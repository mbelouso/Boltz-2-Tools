'''
PyQt6 desktop GUI for correlate_data.py.

Provides a form for the correlation parameters (input CSV, output directory, minimum
paired points), a Run button that executes the correlation pipeline in a background
thread, a live log panel, and an in-app preview of the generated per-receptor plots.

Usage:
python correlate_data_gui.py

Requires PyQt6 (pip install PyQt6) in addition to correlate_data.py's own dependencies.
'''

import sys
import logging

# Must happen before `import correlate_data` pulls in matplotlib.pyplot -- the worker
# thread calls savefig() off the Qt main thread, and matplotlib's default interactive
# backend is not safe to touch from a background thread inside a running Qt app.
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from pathlib import Path
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QSplitter,
    QLineEdit, QPushButton, QTextEdit, QListWidget, QFileDialog, QLabel, QMessageBox,
)

import correlate_data as cd


class QtLogHandler(logging.Handler, QObject):
    """Routes correlate_data's logger output into the GUI's log panel via a Qt signal."""

    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        self.log_signal.emit(self.format(record))


class CorrelationWorker(QThread):
    """Runs the per-receptor correlation/plotting pipeline on a background thread so
    matplotlib rendering doesn't block the Qt event loop."""

    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, input_csv: str, output_dir: str, min_points: int, parent=None):
        super().__init__(parent)
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.min_points = min_points

    def run(self):
        try:
            df = pd.read_csv(self.input_csv)

            receptors = cd.find_receptors(df)
            if not receptors:
                raise ValueError(f"No '<receptor>_affinity_pred_value' columns found in '{self.input_csv}'")

            expected_columns = [f"{r}_{a}_nm" for r in receptors for a in cd.ACTIVITY_TYPES]
            if not any(col in df.columns for col in expected_columns):
                raise ValueError(
                    f"'{self.input_csv}' has none of the expected <receptor>_{{Ki,IC50,EC50}}_nm columns -- "
                    f"run combine_results.py with --input first"
                )

            cd.logger.info(f"Found {len(receptors)} receptors: {receptors}")
            cd.logger.info(f"Activity types to plot: {cd.ACTIVITY_TYPES}")

            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            generated = []
            for receptor in receptors:
                if cd.plot_receptor_correlations(df, receptor, cd.ACTIVITY_TYPES, self.min_points, self.output_dir):
                    png_path = str(Path(self.output_dir) / f"{receptor}_correlation.png")
                    generated.append((receptor, png_path))

            cd.logger.info(f"Generated correlation plots for {len(generated)}/{len(receptors)} receptors")
            self.finished_ok.emit(generated)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boltz-2 Correlation Plotter")
        self.resize(1100, 750)
        self.worker = None
        self.plot_paths = {}

        # Route correlate_data's own logger into the log panel (not the root logger, so
        # unrelated library log chatter doesn't flood the GUI).
        self.log_handler = QtLogHandler()
        self.log_handler.log_signal.connect(self._append_log)
        cd.logger.addHandler(self.log_handler)
        cd.logger.setLevel(logging.INFO)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form_widget = QWidget()
        form = QFormLayout(form_widget)

        self.input_edit = QLineEdit("combined_results.csv")
        input_browse_btn = QPushButton("Browse...")
        input_browse_btn.clicked.connect(self._browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit)
        input_row.addWidget(input_browse_btn)
        input_row_widget = QWidget()
        input_row_widget.setLayout(input_row)
        form.addRow("Input CSV (combine_results.py output):", input_row_widget)

        self.output_dir_edit = QLineEdit(".")
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(output_browse_btn)
        output_row_widget = QWidget()
        output_row_widget.setLayout(output_row)
        form.addRow("Output directory:", output_row_widget)

        self.min_points_edit = QLineEdit("4")
        form.addRow("Minimum paired points:", self.min_points_edit)

        layout.addWidget(form_widget)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run_correlation)
        layout.addWidget(self.run_btn)

        layout.addWidget(QLabel("Log:"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, stretch=1)

        layout.addWidget(QLabel("Correlation plots:"))
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.plot_list = QListWidget()
        self.plot_list.currentTextChanged.connect(self._show_plot)
        self.plot_list.setMaximumWidth(200)
        splitter.addWidget(self.plot_list)

        self.image_preview = QLabel("(no plots yet)")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(400, 300)
        splitter.addWidget(self.image_preview)

        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=2)

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select combine_results.py output CSV", "", "CSV Files (*.csv)")
        if path:
            self.input_edit.setText(path)

    def _browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def _run_correlation(self) -> None:
        input_csv = self.input_edit.text().strip()
        if not input_csv:
            QMessageBox.warning(self, "Missing input", "Input CSV is required.")
            return

        try:
            min_points = int(self.min_points_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Minimum paired points must be an integer.")
            return

        self.run_btn.setEnabled(False)
        self.log_view.clear()
        self.plot_list.clear()
        self.plot_paths = {}
        self.image_preview.setText("(no plots yet)")
        self.image_preview.setPixmap(QPixmap())

        # Keep the worker as an instance attribute — a local variable can be garbage
        # collected mid-run and crash the app.
        self.worker = CorrelationWorker(input_csv, self.output_dir_edit.text().strip() or '.', min_points)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, generated: list) -> None:
        self.run_btn.setEnabled(True)
        if not generated:
            self._append_log("Done — no receptor had enough data to plot.")
            return

        for receptor, png_path in generated:
            self.plot_paths[receptor] = png_path
            self.plot_list.addItem(receptor)

        self._append_log(f"Done — {len(generated)} plot(s) generated.")
        self.plot_list.setCurrentRow(0)

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Correlation run failed", message)

    def _show_plot(self, receptor: str) -> None:
        if not receptor or receptor not in self.plot_paths:
            return
        pixmap = QPixmap(self.plot_paths[receptor])
        self.image_preview.setPixmap(pixmap.scaled(
            self.image_preview.width(), self.image_preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        current = self.plot_list.currentItem()
        if current is not None:
            self._show_plot(current.text())


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
