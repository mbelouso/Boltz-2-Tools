'''
PyQt6 desktop GUI for collate_boltz2.py.

Provides a form for the collation parameters (working directory, binding site residues,
output prefix), a Run button that executes the collation pipeline in a background thread,
a live log panel, and a results table shown after the run completes.

Usage:
python collate_boltz2_gui.py

Requires PyQt6 (pip install PyQt6) in addition to collate_boltz2.py's own dependencies.
'''

import sys
import logging
import multiprocessing as mp

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QAbstractTableModel, QModelIndex
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QTableView, QFileDialog, QLabel, QMessageBox,
)

import collate_boltz2 as cb


class QtLogHandler(logging.Handler, QObject):
    """Routes collate_boltz2's logger output into the GUI's log panel via a Qt signal."""

    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        self.log_signal.emit(self.format(record))


class CollationWorker(QThread):
    """Runs BoltzCollator.run_collation() on a background thread so the internal
    multiprocessing.Pool calls don't block the Qt event loop."""

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            collator = cb.BoltzCollator(self.config)
            results_df = collator.run_collation()
            self.finished_ok.emit(results_df)
        except Exception as e:
            self.failed.emit(str(e))


class PandasModel(QAbstractTableModel):
    """Read-only QAbstractTableModel wrapping a pandas DataFrame for display in a QTableView."""

    def __init__(self, df: pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self._df.iat[index.row(), index.column()]
        if pd.isna(value):
            return ""
        return str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boltz-2 Collator")
        self.resize(1000, 700)
        self.worker = None

        # Route collate_boltz2's own logger into the log panel (not the root logger, so
        # unrelated library log chatter doesn't flood the GUI).
        self.log_handler = QtLogHandler()
        self.log_handler.log_signal.connect(self._append_log)
        cb.logger.addHandler(self.log_handler)
        cb.logger.setLevel(logging.INFO)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form_widget = QWidget()
        form = QFormLayout(form_widget)

        self.working_dir_edit = QLineEdit(".")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_working_dir)
        working_dir_row = QHBoxLayout()
        working_dir_row.addWidget(self.working_dir_edit)
        working_dir_row.addWidget(browse_btn)
        working_dir_row_widget = QWidget()
        working_dir_row_widget.setLayout(working_dir_row)
        form.addRow("Working directory:", working_dir_row_widget)

        self.residue1_edit = QLineEdit("157")
        form.addRow("Binding site residue 1 (required, orthosteric site):", self.residue1_edit)

        self.residue2_edit = QLineEdit("")
        form.addRow("Binding site residue 2 (optional):", self.residue2_edit)

        self.residue3_edit = QLineEdit("")
        form.addRow("Binding site residue 3 (optional):", self.residue3_edit)

        self.output_prefix_edit = QLineEdit("boltz_results")
        form.addRow("Output prefix:", self.output_prefix_edit)

        load_config_btn = QPushButton("Load config JSON...")
        load_config_btn.clicked.connect(self._load_config_file)
        form.addRow("", load_config_btn)

        layout.addWidget(form_widget)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run_collation)
        layout.addWidget(self.run_btn)

        layout.addWidget(QLabel("Log:"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, stretch=1)

        layout.addWidget(QLabel("Results:"))
        self.table_model = PandasModel()
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        layout.addWidget(self.table_view, stretch=2)

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)

    def _browse_working_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Boltz-2 base directory")
        if directory:
            self.working_dir_edit.setText(directory)

    def _load_config_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select config JSON", "", "JSON Files (*.json)")
        if not path:
            return
        file_config = cb.load_config_from_file(path)
        if 'working_directory' in file_config:
            self.working_dir_edit.setText(str(file_config['working_directory']))
        if 'binding_site_residue1' in file_config and file_config['binding_site_residue1'] is not None:
            self.residue1_edit.setText(str(file_config['binding_site_residue1']))
        if file_config.get('binding_site_residue2') is not None:
            self.residue2_edit.setText(str(file_config['binding_site_residue2']))
        if file_config.get('binding_site_residue3') is not None:
            self.residue3_edit.setText(str(file_config['binding_site_residue3']))
        if 'output_prefix' in file_config:
            self.output_prefix_edit.setText(str(file_config['output_prefix']))

    @staticmethod
    def _parse_optional_int(text: str):
        text = text.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _run_collation(self) -> None:
        residue1_text = self.residue1_edit.text().strip()
        if not residue1_text:
            QMessageBox.warning(self, "Missing input", "Binding site residue 1 is required.")
            return
        try:
            residue1 = int(residue1_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Binding site residue 1 must be an integer.")
            return

        config = {
            'working_directory': self.working_dir_edit.text().strip() or '.',
            'binding_site_residue1': residue1,
            'binding_site_residue2': self._parse_optional_int(self.residue2_edit.text()),
            'binding_site_residue3': self._parse_optional_int(self.residue3_edit.text()),
            'output_prefix': self.output_prefix_edit.text().strip() or 'boltz_results',
        }

        self.run_btn.setEnabled(False)
        self.log_view.clear()

        # Keep the worker as an instance attribute — a local variable can be garbage
        # collected mid-run and crash the app.
        self.worker = CollationWorker(config)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, results_df: pd.DataFrame) -> None:
        self.run_btn.setEnabled(True)
        self.table_model.set_dataframe(results_df)
        self._append_log(f"Done — {len(results_df)} models collated.")

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Collation failed", message)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    main()
