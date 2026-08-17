'''
PyQt6 desktop GUI for correlate_data.py.

Loads a collate_combine_boltz2.py anchor-merged CSV in a background thread, then lets
the user pick receptors / activity type / per-receptor filters and see a live, embedded
matplotlib figure (FigureCanvasQTAgg + NavigationToolbar2QT for pan/zoom) -- no PNG
round-trip. All selected receptors always share one plot (correlate_data.py's facet
layout is CLI-only; the GUI never exposes it). Any control change re-applies filters
and redraws, debounced ~200ms so rapid changes (e.g. dragging a spinbox) don't trigger
a redraw per keystroke. Figure generation still runs in a background QThread
(PlotWorker) so a slow regression/CI-band computation never freezes the UI.

Nothing here does matplotlib backend selection -- correlate_data.py builds every Figure
directly (`from matplotlib.figure import Figure`, never pyplot), so it's safe to hand
those Figure objects to FigureCanvasQTAgg from any thread as long as the canvas itself
is only created/drawn on the Qt main thread (which is what PlotWorker's finished_ok
signal, delivered back on the main thread, is for).

Filter controls (see RangeFilterSlider) are superqt's QLabeledDoubleRangeSlider --
dual-handle range sliders with built-in editable numeric labels, so a filter can be
set by dragging or by typing an exact number. There's no separate on/off checkbox per
filter; see RangeFilterSlider's docstring for the "handle at its own extreme = off"
convention this relies on.

The "Weighted regression" checkbox (see _build_selection_group) swaps make_figure()'s
fit between OLS (off, default) and WLS weighted by each point's
affinity_probability_binary (on) -- toggling it just re-triggers the same debounced
redraw as any other control, so the displayed line, confidence band, and R2/slope/
Pearson/Spearman stats always match whichever fit is currently checked; only one line
is ever drawn. Weighted mode also scales each scatter point's size by that same
probability (see correlate_data.py's _marker_sizes()).

The right-most panel (see _build_compound_panel) lists every currently plotted point
across every selected receptor in a table (CompoundDetailPanel below it renders the
selected row's full info + a 2D structure from canonical_smiles via RDKit). Clicking a
plotted point on the chart selects/scrolls to that row in the table instead of opening
a separate window -- see make_figure()'s third return value (per-receptor pair_df) and
each scatter artist's gid, which is what makes a matplotlib pick_event resolvable back
to a specific compound's row (and hence table row) with no extra lookup.

Usage:
python correlate_data_gui.py

Requires PyQt6 and superqt (pip install PyQt6 superqt) in addition to
correlate_data.py's own dependencies. RDKit (conda install -c conda-forge rdkit) is
optional -- without it, the compound detail panel still shows all the tabular info,
just without the structure image.
'''

import math
import sys

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QRadioButton,
    QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)
from superqt import QLabeledDoubleRangeSlider

import pandas as pd

import correlate_data as cd

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


_STRUCTURE_LOGICAL_SIZE = (300, 220)
_STRUCTURE_RENDER_SCALE = 3  # physical pixels rendered per logical point -- see below


def _render_structure(smiles) -> "QPixmap | None":
    """Renders a SMILES string to a 2D-depiction QPixmap via RDKit, or None if RDKit
    isn't installed, smiles is missing/not a string, or it fails to parse. Rendered at
    _STRUCTURE_RENDER_SCALE times _STRUCTURE_LOGICAL_SIZE's physical pixel count, with
    the QPixmap's devicePixelRatio set to match -- so it still lays out at the same
    logical size in the dialog, but supersampled/HiDPI-sharp instead of a single
    300x220 raster stretched to fit a Retina screen. The QImage is built from a PIL
    image's raw buffer and then .copy()'d so the QPixmap owns its own memory rather
    than aliasing a buffer that goes out of scope with this frame."""
    if not _RDKIT_AVAILABLE or not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    render_size = (_STRUCTURE_LOGICAL_SIZE[0] * _STRUCTURE_RENDER_SCALE,
                   _STRUCTURE_LOGICAL_SIZE[1] * _STRUCTURE_RENDER_SCALE)
    img = Draw.MolToImage(mol, size=render_size).convert("RGBA")
    qimage = QImage(img.tobytes("raw", "RGBA"), img.width, img.height, QImage.Format.Format_RGBA8888)
    pixmap = QPixmap.fromImage(qimage.copy())
    pixmap.setDevicePixelRatio(_STRUCTURE_RENDER_SCALE)
    return pixmap


class DataLoadWorker(QThread):
    """Loads and parses the input CSV (cd.load_data -- also runs the missing-ortholog
    diagnostic) on a background thread, since a large CSV parse can take a moment."""

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, input_csv: str, parent=None):
        super().__init__(parent)
        self.input_csv = input_csv

    def run(self):
        try:
            df = cd.load_data(self.input_csv)
            self.finished_ok.emit(df)
        except Exception as e:
            self.failed.emit(str(e))


class PlotWorker(QThread):
    """Runs apply_filters + regression + figure construction (cd.make_figure) on a
    background thread. Builds a bare matplotlib Figure (no canvas, no pyplot) -- see
    module docstring for why that's safe to hand back to the main thread."""

    finished_ok = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, df, receptors, activity, config, layout, min_points,
                 annotate_points, target_map, include_orthologs, excluded_uniprots,
                 weighted, parent=None):
        super().__init__(parent)
        self.df = df
        self.receptors = receptors
        self.activity = activity
        self.config = config
        self.layout = layout
        self.min_points = min_points
        self.annotate_points = annotate_points
        self.target_map = target_map
        self.include_orthologs = include_orthologs
        self.excluded_uniprots = excluded_uniprots
        self.weighted = weighted

    def run(self):
        try:
            fig, filter_reports, point_index = cd.make_figure(
                self.df, self.receptors, self.activity, self.config, layout=self.layout,
                min_points=self.min_points, annotate_points=self.annotate_points,
                target_map=self.target_map, include_orthologs=self.include_orthologs,
                excluded_uniprots=self.excluded_uniprots, weighted=self.weighted,
            )
            self.finished_ok.emit(fig, filter_reports, point_index)
        except Exception as e:
            self.failed.emit(str(e))


class RangeFilterSlider(QWidget):
    """One FilterConfig min/max pair as a single dual-handle range slider (superqt's
    QLabeledDoubleRangeSlider -- drag a handle, or type directly into its built-in
    spinboxes). There's no separate on/off checkbox: dragging a handle to its own
    extreme end (the widget's absolute min/max) means "no constraint on that side",
    which is what min_value()/max_value() report as None -- exactly what apply_filters
    treats as "not applied". A min or max threshold that happens to equal the absolute
    bound is a no-op filter anyway, so this never silently drops real data."""

    changed = pyqtSignal()

    def __init__(self, label: str, absolute_min: float, absolute_max: float, step: float,
                 decimals: int = 2, suffix: str = '', tooltip: str = '', parent=None):
        super().__init__(parent)
        self.absolute_min = absolute_min
        self.absolute_max = absolute_max

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        title = f"{label} ({suffix})" if suffix else label
        label_widget = QLabel(title)
        layout.addWidget(label_widget)

        self.slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(absolute_min, absolute_max)
        self.slider.setSingleStep(step)
        self.slider.setDecimals(decimals)
        self.slider.setValue((absolute_min, absolute_max))
        if tooltip:
            label_widget.setToolTip(tooltip)
            self.slider.setToolTip(tooltip)
        self.slider.valueChanged.connect(self.changed)
        layout.addWidget(self.slider)

    def min_value(self):
        low, _ = self.slider.value()
        return None if math.isclose(low, self.absolute_min, abs_tol=1e-6) else low

    def max_value(self):
        _, high = self.slider.value()
        return None if math.isclose(high, self.absolute_max, abs_tol=1e-6) else high

    def set_range_value(self, low, high) -> None:
        blocked = self.slider.blockSignals(True)
        self.slider.setValue((
            self.absolute_min if low is None else float(low),
            self.absolute_max if high is None else float(high),
        ))
        self.slider.blockSignals(blocked)


class SaveFigureDialog(QDialog):
    """Format / DPI / figure size / transparency controls for exporting the currently
    displayed figure (Sec 6: PNG, SVG, PDF, TIFF; DPI 72-1200 default 300, disabled --
    not just ignored -- for the vector formats since it doesn't apply to them; figure
    size in inches, default 6x6 (square, matching make_figure()'s equal-scale axes);
    optional transparent background). The GUI always
    shows every selected receptor on one shared plot (no facet layout), so there's
    only ever one panel to size."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save figure...")
        layout = QFormLayout(self)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "SVG", "PDF", "TIFF"])
        layout.addRow("Format:", self.format_combo)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        layout.addRow("DPI:", self.dpi_spin)

        size_row = QWidget()
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 40.0)
        self.width_spin.setValue(6.0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 40.0)
        self.height_spin.setValue(6.0)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.height_spin)
        size_layout.addWidget(QLabel("in"))
        layout.addRow("Figure size:", size_row)

        self.transparent_checkbox = QCheckBox("Transparent background")
        layout.addRow("", self.transparent_checkbox)

        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self._on_format_changed(self.format_combo.currentText())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_format_changed(self, fmt: str) -> None:
        vector = fmt in ("SVG", "PDF")
        self.dpi_spin.setEnabled(not vector)
        self.dpi_spin.setToolTip("DPI does not apply to vector formats" if vector else "")


class CompoundDetailPanel(QWidget):
    """Embedded 'compound detail' pane -- structure + full info for whichever row is
    currently selected in MainWindow's compound table (see _build_compound_panel).
    Updated in place via set_compound() on every table-selection change, whether that
    selection came from clicking a row directly or from clicking a plotted point (see
    MainWindow._on_point_pick, which just selects the matching table row)."""

    FIELDS = [
        "ChEMBL ID", "Receptor", "Activity", "Target name", "Target UniProt",
        "Predicted (p-scale)", "Experimental (p-scale)", "Experimental (raw)",
        "Distance to orthosteric site", "Confidence score", "Binding probability",
        "Molecular weight", "AlogP",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.structure_label = QLabel("(click a compound to see its details)")
        self.structure_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.structure_label.setMinimumSize(*_STRUCTURE_LOGICAL_SIZE)
        self.structure_label.setWordWrap(True)
        self.structure_label.setStyleSheet(
            f"background-color: {cd.CHART_SURFACE}; border: 1px solid {cd.AXIS_COLOR};"
        )
        layout.addWidget(self.structure_label)

        form = QFormLayout()
        self.info_labels = {}
        for field in self.FIELDS:
            value_label = QLabel("-")
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(f"{field}:", value_label)
            self.info_labels[field] = value_label
        layout.addLayout(form)

        layout.addWidget(QLabel("SMILES:"))
        self.smiles_label = QLabel("-")
        self.smiles_label.setWordWrap(True)
        self.smiles_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.smiles_label)

    def set_compound(self, row: pd.Series, receptor: str, activity: str) -> None:
        def fmt(value, spec="{}"):
            return "-" if pd.isna(value) else spec.format(value)

        pixmap = _render_structure(row.get("canonical_smiles"))
        if pixmap is not None:
            self.structure_label.setPixmap(pixmap)
        else:
            self.structure_label.setPixmap(QPixmap())
            self.structure_label.setText(
                "(no structure available)" if _RDKIT_AVAILABLE
                else "(install rdkit to view structure: conda install -c conda-forge rdkit)"
            )

        self.info_labels["ChEMBL ID"].setText(fmt(row.get("chembl_id")))
        self.info_labels["Receptor"].setText(receptor)
        self.info_labels["Activity"].setText(activity or "-")
        self.info_labels["Target name"].setText(fmt(row.get("target_name")))
        self.info_labels["Target UniProt"].setText(fmt(row.get("target_uniprot")))
        self.info_labels["Predicted (p-scale)"].setText(fmt(row.get("pred_p"), "{:.2f}"))
        self.info_labels["Experimental (p-scale)"].setText(fmt(row.get("exp_p"), "{:.2f}"))
        self.info_labels["Experimental (raw)"].setText(fmt(row.get("experimental_value_nm"), "{:.1f} nM"))
        self.info_labels["Distance to orthosteric site"].setText(fmt(row.get("distance_to_orthosteric_site"), "{:.2f} A"))
        self.info_labels["Confidence score"].setText(fmt(row.get("confidence_score"), "{:.3f}"))
        self.info_labels["Binding probability"].setText(fmt(row.get("affinity_probability_binary"), "{:.3f}"))
        self.info_labels["Molecular weight"].setText(fmt(row.get("molecular_weight"), "{:.1f}"))
        self.info_labels["AlogP"].setText(fmt(row.get("alogp"), "{:.2f}"))
        self.smiles_label.setText(fmt(row.get("canonical_smiles")))


class MainWindow(QMainWindow):
    REDRAW_DEBOUNCE_MS = 200

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boltz-2 Correlation Plotter")
        self.resize(1600, 900)

        self.df = None
        self.target_map = None
        self.current_figure = None
        self.canvas = None
        self.toolbar = None
        self.histogram_canvas = None
        self.histogram_toolbar = None
        self.receptor_checkboxes = {}
        self._pending_receptor_selection = None
        self.uniprot_checkboxes = {}
        self._pending_excluded_uniprots = None
        self._plot_generation = 0

        # Kept as instance attributes -- a local variable can be garbage collected
        # mid-run and crash the app.
        self.load_worker = None
        self.plot_worker = None

        # Snapshot of what's actually on screen right now (may lag the controls while
        # a redraw is debouncing/running) -- Save Figure / Export CSV always act on
        # this, never on whatever the controls currently say.
        self._shown_receptors = []
        self._shown_activity = None
        self._shown_layout = None
        self._shown_config = None
        self._shown_excluded_uniprots = None
        self._shown_weighted = None

        # {receptor: pair_df} for whatever's currently on screen -- make_figure()'s
        # third return value, indexed by each scatter artist's gid (see
        # _on_point_pick) to resolve a clicked point back to its full compound row.
        self._point_index = {}
        # Parallel to self.compound_table's rows: (receptor, row) per table row, in
        # the same order the table was populated -- lets a table-selection change
        # look up the full pd.Series for CompoundDetailPanel.set_compound().
        self._compound_table_rows = []
        # {receptor: first table row index for that receptor} -- since each
        # receptor's rows are appended as one contiguous block (see
        # _rebuild_compound_table), a pick_event's (gid, event.ind[0]) maps straight
        # to a table row via offset + event.ind[0], no string/id matching needed.
        self._compound_table_offsets = {}

        self.settings = QSettings("Boltz2Tools", "CorrelationApp")

        # No in-app log panel -- correlate_data.py's own logging.basicConfig() already
        # sends everything (warnings, matching diagnostics, save/export confirmations)
        # to the terminal this app was launched from.

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._trigger_redraw)

        self._build_ui()
        self._restore_settings()

    # --- UI construction ---------------------------------------------------------

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        correlation_page = QWidget()
        root_layout = QHBoxLayout(correlation_page)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # A splitter (not a plain QHBoxLayout) so the control panel's width is
        # draggable too, same as the plot/filter-report and table/detail splits
        # inside _build_display_panel() -- every major region on screen resizes the
        # same way.
        root_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_splitter.addWidget(self._build_control_panel())
        root_splitter.addWidget(self._build_display_panel())
        root_splitter.setStretchFactor(0, 0)
        root_splitter.setStretchFactor(1, 1)
        root_splitter.setSizes([360, 1240])
        # Every pane in every splitter in this window (this one and the ones inside
        # _build_display_panel/_build_compound_panel) has an explicit minimum size,
        # with childrenCollapsible left at its Qt default (true): dragging a handle
        # can shrink a pane down to that minimum, and past that point Qt snaps it
        # straight to fully hidden (0) -- so a pane is always either fully readable
        # or fully collapsed, never stuck at some in-between size with clipped text.
        root_layout.addWidget(root_splitter)

        tabs.addTab(correlation_page, "Correlation")
        tabs.addTab(self._build_distribution_page(), "Distributions")

    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_selection_group())
        layout.addWidget(self._build_filters_group())
        layout.addWidget(self._build_target_group())
        layout.addStretch(1)

        scroll.setWidget(panel)
        return scroll

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Data")
        form = QFormLayout(group)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Browse for a file, or paste a path...")
        input_browse_btn = QPushButton("Browse...")
        input_browse_btn.clicked.connect(self._browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit)
        input_row.addWidget(input_browse_btn)
        input_row_widget = QWidget()
        input_row_widget.setLayout(input_row)
        form.addRow("Input CSV:", input_row_widget)

        self.target_map_edit = QLineEdit()
        self.target_map_edit.setPlaceholderText("(optional) override RECEPTOR_UNIPROTS")
        target_map_browse_btn = QPushButton("Browse...")
        target_map_browse_btn.clicked.connect(self._browse_target_map)
        target_map_row = QHBoxLayout()
        target_map_row.addWidget(self.target_map_edit)
        target_map_row.addWidget(target_map_browse_btn)
        target_map_row_widget = QWidget()
        target_map_row_widget.setLayout(target_map_row)
        form.addRow("Target map:", target_map_row_widget)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self._on_load_clicked)
        form.addRow(self.load_btn)

        return group

    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("Selection")
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Receptors:"))
        self.receptor_box = QWidget()
        self.receptor_layout = QVBoxLayout(self.receptor_box)
        self.receptor_layout.setContentsMargins(4, 0, 0, 4)
        self.receptor_placeholder = QLabel("(load data to populate)")
        self.receptor_placeholder.setStyleSheet("color: gray")
        self.receptor_layout.addWidget(self.receptor_placeholder)
        layout.addWidget(self.receptor_box)

        layout.addWidget(QLabel("Activity type:"))
        activity_row = QHBoxLayout()
        self.activity_group = QButtonGroup(self)
        for i, activity in enumerate(cd.ACTIVITY_COLUMNS):
            radio = QRadioButton(activity)
            if i == 0:
                radio.setChecked(True)
            radio.toggled.connect(self._schedule_redraw)
            self.activity_group.addButton(radio)
            activity_row.addWidget(radio)
        layout.addLayout(activity_row)

        form = QFormLayout()
        self.min_points_spin = QSpinBox()
        self.min_points_spin.setRange(2, 1000)
        self.min_points_spin.setValue(4)
        self.min_points_spin.valueChanged.connect(self._schedule_redraw)
        form.addRow("Min points for regression:", self.min_points_spin)
        layout.addLayout(form)

        self.weighted_checkbox = QCheckBox("Weighted regression (by binding probability)")
        self.weighted_checkbox.setToolTip(
            "Off: OLS, all points the same size.\n"
            "On: WLS weighted by each point's affinity_probability_binary -- higher "
            "probability pulls the line harder and draws as a bigger point."
        )
        self.weighted_checkbox.toggled.connect(self._schedule_redraw)
        layout.addWidget(self.weighted_checkbox)

        return group

    def _build_target_group(self) -> QGroupBox:
        group = QGroupBox("Target UniProt accessions (all included by default)")
        layout = QVBoxLayout(group)

        hint = QLabel(
            "Every accession (human and every known ortholog) is included by default -- "
            "uncheck one to exclude that specific species/ortholog from target matching."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        layout.addWidget(hint)

        self.uniprot_box = QWidget()
        self.uniprot_layout = QVBoxLayout(self.uniprot_box)
        self.uniprot_layout.setContentsMargins(4, 0, 0, 4)
        self.uniprot_placeholder = QLabel("(load data to populate)")
        self.uniprot_placeholder.setStyleSheet("color: gray")
        self.uniprot_layout.addWidget(self.uniprot_placeholder)
        layout.addWidget(self.uniprot_box)

        return group

    def _build_filters_group(self) -> QGroupBox:
        group = QGroupBox("Filters (per receptor, off by default)")
        layout = QVBoxLayout(group)

        self.distance_filter = RangeFilterSlider(
            "Distance to orthosteric site", 0.0, 30.0, 0.1, decimals=1, suffix="A",
            tooltip="Observed range in this dataset is ~6.6-19.0 A. The orthosteric-vs-"
                    "extracellular-vestibule boundary is scientifically interesting around 8-12 A.")
        self.confidence_filter = RangeFilterSlider("Confidence score", 0.0, 1.0, 0.01, decimals=2)
        self.probability_filter = RangeFilterSlider("Binding probability", 0.0, 1.0, 0.01, decimals=2)

        self._filter_widgets = [
            self.distance_filter, self.confidence_filter, self.probability_filter,
        ]
        for widget in self._filter_widgets:
            widget.changed.connect(self._schedule_redraw)
            layout.addWidget(widget)

        return group

    def _build_display_panel(self) -> QWidget:
        outer_splitter = QSplitter(Qt.Orientation.Horizontal)

        plot_splitter = QSplitter(Qt.Orientation.Vertical)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.plot_layout = plot_layout

        button_row = QHBoxLayout()
        self.save_figure_btn = QPushButton("Save Figure...")
        self.save_figure_btn.setEnabled(False)
        self.save_figure_btn.clicked.connect(self._on_save_figure_clicked)
        self.export_csv_btn = QPushButton("Export Filtered Data as CSV...")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.clicked.connect(self._on_export_csv_clicked)
        button_row.addWidget(self.save_figure_btn)
        button_row.addWidget(self.export_csv_btn)
        button_row.addStretch(1)
        plot_layout.addLayout(button_row)

        click_hint = QLabel("Click a point on the plot to locate it in the Compounds panel on the right.")
        click_hint.setStyleSheet("color: gray")
        plot_layout.addWidget(click_hint)

        self.plot_placeholder = QLabel("Load data and select at least one receptor to see a plot.")
        self.plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_layout.addWidget(self.plot_placeholder, stretch=1)

        plot_panel.setMinimumHeight(200)
        plot_splitter.addWidget(plot_panel)

        # No in-app log panel -- see the note in __init__. correlate_data.py's own
        # logging already goes to the terminal this app was launched from.
        report_group = QGroupBox("Filter report")
        report_layout = QVBoxLayout(report_group)
        self.filter_report_view = QTextEdit()
        self.filter_report_view.setReadOnly(True)
        report_layout.addWidget(self.filter_report_view)

        report_group.setMinimumHeight(80)
        plot_splitter.addWidget(report_group)
        plot_splitter.setStretchFactor(0, 3)
        plot_splitter.setStretchFactor(1, 1)
        plot_splitter.setMinimumWidth(400)

        compound_panel = self._build_compound_panel()
        compound_panel.setMinimumWidth(280)

        outer_splitter.addWidget(plot_splitter)
        outer_splitter.addWidget(compound_panel)
        outer_splitter.setStretchFactor(0, 3)
        outer_splitter.setStretchFactor(1, 2)
        return outer_splitter

    def _build_compound_panel(self) -> QWidget:
        """Right-most panel: a table of every currently plotted point across every
        selected receptor, plus a detail sub-panel (CompoundDetailPanel) that shows
        the selected row's full info + structure. Clicking a point on the chart
        selects/scrolls to its row here (see _on_point_pick) instead of opening a
        separate window."""
        splitter = QSplitter(Qt.Orientation.Vertical)

        table_group = QGroupBox("Compounds (all plotted points)")
        table_layout = QVBoxLayout(table_group)
        self.compound_table = QTableWidget(0, 4)
        self.compound_table.setHorizontalHeaderLabels(
            ["Receptor", "ChEMBL ID", "Pred. (p)", "Exp. (p)"]
        )
        self.compound_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.compound_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.compound_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.compound_table.verticalHeader().setVisible(False)
        self.compound_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.compound_table.itemSelectionChanged.connect(self._on_compound_table_selection_changed)
        table_layout.addWidget(self.compound_table)
        table_group.setMinimumHeight(100)
        splitter.addWidget(table_group)

        detail_group = QGroupBox("Compound detail")
        detail_layout = QVBoxLayout(detail_group)
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        self.compound_detail = CompoundDetailPanel()
        detail_scroll.setWidget(self.compound_detail)
        detail_layout.addWidget(detail_scroll)
        detail_group.setMinimumHeight(150)
        splitter.addWidget(detail_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _build_distribution_page(self) -> QWidget:
        """Second tab: a frequency histogram of any one numeric field (see
        cd.HISTOGRAM_FIELDS) across every currently plotted point, stacked/colored by
        receptor the same way the Correlation tab's legend is. Driven entirely by
        self._point_index -- the exact same filtered/matched data the correlation
        plot uses -- so it's always in sync with whatever receptors/activity/filters
        are selected there, with no separate data-loading path of its own."""
        page = QWidget()
        layout = QVBoxLayout(page)
        self.histogram_layout = layout

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("X axis:"))
        self.histogram_field_combo = QComboBox()
        for label, _ in cd.HISTOGRAM_FIELDS:
            self.histogram_field_combo.addItem(label)
        self.histogram_field_combo.currentIndexChanged.connect(self._redraw_histogram)
        control_row.addWidget(self.histogram_field_combo)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        self.histogram_placeholder = QLabel("Load data and select at least one receptor to see a distribution.")
        self.histogram_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.histogram_placeholder, stretch=1)

        return page

    # --- Settings persistence -----------------------------------------------------

    def _restore_settings(self) -> None:
        # Deliberately NOT restored: input_edit always starts empty (placeholder text
        # only), even across sessions -- browse/paste is required every time, rather
        # than silently reusing whatever CSV was loaded last.
        self.target_map_edit.setText(self.settings.value("target_map", ""))
        self.min_points_spin.setValue(self.settings.value("min_points", 4, type=int))
        self.weighted_checkbox.setChecked(self.settings.value("weighted_regression", False, type=bool))

        activity = self.settings.value("activity", "Ki")
        for button in self.activity_group.buttons():
            if button.text() == activity:
                button.setChecked(True)

        for widget, key in [
            (self.distance_filter, "filter_distance"), (self.confidence_filter, "filter_confidence"),
            (self.probability_filter, "filter_probability"),
        ]:
            raw_low = self.settings.value(f"{key}_low", None)
            raw_high = self.settings.value(f"{key}_high", None)
            low = float(raw_low) if raw_low not in (None, "") else None
            high = float(raw_high) if raw_high not in (None, "") else None
            widget.set_range_value(low, high)

        receptors_raw = self.settings.value("selected_receptors", [])
        if isinstance(receptors_raw, str):
            receptors_raw = [receptors_raw] if receptors_raw else []
        self._pending_receptor_selection = set(receptors_raw)

        excluded_raw = self.settings.value("excluded_uniprots", [])
        if isinstance(excluded_raw, str):
            excluded_raw = [excluded_raw] if excluded_raw else []
        self._pending_excluded_uniprots = set(excluded_raw)

    def _save_settings(self) -> None:
        # input_csv is intentionally not saved -- see _restore_settings().
        self.settings.setValue("target_map", self.target_map_edit.text())
        self.settings.setValue("min_points", self.min_points_spin.value())
        self.settings.setValue("weighted_regression", self.weighted_checkbox.isChecked())
        self.settings.setValue("activity", self._current_activity())
        self.settings.setValue("selected_receptors", self._selected_receptors())
        self.settings.setValue("excluded_uniprots", list(self._current_excluded_uniprots()))

        for widget, key in [
            (self.distance_filter, "filter_distance"), (self.confidence_filter, "filter_confidence"),
            (self.probability_filter, "filter_probability"),
        ]:
            low, high = widget.slider.value()
            self.settings.setValue(f"{key}_low", low)
            self.settings.setValue(f"{key}_high", high)

    # --- Current-state helpers -----------------------------------------------------

    def _selected_receptors(self) -> list:
        return [r for r, cb in self.receptor_checkboxes.items() if cb.isChecked()]

    def _current_activity(self) -> str:
        checked = self.activity_group.checkedButton()
        return checked.text() if checked else "Ki"

    def _current_filter_config(self) -> cd.FilterConfig:
        return cd.FilterConfig(
            min_distance=self.distance_filter.min_value(), max_distance=self.distance_filter.max_value(),
            min_confidence=self.confidence_filter.min_value(), max_confidence=self.confidence_filter.max_value(),
            min_probability=self.probability_filter.min_value(), max_probability=self.probability_filter.max_value(),
        )

    def _current_excluded_uniprots(self) -> set:
        return {accession for accession, cb in self.uniprot_checkboxes.items() if not cb.isChecked()}

    # --- Data loading ------------------------------------------------------------

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select collate_combine_boltz2.py output CSV", "", "CSV Files (*.csv)")
        if path:
            self.input_edit.setText(path)

    def _browse_target_map(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select target map JSON", "", "JSON Files (*.json)")
        if path:
            self.target_map_edit.setText(path)

    def _on_load_clicked(self) -> None:
        input_csv = self.input_edit.text().strip()
        if not input_csv:
            QMessageBox.warning(self, "Missing input", "Input CSV is required.")
            return

        target_map_path = self.target_map_edit.text().strip()
        try:
            self.target_map = cd.load_target_map(target_map_path or None)
        except Exception as e:
            QMessageBox.critical(self, "Target map failed to load", str(e))
            return

        self.load_btn.setEnabled(False)

        self.load_worker = DataLoadWorker(input_csv)
        self.load_worker.finished_ok.connect(self._on_data_loaded)
        self.load_worker.failed.connect(self._on_data_load_failed)
        self.load_worker.start()

    def _on_data_loaded(self, df) -> None:
        self.load_btn.setEnabled(True)
        self.df = df
        receptors = cd.find_receptors(df)
        if not receptors:
            QMessageBox.warning(self, "No receptors found",
                                 "No '<receptor>_affinity_pred_value' columns found in this file.")
            return

        while self.receptor_layout.count():
            item = self.receptor_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.receptor_checkboxes = {}

        for receptor in receptors:
            checkbox = QCheckBox(receptor)
            checked = receptor in self._pending_receptor_selection if self._pending_receptor_selection else True
            checkbox.setChecked(checked)
            checkbox.toggled.connect(self._schedule_redraw)
            self.receptor_layout.addWidget(checkbox)
            self.receptor_checkboxes[receptor] = checkbox
        self._pending_receptor_selection = None

        self._rebuild_target_accession_checkboxes(receptors)

        cd.logger.info(f"Loaded {len(df)} row(s), found {len(receptors)} receptor(s): {receptors}")
        self._schedule_redraw()

    def _rebuild_target_accession_checkboxes(self, receptors: list) -> None:
        """Rebuilds the 'Target UniProt accessions' checklist for the just-loaded
        receptors, using self.target_map (RECEPTOR_UNIPROTS or a --target-map
        override). Each entry's own "species" tag is what's shown, not a fixed list
        position -- see RECEPTOR_UNIPROTS's own comment."""
        while self.uniprot_layout.count():
            item = self.uniprot_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.uniprot_checkboxes = {}

        target_map = self.target_map or cd.RECEPTOR_UNIPROTS
        for receptor in receptors:
            entries = target_map.get(receptor, [])
            if not entries:
                continue
            receptor_label = QLabel(f"{receptor}:")
            receptor_label.setStyleSheet("font-weight: bold;")
            self.uniprot_layout.addWidget(receptor_label)
            for entry in entries:
                accession = entry["accession"]
                species = entry.get("species")
                label = f"{accession} ({species})" if species else accession
                checkbox = QCheckBox(label)
                excluded = accession in self._pending_excluded_uniprots if self._pending_excluded_uniprots else False
                checkbox.setChecked(not excluded)
                checkbox.toggled.connect(self._schedule_redraw)
                self.uniprot_layout.addWidget(checkbox)
                self.uniprot_checkboxes[accession] = checkbox
        self._pending_excluded_uniprots = None

    def _on_data_load_failed(self, message: str) -> None:
        self.load_btn.setEnabled(True)
        cd.logger.error(f"Failed to load data: {message}")
        QMessageBox.critical(self, "Failed to load data", message)

    # --- Redrawing (debounced) ----------------------------------------------------

    def _schedule_redraw(self) -> None:
        self._redraw_timer.start(self.REDRAW_DEBOUNCE_MS)

    def _trigger_redraw(self) -> None:
        if self.df is None:
            return
        receptors = self._selected_receptors()
        if not receptors:
            self.plot_placeholder.setText("Select at least one receptor to see a plot.")
            self.plot_placeholder.show()
            if self.canvas is not None:
                self.canvas.hide()
                self.toolbar.hide()
            self.save_figure_btn.setEnabled(False)
            self.export_csv_btn.setEnabled(False)
            return

        self._plot_generation += 1
        generation = self._plot_generation
        activity = self._current_activity()
        layout = 'overlay'  # all selected receptors always share one plot -- no facet option in the GUI
        config = self._current_filter_config()
        excluded_uniprots = self._current_excluded_uniprots()

        weighted = self.weighted_checkbox.isChecked()

        self.plot_worker = PlotWorker(
            self.df, receptors, activity, config, layout, self.min_points_spin.value(),
            # annotate_points is always False here -- there's no GUI toggle for it.
            # include_orthologs is always True here -- there's no GUI toggle for it
            # either; excluding a specific species is the Target UniProt accessions
            # panel's job instead (see excluded_uniprots below).
            False, self.target_map, True,
            excluded_uniprots, weighted,
        )
        self.plot_worker.finished_ok.connect(
            lambda fig, reports, points, gen=generation, r=receptors, a=activity, l=layout, c=config, eu=excluded_uniprots, w=weighted:
            self._on_plot_ready(fig, reports, points, gen, r, a, l, c, eu, w)
        )
        self.plot_worker.failed.connect(self._on_plot_failed)
        self.plot_worker.start()

    def _on_plot_ready(self, fig, filter_reports: dict, point_index: dict, generation: int,
                        receptors: list, activity: str, layout: str, config, excluded_uniprots: set,
                        weighted: bool) -> None:
        if generation != self._plot_generation:
            return  # a newer redraw superseded this one; discard

        self.current_figure = fig
        self._shown_receptors = receptors
        self._shown_activity = activity
        self._shown_layout = layout
        self._shown_config = config
        self._shown_excluded_uniprots = excluded_uniprots
        self._shown_weighted = weighted
        self._point_index = point_index
        self._rebuild_compound_table()
        self._redraw_histogram()

        if self.canvas is not None:
            # hide() before deleteLater() -- removeWidget() alone only detaches from
            # the layout's geometry management, it doesn't hide the widget, so
            # without this the old canvas/toolbar can still be visibly on screen
            # (overlapping the new one) until Qt gets around to the deferred delete.
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.hide()
            self.toolbar.deleteLater()
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.hide()
            self.canvas.deleteLater()
        self.plot_placeholder.hide()

        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # A fresh canvas is created on every redraw, so the pick handler is
        # reconnected here rather than once in __init__.
        self.canvas.mpl_connect('pick_event', self._on_point_pick)
        self.plot_layout.insertWidget(1, self.toolbar)
        self.plot_layout.insertWidget(2, self.canvas, stretch=1)
        self.canvas.draw()

        self.save_figure_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)

        self.filter_report_view.setPlainText(
            "\n\n".join(report.as_text() for report in filter_reports.values())
        )

    def _on_plot_failed(self, message: str) -> None:
        cd.logger.error(f"Failed to build plot: {message}")

    # --- Compound panel --------------------------------------------------------

    def _rebuild_compound_table(self) -> None:
        """Repopulates the Compounds table from self._point_index -- one contiguous
        block of rows per receptor, in the same order that receptor's points were
        plotted, so _on_point_pick can map a pick_event straight to a table row via
        offset + event.ind[0] instead of searching for it."""
        self.compound_table.setRowCount(0)
        self._compound_table_rows = []
        self._compound_table_offsets = {}

        for receptor, pair_df in self._point_index.items():
            self._compound_table_offsets[receptor] = len(self._compound_table_rows)
            for _, row in pair_df.iterrows():
                self._compound_table_rows.append((receptor, row))

        self.compound_table.setRowCount(len(self._compound_table_rows))
        for i, (receptor, row) in enumerate(self._compound_table_rows):
            self.compound_table.setItem(i, 0, QTableWidgetItem(receptor))
            self.compound_table.setItem(i, 1, QTableWidgetItem(str(row["chembl_id"])))
            self.compound_table.setItem(i, 2, QTableWidgetItem(f"{row['pred_p']:.2f}"))
            self.compound_table.setItem(i, 3, QTableWidgetItem(f"{row['exp_p']:.2f}"))

    def _on_compound_table_selection_changed(self) -> None:
        selected = self.compound_table.selectionModel().selectedRows()
        if not selected:
            return
        receptor, row = self._compound_table_rows[selected[0].row()]
        self.compound_detail.set_compound(row, receptor, self._shown_activity)

    def _on_point_pick(self, event) -> None:
        """Fires when a scatter point (not the regression line/band, which aren't
        pickable) is clicked -- event.artist.get_gid() is the receptor that point
        belongs to (see make_figure()), and event.ind indexes into that receptor's
        block of rows in the compound table (see _rebuild_compound_table). Selecting
        the row triggers _on_compound_table_selection_changed, which updates the
        detail panel -- so a plot click and a direct table click both funnel through
        the same code path."""
        receptor = event.artist.get_gid()
        offset = self._compound_table_offsets.get(receptor)
        if offset is None or len(event.ind) == 0:
            return
        row_index = offset + event.ind[0]
        self.compound_table.selectRow(row_index)
        self.compound_table.scrollToItem(self.compound_table.item(row_index, 0))

    # --- Distribution histogram ----------------------------------------------------

    def _redraw_histogram(self) -> None:
        """Rebuilds the Distributions tab's histogram from self._point_index (set by
        _on_plot_ready -- the same filtered/matched data behind the correlation plot)
        and whichever field is currently selected. Cheap enough (just binning, no
        regression) to run synchronously on the main thread rather than through a
        QThread like PlotWorker."""
        has_data = any(len(pair_df) for pair_df in self._point_index.values())
        if not has_data:
            self.histogram_placeholder.setText("Load data and select at least one receptor to see a distribution.")
            self.histogram_placeholder.show()
            if self.histogram_canvas is not None:
                self.histogram_canvas.hide()
                self.histogram_toolbar.hide()
            return

        field_label, field = cd.HISTOGRAM_FIELDS[self.histogram_field_combo.currentIndex()]
        fig = cd.make_histogram(self._point_index, field, field_label)

        if self.histogram_canvas is not None:
            # hide() before deleteLater() -- see the identical comment in
            # _on_plot_ready(); without it the old canvas/toolbar can still be
            # visibly on screen, overlapping the new one, until Qt processes the
            # deferred delete.
            self.histogram_layout.removeWidget(self.histogram_toolbar)
            self.histogram_toolbar.hide()
            self.histogram_toolbar.deleteLater()
            self.histogram_layout.removeWidget(self.histogram_canvas)
            self.histogram_canvas.hide()
            self.histogram_canvas.deleteLater()
        self.histogram_placeholder.hide()

        self.histogram_canvas = FigureCanvasQTAgg(fig)
        self.histogram_toolbar = NavigationToolbar2QT(self.histogram_canvas, self)
        self.histogram_layout.insertWidget(1, self.histogram_toolbar)
        self.histogram_layout.insertWidget(2, self.histogram_canvas, stretch=1)
        self.histogram_canvas.draw()

    # --- Export ------------------------------------------------------------------

    def _on_save_figure_clicked(self) -> None:
        if self.current_figure is None:
            return
        dialog = SaveFigureDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        fmt = dialog.format_combo.currentText().lower()
        default_name = f"{self._shown_activity}_{'_'.join(self._shown_receptors)}.{fmt}"
        path, _ = QFileDialog.getSaveFileName(self, "Save figure as...", default_name, f"{fmt.upper()} Files (*.{fmt})")
        if not path:
            return

        original_size = self.current_figure.get_size_inches().copy()
        try:
            self.current_figure.set_size_inches(dialog.width_spin.value(), dialog.height_spin.value())
            cd.save_figure(self.current_figure, path, dpi=dialog.dpi_spin.value(),
                            transparent=dialog.transparent_checkbox.isChecked())
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
        finally:
            self.current_figure.set_size_inches(original_size)
            if self.canvas is not None:
                self.canvas.draw()

    def _on_export_csv_clicked(self) -> None:
        if self.df is None or not self._shown_receptors:
            return
        default_name = f"{self._shown_activity}_{'_'.join(self._shown_receptors)}_pairs.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export filtered data as CSV", default_name, "CSV Files (*.csv)")
        if not path:
            return
        try:
            cd.export_paired_csv(
                self.df, self._shown_receptors, self._shown_activity, self._shown_config,
                path, self.target_map, True, self._shown_excluded_uniprots,
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # --- Lifecycle ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
