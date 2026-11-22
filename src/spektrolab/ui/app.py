"""Графическое приложение «СпектроЛаб» на PySide6 и matplotlib.

Приложение работает без железа: по умолчанию подключается виртуальная
камера-симулятор. Реальная USB-камера используется, если установлен OpenCV.
Все вычисления выполняет ядро :mod:`spektrolab.core`, интерфейс только
управляет камерой, отображает данные и формирует протокол.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QCloseEvent, QFont, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import (
    Calibration,
    Measurement,
    PlanckFit,
    Spectrum,
    auto_roi,
    calibrate_from_lines,
    correct_dark,
    detect_peaks,
    extract_spectrum,
    fit_planck,
    identify,
    linear_calibration,
    reflectance,
    to_grayscale,
)
from ..devices import Camera, CameraError
from ..devices.simulator_device import SIMULATOR_SOURCES, SimulatorCamera

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

APP_TITLE = "СпектроЛаб — USB-спектрометр"
APP_SUBTITLE = "340–1050 нм · учебная лаборатория"
DEFAULT_WAVELENGTH_LOW_NM = 340.0
DEFAULT_WAVELENGTH_HIGH_NM = 1050.0
DEFAULT_PIXEL_COUNT = 1024
FRAME_INTERVAL_MS = 100
ELEMENT_CHOICES = ("Hg", "Na", "Ne", "H", "He", "Cd")
ALL_ELEMENTS = "Все"
CALIBRATION_EXAMPLE = ((0, 340.0), (512, 695.0), (1023, 1050.0))

PLOT_BACKGROUND = "#1b1e23"
PLOT_AXES_BACKGROUND = "#20242b"
PLOT_TEXT_COLOR = "#c9d1dc"
PLOT_GRID_COLOR = "#3a4149"
SPECTRUM_COLOR = "#4fc3f7"
PLANCK_COLOR = "#ffb74d"
PEAK_COLOR = "#ff6e6e"
ROI_COLOR = "#4f8cff"


def _format_number(value: float | None, digits: int = 1) -> str:
    """Аккуратно отформатировать число, заменяя NaN и None прочерком."""
    if value is None:
        return "—"
    number = float(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _dark_palette() -> QPalette:
    """Тёмная палитра Qt в тон оформлению приложения."""
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#1b1e23",
        QPalette.ColorRole.WindowText: "#e6e9ee",
        QPalette.ColorRole.Base: "#20242b",
        QPalette.ColorRole.AlternateBase: "#262b32",
        QPalette.ColorRole.ToolTipBase: "#262b32",
        QPalette.ColorRole.ToolTipText: "#e6e9ee",
        QPalette.ColorRole.Text: "#e6e9ee",
        QPalette.ColorRole.Button: "#2b3038",
        QPalette.ColorRole.ButtonText: "#e6e9ee",
        QPalette.ColorRole.BrightText: "#ff6e6e",
        QPalette.ColorRole.Link: "#4f8cff",
        QPalette.ColorRole.Highlight: "#2f6feb",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: "#8d97a5",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#71798a"))
    return palette


DARK_STYLESHEET = """
QWidget { color: #e6e9ee; font-family: "Segoe UI"; font-size: 13px; }
QMainWindow { background-color: #1b1e23; }
QFrame#headerFrame { background-color: #20242b; border-bottom: 1px solid #2c313a; }
QLabel#appTitle { font-size: 22px; font-weight: 700; color: #f2f6fb; }
QLabel#appSubtitle { color: #8d97a5; font-size: 12px; }
QLabel#sourceStatus { color: #b8c0cb; font-size: 12px; }
QLabel#temperatureValue { font-size: 26px; font-weight: 700; color: #ffb74d; }
QLabel#hintLabel { color: #8d97a5; font-size: 11px; }
QGroupBox {
    background-color: #20242b; border: 1px solid #2c313a; border-radius: 8px;
    margin-top: 14px; padding: 10px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: #8ab4f8; font-weight: 600;
}
QPushButton {
    background-color: #2b3038; border: 1px solid #3a4049;
    border-radius: 6px; padding: 6px 10px;
}
QPushButton:hover { background-color: #353b45; }
QPushButton:pressed { background-color: #242931; }
QPushButton:disabled { color: #71798a; background-color: #24282e; border-color: #2c313a; }
QPushButton#primaryButton { background-color: #2f6feb; border-color: #3f7ffb; font-weight: 600; }
QPushButton#primaryButton:hover { background-color: #3f7ffb; }
QPushButton#primaryButton:pressed { background-color: #2a61cf; }
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QPlainTextEdit {
    background-color: #262b32; border: 1px solid #3a4049;
    border-radius: 6px; padding: 4px 6px;
    selection-background-color: #2f6feb; selection-color: #ffffff;
}
QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled {
    color: #71798a;
}
QComboBox QAbstractItemView {
    background-color: #262b32; border: 1px solid #3a4049;
    selection-background-color: #2f6feb; selection-color: #ffffff;
}
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #4a5260;
    border-radius: 4px; background-color: #262b32;
}
QCheckBox::indicator:checked { background-color: #2f6feb; border-color: #3f7ffb; }
QTabWidget::pane {
    border: 1px solid #2c313a; border-radius: 8px;
    top: -1px; background-color: #1e2227;
}
QTabBar::tab {
    background: transparent; padding: 9px 20px;
    color: #98a3b3; border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #ffffff; border-bottom: 2px solid #4f8cff; }
QTabBar::tab:hover { color: #ffffff; }
QTabWidget::tab-bar { left: 8px; }
QTableWidget {
    background-color: #20242b; border: 1px solid #2c313a;
    border-radius: 6px; gridline-color: #2c313a;
    alternate-background-color: #232830;
}
QTableWidget::item:selected { background-color: #2f6feb; color: #ffffff; }
QHeaderView::section {
    background-color: #262b32; color: #98a3b3; border: none;
    border-right: 1px solid #2c313a; padding: 6px;
}
QStatusBar { background-color: #191c21; color: #98a3b3; }
QStatusBar::item { border: none; }
QToolBar { background-color: #20242b; border: none; spacing: 4px; }
QToolTip { background-color: #262b32; color: #e6e9ee; border: 1px solid #3a4049; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #1e2227; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3a4049; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #4a5260; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QScrollBar:horizontal { background: #1e2227; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #3a4049; border-radius: 5px; min-width: 24px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
"""


class MainWindow(QMainWindow):
    """Главное окно: живой спектр, калибровка, линии и протокол измерения."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)
        self.setMinimumSize(1100, 700)

        # --- состояние приложения ---
        self._camera: Camera | None = None
        self._frame: np.ndarray | None = None
        self._roi: dict[str, int] | None = None
        self._last_raw: np.ndarray | None = None
        self._dark: np.ndarray | None = None
        self._reference: np.ndarray | None = None
        self._wavelengths: np.ndarray | None = None
        self._intensity: np.ndarray | None = None
        self._axis_signature: tuple[object, ...] | None = None
        self._linear_axes: dict[int, np.ndarray] = {}
        self._last_fit: PlanckFit | None = None
        self._last_peaks: list[dict[str, float]] = []
        self._calibration: Calibration | None = None
        self._apply_calibration = False
        self._paused = False
        self._frame_times: deque[float] = deque(maxlen=24)
        self._axis_warning_shown = False
        self._stream_error_shown = False
        self._processing_error_shown = False
        self._legend = None

        # --- таймер живого обновления ---
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_timer)

        self._build_ui()
        self._connect_signals()
        self._update_mode_panels()
        self._update_correction_labels()
        self._update_fit_panel()
        self._refresh_calibration_ui()
        self._clear_peak_results()
        self._update_connection_ui()
        self._timer.start()
        QTimer.singleShot(0, self._start_default_source)

    # ------------------------------------------------------------------
    # построение интерфейса
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.measurement_tab = self._build_measurement_tab()
        self.tabs.addTab(self.measurement_tab, "Измерение")
        self.tabs.addTab(self._build_calibration_tab(), "Калибровка")
        self.tabs.addTab(self._build_lines_tab(), "Линии")
        self.tabs.addTab(self._build_protocol_tab(), "Протокол")
        layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self._build_status_bar()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("headerFrame")
        row = QHBoxLayout(header)
        row.setContentsMargins(20, 14, 20, 14)
        row.setSpacing(14)

        title = QLabel("СпектроЛаб")
        title.setObjectName("appTitle")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("appSubtitle")
        self.source_status_label = QLabel()
        self.source_status_label.setObjectName("sourceStatus")

        row.addWidget(title)
        row.addWidget(subtitle, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        row.addWidget(self.source_status_label)
        return header

    def _make_group(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        """Создать группу с заголовком и вертикальной раскладкой."""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)
        return group, layout

    def _build_measurement_tab(self) -> QWidget:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)
        outer.addWidget(self._build_control_panel())
        outer.addWidget(self._build_plot(), 1)
        self._measure_shortcut = QShortcut(QKeySequence("T"), tab)
        self._measure_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._measure_shortcut.activated.connect(self._measure_temperature)
        return tab

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(12)

        # --- источник ---
        source_group, source_layout = self._make_group("Источник")
        source_layout.addWidget(QLabel("Режим источника:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(("Реальная камера", "Симулятор"))
        self.mode_combo.setCurrentText("Симулятор")
        self.mode_combo.setToolTip(
            "Симулятор не требует оборудования и используется по умолчанию."
        )
        source_layout.addWidget(self.mode_combo)

        self.real_panel = QWidget()
        real_row = QHBoxLayout(self.real_panel)
        real_row.setContentsMargins(0, 0, 0, 0)
        real_row.setSpacing(6)
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(140)
        self.refresh_cameras_button = QPushButton("Обновить")
        self.refresh_cameras_button.setToolTip("Найти подключённые USB-камеры (OpenCV)")
        real_row.addWidget(QLabel("Камера:"))
        real_row.addWidget(self.camera_combo, 1)
        real_row.addWidget(self.refresh_cameras_button)
        source_layout.addWidget(self.real_panel)

        self.sim_panel = QWidget()
        sim_layout = QVBoxLayout(self.sim_panel)
        sim_layout.setContentsMargins(0, 0, 0, 0)
        sim_layout.setSpacing(6)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Источник:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(SIMULATOR_SOURCES))
        source_row.addWidget(self.source_combo, 1)
        sim_layout.addLayout(source_row)
        temperature_row = QHBoxLayout()
        temperature_row.addWidget(QLabel("Температура, К:"))
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(1500.0, 10000.0)
        self.temperature_spin.setSingleStep(50.0)
        self.temperature_spin.setDecimals(0)
        self.temperature_spin.setValue(3000.0)
        self.temperature_spin.setSuffix(" К")
        self.temperature_spin.setToolTip("Температура лампы накаливания в симуляторе")
        temperature_row.addWidget(self.temperature_spin, 1)
        sim_layout.addLayout(temperature_row)
        source_layout.addWidget(self.sim_panel)

        buttons_row = QHBoxLayout()
        self.connect_button = QPushButton("Подключить")
        self.connect_button.setObjectName("primaryButton")
        self.pause_button = QPushButton("Пауза")
        self.pause_button.setEnabled(False)
        buttons_row.addWidget(self.connect_button)
        buttons_row.addWidget(self.pause_button)
        source_layout.addLayout(buttons_row)
        column.addWidget(source_group)

        # --- коррекции ---
        correction_group, correction_layout = self._make_group("Коррекции")
        self.dark_button = QPushButton("Тёмновой кадр")
        self.dark_button.setToolTip(
            "Запомнить спектр при закрытом затворе — собственный шум сенсора."
        )
        self.reference_button = QPushButton("Опорный кадр")
        self.reference_button.setToolTip(
            "Запомнить спектр опорного источника для относительных измерений."
        )
        self.reset_corrections_button = QPushButton("Сбросить коррекции")
        correction_layout.addWidget(self.dark_button)
        correction_layout.addWidget(self.reference_button)
        correction_layout.addWidget(self.reset_corrections_button)
        self.dark_state_label = QLabel()
        self.dark_state_label.setObjectName("hintLabel")
        self.reference_state_label = QLabel()
        self.reference_state_label.setObjectName("hintLabel")
        correction_layout.addWidget(self.dark_state_label)
        correction_layout.addWidget(self.reference_state_label)
        column.addWidget(correction_group)

        # --- результат измерения ---
        fit_group, fit_layout = self._make_group("Температура по закону Планка")
        self.measure_button = QPushButton("Измерить (T)")
        self.measure_button.setObjectName("primaryButton")
        fit_layout.addWidget(self.measure_button)
        self.temperature_label = QLabel("—")
        self.temperature_label.setObjectName("temperatureValue")
        self.temperature_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fit_layout.addWidget(self.temperature_label)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(4)
        self.fit_error_label = QLabel("—")
        self.fit_r2_label = QLabel("—")
        self.fit_rmse_label = QLabel("—")
        self.fit_points_label = QLabel("—")
        form.addRow("Погрешность:", self.fit_error_label)
        form.addRow("R²:", self.fit_r2_label)
        form.addRow("RMSE:", self.fit_rmse_label)
        form.addRow("Точек:", self.fit_points_label)
        fit_layout.addLayout(form)

        self.fit_note_label = QLabel("")
        self.fit_note_label.setObjectName("hintLabel")
        self.fit_note_label.setWordWrap(True)
        fit_layout.addWidget(self.fit_note_label)
        column.addWidget(fit_group)
        column.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(380)
        return scroll

    def _build_plot(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.figure = Figure(figsize=(10, 7), dpi=100, facecolor=PLOT_BACKGROUND)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(420)
        grid = self.figure.add_gridspec(
            2,
            1,
            height_ratios=(1.0, 4.2),
            hspace=0.14,
            left=0.075,
            right=0.985,
            top=0.94,
            bottom=0.08,
        )
        self.ax_image = self.figure.add_subplot(grid[0])
        self.ax_spectrum = self.figure.add_subplot(grid[1])
        self._style_axes()

        self._image_artist = self.ax_image.imshow(
            np.zeros((2, 2), dtype=float),
            cmap="gray",
            vmin=0.0,
            vmax=255.0,
            aspect="auto",
            interpolation="nearest",
            extent=(0.0, 2.0, 2.0, 0.0),
        )
        self._roi_patch = Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            edgecolor=ROI_COLOR,
            facecolor="none",
            linewidth=1.2,
            visible=False,
        )
        self.ax_image.add_patch(self._roi_patch)

        self._spectrum_line, = self.ax_spectrum.plot(
            [], [], color=SPECTRUM_COLOR, linewidth=1.3, label="Спектр"
        )
        self._planck_line, = self.ax_spectrum.plot(
            [], [], color=PLANCK_COLOR, linewidth=1.6, linestyle="--",
            label="Планк", visible=False,
        )
        self._peak_scatter = self.ax_spectrum.scatter(
            [], [], marker="v", s=30, color=PEAK_COLOR, zorder=5, label="Линии"
        )
        self.ax_spectrum.set_title("Спектр", color=PLOT_TEXT_COLOR, fontsize=12, pad=8)

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.toolbar)
        return container

    def _style_axes(self) -> None:
        for axis in (self.ax_image, self.ax_spectrum):
            axis.set_facecolor(PLOT_AXES_BACKGROUND)
            axis.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
            for spine in axis.spines.values():
                spine.set_color(PLOT_GRID_COLOR)
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        self.ax_spectrum.grid(
            True, color=PLOT_GRID_COLOR, linestyle=":", linewidth=0.7, alpha=0.8
        )
        self.ax_spectrum.set_xlabel("Длина волны, нм", color=PLOT_TEXT_COLOR, fontsize=10)
        self.ax_spectrum.set_ylabel(
            "Интенсивность, отн. ед.", color=PLOT_TEXT_COLOR, fontsize=10
        )

    def _build_calibration_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.calibration_table = QTableWidget(0, 2)
        self.calibration_table.setHorizontalHeaderLabels(["Пиксель", "Длина волны, нм"])
        self.calibration_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.calibration_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.calibration_table.setAlternatingRowColors(True)
        self.calibration_table.verticalHeader().setVisible(False)
        for pixel, wavelength in CALIBRATION_EXAMPLE:
            row = self.calibration_table.rowCount()
            self.calibration_table.insertRow(row)
            self.calibration_table.setItem(row, 0, QTableWidgetItem(str(pixel)))
            self.calibration_table.setItem(row, 1, QTableWidgetItem(f"{wavelength:g}"))
        left.addWidget(self.calibration_table, 1)

        table_buttons = QHBoxLayout()
        self.add_calibration_row_button = QPushButton("Добавить")
        self.remove_calibration_row_button = QPushButton("Удалить")
        table_buttons.addWidget(self.add_calibration_row_button)
        table_buttons.addWidget(self.remove_calibration_row_button)
        table_buttons.addStretch(1)
        left.addLayout(table_buttons)

        hint = QLabel(
            "Укажите положение эталонных линий на сенсоре (в пикселях) "
            "и их табличные длины волн. Точки можно вводить с запятой."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        left.addWidget(hint)
        layout.addLayout(left, 1)

        side = QWidget()
        side.setMaximumWidth(400)
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)

        degree_row = QHBoxLayout()
        degree_row.addWidget(QLabel("Степень полинома:"))
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 3)
        self.degree_spin.setValue(2)
        degree_row.addWidget(self.degree_spin)
        degree_row.addStretch(1)
        column.addLayout(degree_row)

        self.build_calibration_button = QPushButton("Построить калибровку")
        self.build_calibration_button.setObjectName("primaryButton")
        column.addWidget(self.build_calibration_button)

        result_group, result_layout = self._make_group("Результат")
        result_form = QFormLayout()
        result_form.setContentsMargins(0, 0, 0, 0)
        self.calibration_rms_label = QLabel("—")
        self.calibration_max_error_label = QLabel("—")
        self.calibration_range_label = QLabel("—")
        self.calibration_points_label = QLabel("—")
        result_form.addRow("RMS:", self.calibration_rms_label)
        result_form.addRow("Макс. ошибка:", self.calibration_max_error_label)
        result_form.addRow("Диапазон:", self.calibration_range_label)
        result_form.addRow("Точек привязки:", self.calibration_points_label)
        result_layout.addLayout(result_form)
        column.addWidget(result_group)

        self.apply_calibration_checkbox = QCheckBox("Применять калибровку к спектру")
        self.apply_calibration_checkbox.setToolTip(
            "Строить ось длин волн через Calibration.wavelengths() по числу пикселей кадра."
        )
        column.addWidget(self.apply_calibration_checkbox)

        io_row = QHBoxLayout()
        self.save_calibration_button = QPushButton("Сохранить JSON…")
        self.load_calibration_button = QPushButton("Загрузить JSON…")
        io_row.addWidget(self.save_calibration_button)
        io_row.addWidget(self.load_calibration_button)
        column.addLayout(io_row)
        column.addStretch(1)
        layout.addWidget(side)
        return tab

    def _build_lines_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(QLabel("Элемент:"))
        self.element_combo = QComboBox()
        self.element_combo.addItems([*ELEMENT_CHOICES, ALL_ELEMENTS])
        self.element_combo.setCurrentText("Hg")
        row.addWidget(self.element_combo)
        row.addWidget(QLabel("Допуск, нм:"))
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.5, 5.0)
        self.tolerance_spin.setSingleStep(0.1)
        self.tolerance_spin.setDecimals(1)
        self.tolerance_spin.setValue(2.0)
        row.addWidget(self.tolerance_spin)
        row.addWidget(QLabel("Порог:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 50.0)
        self.threshold_spin.setSingleStep(1.0)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setValue(5.0)
        self.threshold_spin.setSuffix(" %")
        self.threshold_spin.setToolTip("Минимальная высота линии в процентах от максимума спектра")
        row.addWidget(self.threshold_spin)
        self.find_lines_button = QPushButton("Найти линии")
        self.find_lines_button.setObjectName("primaryButton")
        row.addWidget(self.find_lines_button)
        row.addStretch(1)
        layout.addWidget(controls)

        self.peaks_table = QTableWidget(0, 7)
        self.peaks_table.setHorizontalHeaderLabels(
            ["№", "λ изм., нм", "Интенсивность", "FWHM, нм",
             "Элемент", "Линия, нм", "Ошибка, нм"]
        )
        self.peaks_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.peaks_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.peaks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.peaks_table.setAlternatingRowColors(True)
        self.peaks_table.verticalHeader().setVisible(False)
        layout.addWidget(self.peaks_table, 1)

        self.lines_summary_label = QLabel()
        self.lines_summary_label.setObjectName("hintLabel")
        self.lines_summary_label.setWordWrap(True)
        layout.addWidget(self.lines_summary_label)
        return tab

    def _build_protocol_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        operator_row = QHBoxLayout()
        operator_row.addWidget(QLabel("Оператор:"))
        self.operator_edit = QLineEdit()
        self.operator_edit.setPlaceholderText("Фамилия И.О.")
        operator_row.addWidget(self.operator_edit, 1)
        layout.addLayout(operator_row)

        layout.addWidget(QLabel("Примечания:"))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Условия измерения, параметры установки…")
        self.notes_edit.setFixedHeight(80)
        layout.addWidget(self.notes_edit)

        buttons = QHBoxLayout()
        self.refresh_protocol_button = QPushButton("Обновить")
        self.refresh_protocol_button.setObjectName("primaryButton")
        self.save_report_button = QPushButton("Сохранить MD…")
        self.save_json_button = QPushButton("Сохранить JSON…")
        self.save_csv_button = QPushButton("Сохранить CSV…")
        buttons.addWidget(self.refresh_protocol_button)
        buttons.addWidget(self.save_report_button)
        buttons.addWidget(self.save_json_button)
        buttons.addWidget(self.save_csv_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.report_edit = QPlainTextEdit()
        self.report_edit.setPlaceholderText(
            "Нажмите «Обновить», чтобы сформировать протокол по текущему измерению."
        )
        layout.addWidget(self.report_edit, 1)
        return tab

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.stream_label = QLabel("поток: нет данных")
        self.calibration_status_label = QLabel("Калибровка: паспортная 340–1050 нм")
        bar.addPermanentWidget(self.stream_label)
        bar.addPermanentWidget(self.calibration_status_label)
        bar.showMessage(
            "Готово. По умолчанию используется симулятор — прибор работает без камеры.",
            8000,
        )

    def _connect_signals(self) -> None:
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.refresh_cameras_button.clicked.connect(
            lambda *_args: self._refresh_cameras(silent=False)
        )
        self.source_combo.currentTextChanged.connect(self._on_simulator_changed)
        self.temperature_spin.valueChanged.connect(self._on_simulator_changed)
        self.connect_button.clicked.connect(self._toggle_connection)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.dark_button.clicked.connect(self._capture_dark)
        self.reference_button.clicked.connect(self._capture_reference)
        self.reset_corrections_button.clicked.connect(
            lambda *_args: self._reset_corrections()
        )
        self.measure_button.clicked.connect(self._measure_temperature)
        self.add_calibration_row_button.clicked.connect(self._add_calibration_row)
        self.remove_calibration_row_button.clicked.connect(self._remove_calibration_row)
        self.build_calibration_button.clicked.connect(
            lambda *_args: self._build_calibration()
        )
        self.save_calibration_button.clicked.connect(self._save_calibration)
        self.load_calibration_button.clicked.connect(self._load_calibration)
        self.apply_calibration_checkbox.toggled.connect(
            self._on_apply_calibration_toggled
        )
        self.find_lines_button.clicked.connect(self._find_lines)
        self.refresh_protocol_button.clicked.connect(self._refresh_protocol)
        self.save_report_button.clicked.connect(
            lambda *_args: self._save_protocol("md")
        )
        self.save_json_button.clicked.connect(
            lambda *_args: self._save_protocol("json")
        )
        self.save_csv_button.clicked.connect(
            lambda *_args: self._save_protocol("csv")
        )

    # ------------------------------------------------------------------
    # камера и симулятор
    # ------------------------------------------------------------------
    def _is_simulator_mode(self) -> bool:
        return self.mode_combo.currentText() == "Симулятор"

    def _update_mode_panels(self) -> None:
        simulator = self._is_simulator_mode()
        self.sim_panel.setVisible(simulator)
        self.real_panel.setVisible(not simulator)

    def _on_mode_changed(self, _text: str) -> None:
        self._update_mode_panels()
        if self._camera is not None:
            self._disconnect_camera(silent=True)
        if self._is_simulator_mode():
            self._connect_camera()
        else:
            self._refresh_cameras(silent=True)

    def _refresh_cameras(self, silent: bool = False) -> list[int]:
        """Перебрать доступные камеры; при проблемах OpenCV — понятное сообщение."""
        self.camera_combo.clear()
        indices: list[int] = []
        scan_error: Exception | None = None
        try:
            from ..devices.opencv_camera import list_cameras

            indices = [int(index) for index in list_cameras(max_index=5)]
        except Exception as error:  # noqa: BLE001 - нет OpenCV или камера занята
            scan_error = error

        if scan_error is not None:
            self.camera_combo.addItem("OpenCV недоступен", None)
            self.camera_combo.setEnabled(False)
            if not silent:
                self._error(
                    "Камеры недоступны",
                    f"{scan_error}\n\n"
                    "Для реальной камеры установите OpenCV:\n"
                    "pip install opencv-python\n\n"
                    "Симулятор работает без дополнительных пакетов.",
                )
            return []
        if not indices:
            self.camera_combo.addItem("камеры не найдены", None)
            self.camera_combo.setEnabled(False)
            if not silent:
                self._warn(
                    "Камеры не найдены",
                    "OpenCV не обнаружил ни одной камеры. Проверьте подключение "
                    "и разрешения, затем нажмите «Обновить». Для учёбы можно "
                    "переключиться на симулятор.",
                )
            return []

        self.camera_combo.setEnabled(True)
        for index in indices:
            self.camera_combo.addItem(f"Камера #{index}", index)
        return indices

    def _start_default_source(self) -> None:
        """Запуск сразу после старта: симулятор подключается автоматически."""
        self._refresh_cameras(silent=True)
        self._connect_camera()

    def _connect_camera(self) -> None:
        if self._camera is not None:
            return
        try:
            if self._is_simulator_mode():
                camera: Camera = SimulatorCamera(
                    source=self.source_combo.currentText() or SIMULATOR_SOURCES[0],
                    temperature=float(self.temperature_spin.value()),
                )
                camera.open()
            else:
                index = self.camera_combo.currentData()
                if index is None:
                    self._warn(
                        "Камера не выбрана",
                        "Нет доступных камер. Нажмите «Обновить» или "
                        "переключитесь на симулятор.",
                    )
                    return
                from ..devices import OpenCVCamera

                camera = OpenCVCamera(index=int(index), width=1920, height=1080)
                camera.open()
        except Exception as error:  # noqa: BLE001 - источник может быть недоступен
            self._camera = None
            self._update_connection_ui()
            self._error(
                "Не удалось подключить источник",
                f"{error}\n\n"
                "Проверьте, что камера свободна и установлен opencv-python. "
                "Симулятор доступен всегда.",
            )
            return

        self._camera = camera
        self._paused = False
        self._frame_times.clear()
        self._stream_error_shown = False
        self._processing_error_shown = False
        self._reset_corrections(announce=False)
        self._update_connection_ui()
        self.statusBar().showMessage(f"Источник подключён: {camera.name}", 5000)

    def _disconnect_camera(self, silent: bool = False) -> None:
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:  # noqa: BLE001 - выход не должен падать из-за камеры
                pass
            self._camera = None
        self._paused = False
        self._frame_times.clear()
        self._reset_corrections(announce=False)
        self._update_connection_ui()
        self.stream_label.setText("поток: нет данных")
        if not silent:
            self.statusBar().showMessage("Источник отключён.", 5000)

    def _toggle_connection(self) -> None:
        if self._camera is None:
            self._connect_camera()
        else:
            self._disconnect_camera()

    def _toggle_pause(self) -> None:
        if self._camera is None:
            return
        self._paused = not self._paused
        self._frame_times.clear()
        self.pause_button.setText("Продолжить" if self._paused else "Пауза")
        self._update_source_status()
        self.statusBar().showMessage(
            "Поток приостановлен — график заморожен." if self._paused
            else "Поток возобновлён.",
            4000,
        )

    def _update_connection_ui(self) -> None:
        connected = self._camera is not None
        self.connect_button.setText("Отключить" if connected else "Подключить")
        self.pause_button.setEnabled(connected)
        self.pause_button.setText("Продолжить" if self._paused else "Пауза")
        self._update_source_status()

    def _update_source_status(self) -> None:
        if self._camera is None:
            self.source_status_label.setText(
                "<span style='color:#9aa4b0'>●</span> Источник не подключён"
            )
            return
        details = self._camera.name
        resolution = self._camera.resolution
        if resolution is not None:
            details = f"{details} · {resolution[0]}×{resolution[1]}"
        if self._paused:
            self.source_status_label.setText(
                f"<span style='color:#ffb74d'>●</span> Пауза · {details}"
            )
        else:
            self.source_status_label.setText(
                f"<span style='color:#66bb6a'>●</span> {details}"
            )

    def _on_simulator_changed(self, *_args: object) -> None:
        camera = self._camera
        if not isinstance(camera, SimulatorCamera):
            return
        try:
            camera.set_source(self.source_combo.currentText())
            camera.set_temperature(float(self.temperature_spin.value()))
        except ValueError as error:
            self.statusBar().showMessage(str(error), 6000)
            return
        self._reset_fit()
        self._update_source_status()

    # ------------------------------------------------------------------
    # живой поток
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        if self._camera is None or self._paused or not self._camera.is_open:
            return
        try:
            frame = self._camera.read()
        except (CameraError, RuntimeError) as error:
            self._handle_stream_error(error)
            return
        except Exception as error:  # noqa: BLE001 - неожиданная ошибка драйвера
            self._handle_stream_error(error)
            return
        try:
            self._process_frame(np.asarray(frame))
        except Exception as error:  # noqa: BLE001 - один сбой не должен ронять UI
            if not self._processing_error_shown:
                self._processing_error_shown = True
                self.statusBar().showMessage(
                    f"Ошибка обработки кадра: {error}", 8000
                )

    def _handle_stream_error(self, error: Exception) -> None:
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:  # noqa: BLE001 - камера уже могла отключиться
                pass
        self._camera = None
        self._paused = False
        self._frame_times.clear()
        self._update_connection_ui()
        self.stream_label.setText("поток: нет данных")
        self.statusBar().showMessage(f"Поток остановлен: {error}", 8000)
        if not self._stream_error_shown:
            self._stream_error_shown = True
            self._error(
                "Ошибка камеры",
                f"Чтение кадра прервано:\n{error}\n\n"
                "Проверьте подключение и нажмите «Подключить» снова.",
            )

    def _process_frame(self, frame: np.ndarray) -> None:
        """Обработать кадр: ROI, спектр, коррекции, график и подписи."""
        try:
            roi = auto_roi(frame)
        except (ValueError, TypeError):
            height, width = int(frame.shape[0]), int(frame.shape[1])
            roi = {"x": 0, "y": 0, "width": width, "height": height}

        try:
            raw = extract_spectrum(
                frame, roi=(roi["x"], roi["y"], roi["width"], roi["height"])
            )
        except ValueError:
            raw = extract_spectrum(frame)
        raw = np.asarray(raw, dtype=float).ravel()
        if raw.size < 2:
            return

        intensities = self._apply_corrections(raw)
        if intensities.size < 2 or not np.all(np.isfinite(intensities)):
            return

        wavelengths, signature = self._wavelength_axis(intensities.size)
        axis_changed = signature != self._axis_signature
        self._axis_signature = signature
        self._last_raw = raw
        self._frame = frame
        self._roi = roi
        self._wavelengths = wavelengths
        self._intensity = intensities
        if axis_changed:
            self._reset_fit()
            self._clear_peak_results()

        self._update_image(frame, roi)
        self._spectrum_line.set_data(wavelengths, intensities)
        if axis_changed:
            self.ax_spectrum.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
        self._autoscale_plot()
        self._frame_times.append(time.monotonic())
        self._update_stream_info(int(frame.shape[1]), int(frame.shape[0]), roi)
        self.canvas.draw_idle()

    def _apply_corrections(self, raw: np.ndarray) -> np.ndarray:
        """Применить тёмновой и опорный кадры к сырому спектру."""
        values = np.asarray(raw, dtype=float)
        # Кадры коррекций запоминаются для конкретного разрешения сенсора.
        if self._dark is not None and self._dark.shape != values.shape:
            self._dark = None
            self._update_correction_labels()
        if self._reference is not None and self._reference.shape != values.shape:
            self._reference = None
            self._update_correction_labels()
        try:
            if self._reference is not None:
                dark = (
                    self._dark if self._dark is not None else np.zeros_like(values)
                )
                corrected = reflectance(values, dark, self._reference)
            elif self._dark is not None:
                corrected = correct_dark(values, self._dark)
            else:
                corrected = values.copy()
        except ValueError:
            self._reset_corrections(announce=False)
            return values.copy()
        return np.nan_to_num(
            np.asarray(corrected, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )

    def _wavelength_axis(self, count: int) -> tuple[np.ndarray, tuple[object, ...]]:
        """Ось длин волн: калибровка, если включена, иначе паспортная 340–1050 нм."""
        if self._apply_calibration and self._calibration is not None:
            try:
                wavelengths = np.asarray(self._calibration.wavelengths(count), dtype=float)
            except Exception:  # noqa: BLE001 - калибровка не должна ронять поток
                wavelengths = np.empty(0, dtype=float)
            if (
                wavelengths.size == count
                and np.all(np.isfinite(wavelengths))
                and np.all(np.diff(wavelengths) > 0)
            ):
                return wavelengths, ("calibration", count, id(self._calibration))
            if not self._axis_warning_shown:
                self._axis_warning_shown = True
                self.statusBar().showMessage(
                    "Калибровка даёт немонотонную ось — используется "
                    "паспортная 340–1050 нм.",
                    8000,
                )

        cached = self._linear_axes.get(count)
        if cached is None:
            try:
                cached = linear_calibration(
                    count, DEFAULT_WAVELENGTH_LOW_NM, DEFAULT_WAVELENGTH_HIGH_NM
                ).wavelengths(count)
            except ValueError:
                cached = np.linspace(
                    DEFAULT_WAVELENGTH_LOW_NM, DEFAULT_WAVELENGTH_HIGH_NM, count
                )
            self._linear_axes[count] = cached
        return cached, ("linear", count)

    def _update_image(self, frame: np.ndarray, roi: dict[str, int] | None) -> None:
        gray = to_grayscale(frame)
        height, width = int(gray.shape[0]), int(gray.shape[1])
        self._image_artist.set_data(gray)
        self._image_artist.set_extent((0.0, float(width), float(height), 0.0))
        self.ax_image.set_xlim(0.0, float(width))
        self.ax_image.set_ylim(float(height), 0.0)
        if roi is None:
            self._roi_patch.set_visible(False)
            return
        self._roi_patch.set_visible(True)
        self._roi_patch.set_bounds(
            roi["x"], roi["y"], roi["width"], roi["height"]
        )

    def _autoscale_plot(self) -> None:
        """Подобрать вертикальный масштаб по данным и кривой Планка."""
        if self._intensity is None or self._wavelengths is None:
            return
        finite = self._intensity[np.isfinite(self._intensity)]
        if finite.size == 0:
            return
        top = float(np.max(finite))
        bottom = float(np.min(finite))
        if self._planck_line.get_visible():
            _wavelengths, model = self._planck_line.get_data()
            model_values = np.asarray(model, dtype=float)
            if model_values.size:
                top = max(top, float(np.nanmax(model_values)))
                bottom = min(bottom, float(np.nanmin(model_values)))
        if not np.isfinite(top) or not np.isfinite(bottom):
            return
        span = top - bottom
        if span <= 1e-9:
            span = max(abs(top) * 0.1, 1.0)
        self.ax_spectrum.set_ylim(bottom - span * 0.05, top + span * 0.10)

    def _update_stream_info(
        self, width: int, height: int, roi: dict[str, int] | None
    ) -> None:
        if roi is None:
            roi_text = "ROI: весь кадр"
        else:
            roi_text = f"ROI y={roi['y']}, h={roi['height']}"
        self.stream_label.setText(
            f"{width}×{height} px · {roi_text} · {self._current_fps():.1f} кадр/с"
        )

    def _current_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    # ------------------------------------------------------------------
    # коррекции
    # ------------------------------------------------------------------
    def _capture_dark(self) -> None:
        if self._last_raw is None:
            self._warn(
                "Нет кадра",
                "Сначала подключите источник и дождитесь первого кадра, "
                "затем закройте затвор и снимите тёмновой кадр.",
            )
            return
        self._dark = np.asarray(self._last_raw, dtype=float).copy()
        self._update_correction_labels()
        self._reset_fit()
        self.statusBar().showMessage(
            "Тёмновой кадр записан: собственный шум сенсора будет вычитаться.", 6000
        )

    def _capture_reference(self) -> None:
        if self._last_raw is None:
            self._warn(
                "Нет кадра",
                "Сначала подключите источник и дождитесь первого кадра, "
                "затем снимите опорный кадр.",
            )
            return
        self._reference = np.asarray(self._last_raw, dtype=float).copy()
        self._update_correction_labels()
        self._reset_fit()
        self.statusBar().showMessage(
            "Опорный кадр записан: спектр будет пересчитан в относительных единицах.",
            6000,
        )

    def _reset_corrections(self, announce: bool = True) -> None:
        self._dark = None
        self._reference = None
        self._update_correction_labels()
        self._reset_fit()
        if announce:
            self.statusBar().showMessage("Коррекции сброшены.", 4000)

    def _update_correction_labels(self) -> None:
        self.dark_state_label.setText(
            "Тёмновой: записан" if self._dark is not None else "Тёмновой: не записан"
        )
        self.reference_state_label.setText(
            "Опорный: записан"
            if self._reference is not None
            else "Опорный: не записан"
        )

    # ------------------------------------------------------------------
    # измерение температуры
    # ------------------------------------------------------------------
    def _measure_temperature(self) -> None:
        spectrum = self._make_spectrum()
        if spectrum is None:
            self._warn(
                "Нет данных",
                "Подключите источник и дождитесь кадра со спектром, "
                "затем повторите измерение.",
            )
            return
        if spectrum.size < 8:
            self._warn(
                "Мало точек",
                f"Для подгонки нужно минимум 8 точек, сейчас {spectrum.size}. "
                "Проверьте разрешение камеры.",
            )
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
        except Exception as error:  # noqa: BLE001 - фит не должен ронять приложение
            QApplication.restoreOverrideCursor()
            self._error("Подгонка не удалась", str(error))
            return
        QApplication.restoreOverrideCursor()

        if not fit.success:
            self._reset_fit()
            self._warn(
                "Подгонка не удалась",
                fit.message or "Спектр не похож на излучение чёрного тела.",
            )
            return

        self._last_fit = fit
        self._update_fit_panel()
        self._draw_planck(fit)
        if fit.message:
            self.statusBar().showMessage(fit.message, 8000)
        else:
            self.statusBar().showMessage(
                f"Температура: {fit.temperature:.0f} ± "
                f"{fit.temperature_err:.0f} К, R² = {fit.r_squared:.4f}",
                8000,
            )

    def _draw_planck(self, fit: PlanckFit) -> None:
        if self._wavelengths is None:
            return
        low = float(np.min(self._wavelengths))
        high = float(np.max(self._wavelengths))
        grid = np.linspace(low, high, 400)
        self._planck_line.set_data(grid, fit.model(grid))
        self._planck_line.set_visible(True)
        if self._legend is None:
            self._legend = self.ax_spectrum.legend(
                loc="upper right", frameon=False, fontsize=9, labelcolor=PLOT_TEXT_COLOR
            )
        self.ax_spectrum.set_title(
            f"T = {fit.temperature:.0f} ± {fit.temperature_err:.0f} К · "
            f"R² = {fit.r_squared:.4f}",
            color=PLOT_TEXT_COLOR,
            fontsize=12,
            pad=8,
        )
        self._autoscale_plot()
        self.canvas.draw_idle()

    def _reset_fit(self) -> None:
        self._last_fit = None
        self._planck_line.set_data([], [])
        self._planck_line.set_visible(False)
        if self._legend is not None:
            self._legend.remove()
            self._legend = None
        self.ax_spectrum.set_title("Спектр", color=PLOT_TEXT_COLOR, fontsize=12, pad=8)
        self._update_fit_panel()

    def _update_fit_panel(self) -> None:
        fit = self._last_fit
        if fit is None or not fit.success:
            self.temperature_label.setText("—")
            self.fit_error_label.setText("—")
            self.fit_r2_label.setText("—")
            self.fit_rmse_label.setText("—")
            self.fit_points_label.setText("—")
            self.fit_note_label.setText("")
            return
        self.temperature_label.setText(f"{fit.temperature:.0f} К")
        self.fit_error_label.setText(f"± {fit.temperature_err:.0f} К")
        self.fit_r2_label.setText(f"{fit.r_squared:.4f}")
        self.fit_rmse_label.setText(f"{fit.rmse:.3f}")
        self.fit_points_label.setText(str(fit.n_points))
        self.fit_note_label.setText(fit.message)

    # ------------------------------------------------------------------
    # калибровка
    # ------------------------------------------------------------------
    def _pixel_count(self) -> int:
        if self._frame is not None:
            return int(self._frame.shape[1])
        if self._camera is not None and self._camera.resolution is not None:
            return int(self._camera.resolution[0])
        return DEFAULT_PIXEL_COUNT

    def _add_calibration_row(self) -> None:
        row = self.calibration_table.rowCount()
        self.calibration_table.insertRow(row)
        for column in range(self.calibration_table.columnCount()):
            self.calibration_table.setItem(row, column, QTableWidgetItem(""))
        self.calibration_table.setCurrentCell(row, 0)

    def _remove_calibration_row(self) -> None:
        row = self.calibration_table.currentRow()
        if row < 0:
            row = self.calibration_table.rowCount() - 1
        if row >= 0:
            self.calibration_table.removeRow(row)

    def _collect_calibration_points(self) -> tuple[list[float], list[float]]:
        pixels: list[float] = []
        wavelengths: list[float] = []
        for row in range(self.calibration_table.rowCount()):
            pixel_item = self.calibration_table.item(row, 0)
            wavelength_item = self.calibration_table.item(row, 1)
            pixel_text = pixel_item.text().strip().replace(",", ".") if pixel_item else ""
            wavelength_text = (
                wavelength_item.text().strip().replace(",", ".") if wavelength_item else ""
            )
            if not pixel_text and not wavelength_text:
                continue
            if not pixel_text or not wavelength_text:
                raise ValueError(f"Строка {row + 1}: заполните оба столбца.")
            try:
                pixel = float(pixel_text)
                wavelength = float(wavelength_text)
            except ValueError as error:
                raise ValueError(
                    f"Строка {row + 1}: ожидаются числа, получено "
                    f"«{pixel_text}» и «{wavelength_text}»."
                ) from error
            pixels.append(pixel)
            wavelengths.append(wavelength)
        if len(pixels) < 2:
            raise ValueError(
                "Нужно минимум две заполненные строки «пиксель → длина волны»."
            )
        return pixels, wavelengths

    def _set_calibration_rows(self, points: list[dict[str, float]]) -> None:
        self.calibration_table.setRowCount(0)
        if not points:
            for pixel, wavelength in CALIBRATION_EXAMPLE:
                row = self.calibration_table.rowCount()
                self.calibration_table.insertRow(row)
                self.calibration_table.setItem(row, 0, QTableWidgetItem(str(pixel)))
                self.calibration_table.setItem(
                    row, 1, QTableWidgetItem(f"{wavelength:g}")
                )
            return
        for point in points:
            row = self.calibration_table.rowCount()
            self.calibration_table.insertRow(row)
            pixel = float(point.get("pixel", row))
            wavelength = float(point.get("wavelength_nm", 0.0))
            self.calibration_table.setItem(row, 0, QTableWidgetItem(f"{pixel:g}"))
            self.calibration_table.setItem(
                row, 1, QTableWidgetItem(f"{wavelength:g}")
            )

    def _build_calibration(self) -> None:
        try:
            pixels, wavelengths = self._collect_calibration_points()
        except ValueError as error:
            self._warn("Калибровка не построена", str(error))
            return
        degree = int(self.degree_spin.value())
        if len(pixels) < degree + 1:
            self._warn(
                "Недостаточно точек",
                f"Для полинома степени {degree} нужно минимум {degree + 1} линии, "
                f"заполнено {len(pixels)}.",
            )
            return
        pixel_count = self._pixel_count()
        try:
            calibration = calibrate_from_lines(
                pixels, wavelengths, degree=degree, pixel_count=pixel_count
            )
        except ValueError as error:
            self._error("Калибровка не построена", str(error))
            return
        self._calibration = calibration
        self._refresh_calibration_ui()
        self._reset_fit()
        self._clear_peak_results()
        if self._apply_calibration:
            self._axis_signature = None
        self.statusBar().showMessage(
            f"Калибровка построена: RMS {calibration.rms_nm:.3f} нм "
            f"на {pixel_count} пикселях.",
            8000,
        )

    def _refresh_calibration_ui(self) -> None:
        calibration = self._calibration
        if calibration is None:
            self.calibration_rms_label.setText("—")
            self.calibration_max_error_label.setText("—")
            self.calibration_range_label.setText("—")
            self.calibration_points_label.setText("—")
        else:
            low, high = calibration.wavelength_range
            self.calibration_rms_label.setText(f"{calibration.rms_nm:.3f} нм")
            self.calibration_max_error_label.setText(
                f"{calibration.max_error_nm:.3f} нм"
            )
            self.calibration_range_label.setText(f"{low:.1f}–{high:.1f} нм")
            self.calibration_points_label.setText(
                f"{len(calibration.points)} · {calibration.pixel_count} px"
            )
        if self._apply_calibration and calibration is not None:
            text = (
                f"Калибровка: полином {calibration.degree}° · "
                f"RMS {calibration.rms_nm:.2f} нм"
            )
        elif calibration is not None:
            text = "Калибровка: построена, к оси не применяется"
        else:
            text = "Калибровка: паспортная 340–1050 нм"
        self.calibration_status_label.setText(text)

    def _on_apply_calibration_toggled(self, checked: bool) -> None:
        if checked and self._calibration is None:
            self.apply_calibration_checkbox.blockSignals(True)
            self.apply_calibration_checkbox.setChecked(False)
            self.apply_calibration_checkbox.blockSignals(False)
            self._warn(
                "Калибровка не построена",
                "Сначала постройте калибровку по эталонным линиям "
                "или загрузите сохранённый JSON.",
            )
            return
        self._apply_calibration = checked
        self._axis_signature = None
        self._axis_warning_shown = False
        self._reset_fit()
        self._clear_peak_results()
        self._refresh_calibration_ui()
        self.statusBar().showMessage(
            "Калибровка применяется к оси длин волн."
            if checked
            else "Используется паспортная ось 340–1050 нм.",
            5000,
        )

    def _save_calibration(self) -> None:
        if self._calibration is None:
            self._warn(
                "Нет калибровки",
                "Сначала постройте калибровку по эталонным линиям.",
            )
            return
        path = self._ask_save_path(
            "Сохранить калибровку",
            "calibration.json",
            "JSON (*.json);;Все файлы (*)",
        )
        if path is None:
            return
        try:
            self._calibration.save(path)
        except OSError as error:
            self._error("Не удалось сохранить калибровку", str(error))
            return
        self.statusBar().showMessage(f"Калибровка сохранена: {path}", 8000)

    def _load_calibration(self) -> None:
        path_text, _filter = QFileDialog.getOpenFileName(
            self,
            "Загрузить калибровку",
            str(Path.cwd() / "calibration.json"),
            "JSON (*.json);;Все файлы (*)",
        )
        if not path_text:
            return
        try:
            calibration = Calibration.load(path_text)
        except (OSError, ValueError, KeyError) as error:
            self._error("Не удалось загрузить калибровку", str(error))
            return
        self._calibration = calibration
        self.degree_spin.setValue(min(3, max(1, calibration.degree)))
        self._set_calibration_rows(calibration.points)
        self._refresh_calibration_ui()
        self._reset_fit()
        self._clear_peak_results()
        if self._apply_calibration:
            self._axis_signature = None
        self.statusBar().showMessage(f"Калибровка загружена: {path_text}", 8000)

    # ------------------------------------------------------------------
    # спектральные линии
    # ------------------------------------------------------------------
    def _selected_elements(self) -> str | tuple[str, ...]:
        text = self.element_combo.currentText()
        if text == ALL_ELEMENTS:
            return tuple(ELEMENT_CHOICES)
        return text

    def _clear_peak_results(self) -> None:
        self._last_peaks = []
        self.peaks_table.setRowCount(0)
        self._peak_scatter.set_offsets(np.empty((0, 2), dtype=float))
        self.lines_summary_label.setText(
            "Нажмите «Найти линии», чтобы определить спектральные линии."
        )

    def _find_lines(self) -> None:
        spectrum = self._make_spectrum()
        if spectrum is None:
            self._warn(
                "Нет данных",
                "Нечего анализировать: подключите источник и дождитесь кадра.",
            )
            return
        peak_value = float(np.max(spectrum.intensity))
        if not np.isfinite(peak_value) or peak_value <= 0:
            self._warn(
                "Нет сигнала",
                "Спектр пустой или постоянный — линии искать негде. "
                "Проверьте освещённость сенсора.",
            )
            return

        threshold = float(self.threshold_spin.value()) / 100.0
        try:
            peaks = detect_peaks(spectrum, prominence=threshold * peak_value)
        except ValueError as error:
            self._error("Поиск линий не удался", str(error))
            return

        elements = self._selected_elements()
        tolerance = float(self.tolerance_spin.value())
        try:
            matches = identify(
                [float(peak["wavelength_nm"]) for peak in peaks], elements, tolerance
            )
        except (ValueError, TypeError):
            matches = []
        by_wavelength = {
            round(float(match["measured_nm"]), 4): match for match in matches
        }

        self._last_peaks = list(peaks)
        self._fill_peaks_table(peaks, by_wavelength)
        if peaks:
            self._peak_scatter.set_offsets(
                np.array(
                    [
                        [float(peak["wavelength_nm"]), float(peak["intensity"])]
                        for peak in peaks
                    ],
                    dtype=float,
                )
            )
            mode_text = (
                "по калибровке"
                if self._apply_calibration and self._calibration is not None
                else "по паспортной оси"
            )
            self.lines_summary_label.setText(
                f"Найдено линий: {len(peaks)} · распознано: {len(matches)} · "
                f"{mode_text}."
            )
            self.statusBar().showMessage(f"Найдено линий: {len(peaks)}", 5000)
        else:
            self._clear_peak_results()
            self.lines_summary_label.setText(
                "Линии не найдены — уменьшите порог или увеличьте контраст."
            )
        self.canvas.draw_idle()

    def _fill_peaks_table(
        self,
        peaks: list[dict[str, float]],
        by_wavelength: dict[float, dict[str, float | str]],
    ) -> None:
        table = self.peaks_table
        table.setRowCount(len(peaks))
        for row, peak in enumerate(peaks):
            wavelength = float(peak.get("wavelength_nm", 0.0))
            intensity = float(peak.get("intensity", 0.0))
            match = by_wavelength.get(round(wavelength, 4))
            element_text = str(match["element"]) if match else "—"
            line_text = f"{float(match['reference_nm']):.2f}" if match else "—"
            error_text = f"{float(match['error_nm']):+.2f}" if match else "—"
            values = (
                str(row + 1),
                f"{wavelength:.1f}",
                f"{intensity:.1f}",
                _format_number(peak.get("fwhm_nm"), 2),
                element_text,
                line_text,
                error_text,
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)

    # ------------------------------------------------------------------
    # протокол
    # ------------------------------------------------------------------
    def _device_name(self) -> str:
        if self._camera is not None:
            return self._camera.name
        return "источник не подключён"

    def _make_spectrum(self) -> Spectrum | None:
        if self._wavelengths is None or self._intensity is None:
            return None
        wavelengths = np.asarray(self._wavelengths, dtype=float)
        values = np.asarray(self._intensity, dtype=float)
        if wavelengths.size != values.size or values.size < 2:
            return None
        if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(values)):
            return None
        try:
            return Spectrum(
                wavelengths,
                values,
                label=self._device_name(),
                meta={"device": self._device_name()},
            )
        except ValueError:
            return None

    def _make_measurement(self) -> Measurement | None:
        spectrum = self._make_spectrum()
        if spectrum is None:
            return None
        return Measurement(
            spectrum=spectrum,
            fit=self._last_fit,
            peaks=list(self._last_peaks),
            calibration=self._calibration,
            device=self._device_name(),
            operator=self.operator_edit.text().strip(),
            notes=self.notes_edit.toPlainText().strip(),
        )

    def _refresh_protocol(self) -> None:
        measurement = self._make_measurement()
        if measurement is None:
            self._warn(
                "Нет данных для протокола",
                "Подключите источник, дождитесь кадра и повторите.",
            )
            return
        self.report_edit.setPlainText(measurement.report_markdown())
        self.statusBar().showMessage("Протокол обновлён.", 4000)

    def _ask_save_path(
        self, title: str, default_name: str, file_filter: str
    ) -> Path | None:
        path_text, _filter = QFileDialog.getSaveFileName(
            self, title, str(Path.cwd() / default_name), file_filter
        )
        return Path(path_text) if path_text else None

    def _save_protocol(self, kind: str) -> None:
        measurement = self._make_measurement()
        if measurement is None:
            self._warn(
                "Нет данных",
                "Протокол не сформирован: нет спектра. Подключите источник.",
            )
            return
        if kind == "md":
            path = self._ask_save_path(
                "Сохранить протокол", "protocol.md", "Markdown (*.md);;Все файлы (*)"
            )
        elif kind == "json":
            path = self._ask_save_path(
                "Сохранить измерение",
                "measurement.json",
                "JSON (*.json);;Все файлы (*)",
            )
        else:
            path = self._ask_save_path(
                "Сохранить спектр", "spectrum.csv", "CSV (*.csv);;Все файлы (*)"
            )
        if path is None:
            return
        try:
            if kind == "md":
                text = self.report_edit.toPlainText()
                if text.strip() and text != measurement.report_markdown():
                    path.write_text(text, encoding="utf-8")
                else:
                    measurement.save_report(path)
            elif kind == "json":
                measurement.save_json(path)
            else:
                measurement.save_csv(path)
        except (OSError, ValueError) as error:
            self._error("Не удалось сохранить файл", str(error))
            return
        self.statusBar().showMessage(f"Сохранено: {path}", 8000)

    # ------------------------------------------------------------------
    # сообщения и завершение
    # ------------------------------------------------------------------
    def _warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _error(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Корректно освободить камеру и остановить таймер при закрытии окна."""
        self._timer.stop()
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:  # noqa: BLE001 - не мешать закрытию приложения
                pass
            self._camera = None
        super().closeEvent(event)


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить графическое приложение и вернуть код завершения."""
    arguments = list(sys.argv if argv is None else argv)
    app = QApplication.instance()
    if app is None:
        app = QApplication(arguments)
    app.setApplicationName("СпектроЛаб")
    app.setApplicationDisplayName("СпектроЛаб")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(DARK_STYLESHEET)
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()
    return int(app.exec())
