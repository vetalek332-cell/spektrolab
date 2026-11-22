import sys
import cv2
import numpy as np
import os
import time
from datetime import datetime
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import simpson

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel,
    QHBoxLayout, QGroupBox, QComboBox, QStatusBar, QMessageBox, QTabWidget,
    QSpinBox, QDoubleSpinBox, QFileDialog, QSplitter, QCheckBox, QFrame,
    QTableWidget, QTableWidgetItem, QScrollArea, QSizePolicy, QProgressBar,
    QToolBar, QDockWidget
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QSize
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPalette, QIcon, QAction
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PolyCollection

# Добавим функцию для аппроксимации Планка
def planck_law(wavelength_nm, T, scale=1.0, offset=0.0):
    """Функция закона Планка для аппроксимации"""
    # Физические константы
    h = 6.62607015e-34  # Постоянная Планка
    c = 299792458       # Скорость света
    k = 1.380649e-23    # Постоянная Больцмана
    
    wavelength_m = wavelength_nm * 1e-9
    exp_arg = (h * c) / (wavelength_m * k * T)
    
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        intensity = (2 * h * c ** 2 / (wavelength_m ** 5)) * (1 / (np.exp(exp_arg) - 1))
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Нормализуем и применяем масштабирование
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity) * scale + offset
    
    return intensity

class PlanckCalculator(QObject):
    """Класс для вычисления закона Планка с кэшированием"""
    calculated = Signal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.h = 6.62607015e-34
        self.c = 299792458
        self.k = 1.380649e-23

    def calculate(self, temperature: float):
        wavelengths = np.linspace(340, 1050, 1000)
        wavelengths_m = wavelengths * 1e-9
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            exp_arg = (self.h * self.c) / (wavelengths_m * self.k * temperature)
            intensity = (2 * self.h * self.c ** 2 / (wavelengths_m ** 5)) * \
                        (1 / (np.exp(exp_arg) - 1))
            intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
        return wavelengths, intensity

class SpectrometerWorker(QObject):
    frame_ready = Signal(np.ndarray)
    spectrum_ready = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._running = False
        self._paused = False
        self._dark_data = None
        self._reference_data = None
        self.roi = (100, 100, 600, 100)
        self.spectrum_center = None
        self.wavelengths = np.linspace(390, 1100, 1024)
        self.wavelength_shift = 0  # Сдвиг спектра в нм

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def pause(self, state):
        self._paused = state

    def set_roi(self, roi):
        self.roi = roi
        self.spectrum_center = roi[1] + roi[3] // 2

    def set_dark_data(self, data):
        self._dark_data = data

    def set_reference_data(self, data):
        self._reference_data = data

    def set_wavelength_shift(self, shift):
        self.wavelength_shift = shift

    def process_frame(self, frame, return_data=False):
        if not self._running or self._paused or frame is None:
            return None
        try:
            x, y, w, h = self.roi
            roi_frame = frame[y:y + h, x:x + w]
            if roi_frame.size == 0:
                return None

            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

            if self.spectrum_center is not None:
                center_in_roi = self.spectrum_center - y
                strip_height = 10
                y_start = max(0, center_in_roi - strip_height // 2)
                y_end = min(h, center_in_roi + strip_height // 2)
                spectrum = np.mean(gray[y_start:y_end, :], axis=0)
            else:
                spectrum = np.mean(gray, axis=0)

            if len(spectrum) != len(self.wavelengths):
                spectrum = np.interp(
                    np.linspace(0, len(spectrum) - 1, len(self.wavelengths)),
                    np.arange(len(spectrum)),
                    spectrum
                )

            if return_data:
                return spectrum.copy()
            else:
                self.spectrum_ready.emit(spectrum)

        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            return None

class SpectrometerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор спектра PRO v4.0")
        self.resize(1800, 1000)

        # Физические константы
        self.h = 6.62607015e-34
        self.c = 299792458
        self.k = 1.380649e-23
        self.sigma = 5.670374419e-8
        self.wien_const = 2.8977719e-3

        # Параметры
        self.wavelengths = np.linspace(390, 1100, 1024)
        self.integration_time = 100
        self.temperature = 5000
        self.connected = False
        self.paused = False
        self.fill_under_curve = True
        self.auto_calc_temp = True
        self.show_planck_on_spectrum = False
        self.show_boltzmann_law = False  # Выключено по умолчанию
        self.show_wien_law = False       # Выключено по умолчанию
        self.wavelength_shift = 0  # Сдвиг спектра в нм

        # Данные
        self.spectrum_data = np.zeros(len(self.wavelengths))
        self.raw_spectrum = None
        self.dark_data = None
        self.reference_data = None
        self.planck_data = None
        self.last_save_path = os.path.expanduser("~")

        # Настройки
        self.realtime_plot = True
        self.peak_detection_enabled = False
        self.auto_peak_params = True

        # Камера
        self.cap = None
        self.frame = None
        self.roi = (0, 220, 640, 90)
        self.spectrum_center = None
        self.mirror_camera = False
        self._updating_plot = False 
        self.zoom_enabled = False
        self.rectangle_selector = None
        self.current_xlim = (340, 1050)  # Начальные границы по X
        self.current_ylim = (0, 1)       # Начальные границы по Y
        
        # Аппроксимация
        self.approximation_results = {}  # Для хранения результатов аппроксимации
        
        # Временные данные
        self.time_data = []  # Список для хранения временных данных
        self.time_spectra = []  # Список спектров во времени
        self.time_timestamps = []  # Временные метки
        self.recording_time_data = False
        self.recording_start_time = None
        self.recording_interval = 1000  # Интервал записи в мс
        self.time_approximation_results = {}  # Результаты аппроксимации для временных данных
        
        # Инициализация
        self._planck_cache = {}
        self.peaks = []
        self.peak_positions = []
        self.peak_properties = {}
        self.show_peak_lines = True
        self.continuous_measurement = False
        self.create_colormap()
        self.init_timers()
        self.init_ui()
        self.init_workers()
        
        # Автоподключение к камере при запуске
        QTimer.singleShot(1000, self.auto_connect_camera)

    def auto_connect_camera(self):
        """Автоматическое подключение к камере при запуске"""
        self.refresh_cameras()
        if self.camera_combo.count() > 0 and self.camera_combo.currentText() != "Камеры не найдены":
            self.connect_camera()
            # Установка максимального разрешения
            self.camera_size_combo.setCurrentText("Макс. разрешение")
            self.change_camera_resolution()

    def init_workers(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=6)

        self.worker = SpectrometerWorker()
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.spectrum_ready.connect(self.process_spectrum_data)
        self.worker_thread.start()

        self.planck_calculator = PlanckCalculator()
        self.planck_thread = QThread()
        self.planck_calculator.moveToThread(self.planck_thread)
        self.planck_thread.start()

    def init_timers(self):
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.capture_frame)
        self.camera_timer.start(30)  # ~30 FPS

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(30)  # 10 FPS

        self.peak_timer = QTimer()
        self.peak_timer.setSingleShot(True)
        self.peak_timer.timeout.connect(self.detect_peaks)
        
        # Таймер для записи временных данных
        self.time_recording_timer = QTimer()
        self.time_recording_timer.timeout.connect(self.record_time_data)

    def create_colormap(self):
        colors = [
            (0.3, 0.0, 0.5),  # Фиолетовый (340 нм)
            (0.0, 0.0, 1.0),  # Синий (450 нм)
            (0.0, 1.0, 1.0),  # Голубой (500 нм)
            (0.0, 1.0, 0.0),  # Зеленый (550 нм)
            (1.0, 1.0, 0.0),  # Желтый (600 нм)
            (1.0, 0.6, 0.0),  # Оранжевый (650 нм)
            (1.0, 0.0, 0.0),  # Красный (700 нм)
            (0.5, 0.0, 0.0)   # Темно-красный (1050 нм)
        ]
        from matplotlib.colors import LinearSegmentedColormap
        self.colormap = LinearSegmentedColormap.from_list("spectrum", colors)

    @staticmethod
    def wavelength_to_color(wl):
        wl = np.clip(wl, 380, 780)
        if wl < 440:
            r = (440 - wl) / 60
            g = 0.0
            b = 1.0
        elif wl < 490:
            r = 0.0
            g = (wl - 440) / 50
            b = 1.0
        elif wl < 510:
            r = 0.0
            g = 1.0
            b = (510 - wl) / 20
        elif wl < 580:
            r = (wl - 510) / 70
            g = 1.0
            b = 0.0
        elif wl < 645:
            r = 1.0
            g = (645 - wl) / 65
            b = 0.0
        else:
            r = 1.0
            g = 0.0
            b = 0.0

        if wl > 700:
            factor = 0.3 + 0.7 * (780 - wl) / 80
        elif wl < 420:
            factor = 0.3 + 0.7 * (wl - 380) / 40
        else:
            factor = 1.0

        return (r * factor, g * factor, b * factor)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.create_toolbar()
        self.create_settings_docks()

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setDocumentMode(True)

        self.camera_tab = self.create_camera_tab()
        self.tab_widget.addTab(self.camera_tab, "Спектр")

        self.peaks_tab = self.create_peaks_tab()
        self.tab_widget.addTab(self.peaks_tab, "Пики")

        self.wien_tab = self.create_wien_tab()
        self.tab_widget.addTab(self.wien_tab, "Закон Вина")

        self.boltzmann_tab = self.create_boltzmann_tab()
        self.tab_widget.addTab(self.boltzmann_tab, "Закон Больцмана")
        
        # Новая вкладка для временных графиков
        self.time_series_tab = self.create_time_series_tab()
        self.tab_widget.addTab(self.time_series_tab, "Временные графики")

        self.log_tab = self.create_log_tab()
        self.tab_widget.addTab(self.log_tab, "Лог")

        main_layout.addWidget(self.tab_widget)

        self.status_bar = QStatusBar()
        self.status_bar.setFont(QFont("Segoe UI", 9))
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(250)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)

        self.apply_style("Темный")

    def create_time_series_tab(self):
        """Создает вкладку для временных графиков"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Панель управления записью
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        
        self.record_btn = QPushButton("Начать запись")
        self.record_btn.clicked.connect(self.toggle_time_recording)
        self.record_btn.setCheckable(True)
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 60000)
        self.interval_spin.setValue(1000)
        self.interval_spin.setSuffix(" мс")
        self.interval_spin.valueChanged.connect(self.update_recording_interval)
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 3600)
        self.duration_spin.setValue(60)
        self.duration_spin.setSuffix(" сек")
        
        control_layout.addWidget(QLabel("Интервал:"))
        control_layout.addWidget(self.interval_spin)
        control_layout.addWidget(QLabel("Длительность:"))
        control_layout.addWidget(self.duration_spin)
        control_layout.addWidget(self.record_btn)
        control_layout.addStretch()
        
        # График временных данных
        self.time_figure = Figure(figsize=(10, 6), dpi=100)
        self.time_canvas = FigureCanvas(self.time_figure)
        self.time_ax = self.time_figure.add_subplot(111)
        self.time_ax.grid(True, linestyle=':', alpha=0.5)
        self.time_ax.set_xlabel("Время (сек)")
        self.time_ax.set_ylabel("Интенсивность")
        
        # Кнопки для анализа
        analyze_frame = QFrame()
        analyze_layout = QHBoxLayout(analyze_frame)
        
        self.approximate_time_btn = QPushButton("Аппроксимировать Планком")
        self.approximate_time_btn.clicked.connect(self.approximate_time_data)
        
        self.clear_time_btn = QPushButton("Очистить данные")
        self.clear_time_btn.clicked.connect(self.clear_time_data)
        
        self.save_time_btn = QPushButton("Сохранить данные")
        self.save_time_btn.clicked.connect(self.save_time_data)
        
        analyze_layout.addWidget(self.approximate_time_btn)
        analyze_layout.addWidget(self.clear_time_btn)
        analyze_layout.addWidget(self.save_time_btn)
        analyze_layout.addStretch()
        
        # Таблица для отображения параметров аппроксимации
        self.time_table = QTableWidget(0, 4)
        self.time_table.setHorizontalHeaderLabels(["Время", "Температура", "R²", "Статус"])
        self.time_table.horizontalHeader().setStretchLastSection(True)
        
        # Компоновка
        layout.addWidget(control_frame)
        layout.addWidget(self.time_canvas)
        layout.addWidget(analyze_frame)
        layout.addWidget(self.time_table)
        
        return tab

    def toggle_time_recording(self):
        """Включение/выключение записи временных данных"""
        if not self.connected:
            QMessageBox.warning(self, "Ошибка", "Камера не подключена")
            self.record_btn.setChecked(False)
            return
            
        self.recording_time_data = not self.recording_time_data
        
        if self.recording_time_data:
            # Начало записи
            self.recording_start_time = time.time()
            self.time_data = []
            self.time_spectra = []
            self.time_timestamps = []
            self.time_recording_timer.start(self.recording_interval)
            self.record_btn.setText("Остановить запись")
            self.log_message("Начата запись временных данных")
        else:
            # Остановка записи
            self.time_recording_timer.stop()
            self.record_btn.setText("Начать запись")
            self.log_message(f"Запись временных данных остановлена. Записано {len(self.time_data)} точек")
            self.update_time_plot()

    def update_recording_interval(self):
        """Обновление интервала записи"""
        self.recording_interval = self.interval_spin.value()
        if self.recording_time_data:
            self.time_recording_timer.setInterval(self.recording_interval)

    def record_time_data(self):
        """Запись одной точки временных данных"""
        if not self.connected or self.frame is None:
            return
            
        # Проверяем длительность записи
        current_time = time.time()
        elapsed = current_time - self.recording_start_time
        if elapsed > self.duration_spin.value():
            self.toggle_time_recording()
            return
            
        # Получаем текущий спектр
        spectrum = self.worker.process_frame(self.frame, return_data=True)
        if spectrum is None:
            return
            
        # Вычисляем интегральную интенсивность
        total_intensity = np.trapz(spectrum, self.wavelengths)
        
        # Сохраняем данные
        self.time_data.append(total_intensity)
        self.time_spectra.append(spectrum.copy())
        self.time_timestamps.append(elapsed)
        
        # Обновляем график
        if len(self.time_data) % 5 == 0:  # Обновляем каждые 5 точек для производительности
            self.update_time_plot()

    def update_time_plot(self):
        """Обновление графика временных данных"""
        if not hasattr(self, 'time_ax') or len(self.time_data) == 0:
            return
            
        self.time_ax.clear()
        
        # Рисуем исходные данные
        self.time_ax.plot(self.time_timestamps, self.time_data, 'b-', 
                         label='Интегральная интенсивность', alpha=0.7)
        
        # Рисуем аппроксимацию, если есть
        if hasattr(self, 'time_approximation_results') and self.time_approximation_results:
            for result in self.time_approximation_results.values():
                if result['success']:
                    x_fit = result['x_fit']
                    y_fit = result['y_fit']
                    self.time_ax.plot(x_fit, y_fit, 'r--', 
                                    label=f'Аппроксимация (T={result["temperature"]:.0f}K)')
        
        self.time_ax.set_xlabel("Время (сек)")
        self.time_ax.set_ylabel("Интенсивность")
        self.time_ax.legend()
        self.time_ax.grid(True)
        
        self.apply_style_to_axes(self.time_ax, self.time_figure)
        self.time_canvas.draw_idle()

    def approximate_time_data(self):
        """Аппроксимация временных данных законом Планка"""
        if len(self.time_spectra) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных для аппроксимации")
            return
            
        try:
            # Очищаем предыдущие результаты
            self.time_approximation_results = {}
            self.time_table.setRowCount(0)
            
            for i, (timestamp, spectrum) in enumerate(zip(self.time_timestamps, self.time_spectra)):
                # Аппроксимируем каждый спектр
                try:
                    result = self.approximate_single_spectrum(spectrum, f"t={timestamp:.1f}s")
                    self.time_approximation_results[i] = result
                    
                    # Добавляем в таблицу
                    row = self.time_table.rowCount()
                    self.time_table.insertRow(row)
                    
                    self.time_table.setItem(row, 0, QTableWidgetItem(f"{timestamp:.1f}"))
                    self.time_table.setItem(row, 1, QTableWidgetItem(f"{result['temperature']:.1f}"))
                    self.time_table.setItem(row, 2, QTableWidgetItem(f"{result.get('r_squared', 0):.3f}"))
                    self.time_table.setItem(row, 3, QTableWidgetItem("Успешно" if result['success'] else "Ошибка"))
                    
                except Exception as e:
                    self.log_message(f"Ошибка аппроксимации для t={timestamp:.1f}s: {e}")
            
            # Обновляем график
            self.update_time_plot()
            self.log_message(f"Аппроксимация завершена для {len(self.time_approximation_results)} спектров")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка аппроксимации: {e}")
            self.log_message(f"Ошибка аппроксимации временных данных: {e}")

    def approximate_single_spectrum(self, spectrum, label):
        """Аппроксимация одного спектра законом Планка"""
        try:
            # Подготовка данных
            x_data = self.wavelengths
            y_data = spectrum
            
            # Нормализация
            y_norm = y_data / np.max(y_data) if np.max(y_data) > 0 else y_data
            
            # Начальные параметры
            T_guess = self.temperature if self.temperature > 0 else 5000
            scale_guess = np.max(y_data)
            offset_guess = np.min(y_data)
            
            # Границы параметров
            bounds = (
                [100, 0.1 * scale_guess, -0.1 * scale_guess],
                [20000, 10 * scale_guess, 0.1 * scale_guess]
            )
            
            # Аппроксимация
            popt, pcov = curve_fit(
                planck_law, 
                x_data, 
                y_data,
                p0=[T_guess, scale_guess, offset_guess],
                bounds=bounds,
                maxfev=10000
            )
            
            T_fit, scale_fit, offset_fit = popt
            perr = np.sqrt(np.diag(pcov)) if np.any(pcov) else [0, 0, 0]
            
            # Вычисление R²
            y_pred = planck_law(x_data, T_fit, scale_fit, offset_fit)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'temperature': T_fit,
                'scale': scale_fit,
                'offset': offset_fit,
                'errors': perr,
                'r_squared': r_squared,
                'success': True,
                'x_fit': x_data,
                'y_fit': y_pred
            }
            
        except Exception as e:
            return {
                'temperature': 0,
                'scale': 0,
                'offset': 0,
                'errors': [0, 0, 0],
                'r_squared': 0,
                'success': False,
                'error_message': str(e)
            }

    def clear_time_data(self):
        """Очистка временных данных"""
        self.time_data = []
        self.time_spectra = []
        self.time_timestamps = []
        self.time_approximation_results = {}
        self.time_table.setRowCount(0)
        self.update_time_plot()
        self.log_message("Временные данные очищены")

    def save_time_data(self):
        """Сохранение временных данных"""
        if len(self.time_data) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
            
        filename = f"time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить временные данные",
            os.path.join(self.last_save_path, filename),
            "CSV (*.csv);;Все файлы (*)"
        )
        
        if path:
            try:
                # Сохраняем основные данные
                with open(path, 'w') as f:
                    f.write("Time(s),Intensity,Temperature(K),R_squared\n")
                    for i, (t, intensity) in enumerate(zip(self.time_timestamps, self.time_data)):
                        temp = self.time_approximation_results[i]['temperature'] if i in self.time_approximation_results else 0
                        r2 = self.time_approximation_results[i].get('r_squared', 0) if i in self.time_approximation_results else 0
                        f.write(f"{t:.3f},{intensity:.3f},{temp:.1f},{r2:.3f}\n")
                
                # Сохраняем спектры в отдельный файл
                spec_path = path.replace('.csv', '_spectra.npy')
                np.save(spec_path, np.array(self.time_spectra))
                
                self.log_message(f"Временные данные сохранены: {path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения: {e}")
                self.log_message(f"Ошибка сохранения временных данных: {e}")

    def create_toolbar(self):
        toolbar = QToolBar("Основные инструменты")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Добавляем существующие действия
        toolbar.addActions(self.create_device_actions())
        toolbar.addSeparator()
        toolbar.addActions(self.create_measurement_actions())
        toolbar.addSeparator()
        toolbar.addActions(self.create_view_actions())
        toolbar.addSeparator()
        toolbar.addActions(self.create_save_actions())
        
        # Добавляем новый разделитель и кнопку выбора стиля
        toolbar.addSeparator()
        toolbar.addWidget(self.create_style_selector())
        
        # Добавляем кнопку аппроксимации
        toolbar.addSeparator()
        self.approximate_action = QAction(QIcon.fromTheme("edit-find"), "Аппроксимировать Планком", self)
        self.approximate_action.triggered.connect(self.approximate_with_planck)
        toolbar.addAction(self.approximate_action)
        
        return toolbar

    def approximate_with_planck(self):
        """Аппроксимация текущего спектра законом Планка"""
        if len(self.spectrum_data) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных спектра для аппроксимации")
            return
            
        try:
            # Подготовка данных для аппроксимации
            x_data = self.wavelengths
            y_data = self.spectrum_data
            
            # Нормализация данных
            y_norm = y_data / np.max(y_data) if np.max(y_data) > 0 else y_data
            
            # Начальные параметры для аппроксимации
            T_guess = self.temperature if self.temperature > 0 else 5000
            scale_guess = np.max(y_data)
            offset_guess = np.min(y_data)
            
            # Границы параметров
            bounds = (
                [100, 0.1 * scale_guess, -0.1 * scale_guess],  # нижние границы
                [20000, 10 * scale_guess, 0.1 * scale_guess]   # верхние границы
            )
            
            # Аппроксимация
            popt, pcov = curve_fit(
                planck_law, 
                x_data, 
                y_data,
                p0=[T_guess, scale_guess, offset_guess],
                bounds=bounds,
                maxfev=10000
            )
            
            # Извлечение параметров
            T_fit, scale_fit, offset_fit = popt
            perr = np.sqrt(np.diag(pcov)) if np.any(pcov) else [0, 0, 0]
            
            # Сохранение результатов
            self.approximation_results = {
                'temperature': T_fit,
                'scale': scale_fit,
                'offset': offset_fit,
                'errors': perr,
                'success': True
            }
            
            # Обновление температуры
            self.temperature = T_fit
            self.temp_spin.setValue(T_fit)
            
            # Обновление графиков
            self.update_planck_curve()
            self.update_plot()
            
            # Сообщение о результате
            self.log_message(f"Аппроксимация Планком: T={T_fit:.1f}±{perr[0]:.1f} K")
            self.update_status(f"Аппроксимация: T={T_fit:.1f} K")
            
        except Exception as e:
            self.log_message(f"Ошибка аппроксимации: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось выполнить аппроксимацию: {e}")

    def create_style_selector(self):
        """Создает выпадающий список для выбора стиля интерфейса"""
        style_combo = QComboBox()
        style_combo.addItems(["Темный", "Светлый", "Синий"])
        style_combo.setCurrentText("Темный")
        style_combo.currentTextChanged.connect(self.apply_style)
        style_combo.setToolTip("Выберите стиль интерфейса")
        style_combo.setMinimumWidth(100)
        return style_combo

    def create_device_actions(self):
        self.connect_action = QAction(QIcon.fromTheme("camera-web"), "Подключить камеру", self)
        self.connect_action.triggered.connect(self.toggle_camera_connection)
        self.connect_action.setCheckable(True)

        self.refresh_action = QAction(QIcon.fromTheme("view-refresh"), "Обновить список камер", self)
        self.refresh_action.triggered.connect(self.refresh_cameras)

        return [self.connect_action, self.refresh_action]

    def create_measurement_actions(self):
        self.measure_action = QAction(QIcon.fromTheme("media-record"), "Измерить", self)
        self.measure_action.triggered.connect(self.start_measurement)

        self.temp_measure_action = QAction(QIcon.fromTheme("media-playback-start"), "Измерить (временный)", self)
        self.temp_measure_action.triggered.connect(self.temporary_measurement)

        self.dark_action = QAction(QIcon.fromTheme("weather-clear-night"), "Темновой", self)
        self.dark_action.triggered.connect(self.capture_dark)

        self.reference_action = QAction(QIcon.fromTheme("weather-clear"), "Референс", self)
        self.reference_action.triggered.connect(self.capture_reference)

        self.pause_action = QAction(QIcon.fromTheme("media-playback-pause"), "Пауза", self)
        self.pause_action.triggered.connect(self.toggle_pause)
        self.pause_action.setCheckable(True)

        return [self.measure_action, self.temp_measure_action, self.dark_action, self.reference_action, self.pause_action]  
    
    def temporary_measurement(self):
        if not self.connected:
            QMessageBox.warning(self, "Ошибка", "Камера не подключена")
            return
        if self.frame is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных с камеры")
            return

        spectrum = self.worker.process_frame(self.frame, return_data=True)
        if spectrum is None:
            return

        # Не обновляем self.spectrum_data и не вызываем process_spectrum_data
        # Просто добавляем временный спектр в сохранённые
        self.add_spectrum_to_saved(spectrum, temporary=True)

        # Сохраняем график (если нужно)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = os.path.join(os.path.expanduser("~"), "temp_spectra")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_spectrum_{timestamp}.png")

        try:
            self.figure.savefig(temp_path, dpi=150, bbox_inches='tight')
            self.log_message(f"Временный график сохранен: {temp_path}")
        except Exception as e:
            self.log_message(f"Ошибка сохранения временного графика: {e}")

        if self.auto_calc_temp:
            self.calculate_temp_from_spectrum()
        self.update_plot()

    def add_spectrum_to_saved(self, spectrum, temporary=False):
        if temporary:
            label = f"Временный {datetime.now().strftime('%H:%M:%S')}"
            color = f"C{len(self.saved_spectra)}"
        else:
            label = f"Спектр {datetime.now().strftime('%H:%M:%S')}"
            color = f"C{len(self.saved_spectra)}"
            
        check_box = QCheckBox(label)
        check_box.setChecked(True)
        check_box.stateChanged.connect(self.update_plot)

        self.saved_spectra.append((label, spectrum.copy(), color))
        self.spectra_list_layout.insertWidget(0, check_box)

        if len(self.saved_spectra) > 10:
            oldest_widget = self.spectra_list_layout.itemAt(self.spectra_list_layout.count() - 1).widget()
            if oldest_widget:
                oldest_widget.deleteLater()
            self.saved_spectra.pop()

        if not temporary:
            self.log_message(f"Спектр добавлен в сохраненные: {label}")

    def create_view_actions(self):
        self.mirror_action = QAction(QIcon.fromTheme("object-flip-horizontal"), "Отзеркалить", self)
        self.mirror_action.triggered.connect(self.toggle_mirror)
        self.mirror_action.setCheckable(True)

        self.fill_action = QAction(QIcon.fromTheme("format-fill-color"), "Заливка под кривой", self)
        self.fill_action.triggered.connect(self.toggle_fill_under_curve)
        self.fill_action.setCheckable(True)
        self.fill_action.setChecked(True)

        self.peak_lines_action = QAction(QIcon.fromTheme("draw-line"), "Линии пиков", self)
        self.peak_lines_action.triggered.connect(self.toggle_peak_lines)
        self.peak_lines_action.setCheckable(True)
        self.peak_lines_action.setChecked(True)

        # Добавляем новые действия для управления отображением законов
        self.planck_action = QAction(QIcon.fromTheme("view-visible"), "Закон Планка", self)
        self.planck_action.triggered.connect(self.toggle_planck_display)
        self.planck_action.setCheckable(True)
        self.planck_action.setChecked(False)  # Выключено по умолчанию

        self.wien_action = QAction(QIcon.fromTheme("view-visible"), "Закон Вина", self)
        self.wien_action.triggered.connect(self.toggle_wien_law_display)
        self.wien_action.setCheckable(True)
        self.wien_action.setChecked(False)  # Выключено по умолчанию

        self.boltzmann_action = QAction(QIcon.fromTheme("view-visible"), "Закон Больцмана", self)
        self.boltzmann_action.triggered.connect(self.toggle_boltzmann_law_display)
        self.boltzmann_action.setCheckable(True)
        self.boltzmann_action.setChecked(False)  # Выключено по умолчанию

        return [self.mirror_action, self.fill_action, self.peak_lines_action,
                self.planck_action, self.wien_action, self.boltzmann_action]

    def create_save_actions(self):
        self.save_spectrum_action = QAction(QIcon.fromTheme("document-save"), "Сохранить спектр", self)
        self.save_spectrum_action.triggered.connect(self.save_spectrum)

        self.save_plot_action = QAction(QIcon.fromTheme("image-x-generic"), "Сохранить график", self)
        self.save_plot_action.triggered.connect(self.save_plot)

        self.save_all_action = QAction(QIcon.fromTheme("document-save-all"), "Сохранить все", self)
        self.save_all_action.triggered.connect(self.save_all_data)

        return [self.save_spectrum_action, self.save_plot_action, self.save_all_action]

    def create_settings_docks(self):
        # Док-виджет для настроек устройства
        device_dock = QDockWidget("Настройки устройства", self)
        device_dock.setObjectName("DeviceDock")
        device_widget = QWidget()
        device_layout = QVBoxLayout(device_widget)

        self.camera_combo = QComboBox()
        self.camera_size_combo = QComboBox()
        self.camera_size_combo.addItems(["320x240", "640x480", "800x600", "1024x768", "Макс. разрешение"])
        self.camera_size_combo.currentIndexChanged.connect(self.change_camera_resolution)

        device_layout.addWidget(QLabel("Разрешение:"))
        device_layout.addWidget(self.camera_size_combo)
        device_layout.addWidget(QLabel("Камера:"))
        device_layout.addWidget(self.camera_combo)
        device_layout.addStretch()

        device_dock.setWidget(device_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, device_dock)
        device_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        device_dock.hide()

        # Док-виджет для ROI и калибровки
        roi_dock = QDockWidget("ROI и калибровка", self)
        roi_widget = QWidget()
        roi_layout = QVBoxLayout(roi_widget)

        self.roi_x_spin = QSpinBox()
        self.roi_y_spin = QSpinBox()
        self.roi_w_spin = QSpinBox()
        self.roi_h_spin = QSpinBox()

        params = ["X:", "Y:", "Ширина:", "Высота:"]
        spins = [self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin]

        for i, (param, spin) in enumerate(zip(params, spins)):
            row = QHBoxLayout()
            row.addWidget(QLabel(param))
            spin.setRange(0, 2000)
            spin.setValue(self.roi[i])
            spin.valueChanged.connect(self.update_roi)
            row.addWidget(spin)
            roi_layout.addLayout(row)

        # Добавляем настройку сдвига спектра
        self.wavelength_shift_spin = QDoubleSpinBox()
        self.wavelength_shift_spin.setRange(-100, 100)
        self.wavelength_shift_spin.setValue(self.wavelength_shift)
        self.wavelength_shift_spin.setSingleStep(1)
        self.wavelength_shift_spin.valueChanged.connect(self.update_wavelength_shift)
        
        shift_layout = QHBoxLayout()
        shift_layout.addWidget(QLabel("Сдвиг спектра (нм):"))
        shift_layout.addWidget(self.wavelength_shift_spin)
        roi_layout.addLayout(shift_layout)

        self.auto_roi_btn = QPushButton("Авто ROI")
        self.auto_roi_btn.clicked.connect(self.auto_detect_roi)
        roi_layout.addWidget(self.auto_roi_btn)

        calib_btn = QPushButton("Калибровка")
        calib_btn.clicked.connect(self.start_calibration)
        roi_layout.addWidget(calib_btn)

        btn_layout = QHBoxLayout()
        load_calib_btn = QPushButton("Загрузить калибровку")
        load_calib_btn.clicked.connect(self.load_calibration)
        save_calib_btn = QPushButton("Сохранить калибровку")
        save_calib_btn.clicked.connect(self.save_calibration)
        btn_layout.addWidget(load_calib_btn)
        btn_layout.addWidget(save_calib_btn)
        roi_layout.addLayout(btn_layout)

        roi_layout.addStretch()

        roi_dock.setWidget(roi_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, roi_dock)
        roi_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        roi_dock.hide()

        # Док-виджет для настроек измерений
        measure_dock = QDockWidget("Настройки измерений", self)
        measure_widget = QWidget()
        measure_layout = QVBoxLayout(measure_widget)

        self.integration_spin = QSpinBox()
        self.integration_spin.setRange(1, 10000)
        self.integration_spin.setValue(100)
        self.integration_spin.valueChanged.connect(self.update_integration_time)

        self.continuous_check = QCheckBox("Непрерывное измерение")
        self.continuous_check.toggled.connect(self.on_continuous_check_toggled)
        self.realtime_check = QCheckBox("Реальное время")
        self.realtime_check.setChecked(True)
        self.realtime_check.toggled.connect(self.on_realtime_check_toggled)

        measure_layout.addWidget(QLabel("Время интегрирования (мс):"))
        measure_layout.addWidget(self.integration_spin)
        measure_layout.addWidget(self.continuous_check)
        measure_layout.addWidget(self.realtime_check)
        measure_layout.addStretch()

        measure_dock.setWidget(measure_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, measure_dock)
        measure_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        measure_dock.hide()

        # Док-виджет для обработки данных
        process_dock = QDockWidget("Обработка данных", self)
        process_widget = QWidget()
        process_layout = QVBoxLayout(process_widget)

        self.dark_corr_check = QCheckBox("Темновая коррекция")
        self.dark_corr_check.toggled.connect(self.toggle_dark_correction)
        self.dark_corr_check.setEnabled(False)

        self.ref_corr_check = QCheckBox("Референсная коррекция")
        self.ref_corr_check.toggled.connect(self.toggle_reference_correction)
        self.ref_corr_check.setEnabled(False)

        self.normalize_check = QCheckBox("Нормировать спектр")
        self.normalize_check.setChecked(False)
        self.normalize_check.toggled.connect(self.toggle_normalize)

        self.smooth_check = QCheckBox("Сглаживание спектра")
        self.smooth_check.setChecked(True)
        self.smooth_check.toggled.connect(self.toggle_smooth)

        process_layout.addWidget(self.dark_corr_check)
        process_layout.addWidget(self.ref_corr_check)
        process_layout.addWidget(self.normalize_check)
        process_layout.addWidget(self.smooth_check)
        process_layout.addStretch()

        process_dock.setWidget(process_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, process_dock)
        process_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        process_dock.hide()

        # Док-виджет для анализа пиков
        peaks_dock = QDockWidget("Анализ пиков", self)
        peaks_widget = QWidget()
        peaks_layout = QVBoxLayout(peaks_widget)

        self.peak_detect_check = QCheckBox("Обнаружение пиков")
        self.peak_detect_check.toggled.connect(self.toggle_peak_detection)

        self.auto_peak_check = QCheckBox("Автоматические параметры")
        self.auto_peak_check.setChecked(True)
        self.auto_peak_check.toggled.connect(self.toggle_auto_peak_params)

        self.manual_params_frame = QFrame()
        manual_layout = QVBoxLayout(self.manual_params_frame)

        self.peak_height_spin = QSpinBox()
        self.peak_prominence_spin = QSpinBox()
        self.peak_distance_spin = QSpinBox()

        params = [
            ("Минимальная высота:", self.peak_height_spin, 1, 65535, 1000),
            ("Выступ пика:", self.peak_prominence_spin, 1, 65535, 500),
            ("Минимальное расстояние:", self.peak_distance_spin, 1, 100, 10)
        ]

        for label, spin, min_, max_, default in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin.setRange(min_, max_)
            spin.setValue(default)
            spin.valueChanged.connect(self.detect_peaks)
            row.addWidget(spin)
            manual_layout.addLayout(row)

        self.manual_params_frame.setVisible(False)

        peaks_layout.addWidget(self.peak_detect_check)
        peaks_layout.addWidget(self.auto_peak_check)
        peaks_layout.addWidget(self.manual_params_frame)
        peaks_layout.addStretch()

        peaks_dock.setWidget(peaks_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, peaks_dock)
        peaks_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        peaks_dock.hide()

        # Док-виджет для анализа излучения
        radiation_dock = QDockWidget("Анализ излучения", self)
        radiation_widget = QWidget()
        radiation_layout = QVBoxLayout(radiation_widget)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0, 100000)
        self.temp_spin.setValue(self.temperature)
        self.temp_spin.setSingleStep(100)
        self.temp_spin.valueChanged.connect(self.on_temperature_changed)

        self.auto_temp_check = QCheckBox("Автообновление температуры")
        self.auto_temp_check.setChecked(True)
        self.auto_temp_check.toggled.connect(self.toggle_auto_temperature_update)

        self.planck_on_spectrum_check = QCheckBox("Наложить закон Планка")
        self.planck_on_spectrum_check.toggled.connect(self.toggle_planck_display)

        self.calc_temp_btn = QPushButton("Рассчитать T")
        self.calc_temp_btn.clicked.connect(self.calculate_temp_from_spectrum)

        # Добавляем кнопку для аппроксимации Планком
        self.approximate_btn = QPushButton("Аппроксимировать Планком")
        self.approximate_btn.clicked.connect(self.approximate_with_planck)

        radiation_layout.addWidget(QLabel("Температура (K):"))
        radiation_layout.addWidget(self.temp_spin)
        radiation_layout.addWidget(self.auto_temp_check)
        radiation_layout.addWidget(self.planck_on_spectrum_check)
        radiation_layout.addWidget(self.calc_temp_btn)
        radiation_layout.addWidget(self.approximate_btn)
        radiation_layout.addStretch()

        radiation_dock.setWidget(radiation_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, radiation_dock)
        radiation_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        radiation_dock.hide()

        # Док-виджет для сохраненных спектров
        spectra_dock = QDockWidget("Сохраненные спектры", self)
        spectra_widget = QWidget()
        spectra_layout = QVBoxLayout(spectra_widget)

        self.saved_spectra = []
        self.spectra_list_layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(self.spectra_list_layout)
        scroll_area.setWidget(container)
        spectra_layout.addWidget(scroll_area)

        clear_btn = QPushButton("Очистить все")
        clear_btn.clicked.connect(self.clear_saved_spectra)
        spectra_layout.addWidget(clear_btn)

        spectra_dock.setWidget(spectra_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, spectra_dock)
        spectra_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.create_docks_menu()

    def update_wavelength_shift(self):
        """Обновляет сдвиг спектра и пересчитывает данные"""
        new_shift = self.wavelength_shift_spin.value()
        if abs(new_shift - self.wavelength_shift) < 0.1:
            return
            
        self.wavelength_shift = new_shift
        self.worker.set_wavelength_shift(new_shift)
        
        # Пересчитываем длины волн с учетом сдвига
        self.wavelengths = np.linspace(390 + new_shift, 1100 + new_shift, 1024)
        
        self.log_message(f"Установлен сдвиг спектра: {new_shift:.1f} нм")
        
        # Обновляем графики
        if len(self.spectrum_data) > 0:
            self.update_plot()
            self.update_wien_law()
            self.update_boltzmann_law()

    def clear_saved_spectra(self):
        self.saved_spectra.clear()
        while self.spectra_list_layout.count():
            item = self.spectra_list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.update_plot()

    def create_docks_menu(self):
        docks_menu = self.menuBar().addMenu("Окна")
        for dock in self.findChildren(QDockWidget):
            docks_menu.addAction(dock.toggleViewAction())

    def toggle_peak_lines(self, checked):
        self.show_peak_lines = checked
        self.update_plot()

    def toggle_planck_display(self, checked):
        self.show_planck_on_spectrum = checked
        self.planck_action.setChecked(checked)
        self.update_plot()
        self.log_message(f"Отображение закона Планка {'включено' if checked else 'выключено'}")

    def toggle_wien_law_display(self, checked):
        self.show_wien_law = checked
        self.wien_action.setChecked(checked)
        if checked:
            self.update_wien_law()
        else:
            self.wien_ax.clear()
            self.wien_ax.text(0.5, 0.5, "Отображение закона Вина отключено", ha='center', va='center')
            self.wien_canvas.draw_idle()
        self.log_message(f"Отображение закона Вина {'включено' if checked else 'выключено'}")

    def toggle_boltzmann_law_display(self, checked):
        self.show_boltzmann_law = checked
        self.boltzmann_action.setChecked(checked)
        if checked:
            self.update_boltzmann_law()
        else:
            self.boltzmann_ax.clear()
            self.boltzmann_ax.text(0.5, 0.5, "Отображение закона Стефана-Больцмана отключено", ha='center', va='center')
            self.boltzmann_canvas.draw_idle()
        self.log_message(f"Отображение закона Стефана-Больцмана {'включено' if checked else 'выключено'}")

    def save_calibration(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить калибровку", self.last_save_path, "JSON (*.json);;Все файлы (*)")
        if path:
            import json
            calib = {
                "roi": self.roi, 
                "spectrum_center": self.spectrum_center,
                "wavelength_shift": self.wavelength_shift
            }
            try:
                with open(path, 'w') as f:
                    json.dump(calib, f)
                self.log_message(f"Калибровка сохранена в {path}")
                self.last_save_path = os.path.dirname(path)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить калибровку: {e}")

    def load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить калибровку", self.last_save_path, "JSON (*.json);;Все файлы (*)")
        if path:
            import json
            try:
                with open(path, 'r') as f:
                    calib = json.load(f)
                roi = calib.get("roi", self.roi)
                self.roi_x_spin.setValue(roi[0])
                self.roi_y_spin.setValue(roi[1])
                self.roi_w_spin.setValue(roi[2])
                self.roi_h_spin.setValue(roi[3])
                self.spectrum_center = calib.get("spectrum_center", None)
                self.wavelength_shift = calib.get("wavelength_shift", 0)
                self.wavelength_shift_spin.setValue(self.wavelength_shift)
                self.update_roi()
                self.update_wavelength_shift()
                self.log_message(f"Калибровка загружена из {path}")
                self.last_save_path = os.path.dirname(path)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить калибровку: {e}")

    def apply_style(self, style_name):
        """
        Применяет выбранный стиль ко всему интерфейсу и графикам
        Аргументы:
            style_name (str): название стиля ("Темный", "Светлый" или "Синий")
        """
        self.current_style = style_name
        # Устанавливаем параметры стиля для графиков
        if style_name == "Темный":
            # Настройки для темной темы
            self.graph_bgcolor = '#2b2b2b'      # Цвет фона графиков
            self.graph_textcolor = 'white'      # Цвет текста
            self.graph_gridcolor = '#666'       # Цвет сетки
            self.graph_linecolor = '#42a5f5'    # Цвет линий графиков
            
            # Настройки палитры для интерфейса Qt
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
            
        elif style_name == "Светлый":
            # Настройки для светлой темы
            self.graph_bgcolor = 'white'
            self.graph_textcolor = 'black'
            self.graph_gridcolor = '#ccc'
            self.graph_linecolor = '#1976d2'
            
            # Сбрасываем палитру на стандартную
            palette = QApplication.instance().style().standardPalette()
            
        elif style_name == "Синий":
            # Настройки для синей темы
            self.graph_bgcolor = '#1a2b5b'
            self.graph_textcolor = 'white'
            self.graph_gridcolor = '#5a8aff'
            self.graph_linecolor = '#ff9800'
            
            # Настройки палитры для интерфейса Qt
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(30, 60, 120))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(10, 30, 80))
            palette.setColor(QPalette.AlternateBase, QColor(30, 60, 120))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(30, 60, 120))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(100, 150, 255))
            palette.setColor(QPalette.Highlight, QColor(100, 150, 255))
            palette.setColor(QPalette.HighlightedText, Qt.black)
        
        # Применяем палитру к приложению
        QApplication.instance().setPalette(palette)
        
        # Обновляем все графики
        self.update_graph_styles()
        
        # Обновляем статус
        self.log_message(f"Применен стиль: {style_name}")
        
    def update_graph_styles(self):
        """Обновляет стили всех графиков в соответствии с текущей темой"""
        if hasattr(self, 'ax') and self.ax is not None:
            self.apply_style_to_axes(self.ax, self.figure)
        if hasattr(self, 'wien_ax') and self.wien_ax is not None:
            self.apply_style_to_axes(self.wien_ax, self.wien_figure)
        if hasattr(self, 'boltzmann_ax') and self.boltzmann_ax is not None:
            self.apply_style_to_axes(self.boltzmann_ax, self.boltzmann_figure)
        if hasattr(self, 'time_ax') and self.time_ax is not None:
            self.apply_style_to_axes(self.time_ax, self.time_figure)

    def start_calibration(self):
        if not self.connected:
            QMessageBox.warning(self, "Ошибка", "Камера не подключена")
            return
        self.calibration_data = {'original_roi': self.roi, 'frames': []}
        if self.frame is not None:
            self.roi = (0, 0, self.frame.shape[1], self.frame.shape[0])
            self.update_roi()
            self.log_message("Начата калибровка. Наведите камеру на источник света.")
        else:
            self.log_message("Нет кадра для калибровки")
    def auto_detect_roi(self):
        if not self.connected or self.frame is None:
            return
        try:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            vertical_profile = np.mean(gray, axis=1)
            threshold = np.max(vertical_profile) * 0.5
            indices = np.where(vertical_profile > threshold)[0]
            if len(indices) == 0:
                self.log_message("Не удалось определить ROI автоматически")
                return
            y_min, y_max = indices[0], indices[-1]
            height = y_max - y_min
            x = 0
            y = max(0, y_min - 10)
            w = self.frame.shape[1]
            h = min(self.frame.shape[0] - y, height + 20)

            self.roi_x_spin.setValue(x)
            self.roi_y_spin.setValue(y)
            self.roi_w_spin.setValue(w)
            self.roi_h_spin.setValue(h)
            self.spectrum_center = y + h // 2
            self.worker.set_roi((x, y, w, h))
            self.log_message(f"Автоматически определен ROI: X={x}, Y={y}, W={w}, H={h}; Центр спектра: {self.spectrum_center}")
        except Exception as e:
            self.log_message(f"Ошибка автоматического определения ROI: {e}")

    def toggle_dark_correction(self, checked: bool):
        self.log_message(f"Темновая коррекция {'включена' if checked else 'выключена'}")
        self.reprocess_spectrum()

    def toggle_reference_correction(self, checked: bool):
        self.log_message(f"Референсная коррекция {'включена' if checked else 'выключена'}")
        self.reprocess_spectrum()

    def toggle_normalize(self, checked: bool):
        self.log_message(f"Нормализация спектра {'включена' if checked else 'выключена'}")
        self.reprocess_spectrum()

    def toggle_smooth(self, checked: bool):
        self.log_message(f"Сглаживание спектра {'включено' if checked else 'выключено'}")
        self.reprocess_spectrum()

    def reprocess_spectrum(self):
        if self.raw_spectrum is not None and len(self.raw_spectrum) > 0:
            self.process_spectrum_data(self.raw_spectrum)

    def toggle_peak_detection(self, checked: bool):
        self.peak_detection_enabled = checked
        if self.peak_detection_enabled and len(self.spectrum_data) > 0:
            self.detect_peaks()
        else:
            self.peaks = []
            self.peak_positions = []
            self.update_plot()

    def toggle_auto_peak_params(self, checked: bool):
        self.auto_peak_params = checked
        self.manual_params_frame.setVisible(not self.auto_peak_params)
        self.log_message(f"Автопараметры пиков {'включены' if self.auto_peak_params else 'выключены'}")

    def toggle_auto_temperature_update(self, checked: bool):
        self.auto_calc_temp = checked
        if self.auto_calc_temp and len(self.spectrum_data) > 0:
            self.calculate_temp_from_spectrum()
        self.log_message(f"Автообновление температуры {'включено' if self.auto_calc_temp else 'выключено'}")

    def toggle_planck_display(self, checked: bool):
        if checked != self.show_planck_on_spectrum:
            self.show_planck_on_spectrum = checked
            self.update_plot()
            self.log_message(f"Отображение закона Планка {'включено' if checked else 'выключено'}")

    def toggle_mirror(self, state: bool):
        self.mirror_camera = state
        self.log_message(f"Отзеркаливание камеры {'включено' if state else 'выключено'}")

    def toggle_fill_under_curve(self, checked: bool):
        self.fill_under_curve = checked
        self.update_plot()
        self.log_message(f"Заливка под кривой {'включена' if self.fill_under_curve else 'выключена'}")

    def on_continuous_check_toggled(self, checked: bool):
        self.log_message(f"Непрерывное измерение {'включено' if checked else 'выключено'}")

    def on_realtime_check_toggled(self, checked: bool):
        self.log_message(f"Отображение в реальном времени {'включено' if checked else 'выключено'}")

    def refresh_cameras(self):
        self.camera_combo.clear()
        available_cams = []
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                available_cams.append(f"Камера {i}")
                cap.release()
        if available_cams:
            self.camera_combo.addItems(available_cams)
            self.update_status(f"Найдено камер: {len(available_cams)}")
            self.log_message(f"Обнаружено камер: {len(available_cams)}")
        else:
            self.camera_combo.addItem("Камеры не найдены")
            self.update_status("Камеры не обнаружены")
            self.log_message("Камеры не обнаружены")

    def toggle_camera_connection(self):
        if self.connected:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def connect_camera(self):
        if self.camera_combo.currentText() == "Камеры не найдены":
            QMessageBox.warning(self, "Ошибка", "Нет доступных камер")
            return
        try:
            cam_idx = int(self.camera_combo.currentText().split()[-1])
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError("Не удалось открыть камеру")
            for _ in range(3):
                ret, _ = self.cap.read()
                if ret:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Камера не возвращает кадры")
            self.connected = True
            self.worker.start()
            self.connect_action.setChecked(True)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.update_status(f"Подключено: Камера {cam_idx} ({width}x{height}, {fps:.1f} FPS)")
            self.log_message(f"Подключена камера {cam_idx} ({width}x{height}, {fps:.1f} FPS)")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось подключиться: {e}")
            if self.cap:
                self.cap.release()
            self.connected = False
            self.connect_action.setChecked(False)
            self.update_status(f"Ошибка подключения: {e}")
            self.log_message(f"Ошибка подключения камеры: {e}")

    def disconnect_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.worker.stop()
        self.connected = False
        self.connect_action.setChecked(False)
        self.update_status("Отключено")
        self.log_message("Камера отключена")

    def change_camera_resolution(self):
        if not self.connected or not self.cap:
            return
        index = self.camera_size_combo.currentIndex()
        resolutions = [
            (320, 240),
            (640, 480),
            (800, 600),
            (1024, 768),
            None
        ]
        if index < len(resolutions):
            res = resolutions[index]
            if res is None:
                standard_resolutions = [
                    (1920, 1080), (1600, 1200), (1280, 1024),
                    (1280, 720), (1024, 768), (800, 600),
                    (640, 480), (320, 240)
                ]
                max_width, max_height = 0, 0
                for w, h in standard_resolutions:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if actual_w >= max_width and actual_h >= max_height:
                        max_width, max_height = actual_w, actual_h
                self.log_message(f"Установлено максимальное разрешение: {max_width}x{max_height}")
            else:
                width, height = res
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.log_message(f"Установлено разрешение камеры: {width}x{height}")

            QTimer.singleShot(500, self.update_camera_info)

    def update_camera_info(self):
        if self.connected and self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.update_status(f"Разрешение: {width}x{height}, FPS: {fps:.1f}")
    def capture_frame(self):
        if not self.connected or not self.cap or not self.cap.isOpened():
            return
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
            if self.mirror_camera:
                frame = cv2.flip(frame, 1)
            self.frame = frame
            display_frame = frame.copy()
            x, y, w, h = self.roi
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = rgb_image.shape
            label_size = self.camera_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                aspect_ratio = w_img / h_img
                target_w = label_size.width()
                target_h = label_size.height()
                if target_w / target_h > aspect_ratio:
                    target_w = int(target_h * aspect_ratio)
                else:
                    target_h = int(target_w / aspect_ratio)
                scaled_image = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
                bytes_per_line = scaled_image.strides[0]
                q_img = QImage(scaled_image.copy().data, target_w, target_h, bytes_per_line, QImage.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(q_img))

            if not self.paused and self.realtime_check.isChecked():
                self.worker.process_frame(frame)

            if self.continuous_measurement and not self.paused:
                spectrum = self.worker.process_frame(frame, return_data=True)
                if spectrum is not None:
                    self.process_spectrum_data(spectrum)
                    if self.auto_calc_temp:
                        self.calculate_temp_from_spectrum()

        except Exception as e:
            print(f"Ошибка захвата кадра: {e}")
            self.log_message(f"Ошибка захвата кадра: {e}")

    def process_spectrum_data(self, spectrum, save=False):
        if spectrum is None or len(spectrum) == 0:
            print("Ошибка: Нет данных спектра для обработки")
            return

        if self.dark_corr_check.isChecked() and self.dark_data is not None:
            spectrum = np.maximum(0, spectrum - self.dark_data)

        if self.ref_corr_check.isChecked() and self.reference_data is not None:
            spectrum = np.divide(spectrum,
                                 np.where(self.reference_data > 1, self.reference_data, 1))

        self.raw_spectrum = spectrum.copy()

        if self.smooth_check.isChecked():
            spectrum = np.convolve(spectrum, np.ones(5) / 5, mode='same')

        if self.peak_detection_enabled:
            self.detect_peaks()

        if self.auto_calc_temp:
            self.calculate_temp_from_spectrum()

        if self.normalize_check.isChecked():
            max_val = np.max(spectrum)
            if max_val > 0:
                spectrum = spectrum / max_val

        self.spectrum_data = spectrum
        self.fill_under_curve = self.fill_action.isChecked()

        if save:
            self.add_spectrum_to_saved(spectrum)

        self.update_plot()
        self.update_boltzmann_law()

    def update_plot(self):
        """Обновление графика спектра с сохранением масштабирования при включенном зуме"""
        if not hasattr(self, 'ax') or not hasattr(self, 'canvas'):
            return
        
        try:
            # Сохраняем текущие границы перед обновлением
            prev_xlim = self.ax.get_xlim()
            prev_ylim = self.ax.get_ylim()
            
            # Очищаем график
            self.ax.clear()
            
            # Получаем текущие данные с учетом сдвига
            x = self.wavelengths
            y = self.spectrum_data
            
            # Рисуем основной спектр с цветовой градиентной линией
            if len(x) > 0 and len(y) > 0:
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                colors = [self.wavelength_to_color((x[i] + x[i+1])/2) for i in range(len(x)-1)]
                
                lc = LineCollection(segments, colors=colors, linewidths=1.5, zorder=3)
                self.ax.add_collection(lc)
                
                # Заливка под кривой
                if self.fill_under_curve:
                    verts = [[(x[i], 0), (x[i], y[i]), (x[i+1], y[i+1]), (x[i+1], 0)] 
                            for i in range(len(x)-1)]
                    poly = PolyCollection(verts, facecolors=[(*c, 0.3) for c in colors], zorder=2)
                    self.ax.add_collection(poly)
            
            # Рисуем сохраненные спектры
            max_y = np.max(y) if len(y) > 0 else 1.0
            for i in range(self.spectra_list_layout.count()):
                widget = self.spectra_list_layout.itemAt(i).widget()
                if isinstance(widget, QCheckBox) and widget.isChecked():
                    label, spec, color = self.saved_spectra[i]
                    self.ax.plot(x, spec, color=color, alpha=0.6, label=label)
                    max_y = max(max_y, np.max(spec))
                    
                    if self.show_peak_lines:
                        max_idx = np.argmax(spec)
                        max_wl = x[max_idx]
                        max_intensity = spec[max_idx]
                        self.ax.axvline(max_wl, color=color, linestyle='--', alpha=0.5)
                        self.ax.text(max_wl, max_intensity, f"{max_wl:.1f} nm",
                                    color=color, ha='center', va='bottom')
            
            # Отображаем пики, если включено
            if self.peak_detection_enabled and hasattr(self, 'peaks') and len(self.peaks) > 0:
                for i, peak_idx in enumerate(self.peaks):
                    wl = self.peak_positions[i]
                    peak_val = y[peak_idx] if len(y) > peak_idx else 0
                    peak_color = self.wavelength_to_color(wl)
                    
                    self.ax.plot(wl, peak_val, 'o',
                                markerfacecolor='white',
                                markeredgecolor=peak_color,
                                markersize=8,
                                zorder=4)
                    if self.show_peak_lines:
                        self.ax.axvline(wl, color=peak_color, linestyle='--', alpha=0.5, zorder=1)
            
            # Кривая Планка
            if self.show_planck_on_spectrum and self.planck_data is not None:
                planck_wl, planck_intensity = self.planck_data
                if np.max(planck_intensity) > 0:
                    planck_intensity = planck_intensity / np.max(planck_intensity) * max_y
                    self.ax.plot(planck_wl, planck_intensity, 'r--',
                                label=f'Закон Планка (T={self.temperature:.0f} K)', zorder=2)
            
            # Аппроксимация Планком для сохраненных спектров
            for i, (label, spec, color) in enumerate(self.saved_spectra):
                if f"approximation_{label}" in self.approximation_results:
                    approx_data = self.approximation_results[f"approximation_{label}"]
                    if approx_data['success']:
                        T_fit = approx_data['temperature']
                        planck_fit = planck_law(x, T_fit, approx_data['scale'], approx_data['offset'])
                        self.ax.plot(x, planck_fit, color=color, linestyle=':', 
                                   label=f'Аппроксимация {label} (T={T_fit:.0f} K)', zorder=2)
            
            # Настройка осей
            self.ax.set_xlabel("Длина волны (нм)")
            self.ax.set_ylabel("Интенсивность")
            
            # Применяем сохраненные границы или устанавливаем новые
            if self.zoom_enabled and hasattr(self, 'current_xlim'):
                # Сохраняем текущий зум
                self.ax.set_xlim(*self.current_xlim)
                self.ax.set_ylim(*self.current_ylim)
            else:
                # Автомасштабирование при выключенном зуме
                self.ax.set_xlim(340 + self.wavelength_shift, 1050 + self.wavelength_shift)
                y_max = max(max_y * 1.2, 0.1)  # Минимум 0.1 чтобы избежать ошибки
                self.ax.set_ylim(0, y_max)
                self.current_xlim = (340 + self.wavelength_shift, 1050 + self.wavelength_shift)
                self.current_ylim = (0, y_max)
            
            # Легенда
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:
                legend = self.ax.legend()
                self.apply_style_to_legend(legend, self.current_style if hasattr(self, 'current_style') else "Темный")
            
            # Стиль графика
            self.apply_style_to_axes(self.ax, self.figure, self.current_style if hasattr(self, 'current_style') else "Темный")
            
            # Обновляем canvas
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Ошибка в update_plot: {e}")
            self.log_message(f"Ошибка обновления графика: {e}")

    def apply_style_to_axes(self, ax, figure, style=None):
        """Применяет текущий стиль к осям графика"""
        if style is None:
            # Используем текущие установки стиля
            bgcolor = self.graph_bgcolor
            textcolor = self.graph_textcolor
            gridcolor = self.graph_gridcolor
        else:
            # Устанавливаем стиль на основе переданного параметра
            if style == "Темный":
                bgcolor = '#2b2b2b'
                textcolor = 'white'
                gridcolor = '#666'
            elif style == "Светлый":
                bgcolor = 'white'
                textcolor = 'black'
                gridcolor = '#ccc'
            elif style == "Синий":
                bgcolor = '#1a2b5b'
                textcolor = 'white'
                gridcolor = '#5a8aff'
        
        ax.grid(True, color=gridcolor, linestyle=':', alpha=0.5)
        ax.set_facecolor(bgcolor)
        figure.set_facecolor(bgcolor)
        
        ax.tick_params(colors=textcolor)
        ax.xaxis.label.set_color(textcolor)
        ax.yaxis.label.set_color(textcolor)
        for spine in ax.spines.values():
            spine.set_color(textcolor)
        
        # Обновляем легенду
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_color(textcolor)
        
        # Перерисовываем соответствующий canvas
        if ax == self.ax and hasattr(self, 'canvas'):
            self.canvas.draw_idle()
        elif ax == self.wien_ax and hasattr(self, 'wien_canvas'):
            self.wien_canvas.draw_idle()
        elif ax == self.boltzmann_ax and hasattr(self, 'boltzmann_canvas'):
            self.boltzmann_canvas.draw_idle()
        elif ax == self.time_ax and hasattr(self, 'time_canvas'):
            self.time_canvas.draw_idle()

    @staticmethod
    def apply_style_to_legend(legend, style):
        color_map = {
            "Темный": '#e0e0e0',
            "Светлый": '#333333',
            "Синий": '#ffffff'
        }
        color = color_map.get(style, 'black')
        for text in legend.get_texts():
            text.set_color(color)

    def update_planck_curve(self):
        if not hasattr(self, 'temperature') or self.temperature < 100:
            return
        try:
            if self.temperature not in self._planck_cache:
                wavelengths = np.linspace(340 + self.wavelength_shift, 1050 + self.wavelength_shift, 1000)
                wavelengths_m = (wavelengths - self.wavelength_shift) * 1e-9  # Учитываем сдвиг только для отображения
                exp_arg = (self.h * self.c) / (wavelengths_m * self.k * self.temperature)
                intensity = (2 * self.h * self.c ** 2 / (wavelengths_m ** 5)) * \
                            (1 / (np.exp(exp_arg) - 1))
                intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
                self._planck_cache[self.temperature] = (wavelengths, intensity)
            self.planck_data = self._planck_cache[self.temperature]
            if self.show_planck_on_spectrum:
                self.update_plot()
        except Exception as e:
            print(f"Ошибка обновления кривой Планка: {e}")
            self.log_message(f"Ошибка обновления кривой Планка: {e}")

    def detect_peaks(self):
        if not self.peak_detection_enabled:
            return
        try:
            if not hasattr(self, 'spectrum_data') or len(self.spectrum_data) == 0:
                return

            if self.auto_peak_params:
                max_val = np.max(self.spectrum_data)
                min_val = np.min(self.spectrum_data)
                height = max(min_val + (max_val - min_val) * 0.1, 10)
                prominence = max((max_val - min_val) * 0.05, 5)
                distance = max(len(self.wavelengths) // 20, 5)
            else:
                height = max(self.peak_height_spin.value(), 1)
                prominence = max(self.peak_prominence_spin.value(), 1)
                distance = max(self.peak_distance_spin.value(), 1)

            peaks, properties = find_peaks(
                self.spectrum_data,
                height=height,
                prominence=prominence,
                distance=distance,
                width=2
            )

            if len(peaks) > 0:
                self.peaks = peaks
                self.peak_positions = self.wavelengths[peaks]
                self.peak_properties = properties
            else:
                self.peaks = []
                self.peak_positions = []
                self.peak_properties = {}

            self.update_peaks_table()
            self.update_plot()
        except Exception as e:
            print(f"Ошибка в detect_peaks(): {e}")

    def update_peaks_table(self):
        self.peaks_table.setRowCount(len(self.peaks))
        for i, (idx, wl) in enumerate(zip(self.peaks, self.peak_positions)):
            self.peaks_table.setItem(i, 0, QTableWidgetItem(f"{i + 1}"))
            self.peaks_table.setItem(i, 1, QTableWidgetItem(f"{wl:.2f}"))
            intensity = self.spectrum_data[idx]
            self.peaks_table.setItem(i, 2, QTableWidgetItem(f"{intensity:.0f}"))

            if 'widths' in self.peak_properties:
                fwhm = self.peak_properties['widths'][i] * (self.wavelengths[1] - self.wavelengths[0])
                self.peaks_table.setItem(i, 3, QTableWidgetItem(f"{fwhm:.2f}"))
            else:
                self.peaks_table.setItem(i, 3, QTableWidgetItem("N/A"))

            temp = self.wien_const / (wl * 1e-9) if wl > 0 else 0
            self.peaks_table.setItem(i, 4, QTableWidgetItem(f"{temp:.0f}"))

    def update_roi(self):
        self.roi = (
            self.roi_x_spin.value(),
            self.roi_y_spin.value(),
            self.roi_w_spin.value(),
            self.roi_h_spin.value()
        )
        self.worker.set_roi(self.roi)
        self.log_message(f"ROI обновлен: X={self.roi[0]}, Y={self.roi[1]}, W={self.roi[2]}, H={self.roi[3]}")

    def capture_dark(self):
        if not self.connected or self.frame is None:
            return
        spectrum = self.worker.process_frame(self.frame, return_data=True)
        if spectrum is not None:
            self.dark_data = spectrum
            self.update_status("Темновой спектр захвачен")
            self.dark_corr_check.setEnabled(True)
            self.log_message("Темновой спектр захвачен")

    def capture_reference(self):
        if not self.connected or self.frame is None:
            return
        spectrum = self.worker.process_frame(self.frame, return_data=True)
        if spectrum is not None:
            self.reference_data = spectrum
            self.update_status("Референсный спектр захвачен")
            self.ref_corr_check.setEnabled(True)
            self.log_message("Референсный спектр захвачен")

    def start_measurement(self):
        if not self.connected:
            QMessageBox.warning(self, "Ошибка", "Камера не подключена")
            return
        if self.continuous_check.isChecked():
            self.continuous_measurement = not self.continuous_measurement
            self.measure_action.setIcon(QIcon.fromTheme("media-playback-stop" if self.continuous_measurement else "media-record"))
            self.measure_action.setText("Стоп" if self.continuous_measurement else "Измерить")
            self.log_message(f"Непрерывное измерение {'включено' if self.continuous_measurement else 'выключено'}")
        else:
            if self.frame is None:
                QMessageBox.warning(self, "Ошибка", "Нет данных с камеры")
                return
            spectrum = self.worker.process_frame(self.frame, return_data=True)
            if spectrum is not None:
                self.process_spectrum_data(spectrum)  # без save=True
                self.calculate_temp_from_spectrum()
                self.log_message("Измерение спектра выполнено")

    def update_integration_time(self):
        self.integration_time = self.integration_spin.value()
        self.log_message(f"Время интегрирования установлено: {self.integration_time} мс")

    def save_current_spectrum(self):
        if len(self.spectrum_data) == 0:
            return
        self.add_spectrum_to_saved(self.spectrum_data, temporary=False)

    def toggle_pause(self):
        self.paused = not self.paused
        self.worker.pause(self.paused)
        self.pause_action.setChecked(self.paused)
        self.update_status("Приостановлено" if self.paused else "Продолжено")
        self.log_message("Измерение приостановлено" if self.paused else "Измерение продолжено")

    def on_temperature_changed(self):
        try:
            new_temp = self.temp_spin.value()
            if abs(new_temp - self.temperature) < 1e-3:
                return
            self.temperature = new_temp
            if self.temperature >= 100:
                self.update_planck_curve()
                self.update_wien_law()
                self.update_boltzmann_law()
                self.update_plot()
                self.log_message(f"Температура установлена: {new_temp:.0f} K")
        except Exception as e:
            print(f"Ошибка изменения температуры: {e}")
            self.log_message(f"Ошибка изменения температуры: {e}")

    def update_wien_law(self):
        if not hasattr(self, 'wien_ax'):
            return
        try:
            self.wien_ax.clear()
            if not self.show_wien_law:
                self.wien_ax.text(0.5, 0.5, "Отображение закона Вина отключено", ha='center', va='center')
                self.wien_canvas.draw_idle()
                return
                
            if self.temperature < 100:
                self.wien_ax.text(0.5, 0.5, "Установите T > 100 K", ha='center', va='center')
                self.wien_canvas.draw()
                return

            wavelengths, intensity = self.planck_data if self.planck_data else self.calculate_planck_law(self.temperature)
            self.wien_ax.plot(wavelengths, intensity, 'b-', label=f'Теория (T={int(self.temperature)} K)')

            if len(self.spectrum_data) > 0 and np.max(self.spectrum_data) > 0:
                exp_data = self.spectrum_data / np.max(self.spectrum_data)
                self.wien_ax.plot(self.wavelengths, exp_data, 'g-', alpha=0.7, label='Эксперимент')

                interp_theory = np.interp(self.wavelengths, wavelengths, intensity)
                if np.any(interp_theory > 0):
                    corr = np.corrcoef(exp_data, interp_theory)[0, 1]
                    self.wien_ax.text(0.02, 0.95, f"Корреляция: {corr:.2f}",
                                     transform=self.wien_ax.transAxes,
                                     bbox=dict(facecolor='white', alpha=0.7))

            wien_displacement = self.wien_const / self.temperature * 1e9
            self.wien_ax.axvline(wien_displacement + self.wavelength_shift, color='r', linestyle='--', 
                                label=f'λ_max={wien_displacement:.1f} нм (с учетом сдвига)')

            self.wien_ax.set_title("Закон Вина")
            self.wien_ax.set_xlabel("Длина волны (нм)")
            self.wien_ax.set_ylabel("Относительная интенсивность")
            self.wien_ax.legend()

            self.apply_style_to_axes(self.wien_ax, self.wien_figure)

            self.wien_canvas.draw_idle()
        except Exception as e:
            print(f"Ошибка отрисовки закона Вина: {e}")
            self.log_message(f"Ошибка отрисовки закона Вина: {e}")

    def calculate_planck_law(self, temperature):
        wavelengths = np.linspace(340 + self.wavelength_shift, 1050 + self.wavelength_shift, 1000)
        wavelengths_m = (wavelengths - self.wavelength_shift) * 1e-9  # Учитываем сдвиг только для отображения
        exp_arg = (self.h * self.c) / (wavelengths_m * self.k * temperature)
        intensity = (2 * self.h * self.c ** 2 / (wavelengths_m ** 5)) * \
                    (1 / (np.exp(exp_arg) - 1))
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
        return wavelengths, intensity

    def update_boltzmann_law(self):
        if not hasattr(self, 'boltzmann_ax'):
            return
        try:
            self.boltzmann_ax.clear()
            if not self.show_boltzmann_law:
                self.boltzmann_ax.text(0.5, 0.5, "Отображение закона Стефана-Больцмана отключено", ha='center', va='center')
                self.boltzmann_canvas.draw_idle()
                return
                
            if not hasattr(self, 'temperature') or self.temperature < 100:
                self.boltzmann_ax.text(0.5, 0.5, "Установите T > 100 K", ha='center', va='center')
                self.boltzmann_canvas.draw_idle()
                return

            power_theoretical = self.sigma * (self.temperature ** 4)

            temps = np.linspace(100, max(6000, self.temperature * 1.2), 100)
            powers = self.sigma * (temps ** 4)

            self.boltzmann_ax.plot(temps, powers, 'r-', label='Теоретическая кривая')
            self.boltzmann_ax.plot(int(self.temperature), power_theoretical, 'ro', label=f'T={int(self.temperature)} K')

            exp_power = None
            exp_temp = None
            if hasattr(self, 'spectrum_data') and len(self.spectrum_data) > 0:
                try:
                    exp_power = simpson(self.spectrum_data, self.wavelengths)
                    if exp_power > 0:
                        exp_temp = (exp_power / self.sigma) ** 0.25
                    else:
                        exp_temp = 0
                    self.boltzmann_ax.plot(int(exp_temp), exp_power, 'go', label=f'Эксперимент (T≈{int(exp_temp)} K)')
                except Exception:
                    exp_power = None

            correlation = 0
            if exp_power and exp_power > 0 and power_theoretical > 0:
                correlation = min(exp_power / power_theoretical, power_theoretical / exp_power)

            title = (f"Закон Стефана-Больцмана\n"
                     f"Температура: {int(self.temperature)} K\n"
                     f"Мощность излучения (теория): {power_theoretical:.2e} Вт/м²\n")
            if exp_power:
                title += f"Мощность излучения (эксперимент): {exp_power:.2e} у.е.\nКорреляция: {correlation:.2%}"
            else:
                title += "Мощность излучения (эксперимент): -\nКорреляция: -"

            self.boltzmann_ax.set_title(title)
            self.boltzmann_ax.set_xlabel("Температура (K)")
            self.boltzmann_ax.set_ylabel("Мощность излучения (Вт/м²)")
            self.boltzmann_ax.legend()

            self.boltzmann_ax.set_yscale('log' if power_theoretical > 1e6 else 'linear')

            self.apply_style_to_axes(self.boltzmann_ax, self.boltzmann_figure)

            self.boltzmann_canvas.draw_idle()
        except Exception as e:
            print(f"Ошибка отрисовки закона Стефана-Больцмана: {e}")
            self.log_message(f"Ошибка отрисовки закона Стефана-Больцмана: {e}")

    def calculate_temp_from_spectrum(self):
        if self.spectrum_data is None or len(self.spectrum_data) == 0:
            print("Ошибка: Нет данных спектра")
            return
        try:
            y_data = self.raw_spectrum if self.raw_spectrum is not None else self.spectrum_data
            if np.max(y_data) < 10:
                print("Ошибка: Слишком слабый сигнал")
                return

            smoothed = np.convolve(y_data, np.ones(5) / 5, mode='same')

            peaks, props = find_peaks(
                smoothed,
                height=np.max(smoothed) * 0.3,
                prominence=np.max(smoothed) * 0.2,
                width=5,
                distance=50
            )

            if len(peaks) == 0:
                print("Ошибка: Не найдены пики в спектре")
                return

            main_idx = peaks[np.argmax(props['peak_heights'])]

            if 0 < main_idx < len(smoothed) - 1:
                x = np.array([main_idx - 1, main_idx, main_idx + 1])
                y = smoothed[x]
                coeffs = np.polyfit(x, y, 2)
                precise_pos = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else main_idx
                lambda_max = np.interp(precise_pos, np.arange(len(self.wavelengths)), self.wavelengths)
            else:
                lambda_max = self.wavelengths[main_idx]

            if not (340 <= lambda_max - self.wavelength_shift <= 1050):  # Учитываем сдвиг при проверке диапазона
                print(f"Ошибка: Пик вне диапазона ({lambda_max:.1f} нм)")
                return

            new_temp = self.wien_const / ((lambda_max - self.wavelength_shift) * 1e-9)  # Учитываем сдвиг при расчете температуры
            if not (500 <= new_temp <= 20000):
                print(f"Ошибка: Нереальная температура ({new_temp:.0f} K)")
                return

            self.temperature = new_temp
            self.temp_spin.blockSignals(True)
            self.temp_spin.setValue(new_temp)
            self.temp_spin.blockSignals(False)

            self.update_planck_curve()
            self.update_wien_law()
            self.update_boltzmann_law()
            self.update_plot()

            self.update_status(f"Авторасчет T: {new_temp:.0f} K (λ={lambda_max:.1f} нм)")
            self.log_message(f"Рассчитана температура: {new_temp:.0f} K (по пику {lambda_max:.1f} нм)")

        except Exception as e:
            print(f"Ошибка расчёта температуры: {e}")
            self.update_status(f"Ошибка расчета температуры: {e}")
            self.log_message(f"Ошибка расчета температуры: {e}")

    def save_spectrum(self):
        if len(self.spectrum_data) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных спектра для сохранения")
            return
        filename = f"spectrum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить спектр",
            os.path.join(self.last_save_path, filename),
            "CSV (*.csv);;Все файлы (*)"
        )
        if path:
            try:
                np.savetxt(path, np.column_stack((self.wavelengths, self.spectrum_data)),
                           delimiter=",", header="wavelength(nm),intensity", comments="")
                self.update_status(f"Спектр сохранен: {os.path.basename(path)}")
                self.last_save_path = os.path.dirname(path)
                self.log_message(f"Спектр сохранен в {path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")
                self.log_message(f"Ошибка сохранения спектра: {e}")

    def save_plot(self):
        if len(self.spectrum_data) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения графика")
            return
        filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график",
            os.path.join(self.last_save_path, filename),
            "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)"
        )
        if path:
            try:
                self.figure.savefig(path, dpi=300, bbox_inches='tight')
                self.update_status(f"График сохранен: {os.path.basename(path)}")
                self.last_save_path = os.path.dirname(path)
                self.log_message(f"График сохранен в {path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")
                self.log_message(f"Ошибка сохранения графика: {e}")

    def save_all_data(self):
        if len(self.spectrum_data) == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
        dir_path = QFileDialog.getExistingDirectory(
            self, "Выберите папку для сохранения",
            self.last_save_path
        )
        if dir_path:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                np.savetxt(
                    os.path.join(dir_path, f"spectrum_{timestamp}.csv"),
                    np.column_stack((self.wavelengths, self.spectrum_data)),
                    delimiter=",", header="wavelength(nm),intensity", comments=""
                )
                self.progress_bar.setValue(20)

                if self.dark_data is not None:
                    np.savetxt(
                        os.path.join(dir_path, f"dark_{timestamp}.csv"),
                        np.column_stack((self.wavelengths, self.dark_data)),
                        delimiter=",", header="wavelength(nm),intensity", comments=""
                    )
                self.progress_bar.setValue(40)

                if self.reference_data is not None:
                    np.savetxt(
                        os.path.join(dir_path, f"reference_{timestamp}.csv"),
                        np.column_stack((self.wavelengths, self.reference_data)),
                        delimiter=",", header="wavelength(nm),intensity", comments=""
                    )
                self.progress_bar.setValue(60)

                if hasattr(self, 'peaks') and len(self.peaks) > 0:
                    with open(os.path.join(dir_path, f"peaks_{timestamp}.csv"), 'w') as f:
                        f.write("wavelength(nm),intensity,fwhm(nm),temperature(K)\n")
                        for i in range(self.peaks_table.rowCount()):
                            wl = self.peaks_table.item(i, 1).text()
                            intensity = self.peaks_table.item(i, 2).text()
                            fwhm = self.peaks_table.item(i, 3).text()
                            temp = self.peaks_table.item(i, 4).text()
                            f.write(f"{wl},{intensity},{fwhm},{temp}\n")
                self.progress_bar.setValue(80)

                self.figure.savefig(
                    os.path.join(dir_path, f"plot_{timestamp}.png"),
                    dpi=300, bbox_inches='tight'
                )
                self.progress_bar.setValue(100)

                self.update_status(f"Все данные сохранены в: {dir_path}")
                self.last_save_path = dir_path
                self.log_message(f"Все данные сохранены в {dir_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить данные: {e}")
                self.log_message(f"Ошибка сохранения всех данных: {e}")
            finally:
                self.progress_bar.setVisible(False)

    def update_status(self, message):
        self.status_bar.showMessage(message)

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        current_text = self.log_text.text()
        new_text = f"[{timestamp}] {message}<br>{current_text}"
        self.log_text.setText(new_text)

    def closeEvent(self, event):
        self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()

        self.planck_thread.quit()
        self.planck_thread.wait()

        self.thread_pool.shutdown(wait=False)
        if self.cap:
            self.cap.release()

        event.accept()

    def create_camera_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        splitter = QSplitter(Qt.Vertical)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(1, 1)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.set_xlabel("Длина волны (нм)")
        self.ax.set_ylabel("Интенсивность")

        zoom_layout = QHBoxLayout()
        self.zoom_btn = QPushButton("Вкл/Выкл Зум")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.toggled.connect(self.toggle_zoom)
        self.reset_zoom_btn = QPushButton("Сбросить масштаб")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(self.zoom_btn)
        zoom_layout.addWidget(self.reset_zoom_btn)
        layout.addLayout(zoom_layout) 

        splitter.addWidget(self.camera_label)
        layout.addLayout(zoom_layout)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
        self.camera_splitter = splitter

        self.zoom_enabled = False
        self.rectangle_selector = None

        return tab
    def toggle_zoom(self, checked):
        self.zoom_enabled = checked
        if checked:
            if self.rectangle_selector is None:
                self.rectangle_selector = RectangleSelector(
                    self.ax, 
                    self.on_select_zoom,
                    useblit=True,
                    button=[1],
                    minspanx=5, 
                    minspany=5,
                    spancoords='pixels',
                    interactive=True,
                    props=dict(facecolor='white', edgecolor='black', alpha=0.1, fill=True)
                )
            else:
                self.rectangle_selector.set_visible(True)  # Делаем видимым
            self.rectangle_selector.set_active(True)
        else:
            if self.rectangle_selector is not None:
                self.rectangle_selector.set_active(False)
                self.rectangle_selector.set_visible(False)  # Скрываем прямоугольник
                self.canvas.draw_idle()  # Немедленно обновляем график
            self.reset_zoom()

    def on_select_zoom(self, eclick, erelease):
        try:
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            
            if None in (x1, y1, x2, y2):
                return
                
            # Минимальный размер области зума
            if abs(x1 - x2) < 5:
                x2 = x1 + 5 if x1 + 5 <= 1050 + self.wavelength_shift else x1 - 5
                
            if abs(y1 - y2) < 0.05:
                y2 = y1 + 0.05 if y1 + 0.05 <= 1 else y1 - 0.05
                
            self.current_xlim = (min(x1, x2), max(x1, x2))
            self.current_ylim = (min(y1, y2), max(y1, y2))
            
            self.ax.set_xlim(*self.current_xlim)
            self.ax.set_ylim(*self.current_ylim)
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Zoom error: {e}")

    def reset_zoom(self):
        if hasattr(self, 'rectangle_selector') and self.rectangle_selector is not None:
            self.rectangle_selector.extents = (0, 0, 0, 0)  # Сбрасываем размеры прямоугольника
        
        self.current_xlim = (340 + self.wavelength_shift, 1050 + self.wavelength_shift)
        y_max = np.max(self.spectrum_data) * 1.2 if hasattr(self, 'spectrum_data') and np.max(self.spectrum_data) > 0 else 1.0
        self.current_ylim = (0, max(y_max, 0.1))  # Минимум 0.1 разница
        
        if hasattr(self, 'ax'):
            self.ax.set_xlim(*self.current_xlim)
            self.ax.set_ylim(*self.current_ylim)
            self.canvas.draw_idle()

    def create_peaks_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        self.peaks_table = QTableWidget(0, 5)
        self.peaks_table.setHorizontalHeaderLabels(["№", "λ (нм)", "Инт.", "FWHM", "Температура (K)"])
        self.peaks_table.horizontalHeader().setStretchLastSection(True)
        self.peaks_table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.peaks_table)
        return tab
    def create_wien_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        self.wien_figure = Figure(figsize=(8, 5), dpi=100)
        self.wien_canvas = FigureCanvas(self.wien_figure)
        self.wien_ax = self.wien_figure.add_subplot(111)

        self.wien_ax.grid(True, linestyle=':', alpha=0.5)
        self.wien_ax.set_xlabel("Длина волны (нм)")
        self.wien_ax.set_ylabel("Относительная интенсивность")

        layout.addWidget(self.wien_canvas)

        return tab

    def create_boltzmann_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        self.boltzmann_figure = Figure(figsize=(8, 5), dpi=100)
        self.boltzmann_canvas = FigureCanvas(self.boltzmann_figure)
        self.boltzmann_ax = self.boltzmann_figure.add_subplot(111)

        self.boltzmann_ax.grid(True, linestyle=':', alpha=0.5)
        self.boltzmann_ax.set_xlabel("Температура (K)")
        self.boltzmann_ax.set_ylabel("Мощность излучения (Вт/м²)")

        layout.addWidget(self.boltzmann_canvas)

        return tab

    def create_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        self.log_text = QLabel()
        self.log_text.setWordWrap(True)
        self.log_text.setAlignment(Qt.AlignTop)
        self.log_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.log_text)
        
        clear_btn = QPushButton("Очистить лог")
        clear_btn.clicked.connect(lambda: self.log_text.setText(""))
        
        layout.addWidget(scroll)
        layout.addWidget(clear_btn)
        
        return tab

    def approximate_saved_spectrum(self, index):
        """Аппроксимация конкретного сохраненного спектра законом Планка"""
        if index < 0 or index >= len(self.saved_spectra):
            return
            
        label, spectrum, color = self.saved_spectra[index]
        
        try:
            # Подготовка данных для аппроксимации
            x_data = self.wavelengths
            y_data = spectrum
            
            # Нормализация данных
            y_norm = y_data / np.max(y_data) if np.max(y_data) > 0 else y_data
            
            # Начальные параметры для аппроксимации
            T_guess = self.temperature if self.temperature > 0 else 5000
            scale_guess = np.max(y_data)
            offset_guess = np.min(y_data)
            
            # Границы параметров
            bounds = (
                [100, 0.1 * scale_guess, -0.1 * scale_guess],  # нижние границы
                [20000, 10 * scale_guess, 0.1 * scale_guess]   # верхние границы
            )
            
            # Аппроксимация
            popt, pcov = curve_fit(
                planck_law, 
                x_data, 
                y_data,
                p0=[T_guess, scale_guess, offset_guess],
                bounds=bounds,
                maxfev=10000
            )
            
            # Извлечение параметров
            T_fit, scale_fit, offset_fit = popt
            perr = np.sqrt(np.diag(pcov)) if np.any(pcov) else [0, 0, 0]
            
            # Сохранение результатов
            self.approximation_results[f"approximation_{label}"] = {
                'temperature': T_fit,
                'scale': scale_fit,
                'offset': offset_fit,
                'errors': perr,
                'success': True
            }
            
            self.log_message(f"Аппроксимация {label}: T={T_fit:.1f}±{perr[0]:.1f} K")
            self.update_plot()
            
        except Exception as e:
            self.log_message(f"Ошибка аппроксимации {label}: {e}")

    def approximate_all_spectra(self):
        """Аппроксимация всех сохраненных спектров"""
        for i in range(len(self.saved_spectra)):
            self.approximate_saved_spectrum(i)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = SpectrometerApp()
    window.show()

    sys.exit(app.exec())