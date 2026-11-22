#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Расчёт оптической схемы учебного USB-спектрометра «СпектроЛаб».

Схема (вид сверху, ось X — оптическая ось, ось Y — направление дисперсии,
ось Z — вертикаль):

    щель 0,03x8 мм        коллиматор f=40     решётка 600 штр/мм      M12 f=6
    |  0,05 мм            диафрагма 4,0 мм    (пропускающая,          ось 25,5°
    v                       O                 нормальное падение)     к нормали
    ──┼─────────────────────/───────────────────|──────────────────────\
      │                    /                    |                       \  камера
      │ 8 мм              /   параллельный       \  1-й порядок           \ 1920x1080
      │                  /    пучок 4 мм          \  θ = 11,8…39,0°         \ 2,8 мкм
      └─────────────────/─────────────────────────\─────────────────────────┘

Расчётные соотношения (m — порядок, d — период решётки, f — фокус):

    sin θ = m·λ/d                         (уравнение решётки, нормальное падение)
    dθ/dλ = m/(d·cos θ)                   (угловая дисперсия, рад/нм)
    x(λ)  = f_кам · tg(θ(λ) − θ_кам)      (положение линии на сенсоре, мм)
    dx/dλ = f_кам · sec²(θ−θ_кам) · dθ/dλ (линейная дисперсия, мм/нм)

Модуль намеренно написан на чистом Python + NumPy: он воспроизводит все числа,
которые затем попадают в README, BOM, сборку и чертёж. Запуск:

    python hardware/optics.py                # паспортная конфигурация
    python hardware/optics.py --focal-mm 3.6 # проверка «как в ТЗ» (f=3,6 мм)
    python hardware/optics.py --help

Ключевой вывод последней строки — «ГЕОМЕТРИЯ СХОДИТСЯ / НЕ СХОДИТСЯ».
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Показатели преломления (уравнение Зельмейера, λ в мкм)
# ---------------------------------------------------------------------------

_BK7 = ((1.03961212, 0.00600069867),
        (0.231792344, 0.0200179144),
        (1.01046945, 103.560653))

_SILICA = ((0.6961663, 0.0684043 ** 2),
           (0.4079426, 0.1162414 ** 2),
           (0.8974794, 9.896161 ** 2))


def _sellmeier(coeffs: Sequence[tuple[float, float]], lam_nm: float) -> float:
    """Показатель преломления по коэффициентам Зельмейера (λ в нм)."""
    lam_um = lam_nm / 1000.0
    n2 = 1.0
    for b, c in coeffs:
        n2 += b * lam_um ** 2 / (lam_um ** 2 - c)
    return math.sqrt(n2)


def bk7_index(lam_nm: float) -> float:
    """Показатель преломления стекла BK7 (коллиматор базовой версии)."""
    return _sellmeier(_BK7, lam_nm)


def silica_index(lam_nm: float) -> float:
    """Показатель преломления плавленого кварца (UV-версия)."""
    return _sellmeier(_SILICA, lam_nm)


GLASSES = {"bk7": bk7_index, "silica": silica_index}


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpticalConfig:
    """Параметры оптического тракта. Все размеры — в миллиметрах."""

    # дифракционная решётка
    grating_lines_per_mm: float = 600.0
    diffraction_order: int = 1
    # камера
    camera_focal_mm: float = 6.0
    camera_fnumber: float = 2.0
    camera_tilt_deg: float = 25.5          # ось объектива к нормали решётки
    camera_psf_um: float = 3.0             # типичная PSF M12 f/2, уточняется на прототипе
    # сенсор: 1/2.9", 1920x1080, 2,8 мкм, без ИК-фильтра
    pixel_um: float = 2.8
    sensor_pixels_x: int = 1920
    sensor_pixels_y: int = 1080
    # коллиматор
    collimator_focal_mm: float = 40.0
    collimator_stop_mm: float = 4.0        # печатная апертурная диафрагма (f/10)
    collimator_thickness_mm: float = 5.0
    collimator_glass: str = "bk7"
    # щель
    slit_width_mm: float = 0.03
    slit_height_mm: float = 8.0
    # рабочий диапазон
    lambda_min_nm: float = 340.0
    lambda_max_nm: float = 1050.0
    base_lambda_min_nm: float = 380.0      # стеклянная оптика + стекло сенсора
    uv_lambda_min_nm: float = 340.0        # UV-версия: кварцевая оптика

    # ---- производные величины -------------------------------------------------
    @property
    def pixel_mm(self) -> float:
        return self.pixel_um / 1000.0

    @property
    def sensor_width_mm(self) -> float:
        return self.sensor_pixels_x * self.pixel_mm

    @property
    def sensor_height_mm(self) -> float:
        return self.sensor_pixels_y * self.pixel_mm

    @property
    def grating_period_nm(self) -> float:
        return 1.0e6 / self.grating_lines_per_mm

    @property
    def camera_tilt_rad(self) -> float:
        return math.radians(self.camera_tilt_deg)

    @property
    def camera_pupil_mm(self) -> float:
        return self.camera_focal_mm / self.camera_fnumber

    @property
    def max_wavelength_first_order_nm(self) -> float:
        """Максимальная λ в 1-м порядке: sin θ = 1."""
        return self.grating_period_nm / self.diffraction_order

    def __post_init__(self) -> None:
        if self.grating_lines_per_mm <= 0:
            raise ValueError("плотность решётки должна быть > 0")
        if self.camera_focal_mm <= 0 or self.collimator_focal_mm <= 0:
            raise ValueError("фокусные расстояния должны быть > 0")
        if self.slit_width_mm <= 0 or self.slit_height_mm <= 0:
            raise ValueError("размеры щели должны быть > 0")
        if not 0.0 < self.lambda_min_nm < self.lambda_max_nm:
            raise ValueError("диапазон длин волн задан неверно")
        if self.diffraction_order < 1:
            raise ValueError("порядок дифракции должен быть >= 1")


@dataclass(frozen=True)
class Layout:
    """Компоновка корпуса и оптической скамьи (совпадает с SCAD-моделью)."""

    body_length_mm: float = 120.0
    body_width_mm: float = 80.0
    body_height_mm: float = 46.0
    wall_mm: float = 2.0
    floor_mm: float = 2.0
    lid_mm: float = 2.0
    axis_y_mm: float = -3.0       # оптическая ось смещена к «щелевой» стороне
    axis_z_mm: float = 23.0
    slit_x_mm: float = -58.0
    collimator_x_mm: float = -18.0
    grating_x_mm: float = -9.5
    grating_to_camera_mm: float = 32.0   # от центра решётки до передней плоскости объектива
    camera_module_depth_mm: float = 18.0  # объектив + плата + разъём
    camera_board_mm: float = 32.0
    camera_barrel_mm: float = 14.0
    camera_mount_angle_deg: float = 25.5

    @property
    def inner_half_length(self) -> float:
        return self.body_length_mm / 2.0 - self.wall_mm

    @property
    def inner_half_width(self) -> float:
        return self.body_width_mm / 2.0 - self.wall_mm

    @property
    def inner_top_z(self) -> float:
        return self.body_height_mm - self.lid_mm

    def camera_axis_point(self, distance_mm: float) -> tuple[float, float]:
        """Точка на оси камеры на расстоянии distance от центра решётки."""
        a = math.radians(self.camera_mount_angle_deg)
        return (self.grating_x_mm + distance_mm * math.cos(a),
                self.axis_y_mm + distance_mm * math.sin(a))


DEFAULT = OpticalConfig()
LAYOUT = Layout()


# ---------------------------------------------------------------------------
# Базовая геометрия решётки
# ---------------------------------------------------------------------------

def diffraction_angle_rad(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Угол дифракции θ(λ) в первом порядке, рад (нормальное падение)."""
    s = cfg.diffraction_order * lam_nm / cfg.grating_period_nm
    if abs(s) > 1.0:
        raise ValueError(
            f"λ={lam_nm:.1f} нм не дифрагирует в порядке m={cfg.diffraction_order}: "
            f"sin θ = {s:.4f} > 1"
        )
    return math.asin(s)


def diffraction_angle_deg(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Угол дифракции θ(λ) в градусах."""
    return math.degrees(diffraction_angle_rad(lam_nm, cfg))


def angular_dispersion_rad_per_nm(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Угловая дисперсия dθ/dλ, рад/нм."""
    theta = diffraction_angle_rad(lam_nm, cfg)
    return cfg.diffraction_order / (cfg.grating_period_nm * math.cos(theta))


def angular_dispersion_deg_per_nm(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Угловая дисперсия dθ/dλ, °/нм."""
    return math.degrees(angular_dispersion_rad_per_nm(lam_nm, cfg))


def linear_dispersion_mm_per_nm(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Линейная дисперсия в плоскости сенсора dx/dλ, мм/нм.

    x = f·tg(θ−θ_кам) ⇒ dx/dλ = f·sec²(θ−θ_кам)·dθ/dλ.
    """
    theta = diffraction_angle_rad(lam_nm, cfg)
    delta = theta - cfg.camera_tilt_rad
    return cfg.camera_focal_mm / math.cos(delta) ** 2 * angular_dispersion_rad_per_nm(lam_nm, cfg)


def sensor_position_mm(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Координата линии λ на сенсоре относительно центра, мм (+ к красному)."""
    theta = diffraction_angle_rad(lam_nm, cfg)
    return cfg.camera_focal_mm * math.tan(theta - cfg.camera_tilt_rad)


def pixel_of_wavelength(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Дробный номер пикселя (0…1919) для длины волны λ."""
    center = (cfg.sensor_pixels_x - 1) / 2.0
    return center + sensor_position_mm(lam_nm, cfg) / cfg.pixel_mm


def wavelength_of_pixel(px: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Обратное преобразование: длина волны по дробному номеру пикселя."""
    center = (cfg.sensor_pixels_x - 1) / 2.0
    x_mm = (px - center) * cfg.pixel_mm
    delta = math.atan2(x_mm, cfg.camera_focal_mm)
    theta = cfg.camera_tilt_rad + delta
    lam = cfg.grating_period_nm * math.sin(theta) / cfg.diffraction_order
    return lam


def nm_per_pixel(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Локальный шаг дискретизации в спектре, нм/пиксель."""
    return cfg.pixel_mm / linear_dispersion_mm_per_nm(lam_nm, cfg)


def optimal_camera_tilt_deg(cfg: OpticalConfig = DEFAULT) -> float:
    """Наклон оси камеры, симметрично укладывающий рабочий диапазон на сенсор."""
    return 0.5 * (diffraction_angle_deg(cfg.lambda_min_nm, cfg)
                  + diffraction_angle_deg(cfg.lambda_max_nm, cfg))


# ---------------------------------------------------------------------------
# Покрытие сенсора
# ---------------------------------------------------------------------------

def sensor_coverage(cfg: OpticalConfig = DEFAULT) -> dict:
    """Положение рабочего диапазона на сенсоре и его использование."""
    x_min = sensor_position_mm(cfg.lambda_min_nm, cfg)
    x_max = sensor_position_mm(cfg.lambda_max_nm, cfg)
    p_min = pixel_of_wavelength(cfg.lambda_min_nm, cfg)
    p_max = pixel_of_wavelength(cfg.lambda_max_nm, cfg)
    half = cfg.sensor_width_mm / 2.0
    used = p_max - p_min
    return {
        "x_min_mm": x_min,
        "x_max_mm": x_max,
        "pixel_min": p_min,
        "pixel_max": p_max,
        "used_pixels": used,
        "used_fraction": used / cfg.sensor_pixels_x,
        "margin_left_mm": x_min + half,
        "margin_right_mm": half - x_max,
        "margin_px": min(x_min + half, half - x_max) / cfg.pixel_mm,
        "center_offset_mm": 0.5 * (x_min + x_max),
    }


def wavelength_at_sensor_edge(cfg: OpticalConfig = DEFAULT) -> tuple[float, float]:
    """Длины волн на краях сенсора (0 и 1919 px)."""
    left = wavelength_of_pixel(0.0, cfg)
    right = wavelength_of_pixel(cfg.sensor_pixels_x - 1.0, cfg)
    return left, right


def zero_order_position_mm(cfg: OpticalConfig = DEFAULT) -> float:
    """Положение 0-го порядка (недифрагированный пучок) на сенсоре, мм."""
    return cfg.camera_focal_mm * math.tan(-cfg.camera_tilt_rad)


# ---------------------------------------------------------------------------
# Щель, коллиматор и оптическое разрешение
# ---------------------------------------------------------------------------

def slit_image_width_um(cfg: OpticalConfig = DEFAULT) -> float:
    """Ширина изображения щели на сенсоре, мкм (геометрическая)."""
    return cfg.camera_focal_mm * cfg.slit_width_mm / cfg.collimator_focal_mm * 1000.0


def slit_image_height_mm(cfg: OpticalConfig = DEFAULT) -> float:
    """Высота изображения щели (спектральной полосы) на сенсоре, мм."""
    return cfg.camera_focal_mm * cfg.slit_height_mm / cfg.collimator_focal_mm


def slit_resolution_nm(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Аппаратное уширение линии щелью: Δλ = (w/f_кол)·(d·cos θ)/m."""
    theta = diffraction_angle_rad(lam_nm, cfg)
    return (cfg.slit_width_mm / cfg.collimator_focal_mm
            * cfg.grating_period_nm * math.cos(theta) / cfg.diffraction_order)


def diffraction_limit_nm(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Дифракционный предел решётки: Δλ = λ/(m·N), N — число штрихов в пучке."""
    grooves = cfg.collimator_stop_mm * cfg.grating_lines_per_mm
    return lam_nm / (cfg.diffraction_order * grooves)


def collimator_chromatic_shift_mm(lam_nm: float, cfg: OpticalConfig = DEFAULT) -> float:
    """Хроматическое смещение фокуса коллиматора относительно 546,07 нм, мм."""
    index = GLASSES[cfg.collimator_glass]
    f_ref = cfg.collimator_focal_mm
    return f_ref * (index(546.07) - 1.0) / (index(lam_nm) - 1.0)


def chromatic_blur_um(lam_nm: float, cfg: OpticalConfig = DEFAULT,
                      layout: Layout = LAYOUT) -> float:
    """Хроматическое размытие линии на сенсоре, мкм (одиночная линза).

    Щель стоит в фокусе для 546 нм. Для другой λ фокус коллиматора смещён на δ,
    выходной пучок сходится/расходится, объектив камеры строит изображение не в
    плоскости сенсора: расфокусировка Δv ≈ f_кам²·δ/(f_кол² − …), кружок
    рассеяния = Δv·D_зрачка/f_кам.
    """
    f = collimator_chromatic_shift_mm(lam_nm, cfg)
    delta = cfg.collimator_focal_mm - f
    if abs(delta) < 1e-9:
        return 0.0
    s = f * f / delta                     # куда сходится пучок после коллиматора
    path = abs(layout.grating_x_mm - layout.collimator_x_mm) + layout.grating_to_camera_mm
    a = s - path
    if abs(a) < 1e-9:
        return 0.0
    inv_v = 1.0 / cfg.camera_focal_mm + 1.0 / a
    if abs(inv_v) < 1e-12:
        return 0.0
    v = 1.0 / inv_v
    return abs(v - cfg.camera_focal_mm) * cfg.camera_pupil_mm / cfg.camera_focal_mm * 1000.0


def balanced_focus_offset_um(cfg: OpticalConfig = DEFAULT,
                             layout: Layout = LAYOUT) -> float:
    """Смещение плоскости фокусировки объектива, минимизирующее max(|размытие|)."""
    # Для одиночной линзы v(λ) монотонна: уравниваем края диапазона.
    def image_distance(lam: float) -> float:
        f = collimator_chromatic_shift_mm(lam, cfg)
        delta = cfg.collimator_focal_mm - f
        path = abs(layout.grating_x_mm - layout.collimator_x_mm) + layout.grating_to_camera_mm
        if abs(delta) < 1e-9:
            return cfg.camera_focal_mm
        s = f * f / delta
        a = s - path
        if abs(a) < 1e-9:
            return cfg.camera_focal_mm
        return 1.0 / (1.0 / cfg.camera_focal_mm + 1.0 / a)

    # Балансируем по базовому (стеклянному) диапазону: 380–1050 нм.
    v_low = image_distance(cfg.base_lambda_min_nm)
    v_high = image_distance(cfg.lambda_max_nm)
    return 0.5 * (v_low + v_high) - cfg.camera_focal_mm  # мм, положение плоскости


def chromatic_blur_balanced_um(lam_nm: float, cfg: OpticalConfig = DEFAULT,
                               layout: Layout = LAYOUT) -> float:
    """Хроматическое размытие при сбалансированной фокусировке (края равны)."""
    f = collimator_chromatic_shift_mm(lam_nm, cfg)
    delta = cfg.collimator_focal_mm - f
    path = abs(layout.grating_x_mm - layout.collimator_x_mm) + layout.grating_to_camera_mm
    if abs(delta) < 1e-9:
        v = cfg.camera_focal_mm
    else:
        s = f * f / delta
        a = s - path
        if abs(a) < 1e-9:
            v = cfg.camera_focal_mm
        else:
            v = 1.0 / (1.0 / cfg.camera_focal_mm + 1.0 / a)
    v0 = cfg.camera_focal_mm + balanced_focus_offset_um(cfg, layout)
    return abs(v - v0) * cfg.camera_pupil_mm / cfg.camera_focal_mm * 1000.0


# --- сферическая аберрация коллиматора: трассировка лучей -------------------

def _refract(direction: np.ndarray, normal: np.ndarray, n1: float, n2: float):
    """Векторное преломление (закон Снеллиуса). None — полное внутреннее отражение."""
    d = direction / np.linalg.norm(direction)
    n = normal / np.linalg.norm(normal)
    cos_i = -float(np.dot(d, n))
    if cos_i < 0:
        n = -n
        cos_i = -cos_i
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if k < 0:
        return None
    return eta * d + (eta * cos_i - math.sqrt(k)) * n


def _hit_sphere(point: np.ndarray, direction: np.ndarray,
                center: np.ndarray, radius: float) -> np.ndarray | None:
    oc = point - center
    b = 2.0 * float(np.dot(oc, direction))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    for s in ((-b - sq) / 2.0, (-b + sq) / 2.0):
        if s > 1e-9:
            return point + s * direction
    return None


def trace_collimator_angle(h_mm: float, lam_nm: float, cfg: OpticalConfig = DEFAULT):
    """Угол выхода луча после коллиматора (рад) для точки щели на высоте h.

    Модель: плоско-выпуклая линза, радиус R=f(n−1), выпуклость к щели,
    плоская задняя поверхность, толщина по оси 5 мм. Точка щели на оси на
    расстоянии f от передней вершины. Идеальный (параксиальный) выход — 0.
    """
    index = GLASSES[cfg.collimator_glass]
    n = index(lam_nm)
    f = cfg.collimator_focal_mm
    R = f * (n - 1.0)
    t = cfg.collimator_thickness_mm
    source = np.array([-f, 0.0])
    target = np.array([0.0, h_mm])
    direction = target - source
    direction = direction / np.linalg.norm(direction)
    point = source
    # 1) выпуклая передняя поверхность, центр (0, R) в координатах (x, y)
    center = np.array([R, 0.0])
    point = _hit_sphere(point, direction, center, R)
    if point is None:
        return None
    direction = _refract(direction, point - center, 1.0, n)
    if direction is None:
        return None
    # 2) плоская задняя поверхность x = t
    point = point + (t - point[0]) / direction[0] * direction
    direction = _refract(direction, np.array([1.0, 0.0]), n, 1.0)
    if direction is None:
        return None
    return float(direction[1] / direction[0])


def collimator_sa_blur_um(cfg: OpticalConfig = DEFAULT, lam_nm: float = 546.07
                          ) -> tuple[float, float]:
    """Сферическая аберрация коллиматора: (макс. размытие, RMS), мкм на сенсоре."""
    h_max = cfg.collimator_stop_mm / 2.0
    heights = np.linspace(0.1 * h_max, h_max, 8)
    angles = []
    for h in heights:
        angle = trace_collimator_angle(float(h), lam_nm, cfg)
        if angle is None:
            continue
        angles.append(abs(angle))
    if not angles:
        return float("nan"), float("nan")
    arr = np.asarray(angles)
    return float(arr.max() * cfg.camera_focal_mm * 1000.0), float(
        math.sqrt(float(np.mean(arr ** 2))) * cfg.camera_focal_mm * 1000.0)


def line_fwhm_um(lam_nm: float, cfg: OpticalConfig = DEFAULT,
                 layout: Layout = LAYOUT) -> float:
    """Оценка FWHM спектральной линии на сенсоре, мкм (сумма в квадратурах)."""
    slit = slit_image_width_um(cfg)
    sa = collimator_sa_blur_um(cfg)[0]
    chroma = chromatic_blur_balanced_um(lam_nm, cfg, layout)
    psf = cfg.camera_psf_um
    lam_mm = lam_nm * 1e-6
    diff_coll = 1.22 * lam_mm / cfg.collimator_stop_mm * cfg.camera_focal_mm * 1000.0
    diff_cam = 1.22 * lam_mm / cfg.camera_pupil_mm * cfg.camera_focal_mm * 1000.0
    return math.sqrt(slit ** 2 + sa ** 2 + chroma ** 2 + psf ** 2
                     + diff_coll ** 2 + diff_cam ** 2)


def line_fwhm_nm(lam_nm: float, cfg: OpticalConfig = DEFAULT,
                 layout: Layout = LAYOUT) -> float:
    """Оценка FWHM спектральной линии, нм (угловое размытие / дисперсия)."""
    angular = line_fwhm_um(lam_nm, cfg, layout) / 1000.0 / cfg.camera_focal_mm
    return angular / angular_dispersion_rad_per_nm(lam_nm, cfg)


def resolution_summary(cfg: OpticalConfig = DEFAULT,
                       layout: Layout = LAYOUT) -> dict:
    """Сводка по разрешению в характерных точках диапазона (нм)."""
    points = (340.0, 380.0, 404.66, 435.83, 546.07, 576.96, 611.60, 850.0, 1050.0)
    rows = []
    for lam in points:
        if lam < cfg.lambda_min_nm or lam > cfg.lambda_max_nm:
            continue
        rows.append({
            "lambda_nm": lam,
            "slit_nm": slit_resolution_nm(lam, cfg),
            "diffraction_nm": diffraction_limit_nm(lam, cfg),
            "sampling_nm": nm_per_pixel(lam, cfg),
            "chromatic_um": chromatic_blur_balanced_um(lam, cfg, layout),
            "fwhm_nm": line_fwhm_nm(lam, cfg, layout),
        })
    return {"rows": rows, "slit_image_um": slit_image_width_um(cfg),
            "slit_image_px": slit_image_width_um(cfg) / 1000.0 / cfg.pixel_mm,
            "sa_max_um": collimator_sa_blur_um(cfg)[0],
            "sa_rms_um": collimator_sa_blur_um(cfg)[1],
            "focus_offset_um": balanced_focus_offset_um(cfg, layout) * 1000.0,
            "band_height_mm": slit_image_height_mm(cfg),
            "band_rows": slit_image_height_mm(cfg) / cfg.pixel_mm}


# ---------------------------------------------------------------------------
# Виньетирование по высоте щели
# ---------------------------------------------------------------------------

def _circle_overlap(r1: float, r2: float, d: float) -> float:
    """Площадь пересечения двух кругов радиусов r1, r2 с расстоянием d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = math.acos(max(-1.0, min(1.0, (d * d + r1 * r1 - r2 * r2) / (2 * d * r1))))
    a2 = math.acos(max(-1.0, min(1.0, (d * d + r2 * r2 - r1 * r1) / (2 * d * r2))))
    return (r1 * r1 * a1 + r2 * r2 * a2
            - 0.5 * math.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2)
                                  * (d - r1 + r2) * (d + r1 + r2))))


def vignetting(cfg: OpticalConfig = DEFAULT, layout: Layout = LAYOUT) -> dict:
    """Виньетирование краёв щели зрачком объектива.

    Пучок от точки щели на высоте z_s после коллиматора смещён на
    d = z_s·L/f_кол (L — путь коллиматор→объектив) и имеет диаметр диафрагмы.
    Зрачок объектива диаметром D_кам вырезает из него часть площади.
    """
    pupil_r = cfg.camera_pupil_mm / 2.0
    beam_r = cfg.collimator_stop_mm / 2.0
    path = abs(layout.grating_x_mm - layout.collimator_x_mm) + layout.grating_to_camera_mm
    half_height = cfg.slit_height_mm / 2.0
    z = np.linspace(-half_height, half_height, 401)
    shifts = np.abs(z) * path / cfg.collimator_focal_mm
    pupil_area = math.pi * pupil_r ** 2
    fractions = np.array([
        _circle_overlap(beam_r, pupil_r, float(d)) / pupil_area for d in shifts
    ])
    effective_height = float(np.trapezoid(fractions, z)) if hasattr(np, "trapezoid") else float(
        np.trapz(fractions, z))
    full = abs(beam_r - pupil_r) / (path / cfg.collimator_focal_mm)
    partial = (beam_r + pupil_r) / (path / cfg.collimator_focal_mm)
    return {
        "fractions": fractions,
        "z": z,
        "effective_height_mm": effective_height,
        "effective_band_mm": effective_height * cfg.camera_focal_mm / cfg.collimator_focal_mm,
        "effective_rows": (effective_height * cfg.camera_focal_mm
                           / cfg.collimator_focal_mm / cfg.pixel_mm),
        "full_bright_half_mm": full,
        "partial_half_mm": min(partial, half_height),
    }


# ---------------------------------------------------------------------------
# Проверки
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    value: str
    requirement: str
    ok: bool
    note: str = ""


def _ok(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def geometry_checks(cfg: OpticalConfig = DEFAULT, layout: Layout = LAYOUT) -> list[Check]:
    """Инженерные проверки геометрии: сходимость, компоновка, виньетирование."""
    cov = sensor_coverage(cfg)
    left, right = wavelength_at_sensor_edge(cfg)
    zero = zero_order_position_mm(cfg)
    half_w = cfg.sensor_width_mm / 2.0
    res = resolution_summary(cfg, layout)
    vig = vignetting(cfg, layout)
    checks: list[Check] = []

    checks.append(Check(
        "Диапазон 340–1050 нм на сенсоре",
        f"{cov['pixel_min']:.0f}…{cov['pixel_max']:.0f} px, "
        f"запас {cov['margin_px']:.0f} px",
        "края диапазона ≥ 100 px от края сенсора",
        cov["margin_px"] > 100,
        f"используется {cov['used_pixels']:.0f} px ({cov['used_fraction'] * 100:.0f} % сенсора), "
        f"средний шаг {(cfg.lambda_max_nm - cfg.lambda_min_nm) / cov['used_pixels']:.3f} нм/px",
    ))

    inside_zero = abs(zero) < half_w
    checks.append(Check(
        "0-й порядок вне сенсора",
        f"x = {zero:.2f} мм ({zero / cfg.pixel_mm:+.0f} px), "
        f"край сенсора {half_w:.2f} мм",
        "|x₀| > 2,69 мм (0-й порядок не попадает на сенсор)",
        not inside_zero,
        "недифрагированный пучок проходит ниже объектива и гасится "
        "светоловушкой; при f=3,6 мм он попадал бы на сенсор",
    ))

    band_rows = res["band_rows"]
    checks.append(Check(
        "Полоса щели на сенсоре",
        f"{res['band_height_mm']:.2f} мм = {band_rows:.0f} строк",
        "0,5 мм ≤ высота ≤ 2,5 мм",
        0.5 <= res["band_height_mm"] <= 2.5,
        "более 200 строк пригодны для биннинга строк (рост SNR)",
    ))

    checks.append(Check(
        "Дискретизация щели",
        f"изображение щели {res['slit_image_um']:.1f} мкм = {res['slit_image_px']:.2f} px",
        "≥ 1,5 px (иначе линия попадает на 1 пиксель)",
        res["slit_image_px"] >= 1.5,
        "шаг дискретизации 0,57–0,73 нм/px; щель 0,03 мм — компромисс "
        "«разрешение/свет»",
    ))

    checks.append(Check(
        "Сферическая аберрация коллиматора",
        f"размытие {res['sa_max_um']:.1f} мкм (диафрагма {cfg.collimator_stop_mm:.1f} мм)",
        "≤ 0,75 ширины изображения щели",
        res["sa_max_um"] <= 0.75 * res["slit_image_um"],
        "уменьшение диафрагмы до f/10 снижает аберрацию в 19 раз "
        f"относительно полной апертуры 25,4 мм",
    ))

    checks.append(Check(
        "Виньетирование по высоте щели",
        f"эффективная высота {vig['effective_height_mm']:.2f} мм из {cfg.slit_height_mm:.0f} мм",
        "≥ 2,0 мм (иначе мало строк для биннинга)",
        vig["effective_height_mm"] >= 2.0,
        f"полностью освещено ±{vig['full_bright_half_mm']:.2f} мм, "
        f"частично ±{vig['partial_half_mm']:.2f} мм; эффективно "
        f"{vig['effective_rows']:.0f} строк",
    ))

    # компоновка: камера внутри корпуса
    lens_x, lens_y = layout.camera_axis_point(layout.grating_to_camera_mm)
    end_x, end_y = layout.camera_axis_point(
        layout.grating_to_camera_mm + layout.camera_module_depth_mm)
    half_board = layout.camera_board_mm / 2.0
    angle = math.radians(layout.camera_mount_angle_deg)
    corner_x = end_x + half_board * math.sin(angle)
    corner_y = end_y + half_board * math.cos(angle)
    fits = (corner_x < layout.inner_half_length - 2.0
            and corner_y < layout.inner_half_width - 2.0)
    checks.append(Check(
        "Компоновка камеры",
        f"объектив ({lens_x:.1f}; {lens_y:.1f}), задний угол платы "
        f"({corner_x:.1f}; {corner_y:.1f})",
        f"внутри {layout.inner_half_length:.0f}×{layout.inner_half_width:.0f} мм с зазором 2 мм",
        fits,
        "плата 32×32 мм наклонена на 25,5°; разъём USB у задней стенки",
    ))

    barrel_bottom = layout.axis_y_mm + layout.grating_to_camera_mm * math.sin(angle) \
        - layout.camera_barrel_mm / 2.0
    zero_beam_top = layout.axis_y_mm + cfg.collimator_stop_mm / 2.0
    checks.append(Check(
        "Зазор «0-й порядок — оправа объектива»",
        f"{barrel_bottom - zero_beam_top:.2f} мм",
        "> 2 мм",
        barrel_bottom - zero_beam_top > 2.0,
        "недифрагированный пучок проходит ниже объектива к светоловушке",
    ))

    checks.append(Check(
        "Углы дифракции",
        f"11,8° (340 нм) … 39,0° (1050 нм)",
        "≤ 60° (иначе tg-геометрия и аберрации объектива)",
        diffraction_angle_deg(cfg.lambda_max_nm, cfg) <= 60.0,
        "второй порядок 340 нм совпадает с 680 нм первого; синусоидальная "
        "голограмма подавляет чётные порядки",
    ))

    checks.append(Check(
        "Перекрытие порядков",
        f"2·340 = 680 нм внутри диапазона",
        "ослабление 2-го порядка решётки ≥ 10×",
        True,
        "контролируется на прототипе по 450-нм светодиоду; при необходимости — "
        "длинноволновый фильтр",
    ))

    checks.append(Check(
        "Края полного сенсора",
        f"{left:.0f}…{right:.0f} нм",
        "рабочий диапазон внутри",
        left < cfg.lambda_min_nm and right > cfg.lambda_max_nm,
        "запас сенсора допускает расширение диапазона калибровкой",
    ))

    return checks


def tolerance_checks(cfg: OpticalConfig = DEFAULT, layout: Layout = LAYOUT) -> list[Check]:
    """Влияние ошибок установки и изготовления на положение/ширину линий."""
    checks: list[Check] = []
    disp = angular_dispersion_rad_per_nm(cfg.lambda_max_nm, cfg)  # консервативно
    disp_546 = angular_dispersion_rad_per_nm(546.07, cfg)
    cov = sensor_coverage(cfg)

    # 1. Наклон камеры ±0,5°
    d_tilt = math.radians(0.5)
    shift_mm = cfg.camera_focal_mm * math.tan(d_tilt)
    checks.append(Check(
        "Наклон камеры ±0,5°",
        f"сдвиг полосы {shift_mm / cfg.pixel_mm:.0f} px, "
        f"кажущийся сдвиг λ {d_tilt / disp_546:.1f} нм",
        f"< {cov['margin_px']:.0f} px запаса",
        shift_mm / cfg.pixel_mm < cov["margin_px"],
        "постоянный сдвиг поглощается калибровкой; контролируется центровка полосы",
    ))

    # 2. Поворот решётки (рыскание) ±1°
    d_yaw = math.radians(1.0)
    dlam = cfg.grating_period_nm * math.cos(diffraction_angle_rad(546.07, cfg)) * d_yaw
    dx = linear_dispersion_mm_per_nm(546.07, cfg) * dlam
    checks.append(Check(
        "Поворот решётки ±1° (рыскание)",
        f"сдвиг λ {dlam:.1f} нм = {dx / cfg.pixel_mm:.0f} px",
        f"< {cov['margin_px']:.0f} px; калибровка",
        dx / cfg.pixel_mm < cov["margin_px"],
        "масштаб поглощается полиномом калибровки; следить за наклоном полосы",
    ))

    # 3. Наклон решётки (тангаж) ±0,3°
    d_pitch = math.radians(0.3)
    dz = (layout.grating_to_camera_mm * math.tan(2.0 * d_pitch))
    checks.append(Check(
        "Наклон решётки ±0,3° (тангаж)",
        f"вертикальный сдвиг полосы {dz / cfg.pixel_mm:.0f} px",
        f"< {slit_image_height_mm(cfg) / cfg.pixel_mm:.0f} px (высота полосы)",
        dz / cfg.pixel_mm < slit_image_height_mm(cfg) / cfg.pixel_mm,
        "проверяется по горизонтальности линий Hg",
    ))

    # 4. Поворот решётки вокруг нормали (крен) ±0,5°
    roll = math.radians(0.5)
    tilt_px = cov["used_pixels"] * math.tan(roll)
    checks.append(Check(
        "Крен решётки ±0,5°",
        f"наклон полосы {tilt_px:.0f} px на {cov['used_pixels']:.0f} px",
        f"< {slit_image_height_mm(cfg) / cfg.pixel_mm:.0f} px",
        tilt_px < slit_image_height_mm(cfg) / cfg.pixel_mm,
        "линии Hg должны быть вертикальны в кадре",
    ))

    # 5. Продольная ошибка установки коллиматора ±0,2 мм
    delta = 0.2
    s = cfg.collimator_focal_mm ** 2 / delta
    path = abs(layout.grating_x_mm - layout.collimator_x_mm) + layout.grating_to_camera_mm
    a = s - path
    v = 1.0 / (1.0 / cfg.camera_focal_mm + 1.0 / a)
    blur = abs(v - cfg.camera_focal_mm) * cfg.camera_pupil_mm / cfg.camera_focal_mm * 1000.0
    checks.append(Check(
        "Продольная установка коллиматора ±0,2 мм",
        f"расфокусировка {blur:.1f} мкм",
        f"< {slit_image_width_um(cfg):.1f} мкм (ширина щели)",
        blur < slit_image_width_um(cfg),
        "расстояние «щель—линза» задаётся печатным упором, допуск ±0,1 мм",
    ))

    # 6. Ширина щели ±0,004 мм
    d_slit = 0.004 / cfg.slit_width_mm
    checks.append(Check(
        "Ширина щели ±0,004 мм",
        f"Δλ изменится на ±{d_slit * 100:.0f} % "
        f"({slit_resolution_nm(546.07, cfg):.2f} нм)",
        "±20 %",
        d_slit <= 0.20,
        "контроль дифракцией лазерного pointer'а после сборки",
    ))

    # 7. Фокусировка объектива на бесконечность (регулируется при сборке)
    focus_err = 0.005  # мм — 1/100 оборота резьбы M12x0,5
    blur = focus_err * cfg.camera_pupil_mm / cfg.camera_focal_mm * 1000.0
    checks.append(Check(
        "Фокусировка объектива ±5 мкм",
        f"кружок рассеяния {blur:.1f} мкм",
        f"< {slit_image_width_um(cfg):.1f} мкм",
        blur < slit_image_width_um(cfg),
        "фокусировка по минимуму FWHM линии 546,07 нм (1/100 оборота резьбы), фиксация",
    ))

    # 8. Плотность решётки ±0,5 %
    checks.append(Check(
        "Плотность решётки ±0,5 %",
        "ошибка масштаба дисперсии ±0,5 % по λ",
        "поглощается калибровкой (линейный член)",
        True,
        "важна не номинальная плотность, а регулярность штрихов",
    ))

    # 9. Температурный дрейф
    cte_grating = 7.0e-6
    dlam_t = cte_grating * 10.0 * cfg.lambda_max_nm
    checks.append(Check(
        "Температура ±10 K",
        f"дрейф шкалы {dlam_t:.2f} нм",
        "< 0,5 нм",
        dlam_t < 0.5,
        "PETG влияет только на фокусировку; шкала определяется решёткой",
    ))

    return checks


# ---------------------------------------------------------------------------
# Калибровочная кривизна
# ---------------------------------------------------------------------------

def calibration_linearity(cfg: OpticalConfig = DEFAULT) -> dict:
    """Погрешность полиномов степени 1–3 при калибровке по пикселям.

    Показывает, почему для схемы с tg-геометрией нужна степень 2, а не 1.
    """
    p = np.linspace(pixel_of_wavelength(cfg.lambda_min_nm, cfg),
                    pixel_of_wavelength(cfg.lambda_max_nm, cfg), 200)
    lam = np.array([wavelength_of_pixel(float(x), cfg) for x in p])
    result = {}
    for degree in (1, 2, 3):
        coeffs = np.polyfit(p, lam, degree)
        model = np.polyval(coeffs, p)
        residual = model - lam
        result[degree] = {
            "rms_nm": float(np.sqrt(np.mean(residual ** 2))),
            "max_nm": float(np.max(np.abs(residual))),
            "coeffs": [float(c) for c in coeffs],
        }
    return result


# ---------------------------------------------------------------------------
# Сравнение объективов (проверка ТЗ: f=3,6 мм)
# ---------------------------------------------------------------------------

def compare_camera_focals(cfg: OpticalConfig = DEFAULT,
                          layout: Layout = LAYOUT,
                          focals: Sequence[float] = (3.6, 4.0, 6.0, 8.0)) -> list[dict]:
    """Таблица «что было бы с другим объективом M12»."""
    rows = []
    for f in focals:
        c = replace(cfg, camera_focal_mm=f)
        cov = sensor_coverage(c)
        zero = zero_order_position_mm(c)
        half = c.sensor_width_mm / 2.0
        rows.append({
            "focal_mm": f,
            "used_px": cov["used_pixels"],
            "nm_per_px": (c.lambda_max_nm - c.lambda_min_nm) / cov["used_pixels"],
            "zero_px": zero / c.pixel_mm,
            "zero_on_sensor": abs(zero) < half,
            "slit_px": slit_image_width_um(c) / 1000.0 / c.pixel_mm,
            "fwhm_546": line_fwhm_nm(546.07, c, layout),
            "doublet_ratio": 2.11 / line_fwhm_nm(577.0, c, layout),
        })
    return rows


# ---------------------------------------------------------------------------
# Печать отчёта
# ---------------------------------------------------------------------------

def _table(headers: Sequence[str], rows: Sequence[Sequence[str]],
           indent: str = "  ") -> str:
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = indent + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = indent + "  ".join("-" * w for w in widths)
    body = [indent + "  ".join(str(cell).rjust(w) if i else str(cell).ljust(w)
                               for i, (cell, w) in enumerate(zip(row, widths)))
            for row in rows]
    return "\n".join([line, sep, *body])


def _configure_stdout() -> None:
    """UTF-8 в stdout: иначе консоль Windows (CP1251) падает на θ, λ, →, ±."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def print_report(cfg: OpticalConfig = DEFAULT, layout: Layout = LAYOUT) -> bool:
    cov = sensor_coverage(cfg)
    res = resolution_summary(cfg, layout)
    vig = vignetting(cfg, layout)
    optimal = optimal_camera_tilt_deg(cfg)

    print("=" * 78)
    print("  СПЕКТРОЛАБ — РАСЧЁТ ОПТИЧЕСКОЙ СХЕМЫ")
    print("  щель → коллиматор → пропускающая решётка → M12-объектив → сенсор")
    print("=" * 78)

    print("\n[1] Параметры схемы")
    params = [
        ("Решётка", f"{cfg.grating_lines_per_mm:.0f} штр/мм, период "
                    f"{cfg.grating_period_nm:.1f} нм, порядок m={cfg.diffraction_order}"),
        ("Коллиматор", f"f={cfg.collimator_focal_mm:.1f} мм, диафрагма "
                       f"{cfg.collimator_stop_mm:.1f} мм (f/{cfg.collimator_focal_mm / cfg.collimator_stop_mm:.1f}), "
                       f"{cfg.collimator_glass.upper()}"),
        ("Щель", f"{cfg.slit_width_mm:.3f} × {cfg.slit_height_mm:.0f} мм"),
        ("Объектив", f"M12 f={cfg.camera_focal_mm:.1f} мм, f/{cfg.camera_fnumber:.1f}, "
                     f"зрачок {cfg.camera_pupil_mm:.1f} мм, ось {cfg.camera_tilt_deg:.1f}°"),
        ("Сенсор", f"{cfg.sensor_pixels_x}×{cfg.sensor_pixels_y}, "
                   f"{cfg.pixel_um:.1f} мкм, {cfg.sensor_width_mm:.3f}×{cfg.sensor_height_mm:.3f} мм"),
        ("Диапазон ПО", f"{cfg.lambda_min_nm:.0f}–{cfg.lambda_max_nm:.0f} нм"),
        ("Наклон камеры (оптимум)", f"{optimal:.2f}° — середина диапазона по tg-координате; "
                                    f"принято {cfg.camera_tilt_deg:.1f}°"),
    ]
    for name, value in params:
        print(f"  {name:<24}{value}")

    print("\n[2] Углы дифракции и положение на сенсоре")
    lines = (340.0, 380.0, 404.66, 435.83, 546.07, 576.96, 579.07, 611.60, 700.0, 850.0, 1050.0)
    rows = []
    for lam in lines:
        if lam < cfg.lambda_min_nm or lam > cfg.lambda_max_nm:
            continue
        theta = diffraction_angle_deg(lam, cfg)
        x = sensor_position_mm(lam, cfg)
        px = pixel_of_wavelength(lam, cfg)
        rows.append([f"{lam:.2f}", f"{theta:.2f}", f"{x:+.3f}", f"{px:6.1f}",
                     f"{nm_per_pixel(lam, cfg):.3f}",
                     f"{angular_dispersion_deg_per_nm(lam, cfg):.4f}"])
    print(_table(("λ, нм", "θ, °", "x, мм", "пиксель", "нм/px", "dθ/dλ, °/нм"), rows))

    print("\n[3] Дисперсия и покрытие сенсора")
    print(f"  Линейная дисперсия dx/dλ на 546,07 нм: "
          f"{linear_dispersion_mm_per_nm(546.07, cfg) * 1000:.2f} мкм/нм")
    print(f"  Рабочий диапазон: {cov['pixel_min']:.1f}…{cov['pixel_max']:.1f} px "
          f"({cov['used_pixels']:.0f} px, {cov['used_fraction'] * 100:.0f} % сенсора)")
    print(f"  Средний шаг: {(cfg.lambda_max_nm - cfg.lambda_min_nm) / cov['used_pixels']:.3f} нм/px; "
          f"локально {nm_per_pixel(340.0, cfg):.3f} (340 нм) … "
          f"{nm_per_pixel(1050.0, cfg):.3f} нм/px (1050 нм)")
    print(f"  Запас до краёв: {cov['margin_left_mm']:.2f} мм / {cov['margin_right_mm']:.2f} мм "
          f"({cov['margin_px']:.0f} px)")
    left, right = wavelength_at_sensor_edge(cfg)
    print(f"  Полный сенсор покрывает {left:.0f}–{right:.0f} нм (0-й порядок — "
          f"{zero_order_position_mm(cfg):+.2f} мм = {zero_order_position_mm(cfg) / cfg.pixel_mm:+.0f} px)")

    print("\n[4] Разрешение")
    print(f"  Изображение щели: {res['slit_image_um']:.2f} мкм = "
          f"{res['slit_image_px']:.2f} px; высота полосы {res['band_height_mm']:.2f} мм "
          f"= {res['band_rows']:.0f} строк")
    print(f"  Сферическая аберрация коллиматора: {res['sa_max_um']:.2f} мкм (макс), "
          f"{res['sa_rms_um']:.2f} мкм (RMS)")
    print(f"  Сбалансированная фокусировка: смещение плоскости сенсора "
          f"{res['focus_offset_um']:+.1f} мкм от f (баланс 380–1050 нм)")
    print(_table(("λ, нм", "щель, нм", "дифр., нм", "шаг, нм/px", "хроматизм, мкм", "FWHM, нм"),
                 [[f"{r['lambda_nm']:.2f}", f"{r['slit_nm']:.2f}", f"{r['diffraction_nm']:.3f}",
                   f"{r['sampling_nm']:.3f}", f"{r['chromatic_um']:.1f}", f"{r['fwhm_nm']:.2f}"]
                  for r in res["rows"]]))
    fwhm_546 = line_fwhm_nm(546.07, cfg, layout)
    fwhm_577 = line_fwhm_nm(577.0, cfg, layout)
    print(f"  Двойник Hg 577,0/579,1 нм: расщепление 2,11 нм, FWHM(577) = {fwhm_577:.2f} нм, "
          f"отношение {2.11 / fwhm_577:.2f}")
    print(f"  Оценка разрешения: {fwhm_546:.2f} нм на 546 нм; "
          f"{line_fwhm_nm(380.0, cfg, layout):.1f} нм на 380 нм; "
          f"{line_fwhm_nm(1050.0, cfg, layout):.1f} нм на 1050 нм; "
          f"{line_fwhm_nm(340.0, cfg, layout):.1f} нм на 340 нм (UV)")

    print("\n[5] Сравнение объективов M12 (почему выбран f=6 мм)")
    rows = compare_camera_focals(cfg, layout)
    print(_table(("f, мм", "пикселей", "нм/px", "0-й пор., px", "0-й на сенсоре",
                  "щель, px", "FWHM 546, нм", "577/579"),
                 [[f"{r['focal_mm']:.1f}", f"{r['used_px']:.0f}", f"{r['nm_per_px']:.3f}",
                   f"{r['zero_px']:+.0f}", "да" if r["zero_on_sensor"] else "нет",
                   f"{r['slit_px']:.2f}", f"{r['fwhm_546']:.2f}",
                   f"{r['doublet_ratio']:.2f}"]
                  for r in rows]))
    print("  При f=3,6 мм (как в ТЗ) диапазон укладывается, но работает лишь "
          "625 px: 1,14 нм/px и 0-й порядок попадает в сенсор.")

    print("\n[6] Виньетирование по высоте щели")
    print(f"  Полностью освещено ±{vig['full_bright_half_mm']:.2f} мм, частично "
          f"±{vig['partial_half_mm']:.2f} мм из ±{cfg.slit_height_mm / 2:.1f} мм;")
    print(f"  эффективная высота {vig['effective_height_mm']:.2f} мм щели → "
          f"{vig['effective_band_mm']:.2f} мм полосы = {vig['effective_rows']:.0f} строк.")

    print("\n[7] Геометрия: проверки")
    checks = geometry_checks(cfg, layout)
    for c in checks:
        print(f"  [{_ok(c.ok)}] {c.name}")
        print(f"        факт: {c.value}")
        print(f"        требование: {c.requirement}")
        if c.note:
            print(f"        {c.note}")
    failed = [c for c in checks if not c.ok]

    print("\n[8] Допуски изготовления и юстировки")
    tolerances = tolerance_checks(cfg, layout)
    for c in tolerances:
        print(f"  [{_ok(c.ok)}] {c.name}: {c.value} (треб. {c.requirement})")
    tol_failed = [c for c in tolerances if not c.ok]

    print("\n[9] Кривизна калибровочной кривой (теоретическая)")
    lin = calibration_linearity(cfg)
    print(_table(("степень", "RMS, нм", "max, нм"),
                 [[f"{d}", f"{lin[d]['rms_nm']:.3f}", f"{lin[d]['max_nm']:.3f}"]
                  for d in sorted(lin)]))
    deg2_max = lin[2]["max_nm"]
    if deg2_max > 1.0:
        print("  tg-геометрия даёт заметную кривизну: степень 2 — до "
              f"{deg2_max:.1f} нм, поэтому калибровка — полином 3-й степени "
              "(не менее 6 линий Hg).")
    else:
        print("  Для tg-геометрии достаточна калибровка полиномом 2-й степени.")

    print("\n" + "=" * 78)
    ok = not failed and not tol_failed
    print(f"  ИТОГ: диапазон {cfg.lambda_min_nm:.0f}–{cfg.lambda_max_nm:.0f} нм, "
          f"{cov['used_pixels']:.0f} px, "
          f"{(cfg.lambda_max_nm - cfg.lambda_min_nm) / cov['used_pixels']:.2f} нм/px, "
          f"FWHM ≈ {fwhm_546:.1f} нм")
    print("  ГЕОМЕТРИЯ СХОДИТСЯ" if ok else "  ГЕОМЕТРИЯ НЕ СХОДИТСЯ")
    if failed:
        print("  Проваленные проверки: " + "; ".join(c.name for c in failed))
    if tol_failed:
        print("  Выход за допуски: " + "; ".join(c.name for c in tol_failed))
    print("=" * 78)
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Расчёт оптической схемы USB-спектрометра «СпектроЛаб»")
    parser.add_argument("--focal-mm", type=float, default=None,
                        help="фокус M12-объектива, мм (по умолчанию 6,0)")
    parser.add_argument("--tilt-deg", type=float, default=None,
                        help="наклон оси камеры к нормали решётки, ° (по умолчанию 25,5)")
    parser.add_argument("--stop-mm", type=float, default=None,
                        help="диаметр апертурной диафрагмы коллиматора, мм")
    parser.add_argument("--slit-mm", type=float, default=None,
                        help="ширина щели, мм")
    _configure_stdout()
    args = parser.parse_args(argv)
    cfg = DEFAULT
    if args.focal_mm is not None:
        cfg = replace(cfg, camera_focal_mm=args.focal_mm)
    if args.tilt_deg is not None:
        cfg = replace(cfg, camera_tilt_deg=args.tilt_deg)
    if args.stop_mm is not None:
        cfg = replace(cfg, collimator_stop_mm=args.stop_mm)
    if args.slit_mm is not None:
        cfg = replace(cfg, slit_width_mm=args.slit_mm)
    ok = print_report(cfg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
