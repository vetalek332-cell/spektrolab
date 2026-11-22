"""Физика теплового излучения: закон Планка, Вина, Стефана — Больцмана."""

from __future__ import annotations

import numpy as np

# Фундаментальные константы (СИ)
H = 6.62607015e-34  # постоянная Планка, Дж·с
C = 299792458.0  # скорость света, м/с
K = 1.380649e-23  # постоянная Больцмана, Дж/К
SIGMA = 5.670374419e-8  # постоянная Стефана — Больцмана, Вт/(м²·К⁴)
WIEN_B = 2.897771955e-3  # постоянная Вина, м·К


def planck_spectral_radiance(wavelength_nm, temperature: float) -> np.ndarray:
    """Спектральная плотность энергетической яркости абсолютно чёрного тела.

    Возвращает значения в Вт·ср⁻¹·м⁻³ по длине волны. Для больших чисел это
    нормально: в подгонке используется нормировка, поэтому абсолютный
    масштаб не важен.
    """
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Температура должна быть положительным числом")
    wl_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    if np.any(wl_m <= 0):
        raise ValueError("Длина волны должна быть положительной")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        exponent = H * C / (wl_m * K * temperature)
        radiance = (2 * H * C**2 / wl_m**5) / np.expm1(exponent)
    return np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)


def planck_normalized(wavelength_nm, temperature: float) -> np.ndarray:
    """Закон Планка, нормированный на максимум (удобно для графиков и фита)."""
    values = planck_spectral_radiance(wavelength_nm, temperature)
    peak = float(np.max(values)) if values.size else 0.0
    return values / peak if peak > 0 else values


def wien_peak_nm(temperature: float) -> float:
    """Длина волны максимума излучения по закону смещения Вина, нм."""
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("Температура должна быть положительной")
    return WIEN_B * 1e9 / temperature


def wien_temperature(peak_nm: float) -> float:
    """Температура тела по положению максимума, К."""
    peak_nm = float(peak_nm)
    if peak_nm <= 0:
        raise ValueError("Длина волны должна быть положительной")
    return WIEN_B * 1e9 / peak_nm


def stefan_boltzmann(temperature: float) -> float:
    """Полная энергетическая светимость чёрного тела, Вт/м²."""
    temperature = float(temperature)
    if temperature < 0:
        raise ValueError("Температура не может быть отрицательной")
    return SIGMA * temperature**4


def measured_peak_nm(wavelengths, intensity) -> float:
    """Положение максимума измеренного спектра с параболическим уточнением."""
    wl = np.asarray(wavelengths, dtype=float)
    values = np.asarray(intensity, dtype=float)
    if wl.size < 3:
        return float(wl[int(np.argmax(values))])
    index = int(np.argmax(values))
    if index == 0 or index == wl.size - 1:
        return float(wl[index])

    y0, y1, y2 = values[index - 1], values[index], values[index + 1]
    denominator = y0 - 2 * y1 + y2
    if denominator == 0:
        return float(wl[index])
    shift = 0.5 * (y0 - y2) / denominator
    step = wl[index + 1] - wl[index]
    return float(wl[index] + shift * step)


def blackbody_rgb(temperature: float) -> tuple[int, int, int]:
    """Приблизительный цвет чёрного тела (для подсветки интерфейса)."""
    t = float(np.clip(temperature, 1000.0, 40000.0)) / 100.0
    if t <= 66:
        red = 255.0
        green = 99.4708025861 * np.log(t) - 161.1195681661
        blue = 0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307
    else:
        red = 329.698727446 * (t - 60) ** -0.1332047592
        green = 288.1221695283 * (t - 60) ** -0.0755148492
        blue = 255.0
    clip = lambda value: int(np.clip(value, 0, 255))
    return clip(red), clip(green), clip(blue)
