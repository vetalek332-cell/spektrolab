"""Синтетические спектры и кадры — для тестов, демо и работы без камеры."""

from __future__ import annotations

import numpy as np

from .planck import planck_spectral_radiance
from .spectrum import Spectrum

__all__ = [
    "default_wavelengths",
    "blackbody_spectrum",
    "line_spectrum",
    "mercury_spectrum",
    "sodium_spectrum",
    "hydrogen_spectrum",
    "led_spectrum",
    "sensor_frame",
]


def default_wavelengths(
    low_nm: float = 340.0, high_nm: float = 1050.0, points: int = 1024
) -> np.ndarray:
    """Рабочая сетка длин волн прибора."""
    return np.linspace(float(low_nm), float(high_nm), int(points))


def blackbody_spectrum(
    temperature: float = 3000.0,
    wavelengths: np.ndarray | None = None,
    noise: float = 0.0,
    offset: float = 0.0,
    seed: int | None = None,
) -> Spectrum:
    """Спектр абсолютно чёрного тела с шумом."""
    wl = default_wavelengths() if wavelengths is None else np.asarray(wavelengths, dtype=float)
    values = planck_spectral_radiance(wl, temperature)
    peak = float(np.max(values))
    if peak > 0:
        values = values / peak
    rng = np.random.default_rng(seed)
    values = values + float(offset)
    if noise > 0:
        values = values + rng.normal(0.0, float(noise), values.shape)
    values = np.clip(values, 0.0, None)
    return Spectrum(wl, values, label=f"Чёрное тело, {temperature:.0f} К")


def line_spectrum(
    lines_nm: dict[str, float] | list[float],
    width_nm: float = 2.0,
    amplitudes: list[float] | None = None,
    wavelengths: np.ndarray | None = None,
    noise: float = 0.0,
    seed: int | None = None,
) -> Spectrum:
    """Линейчатый спектр: набор гауссовых линий поверх слабого фона."""
    wl = default_wavelengths() if wavelengths is None else np.asarray(wavelengths, dtype=float)
    centres = list(lines_nm.values()) if isinstance(lines_nm, dict) else list(lines_nm)
    if amplitudes is None:
        amplitudes = [1.0] * len(centres)
    if len(amplitudes) != len(centres):
        raise ValueError("Количество амплитуд и линий не совпадает")

    values = np.full_like(wl, 0.01, dtype=float)
    for centre, amplitude in zip(centres, amplitudes):
        values += amplitude * np.exp(-0.5 * ((wl - centre) / width_nm) ** 2)

    rng = np.random.default_rng(seed)
    if noise > 0:
        values = values + rng.normal(0.0, float(noise), values.shape)
    values = np.clip(values, 0.0, None)
    return Spectrum(wl, values, label="Линейчатый спектр")


def mercury_spectrum(
    wavelengths: np.ndarray | None = None, noise: float = 0.0, seed: int | None = None
) -> Spectrum:
    """Спектр ртутной лампы (линии 404.7, 435.8, 546.1, 577.0, 579.1, 611.6 нм)."""
    lines = {
        "404.66": 404.66,
        "435.83": 435.83,
        "546.07": 546.07,
        "576.96": 576.96,
        "579.07": 579.07,
        "611.60": 611.60,
    }
    amplitudes = [0.35, 0.75, 1.0, 0.55, 0.45, 0.18]
    return line_spectrum(
        lines,
        width_nm=1.2,
        amplitudes=amplitudes,
        wavelengths=wavelengths,
        noise=noise,
        seed=seed,
    )


def sodium_spectrum(
    wavelengths: np.ndarray | None = None, noise: float = 0.0, seed: int | None = None
) -> Spectrum:
    """Спектр натриевой лампы, дублет D 589.0/589.6 нм."""
    return line_spectrum(
        {"589.0": 589.0, "589.6": 589.59},
        width_nm=1.0,
        amplitudes=[1.0, 0.9],
        wavelengths=wavelengths,
        noise=noise,
        seed=seed,
    )


def hydrogen_spectrum(
    wavelengths: np.ndarray | None = None, noise: float = 0.0, seed: int | None = None
) -> Spectrum:
    """Спектр водорода: Hα 656.3, Hβ 486.1, Hγ 434.0 нм."""
    return line_spectrum(
        {"Hα": 656.28, "Hβ": 486.13, "Hγ": 434.05},
        width_nm=1.5,
        amplitudes=[0.8, 0.4, 0.3],
        wavelengths=wavelengths,
        noise=noise,
        seed=seed,
    )


def led_spectrum(
    centre_nm: float = 450.0,
    width_nm: float = 20.0,
    wavelengths: np.ndarray | None = None,
    noise: float = 0.0,
    seed: int | None = None,
) -> Spectrum:
    """Широкий спектр светодиода (гауссов профиль)."""
    wl = default_wavelengths() if wavelengths is None else np.asarray(wavelengths, dtype=float)
    values = np.exp(-0.5 * ((wl - centre_nm) / width_nm) ** 2)
    values += 0.05 * np.exp(-0.5 * ((wl - (centre_nm + 110.0)) / (width_nm * 2.0)) ** 2)
    rng = np.random.default_rng(seed)
    if noise > 0:
        values = values + rng.normal(0.0, float(noise), values.shape)
    values = np.clip(values, 0.0, None)
    return Spectrum(wl, values, label=f"Светодиод, {centre_nm:.0f} нм")


def sensor_frame(
    spectrum: int | float | np.ndarray,
    rows: int = 240,
    background: float = 6.0,
    noise: float = 2.0,
    band_height: float = 0.18,
    seed: int | None = None,
) -> np.ndarray:
    """Смоделировать кадр камеры с горизонтальной спектральной полосой.

    Возвращает серое изображение ``rows × N``: яркая полоса в центре,
    внутри полосы — значения спектра, снаружи — фон и шум сенсора.
    """
    if np.isscalar(spectrum):
        width = 640
        profile = np.full(width, float(spectrum), dtype=float)
    else:
        profile = np.asarray(spectrum, dtype=float).ravel()
        width = profile.size
    if width < 2:
        raise ValueError("Спектр слишком короткий")

    rng = np.random.default_rng(seed)
    frame = np.full((int(rows), width), float(background), dtype=float)

    half_band = max(2, int(rows * band_height / 2))
    centre = rows // 2
    top, bottom = max(0, centre - half_band), min(rows, centre + half_band)

    peak = float(np.max(profile)) if profile.size else 0.0
    scaled = profile / peak * 230.0 if peak > 0 else profile
    frame[top:bottom, :] = scaled[None, :]
    if noise > 0:
        frame = frame + rng.normal(0.0, float(noise), frame.shape)
    return np.clip(frame, 0.0, 255.0)
