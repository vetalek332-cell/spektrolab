"""Поиск и описание спектральных линий (реализация без SciPy)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .spectrum import Spectrum

__all__ = ["find_peaks", "peak_shape", "detect_peaks", "peak_prominence"]


def peak_prominence(values: np.ndarray, index: int) -> float:
    """Высота пика над ближайшей седловиной."""
    peak = values[index]
    left_min = peak
    cursor = index - 1
    while cursor >= 0 and values[cursor] <= peak:
        left_min = min(left_min, values[cursor])
        cursor -= 1

    right_min = peak
    cursor = index + 1
    while cursor < values.size and values[cursor] <= peak:
        right_min = min(right_min, values[cursor])
        cursor += 1
    return float(peak - max(left_min, right_min))


def find_peaks(
    values: Iterable[float],
    height: float | None = None,
    prominence: float | None = None,
    distance: int | None = None,
) -> np.ndarray:
    """Индексы локальных максимумов (аналог ``scipy.signal.find_peaks``).

    distance — минимальное расстояние между пиками в отсчётах; при конфликте
    остаётся более высокий пик.
    """
    y = np.asarray(values, dtype=float).ravel()
    if y.size < 3:
        return np.array([], dtype=int)

    left = y[:-2] < y[1:-1]
    right = y[1:-1] >= y[2:]
    candidates = np.flatnonzero(left & right) + 1
    if candidates.size == 0:
        return np.array([], dtype=int)

    if height is not None:
        candidates = candidates[y[candidates] >= float(height)]
    if prominence is not None:
        candidates = np.array(
            [index for index in candidates if peak_prominence(y, int(index)) >= prominence],
            dtype=int,
        )

    if distance is not None and candidates.size > 1:
        distance = int(distance)
        accepted: list[int] = []
        for index in candidates[np.argsort(y[candidates])[::-1]]:
            if all(abs(int(index) - other) >= distance for other in accepted):
                accepted.append(int(index))
        candidates = np.array(sorted(accepted), dtype=int)

    return candidates


def peak_shape(wavelengths, intensity, index: int) -> dict[str, float]:
    """Параметры линии: центр, высота, FWHM, площадь."""
    wl = np.asarray(wavelengths, dtype=float)
    y = np.asarray(intensity, dtype=float)
    index = int(index)
    if not 0 <= index < y.size:
        raise IndexError("Индекс пика вне массива")

    step = float(np.median(np.diff(wl))) if wl.size > 1 else 1.0
    center = float(wl[index])
    height = float(y[index])

    if 0 < index < y.size - 1:
        denominator = y[index - 1] - 2 * y[index] + y[index + 1]
        if denominator != 0:
            shift = 0.5 * (y[index - 1] - y[index + 1]) / denominator
            center = float(wl[index] + shift * step)
            height = float(y[index] - 0.25 * (y[index - 1] - y[index + 1]) * shift)

    fwhm = float("nan")
    area = float("nan")
    half = height / 2.0
    if height > 0:
        left_cross = None
        cursor = index
        while cursor > 0 and y[cursor] > half:
            cursor -= 1
        if y[cursor] <= half < y[cursor + 1]:
            fraction = (half - y[cursor]) / (y[cursor + 1] - y[cursor])
            left_cross = wl[cursor] + fraction * (wl[cursor + 1] - wl[cursor])

        right_cross = None
        cursor = index
        while cursor < y.size - 1 and y[cursor] > half:
            cursor += 1
        if y[cursor] <= half < y[cursor - 1]:
            fraction = (half - y[cursor]) / (y[cursor - 1] - y[cursor])
            right_cross = wl[cursor] + fraction * (wl[cursor - 1] - wl[cursor])

        if left_cross is not None and right_cross is not None:
            fwhm = float(abs(right_cross - left_cross))
            left_index = int(np.searchsorted(wl, min(left_cross, right_cross)))
            right_index = int(np.searchsorted(wl, max(left_cross, right_cross)))
            left_index = max(0, left_index - 1)
            right_index = min(y.size, right_index + 1)
            area = float(np.trapezoid(y[left_index:right_index], wl[left_index:right_index]))

    return {
        "index": index,
        "wavelength_nm": center,
        "intensity": height,
        "fwhm_nm": fwhm,
        "area": area,
        "prominence": peak_prominence(y, index),
    }


def detect_peaks(
    spectrum: Spectrum,
    height: float | None = None,
    prominence: float | None = None,
    min_distance_nm: float | None = None,
) -> list[dict[str, float]]:
    """Найти линии в спектре и вернуть их параметры, отсортированные по λ."""
    step = spectrum.step_nm
    distance = None
    if min_distance_nm is not None and step > 0:
        distance = max(1, int(round(min_distance_nm / step)))

    indices = find_peaks(
        spectrum.intensity, height=height, prominence=prominence, distance=distance
    )
    peaks = [peak_shape(spectrum.wavelengths, spectrum.intensity, int(i)) for i in indices]
    return sorted(peaks, key=lambda item: item["wavelength_nm"])
