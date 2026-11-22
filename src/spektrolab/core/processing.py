"""Обработка кадра и одномерного спектра (без OpenCV и SciPy)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_grayscale",
    "extract_spectrum",
    "auto_roi",
    "correct_dark",
    "reflectance",
    "moving_average",
    "normalize",
    "remove_baseline",
    "resample",
]


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Привести кадр к двумерному серому изображению."""
    image = np.asarray(frame)
    if image.ndim == 2:
        return image.astype(float, copy=False)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels >= 3:
            b, g, r = image[..., 0], image[..., 1], image[..., 2]
            return (0.114 * b + 0.587 * g + 0.299 * r).astype(float)
        return image[..., 0].astype(float)
    raise ValueError(f"Неизвестная размерность кадра: {image.shape}")


def extract_spectrum(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    aggregate: str = "mean",
) -> np.ndarray:
    """Выделить одномерный спектр из кадра.

    ``roi`` — кортеж ``(x, y, width, height)``; если не задан, берётся весь
    кадр. Интенсивность спектра — среднее (или медиана) по вертикали.
    """
    gray = to_grayscale(frame)
    if roi is not None:
        x, y, width, height = (int(round(value)) for value in roi)
        x = max(x, 0)
        y = max(y, 0)
        width = max(width, 0)
        height = max(height, 0)
        if width == 0 or height == 0:
            raise ValueError("ROI имеет нулевой размер")
        patch = gray[y : y + height, x : x + width]
    else:
        patch = gray
    if patch.size == 0:
        raise ValueError("Пустой ROI")

    if aggregate == "mean":
        return patch.mean(axis=0)
    if aggregate == "median":
        return np.median(patch, axis=0)
    if aggregate == "max":
        return patch.max(axis=0)
    raise ValueError(f"Неизвестный способ усреднения: {aggregate}")


def auto_roi(frame: np.ndarray, threshold: float = 0.5, margin: int = 10) -> dict[str, int]:
    """Автоматически найти яркую горизонтальную полосу спектра в кадре."""
    gray = to_grayscale(frame)
    profile = gray.mean(axis=1)
    if profile.size == 0 or not np.isfinite(profile).any():
        raise ValueError("Не удалось построить профиль кадра")
    level = profile.max() * float(threshold)
    indices = np.flatnonzero(profile > level)
    if indices.size == 0:
        return {"x": 0, "y": 0, "width": int(gray.shape[1]), "height": int(gray.shape[0])}

    y_min = int(indices[0])
    y_max = int(indices[-1])
    y = max(0, y_min - margin)
    height = min(gray.shape[0] - y, (y_max - y_min) + 2 * margin)
    return {"x": 0, "y": y, "width": int(gray.shape[1]), "height": int(height)}


def _check_same_shape(reference: np.ndarray, other: np.ndarray, name: str) -> None:
    if reference.shape != other.shape:
        raise ValueError(
            f"Размер {name} ({other.shape}) не совпадает со спектром ({reference.shape})"
        )


def correct_dark(measured: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Вычесть тёмновой кадр (собственный шум сенсора)."""
    values = np.asarray(measured, dtype=float)
    dark_values = np.asarray(dark, dtype=float)
    _check_same_shape(values, dark_values, "тёмнового кадра")
    return values - dark_values


def reflectance(measured: np.ndarray, dark: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Относительная спектральная характеристика (S − D) / (R − D)."""
    values = np.asarray(measured, dtype=float)
    dark_values = np.asarray(dark, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    _check_same_shape(values, dark_values, "тёмнового кадра")
    _check_same_shape(values, reference_values, "опорного кадра")

    denominator = reference_values - dark_values
    safe = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
    return (values - dark_values) / safe


def moving_average(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Скользящее среднее с выравниванием по краям (без сдвига фазы)."""
    values = np.asarray(signal, dtype=float)
    window = int(window)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    if window > values.size:
        return values.copy()

    half = window // 2
    padded = np.pad(values, (half, half), mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def normalize(signal: np.ndarray) -> np.ndarray:
    """Нормировать сигнал на максимум."""
    values = np.asarray(signal, dtype=float)
    peak = float(np.max(values)) if values.size else 0.0
    return values / peak if peak > 0 else values.copy()


def remove_baseline(signal: np.ndarray, degree: int = 3) -> np.ndarray:
    """Вычесть полиномиальную базовую линию (для спектров поглощения)."""
    values = np.asarray(signal, dtype=float)
    if values.size < degree + 2:
        return values.copy()
    x = np.linspace(-1.0, 1.0, values.size)
    coefficients = np.polyfit(x, values, int(degree))
    return values - np.polyval(coefficients, x)


def resample(wavelengths, signal, new_wavelengths) -> np.ndarray:
    """Линейная интерполяция сигнала на новую сетку длин волн."""
    return np.interp(
        np.asarray(new_wavelengths, dtype=float),
        np.asarray(wavelengths, dtype=float),
        np.asarray(signal, dtype=float),
    )
