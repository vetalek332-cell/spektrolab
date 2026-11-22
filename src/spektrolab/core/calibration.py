"""Калибровка: пиксели сенсора → длины волн."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "Calibration",
    "calibrate_from_lines",
    "linear_calibration",
]


@dataclass
class Calibration:
    """Полиномиальная калибровка ``λ(p) = c0 + c1·p + c2·p² + …``."""

    coefficients: tuple[float, ...]
    pixel_count: int = 1280
    rms_nm: float = 0.0
    max_error_nm: float = 0.0
    points: list[dict[str, float]] = field(default_factory=list)
    device: str = ""
    note: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        self.coefficients = tuple(float(value) for value in self.coefficients)
        if len(self.coefficients) < 2:
            raise ValueError("Нужен хотя бы линейный полином (2 коэффициента)")
        self.pixel_count = int(self.pixel_count)
        if self.pixel_count < 2:
            raise ValueError("pixel_count должен быть не меньше 2")
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    @property
    def wavelength_range(self) -> tuple[float, float]:
        values = self.wavelengths()
        return float(values[0]), float(values[-1])

    def wavelengths(self, pixel_count: int | None = None) -> np.ndarray:
        """Длины волн для всех пикселей сенсора."""
        count = int(pixel_count or self.pixel_count)
        pixels = np.arange(count, dtype=float)
        return np.polynomial.polynomial.polyval(pixels, self.coefficients)

    def pixel_of(self, wavelength_nm: float) -> float:
        """Обратное преобразование: длина волны → (дробный) пиксель."""
        values = self.wavelengths()
        low, high = float(values[0]), float(values[-1])
        if wavelength_nm < min(low, high) or wavelength_nm > max(low, high):
            raise ValueError(
                f"{wavelength_nm} нм вне калиброванного диапазона "
                f"{min(low, high):.1f}–{max(low, high):.1f} нм"
            )
        pixels = np.arange(values.size, dtype=float)
        if values[0] <= values[-1]:
            return float(np.interp(wavelength_nm, values, pixels))
        return float(np.interp(wavelength_nm, values[::-1], pixels[::-1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "spektrolab.calibration",
            "version": 1,
            "coefficients": list(self.coefficients),
            "pixel_count": self.pixel_count,
            "rms_nm": self.rms_nm,
            "max_error_nm": self.max_error_nm,
            "points": self.points,
            "device": self.device,
            "note": self.note,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Calibration":
        if payload.get("format") not in (None, "spektrolab.calibration"):
            raise ValueError("Это не файл калибровки SpektroLab")
        return cls(
            coefficients=tuple(payload["coefficients"]),
            pixel_count=int(payload.get("pixel_count", 1280)),
            rms_nm=float(payload.get("rms_nm", 0.0)),
            max_error_nm=float(payload.get("max_error_nm", 0.0)),
            points=list(payload.get("points", [])),
            device=str(payload.get("device", "")),
            note=str(payload.get("note", "")),
            created_at=str(payload.get("created_at", "")),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def calibrate_from_lines(
    pixel_positions: Sequence[float],
    known_wavelengths_nm: Sequence[float],
    degree: int = 2,
    pixel_count: int | None = None,
    device: str = "",
    note: str = "",
) -> Calibration:
    """Построить калибровку по эталонным линиям.

    Parameters
    ----------
    pixel_positions:
        Положения известных линий на сенсоре (дробные пиксели).
    known_wavelengths_nm:
        Их табличные длины волн.
    degree:
        Степень полинома: 1 — линейная, 2 — квадратичная (рекомендуется).
    """
    pixels = np.asarray(pixel_positions, dtype=float).ravel()
    wavelengths = np.asarray(known_wavelengths_nm, dtype=float).ravel()
    if pixels.size != wavelengths.size:
        raise ValueError("Количество пикселей и длин волн не совпадает")
    if pixels.size < degree + 1:
        raise ValueError(f"Для степени {degree} нужно минимум {degree + 1} линий")
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(wavelengths)):
        raise ValueError("В точках калибровки есть NaN")

    order = np.argsort(pixels)
    pixels, wavelengths = pixels[order], wavelengths[order]
    if np.any(np.diff(pixels) <= 0):
        raise ValueError("Позиции пикселей должны быть уникальными")
    if not np.all(np.diff(wavelengths) > 0) and not np.all(np.diff(wavelengths) < 0):
        raise ValueError("Длины волн должны монотонно расти или убывать вместе с пикселями")

    coefficients = np.polynomial.polynomial.polyfit(pixels, wavelengths, int(degree))
    residual = np.polynomial.polynomial.polyval(pixels, coefficients) - wavelengths
    rms = float(np.sqrt(np.mean(residual**2)))
    max_error = float(np.max(np.abs(residual)))

    if pixel_count is None:
        pixel_count = int(np.ceil(float(pixels[-1]))) + 1

    points = [
        {
            "pixel": float(pixel),
            "wavelength_nm": float(wavelength),
            "residual_nm": float(error),
        }
        for pixel, wavelength, error in zip(pixels, wavelengths, residual)
    ]

    return Calibration(
        coefficients=tuple(float(value) for value in coefficients),
        pixel_count=int(pixel_count),
        rms_nm=rms,
        max_error_nm=max_error,
        points=points,
        device=device,
        note=note,
    )


def linear_calibration(
    pixel_count: int,
    wavelength_start_nm: float,
    wavelength_end_nm: float,
    device: str = "",
) -> Calibration:
    """Простая линейная калибровка «по паспорту» — до точной калибровки по линиям."""
    if wavelength_end_nm == wavelength_start_nm:
        raise ValueError("Начальная и конечная длины волн совпадают")
    coefficients = (float(wavelength_start_nm), (float(wavelength_end_nm) - float(wavelength_start_nm)) / (pixel_count - 1))
    return Calibration(
        coefficients=coefficients,
        pixel_count=int(pixel_count),
        device=device,
        note="Линейная калибровка по паспорту, требует уточнения по эталонным линиям",
    )

