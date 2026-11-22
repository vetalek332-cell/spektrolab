"""Спектр: длины волн + интенсивности и базовые операции над ним."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class Spectrum:
    """Одномерный спектр.

    Parameters
    ----------
    wavelengths:
        Длины волн в нанометрах. Могут быть в любом порядке — массив будет
        отсортирован по возрастанию.
    intensity:
        Интенсивность в произвольных единицах (АЦП, относительные единицы).
    label:
        Подпись для интерфейса и протокола.
    meta:
        Произвольные метаданные (устройство, экспозиция, оператор и т. п.).
    """

    wavelengths: np.ndarray
    intensity: np.ndarray
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wl = np.asarray(self.wavelengths, dtype=float).ravel()
        values = np.asarray(self.intensity, dtype=float).ravel()
        if wl.size != values.size:
            raise ValueError(
                f"Длина волн ({wl.size}) и интенсивность ({values.size}) не совпадают"
            )
        if wl.size < 2:
            raise ValueError("Для спектра нужно минимум две точки")
        if not np.all(np.isfinite(wl)):
            raise ValueError("В длинах волн есть NaN или бесконечность")
        if not np.all(np.isfinite(values)):
            raise ValueError("В интенсивности есть NaN или бесконечность")

        order = np.argsort(wl)
        wl = wl[order]
        values = values[order]
        if np.any(np.diff(wl) <= 0):
            raise ValueError("Длины волн должны быть уникальными")

        self.wavelengths = wl
        self.intensity = values

    # ------------------------------------------------------------------
    # свойства
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return int(self.wavelengths.size)

    @property
    def wavelength_range(self) -> tuple[float, float]:
        return float(self.wavelengths[0]), float(self.wavelengths[-1])

    @property
    def step_nm(self) -> float:
        return float(np.median(np.diff(self.wavelengths)))

    @property
    def is_uniform(self) -> bool:
        if self.size < 3:
            return True
        diffs = np.diff(self.wavelengths)
        return bool(np.allclose(diffs, diffs.mean(), rtol=1e-3, atol=1e-6))

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------
    # преобразования
    # ------------------------------------------------------------------
    def normalized(self) -> "Spectrum":
        """Спектр, нормированный на максимум."""
        peak = float(np.max(self.intensity))
        if peak <= 0:
            return Spectrum(self.wavelengths, self.intensity, self.label, dict(self.meta))
        return Spectrum(self.wavelengths, self.intensity / peak, self.label, dict(self.meta))

    def cropped(self, low_nm: float, high_nm: float) -> "Spectrum":
        """Часть спектра в диапазоне [low_nm, high_nm]."""
        if low_nm >= high_nm:
            raise ValueError("low_nm должен быть меньше high_nm")
        mask = (self.wavelengths >= low_nm) & (self.wavelengths <= high_nm)
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"В диапазоне {low_nm}–{high_nm} нм меньше двух точек"
            )
        return Spectrum(
            self.wavelengths[mask], self.intensity[mask], self.label, dict(self.meta)
        )

    def smoothed(self, window: int = 5) -> "Spectrum":
        """Скользящее среднее с сохранением длины массива."""
        from .processing import moving_average

        return Spectrum(
            self.wavelengths,
            moving_average(self.intensity, window),
            self.label,
            dict(self.meta),
        )

    def baseline_removed(self, degree: int = 3) -> "Spectrum":
        """Вычитание полиномиальной базовой линии."""
        from .processing import remove_baseline

        return Spectrum(
            self.wavelengths,
            remove_baseline(self.intensity, degree),
            self.label,
            dict(self.meta),
        )

    def resampled(self, wavelengths: np.ndarray) -> "Spectrum":
        """Пересэмплирование на новую сетку длин волн (линейная интерполяция)."""
        grid = np.asarray(wavelengths, dtype=float)
        values = np.interp(grid, self.wavelengths, self.intensity)
        return Spectrum(grid, values, self.label, dict(self.meta))

    def with_label(self, label: str) -> "Spectrum":
        return Spectrum(self.wavelengths, self.intensity, label, dict(self.meta))

    # ------------------------------------------------------------------
    # файлы
    # ------------------------------------------------------------------
    def to_rows(self) -> Iterable[tuple[float, float]]:
        return zip(self.wavelengths.tolist(), self.intensity.tolist())

    def save_csv(self, path: str | Path) -> Path:
        """Сохранить спектр в CSV: ``wavelength_nm;intensity``.

        Десятичная запятая не используется — файл читается и Excel, и pandas.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(["wavelength_nm", "intensity"])
            writer.writerows(self.to_rows())
        return target

    @classmethod
    def load_csv(cls, path: str | Path, label: str | None = None) -> "Spectrum":
        """Загрузить спектр из CSV с двумя колонками (запятая или точка с запятой)."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        lines = []
        with source.open("r", encoding="utf-8-sig", newline="") as fh:
            for raw in fh:
                line = raw.strip()
                if line and not line.startswith("#"):
                    lines.append(line)
        if not lines:
            raise ValueError(f"Файл {source} пуст")

        delimiter = ";" if lines[0].count(";") >= lines[0].count(",") else ","
        header = lines[0].lower()
        has_header = any(word in header for word in ("wavelength", "длина", "nm"))
        data_lines = lines[1:] if has_header else lines

        pairs: list[tuple[float, float]] = []
        for line in data_lines:
            parts = [part.strip() for part in line.replace(",", ".").split(delimiter)]
            if len(parts) < 2:
                continue
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if len(pairs) < 2:
            raise ValueError(f"В файле {source} не нашлось пар чисел")

        array = np.asarray(pairs, dtype=float)
        return cls(
            array[:, 0],
            array[:, 1],
            label=label or source.stem,
            meta={"source": str(source)},
        )
