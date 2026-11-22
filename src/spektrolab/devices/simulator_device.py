"""Виртуальная камера: те же кадры, что и у реального прибора, но без железа."""

from __future__ import annotations

import numpy as np

from ..core.simulator import (
    blackbody_spectrum,
    hydrogen_spectrum,
    led_spectrum,
    mercury_spectrum,
    sensor_frame,
    sodium_spectrum,
)
from ..core.spectrum import Spectrum
from .base import Camera

__all__ = ["SimulatorCamera", "SIMULATOR_SOURCES"]

SIMULATOR_SOURCES = (
    "Лампа накаливания",
    "Ртутная лампа",
    "Натриевая лампа",
    "Водородная лампа",
    "Светодиод",
)


class SimulatorCamera(Camera):
    """Симулятор USB-спектрометра: кадр с яркой спектральной полосой.

    Позволяет запускать приложение и снимать протоколы без камеры и лампы.
    Источник и температуру можно менять на лету.
    """

    def __init__(
        self,
        source: str = "Лампа накаливания",
        temperature: float = 3000.0,
        noise: float = 0.01,
        rows: int = 240,
        seed: int | None = 42,
    ):
        self.source = source
        self.temperature = float(temperature)
        self.noise = float(noise)
        self.rows = int(rows)
        self.seed = seed
        self._open = False
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"Симулятор · {self.source}"

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def resolution(self) -> tuple[int, int]:
        return (1024, self.rows)

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def set_source(self, source: str) -> None:
        if source not in SIMULATOR_SOURCES:
            raise ValueError(f"Неизвестный источник: {source}")
        self.source = source

    def set_temperature(self, temperature: float) -> None:
        self.temperature = float(temperature)

    def spectrum(self) -> Spectrum:
        """Спектр, который «видит» симулятор в данный момент."""
        if self.source == "Ртутная лампа":
            return mercury_spectrum()
        if self.source == "Натриевая лампа":
            return sodium_spectrum()
        if self.source == "Водородная лампа":
            return hydrogen_spectrum()
        if self.source == "Светодиод":
            return led_spectrum(centre_nm=450.0, width_nm=18.0)
        return blackbody_spectrum(temperature=self.temperature)

    def read(self) -> np.ndarray:
        if not self._open:
            raise RuntimeError("Симулятор не открыт")
        spectrum = self.spectrum()
        frame = sensor_frame(
            spectrum.intensity,
            rows=self.rows,
            background=6.0,
            noise=self.noise * 255.0,
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )
        return np.stack([frame, frame, frame], axis=-1).astype(np.uint8)
