"""Абстракция камеры: реальная USB-камера или симулятор."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Camera", "CameraError"]


class CameraError(RuntimeError):
    """Ошибка работы с камерой."""


class Camera(ABC):
    """Интерфейс источника кадров.

    ``read()`` возвращает кадр в формате BGR (H×W×3, uint8) или серый (H×W),
    как обычная камера OpenCV.
    """

    name = "камера"

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def read(self) -> np.ndarray:
        ...

    @property
    def resolution(self) -> tuple[int, int] | None:
        return None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
