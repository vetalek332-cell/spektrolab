"""Реальная USB-камера через OpenCV."""

from __future__ import annotations

import os

import numpy as np

from .base import Camera, CameraError

__all__ = ["OpenCVCamera", "list_cameras"]


def _import_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - зависит от окружения
        raise CameraError(
            "Не установлен opencv-python. Установите: pip install opencv-python"
        ) from error
    return cv2


class OpenCVCamera(Camera):
    """USB-камера (или встроенная веб-камера), открытая через OpenCV."""

    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080):
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self._capture = None
        self._resolution: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return f"USB-камера #{self.index}"

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def resolution(self) -> tuple[int, int] | None:
        return self._resolution

    def open(self) -> None:
        if self.is_open:
            return
        cv2 = _import_cv2()
        backends = [cv2.CAP_ANY]
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

        capture = None
        for backend in backends:
            candidate = cv2.VideoCapture(self.index, backend)
            if candidate.isOpened():
                capture = candidate
                break
            candidate.release()
        if capture is None:
            raise CameraError(
                f"Не удалось открыть камеру #{self.index}. "
                "Проверьте подключение и разрешения."
            )

        if self.width and self.height:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (actual_width, actual_height)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._resolution = None

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Камера не открыта")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Не удалось прочитать кадр с камеры")
        return frame


def list_cameras(max_index: int = 5) -> list[int]:
    """Перебрать индексы камер и вернуть те, что открылись."""
    cv2 = _import_cv2()
    available: list[int] = []
    for index in range(int(max_index)):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                available.append(index)
        finally:
            capture.release()
    return available
