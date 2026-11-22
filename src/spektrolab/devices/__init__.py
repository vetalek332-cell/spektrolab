"""Источники кадров: реальные камеры и симулятор."""

from .base import Camera, CameraError
from .simulator_device import SIMULATOR_SOURCES, SimulatorCamera

__all__ = ["Camera", "CameraError", "OpenCVCamera", "SimulatorCamera", "SIMULATOR_SOURCES"]


def __getattr__(name: str):
    """Ленивый импорт OpenCVCamera, чтобы пакет работал без OpenCV."""
    if name == "OpenCVCamera":
        from .opencv_camera import OpenCVCamera

        return OpenCVCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
