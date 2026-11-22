"""Ядро SpektroLab: физика, обработка, калибровка и протоколы."""

from .calibration import Calibration, calibrate_from_lines, linear_calibration
from .fit import PlanckFit, fit_planck, fit_planck_spectrum
from .lines import CALIBRATION_SOURCES, REFERENCE_LINES, identify, lines_for, nearest_line
from .peaks import detect_peaks, find_peaks, peak_shape
from .planck import (
    blackbody_rgb,
    measured_peak_nm,
    planck_normalized,
    planck_spectral_radiance,
    stefan_boltzmann,
    wien_peak_nm,
    wien_temperature,
)
from .processing import (
    auto_roi,
    correct_dark,
    extract_spectrum,
    moving_average,
    normalize,
    reflectance,
    remove_baseline,
    resample,
    to_grayscale,
)
from .protocol import Measurement
from .spectrum import Spectrum

__all__ = [
    "Spectrum",
    "PlanckFit",
    "fit_planck",
    "fit_planck_spectrum",
    "Calibration",
    "calibrate_from_lines",
    "linear_calibration",
    "identify",
    "nearest_line",
    "lines_for",
    "REFERENCE_LINES",
    "CALIBRATION_SOURCES",
    "detect_peaks",
    "find_peaks",
    "peak_shape",
    "planck_spectral_radiance",
    "planck_normalized",
    "wien_peak_nm",
    "wien_temperature",
    "stefan_boltzmann",
    "measured_peak_nm",
    "blackbody_rgb",
    "auto_roi",
    "correct_dark",
    "extract_spectrum",
    "moving_average",
    "normalize",
    "reflectance",
    "remove_baseline",
    "resample",
    "to_grayscale",
    "Measurement",
]
