# -*- coding: utf-8 -*-
"""Тесты симулятора и виртуальной камеры."""

import unittest

import numpy as np

from spektrolab.core.fit import fit_planck
from spektrolab.core.peaks import detect_peaks
from spektrolab.core.simulator import (
    blackbody_spectrum,
    hydrogen_spectrum,
    led_spectrum,
    mercury_spectrum,
    sensor_frame,
    sodium_spectrum,
)
from spektrolab.devices.simulator_device import SimulatorCamera


class SimulatorTests(unittest.TestCase):
    def test_blackbody_can_be_fitted(self):
        spectrum = blackbody_spectrum(temperature=3400.0, noise=0.003, seed=9)
        fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
        self.assertLess(abs(fit.temperature - 3400.0) / 3400.0, 0.01)

    def test_line_spectra_have_expected_lines(self):
        cases = {
            "mercury": (mercury_spectrum(noise=0.001), 546.07),
            "sodium": (sodium_spectrum(noise=0.001), 589.0),
            "hydrogen": (hydrogen_spectrum(noise=0.001), 656.28),
        }
        for name, (spectrum, expected) in cases.items():
            peaks = detect_peaks(spectrum, prominence=0.05)
            found = [peak["wavelength_nm"] for peak in peaks]
            self.assertTrue(
                any(abs(value - expected) < 1.0 for value in found),
                f"{name}: {found}",
            )

    def test_led_spectrum_is_broad(self):
        spectrum = led_spectrum(centre_nm=450.0, width_nm=20.0)
        half = spectrum.intensity.max() / 2
        width = np.count_nonzero(spectrum.intensity > half) * spectrum.step_nm
        self.assertGreater(width, 20.0)

    def test_sensor_frame_shape_and_range(self):
        profile = blackbody_spectrum().intensity
        frame = sensor_frame(profile, rows=120, noise=1.0, seed=1)
        self.assertEqual(frame.shape, (120, profile.size))
        self.assertGreaterEqual(float(frame.min()), 0.0)
        self.assertLessEqual(float(frame.max()), 255.0)

    def test_simulator_camera_frames(self):
        camera = SimulatorCamera(source="Ртутная лампа", rows=96)
        with camera:
            self.assertTrue(camera.is_open)
            frame = camera.read()
        self.assertEqual(frame.shape, (96, 1024, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertFalse(camera.is_open)

    def test_simulator_source_switching(self):
        camera = SimulatorCamera(source="Лампа накаливания", temperature=2500)
        self.assertIn("2500", camera.spectrum().label.replace("Чёрное тело, ", "2500"))
        camera.set_temperature(4000)
        self.assertIn("4000", camera.spectrum().label)


if __name__ == "__main__":
    unittest.main()
