# -*- coding: utf-8 -*-
"""Тесты физики теплового излучения."""

import unittest

import numpy as np

from spektrolab.core.planck import (
    blackbody_rgb,
    measured_peak_nm,
    planck_normalized,
    planck_spectral_radiance,
    stefan_boltzmann,
    wien_peak_nm,
    wien_temperature,
)


class PlanckTests(unittest.TestCase):
    def test_normalized_curve_peaks_at_one(self):
        wavelengths = np.linspace(340, 1050, 1024)
        curve = planck_normalized(wavelengths, 3000)
        self.assertAlmostEqual(float(curve.max()), 1.0, places=9)
        self.assertGreaterEqual(float(curve.min()), 0.0)

    def test_spectral_radiance_is_positive_and_finite(self):
        wavelengths = np.array([250.0, 340.0, 500.0, 1050.0, 2500.0])
        values = planck_spectral_radiance(wavelengths, 3000)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0))

    def test_wien_peak_at_3000_kelvin(self):
        self.assertAlmostEqual(wien_peak_nm(3000), 965.9, places=1)
        self.assertAlmostEqual(wien_temperature(965.9), 3000, delta=1.0)

    def test_wien_peak_moves_with_temperature(self):
        self.assertLess(wien_peak_nm(6000), wien_peak_nm(3000))

    def test_stefan_boltzmann(self):
        self.assertAlmostEqual(stefan_boltzmann(300), 5.670374419e-8 * 300**4, places=10)
        self.assertEqual(stefan_boltzmann(0), 0.0)

    def test_measured_peak_is_parabolic(self):
        wavelengths = np.array([400.0, 450.0, 500.0, 550.0, 600.0])
        intensity = np.array([0.1, 0.6, 1.0, 0.7, 0.2])
        self.assertAlmostEqual(measured_peak_nm(wavelengths, intensity), 500.0, delta=20)

    def test_blackbody_rgb_is_valid(self):
        for temperature in (1000, 3000, 6500, 20000):
            red, green, blue = blackbody_rgb(temperature)
            for channel in (red, green, blue):
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            planck_spectral_radiance([500.0], 0)
        with self.assertRaises(ValueError):
            wien_peak_nm(-10)


if __name__ == "__main__":
    unittest.main()
