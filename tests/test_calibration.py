# -*- coding: utf-8 -*-
"""Тесты калибровки."""

import unittest

import numpy as np

from _helpers import artifact
from spektrolab.core.calibration import (
    Calibration,
    calibrate_from_lines,
    linear_calibration,
)


class CalibrationTests(unittest.TestCase):
    def test_quadratic_calibration_is_exact(self):
        pixels = np.array([100.0, 300.0, 600.0, 900.0])
        wavelengths = 340.0 + 0.68 * pixels + 1.2e-4 * pixels**2
        calibration = calibrate_from_lines(
            pixels, wavelengths, degree=2, pixel_count=1024
        )
        self.assertEqual(calibration.degree, 2)
        self.assertLess(calibration.rms_nm, 1e-9)
        grid = calibration.wavelengths()
        self.assertEqual(grid.size, 1024)
        self.assertTrue(np.all(np.diff(grid) > 0))

    def test_pixel_lookup_roundtrip(self):
        pixels = np.array([0.0, 400.0, 1023.0])
        wavelengths = np.array([340.0, 700.0, 1050.0])
        calibration = calibrate_from_lines(pixels, wavelengths, degree=2, pixel_count=1024)
        for pixel in (0.0, 250.5, 700.0, 1023.0):
            wavelength = float(
                np.polynomial.polynomial.polyval(pixel, calibration.coefficients)
            )
            self.assertAlmostEqual(calibration.pixel_of(wavelength), pixel, delta=0.01)

    def test_save_and_load_roundtrip(self):
        pixels = [10.0, 500.0, 990.0]
        wavelengths = [350.0, 700.0, 1040.0]
        calibration = calibrate_from_lines(pixels, wavelengths, degree=1, pixel_count=1024)
        path = artifact("calibration.json")
        calibration.save(path)
        loaded = Calibration.load(path)
        self.assertEqual(loaded.coefficients, calibration.coefficients)
        self.assertEqual(loaded.pixel_count, calibration.pixel_count)
        self.assertEqual(len(loaded.points), 3)

    def test_linear_calibration(self):
        calibration = linear_calibration(1024, 340.0, 1050.0)
        values = calibration.wavelengths()
        self.assertAlmostEqual(values[0], 340.0, places=9)
        self.assertAlmostEqual(values[-1], 1050.0, places=9)

    def test_out_of_range_lookup(self):
        calibration = linear_calibration(1024, 340.0, 1050.0)
        with self.assertRaises(ValueError):
            calibration.pixel_of(200.0)

    def test_not_enough_points(self):
        with self.assertRaises(ValueError):
            calibrate_from_lines([10.0, 20.0], [500.0, 600.0], degree=2)

    def test_non_monotonic_wavelengths_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_from_lines([10.0, 20.0, 30.0], [500.0, 400.0, 450.0], degree=1)


if __name__ == "__main__":
    unittest.main()
