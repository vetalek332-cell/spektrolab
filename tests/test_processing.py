# -*- coding: utf-8 -*-
"""Тесты обработки кадра и спектра."""

import unittest

import numpy as np

from spektrolab.core.processing import (
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
from spektrolab.core.simulator import blackbody_spectrum, sensor_frame


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.spectrum = blackbody_spectrum(temperature=3000.0, noise=0.0)
        self.frame = sensor_frame(self.spectrum.intensity, rows=200, noise=0.0, seed=1)

    def test_grayscale_from_bgr(self):
        bgr = np.dstack([self.frame, self.frame, self.frame]).astype(np.uint8)
        gray = to_grayscale(bgr)
        self.assertEqual(gray.shape, self.frame.shape)
        self.assertTrue(np.allclose(gray, self.frame, atol=1.0))

    def test_auto_roi_finds_band(self):
        roi = auto_roi(self.frame)
        centre = roi["y"] + roi["height"] / 2
        self.assertAlmostEqual(centre, self.frame.shape[0] / 2, delta=15)
        self.assertGreater(roi["height"], 20)

    def test_extract_from_roi_correlates_with_spectrum(self):
        roi = auto_roi(self.frame)
        extracted = extract_spectrum(
            self.frame, (roi["x"], roi["y"], roi["width"], roi["height"])
        )
        self.assertEqual(extracted.size, self.spectrum.size)
        correlation = np.corrcoef(
            normalize(extracted), normalize(self.spectrum.intensity)
        )[0, 1]
        self.assertGreater(correlation, 0.99)

    def test_dark_correction(self):
        measured = np.array([10.0, 20.0, 30.0])
        dark = np.array([1.0, 2.0, 3.0])
        corrected = correct_dark(measured, dark)
        self.assertTrue(np.allclose(corrected, [9.0, 18.0, 27.0]))

    def test_dark_correction_shape_mismatch(self):
        with self.assertRaises(ValueError):
            correct_dark(np.ones(3), np.ones(4))

    def test_reflectance(self):
        dark = np.zeros(3)
        reference = np.array([100.0, 100.0, 100.0])
        measured = np.array([50.0, 75.0, 100.0])
        result = reflectance(measured, dark, reference)
        self.assertTrue(np.allclose(result, [0.5, 0.75, 1.0]))

    def test_moving_average_reduces_noise(self):
        rng = np.random.default_rng(0)
        noisy = rng.normal(0.0, 1.0, 500)
        smoothed = moving_average(noisy, 9)
        self.assertEqual(smoothed.shape, noisy.shape)
        self.assertLess(float(np.std(smoothed)), float(np.std(noisy)))

    def test_remove_baseline(self):
        x = np.linspace(0.0, 1.0, 200)
        trend = 2.0 + 3.0 * x - 4.0 * x**2
        residual = remove_baseline(trend, degree=3)
        self.assertLess(float(np.max(np.abs(residual))), 1e-9)

    def test_resample_identity(self):
        grid = np.linspace(400, 500, 101)
        values = np.sin(grid)
        self.assertTrue(np.allclose(resample(grid, values, grid), values))


if __name__ == "__main__":
    unittest.main()
