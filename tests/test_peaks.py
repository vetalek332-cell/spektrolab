# -*- coding: utf-8 -*-
"""Тесты поиска линий."""

import unittest

import numpy as np

from spektrolab.core.peaks import detect_peaks, find_peaks, peak_shape
from spektrolab.core.simulator import line_spectrum, mercury_spectrum


class PeakTests(unittest.TestCase):
    def test_finds_simple_peaks(self):
        values = np.array([0, 0, 1, 0, 0, 0, 3, 0, 0, 2, 0], dtype=float)
        indices = find_peaks(values, prominence=0.5)
        self.assertEqual(indices.tolist(), [2, 6, 9])

    def test_distance_filter_keeps_highest(self):
        values = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        indices = find_peaks(values, distance=3)
        self.assertEqual(indices.tolist(), [1, 5])

    def test_gaussian_triplet(self):
        wavelengths = np.linspace(400, 700, 3001)
        spectrum = line_spectrum(
            {"a": 450.0, "b": 550.0, "c": 650.0},
            width_nm=1.0,
            wavelengths=wavelengths,
        )
        peaks = detect_peaks(spectrum, prominence=0.2)
        self.assertEqual(len(peaks), 3)
        centres = [peak["wavelength_nm"] for peak in peaks]
        for expected, measured in zip((450.0, 550.0, 650.0), centres):
            self.assertAlmostEqual(measured, expected, delta=0.3)

    def test_fwhm_of_gaussian(self):
        wavelengths = np.linspace(500, 600, 4001)
        sigma = 1.0
        values = np.exp(-0.5 * ((wavelengths - 550.0) / sigma) ** 2)
        index = int(np.argmax(values))
        shape = peak_shape(wavelengths, values, index)
        expected = 2.3548 * sigma
        self.assertAlmostEqual(shape["fwhm_nm"], expected, delta=0.05)
        self.assertAlmostEqual(shape["wavelength_nm"], 550.0, delta=1e-6)

    def test_mercury_lines_detected(self):
        spectrum = mercury_spectrum(noise=0.002, seed=5)
        peaks = detect_peaks(spectrum, prominence=0.06)
        wavelengths = [peak["wavelength_nm"] for peak in peaks]
        self.assertTrue(any(abs(value - 546.07) < 0.6 for value in wavelengths))
        self.assertTrue(any(abs(value - 435.83) < 0.8 for value in wavelengths))
        self.assertGreaterEqual(len(wavelengths), 4)

    def test_empty_signal(self):
        self.assertEqual(find_peaks(np.zeros(10)).tolist(), [])


if __name__ == "__main__":
    unittest.main()
