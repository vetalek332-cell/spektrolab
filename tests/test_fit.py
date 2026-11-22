# -*- coding: utf-8 -*-
"""Тесты подгонки по закону Планка."""

import unittest

import numpy as np

from spektrolab.core.fit import fit_planck
from spektrolab.core.simulator import blackbody_spectrum


class PlanckFitTests(unittest.TestCase):
    def test_recovers_temperature_with_noise(self):
        for target in (2200.0, 3000.0, 4800.0, 6500.0):
            spectrum = blackbody_spectrum(temperature=target, noise=0.004, seed=11)
            fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
            self.assertTrue(fit.success)
            relative_error = abs(fit.temperature - target) / target
            self.assertLess(relative_error, 0.01, f"{target} К: {fit.temperature:.1f} К")
            self.assertGreater(fit.r_squared, 0.99)
            self.assertGreater(fit.temperature_err, 0.0)

    def test_noise_free_fit_is_accurate(self):
        spectrum = blackbody_spectrum(temperature=3200.0, noise=0.0)
        fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
        self.assertAlmostEqual(fit.temperature, 3200.0, delta=0.5)
        self.assertGreater(fit.r_squared, 0.999999)

    def test_offset_is_separated(self):
        spectrum = blackbody_spectrum(temperature=2700.0, noise=0.0, offset=0.2)
        fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
        self.assertAlmostEqual(fit.temperature, 2700.0, delta=1.0)
        self.assertGreater(fit.offset, 0.0)

    def test_constant_spectrum_is_rejected(self):
        wavelengths = np.linspace(400, 900, 100)
        fit = fit_planck(wavelengths, np.full_like(wavelengths, 0.5))
        self.assertFalse(fit.success)
        self.assertEqual(fit.temperature, 0.0)

    def test_short_input_raises(self):
        with self.assertRaises(ValueError):
            fit_planck(np.linspace(400, 500, 5), np.ones(5))

    def test_wavelength_range_cropping(self):
        spectrum = blackbody_spectrum(temperature=3000.0, noise=0.002, seed=3)
        fit = fit_planck(
            spectrum.wavelengths,
            spectrum.intensity,
            wl_range=(500.0, 800.0),
        )
        self.assertEqual(fit.n_points, int(np.count_nonzero(
            (spectrum.wavelengths >= 500.0) & (spectrum.wavelengths <= 800.0)
        )))
        self.assertEqual(fit.t_range, (700.0, 12000.0))

    def test_model_restores_shape(self):
        spectrum = blackbody_spectrum(temperature=3500.0, noise=0.0)
        fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
        model = fit.model(spectrum.wavelengths)
        self.assertEqual(model.shape, spectrum.intensity.shape)
        self.assertTrue(np.all(np.isfinite(model)))


if __name__ == "__main__":
    unittest.main()
