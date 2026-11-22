# -*- coding: utf-8 -*-
"""Тесты протокола измерения."""

import unittest

from _helpers import artifact
from spektrolab.core.fit import fit_planck
from spektrolab.core.protocol import Measurement
from spektrolab.core.simulator import blackbody_spectrum, mercury_spectrum


class ProtocolTests(unittest.TestCase):
    def _measurement(self) -> Measurement:
        spectrum = blackbody_spectrum(temperature=3000.0, noise=0.002, seed=2)
        fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
        return Measurement(
            spectrum=spectrum,
            fit=fit,
            peaks=[
                {"wavelength_nm": 546.1, "intensity": 1.0, "fwhm_nm": 2.0},
                {"wavelength_nm": 435.8, "intensity": 0.5, "fwhm_nm": float("nan")},
            ],
            device="simulator",
            operator="тест",
        )

    def test_json_roundtrip(self):
        measurement = self._measurement()
        payload = measurement.to_dict()
        restored = Measurement.from_dict(payload)
        self.assertAlmostEqual(restored.fit.temperature, measurement.fit.temperature, places=2)
        self.assertEqual(restored.spectrum.size, measurement.spectrum.size)
        self.assertEqual(len(restored.peaks), 2)
        self.assertIsNone(restored.peaks[1]["fwhm_nm"])

    def test_save_and_load_json(self):
        measurement = self._measurement()
        path = artifact("measurement.json")
        measurement.save_json(path)
        loaded = Measurement.load_json(path)
        self.assertEqual(loaded.device, "simulator")
        self.assertEqual(loaded.operator, "тест")

    def test_csv_has_planck_column(self):
        measurement = self._measurement()
        path = artifact("spectrum.csv")
        measurement.save_csv(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "wavelength_nm;intensity;planck_fit")
        self.assertGreater(len(lines), 1000)

    def test_markdown_report(self):
        measurement = self._measurement()
        report = measurement.report_markdown()
        self.assertIn("Температура по закону Планка", report)
        self.assertIn("546.10", report)
        self.assertIn("R²", report)
        self.assertIn("Подпись оператора", report)

    def test_report_without_fit_or_peaks(self):
        spectrum = mercury_spectrum()
        measurement = Measurement(spectrum=spectrum)
        report = measurement.report_markdown()
        self.assertIn("Протокол измерения", report)
        self.assertNotIn("Температура по закону Планка", report)


if __name__ == "__main__":
    unittest.main()
