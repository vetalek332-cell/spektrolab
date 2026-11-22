# -*- coding: utf-8 -*-
"""Тесты командной строки."""

import contextlib
import io
import unittest

from _helpers import artifact
from spektrolab.cli import main


class CliTests(unittest.TestCase):
    def _run(self, arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main([str(item) for item in arguments])
        return code, output.getvalue()

    def test_selftest(self):
        code, output = self._run(["selftest"])
        self.assertEqual(code, 0, output)
        self.assertIn("Все проверки пройдены", output)

    def test_simulate_and_analyze(self):
        csv_path = artifact("spectrum.csv")
        json_path = artifact("measurement.json")
        report_path = artifact("report.md")

        code, output = self._run(
            ["simulate", "--temperature", "3000", "--csv", csv_path, "--json", json_path]
        )
        self.assertEqual(code, 0, output)
        self.assertTrue(csv_path.exists())
        self.assertTrue(json_path.exists())

        code, output = self._run(
            ["analyze", csv_path, "--peaks", "--element", "Hg", "--report", report_path]
        )
        self.assertEqual(code, 0, output)
        self.assertTrue(report_path.exists())
        self.assertIn("Температура", output)
        self.assertIn("R2", output)

    def test_calibrate_and_info(self):
        path = artifact("calibration.json")
        code, output = self._run(
            ["calibrate", "--pixels", "10", "500", "990",
             "--wavelengths", "350", "700", "1040", "--out", path]
        )
        self.assertEqual(code, 0, output)
        self.assertTrue(path.exists())

        code, output = self._run(["calibration-info", path])
        self.assertEqual(code, 0, output)
        self.assertIn("RMS", output)

    def test_linear_calibration(self):
        path = artifact("linear.json")
        code, output = self._run(
            ["linear-calibration", "--out", path, "--wl-min", "340", "--wl-max", "1050"]
        )
        self.assertEqual(code, 0, output)
        self.assertTrue(path.exists())

    def test_invalid_file_returns_error(self):
        code, output = self._run(["analyze", "no_such_file.csv"])
        self.assertEqual(code, 1)
        self.assertIn("Ошибка", output)

    def test_bad_pixel_wavelength_count(self):
        code, output = self._run(
            ["calibrate", "--pixels", "1", "2", "--wavelengths", "500"]
        )
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
