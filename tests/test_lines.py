# -*- coding: utf-8 -*-
"""Тесты табличных линий и распознавания."""

import unittest

from spektrolab.core.lines import (
    CALIBRATION_SOURCES,
    REFERENCE_LINES,
    identify,
    lines_for,
    nearest_line,
)


class LineTests(unittest.TestCase):
    def test_lines_are_sorted(self):
        lines = lines_for(("Hg", "Ne"))
        self.assertGreater(len(lines), 10)
        wavelengths = [line[1] for line in lines]
        self.assertEqual(wavelengths, sorted(wavelengths))

    def test_identify_mercury(self):
        matches = identify([404.7, 435.9, 546.2, 700.0], elements="Hg", tolerance_nm=2.0)
        self.assertEqual(len(matches), 3)
        self.assertTrue(all(match["element"] == "Hg" for match in matches))
        self.assertLess(abs(matches[0]["error_nm"]), 0.5)

    def test_identify_respects_tolerance(self):
        self.assertEqual(identify([500.0], elements="Hg", tolerance_nm=1.0), [])
        self.assertEqual(len(identify([500.0], elements="Hg", tolerance_nm=5.0)), 0)

    def test_nearest_line(self):
        element, reference, error = nearest_line(546.0, elements="Hg")
        self.assertEqual(element, "Hg")
        self.assertAlmostEqual(reference, 546.07, places=2)
        self.assertAlmostEqual(error, -0.07, places=2)

    def test_calibration_sources_are_known(self):
        for source, elements in CALIBRATION_SOURCES.items():
            for element in elements:
                self.assertIn(element, REFERENCE_LINES, source)

    def test_unknown_element_gives_empty(self):
        self.assertEqual(lines_for("Xx"), [])


if __name__ == "__main__":
    unittest.main()
