"""Tests for finding the colour bar in an inspection export."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from zero_line_detection.colorbar import detect_colorbar


def scan_with_bar(width: int, height: int, bar_x0: int, bar_width: int, *, ticks: bool) -> np.ndarray:
    """A white export with a part blob and a hue ramp down the right margin."""
    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.ellipse(rgb, (width // 2, height // 2), (width // 3, height // 3), 0, 0, 360, (60, 190, 90), -1)
    hue = np.linspace(0, 150, height).astype(np.uint8)
    ramp = cv2.cvtColor(np.stack([hue, np.full(height, 255, np.uint8), np.full(height, 255, np.uint8)], axis=1)
                        .reshape(-1, 1, 3), cv2.COLOR_HSV2RGB)
    rgb[:, bar_x0:bar_x0 + bar_width] = ramp
    if ticks:
        # The export draws tick marks and numbers over the narrow bar itself.
        for y in range(4, height - 4, max(6, height // 12)):
            rgb[y:y + 1, bar_x0:bar_x0 + bar_width] = (30, 30, 30)
            cv2.putText(rgb, "2.00", (bar_x0 + 1, y + 4), cv2.FONT_HERSHEY_PLAIN, 0.3, (20, 20, 20), 1)
    return rgb


class ColorbarDetectionTests(unittest.TestCase):
    def test_plain_bar_is_found(self) -> None:
        rgb = scan_with_bar(600, 400, 570, 14, ticks=False)
        info = detect_colorbar(rgb).info
        self.assertEqual(info.side, "right")
        self.assertLessEqual(abs(info.x0 - 570), 2)

    def test_narrow_bar_with_ticks_and_numbers_printed_on_it_is_still_found(self) -> None:
        """Measured exports draw the scale over the bar: rows are no longer one colour.

        The mean row spread then reaches ~24, over the old limit of 18, and such scans failed with
        "컬러바를 찾지 못했습니다" although the bar is plainly there (7 real scans: 3 of them).
        """
        rgb = scan_with_bar(600, 400, 570, 8, ticks=True)
        info = detect_colorbar(rgb).info
        self.assertEqual(info.side, "right")
        self.assertLessEqual(abs(info.x0 - 570), 2)

    def test_a_scan_without_a_bar_is_refused(self) -> None:
        rgb = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.ellipse(rgb, (300, 200), (200, 130), 0, 0, 360, (60, 190, 90), -1)
        with self.assertRaisesRegex(RuntimeError, "컬러바를 찾지 못했습니다"):
            detect_colorbar(rgb)


if __name__ == "__main__":
    unittest.main()
