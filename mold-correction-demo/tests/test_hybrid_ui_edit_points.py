"""Tests for editable zero-line contour output."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from zero_line_detection.hybrid_ui import _mask_contours_as_lines


class HybridUiEditPointTests(unittest.TestCase):
    def test_case1_contour_is_closed_and_handle_count_is_bounded(self) -> None:
        mask = np.zeros((420, 420), dtype=np.uint8)
        angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        radii = np.where(np.arange(len(angles)) % 2 == 0, 150.0, 135.0)
        points = np.column_stack((
            210 + np.cos(angles) * radii,
            210 + np.sin(angles) * radii,
        )).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)

        lines = _mask_contours_as_lines(mask)

        self.assertEqual(len(lines), 1)
        editable = lines[0]["points"]
        self.assertEqual(editable[0], editable[-1])
        self.assertLessEqual(len(editable) - 1, 32)
        self.assertGreaterEqual(len(editable) - 1, 3)


if __name__ == "__main__":
    unittest.main()
