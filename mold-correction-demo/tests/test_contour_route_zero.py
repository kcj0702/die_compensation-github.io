"""Tests for the outer-zero-point / HSV-correction zero-line algorithm."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from zero_line_detection.contour_route_zero import (
    build_hsv_correction_mask,
    construct_zero_lines,
    render_selection_preview,
    select_outer_zero_points,
)


class ContourRouteZeroTests(unittest.TestCase):
    def _plate(self):
        image = np.full((140, 220, 3), 255, dtype=np.uint8)
        part = np.zeros(image.shape[:2], dtype=bool)
        part[20:121, 20:201] = True
        image[part] = (0, 210, 50)  # RGB green
        values = np.full(part.shape, 0.6, dtype=np.float32)
        return image, part, values

    def test_non_green_material_is_correction(self) -> None:
        image, part, _ = self._plate()
        image[45:85, 80:135] = (235, 30, 25)

        correction = build_hsv_correction_mask(image, part)

        self.assertTrue(correction[60, 100])
        self.assertFalse(correction[30, 30])
        self.assertFalse(correction[5, 5])

    def test_outer_zero_points_use_minus_point_one_to_plus_point_one_mm(self) -> None:
        _image, part, values = self._plate()
        values[part] = 0.2
        values[18:25, 55:105] = 0.05

        _contour, points = select_outer_zero_points(values, part)

        self.assertGreater(len(points), 0)
        self.assertTrue(all(-0.1 <= point["value_mm"] <= 0.1 for point in points))

    def test_interior_region_gets_a_closed_surrounding_line(self) -> None:
        image, part, values = self._plate()
        image[50:85, 90:130] = (230, 25, 25)
        values[part] = 0.0

        result = construct_zero_lines(image, values, part)

        interior = [line for line in result.lines if line["kind"] == "interior_enclosure"]
        self.assertEqual(len(interior), 1)
        self.assertTrue(interior[0]["closed"])
        self.assertEqual(interior[0]["points"][0], interior[0]["points"][-1])

    def test_boundary_region_uses_two_outer_zero_points_and_avoids_it(self) -> None:
        image, part, values = self._plate()
        image[48:88, 20:58] = (230, 25, 25)
        values[part] = 0.0

        result = construct_zero_lines(image, values, part)

        boundary = [line for line in result.lines if line["kind"] == "boundary_connection"]
        self.assertEqual(len(boundary), 1, result.warnings)
        points = np.asarray(boundary[0]["points"], dtype=np.int32)
        self.assertGreaterEqual(len(points), 2)
        crossing = np.zeros(part.shape, dtype=np.uint8)
        cv2.polylines(crossing, [points.reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8)
        # Endpoint portals may touch a dilated edge, but the actual correction
        # component must not be crossed.
        self.assertFalse(np.any((crossing > 0) & result.correction_mask))

    def test_selection_preview_adds_an_explanation_header(self) -> None:
        image, part, values = self._plate()
        image[50:85, 90:130] = (230, 25, 25)
        values[part] = 0.0
        result = construct_zero_lines(image, values, part)

        preview = render_selection_preview(image, part, result)

        self.assertEqual(preview.shape[2], 3)
        self.assertGreater(preview.shape[0], image.shape[0])
        self.assertGreater(int(np.count_nonzero(preview != 255)), 0)


if __name__ == "__main__":
    unittest.main()
