"""Tests for the final image-rule zero lines (sigma correction regions, outline-only lines)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import json

from zero_line_detection.final_zero_line import (
    FinalZeroLineConfig,
    _candidate_pairs,
    _FreeSpace,
    _zeros_on_side,
    build_deviation_from_color_ramp,
    build_sigma_correction_mask,
    construct_final_zero_lines,
    deviation_sigma,
    render_final_preview,
    zero_point_labels,
)


def plate():
    image = np.full((160, 260, 3), 255, dtype=np.uint8)
    part = np.zeros(image.shape[:2], dtype=bool)
    part[20:141, 20:241] = True
    image[part] = (0, 210, 50)
    values = np.zeros(part.shape, dtype=np.float32)
    return image, part, values


class ColourReadingTests(unittest.TestCase):
    """Grey drawing lines have no hue; they must not be read as the red end of the colour bar."""

    @staticmethod
    def _ramp():
        hue = np.linspace(0, 150, 200).astype(np.uint8)                    # red at the top ... magenta at the bottom
        hsv = np.stack([hue, np.full(200, 255, np.uint8), np.full(200, 255, np.uint8)], axis=1)
        return cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)

    def _image(self):
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        middle = cv2.cvtColor(np.array([[[75, 255, 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]   # the 0 mm colour
        image[20:180, 20:280] = middle
        for x in (100, 104, 108):                                          # three grey CAD lines side by side
            image[20:180, x:x + 2] = (110, 110, 110)
        positive = cv2.cvtColor(np.array([[[60, 255, 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]  # about +0.4 mm
        image[110:170, 190:260] = positive
        image[120:160, 200:250] = (150, 150, 150)                          # a wide grey (out-of-range) surface inside it
        return image

    def test_drawing_lines_take_the_surrounding_value(self) -> None:
        values, part, report = build_deviation_from_color_ramp(self._image(), self._ramp(), -2.0, 2.0)
        self.assertTrue(part[100, 104])
        self.assertLess(float(np.max(np.abs(values[30:110, 98:112]))), 0.2, "lines read as 0 mm like their surroundings")
        self.assertGreater(report["unreadable_line_px"], 0)

    def test_a_wide_grey_surface_is_still_out_of_range(self) -> None:
        values, _part, report = build_deviation_from_color_ramp(self._image(), self._ramp(), -2.0, 2.0)
        # Surrounded by positive deviation, so it is beyond the positive end of the bar.
        self.assertGreater(float(np.median(values[128:152, 208:242])), 2.0)
        self.assertGreater(report["gray_surface_px"], 0)


class SigmaTests(unittest.TestCase):
    def test_sigma_is_the_spread_about_zero_not_about_the_mean(self) -> None:
        part = np.ones((10, 10), dtype=bool)
        values = np.full(part.shape, 1.0)            # every point +1 mm: std 0, but 1 mm from zero
        self.assertAlmostEqual(deviation_sigma(values, part), 1.0)
        self.assertAlmostEqual(deviation_sigma(values, part, source="colorbar", colorbar_range=(-2.0, 1.5)), 2.0 / 3.0)

    def test_regions_are_outside_one_sigma_and_noise_is_removed(self) -> None:
        _image, part, values = plate()
        values[40:80, 60:120] = 2.0                  # a real region
        values[110:113, 200:203] = -3.0              # a 3 x 3 speck
        mask, report = build_sigma_correction_mask(values, part, threshold_mm=0.5, min_region_ratio=0.0005)
        self.assertTrue(mask[60, 90])
        self.assertFalse(mask[111, 201], "a speck is noise")
        self.assertFalse(mask[100, 150])
        self.assertGreaterEqual(report["noise_regions_removed"], 1)


class LineTests(unittest.TestCase):
    def _run(self, values, part, image, **config):
        return construct_final_zero_lines(image, values, part, config=FinalZeroLineConfig(**config))

    def test_region_on_the_outline_gets_a_line_that_avoids_it(self) -> None:
        image, part, values = plate()
        values[20:70, 100:150] = 2.0                 # touches the top edge
        result, report = self._run(values, part, image, near_outline_ratio=0.05)
        self.assertEqual(len(result.lines), 1, result.warnings)
        self.assertEqual(report["regions"][0]["kind"], "touching_outline")
        points = np.asarray(result.lines[0]["points"], dtype=np.int32)
        drawn = np.zeros(part.shape, dtype=np.uint8)
        cv2.polylines(drawn, [points.reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8)
        self.assertFalse(np.any((drawn > 0) & result.correction_mask))
        xs = sorted([points[0][0], points[-1][0]])
        self.assertLess(xs[0], 100)
        self.assertGreater(xs[1], 150)

    def test_lines_keep_the_shape_the_viewer_and_the_sheet_need(self) -> None:
        """The 3D overlay and the correction sheet read id + points in scan pixels, nothing else.

        server.analyze_image stores line["points"] as lab_zero_lines, cad_overlay_for unprojects
        those pixels onto the mesh and page.tsx turns them into sheet percentages; the UI edit index
        is positional, so ids must be unique and the order stable. The whole list is also written to
        the .npz cache as JSON, so every value has to survive a JSON round trip.
        """
        image, part, values = plate()
        values[20:70, 100:150] = 2.0
        values[20:60, 30:60] = -2.5                  # a second region, so the order matters
        result, _report = self._run(values, part, image, near_outline_ratio=0.05)
        self.assertGreaterEqual(len(result.lines), 2, result.warnings)
        restored = json.loads(json.dumps(result.lines))
        self.assertEqual(restored, result.lines, "lines must survive the cache's JSON round trip")
        self.assertEqual([line["id"] for line in result.lines], list(range(1, len(result.lines) + 1)))
        height, width = part.shape
        for line in result.lines:
            points = np.asarray(line["points"])
            self.assertGreaterEqual(len(points), 2)
            self.assertEqual(points.shape[1], 2)
            self.assertTrue((points[:, 0] >= 0).all() and (points[:, 0] < width).all())
            self.assertTrue((points[:, 1] >= 0).all() and (points[:, 1] < height).all())

    def test_interior_region_gets_no_line(self) -> None:
        image, part, values = plate()
        values[70:100, 110:150] = 2.0                # 50 px from every edge
        result, report = self._run(values, part, image, near_outline_ratio=0.05)
        self.assertEqual(result.lines, [])
        self.assertEqual(report["regions"][0]["kind"], "interior_no_line")

    def test_region_close_to_the_outline_counts_as_touching_it(self) -> None:
        image, part, values = plate()
        values[27:60, 100:150] = 2.0                 # 7 px below the top edge, not touching
        far = self._run(values, part, image, near_outline_ratio=0.0)
        near = self._run(values, part, image, near_outline_ratio=0.1)
        self.assertEqual(far[0].lines, [])
        self.assertEqual(near[1]["regions"][0]["kind"], "near_outline")
        self.assertEqual(len(near[0].lines), 1, near[0].warnings)
        # The gap to the edge is filled, so the line goes round the region's inner side instead of
        # running along the outline through the gap.
        self.assertTrue(near[0].correction_mask[23, 125], "the gap above the region is filled")
        self.assertGreater(near[1]["gap_filled_px"], 0)
        points = np.asarray(near[0].lines[0]["points"], dtype=np.int32)
        self.assertGreater(points[:, 1].max(), 60, "the route passes below the region")

    def test_zero_points_are_numbered_once_per_endpoint(self) -> None:
        lines = [{"kind": "boundary_connection", "region": "M1", "points": [[0, 0], [50, 0]], "zeroPointValuesMm": [0.0, 0.05]},
                 {"kind": "boundary_connection", "region": "M2", "points": [[52, 1], [90, 0]], "zeroPointValuesMm": [0.05, -0.02]}]
        labels = zero_point_labels(lines)
        self.assertEqual([item["label"] for item in labels], ["Z1", "Z2", "Z3"])
        self.assertEqual(labels[1]["regions"], ["M1", "M2"])

    def test_zero_points_on_a_side_come_nearest_first_and_skip_the_contact(self) -> None:
        contact = np.zeros(20, dtype=bool)
        contact[8:12] = True
        zeros = {index: {"sample_index": index, "point": [index, 0], "value_mm": 0.0} for index in (2, 5, 9, 14, 17)}
        left = _zeros_on_side(zeros, contact, 8, -1)
        right = _zeros_on_side(zeros, contact, 11, 1)
        self.assertEqual([z["sample_index"] for z in left][:2], [5, 2])
        self.assertEqual([z["sample_index"] for z in right][:2], [14, 17])
        self.assertNotIn(9, [z["sample_index"] for z in left + right], "a zero point inside the contact is not used")

    def test_a_zero_point_trapped_in_a_pocket_is_skipped_for_the_next_one(self) -> None:
        strict = np.zeros((60, 100), dtype=np.uint8)
        strict[20:24, 0:44] = 255                     # correction below ...
        strict[0:24, 40:44] = 255                     # ... and beside: a sealed pocket in the top-left corner
        free = _FreeSpace(strict, search_px=1)
        trapped = {"sample_index": 1, "point": [10, 5], "value_mm": 0.0}
        open_left = {"sample_index": 2, "point": [10, 40], "value_mm": 0.0}
        further = {"sample_index": 3, "point": [30, 50], "value_mm": 0.0}
        right = {"sample_index": 4, "point": [80, 30], "value_mm": 0.0}
        self.assertNotEqual(free.piece(trapped["point"]), free.piece(right["point"]))
        pairs = _candidate_pairs([trapped, open_left, further], [right], free)
        self.assertEqual(pairs[0][2], (0, 0), "the classic first pair is always tried first")
        self.assertEqual(pairs[1][2], (1, 0), "then the nearest pair the free space connects")
        self.assertEqual(pairs[2][2], (2, 0))

    def test_preview_renders(self) -> None:
        image, part, values = plate()
        values[20:70, 100:150] = 2.0
        result, report = self._run(values, part, image)
        preview = render_final_preview(image, part, result, report)
        self.assertEqual(preview.shape[2], 3)
        self.assertGreater(preview.shape[0], 150)


if __name__ == "__main__":
    unittest.main()
