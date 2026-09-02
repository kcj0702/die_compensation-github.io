"""Regression tests for label/measurement-point removal."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from label_removal.remove_labels import detect_exact_hsv_leader_lines


class ExactLeaderDetectionTests(unittest.TestCase):
    def test_five_pixel_direct_contact_leader_keeps_exact_endpoint(self) -> None:
        image = np.full((60, 80, 3), (40, 180, 40), dtype=np.uint8)
        image[30, 20:25] = (255, 0, 0)
        scan_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        label_box = (25, 24, 45, 38)

        _, point_specs, point_boxes = detect_exact_hsv_leader_lines(
            image,
            [label_box],
            scan_mask,
            return_point_boxes=True,
        )

        self.assertEqual(point_boxes, [label_box])
        self.assertEqual(len(point_specs), 1)
        self.assertEqual(point_specs[0][:2], (20, 30))


if __name__ == "__main__":
    unittest.main()
