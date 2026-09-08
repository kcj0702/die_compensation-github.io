"""Tests for editable zero-line contour output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from zero_line_detection import hybrid_ui


_mask_contours_as_lines = hybrid_ui._mask_contours_as_lines


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

    def test_exact_image_result_is_loaded_from_disk_cache(self) -> None:
        image = np.full((24, 36, 3), 127, dtype=np.uint8)
        expected = hybrid_ui.HybridZeroLineOutput(
            mask=np.eye(24, 36, dtype=bool),
            overlay_rgb=np.full((24, 36, 3), 80, dtype=np.uint8),
            case=2,
            regions=1,
            ratio=0.25,
            lines=[{"id": 1, "points": [[1, 2], [3, 4]]}],
            warnings=["test warning"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(hybrid_ui, "_cache_dir", return_value=Path(temp_dir)), mock.patch.object(
                hybrid_ui,
                "_detect_hybrid_zero_line_uncached",
                return_value=expected,
            ) as uncached:
                first = hybrid_ui.detect_hybrid_zero_line(image, "sample.png")
                second = hybrid_ui.detect_hybrid_zero_line(image, "sample.png")

        self.assertEqual(uncached.call_count, 1)
        np.testing.assert_array_equal(first.mask, second.mask)
        np.testing.assert_array_equal(first.overlay_rgb, second.overlay_rgb)
        self.assertEqual(first.lines, second.lines)
        self.assertEqual(first.warnings, second.warnings)

    def test_uncached_generic_path_reuses_supplied_basic_detection(self) -> None:
        image = np.full((24, 36, 3), 255, dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        base = SimpleNamespace(
            part_mask=np.ones(image.shape[:2], dtype=bool),
            mask=mask,
            centerline=None,
            zero_crossing=mask,
            values=np.zeros(image.shape[:2], dtype=np.float32),
            result=SimpleNamespace(regions=[]),
            warnings=[],
        )
        with mock.patch.object(hybrid_ui, "_detect_from_review_inputs", return_value=None), mock.patch.object(
            hybrid_ui, "detect_zero_line", side_effect=AssertionError("duplicate detection")
        ):
            result = hybrid_ui._detect_hybrid_zero_line_uncached(
                image, "unregistered-sample.png", base=base
            )

        self.assertEqual(result.mask.shape, image.shape[:2])


if __name__ == "__main__":
    unittest.main()
