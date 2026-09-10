"""Tests for editable zero-line contour output."""

from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from zero_line_detection import hybrid_ui


_mask_contours_as_lines = hybrid_ui._mask_contours_as_lines


class HybridUiEditPointTests(unittest.TestCase):

    def test_case2_failure_is_not_returned_or_cached_as_case1(self) -> None:
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        base = SimpleNamespace(
            values=np.zeros((8, 12), dtype=np.float32),
            part_mask=np.ones((8, 12), dtype=bool),
            colorbar=SimpleNamespace(
                info=SimpleNamespace(y0=0, y1=8, x0=0, x1=2),
                colors_rgb=np.zeros((8, 3), dtype=np.uint8),
            ),
            warnings=[],
        )
        common = {
            "part": np.ones((8, 12), dtype=bool),
            "part_px": 96,
            "positive": np.zeros((8, 12), dtype=bool),
            "negative": np.zeros((8, 12), dtype=bool),
            "zero": np.ones((8, 12), dtype=bool),
            "zero_ratio": 0.8,
            "zero_count": 1,
        }
        from zero_line_detection import case2_route_adapter as case2
        from zero_line_detection import generate_final_hybrid_zero_line as hybrid

        with mock.patch.object(
            hybrid, "build_common_from_color_ramp", return_value=common
        ), mock.patch.object(
            case2, "run_original_case2_pipeline", side_effect=ValueError("route unavailable")
        ):
            with self.assertRaisesRegex(RuntimeError, "Case 2 경로 계산 실패"):
                hybrid_ui._detect_hybrid_zero_line_uncached(
                    image,
                    "another-item.png",
                    base=base,
                    colorbar_range_mm=(-2.0, 2.0),
                )

    def test_case2_mask_uses_review_engine_rasterization(self) -> None:
        selections = [
            {
                "closure_validation": {
                    "route": {"path_points": [[2, 5], [17, 5]]}
                }
            }
        ]
        actual = hybrid_ui._routes_to_review_mask(selections, (12, 20))
        expected = np.zeros((12, 20), dtype=np.uint8)
        cv2.polylines(
            expected,
            [np.asarray([[2, 5], [17, 5]], dtype=np.int32).reshape(-1, 1, 2)],
            False,
            255,
            4,
            cv2.LINE_8,
        )
        np.testing.assert_array_equal(actual, expected.astype(bool))

    def test_cache_key_changes_when_supplied_base_values_change(self) -> None:
        image = np.zeros((8, 12, 3), dtype=np.uint8)

        def base(value: float):
            values = np.full(image.shape[:2], value, dtype=np.float32)
            return SimpleNamespace(
                values=values,
                part_mask=np.ones(image.shape[:2], dtype=bool),
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                zero_crossing=np.zeros(image.shape[:2], dtype=np.uint8),
                colorbar=SimpleNamespace(vmin=-3.0, vmax=3.0),
                result=SimpleNamespace(tolerance=0.3, tolerance_unit="mm"),
            )

        first = hybrid_ui._cache_key(image, "JD_67XX6.png", base(0.0))
        second = hybrid_ui._cache_key(image, "JD_67XX6.png", base(1.0))

        self.assertNotEqual(first, second)

    def test_cache_key_changes_when_ocr_colorbar_range_changes(self) -> None:
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        first = hybrid_ui._cache_key(image, "scan.png", None, (-1.5, 2.0))
        second = hybrid_ui._cache_key(image, "scan.png", None, (-3.0, 3.0))
        self.assertNotEqual(first, second)

    def test_runtime_selection_modules_stay_inside_zero_line_package(self) -> None:
        from zero_line_detection import case2_route_adapter
        from zero_line_detection.adaptive_bundle import generate_adaptive_zero_line_preview

        package_dir = Path(hybrid_ui.__file__).resolve().parent
        runtime_modules = (
            hybrid_ui,
            case2_route_adapter,
            case2_route_adapter.selector,
            generate_adaptive_zero_line_preview,
        )
        for module in runtime_modules:
            with self.subTest(module=module.__name__):
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(package_dir))

        normalized_paths = [str(Path(item).resolve()).replace("\\", "/") for item in sys.path]
        self.assertFalse(
            any("/experiments/zero_line_area_edge_preview" in item for item in normalized_paths)
        )

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
            colorbar=SimpleNamespace(colors_rgb=np.zeros((8, 3), dtype=np.uint8)),
            result=SimpleNamespace(regions=[]),
            warnings=[],
        )
        common = {
            "image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "part": np.ones(image.shape[:2], dtype=bool),
            "part_px": image.shape[0] * image.shape[1],
            "positive": np.zeros(image.shape[:2], dtype=bool),
            "negative": np.zeros(image.shape[:2], dtype=bool),
            "zero": np.ones(image.shape[:2], dtype=bool),
            "zero_ratio": 0.2,
            "zero_count": 2,
        }
        from zero_line_detection import generate_final_hybrid_zero_line as hybrid
        with mock.patch.object(
            hybrid_ui, "detect_zero_line", side_effect=AssertionError("duplicate detection")
        ), mock.patch.object(
            hybrid, "build_common_from_color_ramp", return_value=common
        ), mock.patch.object(
            hybrid, "run_case1", return_value=(mask.astype(bool), {})
        ), mock.patch(
            "zero_line_detection.adaptive_bundle.generate_adaptive_zero_line_preview.build_board",
            return_value=(image, image),
        ):
            result = hybrid_ui._detect_hybrid_zero_line_uncached(
                image,
                "unregistered-sample.png",
                base=base,
                colorbar_range_mm=(-2.0, 2.0),
            )

        self.assertEqual(result.mask.shape, image.shape[:2])


if __name__ == "__main__":
    unittest.main()
