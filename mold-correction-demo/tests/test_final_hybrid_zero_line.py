"""Unit tests for the review-only hybrid zero-line integration."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HYBRID_PATH = ROOT / "zero_line_detection" / "generate_final_hybrid_zero_line.py"
SPEC = importlib.util.spec_from_file_location(
    "final_hybrid_zero_line",
    HYBRID_PATH,
)
assert SPEC is not None and SPEC.loader is not None
hybrid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid)


class HybridCaseSelectionTests(unittest.TestCase):
    def test_case1_requires_low_ratio_and_multiple_components(self) -> None:
        self.assertEqual(hybrid.select_case(0.399, 2), 1)
        self.assertEqual(hybrid.select_case(0.400, 2), 2)
        self.assertEqual(hybrid.select_case(0.200, 1), 2)

    def test_ui_common_inputs_use_final_correction_regions_for_decision(self) -> None:
        part = np.ones((120, 240), dtype=bool)
        values = np.zeros(part.shape, dtype=np.float32)
        # A retained correction band separates the remaining zero area into
        # two components whose combined area is below the 40% Case-1 limit.
        values[:, 36:204] = 1.0

        common = hybrid.build_common_from_detection(
            np.zeros((*part.shape, 3), dtype=np.uint8), values, part
        )

        self.assertEqual(
            hybrid.select_case(common["zero_ratio"], common["zero_count"]), 1
        )
        self.assertLess(common["zero_ratio"], 0.40)
        self.assertGreater(common["zero_count"], 1)

    def test_interior_boundary_sampling_finds_spatially_distinct_crossings(self) -> None:
        part = np.zeros((120, 180), dtype=bool)
        cv2.rectangle(part, (20, 20), (160, 100), 1, -1)
        mapped = part.copy()
        values = np.zeros(part.shape, dtype=np.float32)
        values[:, :90] = -1.0
        values[:, 90:] = 1.0
        contour, _ = hybrid.outer_contour_geometry(part)

        points = hybrid._fallback_sign_crossings(contour, values, mapped, part)

        self.assertGreaterEqual(len(points), 2)
        coordinates = np.asarray([(x, y) for x, y, _ in points], dtype=np.float64)
        self.assertGreater(float(np.ptp(coordinates[:, 1])), 40.0)

    def test_route_mask_uses_all_selected_paths(self) -> None:
        selections = [
            {"closure_validation": {"route": {"path_points": [[5, 5], [40, 5]]}}},
            {"closure_validation": {"route": {"path_points": [[10, 20], [10, 45]]}}},
        ]

        mask = hybrid.routes_to_mask(selections, (60, 60))

        self.assertGreater(int(mask[5, 20]), 0)
        self.assertGreater(int(mask[30, 10]), 0)

    def test_method2_view_does_not_draw_zero_point_numbers(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        case2 = {
            "contour_points": np.asarray([[2, 2], [37, 2], [37, 37], [2, 37]]),
            "zero_points": [{"label": "Z1", "point": np.asarray([20, 2])}],
            "selections": [],
        }

        with mock.patch.object(hybrid.cv2, "putText") as put_text:
            hybrid.draw_team_route_view(
                image,
                case2,
                np.zeros((40, 40), dtype=bool),
            )

        put_text.assert_not_called()

    def test_single_entry_point_dispatches_case1(self) -> None:
        common = {"zero_ratio": 0.2, "zero_count": 2}
        expected = np.ones((3, 4), dtype=bool)
        case_data = {
            "rows": [],
            "split_events": [],
            "corner_squaring_rows": [],
        }
        with mock.patch.object(hybrid, "run_case1", return_value=(expected, case_data)):
            result = hybrid.detect_zero_line(common)

        self.assertEqual(result.selected_case, 1)
        self.assertIs(result.final_mask, expected)
        self.assertEqual(result.method, "case1_kdt_offset_contour_polygon")

    def test_adapter_filters_short_route_without_modifying_selector(self) -> None:
        short = {
            "region": {"label": "R1"},
            "closure_validation": {"route": {"path_length_pixels": 99.9}},
        }
        long = {
            "region": {"label": "R2"},
            "closure_validation": {"route": {"path_length_pixels": 100.0}},
        }
        with (
            mock.patch.object(hybrid.case2_adapter.selector, "extract_regions", return_value=[{}]),
            mock.patch.object(
                hybrid.case2_adapter.selector,
                "select_along_outer_contour",
                return_value=([short, long], 12, 4),
            ),
        ):
            result = hybrid.case2_adapter.run_route_selector(
                correction_mask=np.ones((8, 8), dtype=bool),
                zero_points=[],
                contour_points=np.zeros((3, 2), dtype=np.float64),
                outer_silhouette=np.ones((8, 8), dtype=bool),
            )

        self.assertEqual(result["selections"], [long])
        self.assertEqual(result["rejected_routes"][0]["region_label"], "R1")

    def test_endpoint_anchor_keeps_nearest_candidate_order(self) -> None:
        planning = np.full((31, 31), 255, dtype=np.uint8)
        planning[12, 15] = 0
        planning[15, 12] = 0
        strict = np.zeros_like(planning)

        anchor = hybrid.case2_adapter.selector.find_endpoint_anchor(
            (15, 15), planning, strict, maximum_radius=6
        )

        self.assertEqual(anchor, (15, 12))


if __name__ == "__main__":
    unittest.main()
