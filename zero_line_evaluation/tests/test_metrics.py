from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_zero_line import evaluate_area, evaluate_line, evaluate_points, rasterize_area, rasterize_lines


class ZeroLineMetricTest(unittest.TestCase):
    def test_identical_lines_score_one(self) -> None:
        lines = [{"points": [[5, 10], [25, 10]]}]
        mask = rasterize_lines((32, 32), lines)
        result = evaluate_line(mask, mask.copy(), tolerance_px=2)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["mean_symmetric_distance_px"], 0.0)

    def test_tolerance_accepts_near_parallel_line(self) -> None:
        pred = rasterize_lines((40, 40), [{"points": [[5, 10], [30, 10]]}])
        gt = rasterize_lines((40, 40), [{"points": [[5, 13], [30, 13]]}])
        self.assertAlmostEqual(evaluate_line(pred, gt, tolerance_px=3)["f1"], 1.0)
        self.assertEqual(evaluate_line(pred, gt, tolerance_px=2)["f1"], 0.0)

    def test_points_report_hit_rate(self) -> None:
        pred = np.zeros((40, 40), dtype=bool)
        cv2.line(pred.view(np.uint8), (5, 10), (30, 10), 1, 1)
        points = np.asarray([[8, 11], [20, 20]], dtype=np.float32)
        result = evaluate_points(pred, points, tolerance_px=2)
        self.assertEqual(result["matched_point_count"], 1)
        self.assertAlmostEqual(result["point_hit_rate"], 0.5)

    def test_identical_areas_score_one(self) -> None:
        square = [{"points": [[5, 5], [25, 5], [25, 25], [5, 25]]}]
        mask = rasterize_area((32, 32), square)
        result = evaluate_area(mask, mask.copy())
        self.assertAlmostEqual(result["iou"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)

    def test_partial_area_overlap_scores_iou(self) -> None:
        pred = rasterize_area((40, 40), [{"points": [[0, 0], [20, 0], [20, 20], [0, 20]]}])
        gt = rasterize_area((40, 40), [{"points": [[10, 0], [30, 0], [30, 20], [10, 20]]}])
        result = evaluate_area(pred, gt)
        # Cross-check against a plain numpy overlap computation rather than an
        # assumed exact pixel count — cv2.fillPoly's edge rasterization doesn't
        # land on a clean analytic fraction.
        expected_iou = np.logical_and(pred, gt).sum() / np.logical_or(pred, gt).sum()
        self.assertAlmostEqual(result["iou"], expected_iou, places=6)
        self.assertLess(result["iou"], 1.0)
        self.assertGreater(result["iou"], 0.0)

    def test_rasterize_area_ignores_open_two_point_lines(self) -> None:
        mask = rasterize_area((20, 20), [{"points": [[2, 2], [10, 10]]}])
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
