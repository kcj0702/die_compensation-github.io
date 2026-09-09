from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_zero_line import evaluate_line, evaluate_points, rasterize_lines


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


if __name__ == "__main__":
    unittest.main()
