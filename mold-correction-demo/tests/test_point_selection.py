"""회사 원본 없이 주요 편차 포인트 선별 계약을 검증한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from point_selection import config, select_key_points  # noqa: E402


def _point(point_id: str, x: int, y: int, value: float) -> dict:
    return {"id": point_id, "xPx": x, "yPx": y, "value": value}


class _Object:
    """dict 대신 속성으로 좌표를 노출하는 입력도 받아야 한다."""

    def __init__(self, point_id: str, x: int, y: int, value: float):
        self.id = point_id
        self.xPx = x
        self.yPx = y
        self.value = value


class SelectionTest(unittest.TestCase):
    def test_empty_input_selects_nothing(self) -> None:
        selection = select_key_points([])

        self.assertEqual(selection.keys, [])
        self.assertEqual(selection.total, 0)
        self.assertEqual(selection.to_dict()["selected"], 0)

    def test_local_extreme_is_kept_and_its_slope_is_not(self) -> None:
        # 가로로 늘어선 값: 가운데가 국소 최소다.
        points = [
            _point("P-01", 0, 0, -0.4),
            _point("P-02", 100, 0, -0.8),
            _point("P-03", 200, 0, -2.0),
            _point("P-04", 300, 0, -0.9),
            _point("P-05", 400, 0, -0.5),
            _point("P-06", 500, 0, -0.4),
        ]

        selection = select_key_points(points, peak_neighbours=2)
        reasons = {key.point_id: key.reasons for key in selection.keys}

        self.assertIn("peak", reasons["P-03"])
        self.assertNotIn("P-04", reasons)

    def test_near_zero_ripple_is_not_a_peak(self) -> None:
        points = [
            _point("P-01", 0, 0, 0.0),
            _point("P-02", 100, 0, 0.1),
            _point("P-03", 200, 0, 0.0),
            _point("P-04", 300, 0, 0.1),
            _point("P-05", 400, 0, 0.0),
        ]

        selection = select_key_points(
            points, peak_neighbours=2, keep_extremes=False
        )

        self.assertEqual(
            [key.point_id for key in selection.keys if "peak" in key.reasons], []
        )

    def test_sign_change_needs_both_sides_to_be_meaningful(self) -> None:
        weak = [_point("P-01", 0, 0, 0.05), _point("P-02", 40, 0, -0.9)]
        strong = [_point("P-03", 0, 0, 0.6), _point("P-04", 40, 0, -0.9)]

        weak_selection = select_key_points(weak, keep_extremes=False)
        strong_selection = select_key_points(strong, keep_extremes=False)

        self.assertEqual(weak_selection.count("sign_change"), 0)
        self.assertGreater(strong_selection.count("sign_change"), 0)

    def test_zero_crossing_run_collapses_to_one_marker(self) -> None:
        # 부호가 번갈아 바뀌는 촘촘한 줄. 전부가 아니라 대표점만 남아야 한다.
        points = [
            _point(f"P-{index:02d}", index * 10, 0, 0.6 if index % 2 else -0.6)
            for index in range(12)
        ]

        selection = select_key_points(
            points, sign_merge_radius=40.0, keep_extremes=False
        )

        self.assertGreater(selection.count("sign_change"), 0)
        self.assertLess(selection.count("sign_change"), len(points) // 2)

    def test_global_extremes_are_always_kept(self) -> None:
        points = [
            _point("P-01", 0, 0, 0.1),
            _point("P-02", 100, 0, 0.2),
            _point("P-03", 200, 0, 5.0),
            _point("P-04", 300, 0, -4.0),
            _point("P-05", 400, 0, 0.1),
        ]

        reasons = {
            key.point_id: key.reasons for key in select_key_points(points).keys
        }

        self.assertIn("extreme", reasons["P-03"])
        self.assertIn("extreme", reasons["P-04"])

    def test_extremes_can_be_turned_off(self) -> None:
        points = [_point("P-01", 0, 0, 0.1), _point("P-02", 100, 0, 0.2)]

        selection = select_key_points(points, keep_extremes=False)

        self.assertEqual(selection.count("extreme"), 0)

    def test_object_input_is_accepted(self) -> None:
        points = [
            _Object("P-01", 0, 0, -0.4),
            _Object("P-02", 100, 0, -2.0),
            _Object("P-03", 200, 0, -0.5),
        ]

        selection = select_key_points(points, peak_neighbours=2)

        self.assertIn("P-02", selection.ids)

    def test_selection_never_returns_more_than_the_input(self) -> None:
        points = [
            _point(f"P-{index:02d}", index * 37 % 400, index * 53 % 300, (index % 7) - 3)
            for index in range(40)
        ]

        selection = select_key_points(points)

        self.assertEqual(selection.total, 40)
        self.assertLessEqual(len(selection.keys), 40)
        self.assertEqual(len(set(selection.ids)), len(selection.ids))

    def test_summary_counts_match_the_reasons(self) -> None:
        points = [
            _point("P-01", 0, 0, 0.8),
            _point("P-02", 60, 0, -0.9),
            _point("P-03", 400, 0, -0.2),
        ]

        summary = select_key_points(points).to_dict()

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["selected"], len(summary["points"]))
        self.assertEqual(
            summary["signChanges"],
            sum(1 for item in summary["points"] if "sign_change" in item["reasons"]),
        )

    def test_defaults_come_from_config(self) -> None:
        self.assertGreaterEqual(config.PEAK_NEIGHBOURS, 1)
        self.assertGreater(config.PEAK_MIN_ABS_MM, 0)
        self.assertGreater(config.SIGN_MERGE_RADIUS_RATIO, 0)


if __name__ == "__main__":
    unittest.main()
