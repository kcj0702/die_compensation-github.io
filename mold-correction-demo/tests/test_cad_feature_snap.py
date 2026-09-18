"""Tests for direction-gated CAD feature snapping."""

from __future__ import annotations

import unittest

import numpy as np

from zero_line_detection.cad_feature_snap import snap_boundary_lines_to_cad


class CadFeatureSnapTests(unittest.TestCase):
    def _boundary_line(self) -> dict:
        return {
            "id": 1,
            "kind": "boundary_connection",
            "region": "M1",
            "closed": False,
            "points": [[10, 50], [110, 50]],
        }

    def test_near_parallel_feature_attracts_boundary_route(self) -> None:
        feature = {
            "id": "fillet:1",
            "kind": "fillet_center",
            "points": np.asarray([[0, 45], [120, 45]], dtype=float),
        }

        lines, records = snap_boundary_lines_to_cad(
            [self._boundary_line()], [feature], (120, 140),
            max_distance_px=8, minimum_run_px=12,
        )

        self.assertTrue(lines[0]["cadSnapped"])
        self.assertGreater(len(records), 0)
        y = np.asarray(lines[0]["points"], dtype=float)[:, 1]
        self.assertAlmostEqual(float(np.median(y[5:-5])), 45.0, delta=0.6)

    def test_near_perpendicular_feature_does_not_attract_route(self) -> None:
        feature = {
            "id": "hole:1",
            "kind": "hole",
            "points": np.asarray([[60, 0], [60, 100]], dtype=float),
        }

        lines, records = snap_boundary_lines_to_cad(
            [self._boundary_line()], [feature], (120, 140),
            max_distance_px=12, minimum_run_px=5,
        )

        self.assertFalse(lines[0].get("cadSnapped", False))
        self.assertEqual(records, [])
        self.assertEqual(lines[0]["points"], self._boundary_line()["points"])

    def test_interior_enclosure_is_not_modified(self) -> None:
        interior = {
            "id": 2,
            "kind": "interior_enclosure",
            "closed": True,
            "points": [[20, 20], [80, 20], [80, 80], [20, 20]],
        }
        feature = {
            "id": "outer:1",
            "kind": "outer",
            "points": np.asarray([[0, 20], [100, 20]], dtype=float),
        }

        lines, records = snap_boundary_lines_to_cad(
            [interior], [feature], (120, 140), max_distance_px=10,
        )

        self.assertEqual(lines[0], interior)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
