"""Tests for label removal that keeps the product outline (label_removal/outline_preserving.py)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from label_removal.outline_preserving import (
    _bridge_annotations,
    estimate_leader_width,
    leader_corridors,
    mesh_matches_points,
    mesh_silhouette,
    product_mask_from_image,
    product_mask_with_cad,
    scan_mask_and_core,
)

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def quad(x0, y0, x1, y1, z):
    a, b, c, d = (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)
    return [(a, b, c), (a, c, d)]


class MeshSilhouetteTests(unittest.TestCase):
    def test_overlapping_skins_leave_no_holes(self) -> None:
        # Top skin, bottom skin and a wall seen edge-on overlap: an even-odd fill would cancel them.
        triangles = np.array(quad(10, 10, 60, 40, 0.0) + quad(10, 10, 60, 40, -2.0) + quad(10, 10, 60, 40, -1.0), float)
        mask = mesh_silhouette(triangles, IDENTITY, (0, 1), (60, 80))
        self.assertTrue(mask[12:39, 12:59].all())
        self.assertFalse(mask[45:, :].any())

    def test_mirrored_mesh_is_rejected(self) -> None:
        triangles = np.array(quad(10, 10, 60, 40, 0.0), float)
        points = triangles.reshape(-1, 3)
        self.assertTrue(mesh_matches_points(triangles, points)["matches"])
        mirrored = triangles.copy()
        mirrored[:, :, 0] *= -1
        self.assertFalse(mesh_matches_points(mirrored, points)["matches"])


class BridgeTests(unittest.TestCase):
    def run_bridge(self, part, box):
        annotation = np.zeros(part.shape, bool)
        annotation[box[1]:box[3], box[0]:box[2]] = True
        body = part & ~annotation
        return _bridge_annotations(body, annotation, 7, 12, background=~part & ~annotation)

    def test_straight_edge_continues_through_a_box(self) -> None:
        part = np.zeros((120, 120), bool)
        part[:, :60] = True
        restored = self.run_bridge(part, (45, 40, 80, 70))
        self.assertTrue(restored[42:68, 46:58].all())
        self.assertFalse(restored[40:70, 62:80].any())

    def test_hole_corner_under_a_box_is_not_filled(self) -> None:
        part = np.ones((120, 120), bool)
        part[50:, 50:] = False                          # hole whose corner lies under the box
        restored = self.run_bridge(part, (35, 35, 70, 70))
        self.assertTrue(restored[36:48, 36:69].all() and restored[36:69, 36:48].all())
        self.assertFalse(restored[53:70, 53:70].any())

    def test_part_corner_under_a_box_is_completed(self) -> None:
        part = np.zeros((120, 120), bool)
        part[:60, :60] = True                           # convex part corner under the box
        restored = self.run_bridge(part, (40, 40, 75, 75))
        self.assertTrue(restored[41:58, 41:58].all())
        self.assertFalse(restored[63:75, 40:75].any() or restored[40:75, 63:75].any())


class ImageOnlyProductTests(unittest.TestCase):
    def test_leader_width_and_stub_removal(self) -> None:
        image = np.full((200, 300, 3), 255, np.uint8)
        image[40:160, 40:200] = (60, 200, 60)
        thin = np.zeros(image.shape[:2], bool)
        for y in (70, 100, 130):                        # 5 px wide leaders from the part edge to the right
            image[y - 2:y + 3, 180:290] = (230, 20, 20)
            thin[y - 2:y + 3, 180:290] = True
        self.assertAlmostEqual(estimate_leader_width(image), 5.0, delta=1.5)
        scan, core = scan_mask_and_core(image, estimate_leader_width(image))
        self.assertFalse(scan[:, 215:].any())            # leaders cut off, not kept as part
        product, _ = product_mask_from_image(image, {"boxes": np.zeros_like(thin), "thin": thin}, scan, core)
        self.assertTrue(product[40:160, 40:200].all())   # edge under the leader ends kept
        self.assertFalse(product[:, 205:].any())

    def test_blue_rim_is_not_a_leader_but_its_continuation_is(self) -> None:
        strokes = np.zeros((120, 200), bool)
        strokes[20:100, 98:101] = True                  # blue deviation rim along the part edge (x < 100 is part)
        strokes[59:62, 98:190] = True                   # leader coming in from the right
        off_part = np.zeros_like(strokes)
        off_part[:, 104:] = True
        corridor = leader_corridors(strokes, off_part, 3.0, reach_px=15)
        self.assertTrue(corridor[60, 92:104].all())
        self.assertFalse(corridor[20:50, 99].any() or corridor[72:100, 99].any())


class CadProductTests(unittest.TestCase):
    def test_label_over_edge_follows_silhouette(self) -> None:
        image = np.full((120, 160, 3), 255, np.uint8)
        image[20:100, 20:100] = (60, 200, 60)
        image[40:70, 85:140] = (0, 0, 230)               # label box across the right edge
        annotation = np.zeros(image.shape[:2], bool)
        annotation[40:70, 85:140] = True
        silhouette = np.zeros(image.shape[:2], bool)
        silhouette[20:100, 20:100] = True
        product, restored = product_mask_with_cad(image, annotation, silhouette)
        self.assertTrue(product[20:100, 20:100].all())
        self.assertFalse(product[:, 104:].any())
        self.assertTrue(restored[40:70, 85:100].all())


if __name__ == "__main__":
    unittest.main()
