"""연결 성분 기반 라벨·리더 검출의 회귀 테스트."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from deviation_extraction.calibrate_colorbar import _end_crop
from deviation_extraction.image_io import read_image, write_image
from deviation_extraction.label_detector import (
    _deduplicate_rectangles,
    build_scan_mask,
    detect_labels,
)
from deviation_extraction.point_extractor import _read_label_values
from deviation_extraction.vlm_reader import LabelValueReader


class LabelRectangleDeduplicationTests(unittest.TestCase):
    def test_fully_contained_inner_outline_is_removed(self) -> None:
        outer = (659, 157, 712, 201)
        inner = (661, 159, 710, 188)

        self.assertEqual(_deduplicate_rectangles([inner, outer]), [outer])


def _scan_image() -> np.ndarray:
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (360, 230), (190, 120), 0, 0, 360, (20, 180, 120), -1)
    return image


def _boundary_scan_image() -> np.ndarray:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (180, 100), (450, 250), (20, 180, 120), -1)
    return image


def _draw_gray_label(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    text: str = "-1.2",
) -> None:
    cv2.rectangle(image, top_left, bottom_right, (155, 155, 155), 2)
    cv2.putText(
        image,
        text,
        (top_left[0] + 8, top_left[1] + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


class ConnectedLeaderTest(unittest.TestCase):
    def test_dense_red_labels_do_not_create_merged_false_candidate(self) -> None:
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 60), (380, 230), (20, 180, 120), -1)
        for x, text, point_x in ((100, "0.8", 120), (140, "0.9", 160)):
            cv2.rectangle(image, (x, 180), (x + 38, 215), (0, 0, 255), -1)
            cv2.rectangle(
                image,
                (x, 180),
                (x + 38, 215),
                (153, 153, 153),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text,
                (x + 4, 204),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.line(image, (x + 19, 180), (point_x, 120), (255, 0, 0), 1)

        candidates = sorted(detect_labels(image), key=lambda item: item.box[0])

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate.traced for candidate in candidates))
        self.assertTrue(all(candidate.box[2] < 50 for candidate in candidates))

    def test_top_clipped_white_label_uses_light_fill_fallback(self) -> None:
        image = _scan_image()
        cv2.rectangle(image, (40, -8), (110, 25), (246, 248, 247), -1)
        cv2.rectangle(
            image,
            (40, -8),
            (110, 25),
            (198, 194, 196),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "0.4",
            (54, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        expected_endpoint = (230, 150)
        cv2.line(image, (75, 25), expected_endpoint, (255, 0, 0), 1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].box[1], 0)
        self.assertEqual(candidates[0].label_color, "white")
        self.assertLessEqual(
            math.dist(candidates[0].point_xy, expected_endpoint), 5.0
        )

    def test_light_neutral_white_label_is_detected_from_text_and_outline(self) -> None:
        image = _scan_image()
        cv2.rectangle(image, (40, 45), (110, 77), (246, 248, 247), -1)
        cv2.rectangle(
            image,
            (40, 45),
            (110, 77),
            (198, 194, 196),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "-0.2",
            (47, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        expected_endpoint = (230, 150)
        cv2.line(image, (110, 61), expected_endpoint, (255, 0, 0), 1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].label_color, "white")
        self.assertTrue(candidates[0].traced)
        self.assertLessEqual(
            math.dist(candidates[0].point_xy, expected_endpoint), 5.0
        )

    def test_small_opening_in_white_label_outline_is_closed(self) -> None:
        image = _boundary_scan_image()
        cv2.rectangle(image, (30, 40), (120, 72), (246, 248, 247), -1)
        cv2.rectangle(
            image,
            (30, 40),
            (120, 72),
            (198, 194, 196),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "-0.2",
            (38, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        # 리더가 테두리를 가르며 나가는 export를 3px 열린 외곽선으로 재현한다.
        cv2.rectangle(image, (74, 69), (76, 74), (255, 255, 255), -1)
        expected_contact = (180, 120)
        cv2.line(image, (75, 72), expected_contact, (255, 0, 0), 1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].label_color, "white")
        self.assertTrue(candidates[0].traced)
        self.assertLessEqual(
            math.dist(candidates[0].point_xy, expected_contact), 8.0
        )

    def test_neutral_close_keeps_dense_white_labels_separate(self) -> None:
        image = np.full((300, 500, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 100), (480, 280), (20, 180, 120), -1)
        expected_points: list[tuple[int, int]] = []
        for x, text in ((60, "0.2"), (133, "-0.4")):
            cv2.rectangle(image, (x, 40), (x + 70, 72), (246, 248, 247), -1)
            cv2.rectangle(
                image,
                (x, 40),
                (x + 70, 72),
                (198, 194, 196),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text,
                (x + 8, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            expected_point = (x + 35, 100)
            expected_points.append(expected_point)
            cv2.line(image, (x + 35, 72), expected_point, (255, 0, 0), 1)

        candidates = sorted(detect_labels(image), key=lambda item: item.box[0])

        self.assertEqual(len(candidates), 2)
        for candidate, expected_point in zip(candidates, expected_points):
            self.assertEqual(candidate.label_color, "white")
            self.assertTrue(candidate.traced)
            self.assertLessEqual(
                math.dist(candidate.point_xy, expected_point), 5.0
            )

    def test_gray_label_and_bent_leader_reach_marker(self) -> None:
        image = _scan_image()
        cv2.circle(image, (430, 230), 35, (255, 0, 0), -1)
        cv2.rectangle(image, (40, 45), (110, 75), (155, 155, 155), 2)
        cv2.putText(
            image,
            "-1.2",
            (48, 67),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
        )
        cv2.line(image, (110, 60), (190, 60), (255, 0, 0), 1)
        cv2.line(image, (190, 60), (280, 165), (255, 0, 0), 1)
        marker_center = (283, 169)
        cv2.circle(image, marker_center, 4, (20, 180, 120), -1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].traced)
        self.assertLessEqual(math.dist(candidates[0].point_xy, marker_center), 5.0)

    def test_two_pixel_leader_does_not_follow_wide_blue_surface(self) -> None:
        image = _scan_image()
        cv2.circle(image, (330, 190), 35, (255, 0, 0), -1)
        cv2.rectangle(image, (40, 45), (110, 75), (155, 155, 155), 2)
        surface_contact = (295, 180)
        cv2.line(image, (110, 60), surface_contact, (255, 0, 0), 2)

        candidate = detect_labels(image)[0]

        self.assertTrue(candidate.traced)
        self.assertLessEqual(math.dist(candidate.point_xy, surface_contact), 8.0)

    def test_line_beyond_allowed_gap_is_not_assigned(self) -> None:
        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 40), (90, 70), (0, 0, 0), 2)
        cv2.line(image, (98, 55), (220, 55), (255, 0, 0), 1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].traced)
        self.assertIsNone(candidates[0].point_xy)

    def test_connected_blue_line_without_scan_has_no_point(self) -> None:
        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        _draw_gray_label(image, (20, 40), (90, 70))
        cv2.line(image, (90, 55), (220, 55), (255, 0, 0), 1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].traced)
        self.assertIsNone(candidates[0].point_xy)

    def test_independent_blue_decoration_is_not_assigned_across_box_gap(self) -> None:
        image = _boundary_scan_image()
        _draw_gray_label(image, (30, 40), (120, 72))
        # 장식선은 스캔과 12px 이내지만 라벨 테두리와 연결되지 않았다.
        cv2.line(image, (125, 56), (167, 120), (255, 0, 0), 1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].traced)
        self.assertIsNone(candidates[0].point_xy)

    def test_leader_that_stops_in_white_background_has_no_point(self) -> None:
        image = _boundary_scan_image()
        _draw_gray_label(image, (30, 40), (120, 72))
        cv2.line(image, (120, 56), (155, 90), (255, 0, 0), 1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].traced)
        self.assertIsNone(candidates[0].point_xy)

    def test_short_gap_to_scan_is_snapped_to_scan_pixel(self) -> None:
        image = _boundary_scan_image()
        _draw_gray_label(image, (30, 40), (120, 72))
        expected_contact = (180, 120)
        cv2.line(image, (120, 56), (175, 117), (255, 0, 0), 1)

        scan_mask = build_scan_mask(image)
        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].traced)
        self.assertIsNotNone(candidates[0].point_xy)
        x, y = candidates[0].point_xy
        self.assertGreater(scan_mask[y, x], 0)
        self.assertLessEqual(math.dist((x, y), expected_contact), 8.0)

    def test_compact_short_leader_directly_on_scan_is_not_discarded(self) -> None:
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 60), (380, 240), (20, 180, 120), -1)
        cv2.rectangle(image, (100, 140), (146, 176), (0, 0, 255), -1)
        cv2.rectangle(
            image,
            (100, 140),
            (146, 176),
            (153, 153, 153),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "-0.5",
            (104, 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(image, (147, 157), (150, 159), (255, 0, 0), -1)

        scan_mask = build_scan_mask(image)
        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].traced)
        self.assertIsNotNone(candidates[0].point_xy)
        x, y = candidates[0].point_xy
        self.assertGreater(scan_mask[y, x], 0)
        self.assertLessEqual(math.dist((x, y), (150, 158)), 5.0)

    def test_internal_hole_marker_on_blue_surface_recovers_white_label(self) -> None:
        image = np.full((360, 600, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (80, 60), (540, 310), (255, 0, 0), -1)
        cv2.rectangle(image, (180, 100), (440, 203), (255, 255, 255), -1)
        cv2.rectangle(image, (270, 212), (340, 244), (246, 248, 247), -1)
        cv2.rectangle(
            image,
            (270, 212),
            (340, 244),
            (198, 194, 196),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "-0.3",
            (278, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        expected_marker = (305, 208)
        cv2.rectangle(image, (300, 205), (310, 211), (90, 90, 90), -1)

        scan_mask = build_scan_mask(image)
        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].label_color, "white")
        self.assertTrue(candidates[0].traced)
        self.assertIsNotNone(candidates[0].point_xy)
        x, y = candidates[0].point_xy
        self.assertGreater(scan_mask[y, x], 0)
        self.assertLessEqual(math.dist((x, y), expected_marker), 4.0)

    def test_white_label_on_blue_surface_without_marker_is_not_connected(self) -> None:
        image = np.full((360, 600, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (80, 60), (540, 310), (255, 0, 0), -1)
        cv2.rectangle(image, (180, 100), (440, 203), (255, 255, 255), -1)
        cv2.rectangle(image, (270, 212), (340, 244), (246, 248, 247), -1)
        cv2.rectangle(
            image,
            (270, 212),
            (340, 244),
            (198, 194, 196),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "-0.3",
            (278, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].traced)
        self.assertIsNone(candidates[0].point_xy)

    def test_diagonal_dark_scan_groove_is_not_a_point_marker(self) -> None:
        image = np.full((360, 600, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (80, 60), (540, 310), (255, 0, 0), -1)
        cv2.rectangle(image, (180, 100), (440, 203), (255, 255, 255), -1)
        cv2.rectangle(image, (270, 212), (340, 244), (246, 248, 247), -1)
        cv2.rectangle(
            image,
            (270, 212),
            (340, 244),
            (198, 194, 196),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "-0.3",
            (278, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        # 중앙을 지나더라도 비스듬히 이어지는 표면 홈은 수직 점 마커가 아니다.
        cv2.line(image, (293, 204), (317, 211), (90, 90, 90), 2)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].traced)
        self.assertIsNone(candidates[0].point_xy)

    def test_export_marker_gaps_up_to_twelve_pixels_are_snapped(self) -> None:
        for endpoint_x in (173, 170, 167):
            with self.subTest(endpoint_x=endpoint_x):
                image = _boundary_scan_image()
                _draw_gray_label(image, (30, 40), (120, 72))
                expected_contact = (180, 120)
                cv2.line(
                    image,
                    (120, 56),
                    (endpoint_x, 120),
                    (255, 0, 0),
                    1,
                )

                scan_mask = build_scan_mask(image)
                candidates = detect_labels(image)

                self.assertEqual(len(candidates), 1)
                self.assertTrue(candidates[0].traced)
                self.assertIsNotNone(candidates[0].point_xy)
                x, y = candidates[0].point_xy
                self.assertGreater(scan_mask[y, x], 0)
                self.assertLessEqual(math.dist((x, y), expected_contact), 8.0)

    def test_scan_touching_component_beats_long_background_component(self) -> None:
        image = _boundary_scan_image()
        _draw_gray_label(image, (90, 125), (155, 157))
        expected_contact = (180, 141)
        cv2.line(image, (155, 141), expected_contact, (255, 0, 0), 1)
        cv2.line(image, (90, 141), (10, 60), (255, 0, 0), 1)

        scan_mask = build_scan_mask(image)
        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].traced)
        x, y = candidates[0].point_xy
        self.assertGreater(scan_mask[y, x], 0)
        self.assertLessEqual(math.dist((x, y), expected_contact), 8.0)

    def test_thin_blue_scan_feature_does_not_pull_point_past_contact(self) -> None:
        image = _boundary_scan_image()
        _draw_gray_label(image, (30, 40), (120, 72))
        expected_contact = (180, 120)
        cv2.line(image, (120, 56), expected_contact, (255, 0, 0), 1)
        cv2.line(image, expected_contact, (420, 120), (255, 0, 0), 1)

        scan_mask = build_scan_mask(image)
        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].traced)
        x, y = candidates[0].point_xy
        self.assertGreater(scan_mask[y, x], 0)
        self.assertLessEqual(math.dist((x, y), expected_contact), 10.0)

    def test_internal_label_keeps_the_drawn_endpoint_without_inward_shift(self) -> None:
        image = np.full((300, 500, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (480, 280), (20, 180, 120), -1)
        _draw_gray_label(image, (210, 70), (300, 102), "-1.8")
        expected_endpoint = (175, 130)
        cv2.line(image, (210, 86), expected_endpoint, (255, 0, 0), 1)

        candidate = detect_labels(image)[0]

        self.assertTrue(candidate.traced)
        self.assertIsNotNone(candidate.point_xy)
        self.assertLessEqual(
            math.dist(candidate.point_xy, expected_endpoint),
            1.5,
            msg=candidate.point_xy,
        )

    def test_straight_scan_feature_does_not_run_to_opposite_boundary(self) -> None:
        image = _boundary_scan_image()
        _draw_gray_label(image, (30, 40), (120, 72))
        expected_contact = (180, 120)
        cv2.line(image, (120, 56), expected_contact, (255, 0, 0), 1)
        cv2.line(image, expected_contact, (300, 248), (255, 0, 0), 1)

        scan_mask = build_scan_mask(image)
        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].traced)
        x, y = candidates[0].point_xy
        self.assertGreater(scan_mask[y, x], 0)
        self.assertLessEqual(math.dist((x, y), expected_contact), 10.0)

    def test_crossing_leaders_for_dense_labels_keep_both_points(self) -> None:
        image = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (180, 100), (500, 270), (20, 180, 120), -1)
        _draw_gray_label(image, (60, 320), (150, 352), "-1.2")
        _draw_gray_label(image, (155, 320), (245, 352), "+0.8")
        expected_points = [(360, 180), (220, 180)]
        cv2.line(image, (105, 320), expected_points[0], (255, 0, 0), 1)
        cv2.line(image, (200, 320), expected_points[1], (255, 0, 0), 1)

        scan_mask = build_scan_mask(image)
        candidates = sorted(detect_labels(image), key=lambda item: item.box[0])

        self.assertEqual(len(candidates), 2)
        for candidate, expected in zip(candidates, expected_points):
            self.assertTrue(candidate.traced)
            self.assertIsNotNone(candidate.point_xy)
            x, y = candidate.point_xy
            self.assertGreater(scan_mask[y, x], 0)
            self.assertLessEqual(math.dist((x, y), expected), 10.0)

    def test_crossing_thin_and_antialiased_leaders_keep_both_points(self) -> None:
        image = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (180, 100), (500, 200), (20, 180, 120), -1)
        _draw_gray_label(image, (60, 320), (150, 352), "-1.2")
        _draw_gray_label(image, (155, 320), (245, 352), "+0.8")
        expected_points = [(360, 180), (220, 180)]
        cv2.line(
            image,
            (200, 320),
            expected_points[1],
            (255, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(image, (105, 320), expected_points[0], (255, 0, 0), 1)

        scan_mask = build_scan_mask(image)
        candidates = sorted(detect_labels(image), key=lambda item: item.box[0])

        self.assertEqual(len(candidates), 2)
        for candidate, expected in zip(candidates, expected_points):
            self.assertTrue(candidate.traced)
            self.assertIsNotNone(candidate.point_xy)
            x, y = candidate.point_xy
            self.assertGreater(scan_mask[y, x], 0)
            self.assertLessEqual(
                math.dist((x, y), expected),
                10.0,
                msg=(candidate.box, candidate.point_xy, expected),
            )

    def test_empty_dark_rectangle_on_scan_is_not_a_label(self) -> None:
        image = _scan_image()
        cv2.rectangle(image, (220, 120), (290, 150), (0, 0, 0), 2)

        self.assertEqual(detect_labels(image), [])

    def test_light_rectangle_without_number_or_leader_is_not_a_label(self) -> None:
        image = _scan_image()
        cv2.rectangle(image, (40, 45), (110, 77), (246, 248, 247), -1)
        cv2.rectangle(image, (40, 45), (110, 77), (198, 194, 196), 2)

        self.assertEqual(detect_labels(image), [])


class IntegrationHelperTest(unittest.TestCase):
    def test_legacy_batch_reader_without_batch_size_keyword(self) -> None:
        class LegacyBatchReader:
            def read_values(self, crops):
                return [1.25 for _ in crops]

        crops = [Image.new("RGB", (20, 10)), Image.new("RGB", (20, 10))]

        values = _read_label_values(LegacyBatchReader(), crops, batch_size=8)

        self.assertEqual(values, [1.25, 1.25])

    def test_number_parser_normalizes_unicode_spacing_and_width(self) -> None:
        self.assertEqual(LabelValueReader._parse_number("−\u00a0.5 mm"), -0.5)
        self.assertEqual(LabelValueReader._parse_number("＋１．２５"), 1.25)

    def test_unicode_image_path_round_trip(self) -> None:
        source = np.full((12, 16, 3), (10, 20, 30), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "측정 결과.png"
            write_image(path, source)
            restored = read_image(path)

        np.testing.assert_array_equal(restored, source)

    def test_colorbar_end_crop_is_not_empty_with_large_margin(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        vertical = _end_crop(image, (5, 5, 10, 10), "min", 30)
        horizontal = _end_crop(image, (5, 5, 20, 10), "max", 30)

        self.assertGreater(vertical.size, 0)
        self.assertGreater(horizontal.size, 0)


if __name__ == "__main__":
    unittest.main()
