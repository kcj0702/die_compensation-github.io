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
from deviation_extraction.label_detector import detect_labels
from deviation_extraction.point_extractor import _read_label_values
from deviation_extraction.vlm_reader import LabelValueReader


def _scan_image() -> np.ndarray:
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (360, 230), (190, 120), 0, 0, 360, (20, 180, 120), -1)
    return image


class ConnectedLeaderTest(unittest.TestCase):
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
        expected_endpoint = (220, 130)
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
        expected_endpoint = (220, 130)
        cv2.line(image, (110, 61), expected_endpoint, (255, 0, 0), 1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].label_color, "white")
        self.assertTrue(candidates[0].traced)
        self.assertLessEqual(
            math.dist(candidates[0].point_xy, expected_endpoint), 5.0
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
