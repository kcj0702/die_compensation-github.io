"""회사 원본 없이 편차값·포인트 좌표 추출 계약을 검증한다."""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT_ROOT / "deviation_extraction"
sys.path.insert(0, str(MODULE_DIR))

import label_detector  # noqa: E402
import point_extractor  # noqa: E402
import vlm_reader  # noqa: E402
from label_detector import LabelCandidate, detect_labels  # noqa: E402
from point_extractor import extract_points, save_csv, save_debug_image  # noqa: E402
from vlm_reader import LabelValueReader  # noqa: E402


class FakeReader:
    def __init__(self, value: float | None):
        self.value = value
        self.crop_modes: list[str] = []

    def read_value(self, crop) -> float | None:
        self.crop_modes.append(crop.mode)
        return self.value


def _synthetic_map() -> tuple[np.ndarray, tuple[int, int]]:
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 40), (90, 70), (0, 0, 0), 2)
    endpoint = (210, 125)
    cv2.line(image, (90, 55), endpoint, (255, 0, 0), 2)
    return image, endpoint


class LabelDetectorTest(unittest.TestCase):
    def test_blue_line_touching_box_does_not_merge_label(self) -> None:
        image, endpoint = _synthetic_map()

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].traced)
        self.assertIsNotNone(candidates[0].point_xy)
        error = math.dist(candidates[0].point_xy, endpoint)
        self.assertLessEqual(error, 3.0)

    def test_red_labels_are_detected_in_stable_order(self) -> None:
        image = np.full((180, 240, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (120, 100), (190, 125), (0, 0, 255), -1)
        cv2.rectangle(image, (20, 20), (90, 45), (0, 0, 255), -1)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 2)
        self.assertLess(candidates[0].box[1], candidates[1].box[1])
        self.assertEqual([item.label_color for item in candidates], ["red", "red"])

    def test_longest_valid_hough_segment_is_selected(self) -> None:
        lines = np.array(
            [
                [[50, 20, 80, 20]],
                [[50, 20, 180, 20]],
                [[50, 20, 400, 20]],
                [[0, 0, 200, 0]],
            ],
            dtype=np.int32,
        )
        with patch.object(label_detector.cv2, "HoughLinesP", return_value=lines):
            endpoint = label_detector._trace_leader_line(
                np.zeros((100, 220), dtype=np.uint8),
                (10, 10, 40, 20),
            )

        self.assertEqual(endpoint, (180, 20))

    def test_missing_line_has_no_placeholder_coordinate(self) -> None:
        image = np.full((100, 140, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 30), (90, 60), (0, 0, 0), 2)

        candidates = detect_labels(image)

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].traced)
        self.assertIsNone(candidates[0].point_xy)


class PointExtractorTest(unittest.TestCase):
    def test_value_coordinate_and_resized_mask_are_combined(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "map.png"
            mask_path = temp_path / "mask.png"
            cv2.imwrite(str(image_path), np.full((120, 160, 3), 255, dtype=np.uint8))

            mask = np.zeros((60, 80), dtype=np.uint8)
            mask[29:32, 39:42] = 255
            cv2.imwrite(str(mask_path), mask)

            candidate = LabelCandidate(
                box=(10, 10, 50, 20),
                point_xy=(80, 60),
                label_color="white",
                traced=True,
            )
            reader = FakeReader(-1.25)
            with patch.object(point_extractor, "detect_labels", return_value=[candidate]):
                points = extract_points(image_path, mask_path, reader=reader)

        self.assertEqual(len(points), 1)
        self.assertEqual((points[0].x_px, points[0].y_px), (80, 60))
        self.assertEqual((points[0].x_norm, points[0].y_norm), (0.5, 0.5))
        self.assertEqual(points[0].value_mm, -1.25)
        self.assertTrue(points[0].in_zero_line)
        self.assertEqual(points[0].confidence, "ok")
        self.assertEqual(reader.crop_modes, ["RGB"])

    def test_failures_are_combined_without_fake_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "map.png"
            cv2.imwrite(str(image_path), np.full((80, 120, 3), 255, dtype=np.uint8))
            candidate = LabelCandidate(
                box=(10, 10, 50, 20),
                point_xy=None,
                label_color="red",
                traced=False,
            )
            with patch.object(point_extractor, "detect_labels", return_value=[candidate]):
                points = extract_points(
                    image_path,
                    temp_path / "missing-mask.png",
                    reader=FakeReader(None),
                )

        self.assertIsNone(points[0].x_px)
        self.assertIsNone(points[0].x_norm)
        self.assertEqual(
            points[0].confidence,
            "value_not_read|leader_line_not_traced",
        )

    def test_no_candidates_does_not_load_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "map.png"
            cv2.imwrite(str(image_path), np.full((40, 60, 3), 255, dtype=np.uint8))
            with patch.object(point_extractor, "detect_labels", return_value=[]):
                points = extract_points(
                    image_path,
                    Path(temp_dir) / "missing-mask.png",
                    reader_factory=lambda: self.fail("reader must stay lazy"),
                )

        self.assertEqual(points, [])

    def test_empty_csv_keeps_public_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "empty.csv"
            save_csv([], csv_path)
            content = csv_path.read_text(encoding="utf-8-sig")

        self.assertEqual(content.splitlines()[0], ",".join(point_extractor.CSV_COLUMNS))

    def test_debug_image_accepts_missing_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "map.png"
            output_path = temp_path / "debug.png"
            cv2.imwrite(str(image_path), np.full((80, 120, 3), 255, dtype=np.uint8))
            point = point_extractor.DeviationPoint(
                point_id="P001",
                x_px=None,
                y_px=None,
                x_norm=None,
                y_norm=None,
                value_mm=None,
                label_color="white",
                in_zero_line=False,
                confidence="value_not_read|leader_line_not_traced",
                label_box=(10, 10, 50, 20),
            )

            save_debug_image(image_path, [point], output_path)

            self.assertTrue(output_path.is_file())


class ValueReaderTest(unittest.TestCase):
    def test_number_parser(self) -> None:
        self.assertEqual(LabelValueReader._parse_number("result: -1.25 mm"), -1.25)
        self.assertEqual(LabelValueReader._parse_number("+3"), 3.0)
        self.assertIsNone(LabelValueReader._parse_number("판독 실패"))

    def test_offline_mode_is_forwarded_to_model_loaders(self) -> None:
        processor = MagicMock()
        model = MagicMock()
        with (
            patch.object(vlm_reader, "require_version"),
            patch.object(
                vlm_reader.AutoProcessor,
                "from_pretrained",
                return_value=processor,
            ) as processor_loader,
            patch.object(
                vlm_reader.AutoModelForImageTextToText,
                "from_pretrained",
                return_value=model,
            ) as model_loader,
        ):
            LabelValueReader(
                model_id="local/demo-model",
                device="cpu",
                local_files_only=True,
            )

        self.assertTrue(processor_loader.call_args.kwargs["local_files_only"])
        self.assertTrue(model_loader.call_args.kwargs["local_files_only"])


class RunCliTest(unittest.TestCase):
    def test_zero_labels_write_header_and_exit_two_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "blank.png"
            csv_path = temp_path / "points.csv"
            cv2.imwrite(str(image_path), np.full((60, 100, 3), 255, dtype=np.uint8))

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "deviation_extraction/run.py",
                    "--image",
                    str(image_path),
                    "--out",
                    str(csv_path),
                    "--offline",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )

            header = csv_path.read_text(encoding="utf-8-sig").splitlines()[0]

        self.assertEqual(result.returncode, 2)
        self.assertEqual(header, ",".join(point_extractor.CSV_COLUMNS))


if __name__ == "__main__":
    unittest.main()
