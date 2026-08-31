"""UI 백엔드가 실제 Qwen 판독값만 편차 포인트로 반환하는지 검증한다."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "ui" / "backend" / "server.py"
SPEC = importlib.util.spec_from_file_location("ajin_ui_backend_server", SERVER_PATH)
assert SPEC is not None and SPEC.loader is not None
backend_server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend_server)

from label_detector import LabelCandidate  # noqa: E402


def _candidate(box: tuple[int, int, int, int], point: tuple[int, int]) -> LabelCandidate:
    return LabelCandidate(box=box, point_xy=point, label_color="white", traced=True)


class _Reader:
    def __init__(self, values: list[float | None] | Exception):
        self.values = values

    def read_values(self, _crops):
        if isinstance(self.values, Exception):
            raise self.values
        return self.values


class _ScriptedReader:
    def __init__(self, responses: list[list[float | None] | Exception]):
        self.responses = list(responses)
        self.calls: list[list[object]] = []

    def read_values(self, crops):
        self.calls.append(list(crops))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FocusedReader(_ScriptedReader):
    def __init__(
        self,
        responses: list[list[float | None] | Exception],
        focused_responses: list[float | None | Exception],
    ):
        super().__init__(responses)
        self.focused_responses = list(focused_responses)
        self.focused_calls: list[object] = []

    def read_value_focused(self, crop):
        self.focused_calls.append(crop)
        response = self.focused_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class UiBackendModelDiscoveryTest(unittest.TestCase):
    def test_find_qwen_model_discovers_complete_workspace_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "Qwen2.5-VL-3B-Instruct"
            cache_dir = root / "empty-cache"
            model_dir.mkdir()
            cache_dir.mkdir()
            for filename in ("config.json", "tokenizer.json"):
                (model_dir / filename).write_text("{}", encoding="utf-8")
            shard_name = "model-00001-of-00001.safetensors"
            (model_dir / shard_name).touch()
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.weight": shard_name}}),
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {"AJIN_QWEN_MODEL_PATH": ""}),
                patch.object(backend_server, "WORKSPACE_QWEN_DIR", model_dir),
                patch.object(backend_server, "QWEN_CACHE_DIR", cache_dir),
            ):
                found = backend_server._find_qwen_model()

        self.assertEqual(found, model_dir.resolve())


class UiBackendStrictReadingTest(unittest.TestCase):
    def setUp(self) -> None:
        backend_server._reader = None
        self.image = np.full((80, 120, 3), 180, dtype=np.uint8)
        self.candidates = [
            _candidate((5, 5, 30, 16), (50, 40)),
            _candidate((70, 8, 30, 16), (90, 55)),
        ]

    def tearDown(self) -> None:
        backend_server._reader = None

    def _analyze_with_reader(self, reader, *, scan_present: bool = True) -> dict:
        reader_patch = (
            patch.object(backend_server, "_get_qwen_reader", side_effect=reader)
            if isinstance(reader, Exception)
            else patch.object(backend_server, "_get_qwen_reader", return_value=reader)
        )
        scan_mask = np.full(
            self.image.shape[:2],
            255 if scan_present else 0,
            dtype=np.uint8,
        )
        with (
            patch.object(backend_server, "detect_label_boxes", return_value=[]),
            patch.object(
                backend_server,
                "create_versions",
                return_value={"2_labels_inpainted": self.image.copy()},
            ),
            patch.object(
                backend_server,
                "detect_zero_line",
                side_effect=RuntimeError("zero test disabled"),
            ),
            patch.object(backend_server, "detect_labels", return_value=self.candidates),
            patch.object(backend_server, "build_scan_mask", return_value=scan_mask),
            reader_patch,
        ):
            return backend_server.analyze_image(self.image, "synthetic.png")

    def test_reader_initialization_failure_returns_no_made_up_values(self) -> None:
        result = self._analyze_with_reader(FileNotFoundError("model unavailable"))

        self.assertEqual(result["points"], [])
        self.assertNotIn("deviation", result["errors"])
        self.assertEqual(result["stats"]["pointsDetected"], 0)
        self.assertEqual(result["stats"]["qwenReads"], 0)
        self.assertEqual(result["valueMode"], "판독 결과 없음")
        self.assertTrue(result["warningsByEngine"]["deviation"])

    def test_short_batch_result_retries_each_exact_crop_in_order(self) -> None:
        reader = _ScriptedReader([[99.0], [1.25], [-0.5]])

        result = self._analyze_with_reader(reader)

        self.assertEqual([point["value"] for point in result["points"]], [1.25, -0.5])
        self.assertEqual(
            [(point["xPx"], point["yPx"]) for point in result["points"]],
            [(50, 40), (90, 55)],
        )
        self.assertEqual([len(call) for call in reader.calls], [2, 1, 1])
        self.assertIs(reader.calls[1][0], reader.calls[0][0])
        self.assertIs(reader.calls[2][0], reader.calls[0][1])
        self.assertEqual(result["stats"]["qwenReads"], 2)
        self.assertNotIn("deviation", result["errors"])
        self.assertTrue(result["warningsByEngine"]["deviation"])

    def test_batch_exception_keeps_successful_singleton_read(self) -> None:
        reader = _ScriptedReader(
            [RuntimeError("batch failed"), [1.25], [None]]
        )

        result = self._analyze_with_reader(reader)

        self.assertEqual(len(result["points"]), 1)
        self.assertEqual(result["points"][0]["value"], 1.25)
        self.assertEqual(
            (result["points"][0]["xPx"], result["points"][0]["yPx"]),
            (50, 40),
        )
        self.assertEqual(result["stats"]["qwenReads"], 1)
        self.assertNotIn("deviation", result["errors"])

    def test_batch_none_is_not_retried_after_reader_internal_variants(self) -> None:
        reader = _ScriptedReader([[1.25, None]])

        result = self._analyze_with_reader(reader)

        self.assertEqual([point["value"] for point in result["points"]], [1.25])
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(len(reader.calls[0]), 2)
        self.assertEqual(result["stats"]["qwenReads"], 1)
        self.assertEqual(result["stats"]["qwenUnread"], 1)

    def test_batch_none_gets_bounded_focused_qwen_retry(self) -> None:
        reader = _FocusedReader([[1.25, None]], [-0.5])

        result = self._analyze_with_reader(reader)

        self.assertEqual(
            [point["value"] for point in result["points"]],
            [1.25, -0.5],
        )
        self.assertEqual([len(call) for call in reader.calls], [2])
        self.assertEqual(len(reader.focused_calls), 1)
        self.assertIs(reader.focused_calls[0], reader.calls[0][1])
        self.assertEqual(result["stats"]["qwenReads"], 2)
        self.assertEqual(result["stats"]["qwenUnread"], 0)

    def test_focused_retry_failure_does_not_change_other_qwen_value(self) -> None:
        reader = _FocusedReader([[1.25, None]], [RuntimeError("retry failed")])

        result = self._analyze_with_reader(reader)

        self.assertEqual([point["value"] for point in result["points"]], [1.25])
        self.assertEqual(result["stats"]["qwenReads"], 1)
        self.assertEqual(result["stats"]["qwenUnread"], 1)
        self.assertTrue(result["warningsByEngine"]["deviation"])

    def test_individual_failure_does_not_discard_another_crop(self) -> None:
        reader = _ScriptedReader(
            [RuntimeError("batch failed"), RuntimeError("first crop failed"), [-0.5]]
        )

        result = self._analyze_with_reader(reader)

        self.assertEqual(len(result["points"]), 1)
        self.assertEqual(result["points"][0]["value"], -0.5)
        self.assertEqual(
            (result["points"][0]["xPx"], result["points"][0]["yPx"]),
            (90, 55),
        )
        self.assertEqual([len(call) for call in reader.calls], [2, 1, 1])

    def test_invalid_singleton_length_is_excluded_without_shifting_mapping(self) -> None:
        reader = _ScriptedReader([[99.0], [], [-0.5]])

        result = self._analyze_with_reader(reader)

        self.assertEqual(len(result["points"]), 1)
        self.assertEqual(result["points"][0]["value"], -0.5)
        self.assertEqual(
            (result["points"][0]["xPx"], result["points"][0]["yPx"]),
            (90, 55),
        )

    def test_dense_label_crops_use_one_pixel_padding_without_overlap(self) -> None:
        for x in range(self.image.shape[1]):
            self.image[:, x, :] = x
        self.candidates = [
            _candidate((10, 10, 10, 8), (15, 30)),
            _candidate((22, 10, 10, 8), (27, 30)),
        ]
        reader = _ScriptedReader([[0.1, -0.2]])

        result = self._analyze_with_reader(reader)

        self.assertEqual(result["stats"]["qwenReads"], 2)
        self.assertEqual(len(reader.calls), 1)
        first_crop, second_crop = (np.asarray(crop) for crop in reader.calls[0])
        self.assertEqual(first_crop.shape[:2], (10, 12))
        self.assertEqual(second_crop.shape[:2], (10, 12))
        self.assertTrue(np.all(first_crop[:, -1, :] == 20))
        self.assertTrue(np.all(second_crop[:, 0, :] == 21))

    def test_only_successfully_read_labels_are_returned(self) -> None:
        result = self._analyze_with_reader(_Reader([1.25, None]))

        self.assertEqual(len(result["points"]), 1)
        self.assertEqual(result["points"][0]["id"], "P-01")
        self.assertEqual(result["points"][0]["value"], 1.25)
        self.assertEqual(
            (result["points"][0]["xPx"], result["points"][0]["yPx"]),
            (50, 40),
        )
        self.assertEqual(result["stats"]["qwenReads"], 1)
        self.assertEqual(result["stats"]["detectedCandidates"], 2)
        self.assertEqual(result["stats"]["validCandidates"], 2)
        self.assertEqual(result["stats"]["qwenUnread"], 1)
        self.assertTrue(result["warningsByEngine"]["deviation"])

    def test_nonfinite_or_malformed_value_does_not_discard_valid_point(self) -> None:
        for invalid_value in (float("nan"), float("inf"), "not-a-number"):
            with self.subTest(invalid_value=invalid_value):
                result = self._analyze_with_reader(_Reader([invalid_value, -0.5]))

                self.assertEqual(len(result["points"]), 1)
                self.assertEqual(result["points"][0]["value"], -0.5)
                self.assertEqual(
                    (result["points"][0]["xPx"], result["points"][0]["yPx"]),
                    (90, 55),
                )
                self.assertEqual(result["stats"]["qwenReads"], 1)
                self.assertEqual(result["stats"]["qwenUnread"], 1)
                self.assertNotIn("deviation", result["errors"])

    def test_scanless_image_returns_no_points(self) -> None:
        self.image = np.full((80, 120, 3), 255, dtype=np.uint8)

        result = self._analyze_with_reader(
            _Reader([1.25, -0.5]),
            scan_present=False,
        )

        self.assertEqual(result["points"], [])
        self.assertEqual(result["stats"]["qwenReads"], 0)
        self.assertIn("3D 스캔 본체", result["warningsByEngine"]["deviation"][0])


def _panel(width: int, height: int) -> np.ndarray:
    """비대칭 노치와 구멍 두 개를 가진 합성 판넬."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (width - 11, height - 11), (60, 200, 60), -1)
    cv2.rectangle(image, (10, 10), (10 + width // 5, 10 + height // 4), (255, 255, 255), -1)
    cv2.circle(image, (width // 3, height // 2), max(4, height // 10), (255, 255, 255), -1)
    cv2.circle(image, (2 * width // 3, height // 2), max(3, height // 16), (255, 255, 255), -1)
    return image


class UiBackendProductAlignmentTest(unittest.TestCase):
    """스캔 좌표가 제품데이터 좌표로 옮겨져 나오는지 검증한다."""

    def setUp(self) -> None:
        backend_server._reader = None
        self.product = _panel(200, 120)
        # 제품데이터를 180도 뒤집고 2배로 키운 가짜 스캔이라 정답 좌표를 알 수 있다.
        self.scan = cv2.resize(
            cv2.flip(self.product, -1), (400, 240), interpolation=cv2.INTER_NEAREST
        )
        self.candidates = [_candidate((5, 5, 30, 16), (100, 60))]

    def tearDown(self) -> None:
        backend_server._reader = None

    def _analyze(self, filename: str, product_upload: np.ndarray | None) -> dict:
        with (
            patch.object(backend_server, "detect_label_boxes", return_value=[]),
            patch.object(
                backend_server,
                "create_versions",
                return_value={"2_labels_inpainted": self.scan.copy()},
            ),
            patch.object(
                backend_server,
                "detect_zero_line",
                side_effect=RuntimeError("zero test disabled"),
            ),
            patch.object(backend_server, "detect_labels", return_value=self.candidates),
            patch.object(
                backend_server,
                "build_scan_mask",
                return_value=np.full(self.scan.shape[:2], 255, dtype=np.uint8),
            ),
            patch.object(
                backend_server, "_get_qwen_reader", return_value=_Reader([-0.7])
            ),
        ):
            return backend_server.analyze_image(
                self.scan, filename, product_upload
            )

    def test_uploaded_product_image_moves_points_onto_it(self) -> None:
        result = self._analyze("JD_64XX2-DR000 3D 스캔.png", self.product)

        self.assertEqual(result["partNumber"], "64XX2-DR000")
        self.assertIsNotNone(result["productImage"])
        self.assertIsNotNone(result["alignmentOverlay"])
        self.assertEqual(result["productSource"], "업로드한 이미지")
        self.assertTrue(result["alignment"]["flipX"])
        self.assertTrue(result["alignment"]["flipY"])
        self.assertTrue(result["alignment"]["confident"])

        self.assertEqual(len(result["points"]), 1)
        point = result["points"][0]
        # 스캔 (100, 60) -> 제품 ((399-100)/2, (239-60)/2) = (149.5, 89.5)
        self.assertAlmostEqual(point["xProduct"], 149.5 / 200 * 100, delta=2.0)
        self.assertAlmostEqual(point["yProduct"], 89.5 / 120 * 100, delta=2.0)
        self.assertEqual(result["stats"]["pointsTransferred"], 1)

    def test_registered_product_is_used_without_a_second_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            library = backend_server.ProductLibrary(Path(temp_dir))
            library.register("64XX2-DR000", self.product)
            with patch.object(backend_server, "PRODUCT_LIBRARY", library):
                result = self._analyze("JD_64XX2-DR000 3D 스캔.png", None)

        self.assertIsNotNone(result["productImage"])
        self.assertEqual(result["productSource"], "등록됨 · 64XX2-DR000")
        self.assertEqual(result["stats"]["pointsTransferred"], 1)

    def test_confirmed_orientation_is_reused_for_a_later_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            library = backend_server.ProductLibrary(root / "product")
            library.register("64XX2-DR000", self.product)
            store = backend_server.AlignmentStore(root / "alignment")

            with (
                patch.object(backend_server, "PRODUCT_LIBRARY", library),
                patch.object(backend_server, "ALIGNMENT_STORE", store),
            ):
                first = self._analyze("JD_64XX2-DR000 3D 스캔.png", None)
                saved = backend_server.Alignment.from_dict(first["alignment"])
                # 사람이 좌우만 반대로 확정했다고 가정한다.
                store.save(
                    "64XX2-DR000",
                    backend_server.estimate_alignment(
                        backend_server.build_part_silhouette(self.scan),
                        backend_server.build_product_mask(self.product),
                        flip_x=not saved.flip_x,
                        flip_y=saved.flip_y,
                    ),
                )
                second = self._analyze("JD_64XX2-DR000 3D 스캔.png", None)

        self.assertTrue(first["alignment"]["flipX"])
        self.assertFalse(second["alignment"]["flipX"])
        self.assertTrue(second["alignment"]["overridden"])
        self.assertTrue(
            any("확정 저장된 방향" in item for item in second["warningsByEngine"]["product"])
        )

    def test_missing_product_leaves_the_existing_result_intact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                backend_server,
                "PRODUCT_LIBRARY",
                backend_server.ProductLibrary(Path(temp_dir)),
            ):
                result = self._analyze("JD_64XX2-DR000 3D 스캔.png", None)

        self.assertIsNone(result["productImage"])
        self.assertIsNone(result["alignment"])
        self.assertEqual(result["stats"]["pointsTransferred"], 0)
        self.assertEqual(len(result["points"]), 1)
        self.assertNotIn("xProduct", result["points"][0])
        self.assertTrue(
            any("등록되어 있지" in item for item in result["warningsByEngine"]["product"])
        )

    def test_scan_without_a_part_number_is_not_treated_as_an_error(self) -> None:
        result = self._analyze("synthetic.png", None)

        self.assertIsNone(result["partNumber"])
        self.assertIsNone(result["productImage"])
        self.assertEqual(result["warningsByEngine"]["product"], [])
        self.assertNotIn("product", result["errors"])

    def test_flip_flag_is_tri_state(self) -> None:
        self.assertIsNone(backend_server._optional_flag({}, "flipX"))
        self.assertIsNone(backend_server._optional_flag({"flipX": ""}, "flipX"))
        self.assertTrue(backend_server._optional_flag({"flipX": "true"}, "flipX"))
        self.assertTrue(backend_server._optional_flag({"flipX": "1"}, "flipX"))
        self.assertFalse(backend_server._optional_flag({"flipX": "false"}, "flipX"))


if __name__ == "__main__":
    unittest.main()
