"""UI 백엔드가 실제 Qwen 판독값만 편차 포인트로 반환하는지 검증한다."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        # 판독 캐시는 모듈 전역이라 테스트 사이에 넘어간다. 같은 합성
        # 이미지를 쓰는 테스트가 앞 테스트의 값을 읽어 버려 스텁이 아예
        # 안 불리는 일이 생긴다 — 실제로 10건이 그렇게 깨졌다.
        backend_server.reset_label_cache()
        self.image = np.full((80, 120, 3), 180, dtype=np.uint8)
        self.candidates = [
            _candidate((5, 5, 30, 16), (50, 40)),
            _candidate((70, 8, 30, 16), (90, 55)),
        ]

    def tearDown(self) -> None:
        backend_server._reader = None
        backend_server.reset_label_cache()

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
                # 세 번 다 같은 그림을 쓰므로 판독 캐시가 첫 결과를
                # 돌려준다. 값마다 따로 확인하려면 매번 비워야 한다.
                backend_server.reset_label_cache()
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


if __name__ == "__main__":
    unittest.main()


class ZeroLineEditTest(unittest.TestCase):
    """보정시트에서 손본 제로라인이 3D 로 그대로 간다."""

    def test_옮김은_광선을_쏘기_전에_더해진다(self) -> None:
        lines = [{"line_id": 1, "points": [[10.0, 10.0], [20.0, 10.0]]},
                 {"line_id": 2, "points": [[30.0, 30.0], [40.0, 30.0]]}]
        edits = [{"index": 0, "dx": 5.0, "dy": -3.0}]
        moved = backend_server.apply_zero_edits(lines, edits)
        self.assertEqual(moved[0]["points"], [[15.0, 7.0], [25.0, 7.0]])
        self.assertEqual(moved[1]["points"], lines[1]["points"],
                         "손대지 않은 선은 그대로여야 한다")

    def test_숨긴_선은_빠진다(self) -> None:
        lines = [{"line_id": 1, "points": [[0.0, 0.0], [1.0, 1.0]]},
                 {"line_id": 2, "points": [[2.0, 2.0], [3.0, 3.0]]}]
        left = backend_server.apply_zero_edits(
            lines, [{"index": 0, "hidden": True}])
        self.assertEqual(len(left), 1)
        self.assertEqual(left[0]["line_id"], 2)

    def test_손댄_것이_없으면_그대로다(self) -> None:
        lines = [{"line_id": 1, "points": [[0.0, 0.0], [1.0, 1.0]]}]
        self.assertEqual(backend_server.apply_zero_edits(lines, None), lines)
        self.assertEqual(backend_server.apply_zero_edits(lines, []), lines)

    def test_없는_번호는_무시한다(self) -> None:
        lines = [{"line_id": 1, "points": [[0.0, 0.0], [1.0, 1.0]]}]
        left = backend_server.apply_zero_edits(
            lines, [{"index": 7, "dx": 100.0}])
        self.assertEqual(left, lines)


class OverlayCacheTest(unittest.TestCase):
    """같은 짝을 다시 얹을 때 다시 계산하지 않는다."""

    def setUp(self) -> None:
        backend_server.reset_overlay_cache()

    def tearDown(self) -> None:
        backend_server.reset_overlay_cache()

    def test_손본_내역이_다르면_다른_열쇠다(self) -> None:
        same = backend_server._overlay_key("c", "a", [{"index": 0, "dx": 1}])
        again = backend_server._overlay_key("c", "a", [{"index": 0, "dx": 1}])
        other = backend_server._overlay_key("c", "a", [{"index": 0, "dx": 2}])
        self.assertEqual(same, again)
        self.assertNotEqual(same, other)

    def test_빈_내역과_없는_내역은_같은_열쇠다(self) -> None:
        self.assertEqual(backend_server._overlay_key("c", "a", None),
                         backend_server._overlay_key("c", "a", []))

    def test_CAD_가_만료되면_캐시를_보기_전에_알린다(self) -> None:
        with self.assertRaises(ValueError) as caught:
            backend_server.cad_overlay_for("없는CAD", "없는분석")
        self.assertIn("CAD", str(caught.exception))


class LabelStoreTest(unittest.TestCase):
    """판독 결과를 디스크에도 남긴다 — 엔진을 다시 띄워도 안 잃는다."""

    def setUp(self) -> None:
        # 진짜 캐시 파일은 건드리지 않는다.
        #
        # 처음에는 백업했다가 되돌리는 식으로 짰는데, 그 사이에 시험이
        # 두 항목짜리 파일을 써 버려서 실제로 데워 둔 캐시가 날아갔다 —
        # 다음 분석이 Qwen 을 71초 다시 돌았다. 시험은 시험용 파일만
        # 쓴다.
        backend_server.reset_label_cache()
        self._real = backend_server._LABEL_STORE
        self._temp = tempfile.TemporaryDirectory()
        backend_server._LABEL_STORE = (
            Path(self._temp.name) / "label_cache.json")

    def tearDown(self) -> None:
        backend_server.reset_label_cache()
        backend_server._LABEL_STORE = self._real
        self._temp.cleanup()

    @property
    def _store(self):
        return backend_server._LABEL_STORE

    def test_남겼다가_다시_읽는다(self) -> None:
        backend_server._remember_labels(["가", "나"], [1.5, None])
        backend_server.reset_label_cache()
        self.assertEqual(len(backend_server._label_cache), 0)
        backend_server._load_label_store()
        self.assertEqual(backend_server._label_cache.get("가"), 1.5)
        self.assertIsNone(backend_server._label_cache.get("나"))
        self.assertIn("나", backend_server._label_cache,
                      "못 읽었다는 사실도 기억해야 다시 안 읽는다")

    def test_파일이_깨져도_안_터진다(self) -> None:
        self._store.write_text("{망가진", encoding="utf-8")
        backend_server._load_label_store()      # 예외가 나면 안 된다
        self.assertEqual(len(backend_server._label_cache), 0)

    def test_같은_그림은_같은_열쇠다(self) -> None:
        from PIL import Image

        left = Image.fromarray(np.full((12, 20, 3), 128, np.uint8))
        right = Image.fromarray(np.full((12, 20, 3), 128, np.uint8))
        other = Image.fromarray(np.full((12, 20, 3), 129, np.uint8))
        self.assertEqual(backend_server._crop_key(left),
                         backend_server._crop_key(right))
        self.assertNotEqual(backend_server._crop_key(left),
                            backend_server._crop_key(other))
