"""UI 백엔드가 Qwen 장애 중에도 편차 포인트를 보존하는지 검증한다."""

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


class UiBackendFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        backend_server._reader = None
        self.image = np.full((80, 120, 3), 180, dtype=np.uint8)
        self.candidates = [
            _candidate((5, 5, 30, 16), (50, 40)),
            _candidate((70, 8, 30, 16), (90, 55)),
        ]

    def tearDown(self) -> None:
        backend_server._reader = None

    def _analyze_with_reader(self, reader) -> dict:
        reader_patch = (
            patch.object(backend_server, "_get_qwen_reader", side_effect=reader)
            if isinstance(reader, Exception)
            else patch.object(backend_server, "_get_qwen_reader", return_value=reader)
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
            reader_patch,
            patch.object(backend_server, "_point_value", side_effect=[-0.2, 0.3]),
        ):
            return backend_server.analyze_image(self.image, "synthetic.png")

    def test_reader_initialization_failure_keeps_all_candidates(self) -> None:
        result = self._analyze_with_reader(FileNotFoundError("model unavailable"))

        self.assertEqual(len(result["points"]), 2)
        self.assertEqual([point["id"] for point in result["points"]], ["P-01", "P-02"])
        self.assertEqual(
            [(point["xPx"], point["yPx"]) for point in result["points"]],
            [(50, 40), (90, 55)],
        )
        self.assertNotIn("deviation", result["errors"])
        self.assertEqual(result["stats"]["pointsDetected"], 2)
        self.assertEqual(result["stats"]["qwenReads"], 0)
        self.assertEqual(result["stats"]["fallbackReads"], 2)
        self.assertTrue(
            all(
                point["confidence"] == "qwen_unavailable|colorbar_fallback"
                for point in result["points"]
            )
        )
        self.assertTrue(result["warningsByEngine"]["deviation"])

    def test_short_batch_result_falls_back_without_dropping_candidates(self) -> None:
        result = self._analyze_with_reader(_Reader([1.25]))

        self.assertEqual(len(result["points"]), 2)
        self.assertEqual([point["value"] for point in result["points"]], [1.25, -0.2])
        self.assertEqual(result["stats"]["qwenReads"], 1)
        self.assertEqual(result["stats"]["fallbackReads"], 1)
        self.assertNotIn("deviation", result["errors"])

    def test_batch_read_failure_falls_back_without_dropping_candidates(self) -> None:
        result = self._analyze_with_reader(_Reader(RuntimeError("inference failed")))

        self.assertEqual(len(result["points"]), 2)
        self.assertEqual([point["value"] for point in result["points"]], [-0.2, 0.3])
        self.assertEqual(result["stats"]["qwenReads"], 0)
        self.assertEqual(result["stats"]["fallbackReads"], 2)
        self.assertNotIn("deviation", result["errors"])


if __name__ == "__main__":
    unittest.main()
