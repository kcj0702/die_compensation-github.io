"""UI 백엔드가 실제 Qwen 판독값만 편차 포인트로 반환하는지 검증한다."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "ui" / "backend" / "server.py"
SPEC = importlib.util.spec_from_file_location("ajin_ui_backend_server", SERVER_PATH)
assert SPEC is not None and SPEC.loader is not None
backend_server = importlib.util.module_from_spec(SPEC)
_SERVER_IMPORT_DB_DIR = tempfile.TemporaryDirectory()
with patch.dict(
    os.environ,
    {
        "AJIN_CORRECTION_DB_PATH": str(
            Path(_SERVER_IMPORT_DB_DIR.name) / "import-correction-history.db"
        )
    },
):
    SPEC.loader.exec_module(backend_server)

from label_detector import LabelCandidate  # noqa: E402


def _candidate(box: tuple[int, int, int, int], point: tuple[int, int]) -> LabelCandidate:
    return LabelCandidate(box=box, point_xy=point, label_color="white", traced=True)


def _json_request(
    payload: object | None = None,
    *,
    method: str = "POST",
    query: dict[str, str] | None = None,
):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    query_string = urlencode(query or {}).encode("ascii")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": "/api/corrections",
        "raw_path": b"/api/corrections",
        "query_string": query_string,
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    return backend_server.Request(scope, receive)


def _response_json(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


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


class UiBackendCorrectionHistoryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "correction-history.db"
        self.db_patch = patch.object(
            backend_server, "CORRECTION_DB_PATH", self.db_path
        )
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)

    def test_legacy_database_is_migrated_without_changing_existing_row(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                """
                CREATE TABLE correction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    part_no TEXT NOT NULL,
                    scan_name TEXT NOT NULL,
                    point_id TEXT NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    worker TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO correction_history "
                "(part_no, scan_name, point_id, old_value, new_value, worker, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "64XX2",
                    "legacy.png",
                    "P-14",
                    -0.2,
                    -0.1,
                    "tester",
                    "2026-08-27T17:39:04",
                ),
            )

        backend_server._init_correction_db()

        with closing(sqlite3.connect(self.db_path)) as conn:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(correction_history)")
            }
            row = conn.execute(
                "SELECT id, part_no, scan_name, point_id, old_value, new_value, "
                "worker, created_at, action, old_mode, new_mode, coefficient, "
                "source_entry_id FROM correction_history"
            ).fetchone()
            user_version = conn.execute("PRAGMA user_version").fetchone()[0]

        self.assertTrue(
            {"action", "old_mode", "new_mode", "coefficient", "source_entry_id"}
            <= columns
        )
        self.assertEqual(
            row,
            (
                1,
                "64XX2",
                "legacy.png",
                "P-14",
                -0.2,
                -0.1,
                "tester",
                "2026-08-27T17:39:04",
                "edit",
                None,
                None,
                None,
                None,
            ),
        )
        self.assertEqual(user_version, 1)

    async def test_post_round_trip_supports_metadata_and_legacy_payloads(self) -> None:
        backend_server._init_correction_db()
        first_response = await backend_server.create_correction(
            _json_request(
                {
                    "partNo": "64XX2",
                    "scanName": "scan-a.png",
                    "pointId": "P-14",
                    "oldValue": -0.24,
                    "newValue": -0.1,
                    "worker": "tester",
                    "action": "edit",
                    "oldMode": "auto",
                    "newMode": "manual",
                    "coefficient": 1.15,
                }
            )
        )
        first = _response_json(first_response)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(first["action"], "edit")
        self.assertEqual(first["oldMode"], "auto")
        self.assertEqual(first["newMode"], "manual")
        self.assertEqual(first["coefficient"], 1.15)
        self.assertIsNone(first["sourceEntryId"])

        restore_response = await backend_server.create_correction(
            _json_request(
                {
                    "partNo": "64XX2",
                    "scanName": "scan-a.png",
                    "pointId": "P-14",
                    "oldValue": -0.1,
                    "newValue": -0.24,
                    "worker": "reviewer",
                    "action": "restore_before",
                    "oldMode": "manual",
                    "newMode": "auto",
                    "coefficient": 1.15,
                    "sourceEntryId": first["id"],
                }
            )
        )
        restored = _response_json(restore_response)
        self.assertEqual(restore_response.status_code, 200)
        self.assertEqual(restored["action"], "restore_before")
        self.assertEqual(restored["sourceEntryId"], first["id"])

        legacy_response = await backend_server.create_correction(
            _json_request(
                {
                    "partNo": "64XX2",
                    "scanName": "scan-a.png",
                    "pointId": "P-15",
                    "oldValue": 0.2,
                    "newValue": None,
                    "worker": "tester",
                }
            )
        )
        legacy = _response_json(legacy_response)
        self.assertEqual(legacy_response.status_code, 200)
        self.assertEqual(legacy["action"], "edit")
        self.assertIsNone(legacy["oldMode"])
        self.assertIsNone(legacy["newMode"])
        self.assertIsNone(legacy["coefficient"])
        self.assertIsNone(legacy["sourceEntryId"])

        with closing(sqlite3.connect(self.db_path)) as conn:
            original = conn.execute(
                "SELECT action, old_mode, new_mode, coefficient, source_entry_id "
                "FROM correction_history WHERE id = ?",
                (first["id"],),
            ).fetchone()
        self.assertEqual(original, ("edit", "auto", "manual", 1.15, None))

    async def test_get_can_filter_by_part_number_and_scan_name(self) -> None:
        backend_server._init_correction_db()
        records = (
            ("64XX2", "scan-a.png", "P-01"),
            ("64XX2", "scan-b.png", "P-02"),
            ("OTHER", "scan-a.png", "P-03"),
        )
        for part_no, scan_name, point_id in records:
            response = await backend_server.create_correction(
                _json_request(
                    {
                        "partNo": part_no,
                        "scanName": scan_name,
                        "pointId": point_id,
                        "oldValue": 0.0,
                        "newValue": 0.1,
                    }
                )
            )
            self.assertEqual(response.status_code, 200)

        part_response = await backend_server.list_corrections(
            _json_request(method="GET", query={"partNo": "64XX2"})
        )
        part_entries = _response_json(part_response)["entries"]
        self.assertEqual(
            [(entry["scanName"], entry["pointId"]) for entry in part_entries],
            [("scan-b.png", "P-02"), ("scan-a.png", "P-01")],
        )

        scan_response = await backend_server.list_corrections(
            _json_request(
                method="GET",
                query={"partNo": "64XX2", "scanName": "scan-a.png"},
            )
        )
        scan_entries = _response_json(scan_response)["entries"]
        self.assertEqual(len(scan_entries), 1)
        self.assertEqual(scan_entries[0]["pointId"], "P-01")

        scan_only_response = await backend_server.list_corrections(
            _json_request(method="GET", query={"scanName": "scan-a.png"})
        )
        scan_only_entries = _response_json(scan_only_response)["entries"]
        self.assertEqual(
            [entry["partNo"] for entry in scan_only_entries], ["OTHER", "64XX2"]
        )

    async def test_post_rejects_nonfinite_numbers_before_inserting(self) -> None:
        backend_server._init_correction_db()
        invalid_fields = (
            ("oldValue", float("nan")),
            ("newValue", float("inf")),
            ("coefficient", float("-inf")),
            ("newValue", "0.3"),
            ("oldValue", True),
            ("oldValue", 10**400),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value):
                payload = {
                    "partNo": "64XX2",
                    "scanName": "scan-a.png",
                    "pointId": "P-01",
                    "oldValue": 0.0,
                    "newValue": 0.1,
                }
                payload[field] = value
                response = await backend_server.create_correction(
                    _json_request(payload)
                )
                self.assertEqual(response.status_code, 422)

        with closing(sqlite3.connect(self.db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM correction_history").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_post_rejects_unknown_action_mode_and_source_id(self) -> None:
        backend_server._init_correction_db()
        invalid_metadata = (
            {"action": "delete"},
            {"oldMode": "unknown"},
            {"newMode": 1},
            {"sourceEntryId": 0},
            {"sourceEntryId": True},
            {"sourceEntryId": 2**63},
        )
        for extra in invalid_metadata:
            with self.subTest(extra=extra):
                response = await backend_server.create_correction(
                    _json_request(
                        {
                            "partNo": "64XX2",
                            "scanName": "scan-a.png",
                            "pointId": "P-01",
                            "oldValue": 0.0,
                            "newValue": 0.1,
                            **extra,
                        }
                    )
                )
                self.assertEqual(response.status_code, 422)


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
        # 이 클래스가 만드는 "64XX2-DR000" 은 실제 개발 PC에서 흔히 쓰는 진짜 품번
        # 이름과 겹친다. analyze_image 가 이제 저장된 확정 방향을 먼저 찾아보므로,
        # 그 자리를 매번 빈 임시 폴더로 갈아 끼우지 않으면 실제 로컬에 남아 있는
        # 확정 정렬 파일을 조용히 읽어와 이 스몰 픽스처와 맞지 않는 값으로 테스트를
        # 오염시킨다. 특정 테스트가 스토어를 직접 지정하면 그 with 블록 동안만
        # 이 기본값을 덮어쓴다.
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store_patcher = patch.object(
            backend_server,
            "ALIGNMENT_STORE",
            backend_server.AlignmentStore(Path(temp_dir.name) / "alignment"),
        )
        store_patcher.start()
        self.addCleanup(store_patcher.stop)
        library_patcher = patch.object(
            backend_server,
            "PRODUCT_LIBRARY",
            backend_server.ProductLibrary(Path(temp_dir.name) / "product"),
        )
        library_patcher.start()
        self.addCleanup(library_patcher.stop)

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
        # self.scan 은 product 를 180도 뒤집고 2배로 키운 자료다. 방향 추정이
        # 확실하면(여기선 확실하다) 좌표만 옮기지 않고 라벨을 읽기 전에 원본
        # 픽셀 자체를 바로 세운다 -- 그래야 인쇄된 숫자도 똑바로 보여 OCR이
        # 위아래가 뒤집힌 텍스트를 오독하지 않는다. 그래서 이미 바로 세운 뒤에
        # 남는 정렬은 배율·평행이동만 필요하고 flip 은 더 이상 필요 없다.
        self.assertFalse(result["alignment"]["flipX"])
        self.assertFalse(result["alignment"]["flipY"])
        self.assertTrue(result["alignment"]["confident"])

        self.assertEqual(len(result["points"]), 1)
        point = result["points"][0]
        # detect_labels 는 이미 바로 세운 이미지에서 (100, 60) 을 찾았다고
        # 가정하는 픽스처다. 바로 세운 스캔(400x240)을 제품(200x120)으로
        # 절반 축소하면 (50, 30) -> 25%, 25%.
        self.assertAlmostEqual(point["xProduct"], 25.0, delta=2.0)
        self.assertAlmostEqual(point["yProduct"], 25.0, delta=2.0)
        self.assertEqual(result["stats"]["pointsTransferred"], 1)

    def test_upside_down_scan_is_righted_before_ocr_reads_it(self) -> None:
        """product_alignment 는 좌표 행렬만 만든다 -- 라벨의 인쇄된 숫자는

        건드리지 않는다. 그래서 스캔이 제품데이터 대비 뒤집혀 있으면(이 픽스처는
        180도), 좌표만 나중에 옮기는 예전 방식으로는 Qwen 이 뒤집힌 채로 인쇄된
        숫자를 그대로 읽어 오독한다("0.8" 이 "80.0" 으로 읽히는 식). 라벨을 찾고
        읽기 전에 원본 픽셀 자체를 먼저 바로 세워야 한다.
        """
        captured: list[np.ndarray] = []

        def _capture_boxes(image: np.ndarray) -> list:
            captured.append(image.copy())
            return []

        with (
            patch.object(backend_server, "detect_label_boxes", side_effect=_capture_boxes),
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
            patch.object(backend_server, "detect_labels", return_value=[]),
            patch.object(
                backend_server,
                "build_scan_mask",
                return_value=np.full(self.scan.shape[:2], 255, dtype=np.uint8),
            ),
        ):
            backend_server.analyze_image(self.scan, "JD_64XX2-DR000 3D 스캔.png", self.product)

        self.assertEqual(len(captured), 1)
        righted = cv2.flip(self.scan, -1)  # self.scan 은 product 를 180도 뒤집은 것이니, 되돌리면 이거다.
        np.testing.assert_array_equal(
            captured[0],
            righted,
            "라벨을 찾기 전에 스캔을 바로 세워야 OCR 이 뒤집힌 숫자를 오독하지 않는다.",
        )

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

        # first 는 자동 판정이 확실해서 이미 픽셀 자체를 바로 세웠으므로 flip 이
        # 남지 않는다 (test_uploaded_product_image_moves_points_onto_it 참고).
        self.assertFalse(first["alignment"]["flipX"])
        # 사람이 확정 저장한 방향은 실제로는 틀렸다(좌우만 반전, 원래는 상하까지
        # 필요) -- 그래도 저장된 값이니 그대로 믿고 픽셀을 그 방향으로 바로
        # 세운다. 여전히 어긋난 상태로 남아 outline IoU 가 낮게 나오고, 그
        # 사실을 경고로 알려야 한다.
        self.assertFalse(second["alignment"]["flipX"])
        self.assertTrue(second["alignment"]["overridden"])
        self.assertFalse(second["alignment"]["confident"])
        self.assertTrue(
            any("확정 저장된 방향" in item for item in second["warningsByEngine"]["product"])
        )
        self.assertTrue(
            any("일치도가 낮습니다" in item for item in second["warningsByEngine"]["product"])
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
