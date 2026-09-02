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
        "AJIN_CORRECTION_DB_URL": "",
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


class UiBackendFileOrganizerTest(unittest.TestCase):
    def test_folder_order_keeps_category_required(self) -> None:
        self.assertFalse(backend_server.is_valid_folder_order(["product"]))
        self.assertTrue(backend_server.is_valid_folder_order(["product", "category"]))
        self.assertTrue(
            backend_server.is_valid_folder_order(["category", "family", "product"])
        )

    def test_folder_order_migration_preserves_detail_tree_without_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            product = root / "JD" / "PNL DASH" / "64XX2-DR000"
            drawing_file = product / "02. 금형도면" / "02. 패턴도" / "OP30" / "pattern.zip"
            document_file = product / "03. 문서" / "02. 보정이력" / "history.xlsx"
            drawing_file.parent.mkdir(parents=True)
            document_file.parent.mkdir(parents=True)
            drawing_file.write_bytes(b"drawing")
            document_file.write_bytes(b"document")

            first = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["product", "category"],
                ["category", "product"],
            )
            self.assertEqual(first.errors, [])
            self.assertEqual(
                (
                    root / "02. 금형도면" / "JD" / "PNL DASH" / "64XX2"
                    / "02. 패턴도" / "OP30" / "pattern.zip"
                ).read_bytes(),
                b"drawing",
            )

            second = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["category", "product"],
                ["product", "category"],
            )
            self.assertEqual(second.errors, [])
            self.assertEqual(
                (
                    root / "JD" / "PNL DASH" / "64XX2" / "02. 금형도면"
                    / "02. 패턴도" / "OP30" / "pattern.zip"
                ).read_bytes(),
                b"drawing",
            )
            self.assertEqual(
                (
                    root / "JD" / "PNL DASH" / "64XX2" / "03. 문서"
                    / "02. 보정이력" / "history.xlsx"
                ).read_bytes(),
                b"document",
            )
            self.assertEqual(
                sorted(path.name for path in root.iterdir() if path.is_dir()),
                ["JD"],
            )

    def test_classify_reuses_family_only_folder_across_different_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination_root = Path(temp) / "organized"
            classifier = backend_server.FilenameClassifier(
                backend_server.FILE_ORGANIZER_RULES,
                destination_root,
                ["product", "category"],
            )

            first = classifier.classify(
                Path("JM_67312-DZ000_DASH LWR_OP30_패턴도_260825.zip")
            )
            self.assertIsNotNone(first.target_dir)
            first.target_dir.mkdir(parents=True)

            # 같은 계열(67312)이지만 상세 코드가 다른 파일 — 품번 폴더는 계열까지만
            # 이름 짓기 때문에, 전체 품번이 달라도 같은 차종/품명/계열 폴더로 모여야 한다.
            second = classifier.classify(
                Path("JM_67312-DZ001_DASH LWR_OP20_완성도_260825.zip")
            )

            first_product_dir = destination_root / "JM" / "DASH LWR" / "67312"
            self.assertEqual(
                first.target_dir.relative_to(first_product_dir).parts,
                ("02. 금형도면", "02. 패턴도", "OP30"),
            )
            self.assertEqual(
                second.target_dir.relative_to(first_product_dir).parts,
                ("02. 금형도면", "03. 완성도", "OP20"),
            )
            self.assertEqual(second.matched_product_folder, "67312")

    def test_excel_detail_tags_select_drawing_document_and_nc_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination_root = Path(temp) / "organized"
            product_root = destination_root / "JD" / "PNL DASH" / "64XX2-DR000"
            for category in ("02. 금형도면", "03. 문서", "06. NC DATA"):
                (product_root / category).mkdir(parents=True)
            classifier = backend_server.FilenameClassifier(
                backend_server.FILE_ORGANIZER_RULES,
                destination_root,
                ["product", "category"],
            )

            drawing = classifier.classify(
                Path("JD_64XX2-DR000_PNL DASH_OP30_패턴도_260825.zip")
            )
            document = classifier.classify(
                Path("JD_64XX2-DR000_보정적용_260825.xlsx")
            )
            nc_data = classifier.classify(
                Path("260825_JD_64XX2-DR000_DASH_OP50_형상_UPRDIE_NC DATA.ZIP")
            )

            self.assertEqual(drawing.category_key, "02")
            self.assertEqual(drawing.detail_path, "02. 패턴도/OP30")
            self.assertEqual(
                drawing.target_dir.relative_to(product_root).parts,
                ("02. 금형도면", "02. 패턴도", "OP30"),
            )
            self.assertEqual(document.detail_path, "02. 보정이력")
            self.assertEqual(
                document.target_dir.relative_to(product_root).parts,
                ("03. 문서", "02. 보정이력"),
            )
            self.assertEqual(nc_data.category_key, "06")
            self.assertEqual(nc_data.detail_path, "OP50")
            self.assertEqual(
                nc_data.target_dir.relative_to(product_root).parts,
                ("06. NC DATA", "OP50"),
            )

    def test_folder_order_migration_recognizes_family_only_product_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            leaf = root / "JD" / "PNL DASH" / "64XX2" / "02. 금형도면" / "01. 구조도" / "OP30"
            leaf.mkdir(parents=True)
            (leaf / "sample.zip").write_bytes(b"demo")

            to_category_first = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["product", "category"],
                ["category", "product"],
            )
            self.assertEqual(to_category_first.errors, [])
            self.assertEqual(to_category_first.moved, 1)
            moved = (
                root / "02. 금형도면" / "JD" / "PNL DASH" / "64XX2"
                / "01. 구조도" / "OP30" / "sample.zip"
            )
            self.assertEqual(moved.read_bytes(), b"demo")

            back_to_product_first = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["category", "product"],
                ["product", "category"],
            )
            self.assertEqual(back_to_product_first.errors, [])
            self.assertEqual(back_to_product_first.moved, 1)
            self.assertEqual(
                (
                    root / "JD" / "PNL DASH" / "64XX2" / "02. 금형도면"
                    / "01. 구조도" / "OP30" / "sample.zip"
                ).read_bytes(),
                b"demo",
            )

    def test_classify_proposes_new_folder_for_unseen_customer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination_root = Path(temp) / "organized"
            destination_root.mkdir()
            classifier = backend_server.FilenameClassifier(
                backend_server.FILE_ORGANIZER_RULES,
                destination_root,
                ["product", "category"],
            )

            result = classifier.classify(Path("JDZ_12345-XX000_NEW PART_260825.dwg"))

            self.assertEqual(result.item_no, "12345-XX000")
            self.assertEqual(result.customer, "JDZ")
            self.assertEqual(result.category_key, "02")
            self.assertIsNotNone(result.target_dir)
            self.assertEqual(
                result.target_dir.relative_to(destination_root).parts,
                ("JDZ", "NEW PART", "12345", "02. 금형도면"),
            )

    def test_scan_classifies_product_data_into_existing_part_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "incoming"
            destination_root = root / "organized"
            source_root.mkdir()
            detail = destination_root / "64XX2" / "01. 3D제품데이터" / "JD PNL DASH 64XX2-DR000"
            detail.mkdir(parents=True)
            source = source_root / "64XX2-DR000 제품데이터.png"
            source.write_bytes(b"demo")
            with patch.object(backend_server, "FILE_SOURCE_ROOT", source_root), patch.object(
                backend_server, "FOLDER_ROOT", destination_root
            ), patch.object(
                backend_server, "_active_folder_order",
                return_value=["family", "category", "product"],
            ):
                items = backend_server._scan_organizer_source()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["itemNo"], "64XX2-DR000")
            self.assertEqual(items[0]["categoryKey"], "01")
            self.assertIn("JD PNL DASH 64XX2-DR000", items[0]["targetDir"])

    def test_execute_copies_file_and_keeps_local_audit_log_without_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "incoming"
            destination_root = root / "organized"
            log_root = root / "logs"
            staging_root = root / "staging"
            source_root.mkdir()
            staging_root.mkdir()
            detail = destination_root / "64XX2" / "01. 3D제품데이터" / "JD PNL DASH 64XX2-DR000"
            detail.mkdir(parents=True)
            source = source_root / "64XX2-DR000 제품데이터.png"
            source.write_bytes(b"demo")
            with patch.object(backend_server, "FILE_SOURCE_ROOT", source_root), patch.object(
                backend_server, "FOLDER_ROOT", destination_root
            ), patch.object(backend_server, "FILE_LOG_ROOT", log_root), patch.object(
                backend_server, "FILE_STAGING_ROOT", staging_root
            ), patch.object(backend_server, "_active_file_database_url", return_value=""), patch.object(
                backend_server, "_active_folder_order",
                return_value=["family", "category", "product"],
            ):
                response = backend_server._execute_file_organizer(
                    {
                        "operation": "copy",
                        "conflict": "rename",
                        "items": [{"sourcePath": str(source)}],
                    }
                )
            copied = detail / source.name
            self.assertTrue(source.exists())
            self.assertEqual(copied.read_bytes(), b"demo")
            self.assertEqual(response["results"][0]["status"], "success")
            self.assertIn("로컬 감사 로그", response["databaseNote"])
            self.assertEqual(len(list(log_root.glob("*.jsonl"))), 1)

    def test_source_path_outside_allowed_roots_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "incoming"
            staging_root = root / "staging"
            source_root.mkdir()
            staging_root.mkdir()
            outside = root / "outside.txt"
            outside.write_text("no", encoding="utf-8")
            with patch.object(backend_server, "FILE_SOURCE_ROOT", source_root), patch.object(
                backend_server, "FILE_STAGING_ROOT", staging_root
            ):
                with self.assertRaisesRegex(ValueError, "허용된 원본"):
                    backend_server._safe_organizer_source(str(outside))


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
        self.env_patch = patch.dict(os.environ, {"AJIN_CORRECTION_DB_URL": ""})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_mysql_url_is_parsed_without_exposing_credentials(self) -> None:
        config = backend_server._mysql_connection_config(
            "mysql://adc_user:p%40ss@db.internal:3307/ajin_adc"
            "?charset=utf8mb4&connect_timeout=12"
        )

        self.assertEqual(config["host"], "db.internal")
        self.assertEqual(config["port"], 3307)
        self.assertEqual(config["user"], "adc_user")
        self.assertEqual(config["password"], "p@ss")
        self.assertEqual(config["database"], "ajin_adc")
        self.assertEqual(config["charset"], "utf8mb4")
        self.assertEqual(config["connection_timeout"], 12)
        self.assertFalse(config["autocommit"])

    def test_mysql_adapter_uses_server_parameter_markers(self) -> None:
        class FakeCursor:
            def __init__(self) -> None:
                self.statement = ""
                self.params: tuple[object, ...] = ()

            def execute(self, statement, params) -> None:
                self.statement = statement
                self.params = params

            def close(self) -> None:
                pass

        class FakeConnection:
            def __init__(self) -> None:
                self.cursor_instance = FakeCursor()
                self.dictionary = False

            def cursor(self, *, dictionary=False):
                self.dictionary = dictionary
                return self.cursor_instance

        raw = FakeConnection()
        connection = backend_server._CorrectionConnection(raw, "mysql")
        cursor = connection.execute(
            "SELECT * FROM correction_history WHERE part_no = ? AND id = ?",
            ("64XX2", 7),
        )

        self.assertTrue(raw.dictionary)
        self.assertEqual(
            cursor.statement,
            "SELECT * FROM correction_history WHERE part_no = %s AND id = %s",
        )
        self.assertEqual(cursor.params, ("64XX2", 7))

    def test_invalid_mysql_url_is_rejected(self) -> None:
        invalid_urls = (
            "postgresql://user:pass@db/ajin",
            "mysql://db/ajin",
            "mysql://user@db",
            "mysql://user@db/ajin?connect_timeout=slow",
        )
        for database_url in invalid_urls:
            with self.subTest(database_url=database_url):
                with self.assertRaises(ValueError):
                    backend_server._mysql_connection_config(database_url)

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

    async def test_database_outage_returns_service_unavailable(self) -> None:
        with patch.object(
            backend_server,
            "_get_correction_db",
            side_effect=backend_server.CorrectionDatabaseError("offline"),
        ):
            response = await backend_server.list_corrections(
                _json_request(method="GET", query={"partNo": "64XX2"})
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("error", _response_json(response))


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


if __name__ == "__main__":
    unittest.main()
