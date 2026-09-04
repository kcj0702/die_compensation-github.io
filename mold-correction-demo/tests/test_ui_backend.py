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
        "AJIN_CORRECTION_DB_URL": "",
        "AJIN_CORRECTION_DB_PATH": str(
            Path(_SERVER_IMPORT_DB_DIR.name) / "import-correction-history.db"
        )
    },
):
    SPEC.loader.exec_module(backend_server)

from core import UNKNOWN_VEHICLE_FOLDER  # noqa: E402
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


class UiBackendMarkerDifferenceTest(unittest.TestCase):
    def test_marker_centers_are_measured_from_version_difference(self) -> None:
        labels_inpainted = np.full((40, 60, 3), 120, dtype=np.uint8)
        labels_points_inpainted = labels_inpainted.copy()
        labels_points_inpainted[8:13, 17:22] = 180
        labels_points_inpainted[25:30, 43:48] = 60

        centers = backend_server._marker_centers_from_version_difference(
            labels_inpainted, labels_points_inpainted
        )

        self.assertEqual(len(centers), 2)
        self.assertTrue(any(np.hypot(x - 19, y - 10) < 0.1 for x, y in centers))
        self.assertTrue(any(np.hypot(x - 45, y - 27) < 0.1 for x, y in centers))


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
    def test_file_organizer_routes_allow_the_methods_the_ui_calls(self) -> None:
        # 화면이 부르는 HTTP 메서드와 라우트 선언이 어긋나면 사용자에게는
        # 'Method Not Allowed' 가 JSON 파싱 오류로만 보인다("원본 스캔" 버튼).
        expected = {
            "/api/file-organizer/status": {"GET"},
            "/api/file-organizer/scan": {"GET"},
            "/api/file-organizer/upload": {"POST"},
            "/api/file-organizer/discard": {"POST"},
            "/api/file-organizer/execute": {"POST"},
            "/api/file-organizer/database": {"POST"},
            "/api/file-organizer/folder-order": {"GET", "POST"},
            "/api/file-organizer/paths": {"GET", "POST"},
            "/api/file-organizer/reveal": {"POST"},
        }
        declared = {
            route.path: set(route.methods or ())
            for route in backend_server.app.routes
            if getattr(route, "path", "").startswith("/api/file-organizer/")
        }
        for path, methods in expected.items():
            self.assertIn(path, declared, f"{path} 라우트가 없습니다")
            self.assertTrue(
                methods <= declared[path],
                f"{path}: 화면은 {sorted(methods)} 를 부르는데 라우트는 {sorted(declared[path])} 만 허용합니다",
            )

    def test_folder_order_accepts_every_four_axis_permutation(self) -> None:
        expected = ["item", "vehicle", "category", "detail"]
        self.assertTrue(backend_server.is_valid_folder_order(expected))
        self.assertTrue(backend_server.is_valid_folder_order(["vehicle", "item", "detail", "category"]))
        self.assertFalse(backend_server.is_valid_folder_order(["item", "vehicle", "category"]))
        self.assertFalse(backend_server.is_valid_folder_order(["item", "item", "category", "detail"]))

    def test_folder_order_migration_preserves_detail_tree_without_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            product = root / "64XX2" / "JD"
            drawing_file = product / "02. 금형도면" / "02. 패턴도" / "OP30" / "pattern.zip"
            document_file = product / "03. 문서" / "02. 보정이력" / "history.xlsx"
            drawing_file.parent.mkdir(parents=True)
            document_file.parent.mkdir(parents=True)
            drawing_file.write_bytes(b"drawing")
            document_file.write_bytes(b"document")

            first = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["category", "detail", "item", "vehicle"],
            )
            self.assertEqual(first.errors, [])
            self.assertEqual(first.moved, 2)
            moved_drawing = (
                root / "02. 금형도면" / "02. 패턴도" / "OP30"
                / "64XX2" / "JD" / "pattern.zip"
            )
            moved_document = (
                root / "03. 문서" / "02. 보정이력"
                / "64XX2" / "JD" / "history.xlsx"
            )
            self.assertEqual(moved_drawing.read_bytes(), b"drawing")
            self.assertEqual(moved_document.read_bytes(), b"document")

            second = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["item", "vehicle", "category", "detail"],
            )
            self.assertEqual(second.errors, [])
            self.assertEqual(second.moved, 2)
            self.assertEqual(drawing_file.read_bytes(), b"drawing")
            self.assertEqual(document_file.read_bytes(), b"document")

            # 같은 순서로 다시 저장하면 이미 제자리인 파일은 건드리지 않는다.
            again = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["item", "vehicle", "category", "detail"],
            )
            self.assertEqual(again.moved, 0)
            self.assertEqual(again.errors, [])

    def test_placeholder_only_tree_reports_the_structure_it_rebuilt(self) -> None:
        """빈 폴더뿐인 트리도 배치가 바뀌었음을 알려야 한다.

        .gitkeep 은 옮긴 파일로 세지 않으므로, 실제 파일이 없는 트리에서는
        moved 가 늘 0 이다. 그것만 보고 화면이 "제자리입니다"라고 알리면
        폴더는 실제로 재배치됐는데 아무 일도 없었던 것처럼 보인다.
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            keep = root / "64XX2" / "JD" / "01. 3D제품데이터" / ".gitkeep"
            keep.parent.mkdir(parents=True)
            keep.touch()

            result = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["category", "vehicle", "item", "detail"],
            )

            self.assertEqual(result.errors, [])
            self.assertEqual(result.moved, 0)
            self.assertEqual(result.structure_moved, 1)
            self.assertTrue(
                (root / "01. 3D제품데이터" / "JD" / "64XX2" / ".gitkeep").exists()
            )
            self.assertFalse(keep.exists())

            # 같은 순서로 다시 저장하면 옮길 것이 없으므로 둘 다 0 이어야 한다.
            again = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["category", "vehicle", "item", "detail"],
            )
            self.assertEqual(again.moved, 0)
            self.assertEqual(again.structure_moved, 0)

    def test_classification_follows_selected_folder_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination_root = Path(temp) / "organized"
            classifier = backend_server.FilenameClassifier(
                backend_server.FILE_ORGANIZER_RULES,
                destination_root,
                ["category", "detail", "item", "vehicle"],
            )
            result = classifier.classify(
                Path("JM_67312-DZ000_DASH LWR_OP30_패턴도_260825.zip")
            )
            self.assertEqual(
                result.target_dir.relative_to(destination_root).parts,
                ("02. 금형도면", "02. 패턴도", "OP30", "67312", "JM"),
            )

    def test_existing_item_folder_is_recognized_when_item_is_not_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination_root = Path(temp) / "organized"
            existing = (
                destination_root / "02. 금형도면" / "02. 패턴도" / "OP30" / "67312"
            )
            existing.mkdir(parents=True)
            classifier = backend_server.FilenameClassifier(
                backend_server.FILE_ORGANIZER_RULES,
                destination_root,
                ["category", "detail", "item", "vehicle"],
            )

            result = classifier.classify(
                Path("JM_67312-DZ000_DASH LWR_OP30_패턴도_260825.zip")
            )

            self.assertEqual(result.matched_product_folder, "67312")

    def test_migration_moves_files_under_unknown_vehicle_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            source = (
                root / "67312" / UNKNOWN_VEHICLE_FOLDER
                / "02. 금형도면" / "02. 패턴도" / "OP30"
            )
            source.mkdir(parents=True)
            (source / "pattern.zip").write_bytes(b"drawing")

            result = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["vehicle", "item", "category", "detail"],
            )

            self.assertEqual(result.errors, [])
            self.assertEqual(result.moved, 1)
            moved = (
                root / UNKNOWN_VEHICLE_FOLDER / "67312"
                / "02. 금형도면" / "02. 패턴도" / "OP30" / "pattern.zip"
            )
            self.assertEqual(moved.read_bytes(), b"drawing")

    def test_migration_recognizes_vehicle_not_registered_in_rules(self) -> None:
        # 분류기는 rules.json 에 없는 새 차종 폴더도 스스로 만든다. 순서 변경이
        # 그 폴더를 못 알아보면 새 차종 자료만 옛 구조에 남는다.
        self.assertNotIn("XM", backend_server.FILE_ORGANIZER_RULES["customers"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            source = root / "71612" / "XM" / "02. 금형도면" / "01. 구조도" / "OP30"
            source.mkdir(parents=True)
            (source / "structure.zip").write_bytes(b"drawing")

            result = backend_server.migrate_folder_structure(
                root,
                backend_server.FILE_ORGANIZER_RULES,
                ["vehicle", "item", "category", "detail"],
            )

            self.assertEqual(result.errors, [])
            self.assertEqual(result.moved, 1)
            moved = (
                root / "XM" / "71612" / "02. 금형도면" / "01. 구조도" / "OP30"
                / "structure.zip"
            )
            self.assertEqual(moved.read_bytes(), b"drawing")

    def test_classify_shares_family_folder_across_detail_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination_root = Path(temp) / "organized"
            classifier = backend_server.FilenameClassifier(
                backend_server.FILE_ORGANIZER_RULES,
                destination_root,
                ["item", "vehicle", "category", "detail"],
            )

            first = classifier.classify(
                Path("JM_67312-DZ000_DASH LWR_OP30_패턴도_260825.zip")
            )
            self.assertIsNotNone(first.target_dir)
            first.target_dir.mkdir(parents=True)

            # 앞자리 계열 코드가 같으면 상세 코드가 달라도 같은 품번 폴더를 쓴다.
            second = classifier.classify(
                Path("JM_67312-DZ001_DASH LWR_OP20_완성도_260825.zip")
            )

            product_dir = destination_root / "67312" / "JM"
            self.assertEqual(
                first.target_dir.relative_to(product_dir).parts,
                ("02. 금형도면", "02. 패턴도", "OP30"),
            )
            self.assertEqual(
                second.target_dir.relative_to(product_dir).parts,
                ("02. 금형도면", "03. 완성도", "OP20"),
            )
            self.assertEqual(second.matched_product_folder, "67312")

    def test_excel_detail_tags_select_drawing_document_and_nc_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination_root = Path(temp) / "organized"
            product_root = destination_root / "64XX2" / "JD"
            for category in ("02. 금형도면", "03. 문서", "06. NC DATA"):
                (product_root / category).mkdir(parents=True)
            classifier = backend_server.FilenameClassifier(
                backend_server.FILE_ORGANIZER_RULES,
                destination_root,
                ["item", "vehicle", "category", "detail"],
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

    def test_legacy_saved_folder_order_falls_back_to_default_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            settings = Path(temp)
            (settings / ".folder_order.json").write_text(
                json.dumps(["product", "category"]), encoding="utf-8"
            )
            self.assertEqual(
                backend_server.load_folder_order(settings),
                ["item", "vehicle", "category", "detail"],
            )

    def test_classify_proposes_new_folder_for_unseen_customer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination_root = Path(temp) / "organized"
            destination_root.mkdir()
            classifier = backend_server.FilenameClassifier(
                backend_server.FILE_ORGANIZER_RULES,
                destination_root,
                ["item", "vehicle", "category", "detail"],
            )

            result = classifier.classify(Path("JDZ_12345-XX000_NEW PART_260825.dwg"))

            self.assertEqual(result.item_no, "12345-XX000")
            self.assertEqual(result.customer, "JDZ")
            self.assertEqual(result.category_key, "02")
            self.assertIsNotNone(result.target_dir)
            self.assertEqual(
                result.target_dir.relative_to(destination_root).parts,
                ("12345", "JDZ", "02. 금형도면"),
            )

    def test_scan_classifies_product_data_into_existing_part_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "incoming"
            destination_root = root / "organized"
            source_root.mkdir()
            detail = destination_root / "64XX2" / "JD" / "01. 3D제품데이터"
            detail.mkdir(parents=True)
            source = source_root / "64XX2-DR000 제품데이터.png"
            source.write_bytes(b"demo")
            with patch.object(backend_server, "FILE_SOURCE_ROOT", source_root), patch.object(
                backend_server, "FOLDER_ROOT", destination_root
            ), patch.object(
                backend_server, "_active_folder_order",
                return_value=["item", "vehicle", "category", "detail"],
            ):
                items = backend_server._scan_organizer_source()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["itemNo"], "64XX2-DR000")
            self.assertEqual(items[0]["categoryKey"], "01")
            self.assertIn("64XX2/JD/01. 3D제품데이터", items[0]["targetDir"])

    def test_execute_copies_file_and_keeps_local_audit_log_without_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "incoming"
            destination_root = root / "organized"
            log_root = root / "logs"
            staging_root = root / "staging"
            source_root.mkdir()
            staging_root.mkdir()
            detail = destination_root / "64XX2" / "JD" / "01. 3D제품데이터"
            detail.mkdir(parents=True)
            source = source_root / "64XX2-DR000 제품데이터.png"
            source.write_bytes(b"demo")
            with patch.object(backend_server, "FILE_SOURCE_ROOT", source_root), patch.object(
                backend_server, "FOLDER_ROOT", destination_root
            ), patch.object(backend_server, "FILE_LOG_ROOT", log_root), patch.object(
                backend_server, "FILE_STAGING_ROOT", staging_root
            ), patch.object(backend_server, "_active_file_database_url", return_value=""), patch.object(
                backend_server, "_active_folder_order",
                return_value=["item", "vehicle", "category", "detail"],
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

    def test_unclassifiable_file_goes_to_the_unclassified_folder_not_the_root(self) -> None:
        """화면이 비운 대상 폴더를 루트로 읽지 않는지 확인한다.

        품번이나 카테고리를 못 읽은 파일은 대상 폴더가 빈 문자열로 내려오고
        화면은 그 자리에 '_미분류'를 보여 준다. 빈 문자열을 "사용자가 고른
        폴더"로 받으면 정리 폴더 루트가 되어, 화면 표시와 달리 파일이 품번
        폴더들 옆에 그대로 쌓인다.
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "incoming"
            destination_root = root / "organized"
            log_root = root / "logs"
            staging_root = root / "staging"
            source_root.mkdir()
            staging_root.mkdir()
            destination_root.mkdir()
            # 품번이 없어 분류되지 않는 이름이다.
            source = source_root / "JM DASH LWR 성형해석 리포트 260825.ppt"
            source.write_bytes(b"demo")
            with patch.object(backend_server, "FILE_SOURCE_ROOT", source_root), patch.object(
                backend_server, "FOLDER_ROOT", destination_root
            ), patch.object(backend_server, "FILE_LOG_ROOT", log_root), patch.object(
                backend_server, "FILE_STAGING_ROOT", staging_root
            ), patch.object(backend_server, "_active_file_database_url", return_value=""), patch.object(
                backend_server, "_active_folder_order",
                return_value=["vehicle", "category", "item", "detail"],
            ):
                scanned = backend_server._scan_organizer_source()
                self.assertEqual(scanned[0]["targetDir"], "")
                response = backend_server._execute_file_organizer(
                    {
                        "operation": "copy",
                        "conflict": "rename",
                        # 화면이 실제로 보내는 모양 그대로다.
                        "items": [
                            {"sourcePath": str(source), "targetDir": scanned[0]["targetDir"]}
                        ],
                    }
                )

            self.assertEqual(response["results"][0]["status"], "success")
            unclassified = destination_root / "_미분류" / source.name
            self.assertTrue(unclassified.exists(), "_미분류 로 가야 한다")
            self.assertFalse(
                (destination_root / source.name).exists(), "정리 폴더 루트에 남으면 안 된다"
            )

    def _classify_names(self, root, names, order=None):
        classifier = backend_server.FilenameClassifier(
            backend_server.FILE_ORGANIZER_RULES,
            root,
            order or ["vehicle", "category", "item", "detail"],
        )
        results = backend_server.classify_batch(classifier, [Path(n) for n in names])
        return dict(zip(names, results))

    def test_missing_item_no_is_borrowed_from_the_same_product_in_the_batch(self) -> None:
        """품번이 빠진 파일은 같은 제품 파일의 품번을 참고해 태깅한다."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            root.mkdir(parents=True)
            donor = "JM_67312-DZ000_DASH LWR_OP20_완성도_260825.zip"
            receiver = "JM DASH LWR 성형해석 리포트 260825.ppt"
            found = self._classify_names(root, [donor, receiver])

            borrowed = found[receiver]
            self.assertEqual(borrowed.item_no, "67312-DZ000")
            self.assertEqual(borrowed.family, "67312")
            self.assertIsNotNone(borrowed.target_dir)
            self.assertEqual(
                borrowed.target_dir.relative_to(root).parts,
                ("JM", "03. 문서", "67312", "01. 성형해석"),
            )
            self.assertTrue(
                any(donor in reason for reason in borrowed.reasons),
                "어느 파일에서 참고했는지 근거에 남아야 한다",
            )
            # 파일명에 품번이 적힌 쪽은 그대로다.
            self.assertEqual(found[donor].item_no, "67312-DZ000")

    def test_borrowed_item_no_scores_lower_than_one_read_from_the_filename(self) -> None:
        """빌린 품번은 파일명에 적힌 품번보다 약한 근거이므로 신뢰도가 낮다."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            root.mkdir(parents=True)
            donor = "JM_67312-DZ000_DASH LWR_OP20_완성도_260825.zip"
            receiver = "JM DASH LWR 성형해석 리포트 260825.ppt"
            found = self._classify_names(root, [donor, receiver])
            self.assertLess(found[receiver].confidence, found[donor].confidence)

    def test_item_no_is_not_borrowed_when_the_same_product_has_two_item_numbers(self) -> None:
        """같은 제품에 품번이 둘이면 지어내지 않고 미분류로 둔다."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            root.mkdir(parents=True)
            receiver = "JM DASH LWR 성형해석 리포트 260825.ppt"
            found = self._classify_names(
                root,
                [
                    "JM_67312-DZ000_DASH LWR_OP20_완성도_260825.zip",
                    "JM_67313-DZ000_DASH LWR_OP30_패턴도_260825.zip",
                    receiver,
                ],
            )
            self.assertEqual(found[receiver].item_no, "")
            self.assertIsNone(found[receiver].target_dir)

    def test_item_no_is_not_borrowed_across_different_vehicles(self) -> None:
        """차종이 다르면 품번을 물려받지 않는다."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            root.mkdir(parents=True)
            receiver = "JM DASH LWR 성형해석 리포트 260825.ppt"
            found = self._classify_names(
                root,
                ["XM_71612-DZ000_DASH LWR_OP20_완성도_260825.zip", receiver],
            )
            self.assertEqual(found[receiver].item_no, "")
            self.assertIsNone(found[receiver].target_dir)

    def test_item_no_is_not_borrowed_across_different_product_names(self) -> None:
        """품명이 다르면 같은 차종이어도 물려받지 않는다."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "organized"
            root.mkdir(parents=True)
            receiver = "JM DASH LWR 성형해석 리포트 260825.ppt"
            found = self._classify_names(
                root,
                ["JM_67312-DZ000_PNL HOOD INR_OP20_완성도_260825.zip", receiver],
            )
            self.assertEqual(found[receiver].item_no, "")

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
