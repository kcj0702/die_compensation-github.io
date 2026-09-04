"""Local-only API that connects the React UI to the three vision engines."""

from __future__ import annotations

import base64
import json
import math
import os
import sqlite3
import sys
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

import cv2
import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.concurrency import run_in_threadpool


UI_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = UI_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEVIATION_DIR = PROJECT_DIR / "deviation_extraction"
FILE_ORGANIZER_DIR = WORKSPACE_DIR / "file_organizer"


def _load_local_environment(env_path: Path) -> None:
    """Load non-committed local settings without overriding process variables."""
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_local_environment(UI_DIR / ".env")

# deviation_extraction currently uses local-style imports (import config), so
# its own folder must precede the project root on sys.path.
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(DEVIATION_DIR))
sys.path.insert(0, str(FILE_ORGANIZER_DIR))

from label_detector import (  # noqa: E402
    build_blue_annotation_mask, build_scan_mask, detect_labels,
)
from colormap_reader import build_lut  # noqa: E402
from point_extractor import _sample_deviation_color  # noqa: E402
from vlm_reader import LabelValueReader  # noqa: E402
from label_removal.remove_labels import (  # noqa: E402
    build_scan_mask as build_label_removal_scan_mask,
    create_versions,
    detect_exact_hsv_leader_lines,
    detect_label_boxes,
)
from product_alignment.alignment import (  # noqa: E402
    Alignment,
    estimate_alignment,
    is_inside,
    map_point,
    warp_scan_mask,
)
from product_alignment.compose import render_alignment_overlay  # noqa: E402
from product_alignment.masks import (  # noqa: E402
    build_product_mask,
    build_scan_mask as build_part_silhouette,
)
from product_alignment.registry import (  # noqa: E402
    AlignmentStore,
    ProductLibrary,
    part_number_from_name,
    read_image,
)
from point_selection import select_key_points  # noqa: E402
from sheet_export import (  # noqa: E402
    SheetAnnotation,
    SheetPoint,
    SheetView,
    TitleBlock,
    build_sheet,
    crop_view,
    stack_workbooks,
)
from cad_import.mesh_io import is_mesh_file, load_mesh, to_web_mesh  # noqa: E402
from zero_line_detection.visualize import make_overlay  # noqa: E402
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line  # noqa: E402
from zero_line_detection.hybrid_ui import detect_hybrid_zero_line  # noqa: E402
from zero_line_detection.zero_criteria import (  # noqa: E402
    candidates_to_mask, find_zero_candidates,
)
from zero_line_detection.polygonize import draw_polygons, polygonize  # noqa: E402
from zero_line_detection.zero_polyline import (  # noqa: E402
    draw_zero_polylines, extract_zero_polylines,
)
from zero_line_detection.zero_boundary import (  # noqa: E402
    draw_zero_boundary, find_boundary_anchors, grow_patches,
)
from core import (  # noqa: E402
    AXES, AXIS_LABELS, FilenameClassifier, classify_batch, execute_batch,
    is_valid_folder_order, load_folder_order, load_rules, migrate_folder_structure,
    save_folder_order, write_history,
)
from storage import (  # noqa: E402
    DatabaseError as FileDatabaseError,
    MariaDBRepository,
    load_database_url,
    safe_database_label,
    save_database_url,
)


FILE_ORGANIZER_RULES = load_rules(FILE_ORGANIZER_DIR / "rules.json")
DEFAULT_FOLDER_ROOT = Path(FILE_ORGANIZER_RULES["destination_root"])
_ORGANIZER_PATHS_FILE = FILE_ORGANIZER_DIR / ".organizer_paths.json"


def _load_organizer_paths_override() -> dict[str, str]:
    """웹에서 저장한 원본/정리 대상 경로 오버라이드. .env 의 AJIN_* 값이 항상 우선한다."""
    if not _ORGANIZER_PATHS_FILE.is_file():
        return {}
    try:
        data = json.loads(_ORGANIZER_PATHS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _initial_organizer_root(env_name: str, override_key: str, rules_value: str) -> Path:
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return Path(env_value).resolve()
    override_value = _load_organizer_paths_override().get(override_key, "").strip()
    if override_value:
        return Path(override_value).resolve()
    return Path(rules_value).resolve()


FOLDER_ROOT = _initial_organizer_root(
    "AJIN_FOLDER_ROOT", "destinationRoot", str(DEFAULT_FOLDER_ROOT)
)
FILE_SOURCE_ROOT = _initial_organizer_root(
    "AJIN_FILE_SOURCE_ROOT", "sourceRoot", FILE_ORGANIZER_RULES["source_root"]
)
FILE_STAGING_ROOT = UI_DIR / "backend" / "file_staging"
FILE_LOG_ROOT = UI_DIR / "backend" / "file_operation_logs"
MAX_FILE_ORGANIZER_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
QWEN_CACHE_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-VL-3B-Instruct"
    / "snapshots"
)
WORKSPACE_QWEN_DIR = WORKSPACE_DIR / "models" / "Qwen2.5-VL-3B-Instruct"
QWEN_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
)
_reader: LabelValueReader | None = None
_reader_lock = threading.Lock()

# 보정치 수동 수정 이력을 남기는 로컬 DB. 스캔 이미지·도면 데이터를 외부로 보낼 수 없는
# 사내 보안정책과 같은 이유로, 외부 SQL 서버가 아니라 이 백엔드가 로컬에 직접 들고 있다.
CORRECTION_DB_PATH = Path(
    os.environ.get(
        "AJIN_CORRECTION_DB_PATH", str(UI_DIR / "backend" / "correction_history.db")
    )
)
CORRECTION_DB_URL = os.environ.get("AJIN_CORRECTION_DB_URL", "").strip()
CORRECTION_ACTIONS = frozenset(
    {"edit", "reset_auto", "reset_all", "restore_before", "reapply", "revise"}
)
CORRECTION_MODES = frozenset({"auto", "manual"})


class CorrectionDatabaseError(RuntimeError):
    """Raised when the shared correction-history database is unavailable."""


def _active_correction_db_url() -> str:
    return os.environ.get("AJIN_CORRECTION_DB_URL", CORRECTION_DB_URL).strip()


def _mysql_connection_config(database_url: str) -> dict[str, Any]:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"mysql", "mysql+mysqlconnector"}:
        raise ValueError(
            "AJIN_CORRECTION_DB_URL must start with mysql:// or "
            "mysql+mysqlconnector://."
        )
    database = unquote(parsed.path.lstrip("/"))
    if not parsed.hostname or not parsed.username or not database:
        raise ValueError(
            "AJIN_CORRECTION_DB_URL requires a host, user, and database name."
        )
    query = parse_qs(parsed.query)
    charset = query.get("charset", ["utf8mb4"])[-1]
    timeout_text = query.get("connect_timeout", ["10"])[-1]
    try:
        connection_timeout = max(1, min(60, int(timeout_text)))
    except ValueError as exc:
        raise ValueError("connect_timeout must be an integer from 1 to 60.") from exc
    config: dict[str, Any] = {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username),
        "password": unquote(parsed.password or ""),
        "database": database,
        "charset": charset,
        "connection_timeout": connection_timeout,
        "autocommit": False,
    }
    ssl_ca = query.get("ssl_ca", [""])[-1].strip()
    if ssl_ca:
        config.update(
            ssl_ca=ssl_ca,
            ssl_verify_cert=True,
            ssl_verify_identity=True,
        )
    return config


class _CorrectionConnection:
    def __init__(self, raw_connection: Any, dialect: str) -> None:
        self.raw_connection = raw_connection
        self.dialect = dialect

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        cursor = (
            self.raw_connection.cursor(dictionary=True)
            if self.dialect == "mysql"
            else self.raw_connection.cursor()
        )
        statement = query.replace("?", "%s") if self.dialect == "mysql" else query
        try:
            cursor.execute(statement, params)
        except Exception as exc:
            cursor.close()
            if self.dialect == "mysql":
                raise CorrectionDatabaseError(
                    "MySQL correction-history query failed."
                ) from exc
            raise
        return cursor


@contextmanager
def _get_correction_db() -> Iterator[_CorrectionConnection]:
    database_url = _active_correction_db_url()
    if database_url:
        try:
            import mysql.connector

            raw_connection = mysql.connector.connect(
                **_mysql_connection_config(database_url)
            )
        except (ImportError, ValueError) as exc:
            raise CorrectionDatabaseError(str(exc)) from exc
        except Exception as exc:
            raise CorrectionDatabaseError(
                "Could not connect to the MySQL correction-history database."
            ) from exc
        connection = _CorrectionConnection(raw_connection, "mysql")
    else:
        raw_connection = sqlite3.connect(CORRECTION_DB_PATH)
        raw_connection.row_factory = sqlite3.Row
        connection = _CorrectionConnection(raw_connection, "sqlite")
    try:
        yield connection
        raw_connection.commit()
    except CorrectionDatabaseError:
        raw_connection.rollback()
        raise
    except Exception as exc:
        raw_connection.rollback()
        if connection.dialect == "mysql":
            raise CorrectionDatabaseError(
                "MySQL correction-history operation failed."
            ) from exc
        raise
    finally:
        raw_connection.close()


def _init_correction_db() -> None:
    with _get_correction_db() as conn:
        if conn.dialect == "mysql":
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS correction_history (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    part_no VARCHAR(255) NOT NULL,
                    scan_name VARCHAR(255) NOT NULL,
                    point_id VARCHAR(128) NOT NULL,
                    old_value DOUBLE NULL,
                    new_value DOUBLE NULL,
                    worker VARCHAR(255) NULL,
                    created_at VARCHAR(32) NOT NULL,
                    action VARCHAR(32) NOT NULL DEFAULT 'edit',
                    old_mode VARCHAR(16) NULL,
                    new_mode VARCHAR(16) NULL,
                    coefficient DOUBLE NULL,
                    source_entry_id BIGINT UNSIGNED NULL,
                    PRIMARY KEY (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            columns = {
                row["Field"]
                for row in conn.execute("SHOW COLUMNS FROM correction_history").fetchall()
            }
            migrations = (
                ("action", "VARCHAR(32) NOT NULL DEFAULT 'edit'"),
                ("old_mode", "VARCHAR(16) NULL"),
                ("new_mode", "VARCHAR(16) NULL"),
                ("coefficient", "DOUBLE NULL"),
                ("source_entry_id", "BIGINT UNSIGNED NULL"),
            )
            for name, declaration in migrations:
                if name not in columns:
                    conn.execute(
                        f"ALTER TABLE correction_history ADD COLUMN {name} {declaration}"
                    )
            indexes = {
                row["Key_name"]
                for row in conn.execute("SHOW INDEX FROM correction_history").fetchall()
            }
            if "idx_correction_history_part_scan_id" not in indexes:
                conn.execute(
                    "CREATE INDEX idx_correction_history_part_scan_id "
                    "ON correction_history (part_no, scan_name, id DESC)"
                )
            return

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS correction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                part_no TEXT NOT NULL,
                scan_name TEXT NOT NULL,
                point_id TEXT NOT NULL,
                old_value REAL,
                new_value REAL,
                worker TEXT,
                created_at TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'edit',
                old_mode TEXT,
                new_mode TEXT,
                coefficient REAL,
                source_entry_id INTEGER
            )
            """
        )
        # The first version of the local history DB only had the columns above
        # ``created_at``.  Keep those rows intact and add metadata in place so a
        # UI/backend update never discards an operator's existing audit trail.
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(correction_history)").fetchall()
        }
        migrations = (
            ("action", "TEXT NOT NULL DEFAULT 'edit'"),
            ("old_mode", "TEXT"),
            ("new_mode", "TEXT"),
            ("coefficient", "REAL"),
            ("source_entry_id", "INTEGER"),
        )
        for name, declaration in migrations:
            if name not in columns:
                conn.execute(
                    f"ALTER TABLE correction_history ADD COLUMN {name} {declaration}"
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_correction_history_part_scan_id "
            "ON correction_history (part_no, scan_name, id DESC)"
        )
        current_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if current_version < 1:
            conn.execute("PRAGMA user_version = 1")


try:
    _init_correction_db()
except CorrectionDatabaseError as exc:
    # 서버 시작 시점에 보정 이력 DB(사내망 MySQL)가 잠깐 안 닿아도, 그것과
    # 무관한 나머지 기능(품번 파일 정리 등)까지 통째로 못 뜨게 하지는 않는다.
    # 보정 이력 쪽 API는 요청 시점에 다시 연결을 시도하고, 여전히 안 되면
    # 이미 503으로 안내한다.
    print(f"[경고] 보정 이력 DB 초기화 실패, 나머지 기능은 계속 시작합니다: {exc}")


def _correction_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "partNo": row["part_no"],
        "scanName": row["scan_name"],
        "pointId": row["point_id"],
        "oldValue": row["old_value"],
        "newValue": row["new_value"],
        "worker": row["worker"],
        "createdAt": row["created_at"],
        "action": row["action"],
        "oldMode": row["old_mode"],
        "newMode": row["new_mode"],
        "coefficient": row["coefficient"],
        "sourceEntryId": row["source_entry_id"],
    }


def _optional_finite_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number or null.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be a finite number or null.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number or null.")
    return number


def _optional_correction_mode(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in CORRECTION_MODES:
        choices = ", ".join(sorted(CORRECTION_MODES))
        raise ValueError(f"{field_name} must be one of: {choices}, or null.")
    return value


def _optional_source_entry_id(value: Any) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 2**63 - 1
    ):
        raise ValueError("sourceEntryId must be a positive integer or null.")
    return value


# 제품데이터는 품번당 한 장으로 고정이고 스캔은 차수마다 새로 들어온다. 한 번
# 등록해 두면 이후 스캔은 지금처럼 파일 하나만 올려도 자동으로 짝이 맞는다.
PRODUCT_LIBRARY = ProductLibrary()
ALIGNMENT_STORE = AlignmentStore()


def _is_complete_qwen_model(candidate: Path) -> bool:
    """Return True only when the local model and every indexed shard exist."""
    try:
        if not candidate.is_dir() or any(
            not (candidate / filename).is_file() for filename in QWEN_REQUIRED_FILES
        ):
            return False
        index = json.loads(
            (candidate / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        shards = set(index.get("weight_map", {}).values())
        return bool(shards) and all((candidate / shard).is_file() for shard in shards)
    except (OSError, TypeError, ValueError):
        return False


def _find_qwen_model() -> Path | None:
    configured = os.environ.get("AJIN_QWEN_MODEL_PATH")
    candidates = [Path(configured)] if configured else []
    candidates.append(WORKSPACE_QWEN_DIR)
    if QWEN_CACHE_DIR.is_dir():
        candidates.extend(
            sorted(QWEN_CACHE_DIR.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
        )
    for candidate in candidates:
        candidate = candidate.expanduser()
        if _is_complete_qwen_model(candidate):
            return candidate.resolve()
    return None


def _get_qwen_reader() -> LabelValueReader:
    global _reader
    if _reader is not None:
        return _reader
    with _reader_lock:
        if _reader is None:
            import torch

            model_path = _find_qwen_model()
            if model_path is None:
                raise FileNotFoundError("Qwen2.5-VL-3B 로컬 모델을 찾지 못했습니다.")
            configured_device = os.environ.get("AJIN_QWEN_DEVICE", "auto").strip().lower()
            if configured_device in {"", "auto"}:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA용 PyTorch를 사용할 수 없어 Qwen 판독을 건너뜁니다."
                    )
                device = "cuda"
            else:
                device = configured_device
            _reader = LabelValueReader(
                model_id=str(model_path),
                device=device,
                local_files_only=True,
                use_8bit=device.startswith("cuda"),
            )
    return _reader


def _read_qwen_values(
    reader: LabelValueReader,
    crops: list[Any],
) -> tuple[list[float | None], str | None]:
    """Read every value from Qwen while preserving the crop-to-result mapping.

    The regular batched reader remains the fast path. Only labels that are
    still unread after that complete result get the reader's bounded focused
    retry. A malformed batch is discarded entirely and re-read crop by crop so
    a short response can never shift values onto neighbouring measurement
    points. No image-colour or substitute value is introduced here.
    """
    batch_problem: str | None = None
    focused_indices: list[int] = []
    try:
        batch_values = list(reader.read_values(crops))
        if len(batch_values) == len(crops):
            values = batch_values
            retry_indices = []
            focused_indices = [
                index for index, value in enumerate(values) if value is None
            ]
        else:
            values = [None] * len(crops)
            retry_indices = list(range(len(crops)))
            batch_problem = (
                "Qwen 일괄 판독 결과 수가 라벨 수와 일치하지 않았습니다 "
                f"({len(batch_values)}/{len(crops)})."
            )
    except Exception as exc:
        values = [None] * len(crops)
        retry_indices = list(range(len(crops)))
        batch_problem = f"Qwen 일괄 판독에 실패했습니다: {exc}"

    singleton_failures = 0
    for index in retry_indices:
        try:
            singleton_values = list(reader.read_values([crops[index]]))
        except Exception:
            singleton_failures += 1
            continue
        if len(singleton_values) != 1:
            singleton_failures += 1
            continue
        values[index] = singleton_values[0]

    focused_failures = 0
    focused_reader = getattr(reader, "read_value_focused", None)
    if callable(focused_reader):
        for index in focused_indices:
            try:
                values[index] = focused_reader(crops[index])
            except Exception:
                focused_failures += 1

    warning_parts: list[str] = []
    if batch_problem:
        warning_parts.append(batch_problem)
    if retry_indices:
        warning_parts.append(
            f"동일 Qwen으로 매핑 불명확 라벨 {len(retry_indices)}개를 개별 재판독했습니다."
        )
    if singleton_failures:
        warning_parts.append(
            f"개별 판독 실패 {singleton_failures}개는 결과에서 제외했습니다."
        )
    if focused_failures:
        warning_parts.append(
            f"집중 판독 오류 {focused_failures}개는 결과에서 제외했습니다."
        )
    warning = " ".join(warning_parts) or None
    return values, warning


def _png_data_url(image: np.ndarray, *, rgb: bool = False) -> str:
    source = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image
    ok, encoded = cv2.imencode(".png", source)
    if not ok:
        raise ValueError("결과 이미지를 PNG로 변환하지 못했습니다.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _decode_image(payload: bytes) -> np.ndarray:
    if not payload:
        raise ValueError("비어 있는 파일입니다.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("이미지는 한 장당 60MB 이하만 처리할 수 있습니다.")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("지원되는 이미지 파일이 아니거나 파일이 손상되었습니다.")
    return image


def _marker_centers_from_version_difference(
    labels_inpainted: np.ndarray,
    labels_points_inpainted: np.ndarray,
) -> list[tuple[float, float]]:
    """Return marker centers isolated by the version-2/version-4 difference."""
    if labels_inpainted.shape != labels_points_inpainted.shape:
        raise ValueError("Label-removal result sizes do not match.")
    difference = np.max(
        cv2.absdiff(labels_inpainted, labels_points_inpainted), axis=2
    )
    difference_mask = np.where(difference >= 8, 255, 0).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        difference_mask, connectivity=8
    )
    centers: list[tuple[float, float]] = []
    maximum_area = max(500, int(difference_mask.size * 0.001))
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if 3 <= area <= maximum_area:
            centers.append(
                (float(centroids[component, 0]), float(centroids[component, 1]))
            )
    return centers


def _refine_candidates_from_removed_markers(
    image: np.ndarray,
    candidates: list,
    labels_inpainted: np.ndarray,
    labels_points_inpainted: np.ndarray,
) -> int:
    """Replace heuristic endpoints with centers measured from removed markers."""
    marker_centers = _marker_centers_from_version_difference(
        labels_inpainted, labels_points_inpainted
    )
    if not marker_centers or not candidates:
        return 0

    label_boxes = detect_label_boxes(image)
    removal_scan_mask = build_label_removal_scan_mask(image)
    _, point_specs, point_boxes = detect_exact_hsv_leader_lines(
        image,
        label_boxes,
        removal_scan_mask,
        return_point_boxes=True,
    )

    unused_centers = set(range(len(marker_centers)))
    center_records: list[
        tuple[tuple[int, int, int, int], tuple[int, int]]
    ] = []
    for spec, point_box in zip(point_specs, point_boxes):
        spec_x, spec_y, radius, _ = spec
        nearest: tuple[float, int] | None = None
        for center_index in unused_centers:
            center_x, center_y = marker_centers[center_index]
            distance = math.hypot(center_x - spec_x, center_y - spec_y)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, center_index)
        maximum_gap = max(8.0, float(radius * 3))
        if nearest is None or nearest[0] > maximum_gap:
            continue
        _, center_index = nearest
        center_records.append(
            (point_box, (int(spec_x), int(spec_y)))
        )
        unused_centers.remove(center_index)

    refined = 0
    used_records: set[int] = set()
    for candidate in candidates:
        x, y, box_width, box_height = candidate.box
        candidate_box = (x, y, x + box_width, y + box_height)
        best_record: tuple[float, int] | None = None
        candidate_area = max(1, box_width * box_height)
        for record_index, (point_box, _) in enumerate(center_records):
            if record_index in used_records:
                continue
            ix0 = max(candidate_box[0], point_box[0])
            iy0 = max(candidate_box[1], point_box[1])
            ix1 = min(candidate_box[2], point_box[2])
            iy1 = min(candidate_box[3], point_box[3])
            intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            point_area = max(
                1, (point_box[2] - point_box[0]) * (point_box[3] - point_box[1])
            )
            overlap = intersection / float(candidate_area + point_area - intersection)
            if best_record is None or overlap > best_record[0]:
                best_record = (overlap, record_index)
        if best_record is not None and best_record[0] >= 0.75:
            _, record_index = best_record
            point = center_records[record_index][1]
            used_records.add(record_index)
        else:
            # Direct-contact markers have no normal leader component/box
            # record. Match their newly added difference component to the
            # already reliable compact-marker fallback coordinate.
            if candidate.point_xy is None:
                continue
            nearest_center: tuple[float, int] | None = None
            for center_index in unused_centers:
                center_x, center_y = marker_centers[center_index]
                distance = math.hypot(
                    center_x - candidate.point_xy[0],
                    center_y - candidate.point_xy[1],
                )
                if nearest_center is None or distance < nearest_center[0]:
                    nearest_center = (distance, center_index)
            if nearest_center is None or nearest_center[0] > 12.0:
                continue
            _, center_index = nearest_center
            center_x, center_y = marker_centers[center_index]
            point = (int(round(center_x)), int(round(center_y)))
            unused_centers.remove(center_index)
        candidate.point_xy = point
        candidate.traced = True
        refined += 1
    return refined


def _resolve_product_image(
    filename: str, uploaded: np.ndarray | None
) -> tuple[np.ndarray | None, str | None, list[str]]:
    """Pick the uploaded product image first, then a registered one."""
    warnings: list[str] = []
    if uploaded is not None:
        return uploaded, "업로드한 이미지", warnings
    part_number = part_number_from_name(filename)
    if part_number is None:
        return None, None, warnings
    match = PRODUCT_LIBRARY.find(part_number)
    if match is None:
        warnings.append(f"품번 {part_number}의 제품데이터가 등록되어 있지 않습니다.")
        return None, None, warnings
    if not match.exact:
        warnings.append(f"{part_number}에 정확히 맞는 제품데이터가 없어 {match.part_number}를 사용했습니다.")
    try:
        return read_image(match.path), f"등록됨 · {match.part_number}", warnings
    except ValueError as exc:
        warnings.append(str(exc))
        return None, None, warnings


def _align_to_product(
    image: np.ndarray,
    product_image: np.ndarray,
    part_number: str | None,
    flip_x: bool | None,
    flip_y: bool | None,
) -> tuple[Any, np.ndarray | None, list[str]]:
    """Estimate the scan-to-product transform, reusing a confirmed direction."""
    warnings: list[str] = []
    scan_silhouette = build_part_silhouette(image)
    product_mask = build_product_mask(product_image)
    if flip_x is None and flip_y is None and part_number:
        saved = ALIGNMENT_STORE.load(part_number)
        if saved is not None:
            flip_x, flip_y = saved.flip_x, saved.flip_y
            warnings.append(f"{part_number}에 확정 저장된 방향을 사용했습니다.")
    alignment = estimate_alignment(scan_silhouette, product_mask, flip_x=flip_x, flip_y=flip_y)
    overlay = render_alignment_overlay(product_image, warp_scan_mask(alignment, scan_silhouette))
    return alignment, overlay, warnings + list(alignment.warnings)


def _apply_flip(image: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    if flip_x and flip_y:
        return cv2.flip(image, -1)
    if flip_x:
        return cv2.flip(image, 1)
    if flip_y:
        return cv2.flip(image, 0)
    return image


def analyze_image(
    image: np.ndarray,
    filename: str,
    product_upload: np.ndarray | None = None,
    flip_x: bool | None = None,
    flip_y: bool | None = None,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    errors: dict[str, str] = {}
    part_number = part_number_from_name(filename)

    product_image, product_source, product_warnings = _resolve_product_image(
        filename, product_upload
    )
    alignment = None
    alignment_overlay: np.ndarray | None = None
    if product_image is not None:
        try:
            # 라벨을 읽기 전에 방향부터 바로잡는다. 사람이 이미 확정한 방향이거나
            # (저장된 정렬, 명시적 flipX/flipY) 자동 판정이 충분히 확실할 때만
            # 실제 픽셀을 뒤집는다 -- 애매한 첫 추측만으로 원본을 뒤집으면 오히려
            # 멀쩡한 스캔을 망가뜨릴 수 있다.
            saved_alignment = (
                ALIGNMENT_STORE.load(part_number)
                if flip_x is None and flip_y is None and part_number
                else None
            )
            probe, _, _ = _align_to_product(image, product_image, part_number, flip_x, flip_y)
            trusted = (
                flip_x is not None
                or flip_y is not None
                or saved_alignment is not None
                or probe.confident
            )
            if trusted and (probe.flip_x or probe.flip_y):
                image = _apply_flip(image, probe.flip_x, probe.flip_y)
                # 픽셀을 이미 바로 세웠으니, 다음 정렬은 반전 없이 배율·평행이동만
                # 다시 잡는다 -- 그러지 않으면 같은 방향을 두 번 뒤집는다.
                flip_x, flip_y = False, False
            alignment, alignment_overlay, alignment_warnings = _align_to_product(
                image, product_image, part_number, flip_x, flip_y
            )
            if saved_alignment is not None:
                # flip_x/flip_y 를 여기서 False 로 확정해 버려서 _align_to_product
                # 내부의 "저장된 방향을 불러왔다" 안내가 두 번째 호출에서는 뜨지
                # 않는다. 어떤 방향을 썼는지는 사용자에게 그대로 알려줘야 한다.
                alignment_warnings = list(alignment_warnings) + [
                    f"{part_number}에 확정 저장된 방향(좌우 {saved_alignment.flip_x}, "
                    f"상하 {saved_alignment.flip_y})을 사용했습니다."
                ]
            product_warnings.extend(alignment_warnings)
        except Exception as exc:  # engine errors must be shown per engine
            errors["product"] = str(exc)

    # 위에서 방향을 바로잡았을 수 있으니 크기는 여기서 읽는다. 순수 반전은
    # 가로세로를 바꾸지 않지만, 그래도 최종 image 기준으로 읽는 편이 안전하다.
    height, width = image.shape[:2]

    clean_image: np.ndarray | None = None
    points_removed_image: np.ndarray | None = None
    label_count = 0
    try:
        label_count = len(detect_label_boxes(image))
        label_versions = create_versions(image)
        clean_image = label_versions["2_labels_inpainted"]
        points_removed_image = label_versions["4_labels_points_inpainted"]
    except Exception as exc:  # engine errors must be shown per engine
        errors["label"] = str(exc)

    zero_output = None
    zero_overlay: np.ndarray | None = None
    zero_candidates: list = []
    zero_datum_mask: np.ndarray | None = None
    zero_lines: list = []
    zero_anchors: list = []
    zero_patches: list = []
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        zero_output = detect_zero_line(rgb, ZeroLineConfig(), source_name=filename)
        overlay_base = cv2.cvtColor(
            clean_image if clean_image is not None else image,
            cv2.COLOR_BGR2RGB,
        )

        # 색만 보고 잡은 0 밴드에서, 실제로 기준이 될 수 있는 곳만 추린다.
        # 편차가 0에 가깝고 + 주변이 평탄한 곳이 스프링백의 기준면/기준선이다.
        zero_candidates, flat, _ = find_zero_candidates(
            zero_output.values,
            zero_output.part_mask,
            float(zero_output.result.tolerance),
        )
        zero_datum_mask = candidates_to_mask(zero_candidates, flat, top_n=8)

        # 2026-08-25 아진산업 방문 확인 사항: 제로라인의 시작/끝점은
        # 부품 가장자리에서 편차 부호가 바뀌는 지점이다. RING SUNROOF
        # 실측 시트로 정량 검증했다 — 실제 패치 7개 중 5개 적중,
        # 평균 위치 오차 대각선의 7.2% (zero_line_detection/README.md 참고).
        # 아직 완벽하지 않으므로 후보로 제시하고 최종 판단은 사람이 한다.
        zero_anchors = find_boundary_anchors(
            zero_output.values, zero_output.part_mask
        )
        zero_patches = grow_patches(
            zero_output.values, zero_output.part_mask, zero_anchors,
            tolerance=float(zero_output.result.tolerance),
        )
        zero_lines = extract_zero_polylines(
            zero_output.values, zero_output.part_mask
        )
        if zero_patches:
            zero_overlay = draw_zero_boundary(overlay_base, zero_anchors, zero_patches)
        elif zero_lines:
            zero_overlay = draw_zero_polylines(overlay_base, zero_lines)
        elif zero_datum_mask is not None and zero_datum_mask.any():
            zero_overlay = draw_polygons(
                overlay_base, polygonize(zero_datum_mask, preset="balanced")
            )
        else:
            zero_overlay = make_overlay(
                overlay_base,
                zero_output.mask,
                zero_output.centerline,
                zero_crossing=zero_output.zero_crossing,
            )
        # UI 응답은 합의한 하이브리드 엔진 결과를 우선한다. 위의 기존
        # 결과는 후보/앵커 호환 필드를 유지하기 위한 보조 계산이다.
        hybrid_zero = detect_hybrid_zero_line(image, filename)
        zero_datum_mask = hybrid_zero.mask
        zero_overlay = hybrid_zero.overlay_rgb
        zero_lines = hybrid_zero.lines
    except Exception as exc:
        errors["zero"] = str(exc)

    points: list[dict[str, Any]] = []
    qwen_reads = 0
    unread_labels = 0
    detected_candidates = 0
    valid_candidates_count = 0
    deviation_warnings: list[str] = []
    try:
        candidates = detect_labels(image)
        if clean_image is not None and points_removed_image is not None:
            _refine_candidates_from_removed_markers(
                image,
                candidates,
                clean_image,
                points_removed_image,
            )
        detected_candidates = len(candidates)
        deviation_scan_mask = build_scan_mask(image)
        scan_present = bool(np.any(deviation_scan_mask))
        valid_candidates = [
            candidate
            for candidate in candidates
            if scan_present
            and candidate.traced
            and candidate.point_xy is not None
            and 0 <= candidate.point_xy[0] < width
            and 0 <= candidate.point_xy[1] < height
            and deviation_scan_mask[candidate.point_xy[1], candidate.point_xy[0]] > 0
        ]
        valid_candidates_count = len(valid_candidates)
        if candidates and not scan_present:
            deviation_warnings.append(
                "3D 스캔 본체를 확인하지 못해 편차값을 추출하지 않았습니다."
            )
        crops = []
        for candidate in valid_candidates:
            box_x, box_y, box_w, box_h = candidate.box
            # Keep dense neighbouring labels and their blue leaders out of the
            # OCR crop while retaining one pixel around the detected pill box.
            pad = 1
            crop = image[
                max(0, box_y - pad):min(height, box_y + box_h + pad),
                max(0, box_x - pad):min(width, box_x + box_w + pad),
            ]
            from PIL import Image as PILImage

            crops.append(PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

        qwen_values: list[float | None] = [None] * len(crops)
        qwen_failure: str | None = None
        if crops:
            try:
                reader = _get_qwen_reader()
                qwen_values, qwen_failure = _read_qwen_values(reader, crops)
            except Exception as exc:
                qwen_failure = str(exc)

        for candidate, qwen_value in zip(valid_candidates, qwen_values):
            x, y = candidate.point_xy
            if qwen_value is None:
                unread_labels += 1
                continue
            try:
                numeric_value = float(qwen_value)
            except (TypeError, ValueError, OverflowError):
                unread_labels += 1
                continue
            if not math.isfinite(numeric_value):
                unread_labels += 1
                continue
            value = round(numeric_value, 3)
            qwen_reads += 1
            point_id = f"P-{qwen_reads:02d}"
            points.append(
                {
                    "id": point_id,
                    "xPx": x,
                    "yPx": y,
                    "x": round(x / width * 100, 3),
                    "y": round(y / height * 100, 3),
                    "value": value,
                    "labelColor": candidate.label_color,
                    "confidence": "ok",
                }
            )
        if qwen_failure:
            deviation_warnings.append(qwen_failure)
        if unread_labels:
            deviation_warnings.append(
                f"Qwen이 숫자를 판독하지 못한 라벨 {unread_labels}개는 "
                "결과에서 제외했습니다."
            )
    except Exception as exc:
        errors["deviation"] = str(exc)

    # 보정시트에는 검출된 라벨을 전부 올리지 않는다. 국소 극값과 부호가 바뀌는
    # 지점만 기본으로 켜고, 나머지는 화면에서 켤 수 있게 남겨 둔다.
    selection = select_key_points(points)
    key_reasons = {key.point_id: list(key.reasons) for key in selection.keys}
    for point in points:
        reasons = key_reasons.get(point["id"])
        if reasons:
            point["keyReasons"] = reasons

    # 좌표만 옮긴다. 편차값을 보정치로 바꾸는 계산은 이 단계가 하지 않는다.
    transferred = 0
    if alignment is not None:
        product_width, product_height = alignment.product_size
        for point in points:
            product_x, product_y = map_point(alignment, point["xPx"], point["yPx"])
            if not is_inside(alignment, product_x, product_y):
                continue
            point["xProduct"] = round(product_x / product_width * 100, 3)
            point["yProduct"] = round(product_y / product_height * 100, 3)
            transferred += 1
        if points and transferred < len(points):
            product_warnings.append(
                f"제품데이터 범위를 벗어난 포인트 {len(points) - transferred}개는 "
                "전사하지 않았습니다."
            )

    hybrid_zero = locals().get("hybrid_zero")
    zero_regions = hybrid_zero.regions if hybrid_zero is not None else (len(zero_output.result.regions) if zero_output is not None else 0)
    zero_ratio = hybrid_zero.ratio if hybrid_zero is not None else (zero_output.result.zero_ratio if zero_output is not None else 0.0)
    zero_warnings = hybrid_zero.warnings if hybrid_zero is not None else (list(zero_output.warnings) if zero_output is not None else [])
    warnings = zero_warnings + deviation_warnings + product_warnings

    if qwen_reads:
        value_mode = "Qwen2.5-VL-3B 로컬 판독"
    else:
        value_mode = "판독 결과 없음"

    return {
        "source": {"name": filename, "width": width, "height": height},
        "partNumber": part_number,
        "cleanImage": _png_data_url(clean_image) if clean_image is not None else None,
        "productImage": (
            _png_data_url(product_image) if product_image is not None else None
        ),
        "productSource": product_source,
        "alignment": alignment.to_dict() if alignment is not None else None,
        "alignmentOverlay": (
            _png_data_url(alignment_overlay)
            if alignment_overlay is not None
            else None
        ),
        "keySelection": selection.to_dict(),
        "zeroOverlay": _png_data_url(zero_overlay, rgb=True) if zero_overlay is not None else None,
        "zeroMask": (
            _png_data_url(zero_datum_mask)
            if zero_datum_mask is not None and zero_datum_mask.any()
            else (_png_data_url(zero_output.mask) if zero_output is not None else None)
        ),
        "zeroCandidates": [c.to_dict() for c in zero_candidates[:8]],
        "zeroLines": zero_lines if hybrid_zero is not None else [l.to_dict() for l in zero_lines],
        "zeroAnchors": [a.to_dict() for a in zero_anchors],
        "zeroPatches": [pt.to_dict() for pt in zero_patches],
        "points": points,
        "stats": {
            "labelsRemoved": label_count,
            "pointsDetected": len(points),
            "detectedCandidates": detected_candidates,
            "validCandidates": valid_candidates_count,
            "zeroRegions": zero_regions,
            "zeroRatio": round(zero_ratio, 4),
            "zeroTolerance": 0.6,
            "qwenReads": qwen_reads,
            "qwenUnread": unread_labels,
            "pointsTransferred": transferred,
        },
        "warnings": warnings,
        "warningsByEngine": {
            "deviation": deviation_warnings,
            "zero": zero_warnings,
            "product": product_warnings,
        },
        "errors": errors,
        "valueMode": value_mode,
    }


async def health(_: Request) -> JSONResponse:
    import torch

    model_path = _find_qwen_model()
    return JSONResponse(
        {
            "ok": True,
            "engines": [
                "label_removal",
                "deviation_extraction",
                "zero_line_detection",
                "product_alignment",
            ],
            "folderAvailable": FOLDER_ROOT.is_dir(),
            "registeredProducts": len(PRODUCT_LIBRARY.registered()),
            "qwenCached": model_path is not None,
            "qwenLoaded": _reader is not None,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "correctionDatabase": (
                "mysql" if _active_correction_db_url() else "sqlite"
            ),
        }
    )


def _optional_flag(form: Any, name: str) -> bool | None:
    """Read a tri-state form flag: absent means 'decide automatically'."""
    raw = form.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


async def analyze(request: Request) -> JSONResponse:
    try:
        form = await request.form(max_files=2, max_fields=6, max_part_size=MAX_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "이미지 파일이 필요합니다."}, status_code=400)
        payload = await upload.read()
        image = _decode_image(payload)

        product_upload = form.get("product")
        product_image = None
        if product_upload is not None and hasattr(product_upload, "read"):
            product_image = _decode_image(await product_upload.read())

        result = await run_in_threadpool(
            analyze_image,
            image,
            getattr(upload, "filename", "scan.png"),
            product_image,
            _optional_flag(form, "flipX"),
            _optional_flag(form, "flipY"),
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


def realign_image(
    image: np.ndarray,
    filename: str,
    product_upload: np.ndarray | None,
    flip_x: bool | None,
    flip_y: bool | None,
    points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Redo only the alignment and the point transfer.

    Changing the orientation must not cost another Qwen pass, so the values
    already read stay untouched and only the coordinates are recomputed.
    """
    part_number = part_number_from_name(filename)
    product_image, product_source, warnings = _resolve_product_image(
        filename, product_upload
    )
    if product_image is None:
        raise ValueError("제품데이터 이미지가 없어 정렬을 다시 계산할 수 없습니다.")

    alignment, overlay, alignment_warnings = _align_to_product(
        image, product_image, part_number, flip_x, flip_y
    )
    warnings.extend(alignment_warnings)

    product_width, product_height = alignment.product_size
    mapped: list[dict[str, Any]] = []
    for point in points:
        try:
            x_px = float(point["xPx"])
            y_px = float(point["yPx"])
        except (KeyError, TypeError, ValueError):
            continue
        product_x, product_y = map_point(alignment, x_px, y_px)
        if not is_inside(alignment, product_x, product_y):
            continue
        mapped.append(
            {
                "id": point.get("id"),
                "xProduct": round(product_x / product_width * 100, 3),
                "yProduct": round(product_y / product_height * 100, 3),
            }
        )
    if points and len(mapped) < len(points):
        warnings.append(
            f"제품데이터 범위를 벗어난 포인트 {len(points) - len(mapped)}개는 "
            "전사하지 않았습니다."
        )

    return {
        "partNumber": part_number,
        "productImage": _png_data_url(product_image),
        "productSource": product_source,
        "alignment": alignment.to_dict(),
        "alignmentOverlay": _png_data_url(overlay),
        "points": mapped,
        "warnings": warnings,
    }


async def realign(request: Request) -> JSONResponse:
    try:
        form = await request.form(max_files=2, max_fields=8, max_part_size=MAX_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "스캔 이미지가 필요합니다."}, status_code=400)
        image = _decode_image(await upload.read())

        product_upload = form.get("product")
        product_image = None
        if product_upload is not None and hasattr(product_upload, "read"):
            product_image = _decode_image(await product_upload.read())

        raw_points = form.get("points")
        points = json.loads(str(raw_points)) if raw_points else []
        if not isinstance(points, list):
            return JSONResponse({"error": "points는 배열이어야 합니다."}, status_code=400)

        result = await run_in_threadpool(
            realign_image,
            image,
            getattr(upload, "filename", "scan.png"),
            product_image,
            _optional_flag(form, "flipX"),
            _optional_flag(form, "flipY"),
            points,
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


_ANNOTATION_KINDS = {"rect", "ellipse", "text", "arrow"}


def _apply_placement(view: SheetView, placement: Any) -> None:
    """Position a view's picture inside the sheet the way the UI shows it.

    ``placement`` is ``{x, y, w, h}`` in percent of the UI's sheet canvas.
    That canvas maps 1:1 to the Excel drawing area (from just below the
    title block to the bottom guide), so multiplying by the drawing area's
    size gives the exact pixel box for the picture. When placement is
    missing the view is left alone and ``default_layout`` falls back to
    its auto-fit behaviour.
    """
    if not isinstance(placement, dict):
        return
    try:
        x = float(placement["x"]) / 100.0
        y = float(placement["y"]) / 100.0
        w = float(placement["w"]) / 100.0
        h = float(placement["h"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return
    if w <= 0 or h <= 0:
        return
    from sheet_export import config as sheet_config  # local — avoid startup cost
    drawing_height = sheet_config.DRAWING_BOTTOM - sheet_config.DRAWING_TOP
    view.box = (
        x * sheet_config.SHEET_WIDTH,
        sheet_config.DRAWING_TOP + y * drawing_height,
        w * sheet_config.SHEET_WIDTH,
        h * drawing_height,
    )


def _sheet_annotations(raw: Any) -> list[SheetAnnotation]:
    """Convert the UI annotation payload into ``SheetAnnotation`` values.

    The frontend keeps ``x``/``y``/``w``/``h`` in percent (0–100) of the
    preview layer, which shares its coordinate frame with the product image
    dropped into the sheet, so a simple ``/100`` gives the ratios the
    exporter expects. Unknown kinds are dropped so a bad payload cannot
    poison the whole sheet.
    """
    if not isinstance(raw, list):
        return []
    result: list[SheetAnnotation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).lower()
        if kind not in _ANNOTATION_KINDS:
            continue
        try:
            x = float(item.get("x", 0.0)) / 100.0
            y = float(item.get("y", 0.0)) / 100.0
            w = float(item.get("w", 0.0)) / 100.0
            h = float(item.get("h", 0.0)) / 100.0
        except (TypeError, ValueError):
            continue
        color = str(item.get("color") or "#e8802f")
        text = str(item.get("text") or "")
        font_size_raw = item.get("fontSize")
        font_size: float | None
        try:
            font_size = float(font_size_raw) if font_size_raw is not None else None
        except (TypeError, ValueError):
            font_size = None
        font_family = item.get("fontFamily")
        font_family = str(font_family) if isinstance(font_family, str) else None
        result.append(SheetAnnotation(
            kind=kind, x_ratio=x, y_ratio=y, w_ratio=w, h_ratio=h,
            color=color, text=text, font_size_px=font_size,
            font_family=font_family,
        ))
    return result


def build_sheet_bytes(
    product_image: np.ndarray, payload: dict[str, Any],
    previous_bytes: bytes | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Turn the sheet the operator sees into the company Excel form.

    Point coordinates arrive as percentages of the product image, which is the
    same frame the UI draws in, so no alignment work is repeated here. When
    ``previous_bytes`` is provided, the new block is appended to that
    workbook via ``sheet_export.stack_workbooks``.
    """
    raw_points = payload.get("points") or []
    points = [
        SheetPoint(
            point_id=str(item.get("id", index)),
            text=str(item.get("text") or f"{float(item['value']):+.1f}"),
            x_ratio=float(item["x"]) / 100.0,
            y_ratio=float(item["y"]) / 100.0,
        )
        for index, item in enumerate(raw_points)
    ]

    raw_annotations = payload.get("annotations") or []
    annotations = _sheet_annotations(raw_annotations)

    title_values = payload.get("title") or {}
    title = TitleBlock(
        management_no=str(title_values.get("managementNo", "")),
        part_name=str(title_values.get("partName", "")),
        process=str(title_values.get("process", "")),
        part_no=str(title_values.get("partNo", "")),
        material=str(title_values.get("material", "")),
        applied_date=str(title_values.get("appliedDate", "")),
    )

    front_view = SheetView(
        image=product_image, points=points, annotations=annotations,
    )
    _apply_placement(front_view, payload.get("frontPlacement"))
    views = [front_view]
    skipped: list[str] = []
    for index, region in enumerate(payload.get("details") or [], start=1):
        label = str(region.get("label") or f"DETAIL {index}")
        try:
            detail_view = crop_view(
                product_image,
                points,
                (
                    float(region["x"]) / 100.0,
                    float(region["y"]) / 100.0,
                    float(region["w"]) / 100.0,
                    float(region["h"]) / 100.0,
                ),
                label,
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append(f"{label}: {exc}")
            continue
        _apply_placement(detail_view, region.get("placement"))
        views.append(detail_view)

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "correction-sheet.xlsx"
        report = build_sheet(
            views,
            title,
            output=output,
            title_fonts=payload.get("titleFonts") if isinstance(payload.get("titleFonts"), dict) else None,
            title_font_sizes=payload.get("titleFontSizes") if isinstance(payload.get("titleFontSizes"), dict) else None,
            point_font_family=str(payload.get("pointFontFamily") or "").strip() or None,
        )
        data = output.read_bytes()

    stacked_warnings: list[str] = []
    if previous_bytes:
        try:
            data = stack_workbooks(previous_bytes, data)
        except Exception as exc:
            stacked_warnings.append(f"이전 시트에 이어붙이지 못했습니다: {exc}")

    summary = {
        "pictures": report.pictures,
        "labels": report.labels,
        "leaders": report.leaders,
        "warnings": report.warnings + skipped + stacked_warnings,
    }
    return data, summary


async def sheet(request: Request) -> Response:
    """Return the correction sheet as an .xlsx download.

    The optional ``previous`` file part carries an existing sheet the new
    block should be appended to. When present, ``sheet_export.stack_workbooks``
    merges the freshly built single-block workbook onto the previous one so
    the two print as consecutive pages of the same file.
    """
    try:
        form = await request.form(max_files=2, max_fields=4, max_part_size=MAX_UPLOAD_BYTES)
        payload = json.loads(str(form.get("payload") or "{}"))
        if not isinstance(payload, dict):
            return JSONResponse({"error": "payload는 객체여야 합니다."}, status_code=400)

        upload = form.get("product")
        product_image = None
        if upload is not None and hasattr(upload, "read"):
            product_image = _decode_image(await upload.read())
        else:
            part_number = str(payload.get("partNumber", "")).strip().upper()
            match = PRODUCT_LIBRARY.find(part_number) if part_number else None
            if match is None:
                return JSONResponse(
                    {"error": "제품데이터 이미지를 찾지 못했습니다."}, status_code=400
                )
            product_image = read_image(match.path)

        previous_upload = form.get("previous")
        previous_bytes: bytes | None = None
        if previous_upload is not None and hasattr(previous_upload, "read"):
            previous_bytes = await previous_upload.read() or None

        data, summary = await run_in_threadpool(
            build_sheet_bytes, product_image, payload, previous_bytes
        )
        name = f"{payload.get('partNumber') or 'correction'}-보정시트.xlsx"
        quoted = quote(name)
        return Response(
            data,
            media_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quoted}",
                # HTTP 헤더는 latin-1만 허용한다. 한글 경고는 JSON escape로
                # 보내고, UI가 필요하면 정상적으로 JSON.parse 할 수 있다.
                "X-Sheet-Summary": json.dumps(summary, ensure_ascii=True),
            },
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def products(request: Request) -> JSONResponse:
    """List the registered product-data images, or register one."""
    if request.method == "GET":
        return JSONResponse(
            {
                "entries": PRODUCT_LIBRARY.entries(),
                "directory": str(PRODUCT_LIBRARY.directory),
            }
        )
    try:
        form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "제품데이터 이미지가 필요합니다."}, status_code=400)

        raw_part_number = str(form.get("partNumber", "")).strip().upper()
        part_number = raw_part_number or part_number_from_name(
            getattr(upload, "filename", "")
        )
        if not part_number:
            return JSONResponse(
                {"error": "품번을 찾지 못했습니다. partNumber를 함께 보내 주세요."},
                status_code=400,
            )
        if part_number_from_name(f"{part_number}.png") != part_number:
            return JSONResponse(
                {"error": f"품번 형식이 올바르지 않습니다: {part_number}"},
                status_code=400,
            )

        image = _decode_image(await upload.read())
        path = await run_in_threadpool(PRODUCT_LIBRARY.register, part_number, image)
        return JSONResponse({"partNumber": part_number, "path": str(path)})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def confirm_alignment(request: Request) -> JSONResponse:
    """Store a human-confirmed orientation so later scans reuse it.

    The orientation cannot always be decided from the masks -- a symmetric
    panel scores every flip almost the same -- so this is what turns one
    person's check into a permanent answer for that part number.
    """
    try:
        payload = await request.json()
        part_number = str(payload.get("partNumber", "")).strip().upper()
        if not part_number:
            return JSONResponse({"error": "품번이 필요합니다."}, status_code=400)

        if payload.get("forget"):
            removed = ALIGNMENT_STORE.forget(part_number)
            return JSONResponse({"partNumber": part_number, "removed": removed})

        alignment_payload = payload.get("alignment")
        if not isinstance(alignment_payload, dict):
            return JSONResponse({"error": "alignment 값이 필요합니다."}, status_code=400)

        alignment = Alignment.from_dict(alignment_payload)
        alignment.overridden = True
        path = ALIGNMENT_STORE.save(part_number, alignment)
        return JSONResponse({"partNumber": part_number, "path": str(path)})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


def sample_point(image: np.ndarray, x_norm: float, y_norm: float) -> dict[str, Any]:
    """클릭한 지점의 표면색을 컬러바로 되짚어 편차값을 추정한다.

    라벨을 판독해 얻는 값과 달리 이건 어디까지나 추정치다. 파이프라인도 색 역산을
    판독값 검증용으로만 쓰고 있어(point_extractor 의 교차검증), 여기서도 같은 함수를
    그대로 불러 써서 기준이 갈라지지 않게 한다.
    """
    height, width = image.shape[:2]
    x = int(round(x_norm / 100.0 * (width - 1)))
    y = int(round(y_norm / 100.0 * (height - 1)))
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("이미지 범위를 벗어난 지점입니다.")

    scan_mask = build_scan_mask(image)
    if np.any(scan_mask) and scan_mask[y, x] == 0:
        raise ValueError("스캔 본체 바깥은 편차값을 추정할 수 없습니다.")

    annotation_mask = build_blue_annotation_mask(image)
    color = _sample_deviation_color(image, (x, y), scan_mask, annotation_mask)
    if color is None:
        raise ValueError("주변 표면색을 읽지 못했습니다. 다른 지점을 눌러 보세요.")

    value = build_lut(image).to_value(color)
    return {
        "xPx": x,
        "yPx": y,
        "x": round(x / width * 100, 3),
        "y": round(y / height * 100, 3),
        "value": round(float(value), 3),
        "source": "colormap",
        "bgr": [int(c) for c in color],
    }


async def sample(request: Request) -> JSONResponse:
    try:
        form = await request.form(max_files=1, max_fields=6, max_part_size=MAX_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "이미지 파일이 필요합니다."}, status_code=400)
        x_norm = float(str(form.get("x", "")))
        y_norm = float(str(form.get("y", "")))
        image = _decode_image(await upload.read())
        result = await run_in_threadpool(sample_point, image, x_norm, y_norm)
        return JSONResponse(result)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


async def create_correction(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise TypeError("Request body must be a JSON object.")
        part_no = str(body.get("partNo") or "").strip()
        point_id = str(body.get("pointId") or "").strip()
        if not part_no or not point_id:
            return JSONResponse({"error": "partNo와 pointId가 필요합니다."}, status_code=400)
        scan_name = str(body.get("scanName") or "").strip()
        old_value = _optional_finite_number(body.get("oldValue"), "oldValue")
        new_value = _optional_finite_number(body.get("newValue"), "newValue")
        worker = str(body.get("worker") or "").strip() or None
        action = body["action"] if "action" in body else "edit"
        if not isinstance(action, str) or action not in CORRECTION_ACTIONS:
            choices = ", ".join(sorted(CORRECTION_ACTIONS))
            raise ValueError(f"action must be one of: {choices}.")
        old_mode = _optional_correction_mode(body.get("oldMode"), "oldMode")
        new_mode = _optional_correction_mode(body.get("newMode"), "newMode")
        coefficient = _optional_finite_number(body.get("coefficient"), "coefficient")
        source_entry_id = _optional_source_entry_id(body.get("sourceEntryId"))
        created_at = datetime.now().isoformat(timespec="seconds")
        with _get_correction_db() as conn:
            cursor = conn.execute(
                "INSERT INTO correction_history "
                "(part_no, scan_name, point_id, old_value, new_value, worker, created_at, "
                "action, old_mode, new_mode, coefficient, source_entry_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    part_no,
                    scan_name,
                    point_id,
                    old_value,
                    new_value,
                    worker,
                    created_at,
                    action,
                    old_mode,
                    new_mode,
                    coefficient,
                    source_entry_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM correction_history WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return JSONResponse(_correction_row_to_dict(row))
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except CorrectionDatabaseError:
        return JSONResponse(
            {"error": "중앙 보정 이력 데이터베이스에 연결할 수 없습니다."},
            status_code=503,
        )


async def delete_correction(request: Request) -> JSONResponse:
    raw_id = request.query_params.get("id")
    try:
        entry_id = int(raw_id) if raw_id is not None else None
    except ValueError:
        entry_id = None
    if not entry_id or entry_id <= 0:
        return JSONResponse({"error": "삭제할 이력 id가 필요합니다."}, status_code=400)
    try:
        with _get_correction_db() as conn:
            cursor = conn.execute(
                "DELETE FROM correction_history WHERE id = ?", (entry_id,)
            )
    except CorrectionDatabaseError:
        return JSONResponse(
            {"error": "중앙 보정 이력 데이터베이스에 연결할 수 없습니다."},
            status_code=503,
        )
    if cursor.rowcount == 0:
        return JSONResponse({"error": "해당 이력을 찾을 수 없습니다."}, status_code=404)
    return JSONResponse({"id": entry_id, "deleted": True})


async def list_corrections(request: Request) -> JSONResponse:
    part_no = request.query_params.get("partNo")
    scan_name = request.query_params.get("scanName")
    query = "SELECT * FROM correction_history"
    conditions: list[str] = []
    params: list[Any] = []
    if part_no:
        conditions.append("part_no = ?")
        params.append(part_no)
    if scan_name:
        conditions.append("scan_name = ?")
        params.append(scan_name)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC LIMIT 200"
    try:
        with _get_correction_db() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
    except CorrectionDatabaseError:
        return JSONResponse(
            {"error": "중앙 보정 이력 데이터베이스에 연결할 수 없습니다."},
            status_code=503,
        )
    return JSONResponse({"entries": [_correction_row_to_dict(row) for row in rows]})


def _safe_folder(relative_path: str) -> Path:
    candidate = (FOLDER_ROOT / relative_path).resolve()
    if candidate != FOLDER_ROOT and FOLDER_ROOT not in candidate.parents:
        raise ValueError("허용된 품번별 폴더 밖은 조회할 수 없습니다.")
    return candidate


def _folder_entry(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": path.relative_to(FOLDER_ROOT).as_posix(),
        "isDirectory": path.is_dir(),
        "size": stat.st_size if path.is_file() else None,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


async def folders(request: Request) -> JSONResponse:
    if not FOLDER_ROOT.is_dir():
        return JSONResponse({"available": False, "entries": []})
    try:
        relative_path = request.query_params.get("path", "")
        folder = _safe_folder(relative_path)
        if not folder.is_dir():
            return JSONResponse({"error": "폴더가 아닙니다."}, status_code=400)
        entries = sorted(
            (_folder_entry(path) for path in folder.iterdir()),
            key=lambda item: (not item["isDirectory"], item["name"].lower()),
        )
        return JSONResponse(
            {
                "available": True,
                "rootName": FOLDER_ROOT.name,
                "path": relative_path.replace("\\", "/").strip("/"),
                "entries": entries,
            }
        )
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _active_file_database_url() -> str:
    return os.environ.get("AJIN_FILE_DB_URL", "").strip() or load_database_url(
        FILE_ORGANIZER_DIR
    )


def _active_folder_order() -> list[str]:
    return load_folder_order(FILE_ORGANIZER_DIR, FILE_ORGANIZER_RULES.get("folder_order"))


def _path_is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _safe_organizer_source(value: str) -> Path:
    candidate = Path(value).resolve()
    allowed_roots = (FILE_SOURCE_ROOT, FILE_STAGING_ROOT.resolve())
    if not any(_path_is_within(candidate, root) for root in allowed_roots):
        raise ValueError("허용된 원본 또는 업로드 폴더 밖의 파일은 처리할 수 없습니다.")
    if not candidate.is_file():
        raise ValueError("원본 파일을 찾을 수 없습니다.")
    return candidate


def _safe_organizer_target(relative_path: str) -> Path:
    candidate = (FOLDER_ROOT / relative_path).resolve()
    if not _path_is_within(candidate, FOLDER_ROOT):
        raise ValueError("허용된 정리 대상 폴더 밖으로 저장할 수 없습니다.")
    return candidate


def _organizer_item(path: Path, classification: Any, *, source_kind: str) -> dict[str, Any]:
    target_dir = classification.target_dir
    if target_dir is None or target_dir == FOLDER_ROOT:
        target_relative = ""
    else:
        try:
            target_relative = target_dir.relative_to(FOLDER_ROOT).as_posix()
        except ValueError:
            target_relative = ""
    stat = path.stat()
    return {
        "id": uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())).hex,
        "name": path.name,
        "sourcePath": str(path),
        "sourceKind": source_kind,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "customer": classification.customer,
        "itemNo": classification.item_no,
        "family": classification.family,
        "productName": classification.product_name,
        "process": classification.process,
        "categoryKey": classification.category_key,
        "categoryLabel": classification.category_label,
        "confidence": classification.confidence,
        "reasons": classification.reasons,
        "targetDir": target_relative,
        "targetPath": (Path(target_relative) / path.name).as_posix(),
        "matchedProductFolder": classification.matched_product_folder,
        "detailPath": classification.detail_path,
    }


def _scan_organizer_source() -> list[dict[str, Any]]:
    if not FILE_SOURCE_ROOT.is_dir():
        return []
    ignored = {name.casefold() for name in FILE_ORGANIZER_RULES.get("ignored_names", [])}
    paths = sorted(
        (
            path for path in FILE_SOURCE_ROOT.rglob("*")
            if path.is_file()
            and not path.name.startswith("~$")
            and path.name.casefold() not in ignored
        ),
        key=lambda path: path.name.casefold(),
    )
    classifier = FilenameClassifier(FILE_ORGANIZER_RULES, FOLDER_ROOT, _active_folder_order())
    classifications = classify_batch(classifier, paths)
    return [
        _organizer_item(path, classification, source_kind="source")
        for path, classification in zip(paths, classifications)
    ]


def _file_database_status(check_connection: bool) -> dict[str, Any]:
    database_url = _active_file_database_url()
    response: dict[str, Any] = {
        "configured": bool(database_url),
        "label": safe_database_label(database_url),
        "connected": None,
        "catalogCount": 0,
        "operationCount": 0,
    }
    if not database_url or not check_connection:
        return response
    try:
        summary = MariaDBRepository(database_url).get_summary()
        response.update(summary, connected=True)
    except FileDatabaseError as exc:
        response.update(connected=False, error=str(exc))
    return response


async def file_organizer_status(request: Request) -> JSONResponse:
    check_database = request.query_params.get("checkDb") == "1"
    database = await run_in_threadpool(_file_database_status, check_database)
    return JSONResponse(
        {
            "sourceRoot": str(FILE_SOURCE_ROOT),
            "destinationRoot": str(FOLDER_ROOT),
            "sourceAvailable": FILE_SOURCE_ROOT.is_dir(),
            "destinationAvailable": FOLDER_ROOT.is_dir(),
            "database": database,
        }
    )


async def file_organizer_scan(_request: Request) -> JSONResponse:
    try:
        items = await run_in_threadpool(_scan_organizer_source)
        return JSONResponse({"items": items, "count": len(items)})
    except OSError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def file_organizer_upload(request: Request) -> JSONResponse:
    FILE_STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        form = await request.form(
            max_files=100,
            max_fields=4,
            max_part_size=MAX_FILE_ORGANIZER_UPLOAD_BYTES,
        )
    except Exception as exc:
        return JSONResponse({"error": f"업로드를 읽지 못했습니다: {exc}"}, status_code=400)
    uploads = form.getlist("files")
    if not uploads:
        return JSONResponse({"error": "업로드할 파일이 없습니다."}, status_code=400)
    destinations: list[Path] = []
    for upload in uploads:
        filename = Path(getattr(upload, "filename", "") or "").name
        if not filename or filename.startswith("~$"):
            continue
        upload_dir = FILE_STAGING_ROOT / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=False)
        destination = upload_dir / filename
        total = 0
        try:
            with destination.open("wb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_FILE_ORGANIZER_UPLOAD_BYTES:
                        raise ValueError("파일 한 개는 500MB 이하만 업로드할 수 있습니다.")
                    stream.write(chunk)
            destinations.append(destination)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            upload_dir.rmdir()
            return JSONResponse({"error": str(exc)}, status_code=400)
        finally:
            await upload.close()
    classifier = FilenameClassifier(FILE_ORGANIZER_RULES, FOLDER_ROOT, _active_folder_order())
    classifications = classify_batch(classifier, destinations)
    items = [
        _organizer_item(destination, classification, source_kind="upload")
        for destination, classification in zip(destinations, classifications)
    ]
    return JSONResponse({"items": items, "count": len(items)})


def _discard_staged_upload(source_path: str) -> None:
    """대기열에서 지운 업로드 파일의 임시 사본을 정리한다.

    원본 폴더(FILE_SOURCE_ROOT)에서 스캔한 실제 회사 파일은 절대 지우지 않도록,
    file_staging 아래에 있는 업로드 임시 파일일 때만 지운다 — 그 밖의 경로는
    조용히 무시한다(대기열에서 빼는 것 자체는 프론트엔드가 이미 처리했으므로).
    """
    try:
        candidate = Path(source_path).resolve()
    except OSError:
        return
    if not _path_is_within(candidate, FILE_STAGING_ROOT.resolve()):
        return
    candidate.unlink(missing_ok=True)
    try:
        candidate.parent.rmdir()
    except OSError:
        pass


async def file_organizer_discard(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        source_path = str(payload.get("sourcePath", ""))
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not source_path:
        return JSONResponse({"error": "sourcePath가 필요합니다."}, status_code=400)
    await run_in_threadpool(_discard_staged_upload, source_path)
    return JSONResponse({"ok": True})


def _execute_file_organizer(payload: dict[str, Any]) -> dict[str, Any]:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items or len(raw_items) > 200:
        raise ValueError("한 번에 1~200개 파일을 선택해야 합니다.")
    operation = str(payload.get("operation", "copy"))
    conflict = str(payload.get("conflict", "rename"))
    if operation not in {"copy", "move"} or conflict not in {"rename", "skip", "overwrite"}:
        raise ValueError("지원하지 않는 파일 작업 설정입니다.")

    parsed_items: list[tuple[Path, str | None]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("파일 항목 형식이 올바르지 않습니다.")
        source = _safe_organizer_source(str(raw_item.get("sourcePath", "")))
        # 분류가 안 된 파일은 화면이 대상 폴더를 빈 문자열로 돌려보내고 그 자리에
        # '_미분류' 를 보여 준다. 빈 값을 "사용자가 고른 폴더"로 받으면 정리 폴더
        # 루트가 되어, 화면 표시와 달리 파일이 품번 폴더들 옆에 쌓인다.
        manual_target = str(raw_item.get("targetDir") or "").strip() or None
        parsed_items.append((source, manual_target))

    classifier = FilenameClassifier(FILE_ORGANIZER_RULES, FOLDER_ROOT, _active_folder_order())
    sources = [source for source, _ in parsed_items]
    batch_classifications = classify_batch(classifier, sources)

    pairs: list[tuple[Path, Path]] = []
    classifications: dict[str, Any] = {}
    for (source, manual_target), classification in zip(parsed_items, batch_classifications):
        target_dir = (
            _safe_organizer_target(manual_target)
            if manual_target is not None
            else classification.target_dir
        )
        if target_dir is None:
            target_dir = FOLDER_ROOT / FILE_ORGANIZER_RULES.get("unclassified_folder", "_미분류")
        pairs.append((source, target_dir / source.name))
        classifications[MariaDBRepository._source_key(source)] = classification

    results = execute_batch(pairs, operation=operation, conflict=conflict)
    write_history(results, FILE_LOG_ROOT)
    database_note = "MariaDB 미설정 · 로컬 감사 로그 저장"
    database_url = _active_file_database_url()
    if database_url:
        try:
            db_result = MariaDBRepository(database_url).record_batch(
                batch_id=str(uuid.uuid4()),
                results=results,
                classifications=classifications,
                storage_root=FOLDER_ROOT,
            )
            database_note = (
                f"MariaDB 이력 {db_result.operation_count}건 · "
                f"카탈로그 {db_result.catalog_count}건"
            )
        except FileDatabaseError as exc:
            database_note = f"MariaDB 기록 실패 · 로컬 감사 로그 보존 ({exc})"

    for source, result in zip(sources, results):
        if result.status == "success" and _path_is_within(source, FILE_STAGING_ROOT.resolve()):
            try:
                source.unlink(missing_ok=True)
                source.parent.rmdir()
            except OSError:
                pass
    return {
        "results": [
            {
                "source": result.source,
                "destination": result.destination,
                "operation": result.operation,
                "status": result.status,
                "message": result.message,
            }
            for result in results
        ],
        "databaseNote": database_note,
    }


async def file_organizer_execute(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        response = await run_in_threadpool(_execute_file_organizer, payload)
        return JSONResponse(response)
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def file_organizer_database(request: Request) -> JSONResponse:
    if request.method == "GET":
        status = await run_in_threadpool(_file_database_status, True)
        return JSONResponse(status)
    try:
        payload = await request.json()
        database_url = str(payload.get("databaseUrl", "")).strip()
        if not database_url:
            raise ValueError("MariaDB 연결 URL을 입력해 주세요.")
        version = await run_in_threadpool(
            MariaDBRepository(database_url).test_and_initialize
        )
        save_database_url(FILE_ORGANIZER_DIR, database_url)
        return JSONResponse(
            {
                "configured": True,
                "connected": True,
                "label": safe_database_label(database_url),
                "version": version,
            }
        )
    except (FileDatabaseError, OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


_AXIS_OPTIONS = [{"id": axis, "label": AXIS_LABELS[axis]} for axis in AXES]


async def file_organizer_folder_order(request: Request) -> JSONResponse:
    if request.method == "GET":
        order = await run_in_threadpool(_active_folder_order)
        return JSONResponse({"folderOrder": order, "axes": _AXIS_OPTIONS})
    try:
        payload = await request.json()
        order = payload.get("folderOrder")
        if not is_valid_folder_order(order):
            raise ValueError(
                "folderOrder는 품번, 차종, 카테고리, 세부 하위폴더를 "
                "각각 한 번씩 포함해야 합니다."
            )
        migration = await run_in_threadpool(
            migrate_folder_structure, FOLDER_ROOT, FILE_ORGANIZER_RULES, order,
        )
        await run_in_threadpool(save_folder_order, FILE_ORGANIZER_DIR, order)
        return JSONResponse(
            {
                "folderOrder": order,
                "axes": _AXIS_OPTIONS,
                "migration": {
                    "moved": migration.moved,
                    "skipped": migration.skipped,
                    "structureMoved": migration.structure_moved,
                    "errors": migration.errors,
                },
            }
        )
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def _organizer_path_locks() -> tuple[bool, bool]:
    source_locked = bool(os.environ.get("AJIN_FILE_SOURCE_ROOT", "").strip())
    destination_locked = bool(os.environ.get("AJIN_FOLDER_ROOT", "").strip())
    return source_locked, destination_locked


def _set_organizer_roots(source_root: str, destination_root: str) -> None:
    global FILE_SOURCE_ROOT, FOLDER_ROOT
    source_path = Path(source_root).expanduser().resolve()
    destination_path = Path(destination_root).expanduser().resolve()
    source_path.mkdir(parents=True, exist_ok=True)
    destination_path.mkdir(parents=True, exist_ok=True)
    FILE_ORGANIZER_DIR.mkdir(parents=True, exist_ok=True)
    _ORGANIZER_PATHS_FILE.write_text(
        json.dumps(
            {"sourceRoot": str(source_path), "destinationRoot": str(destination_path)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    FILE_SOURCE_ROOT = source_path
    FOLDER_ROOT = destination_path


async def file_organizer_paths(request: Request) -> JSONResponse:
    source_locked, destination_locked = _organizer_path_locks()
    if request.method == "GET":
        return JSONResponse(
            {
                "sourceRoot": str(FILE_SOURCE_ROOT),
                "destinationRoot": str(FOLDER_ROOT),
                "sourceLocked": source_locked,
                "destinationLocked": destination_locked,
            }
        )
    try:
        if source_locked or destination_locked:
            raise ValueError(
                "ui/.env 의 AJIN_FILE_SOURCE_ROOT/AJIN_FOLDER_ROOT 로 경로가 고정되어 "
                "있어 웹에서 바꿀 수 없습니다. .env 값을 지우거나 바꾼 뒤 서버를 다시 "
                "시작해 주세요."
            )
        payload = await request.json()
        source_root = str(payload.get("sourceRoot", "")).strip()
        destination_root = str(payload.get("destinationRoot", "")).strip()
        if not source_root or not destination_root:
            raise ValueError("원본 폴더와 정리 대상 폴더 경로를 모두 입력해 주세요.")
        await run_in_threadpool(_set_organizer_roots, source_root, destination_root)
        return JSONResponse(
            {
                "sourceRoot": str(FILE_SOURCE_ROOT),
                "destinationRoot": str(FOLDER_ROOT),
                "sourceLocked": False,
                "destinationLocked": False,
            }
        )
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def file_organizer_reveal(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        which = str(payload.get("which", ""))
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    target = {"source": FILE_SOURCE_ROOT, "destination": FOLDER_ROOT}.get(which)
    if target is None:
        return JSONResponse({"error": "which은 source 또는 destination이어야 합니다."}, status_code=400)
    if not target.is_dir():
        return JSONResponse({"error": "폴더가 아직 없습니다."}, status_code=400)
    if not hasattr(os, "startfile"):
        return JSONResponse({"error": "이 서버 환경에서는 탐색기 열기를 지원하지 않습니다."}, status_code=400)
    try:
        os.startfile(str(target))  # noqa: S606 - 로컬 데스크톱 전용 앱, 사용자 자신의 PC 탐색기를 연다.
    except OSError as exc:
        return JSONResponse({"error": f"탐색기를 열지 못했습니다: {exc}"}, status_code=400)
    return JSONResponse({"ok": True})


def _load_cad_mesh(path: Path, name: str) -> dict[str, Any]:
    if not is_mesh_file(path):
        raise ValueError("현재 UI CAD 뷰어는 STL, PLY, OBJ, GLB, 3MF 형식을 지원합니다. STEP 지원은 별도 OCCT 설치가 필요합니다.")
    return {**to_web_mesh(load_mesh(path), name=name, source_format=path.suffix.lower().lstrip(".")), "holes": [], "planes": [], "counts": {"cylinders": 0, "holes": 0, "planes": 0}}


async def cad(request: Request) -> JSONResponse:
    """Read a local mesh upload and return a compact Three.js payload."""
    try:
        form = await request.form()
        uploaded = form.get("file")
        if uploaded is None or not getattr(uploaded, "filename", ""):
            raise ValueError("CAD 파일을 선택하세요.")
        suffix = Path(uploaded.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(await uploaded.read())
        try:
            payload = await run_in_threadpool(_load_cad_mesh, temp_path, uploaded.filename)
        finally:
            temp_path.unlink(missing_ok=True)
        return JSONResponse(payload)
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


app = Starlette(
    routes=[
        Route("/api/health", health, methods=["GET"]),
        Route("/api/analyze", analyze, methods=["POST"]),
        Route("/api/sample", sample, methods=["POST"]),
        Route("/api/corrections", create_correction, methods=["POST"]),
        Route("/api/corrections", list_corrections, methods=["GET"]),
        Route("/api/corrections", delete_correction, methods=["DELETE"]),
        Route("/api/folders", folders, methods=["GET"]),
        Route("/api/realign", realign, methods=["POST"]),
        Route("/api/sheet", sheet, methods=["POST"]),
        Route("/api/products", products, methods=["GET", "POST"]),
        Route("/api/alignment", confirm_alignment, methods=["POST"]),
        Route("/api/cad", cad, methods=["POST"]),
        Route("/api/file-organizer/status", file_organizer_status, methods=["GET"]),
        # 원본 폴더를 읽어 분류 결과만 돌려주는 조회다(바꾸는 것이 없다) — 화면도
        # GET 으로 부른다. POST 로 열어 두면 "원본 스캔" 버튼이 405 를 받는다.
        Route("/api/file-organizer/scan", file_organizer_scan, methods=["GET"]),
        Route("/api/file-organizer/upload", file_organizer_upload, methods=["POST"]),
        Route("/api/file-organizer/discard", file_organizer_discard, methods=["POST"]),
        Route("/api/file-organizer/execute", file_organizer_execute, methods=["POST"]),
        Route("/api/file-organizer/database", file_organizer_database, methods=["POST"]),
        Route("/api/file-organizer/folder-order", file_organizer_folder_order, methods=["GET", "POST"]),
        Route("/api/file-organizer/paths", file_organizer_paths, methods=["GET", "POST"]),
        Route("/api/file-organizer/reveal", file_organizer_reveal, methods=["POST"]),
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    # 시트 생성 결과 요약은 헤더로 오므로 브라우저가 읽을 수 있게 열어 준다.
    expose_headers=["Content-Disposition", "X-Sheet-Summary"],
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
