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
from collections import OrderedDict
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
from sheet_export import (  # noqa: E402
    SheetAnnotation,
    SheetPoint,
    SheetView,
    TitleBlock,
    build_sheet,
    crop_view,
    stack_workbooks,
)
from zero_line_detection.visualize import make_overlay  # noqa: E402
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line  # noqa: E402
from zero_line_detection.hybrid_ui import detect_hybrid_zero_line  # noqa: E402
from zero_line_detection import zero_shapes  # noqa: E402
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
MAX_CAD_UPLOAD_BYTES = 300 * 1024 * 1024

# 분석 결과와 CAD 파싱 결과는 후속 3D 작업에서 재사용한다. STEP 재파싱과
# Qwen 재판독을 피하고, 여러 CAD 탭을 동시에 열 수 있도록 원 브랜치와
# 동일한 LRU 크기를 유지한다.
_analysis_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_ANALYSIS_CACHE_MAX = 5
_cad_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_CAD_CACHE_MAX = 6


def _cache_analysis(entry: dict[str, Any]) -> str:
    analysis_id = uuid.uuid4().hex
    _analysis_cache[analysis_id] = entry
    while len(_analysis_cache) > _ANALYSIS_CACHE_MAX:
        _analysis_cache.popitem(last=False)
    return analysis_id


def _cache_cad(entry: dict[str, Any]) -> str:
    cad_id = uuid.uuid4().hex
    _cad_cache[cad_id] = entry
    while len(_cad_cache) > _CAD_CACHE_MAX:
        _cad_cache.popitem(last=False)
    return cad_id
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

# Qwen 판독은 같은 라벨 그림에 대해 결정적이므로 내용 해시로 재사용한다.
# 메모리 LRU와 디스크 저장을 함께 써 서버 재시작 뒤에도 긴 재판독을 피한다.
_MISSING = object()
_label_cache: "OrderedDict[str, float | None]" = OrderedDict()
_LABEL_CACHE_MAX = 4000
_LABEL_STORE = Path(
    os.environ.get("ADC_LABEL_CACHE")
    or (Path(__file__).resolve().parent / ".label_cache.json")
)
_label_store_dirty = False


def _load_label_store() -> None:
    try:
        if _LABEL_STORE.exists():
            found = json.loads(_LABEL_STORE.read_text(encoding="utf-8"))
            for key, value in found.items():
                _label_cache[key] = value
    except Exception:
        pass


def _save_label_store() -> None:
    global _label_store_dirty
    if not _label_store_dirty:
        return
    try:
        _LABEL_STORE.write_text(
            json.dumps(dict(_label_cache), ensure_ascii=False), encoding="utf-8"
        )
        _label_store_dirty = False
    except Exception:
        pass


_load_label_store()


def _crop_key(crop: Any) -> str:
    import hashlib

    return hashlib.blake2b(
        np.asarray(crop, dtype=np.uint8).tobytes(), digest_size=16
    ).hexdigest()


def reset_label_cache() -> None:
    global _label_store_dirty
    _label_cache.clear()
    _label_store_dirty = False


def _remember_labels(keys: list[str], values: list[Any]) -> None:
    global _label_store_dirty
    for key, value in zip(keys, values):
        _label_cache[key] = value
        _label_cache.move_to_end(key)
    while len(_label_cache) > _LABEL_CACHE_MAX:
        _label_cache.popitem(last=False)
    _label_store_dirty = True
    _save_label_store()

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
            keys = [_crop_key(crop) for crop in crops]
            cached = [_label_cache.get(key, _MISSING) for key in keys]
            todo = [index for index, value in enumerate(cached) if value is _MISSING]
            if todo:
                try:
                    reader = _get_qwen_reader()
                    fresh, qwen_failure = _read_qwen_values(
                        reader, [crops[index] for index in todo]
                    )
                    _remember_labels([keys[index] for index in todo], fresh)
                    for slot, value in zip(todo, fresh):
                        cached[slot] = value
                except Exception as exc:
                    qwen_failure = str(exc)
                    for slot in todo:
                        cached[slot] = None
            qwen_values = [None if value is _MISSING else value for value in cached]

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

    analysis_id: str | None = None
    if zero_output is not None:
        hybrid_lines = [
            line.get("points", [])
            for line in (getattr(hybrid_zero, "lines", []) if hybrid_zero is not None else [])
            if isinstance(line, dict) and len(line.get("points", [])) >= 2
        ]
        hybrid_case = int(getattr(hybrid_zero, "case", 2)) if hybrid_zero is not None else 2
        analysis_id = _cache_analysis(
            {
                "values": zero_output.values,
                "part_mask": zero_output.part_mask,
                "tolerance": float(zero_output.result.tolerance),
                "anchors": zero_anchors,
                "overlay_base": cv2.cvtColor(
                    clean_image if clean_image is not None else image,
                    cv2.COLOR_BGR2RGB,
                ),
                # 현재 확정된 하이브리드 결과를 CAD 브랜치 입력 형식으로만
                # 변환한다. 제로라인 자체를 다시 계산하지 않는다.
                "lab_zero_lines": hybrid_lines if hybrid_case == 2 else [],
                "lab_zero_areas": hybrid_lines if hybrid_case == 1 else [],
                "simple_zero_lines": [],
                "deviation_points": points,
                "part_no": part_number,
                "zero_reference": None,
            }
        )

    return {
        "analysisId": analysis_id,
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
        heading=str(title_values.get("heading", "보정 적용 내용")),
        management_label=str(title_values.get("managementLabel", "관리 NO")),
        management_no=str(title_values.get("managementNo", "")),
        part_name_label=str(title_values.get("partNameLabel", "PART NAME")),
        part_name=str(title_values.get("partName", "")),
        process_label=str(title_values.get("processLabel", "공정")),
        process=str(title_values.get("process", "")),
        part_no_label=str(title_values.get("partNoLabel", "PART NO")),
        part_no=str(title_values.get("partNo", "")),
        material_label=str(title_values.get("materialLabel", "원소재")),
        material=str(title_values.get("material", "")),
        applied_date_label=str(title_values.get("appliedDateLabel", "적용일자")),
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
        manual_target = raw_item.get("targetDir")
        parsed_items.append((source, str(manual_target) if manual_target is not None else None))

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
                "folderOrder는 차종·상세품번과 자료유형을 반드시 포함하고, "
                + ", ".join(AXES)
                + " 값을 겹치지 않게 사용해야 합니다."
            )
        previous_order = await run_in_threadpool(_active_folder_order)
        migration = await run_in_threadpool(
            migrate_folder_structure, FOLDER_ROOT, FILE_ORGANIZER_RULES,
            previous_order, order,
        )
        await run_in_threadpool(save_folder_order, FILE_ORGANIZER_DIR, order)
        return JSONResponse(
            {
                "folderOrder": order,
                "axes": _AXIS_OPTIONS,
                "migration": {
                    "moved": migration.moved,
                    "skipped": migration.skipped,
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




def _shift_centers(features: list[dict], offset) -> list[dict]:
    """center 를 offset 만큼 평행이동한다. axis/normal 은 방향이라 두 손 뗀다."""
    if not offset:
        return features
    ox, oy, oz = (float(v) for v in offset)
    moved = []
    for feature in features:
        item = dict(feature)
        centre = item.get("center")
        if centre and len(centre) == 3:
            item["center"] = [round(centre[0] - ox, 4),
                              round(centre[1] - oy, 4),
                              round(centre[2] - oz, 4)]
        moved.append(item)
    return moved


def load_cad_payload(payload: bytes, filename: str) -> dict[str, Any]:
    """업로드된 3D 파일을 읽어 뷰어용 메시 + RPS 후보를 만든다.

    STEP 이면 홀·기준평면까지 뽑고, STL/PLY/OBJ 면 메시만 준다 —
    삼각망으로 내보낸 시점에 원통면 정보가 이미 사라지기 때문이다.
    """
    from cad_import import mesh_io, step_reader

    suffix = Path(filename).suffix.lower()
    # OCCT/trimesh 는 경로 기반이라 임시 파일로 떨군다
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (Path(filename).name or "upload")
        path.write_bytes(payload)

        if step_reader.is_step_file(path):
            parsed = step_reader.read_step_full(path)
            web = mesh_io.to_web_mesh(
                parsed["mesh"], name=path.stem, source_format="step")
            # to_web_mesh 는 정점을 원점으로 옮긴다(자동차 부품은 차량
            # 좌표계라 원점에서 수천 mm 떨어져 있다). 홀·평면 좌표도 같은
            # 만큼 옮겨야 형상 위에 얹힌다 — 안 옮기면 실측 001 REINF SIDE
            # OTR 기준 중심이 (1584, -653, 491) 이라 홀이 1.8m 밖에 뜬다.
            # 축과 법선은 방향이라 그대로 둔다.
            offset = web["summary"]["bounds"]["center"] if web.get("recentered") else None
            web["holes"] = _shift_centers(parsed["holes"], offset)
            web["planes"] = _shift_centers(parsed["planes"][:50], offset)
            web["counts"] = parsed["counts"]

            # 오버레이(제로라인·보정량)를 그리려면 원본 삼각망이 필요하다.
            # 화면용 메시는 간략화돼 있어 광선 교차에 쓰면 어긋난다.
            mesh_full = parsed["mesh"]
            web["cadId"] = _cache_cad({
                "mesh": mesh_full,
                "offset": np.asarray(offset, dtype=float) if offset else np.zeros(3),
                "name": path.stem,
                # 히트맵은 화면에 그리는 정점에 입혀야 한다. 원본 삼각망은
                # 간략화 전이라 개수가 달라서 그대로 쓰면 어긋난다.
                "display_vertices": np.asarray(
                    web["positions"], dtype=float).reshape(-1, 3),
                "display_faces": np.asarray(web["indices"]).reshape(-1, 3),
            })
            return web

        if mesh_io.is_mesh_file(path):
            mesh = mesh_io.load_mesh(path)
            web = mesh_io.to_web_mesh(
                mesh, name=path.stem, source_format=suffix.lstrip("."))
            # 삼각망엔 B-Rep 특징이 없다. 빈 배열로 명시해 프론트가
            # "아직 안 읽음"과 "원래 없음"을 구분할 수 있게 한다.
            web["holes"] = []
            web["planes"] = []
            web["counts"] = {"cylinders": 0, "holes": 0, "planes": 0}
            web["note"] = (
                "삼각망 파일이라 홀·기준평면 정보가 없습니다. "
                "RPS 정렬이 필요하면 STEP(AP214)으로 받아야 합니다."
            )
            return web

    if suffix == ".catpart":
        raise ValueError(
            "CATIA 네이티브(.CATPart)는 독자 포맷이라 읽을 수 없습니다. "
            "CATIA에서 STEP(AP214) 또는 STL로 내보내 주세요."
        )
    raise ValueError(
        f"지원하지 않는 형식입니다: {suffix or '확장자 없음'} "
        f"(지원: STEP/STP, STL, PLY, OBJ, GLB/GLTF, 3MF)"
    )


# CAD 파일명과 스캔 품번이 다르다. 현업 제품데이터 폴더가 짝을 보여준다 —
#   64XX1-DR000_HDCT1860.CATPart  <->  64XX2-DR000 제품데이터.png (LH/RH)
#   67XX6-DR050_HDCT1750.CATPart  <->  67XX6-DR050 제품데이터.png
#   71XX1-DR000_HDCT0458.CATPart  <->  71XX2-DR000 제품데이터.png
# 끝자리 1/2 는 좌우 대칭품이라 CAD 는 한쪽만 온다.
CAD_TO_SCAN_PART = {
    "64XX1": "64XX2",
    "71XX1": "71XX2",
    "67XX6": "67XX6",
}


def scan_part_for_cad(cad_name: str) -> str | None:
    """CAD 파일명에서 짝이 되는 스캔 품번을 찾는다."""
    folded = str(cad_name or "").upper().replace("-", "").replace("_", "")
    for cad_key, scan_key in CAD_TO_SCAN_PART.items():
        if cad_key in folded:
            return scan_key
    return None


def apply_zero_edits(raw_lines: list, zero_edits: list | None) -> list:
    """보정시트에서 손본 제로라인을 그대로 따른다.

    3D 의 제로라인은 시트가 정하는 것이라, 시트에서 옮기거나 숨긴 것이
    3D 에 안 비치면 둘이 어긋난다. 옮김은 그림 좌표(px)로 받아 **광선을
    쏘기 전에** 더한다 — 3D 에서 따로 밀면 표면에서 떠 버린다.
    """
    if not zero_edits:
        return raw_lines
    moved: list = []
    for index, line in enumerate(raw_lines):
        edit = next((e for e in zero_edits
                     if int(e.get("index", -1)) == index), None)
        if edit is None:
            moved.append(line)
            continue
        if edit.get("hidden"):
            continue
        dx = float(edit.get("dx") or 0.0)
        dy = float(edit.get("dy") or 0.0)
        moved.append({**line, "points": [[p[0] + dx, p[1] + dy]
                                         for p in line["points"]]})
    return moved


# 적응형 제로라인(zero_line (2) 묶음)을 쓸 부품.
# 이 묶음의 Case 1(윤곽 교점 다각형)로 가는 부품만 넣는다.
ADAPTIVE_PARTS = {"JD_67XX6-DR000"}

# 자세 후보를 몇 개까지 광선으로 검증할지. 하나에 1초 안쪽이다.
OVERLAY_TRIES = 6

# 스캔을 CAD 에 얹은 결과. 같은 짝을 다시 보는 일이 잦다(탭을 옮기거나
# 제로라인을 손봤다 되돌리거나). 실측 20초짜리를 다시 돌릴 이유가 없다.
_overlay_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_OVERLAY_CACHE_MAX = 12


def reset_overlay_cache() -> None:
    """시험에서 캐시가 새어 나가지 않게 비운다."""
    _overlay_cache.clear()


def _overlay_key(cad_id: str, analysis_id: str, zero_edits,
                 fit_adjust=None) -> str:
    return "|".join([cad_id, analysis_id,
                     json.dumps(zero_edits or [], sort_keys=True),
                     json.dumps(fit_adjust or {}, sort_keys=True)])


def cad_overlay_for(cad_id: str, analysis_id: str,
                    zero_edits: list | None = None,
                    fit_adjust: dict | None = None) -> dict[str, Any]:
    """스캔에서 뽑은 제로라인·포인트를 CAD 표면 위의 3D 좌표로 옮긴다.

    보정량은 여기서 계산하지 않는다. 최종 보정시트는 작업자가 값을
    고치고 포인트를 빼기도 하는데, 그 상태는 화면이 들고 있다. 백엔드가
    따로 계산하면 3D 와 시트가 어긋난다 — 위치만 주고 값은 화면이 정한다.

    실루엣을 맞춰 어느 방향에서 본 그림인지 찾고, 그 방향으로 광선을
    쏴서 표면에 얹는다. 데이텀 정합이 아니라 겉모양 정합이라
    fit.iou 를 같이 내보낸다 — 낮으면 화면에서 경고해야 한다.
    """
    from cad_import import overlay as ov

    cad_entry = _cad_cache.get(cad_id)
    if cad_entry is None:
        raise ValueError("CAD 가 만료됐습니다. 3D 파일을 다시 여세요.")
    analysis = _analysis_cache.get(analysis_id)
    if analysis is None:
        raise ValueError("분석 결과가 만료됐습니다. 이미지를 다시 분석하세요.")

    cache_key = _overlay_key(cad_id, analysis_id, zero_edits, fit_adjust)
    if cache_key in _overlay_cache:
        _overlay_cache.move_to_end(cache_key)
        return _overlay_cache[cache_key]

    mesh = cad_entry["mesh"]
    offset = cad_entry["offset"]
    vertices = np.asarray(mesh.vertices, dtype=float) - offset
    faces = np.asarray(mesh.faces)

    import trimesh

    # LH·RH 가 한 파일에 들어 있으면 갈라서 **각각** 맞춰 보고 잘 맞는
    # 쪽을 쓴다. 스캔은 한 짝뿐이라 두 짝을 합친 실루엣과는 맞지 않는다
    # (실측 71XX1: 통째로 30% -> 한 짝만 쓰면 아래 hit_rate 참고).
    best = None
    for half in ov.split_sides(vertices, faces):
        half_vertices, half_faces, cut_axis, cut_mid, cut_side = half
        piece = trimesh.Trimesh(vertices=half_vertices, faces=half_faces,
                                process=False)
        # 겹침 넓이가 아니라 **점이 형상에 얹히는 비율**로 판정한다.
        # 껍질 겹침은 후하고(64XX2 96.9%) 실루엣은 박하다(42.2%) — 둘 다
        # 오버레이가 쓸 만한지를 못 가린다. overlay.MIN_HIT_RATE 참고.
        #
        # 그래서 자세 후보를 몇 개 받아 **광선을 쏴서** 고른다. 겹침으로
        # 하나만 받으면 얹힘이 더 좋은 자세를 놓칠 수 있다.
        seen: set = set()
        for guess in ov.fit_view(half_vertices, half_faces,
                                 analysis["part_mask"], top_k=OVERLAY_TRIES):
            key = (guess.axis, guess.sign, guess.flip_u, guess.flip_v,
                   round(guess.angle, 3), round(guess.mm_per_px, 4),
                   round(guess.origin_u, 1), round(guess.origin_v, 1))
            if key in seen:
                continue          # 같은 자세가 여러 번 올라온다
            seen.add(key)
            guess.hit_rate = round(ov.measure_hit_rate(
                guess, half_vertices, half_faces,
                analysis["part_mask"], piece), 4)
            if best is None or guess.hit_rate > best[0].hit_rate:
                best = (guess, half_vertices, half_faces, piece,
                        cut_axis, cut_mid, cut_side)

    fit, vertices, faces, shifted, cut_axis, cut_mid, cut_side = best

    # 마지막으로 **얹힘 비율을 직접 올린다.** 자세 찾기는 실루엣 겹침을
    # 보는데, 그건 대리 지표라 부품에 따라 크게 어긋난다.
    #   64XX2 99.7 -> 99.7 · 67XX6 88.0 -> 99.0 · 71XX2 67.7 -> 90.0
    fit = ov.polish_by_hit_rate(
        fit, vertices, faces, analysis["part_mask"], shifted)

    # 작업자가 손으로 맞춘 값이 있으면 그대로 따른다. 자동 정합은
    # 실루엣만 보므로 몇 퍼센트가 모자랄 수 있는데, 그때 사람이 조금
    # 돌리고 옮겨서 맞출 수 있어야 쓸 수 있는 도구가 된다.
    if fit_adjust:
        fit = ov.nudge_fit(
            fit, analysis["part_mask"].shape,
            angle_deg=float(fit_adjust.get("angle") or 0.0),
            dx=float(fit_adjust.get("dx") or 0.0),
            dy=float(fit_adjust.get("dy") or 0.0),
            scale=float(fit_adjust.get("scale") or 1.0))
        # 손으로 옮겼으면 얹힘도 다시 잰다 — 나아졌는지 보여야 한다
        fit.hit_rate = round(ov.measure_hit_rate(
            fit, vertices, faces, analysis["part_mask"], shifted), 4)

    fit.reliable = fit.hit_rate >= ov.MIN_HIT_RATE

    # 표면에 얹지 못한 점(광선이 빗나간 자리)은 뺀다. 예전에는 아무
    # 정점으로나 채워서 제로라인이 부품 밖으로 길게 뻗었다.
    def _densify(points: list, step_px: float = 4.0) -> list:
        """선 위를 촘촘히 채운다.

        꼭짓점만 표면에 얹으면 그 사이는 공중을 가로지른다. 촘촘히
        쏴야 곡면을 그대로 따라간다 — 3D 에서 "선을 얹은 느낌" 을
        없애는 진짜 방법이다(표면을 칠하면 리브와 구멍에서 조각난다).
        """
        dense: list = []
        for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
            span = float(np.hypot(bx - ax, by - ay))
            count = max(int(span / step_px), 1)
            for k in range(count):
                t = k / count
                dense.append([ax + (bx - ax) * t, ay + (by - ay) * t])
        dense.append(list(points[-1]))
        return dense

    # 우선순위: 현업 파이프라인 > 시트 등록 정답 > 우리 검출
    raw_lines = [{"line_id": i + 1, "points": pts}
                 for i, pts in enumerate(analysis.get("lab_zero_lines") or [])]
    if not raw_lines:
        raw_lines = [{"line_id": l.get("line_id"), "points": l["points"]}
                     for l in analysis.get("simple_zero_lines", [])]

    raw_lines = apply_zero_edits(raw_lines, zero_edits)

    lines = []
    dropped_line_points = 0
    for line in raw_lines:
        pts = line["points"]
        if len(pts) < 2:
            continue
        placed = ov.unproject(_densify(pts), vertices, faces, fit, shifted)
        kept = [spot for spot in placed if spot is not None]
        dropped_line_points += len(placed) - len(kept)
        if len(kept) < 2:
            continue

        # 표면에 얹힌 구간과 **빈 공간을 지나는 구간**을 나눠 준다.
        #
        # 광선이 빗나갔다는 것은 그 자리에 부품이 없다는 뜻이다(구멍·
        # 개구부). 이어서 실선으로 그리면 없는 자리에 선이 있는 것처럼
        # 보인다 — 받은 파이프라인은 "inner openings and holes are
        # traversable" 라 링 부품에서 실제로 빈 데를 지난다. 그 구간은
        # 화면에서 점선으로 그려 사실대로 보이게 한다.
        runs: list = []
        current: list = []
        for spot in placed:
            if spot is None:
                if len(current) >= 2:
                    runs.append(current)
                current = []
            else:
                current.append(spot)
        if len(current) >= 2:
            runs.append(current)

        gaps = [[runs[i][-1], runs[i + 1][0]] for i in range(len(runs) - 1)]
        lines.append({
            "line_id": line.get("line_id"),
            "points": kept,          # 예전 형식 — 통째로 이은 것
            "runs": runs,            # 표면에 얹힌 구간 (실선)
            "gaps": gaps,            # 빈 공간을 지나는 구간 (점선)
        })

    # 컬러바 범위 밖의 값은 판독 오류다. 실측(JD_67XX6, 컬러바 +3.0~-3.0)
    # 에서 +9.00 이 5건 나왔다. 3D 에 얹으면 화살표 길이 기준을 잡아먹어
    # 진짜 보정량(0.1~3mm)이 전부 점만 해진다.
    from zero_line_detection.simple_zero_line import PRODUCT_COLORBAR_MM
    folded = str(analysis.get("part_no") or "").upper().replace("-", "")
    span = next((v for k, v in PRODUCT_COLORBAR_MM.items() if k in folded), None)
    limit = max(abs(span[0]), abs(span[1])) * 1.05 if span else None

    wanted, rejected = [], []
    for point in analysis.get("deviation_points", []):
        value = float(point.get("value", 0.0))
        if limit is not None and abs(value) > limit:
            rejected.append({"id": point.get("id"), "value": round(value, 3)})
            continue
        wanted.append(point)

    # 광선은 한 번에 쏘는 게 훨씬 빠르다
    placed = ov.unproject([[p["xPx"], p["yPx"]] for p in wanted],
                          vertices, faces, fit, shifted)
    points = [
        {"id": point.get("id"), "position": spot,
         "value": round(float(point.get("value", 0.0)), 3)}
        for point, spot in zip(wanted, placed) if spot is not None
    ]
    dropped_points = sum(1 for spot in placed if spot is None)

    # ── 제로라인을 표면에 칠할 도장 ──────────────────────────
    # 3D 공간에 관(tube)으로 띄우면 곡면 위에서 형상과 떠서 "선을 얹은
    # 느낌" 이 난다. 표면 자체를 칠하면 굴곡을 그대로 따라간다.
    #   1 = 선(띠)   2 = 영역
    display = cad_entry.get("display_vertices")
    stencil = np.zeros(analysis["part_mask"].shape, np.uint8)
    reference = analysis.get("zero_reference") or {}
    # 영역은 **최소 외접 사각형**으로 그린다.
    #
    # 윤곽 그대로 칠했더니 가장자리를 따라 실오라기처럼 갈라져 "물감
    # 칠한 느낌" 이 났다. 시트도 제로 영역을 빗금 친 **네모**로 표기한다 —
    # 경계가 반듯해야 어디까지가 그 영역인지 읽힌다.
    area_contours = list(analysis.get("lab_zero_areas") or [])
    if not area_contours:
        # 시트에 등록해 둔 정답 영역도 같은 규칙으로 도형으로 만든다
        area_contours = zero_shapes.clean(reference.get("contours") or [])

    # 제로 영역의 **테두리**를 따로 만들어 준다.
    #
    # 지금까지는 네모를 표면에 칠하기만 했다. 칠하기는 정점 단위라
    # 경계가 삼각망을 따라 들쭉날쭉해진다 — 네모로 만들어 놓고도
    # 화면에서는 네모로 안 보였다. 네모의 네 변을 제로라인과 똑같이
    # 촘촘히 쏴서 표면에 얹으면 곡면을 타면서도 경계가 반듯하다.
    area_outlines: list = []
    for contour in area_contours:
        pts = np.rint(np.asarray(contour, dtype=float)).astype(np.int32)
        if len(pts) < 3:
            continue

        # 분석 때 이미 네모로 바꿔 두었다(zero_shapes). 여기서 또
        # 손대면 시트에 그린 것과 3D 에 그린 것이 달라진다.
        shape = pts
        cv2.fillPoly(stencil, [shape], 2)
        ring = [[float(x), float(y)] for x, y in shape]
        ring.append(ring[0])          # 닫는다
        area_outlines.append(ring)
    # 우선순위: 현업 파이프라인 > 시트에 등록된 정답 > 우리 검출.
    # 앞의 것일수록 근거가 분명하다.
    # 도장은 **영역**에만 쓴다(67XX6 처럼 시트가 면으로 표기한 경우).
    #
    # 선까지 칠했더니 리브와 구멍이 많은 면에서 조각조각 갈라져
    # "물감 칠한 느낌" 이 났다. 선은 선으로 그리는 게 맞고, 곡면을
    # 따라가게 하려면 촘촘히 쏴서 표면에 얹으면 된다(_densify).

    # 화면용 메시에서 **고른 쪽**만 남기는 가리개.
    #
    # LH·RH 가 한 파일이면 한 짝에 맞춘 자세로 색을 칠하게 되는데,
    # 그대로 두면 반대쪽 살에도 색이 묻는다. 자세를 잡을 때 쓴 것과
    # 같은 평면으로 가른다.
    def _own_side(points: np.ndarray) -> np.ndarray:
        if cut_axis < 0:
            return np.ones(len(points), dtype=bool)
        left = points[:, cut_axis] < cut_mid
        return left if cut_side < 0 else ~left

    # 제로 영역을 표면에 **칠하지 않는다.**
    #
    # 정점 단위로 칠하면 세 꼭짓점이 모두 영역에 든 삼각형만 남아,
    # 리브와 구멍이 많은 면에서 조각조각 갈라진다 — 네모로 만들어
    # 놓고도 화면에서는 물감 칠한 것처럼 보였다. 이제 네모의 테두리를
    # 표면에 얹어 그린다(zero_areas). 도장 계산은 그만큼 뺀다 —
    # 화면용 정점이 40만 개라 공짜가 아니다.
    zero_surface: list = []

    # 영역 테두리를 표면 위로 옮긴다 — 제로라인과 같은 방식이다.
    zero_areas: list = []
    for ring in area_outlines:
        placed = ov.unproject(_densify(ring), vertices, faces, fit, shifted)
        runs, current = [], []
        for spot in placed:
            if spot is None:
                if len(current) >= 2:
                    runs.append(current)
                current = []
            else:
                current.append(spot)
        if len(current) >= 2:
            runs.append(current)
        if runs:
            zero_areas.append({
                "runs": runs,
                "gaps": [[runs[i][-1], runs[i + 1][0]]
                         for i in range(len(runs) - 1)],
            })

    # 표면에 입힐 편차 — 화면용 정점 하나하나에 스캔 값을 찍는다.
    surface = []
    if display is not None:
        spots = np.asarray(display, dtype=float)
        surface = ov.sample_deviation(
            spots, fit, analysis["values"], analysis["part_mask"])
        mine = _own_side(spots)
        if not mine.all():
            surface = [v if mine[i] else None
                       for i, v in enumerate(surface)]

    # 보정 포인트를 **CAD 원래 좌표**로도 준다.
    #
    # CAD(STEP)를 그대로 고쳐 내보내는 것은 못 한다 — 자유곡면 제어점을
    # 연속성 지키며 옮기는 건 전용 소프트웨어 영역이고, 면으로 쪼갠
    # STEP 은 실측 삼각형당 1.6ms · 2.4KB 라 이 부품(363,431 삼각형)이면
    # 10분 · 852MB 다. 대신 CATIA 작업자가 그 자리에 그 값을 넣을 수
    # 있도록 **부품 좌표와 보정량**을 준다. 화면 좌표는 원점을 옮겨
    # 놓았으므로 offset 을 되돌린다.
    back = np.asarray(offset, dtype=float)
    for point in points:
        spot = np.asarray(point["position"], dtype=float) + back
        point["cad"] = [round(float(v), 3) for v in spot]

    answer = {
            "fit": fit.to_dict(), "zeroLines": lines, "points": points,
            "rejected": rejected,
            "zeroSurface": zero_surface,
            "zeroAreas": zero_areas,
            "zeroKind": "areas" if area_contours else (reference.get("kind") or "line"),
            "droppedPoints": dropped_points,
            "droppedLinePoints": dropped_line_points,
            "colorbarLimit": round(limit, 2) if limit else None,
            "scanPart": scan_part_for_cad(cad_entry.get("name", "")),
            "surfaceDeviation": surface,
            "deviationRange": (
                [round(float(np.nanmin(analysis["values"][analysis["part_mask"] > 0])), 2),
                 round(float(np.nanmax(analysis["values"][analysis["part_mask"] > 0])), 2)]
                if analysis.get("part_mask") is not None else None)}

    _overlay_cache[cache_key] = answer
    while len(_overlay_cache) > _OVERLAY_CACHE_MAX:
        _overlay_cache.popitem(last=False)
    return answer

def cad_morph_for(cad_id: str, corrections: dict, positions: dict,
                  reach_ratio: float) -> dict[str, Any]:
    """보정량만큼 민 "보정 후" 형상을 만든다.

    B-Rep 은 건드리지 않는다 — 자유곡면을 연속성 지키며 변형하는 건
    전용 소프트웨어 영역이고, 어설프게 하면 가공 못 할 형상이 나온다.
    삼각망만 밀어서 **비교용**으로 쓴다(cad_import/morph.py 참고).
    """
    import trimesh
    from cad_import import morph as mp

    entry = _cad_cache.get(cad_id)
    if entry is None:
        raise ValueError("CAD 가 만료됐습니다. 3D 파일을 다시 여세요.")

    display = entry.get("display_vertices")
    if display is None:
        raise ValueError("표시용 형상이 없습니다.")
    vertices = np.asarray(display, dtype=float)
    faces = np.asarray(entry["display_faces"])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=float)

    spots, values = [], []
    for point_id, value in corrections.items():
        where = positions.get(point_id)
        if where and len(where) == 3:
            spots.append([float(v) for v in where])
            values.append(float(value))
    if not spots:
        raise ValueError("3D 에 올라간 보정 포인트가 없습니다.")

    moved, shift, stats = mp.morph(
        vertices, faces, normals, spots, values, reach_ratio)

    # 보정량의 **부호가 곧 공정**이다 — + 용접(덧살), - CNC 가공.
    # 두 일은 작업자도 견적도 다르므로 물량을 따로 낸다.
    work = mp.work_volumes(vertices, faces, shift)

    return {
        "positions": [round(float(v), 4) for v in moved.ravel()],
        "shift": [round(float(v), 4) for v in shift],
        "stats": stats.to_dict(),
        "points": len(spots),
        "work": [w.to_dict() for w in work],
    }


async def cad_morph(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        corrections = body.get("corrections") or {}
        positions = body.get("positions") or {}
        result = await run_in_threadpool(
            cad_morph_for, str(body.get("cadId") or ""),
            {str(k): float(v) for k, v in corrections.items()},
            {str(k): v for k, v in positions.items()},
            float(body.get("reachRatio") or 0.04),
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


def open_morph_as_cad(cad_id: str, corrections: dict, positions: dict,
                      reach_ratio: float) -> dict[str, Any]:
    """보정 후 형상을 **새 CAD 로 등록**해 원본처럼 다루게 한다.

    [왜 이렇게 하나]
    화면에서만 밀어 보여 주면, 내보낸 파일이 정말 그 형상인지 알 수 없다.
    고친 형상을 실제로 만들어 다시 올리면 —
      · 원본과 같은 도구(단면·측정·주석·공정 구역)를 그대로 쓴다
      · 탭을 오가며 원본과 견준다
      · 화면에 보이는 것이 곧 내보내는 STL 이다(둘이 갈라지지 않는다)

    B-Rep 은 아니다. 원본 STEP 의 자유곡면을 고쳐 내보내는 것은 전용
    소프트웨어 영역이고, 면으로 쪼갠 STEP 은 실측 삼각형당 1.6ms ·
    2.4KB 라 이 부품(363,431 삼각형)이면 10분 · 852MB 다. 여기서 만드는
    것은 **삼각망**이고, 그것이 STL 로 나가는 것과 같은 형상이다.
    """
    import trimesh
    from cad_import import mesh_io

    entry = _cad_cache.get(cad_id)
    if entry is None:
        raise ValueError("CAD 가 만료됐습니다. 3D 파일을 다시 여세요.")

    result = cad_morph_for(cad_id, corrections, positions, reach_ratio)
    moved = np.asarray(result["positions"], dtype=float).reshape(-1, 3)
    faces = np.asarray(entry["display_faces"])
    # 화면용 정점은 원점을 옮겨 놓았다. 새 CAD 도 같은 자리에 서야
    # 원본과 겹쳐 볼 수 있으므로 되돌리지 않는다.
    mesh = trimesh.Trimesh(vertices=moved, faces=faces, process=False)

    name = f"{entry.get('name', 'part')}_보정후"
    # **다시 원점으로 옮기지 않는다.**
    #
    # 화면용 정점은 이미 원본을 옮겨 놓은 좌표계에 있다. to_web_mesh 가
    # 기본으로 자기 바운딩 상자 중심으로 또 옮기면 새 형상이 원본과
    # 어긋나 겹쳐 볼 수가 없다 — 실측에서 최대 이동이 2.886mm 로 나왔다
    # (보정 최대는 2.000mm 인데 상수 오프셋이 얹힌 것이다).
    web = mesh_io.to_web_mesh(mesh, name=name, source_format="morph",
                              recenter=False)
    web["holes"] = []
    web["planes"] = []
    web["counts"] = {"cylinders": 0, "holes": 0, "planes": 0}
    web["cadId"] = _cache_cad({
        "mesh": mesh,
        "offset": np.asarray(entry.get("offset"), dtype=float),
        "name": name,
        "display_vertices": np.asarray(
            web["positions"], dtype=float).reshape(-1, 3),
        "display_faces": np.asarray(web["indices"]).reshape(-1, 3),
    })
    web["work"] = result.get("work") or []
    # 이 탭이 무엇인지 적어 둔다. 안 적으면 "홀을 찾지 못했습니다" 가
    # 떠서 읽기 실패처럼 보인다 — 파생 형상이라 홀 정보가 없는 게 맞다.
    volumes = " · ".join(
        f"{'용접' if w['kind'] == 'weld' else '가공'} "
        f"{w['volume_mm3'] / 1000:.1f}cc"
        for w in (result.get("work") or []))
    web["note"] = (
        f"보정 후 형상입니다 — {entry.get('name', '원본')} 을(를) "
        f"보정 포인트 {result.get('points', 0)}개로 민 삼각망입니다. "
        f"조립 홀·기준면은 원본 탭에서 봅니다"
        + (f" · {volumes}" if volumes else "") + ".")
    return web


async def cad_morph_open(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await run_in_threadpool(
            open_morph_as_cad, str(body.get("cadId") or ""),
            {str(k): float(v) for k, v in (body.get("corrections") or {}).items()},
            {str(k): v for k, v in (body.get("positions") or {}).items()},
            float(body.get("reachRatio") or 0.04),
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def cad_morph_stl(request: Request) -> Response:
    """보정 후 형상을 STL 로 내보낸다.

    `part` 로 무엇을 담을지 고른다 —
      "after"(기본)  보정 후 형상 전체
      "weld"         살을 붙일 자리만 (용접 지시용)
      "cut"          살을 깎을 자리만 (CNC 가공 지시용)

    현장에서는 전체 형상보다 **자기 공정 자리만** 있는 편이 낫다.
    용접공에게 깎을 자리를 같이 주면 헷갈린다.
    """
    try:
        import trimesh
        from cad_import import morph as mp
        body = await request.json()
        want = str(body.get("part") or "after")
        result = await run_in_threadpool(
            cad_morph_for, str(body.get("cadId") or ""),
            {str(k): float(v) for k, v in (body.get("corrections") or {}).items()},
            {str(k): v for k, v in (body.get("positions") or {}).items()},
            float(body.get("reachRatio") or 0.04),
        )
        entry = _cad_cache.get(str(body.get("cadId") or "")) or {}
        moved = np.asarray(result["positions"], dtype=float).reshape(-1, 3)
        faces = np.asarray(entry["display_faces"])
        tail = "보정후"
        if want in ("weld", "cut"):
            split = mp.split_by_process(faces, result["shift"])
            faces = split[want]
            if not len(faces):
                return JSONResponse(
                    {"error": "그 공정에 해당하는 자리가 없습니다."},
                    status_code=404)
            tail = "덧살(용접)" if want == "weld" else "깎기(가공)"
        mesh = trimesh.Trimesh(vertices=moved, faces=faces, process=False)
        payload = mesh.export(file_type="stl")
        name = f"{entry.get('name', 'part')}_{tail}.stl"
        quoted = quote(name)
        return Response(payload, media_type="model/stl", headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quoted}"})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


def cad_sections_for(cad_id: str, notes: str, side: str) -> dict[str, Any]:
    """보정시트가 적어 둔 단면 위치(H:300 · T:1700)로 제로라인을 계산한다.

    다른 제로라인은 전부 추정이다 — 색을 읽거나 실루엣을 맞춘다. 이건
    아니다. 시트가 준 숫자로 CAD 를 자르기만 하므로 오차가 없다.
    (section_zero.py 에 축을 어떻게 확정했는지 적어 뒀다.)

    좌표는 화면용 메시와 같은 자리로 옮겨서 준다 — to_web_mesh 가 정점을
    원점으로 당겨 놨기 때문에 그만큼 빼지 않으면 형상에서 멀리 뜬다.
    """
    from zero_line_detection.section_zero import (
        parse_notes, zero_lines_from_notes,
    )

    entry = _cad_cache.get(cad_id)
    if entry is None:
        raise ValueError("CAD 가 만료됐습니다. 3D 파일을 다시 여세요.")

    parsed = parse_notes(notes)
    if not parsed:
        raise ValueError(
            "단면 표기를 찾지 못했습니다. 시트에 적힌 대로 "
            "'H : 300' 이나 'T : 1700' 처럼 넣어 주세요.")

    lines = zero_lines_from_notes(entry["mesh"], parsed, side=side)
    # offset 은 numpy 배열일 수 있다 — `or` 로 기본값을 주면
    # "truth value of an array is ambiguous" 로 터진다.
    raw = entry.get("offset")
    offset = [0.0, 0.0, 0.0] if raw is None else [float(v) for v in raw]
    shifted = []
    for line in lines:
        moved = [[[p[0] - offset[0], p[1] - offset[1], p[2] - offset[2]]
                  for p in poly] for poly in line.polylines]
        item = line.to_dict()
        item["polylines"] = moved
        shifted.append(item)

    return {"notes": [f"{k}:{v:g}" for k, v in parsed],
            "sections": shifted,
            "unmatched": [f"{k}:{v:g}" for k, v in parsed
                          if not any(s["label"] == f"{k}:{v:g}" for s in shifted)]}


async def cad_sections(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await run_in_threadpool(
            cad_sections_for,
            str(body.get("cadId") or ""),
            str(body.get("notes") or ""),
            str(body.get("side") or "both"),
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def cad_overlay(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await run_in_threadpool(
            cad_overlay_for,
            str(body.get("cadId") or ""),
            str(body.get("analysisId") or ""),
            body.get("zeroEdits") or None,
            body.get("fitAdjust") or None,
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def cad(request: Request) -> JSONResponse:
    try:
        form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_CAD_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "3D 파일이 필요합니다."}, status_code=400)
        payload = await upload.read()
        if len(payload) > MAX_CAD_UPLOAD_BYTES:
            return JSONResponse(
                {"error": f"파일이 너무 큽니다 ({len(payload) / 1024 / 1024:.0f}MB). "
                          f"최대 {MAX_CAD_UPLOAD_BYTES // 1024 // 1024}MB"},
                status_code=413,
            )
        result = await run_in_threadpool(
            load_cad_payload, payload, getattr(upload, "filename", "part.step")
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


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
        Route("/api/cad-overlay", cad_overlay, methods=["POST"]),
        Route("/api/cad-sections", cad_sections, methods=["POST"]),
        Route("/api/cad-morph", cad_morph, methods=["POST"]),
        Route("/api/cad-morph-open", cad_morph_open, methods=["POST"]),
        Route("/api/cad-morph-stl", cad_morph_stl, methods=["POST"]),
        Route("/api/file-organizer/status", file_organizer_status, methods=["GET"]),
        Route("/api/file-organizer/scan", file_organizer_scan, methods=["GET", "POST"]),
        Route("/api/file-organizer/upload", file_organizer_upload, methods=["POST"]),
        Route("/api/file-organizer/discard", file_organizer_discard, methods=["POST"]),
        Route("/api/file-organizer/execute", file_organizer_execute, methods=["POST"]),
        Route("/api/file-organizer/database", file_organizer_database, methods=["GET", "POST"]),
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
