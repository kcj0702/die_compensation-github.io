"""Local-only API that connects the React UI to the three vision engines."""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

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

# deviation_extraction currently uses local-style imports (import config), so
# its own folder must precede the project root on sys.path.
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(DEVIATION_DIR))

from label_detector import (  # noqa: E402
    build_blue_annotation_mask, build_scan_mask, detect_labels,
)
from colormap_reader import build_lut  # noqa: E402
from point_extractor import _sample_deviation_color  # noqa: E402
from vlm_reader import LabelValueReader  # noqa: E402
from label_removal.remove_labels import create_versions, detect_label_boxes  # noqa: E402
from zero_line_detection.visualize import make_overlay  # noqa: E402
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line  # noqa: E402
from zero_line_detection import zero_shapes  # noqa: E402
from zero_line_detection.zero_criteria import (  # noqa: E402
    candidates_to_mask, find_zero_candidates,
)
from zero_line_detection.polygonize import draw_polygons, polygonize  # noqa: E402
from zero_line_detection.zero_polyline import (  # noqa: E402
    draw_zero_polylines, extract_zero_polylines,
)
from zero_line_detection.zero_boundary import (  # noqa: E402
    draw_zero_boundary, filter_anchors_by_labels, find_boundary_anchors, grow_patches,
)
from zero_line_detection.calibration import (  # noqa: E402
    calibrate_vmin_vmax, calibrate_with_points,
)
from zero_line_detection.zero_valley import (  # noqa: E402
    find_valley_lines, rank_zero_line_candidates,
)
from zero_line_advance.advance import (  # noqa: E402
    AdvanceConfig, detect_advanced_zero_line,
)
from zero_line_detection.sheet_reference import load_library  # noqa: E402
from zero_line_detection.green_belt import find_green_belts  # noqa: E402
from zero_line_detection.simple_zero_line import (  # noqa: E402
    PRODUCT_COLORBAR_MM, colorbar_span_for, find_simple_zero_lines,
)
from zero_line_detection.lab_profile import (  # noqa: E402
    distance_report, lab_shapes_for,
)
from zero_line_detection.zero_points import (  # noqa: E402
    cluster_zero_points, connect_strongest_pair, expand_clusters_to_zones,
    filter_to_key_points, load_key_scores, load_loop_paths, load_zero_points,
    snap_into_mask,
)
from zero_line_detection.register_sheet import part_no_from_name  # noqa: E402
from zero_line_detection.key_points import select as select_key_points  # noqa: E402
from zero_line_detection.file_naming import parse as parse_filename  # noqa: E402
from zero_line_detection import lab_runner  # noqa: E402


DEFAULT_FOLDER_ROOT = Path(
    r"C:\Users\KDT013\Desktop\금형보정치\경북대KDT(14기) 자료\품번별 폴더 정리 자료_예시"
)
FOLDER_ROOT = Path(os.environ.get("AJIN_FOLDER_ROOT", DEFAULT_FOLDER_ROOT)).resolve()
# 금형 STEP 은 CATIA 가 삼각망을 통째로 끼워 넣어 파일이 크다 —
# 실측 64XX1 이 215MB, 67XX6 이 170MB, 71XX1 이 119MB 였다.
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
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
# 품번별로 확정된 제로라인 보관함. 보정시트에서 읽어 등록해두면
# (python -m zero_line_detection.register_sheet) 같은 품번의 스캔이
# 들어왔을 때 추론하지 않고 시트와 동일한 선을 그대로 쓴다.
ZERO_LINE_LIBRARY = PROJECT_DIR / "zero_line_detection" / "zero_line_library.json"
# my_lab 파이프라인(스캔포인트 윤곽선 -> 편차 그래프 -> 0포인트)의 결과.
# 라벨 실측값에서 나온 0포인트라 컬러바 색 잡음에 흔들리지 않는다.
ZERO_POINTS_DIR = PROJECT_DIR / "zero_line_detection" / "zero_points_data"
LOOP_PATHS_DIR = PROJECT_DIR / "my_lab" / "scan_point_contour" / "output"
# 현업 제공 key_zero_point_engine 결과(있으면 0포인트 후보를 컬러바 HSV
# 재검증으로 한 번 더 거른다). 없는 품번은 그냥 원래 후보를 그대로 쓴다.
KEY_ZERO_POINTS_DIR = PROJECT_DIR / "my_lab" / "zero_point_selection" / "output"

_reader: LabelValueReader | None = None
_reader_lock = threading.Lock()

# 분석 1회분(값장·부품마스크·앵커·허용오차)을 잠깐 들고 있는 캐시.
# 로컬 1인용 데모라 세션 관리 없이 메모리 dict 로 충분하다 — 사람이
# 앵커 2개를 클릭해서 "선 잇기" 를 요청할 때 이미지를 다시 안 올리고,
# VLM 라벨 판독도 다시 안 돌리려는 목적.
_analysis_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_ANALYSIS_CACHE_MAX = 5


# 라벨 판독 결과 캐시.
#
# [왜 필요한가 — 실측]
# JD_67XX6 한 장을 분석하는 데 812초가 걸리는데 그중 **739초(96%)가
# Qwen 숫자 판독**이다. 나머지 전부 합쳐도 8초다(라벨 검출 0.3 · 라벨
# 지우기 3.6 · 컬러바 4.2).
#
# 그런데 Qwen 이 하는 일은 **라벨에 적힌 글자를 읽는 것**뿐이다.
# `-1.4` 는 품번이 뭐든 `-1.4` 다. 품번은 컬러바 범위(색->mm)와 제로라인
# 파라미터만 바꾼다. 그래서 같은 그림을 다시 분석할 때 — 품번을 고쳐
# 다시 돌릴 때가 특히 그렇다 — 739초를 그대로 또 쓸 이유가 없다.
#
# 열쇠는 **잘라낸 라벨 그림 자체**의 해시다. 파일 이름이나 크기가 아니라
# 내용으로 잡아야 같은 라벨을 알아본다.
_MISSING = object()          # 캐시에 없음과 '읽었는데 None' 을 가른다
_label_cache: "OrderedDict[str, float | None]" = OrderedDict()
_LABEL_CACHE_MAX = 4000        # 부품 한 장이 라벨 130여 개다

# 판독 결과를 디스크에도 남긴다.
#
# 실측 64XX2 한 장이 Qwen 판독 57초다. 메모리에만 들고 있으면 엔진을
# 다시 띄울 때마다 그 57초를 다시 쓴다 — 파이썬을 고치면 반드시 다시
# 띄워야 하는 프로젝트라 그 일이 잦다. 열쇠가 **잘라낸 그림의 내용
# 해시**라 그림이 같으면 값도 같다.
# 경로는 환경변수로 옮길 수 있다(ADC_LABEL_CACHE). 시험은 이걸로
# 임시 폴더를 가리켜 실제로 데워 둔 캐시를 지키고, 운영에서는 여러
# 사람이 공유하는 자리로 옮길 수 있다.
_LABEL_STORE = Path(os.environ.get("ADC_LABEL_CACHE")
                    or (Path(__file__).resolve().parent / ".label_cache.json"))
_label_store_dirty = False


def _load_label_store() -> None:
    try:
        if _LABEL_STORE.exists():
            found = json.loads(_LABEL_STORE.read_text(encoding="utf-8"))
            for key, value in found.items():
                _label_cache[key] = value
    except Exception:
        pass          # 깨졌으면 그냥 다시 읽는다


def _save_label_store() -> None:
    global _label_store_dirty
    if not _label_store_dirty:
        return
    try:
        _LABEL_STORE.write_text(
            json.dumps(dict(_label_cache), ensure_ascii=False),
            encoding="utf-8")
        _label_store_dirty = False
    except Exception:
        pass


_load_label_store()


def _crop_key(crop) -> str:
    """잘라낸 라벨 그림의 내용 해시."""
    import hashlib

    return hashlib.blake2b(
        np.asarray(crop, dtype=np.uint8).tobytes(), digest_size=16
    ).hexdigest()


def reset_label_cache() -> None:
    """판독 캐시를 비운다. 테스트가 서로 영향을 주지 않게 하는 용도다."""
    global _label_store_dirty
    _label_cache.clear()
    _label_store_dirty = False


def _remember_labels(keys: list[str], values: list) -> None:
    global _label_store_dirty
    for key, value in zip(keys, values):
        _label_cache[key] = value
        _label_cache.move_to_end(key)
    while len(_label_cache) > _LABEL_CACHE_MAX:
        _label_cache.popitem(last=False)
    _label_store_dirty = True
    _save_label_store()


_cad_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
# 여러 개를 열어 놓고 골라 보므로 3개로는 모자란다(파일 4개째를 열면
# 첫 파일이 밀려나 "CAD 가 만료됐습니다" 가 뜬다). 실측 64XX1 STEP 한 개가
# 파싱 후 메모리에서 약 40MB(정점 302,340 · 삼각형 369,082)라 6개까지는
# 감당된다.
_CAD_CACHE_MAX = 6


def _cache_cad(entry: dict[str, Any]) -> str:
    """파싱한 CAD 를 들고 있는다. 215MB STEP 이 42~100초 걸려서
    오버레이를 그릴 때마다 다시 읽을 수는 없다."""
    cad_id = uuid.uuid4().hex
    _cad_cache[cad_id] = entry
    while len(_cad_cache) > _CAD_CACHE_MAX:
        _cad_cache.popitem(last=False)
    return cad_id


def _cache_analysis(entry: dict[str, Any]) -> str:
    analysis_id = uuid.uuid4().hex
    _analysis_cache[analysis_id] = entry
    while len(_analysis_cache) > _ANALYSIS_CACHE_MAX:
        _analysis_cache.popitem(last=False)
    return analysis_id


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


def analyze_image(image: np.ndarray, filename: str,
                  part_no: str | None = None) -> dict[str, Any]:
    """스캔 한 장을 분석한다.

    part_no 를 주면 파일명 대신 그것을 품번으로 쓴다. 품번은 컬러바 범위
    (PRODUCT_COLORBAR_MM)와 제로라인 파라미터를 고르는 열쇠라, 파일명에
    품번이 없으면 제로라인 단계가 통째로 비어 버린다 — 실측으로 확인했다.

        _boundary_anchors.png                  제로라인 0개
        JD_67XX6-DR000 3D 스캔.png (같은 그림)  제로라인 3개

    파일명을 바꾸라고 하는 대신 화면에서 품번을 고를 수 있게 했다.
    """
    height, width = image.shape[:2]
    errors: dict[str, str] = {}
    # 단계별 소요 시간. "왜 이렇게 오래 걸리나" 를 짐작이 아니라 숫자로
    # 답하려는 것이다. 따로 프로파일러를 띄우면 이 서버와 GPU 를 두고
    # 다퉈 값이 왜곡된다 — 실제로 그렇게 재다가 13분을 버렸다.
    spent: dict[str, float] = {}

    from contextlib import contextmanager

    @contextmanager
    def _timed(label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            spent[label] = round(spent.get(label, 0.0)
                                 + time.perf_counter() - start, 2)
    # 품번을 먼저 정한다 — 컬러바 검출이 실패했을 때 대신 쓸 범위를
    # 고르는 데 필요하다(zero_line.detect_zero_line 의 대체 경로).
    part_key = (part_no or "").strip().upper() or part_no_from_name(filename)
    part_span = colorbar_span_for(part_key)
    # 현업 파일명 규칙에서 보정시트 머리말 거리를 읽어 둔다
    # (차종·품명·공정·적용일자). 원소재만 파일명에 없다.
    naming = parse_filename(filename)

    clean_image: np.ndarray | None = None
    label_count = 0
    try:
        with _timed("라벨 박스 검출"):
            label_count = len(detect_label_boxes(image))
        with _timed("라벨 지우기"):
            clean_image = create_versions(image)["2_labels_inpainted"]
    except Exception as exc:  # engine errors must be shown per engine
        errors["label"] = str(exc)

    # clean_image(라벨 제거·복원본)를 값 추출 소스로 써보려 했으나,
    # label_removal 엔진이 컬러바 범례의 눈금 숫자까지 "라벨"로 오인해
    # 범례 전체를 흰색으로 지워버려서 컬러바 검출 자체가 깨진다(확인함).
    # 그래서 원본 이미지를 그대로 쓰고, 주석 제거는 zero_line_detection
    # 자체 마스킹(annotations.py, 컬러바 영역은 건드리지 않음)에 맡긴다.
    zero_output = None
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with _timed("컬러바+제로영역"):
            zero_output = detect_zero_line(
                rgb,
                # PRODUCT_COLORBAR_MM 은 (위, 아래) = (vmax, vmin) 순서다
                ZeroLineConfig(
                    vmax=part_span[0] if part_span else None,
                    vmin=part_span[1] if part_span else None,
                ),
                source_name=filename)
    except Exception as exc:
        errors["zero"] = str(exc)

    points: list[dict[str, Any]] = []
    qwen_reads = 0
    unread_labels = 0
    # 다시 읽지 않고 캐시에서 가져온 라벨 수. 판독까지 못 가고 예외가
    # 나도 아래 응답에서 쓰므로 여기서 잡아 둔다.
    reused_labels = 0
    detected_candidates = 0
    valid_candidates_count = 0
    deviation_warnings: list[str] = []
    try:
        candidates = detect_labels(image)
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
            # 이미 읽어 본 라벨은 다시 읽지 않는다
            keys = [_crop_key(crop) for crop in crops]
            cached = [_label_cache.get(key, _MISSING) for key in keys]
            todo = [i for i, value in enumerate(cached) if value is _MISSING]
            # timings 는 **초**만 담는다. 개수를 같이 넣었더니 합계가
            # 엉뚱하게 나왔다(라벨 79개가 79초로 더해졌다).
            reused_labels = len(crops) - len(todo)
            if todo:
                try:
                    with _timed("Qwen 모델 적재"):
                        reader = _get_qwen_reader()
                    with _timed("Qwen 숫자 판독"):
                        fresh, qwen_failure = _read_qwen_values(
                            reader, [crops[i] for i in todo])
                    _remember_labels([keys[i] for i in todo], fresh)
                    for slot, value in zip(todo, fresh):
                        cached[slot] = value
                except Exception as exc:
                    qwen_failure = str(exc)
                    for slot in todo:
                        cached[slot] = None
            qwen_values = [None if v is _MISSING else v for v in cached]

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

    zero_overlay: np.ndarray | None = None
    zero_candidates: list = []
    zero_datum_mask: np.ndarray | None = None
    zero_lines: list = []
    zero_anchors: list = []
    zero_patches: list = []
    zero_line_candidates: list = []
    calibration_stats: dict | None = None
    calibrated_values: np.ndarray | None = None
    if zero_output is not None:
        try:
            overlay_base = cv2.cvtColor(
                clean_image if clean_image is not None else image,
                cv2.COLOR_BGR2RGB,
            )

            # VLM이 라벨에서 직접 읽은 실측값(points)으로 컬러바 추정치를
            # 보정한다. 부품마다 --vmin/--vmax 를 손으로 넣던 걸 대신한다.
            calibrated_values, calibration_stats = calibrate_with_points(
                zero_output.values, points
            )

            # 색만 보고 잡은 0 밴드에서, 실제로 기준이 될 수 있는 곳만 추린다.
            # 편차가 0에 가깝고 + 주변이 평탄한 곳이 스프링백의 기준면/기준선이다.
            calibration_scale = abs(float((calibration_stats or {}).get("scale", 1.0)))
            calibrated_tolerance = float(zero_output.result.tolerance) * calibration_scale

            zero_candidates, flat, _ = find_zero_candidates(
                calibrated_values,
                zero_output.part_mask,
                calibrated_tolerance,
            )
            zero_datum_mask = candidates_to_mask(zero_candidates, flat, top_n=8)

            # 2026-08-25 아진산업 방문 확인 사항: 제로라인의 시작/끝점은
            # 부품 가장자리에서 편차 부호가 바뀌는 지점이다. RING SUNROOF
            # 실측 시트로 정량 검증했다 — 실제 패치 7개 중 5개 적중,
            # 평균 위치 오차 대각선의 7.2% (zero_line_detection/README.md 참고).
            # 아직 완벽하지 않으므로 후보로 제시하고 최종 판단은 사람이 한다.
            zero_anchors = find_boundary_anchors(
                calibrated_values, zero_output.part_mask
            )
            # 근처 실측 라벨값이 뚜렷하게 크면(0.5mm 초과) 리브·나사구멍
            # 잡음으로 생긴 가짜 앵커일 확률이 높다 — 탈락시킨다("정답
            # 확정"이 아니라 "명백히 아닌 것만 거르는" 필터다).
            zero_anchors = filter_anchors_by_labels(zero_anchors, points)
            zero_patches = grow_patches(
                calibrated_values, zero_output.part_mask, zero_anchors,
                tolerance=calibrated_tolerance,
            )
            zero_lines = extract_zero_polylines(
                calibrated_values, zero_output.part_mask
            )
            # 앵커 쌍을 사람이 고르지 않아도 되도록, 모든 쌍의 경로를
            # "부품을 실제로 둘로 가르는 정도"로 순위 매겨 상위 4개만
            # 내보낸다. 1등이 항상 정답은 아니라서(실측: 정답이 3·4위)
            # 하나로 확정하지 않고 후보 목록으로 준다.
            zero_line_candidates = rank_zero_line_candidates(
                calibrated_values, zero_output.part_mask, zero_anchors, top_n=4
            )
            # 기본 화면에는 검증된 것만 보여준다. zero_lines(전체 내부
            # 스켈레톤 추적)와 zero_datum_mask(평탄도 기반 후보)는 실측
            # 시트 대비 검증에서 지저분하고 신뢰도가 낮았던 예전 방식이라
            # 자동 오버레이에서는 뺐다 — 대신 사람이 앵커 2개를 골라
            # /api/zero-valley-line 으로 선을 그리는 방식(검증됨, 오차
            # 대각선의 3.68%)을 쓴다.
            # 자동 후보 패치의 붉은 외곽선은 보정시트의 기준선처럼 보이지만
            # 실제로는 후보일 뿐이라 화면을 지저분하게 만들었다. 기본 화면은
            # 원본 스캔만 보여주고, 사용자가 앵커 두 개를 고르면 그 사이의
            # 단일 골짜기 경로만 프런트엔드에서 붉은 선으로 표시한다.
            zero_overlay = overlay_base
        except Exception as exc:
            errors["zero"] = str(exc)

    # AI 1차 제안(zero_line_advance): 라벨 숫자를 컬러바가 아니라 직접
    # 읽어서(폰트 템플릿 매칭) "0.0" 표시점과 부호가 바뀌는 지점을 찾고,
    # 그 사이를 꼭짓점 몇 개짜리 깔끔한 직선으로 잇는다. 컬러바 클리핑에
    # 영향받지 않아 사람이 앵커를 고르지 않아도 자동으로 선을 만든다.
    # "0.0" 표시점이 2개 이상이면 신뢰도가 높고, 1개 이하이면 반대쪽
    # 끝점을 추정해야 해서 신뢰도가 낮다 — 후자는 warnings 로 표시하고
    # 사람이 위 앵커-클릭 방식으로 직접 고쳐야 한다(회의록 "AI 제안 →
    # 작업자 수정" 방향).
    advance_line: dict[str, Any] | None = None
    if zero_output is not None and calibration_stats is not None:
        try:
            advance_vmin_vmax = calibrate_vmin_vmax(zero_output.values, calibration_stats)
            if advance_vmin_vmax is not None:
                advance_vmin, advance_vmax = advance_vmin_vmax
                advance_result = detect_advanced_zero_line(
                    image,
                    clean_image if clean_image is not None else image,
                    vmin=advance_vmin,
                    vmax=advance_vmax,
                    config=AdvanceConfig(),
                )
                advance_line = {
                    "points": [
                        [round(float(x), 1), round(float(y), 1)]
                        for x, y in advance_result.smooth_path
                    ],
                    "warnings": advance_result.warnings,
                    "confidence": "low" if advance_result.warnings else "high",
                }
        except Exception as exc:
            errors["zeroAdvance"] = str(exc)

    zero_regions = len(zero_output.result.regions) if zero_output is not None else 0
    zero_ratio = zero_output.result.zero_ratio if zero_output is not None else 0.0
    zero_warnings = list(zero_output.warnings) if zero_output is not None else []
    warnings = zero_warnings + deviation_warnings

    if qwen_reads:
        value_mode = "Qwen2.5-VL-3B 로컬 판독"
    else:
        value_mode = "판독 결과 없음"

    # 라벨 실측값 기반 0포인트 -> 군집(점/존) -> 선으로 잇기
    zero_point_clusters: list = []
    simple_key_points: list = []
    label_zero_line = None
    try:
        points_file = ZERO_POINTS_DIR / f"{part_key}.json"
        if points_file.is_file():
            raw_points = load_zero_points(points_file)
            loop_paths = {}
            for folder in LOOP_PATHS_DIR.glob("*"):
                if part_key in folder.name.upper():
                    candidate = folder / "scan_point_loops.json"
                    if candidate.is_file():
                        loop_paths = load_loop_paths(candidate)
                    break
            # 현업 제공 key_zero_point_engine(2026-08-25)을 두 군데에 쓴다.
            #  1) 필터 — 0포인트 후보 중 "주요 0포인트"(K1..Kn)만 남긴다.
            #  2) 순위 — 그 주요 점들 중 어느 둘을 제로라인 끝점으로 삼을지
            #     엔진이 잰 컬러바 실측 |편차|(mean_abs_deviation_mm)로 고른다.
            # 전에는 1)만 쓰고 끝점은 우리 strength(라벨 부호전환 크기)로
            # 골랐는데, 그건 엔진이 실제로 잰 값을 버리는 셈이었다.
            key_scores = None
            for key_folder in KEY_ZERO_POINTS_DIR.glob("*"):
                if part_key in key_folder.name.upper():
                    key_json = key_folder / "key_zero_points.json"
                    if key_json.is_file():
                        raw_points = filter_to_key_points(raw_points, key_json)
                        key_scores = load_key_scores(key_json)
                    break
            zero_point_clusters = cluster_zero_points(
                raw_points, loop_paths=loop_paths, key_scores=key_scores)
            # 직선 영라인의 시작·끝점은 반드시 이 주요 0포인트여야 한다
            # (현업 zero_line_drawing README). 존으로 넓히기 전 중심을 쓴다.
            if zero_output is not None:
                for cluster in zero_point_clusters:
                    x, y, moved = snap_into_mask(
                        zero_output.part_mask, cluster.center[0], cluster.center[1])
                    if moved <= 60.0:
                        simple_key_points.append(
                            (x, y, max(float(cluster.span) / 2.0, 20.0)))
            if zero_output is not None and zero_point_clusters:
                line = connect_strongest_pair(
                    zero_point_clusters,
                    calibrated_values if calibrated_values is not None else zero_output.values,
                    zero_output.part_mask,
                    float(zero_output.result.tolerance),
                )
                if line is not None:
                    label_zero_line = line.to_dict()
            # 선을 만든 뒤 모든 군집을 존으로 넓힌다(끝점 선택은 원래
            # 군집 중심으로 이미 끝났으므로 영향 없다). 시트가 제로를
            # 여러 존으로 표기하는 부품(67XX6=9개, 71XX2=5개)에서
            # "점 2개를 이은 선 하나"만으로는 정답을 크게 놓쳤다 —
            # 실측 커버리지가 각각 19.8%/42.4%였고 존으로 내면
            # 5.6%/18.6%로 좋아진다(zero_points.py 문서 참고).
            zero_point_clusters = expand_clusters_to_zones(
                zero_point_clusters, loop_paths=loop_paths,
                part_mask=zero_output.part_mask if zero_output is not None else None)
    except Exception as exc:
        errors["zeroPoints"] = str(exc)

    # 현업이 준 영라인 선정 방법(2026-08-25) 그대로 — "녹색 영역" 과
    # "플러스/마이너스 전환대" 가 겹치는 **길쭉한 벨트**를 찾는다.
    # 두 0포인트를 억지로 잇지 않으므로, 측정 근거가 없는 자리에는
    # 아무것도 그리지 않는다(green_belt.py 문서에 실측 근거 있음).
    green_belts: list = []
    try:
        if zero_output is not None:
            belt_values = (
                calibrated_values if calibrated_values is not None
                else zero_output.values
            )
            green_belts = find_green_belts(belt_values, zero_output.part_mask)
    except Exception as exc:
        errors["greenBelts"] = str(exc)

    # 현업 zero_line_drawing(2026-08-26) 방식 — 주요 0포인트를 직선
    # 정렬도로 묶고, 편차 -0.5~+0.5mm 허용범위를 얼마나 지나가는지로
    # 채점한다. 곡선을 쓰지 않으므로 결과가 시트처럼 깔끔한 직선이다.
    simple_zero_lines: list = []
    try:
        if zero_output is not None and len(simple_key_points) >= 2:
            simple_zero_lines = find_simple_zero_lines(
                zero_output.values, zero_output.part_mask,
                simple_key_points, part_no=part_key,
            )
    except Exception as exc:
        errors["simpleZeroLines"] = str(exc)

    # my_lab 파이프라인이 그린 영라인(있는 품번만). 데모 화면은 이것을
    # 영라인으로 표시하고, 우리 검출은 대조용으로 숨겨 둔다.
    #
    # 처음엔 이걸 "승인 도면" 으로 알고 정답처럼 썼는데 아니었다 —
    # my_lab/zero_line_drawing 스크립트의 출력이다(lab_profile.py 참고).
    # 게다가 우리 검출과 같은 key_zero_points 를 끝점으로 쓰므로, 둘이
    # 가깝다는 것이 정확도의 근거가 되지 못한다.
    lab_profile: list = []
    lab_distance = None
    try:
        lab_profile = lab_shapes_for(part_key, width, height)
        if lab_profile and simple_zero_lines:
            lab_distance = distance_report(
                [l.points for l in simple_zero_lines], part_key, width, height)
    except Exception as exc:
        errors["labProfile"] = str(exc)

    # 이 품번에 확정된 제로라인이 등록돼 있으면 그걸 정답으로 쓴다.
    reference_line = None
    try:
        entry = load_library(ZERO_LINE_LIBRARY).get(part_key)
        if entry:
            # 부품에 따라 시트가 제로를 선으로 그리기도, 여러 존(면)으로
            # 칠하기도 한다 — 등록된 형태 그대로 내보낸다.
            reference_line = {
                "kind": entry.get("kind", "line"),
                "points": entry.get("points") or [],
                "contours": entry.get("contours") or [],
                "partNo": entry["part_no"],
                "sourceSheet": Path(entry["source_sheet"]).name,
                "mirrored": entry.get("mirrored", False),
            }
    except Exception as exc:
        errors["zeroReference"] = str(exc)

    # 클릭용 앵커는 라벨 실측값에서 나온 0포인트를 우선 쓴다. 컬러바
    # 색에서 추정한 앵커보다 근거가 확실하고, 부품 윤곽선 위에 있다.
    class _ClusterAnchor:
        def __init__(self, anchor_id, x, y, kind, strength):
            self.anchor_id, self.x, self.y = anchor_id, int(x), int(y)
            self.kind, self.strength = kind, strength

        def to_dict(self):
            return {
                "anchor_id": self.anchor_id, "x": self.x, "y": self.y,
                "boundary_arclen": 0.0, "source": "label_zero_point",
                "kind": self.kind, "strength": self.strength,
            }

    if zero_point_clusters and zero_output is not None:
        snapped = []
        for index, cluster in enumerate(zero_point_clusters, start=1):
            sx, sy, moved = snap_into_mask(
                zero_output.part_mask, cluster.center[0], cluster.center[1])
            if moved <= 60.0:
                snapped.append(_ClusterAnchor(
                    len(snapped) + 1, sx, sy, cluster.kind, cluster.strength))
        if len(snapped) >= 2:
            zero_anchors = snapped

    # 현업이 준 제로라인 파이프라인(lab_pipeline). 근거가 가장 분명한
    # 경로다 — "허용범위 밖 영역을 윤곽 위 제로포인트 둘로 닫는다".
    # 등록된 품번(64XX2·67XX6·71XX2)에서만 돈다.
    lab_zero: dict = {}
    if lab_runner.prefix_for(part_key):
        try:
            with _timed("현업 제로라인"):
                lab_zero = lab_runner.run(image, part_key)
            if lab_zero.get("error"):
                errors["labZero"] = lab_zero["error"]
        except Exception as exc:
            errors["labZero"] = str(exc)

    # 받은 제로 영역을 네모 몇 개로 바꾼다(zero_shapes 참고). 여기서 한
    # 번만 해 두면 시트와 3D 가 같은 도형을 본다 — 서로 다르게 보이던
    # 것이 이 때문이었다.
    lab_areas = zero_shapes.clean(lab_zero.get("areas") or [])

    # 보정시트에 실제로 적을 포인트를 고른다. 스캔에는 수십~백여 개가
    # 찍히지만 현업 시트에 적히는 건 열몇 개다(향후 계획 02번).
    key_points: list = []
    key_rejected: list = []
    try:
        key_points, key_rejected = select_key_points(
            points, width, height, part_no=part_key)
    except Exception as exc:
        errors["keyPoints"] = str(exc)

    analysis_id = None
    if zero_output is not None and zero_anchors and calibrated_values is not None:
        analysis_id = _cache_analysis({
            "values": calibrated_values,
            "part_mask": zero_output.part_mask,
            "tolerance": calibrated_tolerance,
            "anchors": zero_anchors,
            "overlay_base": cv2.cvtColor(
                clean_image if clean_image is not None else image, cv2.COLOR_BGR2RGB
            ),
            # /api/cad-overlay 가 3D 표면 위로 옮길 대상들
            "simple_zero_lines": [l.to_dict() for l in simple_zero_lines],
            # 현업 파이프라인 결과가 있으면 3D 에는 이걸 쓴다
            "lab_zero_lines": lab_zero.get("lines") or [],
            # 67XX6 처럼 7단계(가지 확장)가 있는 부품은 제로라인이
            # 선이 아니라 **영역**으로 나온다. 받은 그대로는 링을 따라
            # 구불구불한 띠라서 시트에도 3D 에도 못 쓴다 — 여기서 한 번만
            # 네모로 바꿔 두고 시트와 3D 가 **같은 것**을 쓴다.
            "lab_zero_areas": lab_areas,
            "deviation_points": points,
            "part_no": part_key,
            # 시트에 등록된 제로 표기. 67XX6 은 선이 아니라 **영역**이라
            # 3D 에서도 면으로 칠해야 한다.
            "zero_reference": reference_line,
        })

    return {
        "analysisId": analysis_id,
        "partNo": part_key,
        "naming": naming.to_dict(),
        "labZeroLines": lab_zero.get("lines") or [],
        "labZeroAreas": lab_areas,
        "labZeroRegions": lab_zero.get("regions") or [],
        "timings": spent,
        # 다시 읽지 않고 캐시에서 가져온 라벨 수. timings 와 단위가
        # 달라 따로 둔다.
        "reusedLabels": reused_labels,
        "keyPoints": [k.to_dict() for k in key_points],
        "keyPointsRejected": key_rejected,
        "knownParts": sorted(PRODUCT_COLORBAR_MM),
        "source": {"name": filename, "width": width, "height": height},
        "cleanImage": _png_data_url(clean_image) if clean_image is not None else None,
        "zeroOverlay": _png_data_url(zero_overlay, rgb=True) if zero_overlay is not None else None,
        "zeroMask": (
            _png_data_url(zero_datum_mask)
            if zero_datum_mask is not None and zero_datum_mask.any()
            else (_png_data_url(zero_output.mask) if zero_output is not None else None)
        ),
        "zeroCandidates": [c.to_dict() for c in zero_candidates[:8]],
        "zeroLines": [l.to_dict() for l in zero_lines],
        "zeroAnchors": [a.to_dict() for a in zero_anchors],
        "zeroPatches": [pt.to_dict() for pt in zero_patches],
        "advanceLine": advance_line,
        "zeroLineCandidates": [c.to_dict() for c in zero_line_candidates],
        "zeroPointClusters": [c.to_dict() for c in zero_point_clusters],
        "greenBelts": [b.to_dict() for b in green_belts],
        "simpleZeroLines": [l.to_dict() for l in simple_zero_lines],
        "labProfile": lab_profile,
        "labDistance": lab_distance,
        "labelZeroLine": label_zero_line,
        "referenceLine": reference_line,
        "points": points,
        "stats": {
            "labelsRemoved": label_count,
            "pointsDetected": len(points),
            "detectedCandidates": detected_candidates,
            "validCandidates": valid_candidates_count,
            "zeroRegions": zero_regions,
            "zeroRatio": round(zero_ratio, 4),
            "zeroTolerance": (
                round(float(zero_output.result.tolerance), 4)
                if zero_output is not None
                else None
            ),
            "qwenReads": qwen_reads,
            "qwenUnread": unread_labels,
            "calibration": calibration_stats,
        },
        "warnings": warnings,
        "warningsByEngine": {
            "deviation": deviation_warnings,
            "zero": zero_warnings,
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
            "engines": ["label_removal", "deviation_extraction", "zero_line_detection"],
            "folderAvailable": FOLDER_ROOT.is_dir(),
            "qwenCached": model_path is not None,
            "qwenLoaded": _reader is not None,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )


def read_sheet_callouts(payload: bytes) -> dict[str, Any]:
    """보정시트 이미지에서 보정치 콜아웃(값+좌표)을 전부 읽는다.

    학습 데이터 생성용 — 같은 위치의 스캔 실측값과 비교해 "측정값을
    그대로 뒤집은 것과 실제 보정치가 얼마나 다른지"를 계산하는 데 쓴다.
    """
    from zero_line_detection.sheet_values import (
        assemble_callouts, build_callout_crops, detect_callout_regions,
        detect_instruction_notes, detect_red_dots,
    )

    image = _decode_image(payload)
    boxes = detect_callout_regions(image)
    dots = detect_red_dots(image)
    crops = build_callout_crops(image, boxes)

    values: list[float | None] = []
    warning: str | None = None
    if crops:
        reader = _get_qwen_reader()
        values, warning = _read_qwen_values(reader, crops)

    callouts = assemble_callouts(boxes, dots, values)
    # "OO ea 절대높이 유지" 같은 지시문은 숫자가 아니라 값으로 안 읽힌다.
    # 어느 점에 적용되는지까지는 자동으로 못 풀어서(sheet_values.py 문서
    # 참고) 위치만 알려주고 사람이 확인하게 한다.
    notes = detect_instruction_notes(image)
    result: dict[str, Any] = {
        "width": image.shape[1],
        "height": image.shape[0],
        "callouts": [c.to_dict() for c in callouts],
        "boxesDetected": len(boxes),
        "dotsDetected": len(dots),
        "instructionNotes": [list(box) for box in notes],
    }
    if warning:
        result["warning"] = warning
    return result


async def sheet_values(request: Request) -> JSONResponse:
    try:
        form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "보정시트 이미지가 필요합니다."}, status_code=400)
        payload = await upload.read()
        result = await run_in_threadpool(read_sheet_callouts, payload)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def analyze(request: Request) -> JSONResponse:
    try:
        form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "이미지 파일이 필요합니다."}, status_code=400)
        payload = await upload.read()
        image = _decode_image(payload)
        part_no = form.get("partNo")
        result = await run_in_threadpool(
            analyze_image, image, getattr(upload, "filename", "scan.png"),
            part_no if isinstance(part_no, str) else None,
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


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


def sheet_excel_for(analysis_id: str, corrections: dict,
                    meta: dict, images: list | None = None) -> bytes:
    """최종 보정시트를 현업 엑셀 양식으로 만든다.

    보정량은 화면이 준다 — 작업자가 고친 값과 계수가 반영된 최종값이다.
    여기서 다시 계산하면 시트와 엑셀이 어긋난다.

    [그림]
    현업 시트("보정 적용 내용")는 스캔 히트맵이 아니라 **3D 형상 그림**을
    쓴다. 그래서 화면에서 찍은 3D 뷰를 받으면 그걸 쓰고, 없으면 스캔에
    콜아웃을 그려 넣은 그림으로 대신한다.
    여러 장을 주면 페이지를 나눠 넣는다 — 실제 시트도 전체도와 확대도를
    따로 싣는다.
    """
    from zero_line_detection.sheet_excel import (
        SheetPoint, build_workbook, draw_sheet_image,
    )

    entry = _analysis_cache.get(analysis_id)
    if entry is None:
        raise ValueError("분석 결과가 만료됐습니다. 이미지를 다시 분석하세요.")

    base_rgb = entry.get("overlay_base")
    if base_rgb is None:
        raise ValueError("시트에 쓸 그림이 없습니다.")
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

    points = []
    for point in entry.get("deviation_points", []):
        point_id = point.get("id")
        if point_id not in corrections:      # 작업자가 숨긴 포인트
            continue
        points.append(SheetPoint(
            point_id=point_id,
            x_px=int(point.get("xPx", 0)),
            y_px=int(point.get("yPx", 0)),
            deviation=float(point.get("value", 0.0)),
            correction=float(corrections[point_id]),
        ))
    points.sort(key=lambda p: p.point_id)

    pages: list = []
    for raw in (images or []):
        text = str(raw or "")
        if "," in text:
            text = text.split(",", 1)[1]
        try:
            buffer = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
            shot = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception:
            shot = None
        if shot is not None:
            pages.append(shot)
    # 1쪽은 **항상** 스캔에 보정치를 그린 전체도다. 예전에는 3D 화면을
    # 담아 두면 그걸로 1쪽을 통째로 대체해서, 정합이 어긋난 CAD 캡처
    # 한 장이 시트가 됐다 — "71XX2 로 만들었는데 다른 제품이 나온다" 는
    # 말이 그것이었다. 현업 시트도 전체도가 먼저고 상세도가 뒤따른다.
    pages = [draw_sheet_image(base_bgr, points)] + pages

    return build_workbook(
        pages, points,
        part_no=str(meta.get("partNo") or entry.get("part_no") or ""),
        part_name=str(meta.get("partName") or ""),
        process=str(meta.get("process") or ""),
        material=str(meta.get("material") or ""),
        control_no=str(meta.get("controlNo") or ""),
        applied_at=str(meta.get("appliedAt") or "") or None,
        coefficient=float(meta.get("coefficient") or 1.0),
        processes=[str(x) for x in (meta.get("processes") or [])],
    )


async def sheet_excel(request: Request) -> Response:
    try:
        body = await request.json()
        corrections = body.get("corrections") or {}
        if not isinstance(corrections, dict) or not corrections:
            return JSONResponse({"error": "보정량이 비어 있습니다."}, status_code=400)
        images = body.get("images")
        if isinstance(images, str):
            images = [images]
        payload = await run_in_threadpool(
            sheet_excel_for, str(body.get("analysisId") or ""),
            {str(k): float(v) for k, v in corrections.items()},
            body.get("meta") or {},
            images if isinstance(images, list) else None,
        )
        name = str(body.get("filename") or "보정시트") + ".xlsx"
        quoted = urllib.parse.quote(name)
        return Response(
            payload,
            media_type=("application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"),
            headers={"Content-Disposition":
                     f"attachment; filename*=UTF-8''{quoted}"},
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


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

    return {
        "positions": [round(float(v), 4) for v in moved.ravel()],
        "shift": [round(float(v), 4) for v in shift],
        "stats": stats.to_dict(),
        "points": len(spots),
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


async def cad_morph_stl(request: Request) -> Response:
    """보정 후 형상을 STL 로 내보낸다."""
    try:
        import trimesh
        body = await request.json()
        result = await run_in_threadpool(
            cad_morph_for, str(body.get("cadId") or ""),
            {str(k): float(v) for k, v in (body.get("corrections") or {}).items()},
            {str(k): v for k, v in (body.get("positions") or {}).items()},
            float(body.get("reachRatio") or 0.04),
        )
        entry = _cad_cache.get(str(body.get("cadId") or "")) or {}
        mesh = trimesh.Trimesh(
            vertices=np.asarray(result["positions"], dtype=float).reshape(-1, 3),
            faces=np.asarray(entry["display_faces"]), process=False)
        payload = mesh.export(file_type="stl")
        name = f"{entry.get('name', 'part')}_보정후.stl"
        quoted = urllib.parse.quote(name)
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
        form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "3D 파일이 필요합니다."}, status_code=400)
        payload = await upload.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"error": f"파일이 너무 큽니다 ({len(payload) / 1024 / 1024:.0f}MB). "
                          f"최대 {MAX_UPLOAD_BYTES // 1024 // 1024}MB"},
                status_code=413,
            )
        result = await run_in_threadpool(
            load_cad_payload, payload, getattr(upload, "filename", "part.step")
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


def zero_valley_line_for(analysis_id: str, anchor_id_a: int, anchor_id_b: int) -> dict[str, Any]:
    entry = _analysis_cache.get(analysis_id)
    if entry is None:
        raise ValueError("분석 결과가 만료됐습니다. 이미지를 다시 분석하세요.")
    anchors = entry["anchors"]
    by_id = {a.anchor_id: a for a in anchors}
    if anchor_id_a not in by_id or anchor_id_b not in by_id:
        raise ValueError("존재하지 않는 앵커 ID 입니다.")
    if anchor_id_a == anchor_id_b:
        raise ValueError("서로 다른 앵커 2개를 선택하세요.")

    pair = [by_id[anchor_id_a], by_id[anchor_id_b]]
    lines = find_valley_lines(
        entry["values"], entry["part_mask"], pair, entry["tolerance"],
        max_quality_ratio=100.0,   # 사람이 직접 고른 쌍이므로 비용으로 거르지 않는다
        min_length_px=0.0,
        max_uses_per_anchor=2,
    )
    if not lines:
        raise ValueError("두 앵커를 잇는 경로를 찾지 못했습니다 (부품 영역 밖일 수 있음).")
    line = lines[0]
    return {"line": line.to_dict()}


async def zero_valley_line(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        analysis_id = body.get("analysisId")
        anchor_ids = body.get("anchorIds")
        if not analysis_id or not isinstance(anchor_ids, list) or len(anchor_ids) != 2:
            return JSONResponse(
                {"error": "analysisId 와 anchorIds(길이 2) 가 필요합니다."}, status_code=400
            )
        result = await run_in_threadpool(
            zero_valley_line_for, analysis_id, int(anchor_ids[0]), int(anchor_ids[1])
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
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


app = Starlette(
    routes=[
        Route("/api/health", health, methods=["GET"]),
        Route("/api/analyze", analyze, methods=["POST"]),
        Route("/api/sheet-values", sheet_values, methods=["POST"]),
        Route("/api/zero-valley-line", zero_valley_line, methods=["POST"]),
        Route("/api/cad", cad, methods=["POST"]),
        Route("/api/cad-overlay", cad_overlay, methods=["POST"]),
        Route("/api/cad-sections", cad_sections, methods=["POST"]),
        Route("/api/cad-morph", cad_morph, methods=["POST"]),
        Route("/api/cad-morph-stl", cad_morph_stl, methods=["POST"]),
        Route("/api/sheet-excel", sheet_excel, methods=["POST"]),
        Route("/api/sample", sample, methods=["POST"]),
        Route("/api/folders", folders, methods=["GET"]),
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
