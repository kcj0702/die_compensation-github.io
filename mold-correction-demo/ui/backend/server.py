"""Local-only API that connects the React UI to the three vision engines."""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import tempfile
import threading
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
from starlette.responses import JSONResponse
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
    find_simple_zero_lines,
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


_cad_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_CAD_CACHE_MAX = 3      # 하나가 수백 MB 라 많이 들고 있으면 안 된다


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


def analyze_image(image: np.ndarray, filename: str) -> dict[str, Any]:
    height, width = image.shape[:2]
    errors: dict[str, str] = {}

    clean_image: np.ndarray | None = None
    label_count = 0
    try:
        label_count = len(detect_label_boxes(image))
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
        zero_output = detect_zero_line(rgb, ZeroLineConfig(), source_name=filename)
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
    part_key = part_no_from_name(filename)
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
        entry = load_library(ZERO_LINE_LIBRARY).get(part_no_from_name(filename))
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
            "deviation_points": points,
            "part_no": part_key,
        })

    return {
        "analysisId": analysis_id,
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
        result = await run_in_threadpool(
            analyze_image, image, getattr(upload, "filename", "scan.png")
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


def cad_overlay_for(cad_id: str, analysis_id: str,
                    coefficient: float = 1.0) -> dict[str, Any]:
    """스캔에서 뽑은 제로라인·보정 포인트를 CAD 표면 위로 옮긴다.

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

    mesh = cad_entry["mesh"]
    offset = cad_entry["offset"]
    vertices = np.asarray(mesh.vertices, dtype=float) - offset
    faces = np.asarray(mesh.faces)

    import trimesh
    shifted = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    fit = ov.fit_view(vertices, faces, analysis["part_mask"])

    lines = []
    for line in analysis.get("simple_zero_lines", []):
        placed = ov.unproject(line["points"], vertices, faces, fit, shifted)
        if placed:
            lines.append({"line_id": line.get("line_id"), "points": placed})

    # 컬러바 범위 밖의 값은 판독 오류다. 실측(JD_67XX6, 컬러바 +3.0~-3.0)
    # 에서 +9.00 이 5건 나왔다. 그대로 두면 화살표 길이 기준을 잡아먹어
    # 진짜 보정량(0.1~3mm)이 전부 점만 해진다.
    from zero_line_detection.simple_zero_line import PRODUCT_COLORBAR_MM
    folded = str(analysis.get("part_no") or "").upper().replace("-", "")
    span = next((v for k, v in PRODUCT_COLORBAR_MM.items() if k in folded), None)
    limit = max(abs(span[0]), abs(span[1])) * 1.05 if span else None

    points, rejected = [], []
    for point in analysis.get("deviation_points", []):
        value = float(point.get("value", 0.0))
        if limit is not None and abs(value) > limit:
            rejected.append({"id": point.get("id"), "value": round(value, 3)})
            continue
        placed = ov.unproject([[point["xPx"], point["yPx"]]],
                              vertices, faces, fit, shifted)
        if not placed:
            continue
        points.append({
            "id": point.get("id"),
            "position": placed[0],
            "value": round(value, 3),
            # 보정은 편차를 뒤집는 것이 기본이다. 계수는 화면에서 조절한다.
            "correction": round(-value * coefficient, 3),
        })

    return {"fit": fit.to_dict(), "zeroLines": lines, "points": points,
            "coefficient": coefficient, "rejected": rejected,
            "colorbarLimit": round(limit, 2) if limit else None}


async def cad_overlay(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await run_in_threadpool(
            cad_overlay_for,
            str(body.get("cadId") or ""),
            str(body.get("analysisId") or ""),
            float(body.get("coefficient") or 1.0),
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
