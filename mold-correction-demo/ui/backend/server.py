"""Local-only API that connects the React UI to the three vision engines."""

from __future__ import annotations

import base64
import json
import math
import os
import sys
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

from label_detector import build_scan_mask, detect_labels  # noqa: E402
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
    draw_zero_boundary, find_boundary_anchors, grow_patches,
)
from zero_line_detection.calibration import (  # noqa: E402
    calibrate_vmin_vmax, calibrate_with_points,
)
from zero_line_detection.zero_valley import find_valley_lines  # noqa: E402
from zero_line_advance.advance import (  # noqa: E402
    AdvanceConfig, detect_advanced_zero_line,
)


DEFAULT_FOLDER_ROOT = Path(
    r"C:\Users\KDT013\Desktop\금형보정치\경북대KDT(14기) 자료\품번별 폴더 정리 자료_예시"
)
FOLDER_ROOT = Path(os.environ.get("AJIN_FOLDER_ROOT", DEFAULT_FOLDER_ROOT)).resolve()
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

# 분석 1회분(값장·부품마스크·앵커·허용오차)을 잠깐 들고 있는 캐시.
# 로컬 1인용 데모라 세션 관리 없이 메모리 dict 로 충분하다 — 사람이
# 앵커 2개를 클릭해서 "선 잇기" 를 요청할 때 이미지를 다시 안 올리고,
# VLM 라벨 판독도 다시 안 돌리려는 목적.
_analysis_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_ANALYSIS_CACHE_MAX = 5


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
            zero_patches = grow_patches(
                calibrated_values, zero_output.part_mask, zero_anchors,
                tolerance=calibrated_tolerance,
            )
            zero_lines = extract_zero_polylines(
                calibrated_values, zero_output.part_mask
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
        Route("/api/zero-valley-line", zero_valley_line, methods=["POST"]),
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
