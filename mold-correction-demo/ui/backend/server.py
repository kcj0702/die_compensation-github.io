"""Local-only API that connects the React UI to the three vision engines."""

from __future__ import annotations

import base64
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
DEVIATION_DIR = PROJECT_DIR / "deviation_extraction"

# deviation_extraction currently uses local-style imports (import config), so
# its own folder must precede the project root on sys.path.
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(DEVIATION_DIR))

from label_detector import detect_labels  # noqa: E402
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
from zero_line_detection.calibration import calibrate_with_points  # noqa: E402
from zero_line_detection.zero_valley import find_valley_lines  # noqa: E402


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


def _find_qwen_model() -> Path | None:
    configured = os.environ.get("AJIN_QWEN_MODEL_PATH")
    candidates = [Path(configured)] if configured else []
    if QWEN_CACHE_DIR.is_dir():
        candidates.extend(
            sorted(QWEN_CACHE_DIR.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
        )
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "config.json").is_file()
            and (candidate / "model.safetensors.index.json").is_file()
            and (candidate / "tokenizer.json").is_file()
        ):
            return candidate
    return None


def _get_qwen_reader() -> LabelValueReader:
    global _reader
    if _reader is not None:
        return _reader
    with _reader_lock:
        if _reader is None:
            model_path = _find_qwen_model()
            if model_path is None:
                raise FileNotFoundError("Qwen2.5-VL-3B 로컬 모델을 찾지 못했습니다.")
            _reader = LabelValueReader(
                model_id=str(model_path),
                device="cuda",
                local_files_only=True,
                use_8bit=True,
            )
    return _reader


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


def _point_value(values: np.ndarray | None, image: np.ndarray, x: int, y: int) -> float:
    """Read a robust local value around a detected leader endpoint."""
    h, w = image.shape[:2]
    x0, x1 = max(0, x - 3), min(w, x + 4)
    y0, y1 = max(0, y - 3), min(h, y + 4)
    if values is not None:
        patch = values[y0:y1, x0:x1]
        finite = patch[np.isfinite(patch)]
        if finite.size:
            return round(float(np.median(finite)), 3)

    # This fallback is used only when zero-line colorbar detection fails.
    # It preserves a useful sign/magnitude estimate without any network model.
    b, g, r = image[y, x].astype(float)
    return round(float(np.clip((r - b) / 255.0 * 3.0, -3.0, 3.0)), 3)


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
    fallback_reads = 0
    try:
        candidates = detect_labels(image)
        values = zero_output.values if zero_output is not None else None
        valid_candidates = [candidate for candidate in candidates if candidate.point_xy is not None]
        reader = _get_qwen_reader() if valid_candidates else None
        crops = []
        for candidate in valid_candidates:
            box_x, box_y, box_w, box_h = candidate.box
            pad = 4
            crop = image[
                max(0, box_y - pad):min(height, box_y + box_h + pad),
                max(0, box_x - pad):min(width, box_x + box_w + pad),
            ]
            from PIL import Image as PILImage

            crops.append(PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        qwen_values = reader.read_values(crops) if reader is not None else [None] * len(crops)

        for index, (candidate, qwen_value) in enumerate(
            zip(valid_candidates, qwen_values), start=1
        ):
            x, y = candidate.point_xy
            if qwen_value is None:
                value = _point_value(values, image, x, y)
                confidence = "qwen_not_read|colorbar_fallback"
                fallback_reads += 1
            else:
                value = round(float(qwen_value), 3)
                confidence = "ok"
                qwen_reads += 1
            points.append(
                {
                    "id": f"P-{index:02d}",
                    "xPx": x,
                    "yPx": y,
                    "x": round(x / width * 100, 3),
                    "y": round(y / height * 100, 3),
                    "value": value,
                    "labelColor": candidate.label_color,
                    "confidence": confidence,
                }
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

    zero_regions = len(zero_output.result.regions) if zero_output is not None else 0
    zero_ratio = zero_output.result.zero_ratio if zero_output is not None else 0.0
    warnings = zero_output.warnings if zero_output is not None else []

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
        "points": points,
        "stats": {
            "labelsRemoved": label_count,
            "pointsDetected": len(points),
            "zeroRegions": zero_regions,
            "zeroRatio": round(zero_ratio, 4),
            "zeroTolerance": (
                round(float(zero_output.result.tolerance), 4)
                if zero_output is not None
                else None
            ),
            "qwenReads": qwen_reads,
            "fallbackReads": fallback_reads,
            "calibration": calibration_stats,
        },
        "warnings": warnings,
        "errors": errors,
        "valueMode": "Qwen2.5-VL-3B GPU 8-bit 판독",
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
