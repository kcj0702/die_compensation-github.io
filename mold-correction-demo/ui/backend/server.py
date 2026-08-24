"""Local-only API that connects the React UI to the three vision engines."""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
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

    zero_output = None
    zero_overlay: np.ndarray | None = None
    zero_candidates: list = []
    zero_datum_mask: np.ndarray | None = None
    zero_lines: list = []
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

        # 보정시트의 제로라인은 면이 아니라 선이다. 편차 부호가 바뀌는
        # 경계를 순서 있는 폴리라인으로 뽑아 시트와 같은 형태로 그린다.
        zero_lines = extract_zero_polylines(
            zero_output.values, zero_output.part_mask
        )
        if zero_lines:
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
    except Exception as exc:
        errors["zero"] = str(exc)

    points: list[dict[str, Any]] = []
    qwen_reads = 0
    fallback_reads = 0
    deviation_warnings: list[str] = []
    try:
        candidates = detect_labels(image)
        values = zero_output.values if zero_output is not None else None
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.point_xy is not None
            and 0 <= candidate.point_xy[0] < width
            and 0 <= candidate.point_xy[1] < height
        ]
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

        qwen_values: list[float | None] = [None] * len(crops)
        qwen_failure: str | None = None
        if crops:
            try:
                reader = _get_qwen_reader()
                read_values = list(reader.read_values(crops))
                qwen_values[: min(len(read_values), len(crops))] = read_values[: len(crops)]
                if len(read_values) != len(crops):
                    qwen_failure = (
                        f"Qwen 판독 결과 수가 후보 수와 다릅니다 "
                        f"({len(read_values)}/{len(crops)})."
                    )
            except Exception as exc:
                qwen_failure = str(exc)

        if qwen_failure:
            deviation_warnings.append(
                f"{qwen_failure} 판독하지 못한 포인트는 컬러바 값으로 대체했습니다."
            )

        for index, (candidate, qwen_value) in enumerate(
            zip(valid_candidates, qwen_values), start=1
        ):
            x, y = candidate.point_xy
            if qwen_value is None:
                value = _point_value(values, image, x, y)
                confidence = (
                    "qwen_unavailable|colorbar_fallback"
                    if qwen_failure
                    else "qwen_not_read|colorbar_fallback"
                )
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

    zero_regions = len(zero_output.result.regions) if zero_output is not None else 0
    zero_ratio = zero_output.result.zero_ratio if zero_output is not None else 0.0
    zero_warnings = list(zero_output.warnings) if zero_output is not None else []
    warnings = zero_warnings + deviation_warnings

    if qwen_reads and fallback_reads:
        value_mode = "Qwen2.5-VL-3B + 컬러바 대체 판독"
    elif qwen_reads:
        value_mode = "Qwen2.5-VL-3B 로컬 판독"
    elif fallback_reads:
        value_mode = "컬러바 기반 대체 판독"
    else:
        value_mode = "판독 결과 없음"

    return {
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
