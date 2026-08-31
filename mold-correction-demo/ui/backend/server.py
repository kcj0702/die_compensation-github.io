"""Local-only API that connects the React UI to the three vision engines."""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
from product_alignment.alignment import (  # noqa: E402
    Alignment, estimate_alignment, is_inside, map_point, warp_scan_mask,
)
from product_alignment.compose import render_alignment_overlay  # noqa: E402
# label_detector에도 같은 이름의 함수가 있다. 이쪽은 label_removal의 규칙을 쓰는
# 부품 실루엣용이라 이름을 구분해 둔다.
from product_alignment.masks import (  # noqa: E402
    build_product_mask, build_scan_mask as build_part_silhouette,
)
from product_alignment.registry import (  # noqa: E402
    AlignmentStore, ProductLibrary, part_number_from_name, read_image,
)
from point_selection import select_key_points  # noqa: E402
from sheet_export import (  # noqa: E402
    SheetPoint, SheetView, TitleBlock, build_sheet, crop_view,
)
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


def _resolve_product_image(
    filename: str, uploaded: np.ndarray | None
) -> tuple[np.ndarray | None, str | None, list[str]]:
    """Pick the product-data image for this scan, upload first then library."""
    warnings: list[str] = []
    if uploaded is not None:
        return uploaded, "업로드한 이미지", warnings

    part_number = part_number_from_name(filename)
    if part_number is None:
        return None, None, warnings

    match = PRODUCT_LIBRARY.find(part_number)
    if match is None:
        warnings.append(
            f"품번 {part_number}의 제품데이터가 등록되어 있지 않습니다. "
            "제품데이터 이미지를 함께 올리면 다음부터 자동으로 사용합니다."
        )
        return None, None, warnings

    if not match.exact:
        warnings.append(
            f"{part_number}에 정확히 맞는 제품데이터가 없어 "
            f"{match.part_number}로 등록된 이미지를 사용했습니다."
        )
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
    """Estimate the scan-to-product transform, reusing a confirmed one."""
    warnings: list[str] = []
    scan_silhouette = build_part_silhouette(image)
    product_mask = build_product_mask(product_image)

    if flip_x is None and flip_y is None and part_number:
        saved = ALIGNMENT_STORE.load(part_number)
        if saved is not None:
            flip_x, flip_y = saved.flip_x, saved.flip_y
            warnings.append(
                f"{part_number}에 확정 저장된 방향(좌우 {saved.flip_x}, "
                f"상하 {saved.flip_y})을 사용했습니다."
            )

    alignment = estimate_alignment(
        scan_silhouette, product_mask, flip_x=flip_x, flip_y=flip_y
    )
    overlay = render_alignment_overlay(
        product_image, warp_scan_mask(alignment, scan_silhouette)
    )
    return alignment, overlay, warnings + list(alignment.warnings)


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
            alignment, alignment_overlay, alignment_warnings = _align_to_product(
                image, product_image, part_number, flip_x, flip_y
            )
            product_warnings.extend(alignment_warnings)
        except Exception as exc:  # engine errors must be shown per engine
            errors["product"] = str(exc)

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

    zero_regions = len(zero_output.result.regions) if zero_output is not None else 0
    zero_ratio = zero_output.result.zero_ratio if zero_output is not None else 0.0
    zero_warnings = list(zero_output.warnings) if zero_output is not None else []
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
        "zeroLines": [l.to_dict() for l in zero_lines],
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
            "zeroTolerance": (
                round(float(zero_output.result.tolerance), 4)
                if zero_output is not None
                else None
            ),
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


def build_sheet_bytes(
    product_image: np.ndarray, payload: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    """Turn the sheet the operator sees into the company Excel form.

    Point coordinates arrive as percentages of the product image, which is the
    same frame the UI draws in, so no alignment work is repeated here.
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

    title_values = payload.get("title") or {}
    title = TitleBlock(
        management_no=str(title_values.get("managementNo", "")),
        part_name=str(title_values.get("partName", "")),
        process=str(title_values.get("process", "")),
        part_no=str(title_values.get("partNo", "")),
        material=str(title_values.get("material", "")),
        applied_date=str(title_values.get("appliedDate", "")),
    )

    views = [SheetView(image=product_image, points=points)]
    skipped: list[str] = []
    for index, region in enumerate(payload.get("details") or [], start=1):
        label = str(region.get("label") or f"DETAIL {index}")
        try:
            views.append(
                crop_view(
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append(f"{label}: {exc}")

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "correction-sheet.xlsx"
        report = build_sheet(views, title, output=output)
        data = output.read_bytes()

    summary = {
        "pictures": report.pictures,
        "labels": report.labels,
        "leaders": report.leaders,
        "warnings": report.warnings + skipped,
    }
    return data, summary


async def sheet(request: Request) -> Response:
    """Return the correction sheet as an .xlsx download."""
    try:
        form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_UPLOAD_BYTES)
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

        data, summary = await run_in_threadpool(
            build_sheet_bytes, product_image, payload
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
                "X-Sheet-Summary": json.dumps(summary, ensure_ascii=False),
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
        Route("/api/sample", sample, methods=["POST"]),
        Route("/api/folders", folders, methods=["GET"]),
        Route("/api/realign", realign, methods=["POST"]),
        Route("/api/sheet", sheet, methods=["POST"]),
        Route("/api/products", products, methods=["GET", "POST"]),
        Route("/api/alignment", confirm_alignment, methods=["POST"]),
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    # 시트 생성 결과 요약은 헤더로 오므로 브라우저가 읽을 수 있게 열어 준다.
    expose_headers=["Content-Disposition", "X-Sheet-Summary"],
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
