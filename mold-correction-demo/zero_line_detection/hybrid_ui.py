"""UI adapter for the agreed hybrid zero-line engine.

Case 1 keeps the in-house area decision.  Case 2 runs the preserved original
route selector as one in-memory pipeline.  Keeping this adapter
here lets the UI use one engine without copying either implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from zero_line_detection.visualize import make_overlay
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line


PACKAGE_DIR = Path(__file__).resolve().parent

_CACHE_SCHEMA = "hybrid-zero-v6"
_CACHE_LIMIT = 32


def _engine_fingerprint() -> str:
    """Hash every source file that can change the hybrid result."""
    files = [
        Path(__file__),
        PACKAGE_DIR / "zero_line.py",
        PACKAGE_DIR / "colorbar.py",
        PACKAGE_DIR / "annotations.py",
        PACKAGE_DIR / "generate_final_hybrid_zero_line.py",
        PACKAGE_DIR / "case2_route_adapter.py",
        PACKAGE_DIR / "case2_route_selector.py",
        PACKAGE_DIR / "adaptive_bundle" / "generate_adaptive_zero_line_preview.py",
    ]
    files.extend(sorted((PACKAGE_DIR / "case2_original_pipeline").glob("*.py")))
    digest = hashlib.sha256(_CACHE_SCHEMA.encode("ascii"))
    for path in files:
        try:
            digest.update(path.relative_to(PACKAGE_DIR).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        except OSError:
            digest.update(str(path).encode("utf-8", errors="replace"))
    return digest.hexdigest()[:20]


_ENGINE_FINGERPRINT = _engine_fingerprint()


def _cache_dir() -> Path:
    configured = os.environ.get("ADC_LAB_CACHE", "").strip()
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parent / ".lab_cache"


def _digest_array(digest, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())


def _digest_base(digest, base: Any) -> None:
    """Include supplied basic-detection inputs that can alter the result."""
    if base is None:
        digest.update(b"base:none")
        return
    digest.update(b"base:supplied")
    for name in ("values", "part_mask", "mask", "zero_crossing"):
        value = getattr(base, name, None)
        digest.update(name.encode("ascii"))
        if value is None:
            digest.update(b":none")
        else:
            _digest_array(digest, value)
    colorbar = getattr(base, "colorbar", None)
    result = getattr(base, "result", None)
    metadata = {
        "colorbar_vmin": getattr(colorbar, "vmin", None),
        "colorbar_vmax": getattr(colorbar, "vmax", None),
        "tolerance": getattr(result, "tolerance", None),
        "tolerance_unit": getattr(result, "tolerance_unit", None),
    }
    digest.update(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8"))


def _cache_key(
    image_bgr: np.ndarray,
    filename: str,
    base: Any = None,
    colorbar_range_mm: tuple[float, float] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(_ENGINE_FINGERPRINT.encode("ascii"))
    digest.update(filename.upper().encode("utf-8", errors="replace"))
    _digest_array(digest, image_bgr)
    _digest_base(digest, base)
    digest.update(json.dumps(colorbar_range_mm).encode("ascii"))
    return digest.hexdigest()


def _cache_path(
    image_bgr: np.ndarray,
    filename: str,
    base: Any = None,
    colorbar_range_mm: tuple[float, float] | None = None,
) -> Path:
    return _cache_dir() / f"{_cache_key(image_bgr, filename, base, colorbar_range_mm)}.npz"


def _load_cached(path: Path, shape: tuple[int, int]) -> "HybridZeroLineOutput | None":
    try:
        with np.load(path, allow_pickle=False) as payload:
            mask = payload["mask"].astype(bool)
            overlay = payload["overlay"].astype(np.uint8)
            metadata = json.loads(payload["metadata"].tobytes().decode("utf-8"))
        if mask.shape != shape or overlay.shape[:2] != shape:
            raise ValueError("cached zero-line shape does not match the input")
        return HybridZeroLineOutput(
            mask=mask,
            overlay_rgb=overlay,
            case=int(metadata["case"]),
            regions=int(metadata["regions"]),
            ratio=float(metadata["ratio"]),
            lines=list(metadata["lines"]),
            warnings=list(metadata["warnings"]),
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _prune_cache(directory: Path) -> None:
    try:
        entries = [
            item
            for item in directory.glob("*.npz")
            if len(item.stem) == 64 and all(character in "0123456789abcdef" for character in item.stem)
        ]
        entries.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        for stale in entries[_CACHE_LIMIT:]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


def _save_cached(path: Path, output: "HybridZeroLineOutput") -> None:
    metadata: dict[str, Any] = {
        "case": output.case,
        "regions": output.regions,
        "ratio": output.ratio,
        "lines": output.lines,
        "warnings": output.warnings,
    }
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
        encoded = np.frombuffer(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        )
        np.savez_compressed(
            temporary,
            mask=output.mask.astype(np.uint8),
            overlay=output.overlay_rgb.astype(np.uint8),
            metadata=encoded,
        )
        os.replace(temporary, path)
        _prune_cache(path.parent)
    except OSError:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


@dataclass
class HybridZeroLineOutput:
    mask: np.ndarray
    overlay_rgb: np.ndarray
    case: int
    regions: int
    ratio: float
    lines: list[dict]
    warnings: list[str]


def _routes_to_review_mask(
    selections: list[dict], shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize case-2 routes with the shared reviewed-engine contract."""
    from zero_line_detection.generate_final_hybrid_zero_line import routes_to_mask

    return routes_to_mask(selections, shape).astype(bool)


def _matching_review_spec(filename: str):
    """Return the bundled review specification for a known standard scan.

    The correction-only review inputs include the inpainted scan, its colour
    map and the gray-area sign assignment.  They are the authoritative input
    for the supplied standard scans; re-estimating those from an uploaded PNG
    changes the connected components and can select the wrong case.
    """
    from zero_line_detection.adaptive_bundle import generate_adaptive_zero_line_preview as kdt

    normalized = filename.upper()
    return next(
        (spec for spec in kdt.SPECS if spec.key.upper() in normalized), None
    )


def _mask_contours_as_lines(mask: np.ndarray) -> list[dict]:
    """Expose each final zero region boundary in the response schema.

    `cv2.findContours` gives an open point list (last point does not repeat
    the first) even though the boundary is a closed loop. Consumers that draw
    this as an open polyline(<polyline> in the UI) would then show every
    region missing its last edge. Repeating the first point closes it.

    Case 1 is produced from a raster area mask, so even CHAIN_APPROX_SIMPLE can
    leave a handle at every tiny pixel stair-step.  Those points add no useful
    editing precision.  Increase Douglas-Peucker tolerance only until the
    editable contour has at most 32 vertices; the detection mask and raster
    overlay remain untouched.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    lines: list[dict] = []
    for index, contour in enumerate(contours, start=1):
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(2.0, perimeter * 0.0015)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        while len(simplified) > 32 and epsilon < perimeter * 0.03:
            epsilon *= 1.35
            simplified = cv2.approxPolyDP(contour, epsilon, True)
        points = simplified.reshape(-1, 2)
        if len(points) >= 2:
            closed = np.vstack([points, points[:1]])
            lines.append({"id": index, "points": closed.tolist()})
    return lines


def _detect_from_review_inputs(
    image_bgr: np.ndarray, filename: str
) -> HybridZeroLineOutput | None:
    """Run the exact review pipeline for a bundled standard scan.

    ``None`` means arbitrary input, which continues through the generic
    in-memory path.  A size mismatch also must not apply a mask at wrong
    pixels.
    """
    from zero_line_detection.adaptive_bundle import generate_adaptive_zero_line_preview as kdt
    from zero_line_detection import generate_final_hybrid_zero_line as hybrid

    spec = _matching_review_spec(filename)
    if spec is None:
        return None
    common = hybrid.load_common_inputs(spec, hybrid.DEFAULT_CORRECTION_DIR)
    if tuple(common["part"].shape) != tuple(image_bgr.shape[:2]):
        return None

    selected_case = hybrid.select_case(common["zero_ratio"], common["zero_count"])
    if selected_case == 1:
        final_mask, details = hybrid.run_case1(common)
        lines = _mask_contours_as_lines(final_mask)
        # Keep the review result untouched: it is drawn on the inpainted
        # source, with the blue zero-area fill and its 4 px boundary.  The
        # uploaded original has labels/noise that were explicitly removed
        # before correction and zero-line detection.
        _board, overlay = kdt.build_board(
            common["image"],
            common["positive"],
            common["negative"],
            common["zero"],
            "case1_contour_polygon",
            details,
            None,
            common["zero_ratio"],
            common["zero_count"],
        )
    else:
        final_mask, details = hybrid.run_case2(common)
        lines = []
        for index, selection in enumerate(details["selections"], start=1):
            points = np.asarray(
                selection["closure_validation"]["route"]["path_points"],
                dtype=np.int32,
            )
            if len(points) >= 2:
                lines.append({"id": index, "points": points.tolist()})
        _construction, overlay = hybrid.draw_team_route_view(
            common["image"].copy(), details, final_mask
        )

    return HybridZeroLineOutput(
        mask=final_mask.astype(bool),
        overlay_rgb=overlay,
        case=selected_case,
        regions=len(lines),
        ratio=float(final_mask.sum()) / max(1, int(common["part_px"])),
        lines=lines,
        warnings=[
            f"검토 입력 재사용: {spec.key} / {selected_case}번 방식 "
            f"(제로 가능영역 {common['zero_ratio']:.2%}, "
            f"{common['zero_count']}개)"
        ],
    )


def _detect_hybrid_zero_line_uncached(
    image_bgr: np.ndarray,
    filename: str,
    base=None,
    decision_bgr: np.ndarray | None = None,
    colorbar_range_mm: tuple[float, float] | None = None,
) -> HybridZeroLineOutput:
    """Detect a UI-ready zero result, with a safe case-1 fallback.

    The distribution rule is shared with the review engine: separated zero
    components whose total area is below 40% choose case 1; all other inputs
    choose the original case-2 routing implementation.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    decision_rgb = cv2.cvtColor(
        image_bgr if decision_bgr is None else decision_bgr, cv2.COLOR_BGR2RGB
    )
    if colorbar_range_mm is None:
        raise ValueError("하이브리드 제로라인 검출에 컬러바 물리 범위가 필요합니다.")
    vmin, vmax = colorbar_range_mm
    if base is None:
        base = detect_zero_line(
            rgb, ZeroLineConfig(vmin=vmin, vmax=vmax), source_name=filename
        )
    fallback_part_px = max(1, int(base.part_mask.sum()))
    fallback_ratio = float(base.mask.astype(bool).sum()) / fallback_part_px
    try:
        from zero_line_detection import case2_route_adapter as case2
        from zero_line_detection import generate_final_hybrid_zero_line as hybrid
        from zero_line_detection.adaptive_bundle import generate_adaptive_zero_line_preview as kdt

        # 판정 입력도 검토 엔진과 같은 보정영역 생성 단계를 거친다. 단순
        # +/-0.6 threshold 는 67XX6의 영역을 합치기 전 상태로 세어 Case 2로
        # 잘못 보낼 수 있었다.
        info = base.colorbar.info
        legend_rgb = rgb[info.y0:info.y1, info.x0:info.x1]
        common = hybrid.build_common_from_review_mapping(
            decision_rgb, legend_rgb, vmin, vmax
        )
        part = common["part"]
        part_px = common["part_px"]
        positive = common["positive"]
        negative = common["negative"]
        zero = common["zero"]
        selected_case = hybrid.select_case(
            common["zero_ratio"], common["zero_count"]
        )
        print(
            f"[zero] decision ratio={common['zero_ratio']:.4f} "
            f"components={common['zero_count']} case={selected_case}",
            flush=True,
        )

        if selected_case == 1:
            final_mask, details = hybrid.run_case1(common)
            # Use the same renderer as the reviewed experiment.  The previous
            # UI-only cyan tint omitted correction shading, the blue polygon
            # fill/boundary and Z labels even when the polygon mask matched.
            _board, overlay = kdt.build_board(
                common["image"],
                common["positive"],
                common["negative"],
                common["zero"],
                "case1_contour_polygon",
                details,
                None,
                common["zero_ratio"],
                common["zero_count"],
            )
            # [버그였던 부분] Case 1(영역/다각형) 은 "선" 이 아니라 "면" 이라는
            # 이유로 lines=[] 를 그냥 박아 뒀다. 하지만 다각형도 윤곽선을 따면
            # 얼마든지 폴리라인으로 낼 수 있다 — 검토용(리뷰 자산) 경로의
            # `_mask_contours_as_lines` 가 이미 이 방식을 쓴다. 여기서도 같은
            # 함수를 재사용하면, 시트 화면(제품데이터 위 SVG 오버레이)에도
            # Case 1 결과가 실제로 그려진다. 지금까지는 review 자산이 없는
            # 이 PC 에서 Case 1 로 떨어진 스캔(67XX6 등)은 항상 "라인 없음"
            # 으로 보였다.
            return HybridZeroLineOutput(
                mask=final_mask.astype(bool), overlay_rgb=overlay, case=1,
                regions=int(cv2.connectedComponents(final_mask.astype(np.uint8))[0] - 1),
                ratio=float(final_mask.sum()) / part_px,
                lines=_mask_contours_as_lines(final_mask.astype(np.uint8)),
                warnings=list(base.warnings) + ["하이브리드 Case 1: ±0.6 mm 보정영역 기반 오프셋 다각형 결과입니다."],
            )

        routed = case2.run_original_case2_pipeline(
            original_bgr=image_bgr,
            colorbar_range_mm=colorbar_range_mm,
        )
        lines: list[dict] = []
        for index, selection in enumerate(routed["selections"], start=1):
            points = np.asarray(selection["closure_validation"]["route"]["path_points"], dtype=np.int32)
            if len(points) < 2:
                continue
            lines.append({"id": index, "points": points.tolist()})
        # Use the review engine's exact 4 px / LINE_8 rasterisation.  Redrawing
        # here with 5 px antialiasing made identical routes look much thicker.
        mask = _routes_to_review_mask(routed["selections"], image_bgr.shape[:2])
        overlay_base = cv2.cvtColor(routed["cleaned_bgr"], cv2.COLOR_BGR2RGB)
        _construction, overlay = hybrid.draw_team_route_view(overlay_base, routed, mask)
        route_ratio = float(mask.sum()) / part_px
        return HybridZeroLineOutput(
            mask=mask, overlay_rgb=overlay, case=2, regions=len(lines), ratio=route_ratio,
            lines=lines, warnings=list(base.warnings),
        )
    except Exception as exc:
        overlay = make_overlay(rgb, base.mask, base.centerline, zero_crossing=base.zero_crossing)
        # 위 Case 1 분기와 같은 이유로, 여기서도 base.mask(영역) 를 폴리라인
        # 윤곽선으로 뽑아 lines 를 채운다 — 컬러바 인식 실패 등으로 case2 가
        # 죽어 이 최종 폴백까지 떨어져도 최소한 "면적 윤곽" 은 벡터로 보인다.
        try:
            fallback_lines = _mask_contours_as_lines(base.mask.astype(np.uint8))
        except Exception:
            fallback_lines = []
        return HybridZeroLineOutput(
            mask=base.mask.astype(bool), overlay_rgb=overlay, case=1,
            regions=len(base.result.regions), ratio=fallback_ratio, lines=fallback_lines,
            warnings=list(base.warnings) + [f"Case 2 경로 계산 실패로 Case 1 후보를 표시했습니다: {exc}"],
        )


def detect_hybrid_zero_line(
    image_bgr: np.ndarray,
    filename: str,
    base=None,
    decision_bgr: np.ndarray | None = None,
    colorbar_range_mm: tuple[float, float] | None = None,
) -> HybridZeroLineOutput:
    """Return a content-cached result and optionally reuse basic detection."""
    cache_image = image_bgr if decision_bgr is None else np.concatenate(
        (image_bgr.reshape(-1, 3), decision_bgr.reshape(-1, 3)), axis=0
    )
    path = _cache_path(cache_image, filename, base, colorbar_range_mm)
    cached = _load_cached(path, image_bgr.shape[:2]) if path.is_file() else None
    if cached is not None:
        return cached

    output = _detect_hybrid_zero_line_uncached(
        image_bgr,
        filename,
        base=base,
        decision_bgr=decision_bgr,
        colorbar_range_mm=colorbar_range_mm,
    )
    _save_cached(path, output)
    return output
