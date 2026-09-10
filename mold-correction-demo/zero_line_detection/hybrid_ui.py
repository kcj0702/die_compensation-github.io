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

_CACHE_SCHEMA = "hybrid-zero-v7-current-input-parity"
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
    selected_case: int | None = None
    try:
        from zero_line_detection import case2_route_adapter as case2
        from zero_line_detection import generate_final_hybrid_zero_line as hybrid
        from zero_line_detection.adaptive_bundle import generate_adaptive_zero_line_preview as kdt

        # Use the colour ramp detected from this exact upload, but map it with
        # the same hue interpolation and correction preparation as the
        # reviewed experiment. ``Colorbar.colors_rgb`` is stored vmin->vmax;
        # the reviewed mapper accepts vmax->vmin (top->bottom), hence reverse.
        # This needs neither a product-specific range nor a saved legend.
        common = hybrid.build_common_from_color_ramp(
            decision_rgb, base.colorbar.colors_rgb[::-1], vmin, vmax
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
        overlay = hybrid.draw_final_selected_overlay(decision_rgb, mask)
        route_ratio = float(mask.sum()) / part_px
        return HybridZeroLineOutput(
            mask=mask, overlay_rgb=overlay, case=2, regions=len(lines), ratio=route_ratio,
            lines=lines, warnings=list(base.warnings),
        )
    except Exception as exc:
        # The distribution decision already selected Case 2.  Returning the
        # basic area mask as ``case=1`` hides the real failure, caches a false
        # classification, and makes unrelated products appear to use Case 1.
        # Propagate the failure so callers can report/retry it without
        # changing the selected algorithm.
        if selected_case == 2:
            message = f"Case 2 경로 계산 실패: {exc}"
        elif selected_case == 1:
            message = f"Case 1 영역 계산 실패: {exc}"
        else:
            message = f"하이브리드 제로라인 계산 실패: {exc}"
        raise RuntimeError(message) from exc


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
