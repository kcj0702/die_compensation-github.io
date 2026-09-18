"""UI adapter for the outer-zero-point / HSV-correction zero-line engine.

The response keeps the existing editable-line schema and ``case=2`` transport
contract, while construction is delegated to :mod:`contour_route_zero`.
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

_CACHE_SCHEMA = "final-zero-line-v1-sigma-outline-rules"
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
        PACKAGE_DIR / "contour_route_zero.py",
        PACKAGE_DIR / "final_zero_line.py",
        PACKAGE_DIR / "cad_feature_snap.py",
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
    mm_per_px: float | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(_ENGINE_FINGERPRINT.encode("ascii"))
    digest.update(filename.upper().encode("utf-8", errors="replace"))
    _digest_array(digest, image_bgr)
    _digest_base(digest, base)
    digest.update(json.dumps(colorbar_range_mm).encode("ascii"))
    # The physical scale decides how far from the outline a correction region still gets a line.
    digest.update(json.dumps(None if mm_per_px is None else round(float(mm_per_px), 4)).encode("ascii"))
    return digest.hexdigest()


def _cache_path(
    image_bgr: np.ndarray,
    filename: str,
    base: Any = None,
    colorbar_range_mm: tuple[float, float] | None = None,
    mm_per_px: float | None = None,
) -> Path:
    return _cache_dir() / f"{_cache_key(image_bgr, filename, base, colorbar_range_mm, mm_per_px)}.npz"


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
    mm_per_px: float | None = None,
) -> HybridZeroLineOutput:
    """Detect UI-ready lines from outer zero points and the sigma correction regions."""
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
    try:
        from zero_line_detection.cad_feature_snap import discover_local_ai_features
        from zero_line_detection.final_zero_line import (
            build_deviation_from_color_ramp,
            construct_final_zero_lines,
        )
        from zero_line_detection import generate_final_hybrid_zero_line as hybrid

        # Use the colour ramp detected from this exact upload, but map it with
        # the same hue interpolation as the reviewed experiment.
        # ``Colorbar.colors_rgb`` is stored vmin->vmax; the mapper accepts
        # vmax->vmin (top->bottom), hence reverse. Unlike the earlier reader,
        # this one fills the grey CAD drawing lines from the nearest readable
        # colour first, so doubled lines on hole rims and fillets no longer
        # read as the red end of the bar.
        values, part, reading = build_deviation_from_color_ramp(
            decision_rgb, np.asarray(base.colorbar.colors_rgb[::-1], dtype=np.uint8), vmin, vmax
        )
        cad_features = discover_local_ai_features(filename, PACKAGE_DIR.parents[2])
        result, report = construct_final_zero_lines(
            decision_rgb,
            values,
            part,
            cad_feature_lines=cad_features,
            mm_per_px=mm_per_px,
            colorbar_range=(vmin, vmax),
        )
        kinds = [row["kind"] for row in report["regions"]]
        print(
            f"[zero] final-rules sigma={report['sigma_mm']:.2f}mm "
            f"zero_points={len(result.zero_points)} regions={len(kinds)} "
            f"(outline {kinds.count('touching_outline')}, near {kinds.count('near_outline')}, "
            f"interior {kinds.count('interior_no_line')}, unresolved {kinds.count('unresolved')}) "
            f"noise_removed={report['noise_regions_removed']} lines={len(result.lines)} "
            f"near_outline={report['near_outline_px']}px mm_per_px={mm_per_px} "
            f"cad_features={len(cad_features)} cad_snaps={len(result.snap_records)} "
            f"grey_filled={reading.get('filled_px')}",
            flush=True,
        )
        overlay = hybrid.draw_final_selected_overlay(decision_rgb, result.mask)
        ratio = float(result.mask.sum()) / max(1, int(part.sum()))
        return HybridZeroLineOutput(
            mask=result.mask,
            overlay_rgb=overlay,
            case=2,
            regions=len(result.lines),
            ratio=ratio,
            lines=result.lines,
            warnings=list(base.warnings) + result.warnings,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Case 2 경로 계산 실패: 외곽 제로점 기반 제로라인 계산 실패: {exc}"
        ) from exc


def detect_hybrid_zero_line(
    image_bgr: np.ndarray,
    filename: str,
    base=None,
    decision_bgr: np.ndarray | None = None,
    colorbar_range_mm: tuple[float, float] | None = None,
    mm_per_px: float | None = None,
) -> HybridZeroLineOutput:
    """Return a content-cached result and optionally reuse basic detection."""
    cache_image = image_bgr if decision_bgr is None else np.concatenate(
        (image_bgr.reshape(-1, 3), decision_bgr.reshape(-1, 3)), axis=0
    )
    path = _cache_path(cache_image, filename, base, colorbar_range_mm, mm_per_px)
    cached = _load_cached(path, image_bgr.shape[:2]) if path.is_file() else None
    if cached is not None:
        return cached

    output = _detect_hybrid_zero_line_uncached(
        image_bgr,
        filename,
        base=base,
        decision_bgr=decision_bgr,
        colorbar_range_mm=colorbar_range_mm,
        mm_per_px=mm_per_px,
    )
    _save_cached(path, output)
    return output
