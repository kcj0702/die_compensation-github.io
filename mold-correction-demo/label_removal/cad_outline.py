"""The part silhouette a CAD file casts on the scan, for outline-preserving label removal.

``outline_preserving`` can use the true part shape instead of guessing it from the image, but it
needs that shape in scan pixels. The 3D overlay already solves the same problem - it fits the mesh
to the scan and projects it - so this module only reuses that fit (``cad_import.overlay``, read
only) and rasterizes the silhouette. Nothing here changes the overlay or the alignment.

Two things are done differently from ``overlay._silhouette_raster`` on purpose:

* every triangle is filled on its own. One ``fillPoly`` call with many triangles fills them with the
  even-odd rule, so where the top skin, bottom skin and walls overlap the triangles cancel and the
  silhouette is riddled with holes - which would then be cut out of the scan as "background".
* the fit is only accepted when it really matches the scan (``MIN_SILHOUETTE_IOU``). A silhouette
  that sits a few pixels off would cut real product away, which is exactly what this work is meant
  to stop; when in doubt the caller falls back to the image-only mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# A fit below this outline overlap is not trusted to decide what is product and what is background.
# Measured on the seven zeroline_datas scans: the fits that keep the outline score 0.976-0.985,
# while the ones that cut thousands of real product pixels away score 0.903-0.925.
MIN_SILHOUETTE_IOU = 0.95
# Even a well-scoring fit is refused when it would cut away more than this share of the product the
# image alone found. Overlap is an average; this is the damage the silhouette would actually do.
MAX_CUT_RATIO = 0.005
# The same tolerance product_mask_with_cad keeps around the silhouette.
TOLERANCE_PX = 3


def silhouette_from_pixels(points_px: np.ndarray, faces: np.ndarray, image_shape) -> np.ndarray:
    """Filled part shadow from vertices already projected to scan pixels, one triangle at a time."""
    height, width = int(image_shape[0]), int(image_shape[1])
    points = np.asarray(points_px, dtype=np.float64)
    triangles = points[np.asarray(faces, dtype=np.int64)]
    inside = ((triangles[:, :, 0].max(axis=1) >= 0) & (triangles[:, :, 0].min(axis=1) < width)
              & (triangles[:, :, 1].max(axis=1) >= 0) & (triangles[:, :, 1].min(axis=1) < height))
    shift = 4                                                  # sub-pixel vertex positions
    polygons = np.round(triangles[inside] * (1 << shift)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, polygon, 1, lineType=cv2.LINE_8, shift=shift)
    return mask.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def silhouette_for_scan(
    vertices: np.ndarray,
    faces: np.ndarray,
    part_mask: np.ndarray,
    *,
    min_iou: float = MIN_SILHOUETTE_IOU,
    max_cut_ratio: float = MAX_CUT_RATIO,
    view_tries: int = 4,
    cache_dir: Path | None = None,
    pose_search: bool = True,
) -> tuple[np.ndarray | None, dict]:
    """(silhouette in scan pixels or None, report) for the mesh seen the way the scan sees it.

    part_mask: the product as the image alone sees it - the mask the fit is measured against.
    None comes back when no fitted view matches it closely enough (min_iou) or when the best one
    would still cut real product away (max_cut_ratio); the caller then keeps the image-only result
    rather than trusting a shape that does not match the scan.
    """
    from cad_import import overlay as ov

    mask = np.asarray(part_mask) > 0
    shape = mask.shape
    stencil = mask.astype(np.uint8) * 255
    report: dict = {"candidates": [], "min_iou": float(min_iou)}
    best: tuple[float, np.ndarray, object] | None = None

    def try_fit(name: str, fit, fit_vertices, fit_faces):
        nonlocal best
        try:
            fit = ov.refine_fit_by_silhouette(fit, fit_vertices, fit_faces, stencil)
            xs, ys = ov.to_pixels(fit_vertices, fit)
            silhouette = silhouette_from_pixels(np.stack([xs, ys], axis=1), fit_faces, shape)
        except Exception as exc:
            report["candidates"].append({"fit": name, "error": f"{type(exc).__name__}: {exc}"})
            return
        overlap = _iou(silhouette, mask)
        report["candidates"].append({"fit": name, "iou": round(overlap, 4), "axis": int(fit.axis),
                                     "sign": int(fit.sign), "mm_per_px": round(float(fit.mm_per_px), 4)})
        if best is None or overlap > best[0]:
            best = (overlap, silhouette, fit)

    # split_sides gives (vertices, faces, axis, middle, side) per half; a file holding an LH+RH pair
    # must be fitted one half at a time, as the 3D overlay does.
    all_vertices = np.asarray(vertices, dtype=np.float64)
    all_faces = np.asarray(faces, dtype=np.int64)
    for index, half in enumerate(ov.split_sides(all_vertices, all_faces)):
        half_vertices, half_faces = half[0], half[1]
        try:
            # Several poses, not just the best silhouette overlap: the top one is often a mirrored
            # or sideways view that scores nearly the same on the convex hull the search uses.
            guesses = ov.fit_view(half_vertices, half_faces, stencil, top_k=view_tries)
        except Exception as exc:                               # a half that cannot be fitted is skipped
            report["candidates"].append({"half": index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for rank, guess in enumerate(guesses if isinstance(guesses, (list, tuple)) else [guesses]):
            try_fit(f"half{index}#{rank}", guess, half_vertices, half_faces)

    # A scan taken at an angle matches no axis-aligned view. The 3D overlay searches all three
    # rotations in that case, so the same search is tried here before giving up.
    if pose_search and (best is None or best[0] < min_iou):
        try:
            from cad_import.pose_search import cached_search_pose, pose_to_view

            outline_iou, pose = cached_search_pose(
                all_vertices, all_faces, stencil, cache_dir,
                log=lambda message: print(f"[label-cad] {message}", file=sys.stderr, flush=True))
            fit3d, rotated, _ = pose_to_view(pose, all_vertices, outline_iou)
            try_fit("pose_search", fit3d, rotated, all_faces)
        except Exception as exc:
            report["candidates"].append({"fit": "pose_search", "error": f"{type(exc).__name__}: {exc}"})
    if best is None:
        report["accepted"] = False
        report["reason"] = "no mesh half could be fitted to the scan"
        return None, report
    overlap, silhouette, fit = best
    size = 2 * int(TOLERANCE_PX) + 1
    near = cv2.dilate(silhouette.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))).astype(bool)
    cut = mask & ~near
    cut_ratio = float(cut.sum()) / max(1, int(mask.sum()))
    report.update(iou=round(overlap, 4), mm_per_px=round(float(fit.mm_per_px), 4),
                  axis=int(fit.axis), sign=int(fit.sign), cut_px=int(cut.sum()), cut_ratio=round(cut_ratio, 5),
                  max_cut_ratio=float(max_cut_ratio))
    report["accepted"] = bool(overlap >= min_iou and cut_ratio <= max_cut_ratio)
    if not report["accepted"]:
        report["reason"] = (f"outline overlap {overlap:.3f} below {min_iou:.2f}" if overlap < min_iou
                            else f"would cut {cut.sum()} product pixels ({cut_ratio:.2%}) away")
        return None, report
    return silhouette, report


def silhouette_for_scan_from_file(
    cad_path: str | Path,
    part_mask: np.ndarray,
    *,
    cache_dir: Path | None = None,
    min_iou: float = MIN_SILHOUETTE_IOU,
) -> tuple[np.ndarray | None, dict]:
    """silhouette_for_scan for a CAD file (STEP/STL/CATPart - whatever mesh_io can open)."""
    from cad_import.mesh_io import load_any

    path = Path(cad_path)
    try:
        mesh = load_any(path, cache_dir=cache_dir) if cache_dir is not None else load_any(path)
    except Exception as exc:
        return None, {"accepted": False, "reason": f"CAD 파일을 열지 못했습니다({path.name}): {exc}"}
    silhouette, report = silhouette_for_scan(mesh.vertices, mesh.faces, part_mask,
                                             min_iou=min_iou, cache_dir=cache_dir)
    report["cad_file"] = path.name
    return silhouette, report


__all__ = [
    "MAX_CUT_RATIO",
    "MIN_SILHOUETTE_IOU",
    "silhouette_for_scan",
    "silhouette_for_scan_from_file",
    "silhouette_from_pixels",
]
