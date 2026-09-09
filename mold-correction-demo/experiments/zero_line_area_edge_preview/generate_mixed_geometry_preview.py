"""Vectorize final zero-line areas as mixed line/curve polygons.

The final raster masks are not reclassified.  Their boundaries are converted
to closed curvilinear polygons made of:
  * explicit corner anchors;
  * straight line segments (SVG ``L``);
  * cubic Bezier curve segments (SVG ``C``).

Outputs include PNG review boards, a fitted raster mask, transparent SVG, and
JSON geometry suitable for later engine integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from generate_preview import add_title, blend_mask, fit_panel, imread_rgb, imwrite_gray, imwrite_rgb


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results_merged_correction_3pct"
DEFAULT_OUTPUT = HERE / "results_merged_correction_3pct_mixed_geometry"

LINE_RGB = (255, 225, 0)
CURVE_RGB = (0, 235, 255)
CORNER_RGB = (255, 0, 205)
FILL_RGB = (65, 220, 115)


def imread_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 3:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6
    points = points[keep]
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= spacing * 3:
        return points
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = np.arange(0.0, perimeter, spacing)
    x = np.interp(samples, cumulative, closed[:, 0])
    y = np.interp(samples, cumulative, closed[:, 1])
    return np.column_stack((x, y))


def smooth_closed(points: np.ndarray, radius: int = 2, sigma: float = 1.25) -> np.ndarray:
    if len(points) < radius * 2 + 3:
        return points.copy()
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    weights /= weights.sum()
    out = np.zeros_like(points, dtype=np.float64)
    for offset, weight in zip(offsets.astype(int), weights):
        out += np.roll(points, offset, axis=0) * weight
    return out


def detect_corners(
    points: np.ndarray,
    spacing: float,
    min_turn_deg: float,
    min_separation_px: float,
    scale_px: float,
    prominence_deg: float,
) -> list[int]:
    n = len(points)
    if n < 8:
        return []
    scale = max(2, int(round(scale_px / max(spacing, 1e-6))))
    turns = np.zeros(n, np.float64)
    for i in range(n):
        before = points[(i - scale) % n] - points[i]
        after = points[(i + scale) % n] - points[i]
        denom = np.linalg.norm(before) * np.linalg.norm(after)
        if denom <= 1e-9:
            continue
        interior = np.degrees(np.arccos(np.clip(np.dot(before, after) / denom, -1.0, 1.0)))
        turns[i] = 180.0 - interior

    min_sep = max(2, int(round(min_separation_px / max(spacing, 1e-6))))
    peak_radius = max(2, min_sep // 2)
    candidates = []
    for i in np.where(turns >= min_turn_deg)[0]:
        indices = [(i + offset) % n for offset in range(-peak_radius, peak_radius + 1)]
        local = turns[indices]
        if turns[i] + 1e-9 < float(local.max()):
            continue
        if turns[i] - float(np.median(local)) < prominence_deg:
            continue
        candidates.append(int(i))
    candidates_arr = np.asarray(candidates, dtype=np.int32)
    order = candidates_arr[np.argsort(turns[candidates_arr])[::-1]] if len(candidates_arr) else []
    selected: list[int] = []
    for idx in order:
        if all(min((idx - kept) % n, (kept - idx) % n) >= min_sep for kept in selected):
            selected.append(int(idx))
    return sorted(selected)


def cyclic_span(points: np.ndarray, start: int, end: int) -> np.ndarray:
    if end > start:
        return points[start:end + 1]
    return np.vstack((points[start:], points[:end + 1]))


def line_error(points: np.ndarray) -> tuple[float, int]:
    start, end = points[0], points[-1]
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        errors = np.linalg.norm(points - start, axis=1)
    else:
        relative = points - start
        errors = np.abs((direction[0] * relative[:, 1] - direction[1] * relative[:, 0]) / norm)
    idx = int(np.argmax(errors))
    return float(errors[idx]), idx


def fit_cubic(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    p0 = points[0]
    p3 = points[-1]
    chord_lengths = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if chord_lengths[-1] <= 1e-9:
        return p0.copy(), p0.copy(), np.repeat(p0[None, :], len(points), axis=0), 0.0, 0
    t = chord_lengths / chord_lengths[-1]
    omt = 1.0 - t
    a = np.column_stack((3.0 * omt * omt * t, 3.0 * omt * t * t))
    base = (omt ** 3)[:, None] * p0 + (t ** 3)[:, None] * p3
    rhs = points - base
    controls, _, _, _ = np.linalg.lstsq(a, rhs, rcond=None)
    p1, p2 = controls[0], controls[1]
    fitted = (
        (omt ** 3)[:, None] * p0
        + (3.0 * omt * omt * t)[:, None] * p1
        + (3.0 * omt * t * t)[:, None] * p2
        + (t ** 3)[:, None] * p3
    )
    errors = np.linalg.norm(fitted - points, axis=1)
    idx = int(np.argmax(errors))
    return p1, p2, fitted, float(errors[idx]), idx


def fit_span(
    points: np.ndarray,
    straight_error_px: float,
    curve_error_px: float,
    depth: int = 0,
) -> list[dict]:
    if len(points) <= 2:
        return [{"type": "line", "start": points[0], "end": points[-1]}]

    max_line_error, _ = line_error(points)
    if max_line_error <= straight_error_px:
        return [{"type": "line", "start": points[0], "end": points[-1]}]

    p1, p2, _, max_curve_error, split_idx = fit_cubic(points)
    if max_curve_error <= curve_error_px or depth >= 8 or len(points) < 7:
        return [{
            "type": "cubic",
            "start": points[0],
            "control1": p1,
            "control2": p2,
            "end": points[-1],
            "fit_error_px": max_curve_error,
        }]

    split_idx = max(2, min(len(points) - 3, split_idx))
    return (
        fit_span(points[:split_idx + 1], straight_error_px, curve_error_px, depth + 1)
        + fit_span(points[split_idx:], straight_error_px, curve_error_px, depth + 1)
    )


def vectorize_contour(
    contour: np.ndarray,
    spacing: float,
    smooth_radius: int,
    smooth_sigma: float,
    corner_turn_deg: float,
    corner_separation_px: float,
    corner_scale_px: float,
    corner_prominence_deg: float,
    straight_error_px: float,
    curve_error_px: float,
) -> dict:
    sampled = resample_closed(contour.reshape(-1, 2), spacing)
    smooth = smooth_closed(sampled, radius=smooth_radius, sigma=smooth_sigma)
    real_corners = detect_corners(
        smooth,
        spacing,
        corner_turn_deg,
        corner_separation_px,
        corner_scale_px,
        corner_prominence_deg,
    )
    if len(real_corners) >= 2:
        anchors = real_corners
    else:
        # A circle or a long smooth loop has no true corner.  Synthetic quarter
        # anchors only divide the closed fit and are not rendered as corners.
        anchors = sorted(set(int(round(i * len(smooth) / 4)) % len(smooth) for i in range(4)))

    segments: list[dict] = []
    for idx, start in enumerate(anchors):
        end = anchors[(idx + 1) % len(anchors)]
        span = cyclic_span(smooth, start, end)
        segments.extend(fit_span(span, straight_error_px, curve_error_px))

    return {
        "segments": segments,
        "corner_points": [smooth[i] for i in real_corners],
        "sample_count": int(len(sampled)),
    }


def cubic_points(segment: dict, count: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)[:, None]
    omt = 1.0 - t
    p0 = segment["start"]
    p1 = segment["control1"]
    p2 = segment["control2"]
    p3 = segment["end"]
    return omt ** 3 * p0 + 3 * omt ** 2 * t * p1 + 3 * omt * t ** 2 * p2 + t ** 3 * p3


def segment_polyline(segment: dict) -> np.ndarray:
    if segment["type"] == "line":
        return np.vstack((segment["start"], segment["end"]))
    return cubic_points(segment)


def path_points(segments: list[dict]) -> np.ndarray:
    pieces = []
    for i, segment in enumerate(segments):
        pts = segment_polyline(segment)
        pieces.append(pts if i == 0 else pts[1:])
    return np.vstack(pieces)


def draw_geometry(image: np.ndarray, geometry: list[dict], thickness: int = 3) -> None:
    for contour in geometry:
        for segment in contour["segments"]:
            points = np.round(segment_polyline(segment)).astype(np.int32).reshape(-1, 1, 2)
            color = LINE_RGB if segment["type"] == "line" else CURVE_RGB
            cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
        for point in contour["corner_points"]:
            center = tuple(np.round(point).astype(int))
            cv2.circle(image, center, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, center, 3, CORNER_RGB, -1, cv2.LINE_AA)


def to_serializable_contour(row: dict) -> dict:
    def point(value: np.ndarray) -> list[float]:
        return [round(float(value[0]), 3), round(float(value[1]), 3)]

    segments = []
    for segment in row["segments"]:
        out = {
            "type": segment["type"],
            "start": point(segment["start"]),
            "end": point(segment["end"]),
        }
        if segment["type"] == "cubic":
            out["control1"] = point(segment["control1"])
            out["control2"] = point(segment["control2"])
            out["fit_error_px"] = round(float(segment["fit_error_px"]), 3)
        segments.append(out)
    return {
        "region_id": row["region_id"],
        "contour_id": row["contour_id"],
        "is_hole": row["is_hole"],
        "sample_count": row["sample_count"],
        "corner_points": [point(p) for p in row["corner_points"]],
        "segments": segments,
    }


def svg_path_data(contour: dict) -> str:
    segments = contour["segments"]
    if not segments:
        return ""
    start = segments[0]["start"]
    commands = [f"M {start[0]:.2f} {start[1]:.2f}"]
    for segment in segments:
        end = segment["end"]
        if segment["type"] == "line":
            commands.append(f"L {end[0]:.2f} {end[1]:.2f}")
        else:
            c1, c2 = segment["control1"], segment["control2"]
            commands.append(
                f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
            )
    commands.append("Z")
    return " ".join(commands)


def write_svg(path: Path, width: int, height: int, geometry: list[dict]) -> None:
    combined = " ".join(svg_path_data(row) for row in geometry)
    elements = [
        f'<path d="{combined}" fill="#41dc73" fill-opacity="0.28" fill-rule="evenodd" stroke="none"/>'
    ]
    for row in geometry:
        for segment in row["segments"]:
            start, end = segment["start"], segment["end"]
            if segment["type"] == "line":
                elements.append(
                    f'<path d="M {start[0]:.2f} {start[1]:.2f} L {end[0]:.2f} {end[1]:.2f}" '
                    'fill="none" stroke="#ffe100" stroke-width="3"/>'
                )
            else:
                c1, c2 = segment["control1"], segment["control2"]
                elements.append(
                    f'<path d="M {start[0]:.2f} {start[1]:.2f} C {c1[0]:.2f} {c1[1]:.2f} '
                    f'{c2[0]:.2f} {c2[1]:.2f} {end[0]:.2f} {end[1]:.2f}" '
                    'fill="none" stroke="#00ebff" stroke-width="3"/>'
                )
        for point in row["corner_points"]:
            elements.append(
                f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="4" fill="#ff00cd" stroke="#000" stroke-width="1"/>'
            )
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    path.write_text(content, encoding="utf-8")


def process_item(
    input_dir: Path,
    output_dir: Path,
    spacing: float,
    smooth_radius: int,
    smooth_sigma: float,
    corner_turn_deg: float,
    corner_separation_px: float,
    corner_scale_px: float,
    corner_prominence_deg: float,
    straight_error_px: float,
    curve_error_px: float,
) -> dict:
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    cutoff_ratio = float(summary.get("minimum_final_zero_ratio", 0.03))
    original = imread_rgb(Path(summary["source_image"]))
    final_mask = imread_gray(input_dir / "final_zero_line_area_mask.png") > 0
    labels = cv2.imdecode(
        np.fromfile(str(input_dir / "final_zero_line_group_labels.png"), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if labels is None:
        raise RuntimeError(f"Cannot read group labels: {input_dir}")

    geometry: list[dict] = []
    vector_mask = np.zeros(final_mask.shape, np.uint8)
    region_summaries = []
    for region_id in sorted(int(v) for v in np.unique(labels) if v > 0):
        region_mask = labels == region_id
        contours, hierarchy = cv2.findContours(
            region_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = hierarchy[0] if hierarchy is not None else np.empty((0, 4), np.int32)
        region_geometry = []
        for contour_id, contour in enumerate(contours, 1):
            if abs(cv2.contourArea(contour)) < 12:
                continue
            fitted = vectorize_contour(
                contour,
                spacing,
                smooth_radius,
                smooth_sigma,
                corner_turn_deg,
                corner_separation_px,
                corner_scale_px,
                corner_prominence_deg,
                straight_error_px,
                curve_error_px,
            )
            fitted.update(
                {
                    "region_id": region_id,
                    "contour_id": contour_id,
                    "is_hole": bool(hierarchy[contour_id - 1][3] >= 0),
                }
            )
            geometry.append(fitted)
            region_geometry.append(fitted)

            polygon = np.round(path_points(fitted["segments"])).astype(np.int32)
            fill_value = 0 if fitted["is_hole"] else 255
            cv2.fillPoly(vector_mask, [polygon], fill_value, cv2.LINE_AA)

        line_count = sum(
            segment["type"] == "line"
            for row in region_geometry for segment in row["segments"]
        )
        curve_count = sum(
            segment["type"] == "cubic"
            for row in region_geometry for segment in row["segments"]
        )
        corner_count = sum(len(row["corner_points"]) for row in region_geometry)
        region_summaries.append(
            {
                "region_id": region_id,
                "original_area_px": int(region_mask.sum()),
                "line_segment_count": int(line_count),
                "curve_segment_count": int(curve_count),
                "corner_count": int(corner_count),
            }
        )

    vector_bool = vector_mask > 0
    union = int((vector_bool | final_mask).sum())
    intersection = int((vector_bool & final_mask).sum())
    iou = intersection / union if union else 1.0
    area_error = (
        (int(vector_bool.sum()) - int(final_mask.sum())) / int(final_mask.sum())
        if final_mask.any() else 0.0
    )

    mask_overlay = original.copy()
    blend_mask(mask_overlay, final_mask, FILL_RGB, 0.58)
    fitted_overlay = original.copy()
    blend_mask(fitted_overlay, vector_bool, FILL_RGB, 0.34)
    draw_geometry(fitted_overlay, geometry, thickness=3)
    geometry_only = np.full_like(original, 255)
    blend_mask(geometry_only, vector_bool, FILL_RGB, 0.28)
    draw_geometry(geometry_only, geometry, thickness=3)

    board = np.vstack(
        (
            np.hstack(
                (
                    add_title(fit_panel(original), "1. Original scan"),
                    add_title(
                        fit_panel(mask_overlay),
                        "2. Final zero-line raster area",
                        f"green: selected area at {cutoff_ratio * 100:g}% cutoff",
                    ),
                )
            ),
            np.hstack(
                (
                    add_title(
                        fit_panel(fitted_overlay),
                        "3. Mixed-geometry boundary",
                        "yellow: line / cyan: cubic curve / magenta: corner",
                    ),
                    add_title(
                        fit_panel(geometry_only),
                        "4. Curvilinear polygon only",
                        f"vector/raster IoU {iou * 100:.1f}% / area error {area_error * 100:+.2f}%",
                    ),
                )
            ),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    imwrite_rgb(output_dir / "review_board.png", board)
    imwrite_rgb(output_dir / "mixed_geometry_overlay.png", fitted_overlay)
    imwrite_rgb(output_dir / "mixed_geometry_only.png", geometry_only)
    imwrite_gray(output_dir / "vectorized_zero_area_mask.png", vector_mask)
    write_svg(output_dir / "mixed_geometry_boundaries.svg", original.shape[1], original.shape[0], geometry)

    serializable = [to_serializable_contour(row) for row in geometry]
    result = {
        "source_result": str(input_dir),
        "image_width": int(original.shape[1]),
        "image_height": int(original.shape[0]),
        "params": {
            "sample_spacing_px": spacing,
            "smooth_radius_samples": smooth_radius,
            "smooth_sigma_samples": smooth_sigma,
            "corner_turn_degrees": corner_turn_deg,
            "corner_min_separation_px": corner_separation_px,
            "corner_measurement_scale_px": corner_scale_px,
            "corner_min_prominence_degrees": corner_prominence_deg,
            "straight_max_error_px": straight_error_px,
            "curve_max_error_px": curve_error_px,
        },
        "raster_vector_iou": iou,
        "vector_area_error_ratio": area_error,
        "regions": region_summaries,
        "contours": serializable,
    }
    (output_dir / "mixed_geometry_boundaries.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-spacing", type=float, default=5.0)
    parser.add_argument("--smooth-radius", type=int, default=3)
    parser.add_argument("--smooth-sigma", type=float, default=1.8)
    parser.add_argument("--corner-turn-deg", type=float, default=50.0)
    parser.add_argument("--corner-separation", type=float, default=28.0)
    parser.add_argument("--corner-scale", type=float, default=18.0)
    parser.add_argument("--corner-prominence", type=float, default=12.0)
    parser.add_argument("--straight-error", type=float, default=3.0)
    parser.add_argument("--curve-error", type=float, default=4.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for item_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        output_dir = args.output_dir / item_dir.name
        result = process_item(
            item_dir,
            output_dir,
            args.sample_spacing,
            args.smooth_radius,
            args.smooth_sigma,
            args.corner_turn_deg,
            args.corner_separation,
            args.corner_scale,
            args.corner_prominence,
            args.straight_error,
            args.curve_error,
        )
        results.append({"item": item_dir.name, **result})
        lines = sum(r["line_segment_count"] for r in result["regions"])
        curves = sum(r["curve_segment_count"] for r in result["regions"])
        corners = sum(r["corner_count"] for r in result["regions"])
        print(
            f"{item_dir.name}: IoU={result['raster_vector_iou'] * 100:.1f}%, "
            f"lines={lines}, curves={curves}, corners={corners}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
