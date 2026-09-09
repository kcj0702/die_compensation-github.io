"""Area-first / edge-following zero-line preview generator.

This is an isolated experiment.  It does not import or modify the production
zero-line engine.  It reads the label-inpainted scan images and their separate
color-map legends, then writes review-only masks and overlays.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEMO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = DEMO_ROOT / "label_removal" / "output" / "2_labels_inpainted"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ScanSpec:
    key: str
    vmin: float
    vmax: float


SPECS = (
    ScanSpec("JD_64XX2-DR000", -1.5, 2.0),
    ScanSpec("JD_67XX6-DR000", -3.0, 3.0),
    ScanSpec("JD_71XX2-DR000", -2.0, 2.0),
)

POS_RGB = (255, 70, 45)
NEG_RGB = (45, 120, 255)
ZERO_RGB = (255, 235, 0)
BOUNDARY_RGB = (255, 255, 255)


def imread_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix or ".png", bgr)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def imwrite_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern!r} in {folder}, got {len(matches)}")
    return matches[0]


def _unwrap_hue(hue: np.ndarray) -> np.ndarray:
    out = hue.astype(np.float32).copy()
    out[out > 170] -= 180.0
    return out


def extract_color_ramp(legend_rgb: np.ndarray) -> np.ndarray:
    """Return the legend ramp ordered from maximum (top) to minimum (bottom)."""
    hsv = cv2.cvtColor(legend_rgb, cv2.COLOR_RGB2HSV)
    colorful = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 50)
    h, _ = colorful.shape
    good_columns = np.where(colorful.mean(axis=0) > 0.72)[0]
    if len(good_columns) == 0:
        raise RuntimeError("Could not locate the vertical color ramp in the legend image")
    runs = np.split(good_columns, np.where(np.diff(good_columns) > 1)[0] + 1)
    run = max(runs, key=len)
    x0, x1 = int(run[0]), int(run[-1] + 1)
    rows = np.where(colorful[:, x0:x1].mean(axis=1) > 0.5)[0]
    if len(rows) < h * 0.65:
        raise RuntimeError("Detected legend ramp is unexpectedly short")
    y0, y1 = int(rows.min()), int(rows.max() + 1)
    # Ignore the black border and the black zero tick that may split the ramp.
    inner0, inner1 = x0 + 2, x1 - 2
    if inner1 <= inner0:
        inner0, inner1 = x0, x1
    ramp = np.median(legend_rgb[y0:y1, inner0:inner1], axis=1).astype(np.uint8)
    invalid_rows = cv2.cvtColor(ramp.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)[:, 1] < 50
    if invalid_rows.any():
        valid_idx = np.where(~invalid_rows)[0]
        for channel in range(3):
            ramp[:, channel] = np.interp(
                np.arange(len(ramp)), valid_idx, ramp[valid_idx, channel]
            ).astype(np.uint8)
    ramp = cv2.GaussianBlur(ramp.reshape(-1, 1, 3), (1, 7), 0).reshape(-1, 3)
    return ramp


def map_deviation(
    image_rgb: np.ndarray,
    ramp_top_to_bottom: np.ndarray,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hue-map an RGB scan using the measured legend rather than a fixed palette."""
    ramp_hsv = cv2.cvtColor(
        ramp_top_to_bottom.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)
    ramp_hue = _unwrap_hue(ramp_hsv[:, 0])
    # The supplied legends run red -> ... -> magenta from top to bottom.
    ramp_hue = np.maximum.accumulate(ramp_hue.astype(np.float64))
    ramp_values = np.linspace(vmax, vmin, len(ramp_hue), dtype=np.float32)
    keep = np.concatenate(([True], np.diff(ramp_hue) > 1e-5))
    xp = ramp_hue[keep].astype(np.float32)
    fp = ramp_values[keep]

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    pixel_hue = _unwrap_hue(hsv[:, :, 0])
    values = np.interp(pixel_hue, xp, fp).astype(np.float32)
    valid = (
        (hsv[:, :, 1] >= 45)
        & (hsv[:, :, 2] >= 35)
        & (pixel_hue >= float(xp.min()) - 3.0)
        & (pixel_hue <= float(xp.max()) + 3.0)
    )
    return values, valid


def detect_part_mask(image_rgb: np.ndarray, white_threshold: int = 245) -> np.ndarray:
    """Find the largest non-white material component without filling true holes."""
    non_white = (~np.all(image_rgb > white_threshold, axis=2)).astype(np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(non_white, k5, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if n <= 1:
        return np.zeros_like(non_white, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    core = (labels == largest).astype(np.uint8)
    restored = cv2.dilate(core, k5, iterations=1) & non_white
    restored = cv2.morphologyEx(restored, cv2.MORPH_CLOSE, k5, iterations=2)
    restored = cv2.morphologyEx(restored, cv2.MORPH_OPEN, k5, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(restored, connectivity=8)
    if n <= 1:
        return np.zeros_like(restored, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def clean_threshold_mask(mask: np.ndarray) -> np.ndarray:
    original = mask.astype(bool)
    binary = mask.astype(np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    # Morphology is used only to suppress salt-and-pepper noise.  Never admit
    # a pixel that failed the user's strict +/-0.5 mm threshold.
    return binary.astype(bool) & original


def select_large_regions(
    positive: np.ndarray,
    negative: np.ndarray,
    values: np.ndarray,
    part_px: int,
    min_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    accepted_pos = np.zeros_like(positive, dtype=bool)
    accepted_neg = np.zeros_like(negative, dtype=bool)
    regions: list[dict] = []
    next_id = 1
    for sign, source, target in (
        ("positive", positive, accepted_pos),
        ("negative", negative, accepted_neg),
    ):
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            source.astype(np.uint8), connectivity=8
        )
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            ratio = area / part_px if part_px else 0.0
            component = labels == i
            accepted = ratio >= min_ratio
            if accepted:
                target[component] = True
            x, y, w, h, _ = (int(v) for v in stats[i])
            component_values = values[component]
            regions.append(
                {
                    "region_id": next_id,
                    "sign": sign,
                    "accepted": accepted,
                    "area_px": area,
                    "ratio_of_part": ratio,
                    "bbox": [x, y, w, h],
                    "centroid": [float(centroids[i][0]), float(centroids[i][1])],
                    "mean_mm": float(component_values.mean()),
                    "min_mm": float(component_values.min()),
                    "max_mm": float(component_values.max()),
                }
            )
            next_id += 1
    regions.sort(key=lambda row: row["area_px"], reverse=True)
    accepted_index = 0
    for row in regions:
        if row["accepted"]:
            accepted_index += 1
            row["accepted_id"] = accepted_index
        else:
            row["accepted_id"] = None
    return accepted_pos, accepted_neg, regions


def correction_boundary(
    mask: np.ndarray,
    domain: np.ndarray | None = None,
    thickness: int = 1,
) -> np.ndarray:
    """Return only correction/non-correction interfaces, not part outer edges.

    ``domain`` is the color-mapped scan surface.  Requiring non-correction
    domain pixels next to a contour removes boundaries against white holes,
    unmapped gray patches, and the outside of the part.
    """
    result = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    if contours:
        cv2.drawContours(result, contours, -1, 255, thickness, cv2.LINE_AA)
    if domain is not None:
        non_correction = domain & ~mask
        near_non_correction = cv2.dilate(
            non_correction.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        result[near_non_correction == 0] = 0
        result[~domain] = 0
    return result


def detect_edge_following_line(
    image_rgb: np.ndarray,
    raw_boundary: np.ndarray,
    part_mask: np.ndarray,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep structural edges lying close to the accepted correction boundary."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    all_edges = cv2.Canny(gray, 55, 145, L2gradient=True)
    all_edges[~part_mask] = 0

    if not np.any(raw_boundary):
        return np.zeros_like(raw_boundary), all_edges, 0.0
    distance, nearest_labels = cv2.distanceTransformWithLabels(
        (all_edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    edge_y, edge_x = np.where(all_edges > 0)
    max_label = int(nearest_labels.max())
    nearest_x = np.full(max_label + 1, -1, np.int32)
    nearest_y = np.full(max_label + 1, -1, np.int32)
    edge_labels = nearest_labels[edge_y, edge_x]
    nearest_x[edge_labels] = edge_x
    nearest_y[edge_labels] = edge_y

    boundary_y, boundary_x = np.where(raw_boundary > 0)
    boundary_labels = nearest_labels[boundary_y, boundary_x]
    supported_boundary = (
        (distance[boundary_y, boundary_x] <= search_radius)
        & (boundary_labels > 0)
        & (nearest_x[boundary_labels] >= 0)
    )
    projected = np.zeros_like(all_edges, np.uint8)
    supported_labels = boundary_labels[supported_boundary]
    projected[nearest_y[supported_labels], nearest_x[supported_labels]] = 1

    # Grow only from the nearest edge assigned to each boundary pixel.  This
    # avoids returning every parallel Canny edge inside the search band.
    bridge = cv2.dilate(projected, np.ones((3, 3), np.uint8))
    bridge = cv2.morphologyEx(bridge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridge, connectivity=8)
    supported = np.zeros_like(bridge)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= 18 or max(w, h) >= 12:
            supported[labels == i] = 1
    line = (all_edges > 0) & (cv2.dilate(supported, np.ones((3, 3), np.uint8)) > 0)
    boundary_pixels = raw_boundary > 0
    support_ratio = float((distance[boundary_pixels] <= search_radius).mean())
    return line.astype(np.uint8) * 255, all_edges, support_ratio


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color_arr = np.asarray(color, dtype=np.float32)
    pixels = image[mask].astype(np.float32)
    image[mask] = np.clip(pixels * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)


def draw_mask_line(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    binary = (mask > 0).astype(np.uint8)
    if width > 1:
        binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width)))
    image[binary > 0] = color


def label_accepted_regions(image: np.ndarray, regions: list[dict]) -> None:
    for row in regions:
        if not row["accepted"]:
            continue
        cx, cy = (int(round(v)) for v in row["centroid"])
        sign = "+" if row["sign"] == "positive" else "-"
        text = f"A{row['accepted_id']} {sign} {row['ratio_of_part'] * 100:.1f}%"
        cv2.putText(image, text, (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def add_title(panel: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    bar_h = 58
    out = np.full((panel.shape[0] + bar_h, panel.shape[1], 3), 248, np.uint8)
    out[bar_h:] = panel
    cv2.putText(out, title, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (65, 65, 65), 1, cv2.LINE_AA)
    return out


def fit_panel(image: np.ndarray, width: int = 900, height: int = 590) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 255, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def build_review_board(
    original: np.ndarray,
    raw_pos: np.ndarray,
    raw_neg: np.ndarray,
    accepted_pos: np.ndarray,
    accepted_neg: np.ndarray,
    boundary: np.ndarray,
    zero_line: np.ndarray,
    regions: list[dict],
    subtitle: str,
) -> np.ndarray:
    threshold = original.copy()
    blend_mask(threshold, raw_pos, POS_RGB, 0.62)
    blend_mask(threshold, raw_neg, NEG_RGB, 0.62)

    accepted = original.copy()
    blend_mask(accepted, accepted_pos, POS_RGB, 0.55)
    blend_mask(accepted, accepted_neg, NEG_RGB, 0.55)
    label_accepted_regions(accepted, regions)

    zero = original.copy()
    blend_mask(zero, accepted_pos, POS_RGB, 0.18)
    blend_mask(zero, accepted_neg, NEG_RGB, 0.18)
    draw_mask_line(zero, boundary, BOUNDARY_RGB, 2)
    draw_mask_line(zero, zero_line, ZERO_RGB, 4)

    panels = [
        add_title(fit_panel(original), "1. Original scan", subtitle),
        add_title(fit_panel(threshold), "2. Raw |deviation| > 0.5 mm", "red: positive / blue: negative"),
        add_title(fit_panel(accepted), "3. Accepted correction regions", "each connected region >= 5% of total part"),
        add_title(fit_panel(zero), "4. Edge-following zero-line candidate", "yellow: detected image edge / white: raw region boundary"),
    ]
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    return np.vstack((top, bottom))


def process_one(
    spec: ScanSpec,
    input_dir: Path,
    output_dir: Path,
    threshold_mm: float,
    min_area_ratio: float,
    edge_search_radius: int,
) -> dict:
    image_path = find_one(input_dir, f"{spec.key}*_2_labels_inpainted.png")
    legend_path = find_one(input_dir / "colormap", f"{spec.key}*.png")
    image = imread_rgb(image_path)
    legend = imread_rgb(legend_path)
    ramp = extract_color_ramp(legend)
    values, color_valid = map_deviation(image, ramp, spec.vmin, spec.vmax)
    values = cv2.medianBlur(values, 5)
    part = detect_part_mask(image)
    mapped = part & color_valid
    part_px = int(part.sum())
    mapped_px = int(mapped.sum())

    raw_pos = clean_threshold_mask(mapped & (values > threshold_mm)) & mapped
    raw_neg = clean_threshold_mask(mapped & (values < -threshold_mm)) & mapped
    accepted_pos, accepted_neg, regions = select_large_regions(
        raw_pos, raw_neg, values, part_px, min_area_ratio
    )
    correction = accepted_pos | accepted_neg
    boundary = correction_boundary(correction, domain=mapped, thickness=1)
    zero_line, all_edges, boundary_support_ratio = detect_edge_following_line(
        image, boundary, part, edge_search_radius
    )

    stem = spec.key
    item_dir = output_dir / stem
    threshold_rgb = np.zeros((*part.shape, 3), np.uint8)
    threshold_rgb[part] = (55, 55, 55)
    threshold_rgb[raw_pos] = POS_RGB
    threshold_rgb[raw_neg] = NEG_RGB
    correction_rgb = np.zeros((*part.shape, 3), np.uint8)
    correction_rgb[accepted_pos] = POS_RGB
    correction_rgb[accepted_neg] = NEG_RGB
    overlay = image.copy()
    blend_mask(overlay, accepted_pos, POS_RGB, 0.22)
    blend_mask(overlay, accepted_neg, NEG_RGB, 0.22)
    draw_mask_line(overlay, boundary, BOUNDARY_RGB, 2)
    draw_mask_line(overlay, zero_line, ZERO_RGB, 4)
    label_accepted_regions(overlay, regions)

    subtitle = (
        f"range {spec.vmin:+.1f}..{spec.vmax:+.1f} mm / "
        f"mapped {mapped_px / part_px * 100:.1f}%" if part_px else "no part detected"
    )
    board = build_review_board(
        image, raw_pos, raw_neg, accepted_pos, accepted_neg,
        boundary, zero_line, regions, subtitle,
    )

    imwrite_rgb(item_dir / "review_board.png", board)
    imwrite_rgb(item_dir / "zero_line_overlay.png", overlay)
    imwrite_rgb(item_dir / "threshold_regions.png", threshold_rgb)
    imwrite_rgb(item_dir / "accepted_correction_regions.png", correction_rgb)
    imwrite_gray(item_dir / "part_mask.png", part.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "mapped_color_mask.png", mapped.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "raw_boundary_mask.png", boundary)
    imwrite_gray(item_dir / "edge_following_zero_line_mask.png", zero_line)
    imwrite_gray(item_dir / "all_detected_edges_reference.png", all_edges)

    accepted_regions = [row for row in regions if row["accepted"]]
    summary = {
        "source_image": str(image_path),
        "legend_image": str(legend_path),
        "value_range_mm": [spec.vmin, spec.vmax],
        "threshold_mm": threshold_mm,
        "minimum_region_ratio": min_area_ratio,
        "edge_search_radius_px": edge_search_radius,
        "part_px": part_px,
        "mapped_color_px": mapped_px,
        "mapped_color_ratio_of_part": mapped_px / part_px if part_px else 0.0,
        "raw_positive_px": int(raw_pos.sum()),
        "raw_negative_px": int(raw_neg.sum()),
        "accepted_region_count": len(accepted_regions),
        "accepted_correction_px": int(correction.sum()),
        "accepted_correction_ratio_of_part": float(correction.sum() / part_px) if part_px else 0.0,
        "raw_boundary_px": int((boundary > 0).sum()),
        "edge_following_zero_line_px": int((zero_line > 0).sum()),
        "boundary_with_nearby_edge_ratio": boundary_support_ratio,
        "regions": regions,
    }
    (item_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_overall_summary(output_dir: Path, summaries: list[dict]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "image", "part_px", "mapped_color_ratio_of_part",
        "raw_positive_ratio_of_part", "raw_negative_ratio_of_part",
        "accepted_region_count", "accepted_correction_ratio_of_part",
        "boundary_with_nearby_edge_ratio", "edge_following_zero_line_px",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            part_px = row["part_px"]
            writer.writerow(
                {
                    "image": Path(row["source_image"]).name,
                    "part_px": part_px,
                    "mapped_color_ratio_of_part": row["mapped_color_ratio_of_part"],
                    "raw_positive_ratio_of_part": row["raw_positive_px"] / part_px if part_px else 0.0,
                    "raw_negative_ratio_of_part": row["raw_negative_px"] / part_px if part_px else 0.0,
                    "accepted_region_count": row["accepted_region_count"],
                    "accepted_correction_ratio_of_part": row["accepted_correction_ratio_of_part"],
                    "boundary_with_nearby_edge_ratio": row["boundary_with_nearby_edge_ratio"],
                    "edge_following_zero_line_px": row["edge_following_zero_line_px"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-mm", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--edge-search-radius", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        process_one(
            spec,
            args.input_dir,
            args.output_dir,
            args.threshold_mm,
            args.min_area_ratio,
            args.edge_search_radius,
        )
        for spec in SPECS
    ]
    write_overall_summary(args.output_dir, summaries)
    for row in summaries:
        print(
            f"{Path(row['source_image']).name}: "
            f"accepted={row['accepted_region_count']}, "
            f"correction={row['accepted_correction_ratio_of_part'] * 100:.2f}%, "
            f"edge-support={row['boundary_with_nearby_edge_ratio'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
