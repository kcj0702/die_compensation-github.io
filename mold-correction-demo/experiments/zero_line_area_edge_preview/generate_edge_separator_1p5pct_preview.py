"""Select 1.5% zero areas and edge-following correction separators.

The accepted zero *area* is a connected neutral region (-0.5..+0.5 mm) that
touches both merged correction signs and covers at least 1.5% of the part.
Inside each accepted area, a signed-distance bisector separates positive and
negative corrections.  The bisector is snapped to nearby structural part
edges; unsupported spans retain the raw bisector as a fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from generate_preview import (
    DEFAULT_INPUT,
    SPECS,
    ScanSpec,
    add_title,
    blend_mask,
    detect_part_mask,
    extract_color_ramp,
    find_one,
    fit_panel,
    imread_rgb,
    imwrite_gray,
    imwrite_rgb,
    map_deviation,
)
from generate_between_signs_preview import (
    GRAY_SENTINEL_RGB,
    NEG_RGB,
    POS_RGB,
    ZERO_CANDIDATE_RGB,
    build_correction_masks,
    detect_unmapped_gray,
    draw_line,
    mask_boundary,
)
from generate_merged_4pct_preview import (
    fill_holes_without_final_zero,
    group_nearby_neutral_regions,
    merge_nearby_correction,
    resolve_sign_overlap,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results_edge_separator_1p5pct"
FINAL_ZERO_RGB = (255, 235, 0)
RAW_SEPARATOR_RGB = (255, 255, 255)
EDGE_SEPARATOR_RGB = (255, 215, 0)
FALLBACK_RGB = (255, 80, 210)
STRUCTURE_EDGE_RGB = (80, 80, 80)
POS_FILL_RGB = (255, 155, 65)
NEG_FILL_RGB = (75, 205, 255)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)
    skel = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary.copy()
    for _ in range(1000):
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(current, opened))
        current = cv2.erode(current, kernel)
        if cv2.countNonZero(current) == 0:
            break
    return skel


def detect_structural_edges(image_rgb: np.ndarray, part: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 55, 145, L2gradient=True)
    part_boundary = cv2.morphologyEx(
        part.astype(np.uint8) * 255,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    edges = cv2.bitwise_or(edges, part_boundary)
    edges[~cv2.dilate(part.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 0
    return edges


def signed_distance_bisector(
    positive: np.ndarray,
    negative: np.ndarray,
    zero_area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist_positive = cv2.distanceTransform((~positive).astype(np.uint8), cv2.DIST_L2, 5)
    dist_negative = cv2.distanceTransform((~negative).astype(np.uint8), cv2.DIST_L2, 5)
    nearer_positive = dist_positive < dist_negative
    nearer_negative = dist_negative < dist_positive
    k3 = np.ones((3, 3), np.uint8)
    boundary = (
        (cv2.dilate(nearer_positive.astype(np.uint8), k3) > 0)
        & (cv2.dilate(nearer_negative.astype(np.uint8), k3) > 0)
        & zero_area
    )
    # If quantisation produces a gap, retain the locally balanced corridor.
    balance = np.abs(dist_positive - dist_negative)
    corridor = zero_area & (balance <= 3.0)
    boundary |= skeletonize(corridor.astype(np.uint8)) > 0
    return skeletonize(boundary.astype(np.uint8)), dist_positive, dist_negative


def snap_separator_to_edges(
    raw_separator: np.ndarray,
    structural_edges: np.ndarray,
    zero_area: np.ndarray,
    snap_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not np.any(raw_separator):
        empty = np.zeros_like(raw_separator, np.uint8)
        return empty, empty, empty, 0.0

    distance, nearest_labels = cv2.distanceTransformWithLabels(
        (structural_edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    edge_y, edge_x = np.where(structural_edges > 0)
    max_label = int(nearest_labels.max())
    nearest_x = np.full(max_label + 1, -1, np.int32)
    nearest_y = np.full(max_label + 1, -1, np.int32)
    edge_labels = nearest_labels[edge_y, edge_x]
    nearest_x[edge_labels] = edge_x
    nearest_y[edge_labels] = edge_y

    raw_y, raw_x = np.where(raw_separator > 0)
    labels = nearest_labels[raw_y, raw_x]
    supported = (
        (distance[raw_y, raw_x] <= snap_radius)
        & (labels > 0)
        & (nearest_x[labels] >= 0)
    )
    projected = np.zeros_like(raw_separator, np.uint8)
    supported_labels = labels[supported]
    projected[nearest_y[supported_labels], nearest_x[supported_labels]] = 255

    # Keep the connected structural edge close to projected separator points.
    projected_zone = cv2.dilate(projected, np.ones((5, 5), np.uint8))
    edge_following = ((structural_edges > 0) & (projected_zone > 0) & zero_area).astype(np.uint8) * 255
    edge_following = skeletonize(cv2.morphologyEx(
        edge_following,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ))

    unsupported = np.zeros_like(raw_separator, np.uint8)
    unsupported[raw_y[~supported], raw_x[~supported]] = 255
    fallback = skeletonize(cv2.dilate(unsupported, np.ones((3, 3), np.uint8)))
    fallback[~zero_area] = 0

    combined = cv2.bitwise_or(edge_following, fallback)
    combined = skeletonize(cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ))
    combined[~zero_area] = 0
    support_ratio = float(supported.mean()) if len(supported) else 0.0
    return combined, edge_following, fallback, support_ratio


def label_zero_regions(image: np.ndarray, rows: list[dict]) -> None:
    for row in rows:
        if not row["accepted"]:
            continue
        cx, cy = (int(round(v)) for v in row["centroid"])
        text = f"Z{row['accepted_id']} {row['ratio_of_part'] * 100:.2f}%"
        cv2.putText(image, text, (cx - 48, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (cx - 48, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def build_board(
    original: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    positive_fill: np.ndarray,
    negative_fill: np.ndarray,
    zero_area: np.ndarray,
    raw_separator: np.ndarray,
    structural_edges: np.ndarray,
    final_separator: np.ndarray,
    edge_following: np.ndarray,
    fallback: np.ndarray,
    zero_rows: list[dict],
    subtitle: str,
    support_ratio: float,
) -> np.ndarray:
    correction = original.copy()
    blend_mask(correction, positive, POS_RGB, 0.48)
    blend_mask(correction, negative, NEG_RGB, 0.48)
    blend_mask(correction, positive_fill, POS_FILL_RGB, 0.88)
    blend_mask(correction, negative_fill, NEG_FILL_RGB, 0.88)

    area_view = original.copy()
    blend_mask(area_view, zero_area, ZERO_CANDIDATE_RGB, 0.60)
    draw_line(area_view, mask_boundary(zero_area, 1), FINAL_ZERO_RGB, 3)
    label_zero_regions(area_view, zero_rows)

    bisector_view = original.copy()
    dim_edges = cv2.dilate((structural_edges > 0).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    bisector_view[dim_edges] = (
        bisector_view[dim_edges].astype(np.float32) * 0.45
        + np.asarray(STRUCTURE_EDGE_RGB, np.float32) * 0.55
    ).astype(np.uint8)
    blend_mask(bisector_view, zero_area, ZERO_CANDIDATE_RGB, 0.15)
    draw_line(bisector_view, raw_separator, RAW_SEPARATOR_RGB, 3)

    final_view = original.copy()
    blend_mask(final_view, zero_area, ZERO_CANDIDATE_RGB, 0.22)
    draw_line(final_view, raw_separator, RAW_SEPARATOR_RGB, 2)
    draw_line(final_view, edge_following, EDGE_SEPARATOR_RGB, 4)
    draw_line(final_view, fallback, FALLBACK_RGB, 3)
    label_zero_regions(final_view, zero_rows)

    panels = [
        add_title(fit_panel(correction), "1. Merged correction areas", subtitle),
        add_title(
            fit_panel(area_view),
            "2. Final zero areas >= 1.5%",
            "green: neutral connected area touching both correction signs",
        ),
        add_title(
            fit_panel(bisector_view),
            "3. Correction-separating bisector",
            "white: equal-distance separator / dark gray: detected part edges",
        ),
        add_title(
            fit_panel(final_view),
            "4. Edge-following final separator",
            f"yellow: snapped to part edge / magenta: fallback / edge support {support_ratio * 100:.1f}%",
        ),
    ]
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))


def process_one(
    spec: ScanSpec,
    input_dir: Path,
    output_dir: Path,
    threshold_mm: float,
    gray_sentinel_mm: float,
    min_zero_ratio: float,
    correction_merge_gap: int,
    sign_adjacency: int,
    edge_snap_radius: int,
    separator_width: int,
) -> dict:
    image_path = find_one(input_dir, f"{spec.key}*_2_labels_inpainted.png")
    legend_path = find_one(input_dir / "colormap", f"{spec.key}*.png")
    image = imread_rgb(image_path)
    legend = imread_rgb(legend_path)
    values, color_valid = map_deviation(
        image, extract_color_ramp(legend), spec.vmin, spec.vmax
    )
    values = cv2.medianBlur(values, 5)
    part = detect_part_mask(image)
    mapped = part & color_valid
    gray = detect_unmapped_gray(image, part, mapped)
    effective_values = values.copy()
    effective_values[gray] = gray_sentinel_mm
    raw_positive, raw_negative, neutral = build_correction_masks(
        effective_values, mapped, gray, threshold_mm
    )
    merged_positive = merge_nearby_correction(raw_positive, correction_merge_gap, part)
    merged_negative = merge_nearby_correction(raw_negative, correction_merge_gap, part)

    part_px = int(part.sum())
    _, final_zero_area, final_labels, zero_rows = group_nearby_neutral_regions(
        neutral,
        merged_positive,
        merged_negative,
        part_px,
        min_zero_ratio,
        0,
        sign_adjacency,
    )
    raw_separator, dist_positive, dist_negative = signed_distance_bisector(
        merged_positive, merged_negative, final_zero_area
    )
    structural_edges = detect_structural_edges(image, part)
    final_separator, edge_following, fallback, support_ratio = snap_separator_to_edges(
        raw_separator, structural_edges, final_zero_area, edge_snap_radius
    )
    separator_area = cv2.dilate(
        (final_separator > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (separator_width, separator_width)),
    ).astype(bool) & final_zero_area

    display_positive, positive_fill, positive_components = fill_holes_without_final_zero(
        merged_positive, final_zero_area, part
    )
    display_negative, negative_fill, negative_components = fill_holes_without_final_zero(
        merged_negative, final_zero_area, part
    )
    display_positive, display_negative = resolve_sign_overlap(
        display_positive, display_negative, raw_positive, raw_negative
    )
    positive_fill &= display_positive
    negative_fill &= display_negative

    subtitle = (
        f"gray +{gray_sentinel_mm:.2f} / correction gap {correction_merge_gap}px / "
        f"zero cutoff {min_zero_ratio * 100:g}%"
    )
    board = build_board(
        image,
        display_positive,
        display_negative,
        positive_fill,
        negative_fill,
        final_zero_area,
        raw_separator,
        structural_edges,
        final_separator,
        edge_following,
        fallback,
        zero_rows,
        subtitle,
        support_ratio,
    )
    overlay = image.copy()
    blend_mask(overlay, display_positive, POS_RGB, 0.12)
    blend_mask(overlay, display_negative, NEG_RGB, 0.12)
    blend_mask(overlay, final_zero_area, ZERO_CANDIDATE_RGB, 0.22)
    draw_line(overlay, edge_following, EDGE_SEPARATOR_RGB, 4)
    draw_line(overlay, fallback, FALLBACK_RGB, 3)
    label_zero_regions(overlay, zero_rows)

    item_dir = output_dir / spec.key
    imwrite_rgb(item_dir / "review_board.png", board)
    imwrite_rgb(item_dir / "final_edge_separator_overlay.png", overlay)
    imwrite_gray(item_dir / "part_mask.png", part.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "gray_assigned_gt3_mask.png", gray.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "merged_positive_correction_mask.png", display_positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "merged_negative_correction_mask.png", display_negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "final_zero_line_area_mask.png", final_zero_area.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "final_zero_line_group_labels.png", final_labels.astype(np.uint16))
    imwrite_gray(item_dir / "raw_distance_bisector_mask.png", raw_separator)
    imwrite_gray(item_dir / "detected_structural_edges.png", structural_edges)
    imwrite_gray(item_dir / "edge_following_separator_mask.png", edge_following)
    imwrite_gray(item_dir / "separator_fallback_mask.png", fallback)
    imwrite_gray(item_dir / "final_separator_line_mask.png", final_separator)
    imwrite_gray(item_dir / "final_separator_area_mask.png", separator_area.astype(np.uint8) * 255)

    accepted_rows = [row for row in zero_rows if row["accepted"]]
    row_by_id = {row["accepted_id"]: row for row in accepted_rows}
    region_rows = []
    for region_id in sorted(int(v) for v in np.unique(final_labels) if v > 0):
        region = final_labels == region_id
        line = region & (final_separator > 0)
        edge_line = region & (edge_following > 0)
        source = row_by_id.get(region_id, {})
        region_rows.append(
            {
                "region_id": region_id,
                "area_px": int(region.sum()),
                "ratio_of_part": float(region.sum() / part_px),
                "separator_line_px": int(line.sum()),
                "edge_following_line_px": int(edge_line.sum()),
                "edge_following_ratio": float(edge_line.sum() / line.sum()) if line.any() else 0.0,
                "touches_positive": source.get("touches_positive", True),
                "touches_negative": source.get("touches_negative", True),
                "mean_distance_balance_px": float(np.abs(dist_positive[region] - dist_negative[region]).mean()),
            }
        )

    summary = {
        "source_image": str(image_path),
        "legend_image": str(legend_path),
        "gray_sentinel_mm": gray_sentinel_mm,
        "correction_threshold_mm": threshold_mm,
        "correction_merge_gap_px": correction_merge_gap,
        "minimum_final_zero_ratio": min_zero_ratio,
        "sign_adjacency_px": sign_adjacency,
        "edge_snap_radius_px": edge_snap_radius,
        "separator_area_width_px": separator_width,
        "part_px": part_px,
        "final_zero_region_count": len(accepted_rows),
        "final_zero_area_px": int(final_zero_area.sum()),
        "final_zero_area_ratio_of_part": float(final_zero_area.sum() / part_px) if part_px else 0.0,
        "raw_separator_line_px": int((raw_separator > 0).sum()),
        "final_separator_line_px": int((final_separator > 0).sum()),
        "edge_following_line_px": int((edge_following > 0).sum()),
        "fallback_line_px": int((fallback > 0).sum()),
        "edge_support_ratio_of_raw_separator": support_ratio,
        "separator_area_px": int(separator_area.sum()),
        "regions": region_rows,
        "zero_groups": zero_rows,
        "positive_correction_components": positive_components,
        "negative_correction_components": negative_components,
    }
    (item_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_summary(output_dir: Path, summaries: list[dict]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "image",
        "final_zero_region_count",
        "final_zero_area_ratio_of_part",
        "raw_separator_line_px",
        "final_separator_line_px",
        "edge_support_ratio_of_raw_separator",
        "separator_area_px",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "image": Path(row["source_image"]).name,
                **{field: row[field] for field in fields[1:]},
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-mm", type=float, default=0.5)
    parser.add_argument("--gray-sentinel-mm", type=float, default=3.01)
    parser.add_argument("--min-zero-ratio", type=float, default=0.015)
    parser.add_argument("--correction-merge-gap", type=int, default=24)
    parser.add_argument("--sign-adjacency", type=int, default=5)
    parser.add_argument("--edge-snap-radius", type=int, default=12)
    parser.add_argument("--separator-width", type=int, default=5)
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
            args.gray_sentinel_mm,
            args.min_zero_ratio,
            args.correction_merge_gap,
            args.sign_adjacency,
            args.edge_snap_radius,
            args.separator_width,
        )
        for spec in SPECS
    ]
    write_summary(args.output_dir, summaries)
    for row in summaries:
        print(
            f"{Path(row['source_image']).name}: areas={row['final_zero_region_count']}, "
            f"area={row['final_zero_area_ratio_of_part'] * 100:.2f}%, "
            f"edge-support={row['edge_support_ratio_of_raw_separator'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
