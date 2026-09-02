"""Generate review-only positive/negative correction areas with a 2% cutoff.

The production engine is not imported or modified.  This experiment:
  * maps scan colors through each supplied color-map legend;
  * treats mapped values > +0.6 mm as positive correction;
  * treats mapped values < -0.6 mm as negative correction;
  * assigns each unmapped gray pixel beyond the color-map range with the sign
    of its nearest mapped surrounding color;
  * merges same-sign candidates, fills holes, and resolves sign overlap;
  * retains merged components with area >= 2% of the total part;
  * removes narrow protrusions with a 17x17 circular-kernel opening, then applies
    the final strict-greater-than-2% component check.
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
    NEG_RGB,
    POS_RGB,
    detect_unmapped_gray,
    draw_line,
    mask_boundary,
    strict_morphology,
)
from generate_merged_4pct_preview import merge_nearby_correction, resolve_sign_overlap


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_correction_only_2pct"
CORRECTION_THRESHOLD_MM = 0.6
OPENING_KERNEL_SIZE_PX = 17
MERGED_POS_RGB = (255, 155, 65)
MERGED_NEG_RGB = (75, 205, 255)
FINAL_POS_RGB = (255, 65, 35)
FINAL_NEG_RGB = (30, 105, 255)
HOLE_RGB = (255, 235, 0)


def assign_gray_by_nearest_mapped_sign(
    values: np.ndarray,
    mapped: np.ndarray,
    gray: np.ndarray,
    positive_out_of_range_mm: float,
    negative_out_of_range_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign gray pixels the sign of their nearest non-zero mapped color."""
    sign_source = mapped & (np.abs(values) > 1e-6)
    if not np.any(sign_source):
        raise RuntimeError("Cannot infer gray sign because no non-zero mapped colors exist")
    _, nearest_labels = cv2.distanceTransformWithLabels(
        (~sign_source).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    label_values = np.zeros(int(nearest_labels.max()) + 1, np.float32)
    source_y, source_x = np.where(sign_source)
    label_values[nearest_labels[source_y, source_x]] = values[source_y, source_x]
    nearest_values = label_values[nearest_labels]
    gray_positive = gray & (nearest_values >= 0.0)
    gray_negative = gray & (nearest_values < 0.0)
    effective_values = values.copy()
    effective_values[gray_positive] = positive_out_of_range_mm
    effective_values[gray_negative] = negative_out_of_range_mm
    return effective_values, gray_positive, gray_negative, nearest_values


def build_signed_correction_masks(
    values: np.ndarray,
    mapped: np.ndarray,
    gray_positive: np.ndarray,
    gray_negative: np.ndarray,
    threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    positive_source = (mapped & (values > threshold_mm)) | gray_positive
    negative_source = (mapped & (values < -threshold_mm)) | gray_negative
    positive = strict_morphology(positive_source, open_size=5, close_size=7)
    negative = strict_morphology(negative_source, open_size=5, close_size=7)
    return positive, negative


def fill_all_internal_holes(
    mask: np.ndarray,
    part: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fill every enclosed hole in each connected correction component."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    filled = mask.copy()
    filled_only = np.zeros_like(mask, dtype=bool)
    rows: list[dict] = []
    for component_id in range(1, n):
        component = labels == component_id
        x, y, w, h, area = (int(v) for v in stats[component_id])
        local = component[y:y + h, x:x + w].astype(np.uint8)
        padded = cv2.copyMakeBorder(local, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood = padded.copy()
        flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 1)
        holes_local = (flood[1:-1, 1:-1] == 0) & (local == 0)
        holes = np.zeros_like(component)
        holes[y:y + h, x:x + w] = holes_local
        holes &= part
        filled[holes] = True
        filled_only[holes] = True
        rows.append(
            {
                "source_component_id": component_id,
                "area_px_before_hole_fill": area,
                "hole_px_filled": int(holes.sum()),
                "area_px_after_hole_fill": area + int(holes.sum()),
            }
        )
    return filled, filled_only, rows


def select_final_regions(
    positive: np.ndarray,
    negative: np.ndarray,
    part_px: int,
    min_ratio: float,
    inclusive: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Select components using either >= or > area threshold semantics."""
    candidates: list[dict] = []
    sign_data = []
    for sign, mask in (("positive", positive), ("negative", negative)):
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        sign_data.append((sign, labels))
        for component_id in range(1, n):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            x, y, w, h, _ = (int(v) for v in stats[component_id])
            ratio = area / part_px if part_px else 0.0
            accepted = ratio >= min_ratio if inclusive else ratio > min_ratio
            candidates.append(
                {
                    "sign": sign,
                    "source_component_id": component_id,
                    "area_px": area,
                    "ratio_of_part": ratio,
                    "accepted": bool(accepted),
                    "centroid": [
                        float(centroids[component_id][0]),
                        float(centroids[component_id][1]),
                    ],
                    "bbox": [x, y, w, h],
                }
            )

    candidates.sort(key=lambda row: row["area_px"], reverse=True)
    accepted_positive = np.zeros_like(positive, dtype=bool)
    accepted_negative = np.zeros_like(negative, dtype=bool)
    accepted_labels = np.zeros(positive.shape, np.uint16)
    labels_by_sign = {sign: labels for sign, labels in sign_data}
    accepted_id = 0
    for row in candidates:
        if not row["accepted"]:
            row["accepted_id"] = None
            continue
        accepted_id += 1
        row["accepted_id"] = accepted_id
        component = labels_by_sign[row["sign"]] == row["source_component_id"]
        if row["sign"] == "positive":
            accepted_positive[component] = True
        else:
            accepted_negative[component] = True
        accepted_labels[component] = accepted_id
    return accepted_positive, accepted_negative, accepted_labels, candidates


def open_narrow_parts(mask: np.ndarray, kernel_size_px: int, part: np.ndarray) -> np.ndarray:
    """Remove narrow protrusions with a circular morphological opening."""
    if kernel_size_px <= 1:
        return mask & part
    size = kernel_size_px if kernel_size_px % 2 else kernel_size_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return (opened > 0) & part


def draw_region_labels(image: np.ndarray, rows: list[dict]) -> None:
    for row in rows:
        if not row["accepted"]:
            continue
        cx, cy = (int(round(v)) for v in row["centroid"])
        sign = "+" if row["sign"] == "positive" else "-"
        text = f"{sign}C{row['accepted_id']} {row['ratio_of_part'] * 100:.2f}%"
        cv2.putText(image, text, (cx - 48, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (cx - 48, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 1, cv2.LINE_AA)


def build_board(
    image: np.ndarray,
    gray_positive: np.ndarray,
    gray_negative: np.ndarray,
    raw_positive: np.ndarray,
    raw_negative: np.ndarray,
    selected_positive: np.ndarray,
    selected_negative: np.ndarray,
    positive_holes: np.ndarray,
    negative_holes: np.ndarray,
    final_positive: np.ndarray,
    final_negative: np.ndarray,
    rows: list[dict],
    merge_gap: int,
    cutoff_ratio: float,
    threshold_mm: float,
) -> np.ndarray:
    panel1 = image.copy()
    blend_mask(panel1, gray_positive, POS_RGB, 0.72)
    blend_mask(panel1, gray_negative, NEG_RGB, 0.72)
    draw_line(panel1, mask_boundary(gray_positive, 1), POS_RGB, 3)
    draw_line(panel1, mask_boundary(gray_negative, 1), NEG_RGB, 3)

    panel2 = image.copy()
    blend_mask(panel2, raw_positive, POS_RGB, 0.58)
    blend_mask(panel2, raw_negative, NEG_RGB, 0.58)

    panel3 = image.copy()
    blend_mask(panel3, selected_positive, MERGED_POS_RGB, 0.52)
    blend_mask(panel3, selected_negative, MERGED_NEG_RGB, 0.52)
    draw_line(panel3, mask_boundary(selected_positive, 1), POS_RGB, 3)
    draw_line(panel3, mask_boundary(selected_negative, 1), NEG_RGB, 3)

    panel4 = image.copy()
    blend_mask(panel4, final_positive, FINAL_POS_RGB, 0.58)
    blend_mask(panel4, final_negative, FINAL_NEG_RGB, 0.58)
    blend_mask(panel4, positive_holes | negative_holes, HOLE_RGB, 0.78)
    draw_line(panel4, mask_boundary(final_positive, 1), (255, 245, 235), 3)
    draw_line(panel4, mask_boundary(final_negative, 1), (235, 245, 255), 3)
    draw_region_labels(panel4, rows)

    panels = (
        add_title(panel1, "1. Gray sign from nearby colors", "red: above color-map max / blue: below color-map min"),
        add_title(
            panel2,
            "2. Raw correction pixels",
            f"red: > +{threshold_mm:g} mm / blue: < -{threshold_mm:g} mm",
        ),
        add_title(
            panel3,
            "3. Same-sign merge + hole fill",
            f"gap <= {merge_gap}px before area selection",
        ),
        add_title(
            panel4,
            "4. Final correction regions",
            f"area >= {cutoff_ratio * 100:g}% / {OPENING_KERNEL_SIZE_PX}px circular opening / yellow: filled holes",
        ),
    )
    panels = [fit_panel(panel) for panel in panels]
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))


def process_one(
    spec: ScanSpec,
    input_dir: Path,
    output_dir: Path,
    threshold_mm: float,
    gray_out_of_range_margin_mm: float,
    min_area_ratio: float,
    correction_merge_gap: int,
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
    positive_out_of_range_mm = max(float(spec.vmax), threshold_mm) + gray_out_of_range_margin_mm
    negative_out_of_range_mm = min(float(spec.vmin), -threshold_mm) - gray_out_of_range_margin_mm
    effective_values, gray_positive, gray_negative, nearest_gray_values = assign_gray_by_nearest_mapped_sign(
        values,
        mapped,
        gray,
        positive_out_of_range_mm,
        negative_out_of_range_mm,
    )
    raw_positive, raw_negative = build_signed_correction_masks(
        effective_values, mapped, gray_positive, gray_negative, threshold_mm
    )

    part_px = int(part.sum())
    merged_positive = merge_nearby_correction(raw_positive, correction_merge_gap, part)
    merged_negative = merge_nearby_correction(raw_negative, correction_merge_gap, part)
    merged_positive_filled, merged_positive_holes, positive_merge_rows = fill_all_internal_holes(
        merged_positive, part
    )
    merged_negative_filled, merged_negative_holes, negative_merge_rows = fill_all_internal_holes(
        merged_negative, part
    )
    merged_positive_filled, merged_negative_filled = resolve_sign_overlap(
        merged_positive_filled, merged_negative_filled, raw_positive, raw_negative
    )

    selected_positive, selected_negative, selected_labels, candidate_rows = select_final_regions(
        merged_positive_filled,
        merged_negative_filled,
        part_px,
        min_area_ratio,
        inclusive=True,
    )
    opened_positive = open_narrow_parts(selected_positive, OPENING_KERNEL_SIZE_PX, part)
    opened_negative = open_narrow_parts(selected_negative, OPENING_KERNEL_SIZE_PX, part)
    opened_positive, opened_negative = resolve_sign_overlap(
        opened_positive, opened_negative, raw_positive, raw_negative
    )
    final_positive = opened_positive
    final_negative = opened_negative
    # Sign-overlap resolution can expose enclosed raster gaps, so close the
    # pipeline by filling all final internal holes once more.
    final_positive, final_positive_holes, positive_final_rows = fill_all_internal_holes(
        final_positive, part
    )
    final_negative, final_negative_holes, negative_final_rows = fill_all_internal_holes(
        final_negative, part
    )
    final_positive, final_negative = resolve_sign_overlap(
        final_positive, final_negative, raw_positive, raw_negative
    )
    final_positive, final_negative, final_labels, final_check_rows = select_final_regions(
        final_positive, final_negative, part_px, min_area_ratio, inclusive=False
    )
    final_region_rows = [row for row in final_check_rows if row["accepted"]]
    positive_holes = (
        merged_positive_holes | final_positive_holes
    ) & final_positive
    negative_holes = (
        merged_negative_holes | final_negative_holes
    ) & final_negative
    accepted_source_rows = [row for row in candidate_rows if row["accepted"]]

    board = build_board(
        image,
        gray_positive,
        gray_negative,
        raw_positive,
        raw_negative,
        selected_positive,
        selected_negative,
        positive_holes,
        negative_holes,
        final_positive,
        final_negative,
        final_region_rows,
        correction_merge_gap,
        min_area_ratio,
        threshold_mm,
    )
    final_overlay = image.copy()
    blend_mask(final_overlay, final_positive, FINAL_POS_RGB, 0.58)
    blend_mask(final_overlay, final_negative, FINAL_NEG_RGB, 0.58)
    draw_line(final_overlay, mask_boundary(final_positive, 1), (255, 245, 235), 3)
    draw_line(final_overlay, mask_boundary(final_negative, 1), (235, 245, 255), 3)
    draw_region_labels(final_overlay, final_region_rows)

    item_dir = output_dir / spec.key
    imwrite_rgb(item_dir / "review_board.png", board)
    imwrite_rgb(item_dir / "final_correction_overlay.png", final_overlay)
    imwrite_gray(item_dir / "part_mask.png", part.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "gray_unmeasured_mask.png", gray.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "gray_assigned_positive_mask.png", gray_positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "gray_assigned_negative_mask.png", gray_negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "raw_positive_correction_mask.png", raw_positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "raw_negative_correction_mask.png", raw_negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "selected_positive_after_2pct_filter_mask.png", selected_positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "selected_negative_after_2pct_filter_mask.png", selected_negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "selected_correction_region_labels.png", selected_labels)
    imwrite_gray(item_dir / "merged_positive_after_hole_fill_mask.png", merged_positive_filled.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "merged_negative_after_hole_fill_mask.png", merged_negative_filled.astype(np.uint8) * 255)
    imwrite_gray(item_dir / f"positive_after_{OPENING_KERNEL_SIZE_PX}px_circular_opening_mask.png", opened_positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / f"negative_after_{OPENING_KERNEL_SIZE_PX}px_circular_opening_mask.png", opened_negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "positive_holes_filled_only_mask.png", positive_holes.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "negative_holes_filled_only_mask.png", negative_holes.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "final_positive_correction_mask.png", final_positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "final_negative_correction_mask.png", final_negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "final_correction_union_mask.png", ((final_positive | final_negative) * 255).astype(np.uint8))
    imwrite_gray(item_dir / "final_correction_region_labels.png", final_labels)

    summary = {
        "source_image": str(image_path),
        "legend_image": str(legend_path),
        "gray_assignment_method": "nearest non-zero mapped color sign",
        "gray_out_of_range_margin_mm": gray_out_of_range_margin_mm,
        "gray_positive_assigned_mm": positive_out_of_range_mm,
        "gray_negative_assigned_mm": negative_out_of_range_mm,
        "correction_threshold_mm": threshold_mm,
        "correction_merge_gap_px": correction_merge_gap,
        "elongated_part_filter": f"{OPENING_KERNEL_SIZE_PX}x{OPENING_KERNEL_SIZE_PX} circular-kernel opening",
        "minimum_correction_area_ratio_exclusive": min_area_ratio,
        "pre_kernel_area_rule": "merged connected ratio_of_part >= minimum_correction_area_ratio",
        "final_area_rule": "final connected ratio_of_part > minimum_correction_area_ratio_exclusive",
        "part_px": part_px,
        "gray_unmeasured_px": int(gray.sum()),
        "gray_assigned_positive_px": int(gray_positive.sum()),
        "gray_assigned_negative_px": int(gray_negative.sum()),
        "gray_nearest_mapped_mean_mm": float(nearest_gray_values[gray].mean()) if np.any(gray) else 0.0,
        "raw_positive_px": int(raw_positive.sum()),
        "raw_negative_px": int(raw_negative.sum()),
        "selected_positive_after_merge_px": int(selected_positive.sum()),
        "selected_negative_after_merge_px": int(selected_negative.sum()),
        "positive_holes_filled_px": int(positive_holes.sum()),
        "negative_holes_filled_px": int(negative_holes.sum()),
        "final_positive_px": int(final_positive.sum()),
        "final_negative_px": int(final_negative.sum()),
        "selected_positive_region_count_after_merge": sum(
            r["accepted"] and r["sign"] == "positive" for r in candidate_rows
        ),
        "selected_negative_region_count_after_merge": sum(
            r["accepted"] and r["sign"] == "negative" for r in candidate_rows
        ),
        "final_positive_region_count": sum(
            r["sign"] == "positive" for r in final_region_rows
        ),
        "final_negative_region_count": sum(
            r["sign"] == "negative" for r in final_region_rows
        ),
        "final_region_count": len(final_region_rows),
        "accepted_source_region_count": len(accepted_source_rows),
        "merged_regions_before_opening": candidate_rows,
        "final_regions_after_merge": final_region_rows,
        "final_region_check": final_check_rows,
        "positive_merge_components": positive_merge_rows,
        "negative_merge_components": negative_merge_rows,
        "positive_final_components": positive_final_rows,
        "negative_final_components": negative_final_rows,
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
        "part_px",
        "gray_unmeasured_ratio_of_part",
        "gray_positive_ratio_of_part",
        "gray_negative_ratio_of_part",
        "positive_region_count",
        "negative_region_count",
        "positive_final_ratio_of_part",
        "negative_final_ratio_of_part",
        "holes_filled_ratio_of_part",
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
                    "gray_unmeasured_ratio_of_part": row["gray_unmeasured_px"] / part_px,
                    "gray_positive_ratio_of_part": row["gray_assigned_positive_px"] / part_px,
                    "gray_negative_ratio_of_part": row["gray_assigned_negative_px"] / part_px,
                    "positive_region_count": row["final_positive_region_count"],
                    "negative_region_count": row["final_negative_region_count"],
                    "positive_final_ratio_of_part": row["final_positive_px"] / part_px,
                    "negative_final_ratio_of_part": row["final_negative_px"] / part_px,
                    "holes_filled_ratio_of_part": (
                        row["positive_holes_filled_px"] + row["negative_holes_filled_px"]
                    ) / part_px,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-mm", type=float, default=CORRECTION_THRESHOLD_MM)
    parser.add_argument("--gray-out-of-range-margin-mm", type=float, default=0.01)
    parser.add_argument("--min-area-ratio", type=float, default=0.02)
    parser.add_argument("--correction-merge-gap", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gray_out_of_range_margin_mm <= 0.0:
        raise ValueError("--gray-out-of-range-margin-mm must be positive")
    if not 0.0 <= args.min_area_ratio < 1.0:
        raise ValueError("--min-area-ratio must be in [0, 1)")
    summaries = [
        process_one(
            spec,
            args.input_dir,
            args.output_dir,
            args.threshold_mm,
            args.gray_out_of_range_margin_mm,
            args.min_area_ratio,
            args.correction_merge_gap,
        )
        for spec in SPECS
    ]
    write_summary(args.output_dir, summaries)
    for row in summaries:
        name = Path(row["source_image"]).name
        print(
            f"{name}: +areas={row['final_positive_region_count']}, "
            f"-areas={row['final_negative_region_count']}, "
            f"+area={row['final_positive_px'] / row['part_px'] * 100:.2f}%, "
            f"-area={row['final_negative_px'] / row['part_px'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
