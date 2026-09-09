"""Merged correction-area / grouped zero-area preview with a 4% cutoff.

This review-only experiment extends ``generate_between_signs_preview.py``:
  * same-sign correction areas separated by <= 24 px are merged;
  * connected neutral zero areas must touch both correction signs and cover >= 4% of
    the total part;
  * holes inside a merged correction component are filled for display only
    when none of the holes contains a final zero-line area.

The production zero-line engine is not imported or modified.
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


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_merged_correction_4pct"
FINAL_ZERO_RGB = (255, 235, 0)
POS_FILL_RGB = (255, 155, 65)
NEG_FILL_RGB = (75, 205, 255)


def _odd_at_least_three(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def merge_nearby_correction(mask: np.ndarray, max_gap_px: int, part: np.ndarray) -> np.ndarray:
    """Join nearby same-sign areas into a broad correction-area mask."""
    size = _odd_at_least_three(max_gap_px + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    merged = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return (merged > 0) & part


def group_nearby_neutral_regions(
    neutral: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    part_px: int,
    min_area_ratio: float,
    zero_group_gap_px: int,
    sign_adjacency_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Group nearby neutral pieces without adding bridge pixels to their area."""
    if zero_group_gap_px > 0:
        group_radius = max(1, int(np.ceil(zero_group_gap_px / 2)))
        group_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (group_radius * 2 + 1, group_radius * 2 + 1)
        )
        expanded = cv2.dilate(neutral.astype(np.uint8), group_kernel)
    else:
        expanded = neutral.astype(np.uint8)
    n, group_labels, _, _ = cv2.connectedComponentsWithStats(expanded, connectivity=8)

    adjacency_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (sign_adjacency_px * 2 + 1, sign_adjacency_px * 2 + 1),
    )
    near_positive = cv2.dilate(positive.astype(np.uint8), adjacency_kernel) > 0
    near_negative = cv2.dilate(negative.astype(np.uint8), adjacency_kernel) > 0

    between = np.zeros_like(neutral, dtype=bool)
    final = np.zeros_like(neutral, dtype=bool)
    accepted_labels = np.zeros(neutral.shape, np.int32)
    rows: list[dict] = []
    accepted_id = 0
    for group_id in range(1, n):
        group_core = neutral & (group_labels == group_id)
        area = int(group_core.sum())
        if area == 0:
            continue
        touches_positive = bool(np.any(group_core & near_positive))
        touches_negative = bool(np.any(group_core & near_negative))
        between_signs = touches_positive and touches_negative
        ratio = area / part_px if part_px else 0.0
        accepted = between_signs and ratio >= min_area_ratio
        if between_signs:
            between[group_core] = True
        if accepted:
            accepted_id += 1
            final[group_core] = True
            accepted_labels[group_core] = accepted_id

        component_count, _, _, _ = cv2.connectedComponentsWithStats(
            group_core.astype(np.uint8), connectivity=8
        )
        ys, xs = np.where(group_core)
        rows.append(
            {
                "group_id": int(group_id),
                "accepted_id": accepted_id if accepted else None,
                "area_px": area,
                "ratio_of_part": ratio,
                "centroid": [float(xs.mean()), float(ys.mean())],
                "bbox": [
                    int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
                ],
                "member_component_count": int(component_count - 1),
                "touches_positive": touches_positive,
                "touches_negative": touches_negative,
                "between_signs": between_signs,
                "accepted": accepted,
            }
        )
    rows.sort(key=lambda row: row["area_px"], reverse=True)

    # Re-number accepted groups in descending area order for readable overlays.
    accepted_id = 0
    remapped_labels = np.zeros_like(accepted_labels)
    for row in rows:
        if not row["accepted"]:
            row["accepted_id"] = None
            continue
        accepted_id += 1
        old_id = row["accepted_id"]
        remapped_labels[accepted_labels == old_id] = accepted_id
        row["accepted_id"] = accepted_id
    return between, final, remapped_labels, rows


def fill_holes_without_final_zero(
    merged_mask: np.ndarray,
    final_zero: np.ndarray,
    part: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fill a correction component's holes only if none contains final zero."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        merged_mask.astype(np.uint8), connectivity=8
    )
    displayed = merged_mask.copy()
    filled_only = np.zeros_like(merged_mask, dtype=bool)
    rows: list[dict] = []
    for component_id in range(1, n):
        component = labels == component_id
        x, y, w, h, area = (int(v) for v in stats[component_id])
        local = component[y:y + h, x:x + w].astype(np.uint8)
        padded = cv2.copyMakeBorder(local, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        flood = padded.copy()
        cv2.floodFill(flood, flood_mask, (0, 0), 1)
        holes_local = (flood[1:-1, 1:-1] == 0) & (local == 0)
        holes = np.zeros_like(component)
        holes[y:y + h, x:x + w] = holes_local
        holes &= part
        contains_final_zero = bool(np.any(holes & final_zero))
        fill_applied = bool(np.any(holes)) and not contains_final_zero
        if fill_applied:
            displayed[holes] = True
            filled_only[holes] = True
        rows.append(
            {
                "component_id": component_id,
                "area_px_before_fill": area,
                "hole_px": int(holes.sum()),
                "contains_final_zero_in_hole": contains_final_zero,
                "hole_fill_applied": fill_applied,
            }
        )
    return displayed, filled_only, rows


def resolve_sign_overlap(
    positive: np.ndarray,
    negative: np.ndarray,
    raw_positive: np.ndarray,
    raw_negative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign display-only overlap to the nearest original correction sign."""
    overlap = positive & negative
    if not np.any(overlap):
        return positive, negative
    dist_positive = cv2.distanceTransform((~raw_positive).astype(np.uint8), cv2.DIST_L2, 3)
    dist_negative = cv2.distanceTransform((~raw_negative).astype(np.uint8), cv2.DIST_L2, 3)
    positive = positive.copy()
    negative = negative.copy()
    positive[overlap & (dist_negative < dist_positive)] = False
    negative[overlap & (dist_positive <= dist_negative)] = False
    return positive, negative


def draw_group_labels(image: np.ndarray, rows: list[dict], accepted_only: bool) -> None:
    for row in rows:
        if not row["between_signs"]:
            continue
        if accepted_only and not row["accepted"]:
            continue
        cx, cy = (int(round(v)) for v in row["centroid"])
        name = f"Z{row['accepted_id']}" if row["accepted"] else f"G{row['group_id']}"
        text = f"{name} {row['ratio_of_part'] * 100:.1f}%"
        cv2.putText(
            image, text, (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )


def build_board(
    original: np.ndarray,
    gray: np.ndarray,
    raw_positive: np.ndarray,
    raw_negative: np.ndarray,
    merged_positive: np.ndarray,
    merged_negative: np.ndarray,
    positive_fill_only: np.ndarray,
    negative_fill_only: np.ndarray,
    between: np.ndarray,
    final_zero: np.ndarray,
    rows: list[dict],
    subtitle: str,
    min_zero_ratio: float,
) -> np.ndarray:
    raw_view = original.copy()
    blend_mask(raw_view, raw_positive, POS_RGB, 0.58)
    blend_mask(raw_view, raw_negative, NEG_RGB, 0.58)
    blend_mask(raw_view, gray, GRAY_SENTINEL_RGB, 0.80)

    merged_view = original.copy()
    blend_mask(merged_view, merged_positive, POS_RGB, 0.50)
    blend_mask(merged_view, merged_negative, NEG_RGB, 0.50)
    blend_mask(merged_view, positive_fill_only, POS_FILL_RGB, 0.90)
    blend_mask(merged_view, negative_fill_only, NEG_FILL_RGB, 0.90)

    candidate_view = original.copy()
    blend_mask(candidate_view, merged_positive, POS_RGB, 0.14)
    blend_mask(candidate_view, merged_negative, NEG_RGB, 0.14)
    blend_mask(candidate_view, between, ZERO_CANDIDATE_RGB, 0.66)
    draw_group_labels(candidate_view, rows, accepted_only=False)

    final_view = merged_view.copy()
    blend_mask(final_view, final_zero, ZERO_CANDIDATE_RGB, 0.62)
    draw_line(final_view, mask_boundary(final_zero, 1), FINAL_ZERO_RGB, 4)
    draw_group_labels(final_view, rows, accepted_only=True)

    panels = [
        add_title(
            fit_panel(raw_view),
            "1. Strict correction pixels",
            subtitle,
        ),
        add_title(
            fit_panel(merged_view),
            "2. Merged correction areas",
            "same sign gap <= 24 px; light red/blue: conditionally filled holes",
        ),
        add_title(
            fit_panel(candidate_view),
            "3. Connected between-sign zero candidates",
            "each connected neutral area is measured separately",
        ),
        add_title(
            fit_panel(final_view),
            "4. Final zero-line areas",
            f"green/yellow: connected area >= {min_zero_ratio * 100:g}% of total part",
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
    zero_group_gap: int,
    sign_adjacency: int,
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
    between, final_zero, final_labels, zero_rows = group_nearby_neutral_regions(
        neutral,
        merged_positive,
        merged_negative,
        part_px,
        min_zero_ratio,
        zero_group_gap,
        sign_adjacency,
    )

    display_positive, positive_fill_only, positive_components = fill_holes_without_final_zero(
        merged_positive, final_zero, part
    )
    display_negative, negative_fill_only, negative_components = fill_holes_without_final_zero(
        merged_negative, final_zero, part
    )
    display_positive, display_negative = resolve_sign_overlap(
        display_positive, display_negative, raw_positive, raw_negative
    )
    positive_fill_only &= display_positive
    negative_fill_only &= display_negative

    subtitle = (
        f"gray +{gray_sentinel_mm:.2f} mm / correction gap {correction_merge_gap}px / "
        f"zero gap {zero_group_gap}px / cutoff {min_zero_ratio * 100:.0f}%"
    )
    board = build_board(
        image,
        gray,
        raw_positive,
        raw_negative,
        display_positive,
        display_negative,
        positive_fill_only,
        negative_fill_only,
        between,
        final_zero,
        zero_rows,
        subtitle,
        min_zero_ratio,
    )
    overlay = image.copy()
    blend_mask(overlay, display_positive, POS_RGB, 0.23)
    blend_mask(overlay, display_negative, NEG_RGB, 0.23)
    blend_mask(overlay, positive_fill_only, POS_FILL_RGB, 0.75)
    blend_mask(overlay, negative_fill_only, NEG_FILL_RGB, 0.75)
    blend_mask(overlay, final_zero, ZERO_CANDIDATE_RGB, 0.62)
    draw_line(overlay, mask_boundary(final_zero, 1), FINAL_ZERO_RGB, 4)
    draw_group_labels(overlay, zero_rows, accepted_only=True)

    item_dir = output_dir / spec.key
    imwrite_rgb(item_dir / "review_board.png", board)
    imwrite_rgb(item_dir / "final_overlay.png", overlay)
    imwrite_gray(item_dir / "part_mask.png", part.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "gray_assigned_gt3_mask.png", gray.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "raw_positive_correction_mask.png", raw_positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "raw_negative_correction_mask.png", raw_negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "merged_positive_correction_mask.png", display_positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "merged_negative_correction_mask.png", display_negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "positive_holes_filled_only_mask.png", positive_fill_only.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "negative_holes_filled_only_mask.png", negative_fill_only.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "between_signs_candidate_mask.png", between.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "final_zero_line_area_mask.png", final_zero.astype(np.uint8) * 255)
    # Store group IDs losslessly for downstream inspection.
    imwrite_gray(item_dir / "final_zero_line_group_labels.png", final_labels.astype(np.uint16))

    accepted_rows = [row for row in zero_rows if row["accepted"]]
    between_rows = [row for row in zero_rows if row["between_signs"]]
    summary = {
        "source_image": str(image_path),
        "legend_image": str(legend_path),
        "gray_sentinel_mm": gray_sentinel_mm,
        "correction_threshold_mm": threshold_mm,
        "correction_merge_gap_px": correction_merge_gap,
        "zero_group_gap_px": zero_group_gap,
        "sign_adjacency_px": sign_adjacency,
        "minimum_final_zero_ratio": min_zero_ratio,
        "part_px": part_px,
        "gray_gt3_px": int(gray.sum()),
        "raw_positive_px": int(raw_positive.sum()),
        "raw_negative_px": int(raw_negative.sum()),
        "merged_positive_display_px": int(display_positive.sum()),
        "merged_negative_display_px": int(display_negative.sum()),
        "positive_holes_filled_px": int(positive_fill_only.sum()),
        "negative_holes_filled_px": int(negative_fill_only.sum()),
        "between_signs_group_count": len(between_rows),
        "between_signs_candidate_px": int(between.sum()),
        "final_zero_region_count": len(accepted_rows),
        "final_zero_line_px": int(final_zero.sum()),
        "final_zero_line_ratio_of_part": float(final_zero.sum() / part_px) if part_px else 0.0,
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
        "part_px",
        "gray_gt3_ratio_of_part",
        "merged_positive_ratio_of_part",
        "merged_negative_ratio_of_part",
        "holes_filled_ratio_of_part",
        "between_signs_group_count",
        "final_zero_region_count",
        "final_zero_line_ratio_of_part",
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
                    "gray_gt3_ratio_of_part": row["gray_gt3_px"] / part_px,
                    "merged_positive_ratio_of_part": row["merged_positive_display_px"] / part_px,
                    "merged_negative_ratio_of_part": row["merged_negative_display_px"] / part_px,
                    "holes_filled_ratio_of_part": (
                        row["positive_holes_filled_px"] + row["negative_holes_filled_px"]
                    ) / part_px,
                    "between_signs_group_count": row["between_signs_group_count"],
                    "final_zero_region_count": row["final_zero_region_count"],
                    "final_zero_line_ratio_of_part": row["final_zero_line_ratio_of_part"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-mm", type=float, default=0.5)
    parser.add_argument("--gray-sentinel-mm", type=float, default=3.01)
    parser.add_argument("--min-zero-ratio", type=float, default=0.04)
    parser.add_argument("--correction-merge-gap", type=int, default=24)
    parser.add_argument("--zero-group-gap", type=int, default=0)
    parser.add_argument("--sign-adjacency", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gray_sentinel_mm <= 3.0:
        raise ValueError("--gray-sentinel-mm must be greater than 3.0")
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
            args.zero_group_gap,
            args.sign_adjacency,
        )
        for spec in SPECS
    ]
    write_summary(args.output_dir, summaries)
    for row in summaries:
        part_px = row["part_px"]
        filled = row["positive_holes_filled_px"] + row["negative_holes_filled_px"]
        print(
            f"{Path(row['source_image']).name}: "
            f"final={row['final_zero_region_count']} "
            f"({row['final_zero_line_ratio_of_part'] * 100:.2f}%), "
            f"holes-filled={filled / part_px * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
