"""Preview zero-line areas lying between positive and negative correction areas.

Rules implemented for this review run:
  * unmapped gray material is assigned the sentinel deviation +3.01 mm;
  * positive correction: deviation > +0.5 mm (including gray sentinel);
  * negative correction: deviation < -0.5 mm;
  * zero-line candidate: -0.5 <= deviation <= +0.5 mm and adjacent to
    both positive and negative correction areas;
  * final zero-line area: each connected candidate >= 5% of total part area.

This script is isolated from the production zero-line engine.
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


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_gray_gt3_between_signs"

POS_RGB = (255, 65, 45)
NEG_RGB = (45, 115, 255)
GRAY_SENTINEL_RGB = (220, 45, 220)
ZERO_CANDIDATE_RGB = (75, 225, 105)
FINAL_ZERO_RGB = (255, 235, 0)


def strict_morphology(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    """Remove noise without adding any pixel that failed the source condition."""
    original = mask.astype(bool)
    out = original.astype(np.uint8)
    if open_size > 0:
        out = cv2.morphologyEx(
            out,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
        )
    if close_size > 0:
        out = cv2.morphologyEx(
            out,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        )
    return out.astype(bool) & original


def detect_unmapped_gray(
    image_rgb: np.ndarray,
    part_mask: np.ndarray,
    mapped_mask: np.ndarray,
) -> np.ndarray:
    """Find gray material pixels that have no valid color-map value."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    gray = (
        part_mask
        & ~mapped_mask
        & (hsv[:, :, 1] <= 45)
        & (hsv[:, :, 2] >= 60)
        & (hsv[:, :, 2] <= 245)
    )
    # A gray *surface* must have appreciable thickness.  The 7x7 opening drops
    # anti-aliased CAD outlines and thin feature lines that are gray but are
    # not unmeasured faces.
    return strict_morphology(gray, open_size=7, close_size=7)


def build_correction_masks(
    values: np.ndarray,
    mapped_mask: np.ndarray,
    gray_mask: np.ndarray,
    threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_source = (mapped_mask & (values > threshold_mm)) | gray_mask
    negative_source = mapped_mask & (values < -threshold_mm)
    neutral_source = mapped_mask & (values >= -threshold_mm) & (values <= threshold_mm)
    positive = strict_morphology(positive_source, open_size=5, close_size=7)
    negative = strict_morphology(negative_source, open_size=5, close_size=7)
    neutral = strict_morphology(neutral_source, open_size=3, close_size=5)
    return positive, negative, neutral


def select_between_signs_zero_regions(
    neutral_mask: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    part_px: int,
    min_area_ratio: float,
    adjacency_radius: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Select neutral components that touch both correction signs."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (adjacency_radius * 2 + 1, adjacency_radius * 2 + 1),
    )
    near_positive = cv2.dilate(positive_mask.astype(np.uint8), kernel) > 0
    near_negative = cv2.dilate(negative_mask.astype(np.uint8), kernel) > 0

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        neutral_mask.astype(np.uint8), connectivity=8
    )
    candidate = np.zeros_like(neutral_mask, dtype=bool)
    final = np.zeros_like(neutral_mask, dtype=bool)
    rows: list[dict] = []
    for i in range(1, n):
        component = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        touches_positive = bool(np.any(component & near_positive))
        touches_negative = bool(np.any(component & near_negative))
        between_signs = touches_positive and touches_negative
        ratio = area / part_px if part_px else 0.0
        accepted = between_signs and ratio >= min_area_ratio
        if between_signs:
            candidate[component] = True
        if accepted:
            final[component] = True
        x, y, w, h, _ = (int(v) for v in stats[i])
        rows.append(
            {
                "component_id": int(i),
                "area_px": area,
                "ratio_of_part": ratio,
                "centroid": [float(centroids[i][0]), float(centroids[i][1])],
                "bbox": [x, y, w, h],
                "touches_positive": touches_positive,
                "touches_negative": touches_negative,
                "between_signs": between_signs,
                "accepted": accepted,
            }
        )

    rows.sort(key=lambda row: row["area_px"], reverse=True)
    accepted_id = 0
    for row in rows:
        if row["accepted"]:
            accepted_id += 1
            row["accepted_id"] = accepted_id
        else:
            row["accepted_id"] = None
    return candidate, final, rows


def draw_component_labels(image: np.ndarray, rows: list[dict], accepted_only: bool) -> None:
    for row in rows:
        if not row["between_signs"]:
            continue
        if accepted_only and not row["accepted"]:
            continue
        cx, cy = (int(round(v)) for v in row["centroid"])
        if row["accepted"]:
            name = f"Z{row['accepted_id']}"
        else:
            name = f"C{row['component_id']}"
        text = f"{name} {row['ratio_of_part'] * 100:.1f}%"
        cv2.putText(
            image, text, (cx - 42, cy), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, (cx - 42, cy), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )


def mask_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    out = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(out, contours, -1, 255, width, cv2.LINE_AA)
    return out


def draw_line(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    binary = (mask > 0).astype(np.uint8)
    if width > 1:
        binary = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width)),
        )
    image[binary > 0] = color


def build_board(
    original: np.ndarray,
    gray_mask: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    neutral: np.ndarray,
    candidate: np.ndarray,
    final: np.ndarray,
    rows: list[dict],
    subtitle: str,
) -> np.ndarray:
    correction_view = original.copy()
    blend_mask(correction_view, positive, POS_RGB, 0.58)
    blend_mask(correction_view, negative, NEG_RGB, 0.58)
    blend_mask(correction_view, gray_mask, GRAY_SENTINEL_RGB, 0.80)

    candidate_view = original.copy()
    blend_mask(candidate_view, positive, POS_RGB, 0.20)
    blend_mask(candidate_view, negative, NEG_RGB, 0.20)
    blend_mask(candidate_view, neutral, (105, 210, 120), 0.10)
    blend_mask(candidate_view, candidate, ZERO_CANDIDATE_RGB, 0.68)
    draw_component_labels(candidate_view, rows, accepted_only=False)

    final_view = original.copy()
    blend_mask(final_view, positive, POS_RGB, 0.12)
    blend_mask(final_view, negative, NEG_RGB, 0.12)
    blend_mask(final_view, final, ZERO_CANDIDATE_RGB, 0.52)
    draw_line(final_view, mask_boundary(final, 1), FINAL_ZERO_RGB, 4)
    draw_component_labels(final_view, rows, accepted_only=True)

    panels = [
        add_title(fit_panel(original), "1. Original scan", subtitle),
        add_title(
            fit_panel(correction_view),
            "2. Correction areas",
            "red: positive / blue: negative / magenta: gray assigned +3.01 mm",
        ),
        add_title(
            fit_panel(candidate_view),
            "3. Neutral areas between both signs",
            "green: -0.5..+0.5 mm component touching positive and negative",
        ),
        add_title(
            fit_panel(final_view),
            "4. Final zero-line areas",
            "green/yellow: each between-sign component >= 5% of total part",
        ),
    ]
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))


def process_one(
    spec: ScanSpec,
    input_dir: Path,
    output_dir: Path,
    threshold_mm: float,
    gray_sentinel_mm: float,
    min_area_ratio: float,
    adjacency_radius: int,
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
    gray = detect_unmapped_gray(image, part, mapped)
    effective_values = values.copy()
    effective_values[gray] = gray_sentinel_mm

    positive, negative, neutral = build_correction_masks(
        effective_values, mapped, gray, threshold_mm
    )
    part_px = int(part.sum())
    candidate, final, rows = select_between_signs_zero_regions(
        neutral,
        positive,
        negative,
        part_px,
        min_area_ratio,
        adjacency_radius,
    )

    item_dir = output_dir / spec.key
    subtitle = (
        f"range {spec.vmin:+.1f}..{spec.vmax:+.1f} mm / "
        f"gray +{gray_sentinel_mm:.2f} mm / total part {part_px:,} px"
    )
    board = build_board(
        image, gray, positive, negative, neutral, candidate, final, rows, subtitle
    )
    overlay = image.copy()
    blend_mask(overlay, positive, POS_RGB, 0.14)
    blend_mask(overlay, negative, NEG_RGB, 0.14)
    blend_mask(overlay, gray, GRAY_SENTINEL_RGB, 0.32)
    blend_mask(overlay, final, ZERO_CANDIDATE_RGB, 0.55)
    draw_line(overlay, mask_boundary(final, 1), FINAL_ZERO_RGB, 4)
    draw_component_labels(overlay, rows, accepted_only=True)

    imwrite_rgb(item_dir / "review_board.png", board)
    imwrite_rgb(item_dir / "final_zero_line_overlay.png", overlay)
    imwrite_gray(item_dir / "part_mask.png", part.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "mapped_color_mask.png", mapped.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "gray_assigned_gt3_mask.png", gray.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "positive_correction_mask.png", positive.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "negative_correction_mask.png", negative.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "neutral_minus05_plus05_mask.png", neutral.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "between_signs_candidate_mask.png", candidate.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "final_zero_line_area_mask.png", final.astype(np.uint8) * 255)

    accepted_rows = [row for row in rows if row["accepted"]]
    between_rows = [row for row in rows if row["between_signs"]]
    summary = {
        "source_image": str(image_path),
        "legend_image": str(legend_path),
        "value_range_mm": [spec.vmin, spec.vmax],
        "gray_sentinel_mm": gray_sentinel_mm,
        "correction_threshold_mm": threshold_mm,
        "minimum_zero_region_ratio": min_area_ratio,
        "sign_adjacency_radius_px": adjacency_radius,
        "part_px": part_px,
        "mapped_color_px": int(mapped.sum()),
        "mapped_color_ratio_of_part": float(mapped.sum() / part_px) if part_px else 0.0,
        "gray_gt3_px": int(gray.sum()),
        "gray_gt3_ratio_of_part": float(gray.sum() / part_px) if part_px else 0.0,
        "positive_correction_px": int(positive.sum()),
        "positive_correction_ratio_of_part": float(positive.sum() / part_px) if part_px else 0.0,
        "negative_correction_px": int(negative.sum()),
        "negative_correction_ratio_of_part": float(negative.sum() / part_px) if part_px else 0.0,
        "neutral_px": int(neutral.sum()),
        "between_signs_component_count": len(between_rows),
        "between_signs_candidate_px": int(candidate.sum()),
        "between_signs_candidate_ratio_of_part": float(candidate.sum() / part_px) if part_px else 0.0,
        "final_zero_region_count": len(accepted_rows),
        "final_zero_line_px": int(final.sum()),
        "final_zero_line_ratio_of_part": float(final.sum() / part_px) if part_px else 0.0,
        "neutral_components": rows,
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
        "mapped_color_ratio_of_part",
        "gray_gt3_ratio_of_part",
        "positive_correction_ratio_of_part",
        "negative_correction_ratio_of_part",
        "between_signs_component_count",
        "between_signs_candidate_ratio_of_part",
        "final_zero_region_count",
        "final_zero_line_ratio_of_part",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "image": Path(row["source_image"]).name,
                    **{field: row[field] for field in fields[2:]},
                    "part_px": row["part_px"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-mm", type=float, default=0.5)
    parser.add_argument("--gray-sentinel-mm", type=float, default=3.01)
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--adjacency-radius", type=int, default=5)
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
            args.min_area_ratio,
            args.adjacency_radius,
        )
        for spec in SPECS
    ]
    write_summary(args.output_dir, summaries)
    for row in summaries:
        print(
            f"{Path(row['source_image']).name}: "
            f"gray>3={row['gray_gt3_ratio_of_part'] * 100:.2f}%, "
            f"between={row['between_signs_component_count']}, "
            f"final={row['final_zero_region_count']} "
            f"({row['final_zero_line_ratio_of_part'] * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
