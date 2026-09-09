"""Review-only separators at the positive/negative correction transition.

This does not alter the production engine. A separator is drawn only in the
non-correction gap where final positive and negative masks are equally near.
Existing outer-boundary sign-transition anchors select review candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from generate_preview import (SPECS, add_title, blend_mask, detect_part_mask,
                              extract_color_ramp, fit_panel, imread_rgb,
                              imwrite_gray, imwrite_rgb, map_deviation)
from generate_between_signs_preview import detect_unmapped_gray, draw_line

HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))
from zero_line_detection.zero_boundary import find_boundary_anchors  # noqa: E402

DEFAULT_CORRECTION_RESULTS = HERE / "results_correction_only_3pct"
DEFAULT_OUTPUT = HERE / "results_correction_sign_boundaries"
ANCHOR_RGB, RAW_RGB, SPLIT_RGB = (40, 230, 85), (150, 150, 150), (255, 230, 0)
POS_POS_RGB, NEG_NEG_RGB = (235, 90, 255), (0, 235, 255)
POS_FILL_RGB, NEG_FILL_RGB = (255, 80, 45), (50, 120, 255)


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    return image > 0


def skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary) > 0
    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary.copy()
    while cv2.countNonZero(current):
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = cv2.erode(current, kernel)
    return skeleton > 0


def pair_bisector(first, second, gap, max_distance_px):
    """One-pixel equal-distance boundary for a pair of final components."""
    dist_positive = cv2.distanceTransform((~first).astype(np.uint8), cv2.DIST_L2, 5)
    dist_negative = cv2.distanceTransform((~second).astype(np.uint8), cv2.DIST_L2, 5)
    near_positive, near_negative = dist_positive < dist_negative, dist_negative < dist_positive
    k3 = np.ones((3, 3), np.uint8)
    touching = ((cv2.dilate(near_positive.astype(np.uint8), k3) > 0)
                & (cv2.dilate(near_negative.astype(np.uint8), k3) > 0))
    balanced = np.abs(dist_positive - dist_negative) <= 3.0
    both_near = (dist_positive <= max_distance_px) & (dist_negative <= max_distance_px)
    return skeletonize(gap & both_near & (touching | balanced))


def component_masks(mask):
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    return [labels == index for index in range(1, count)]


def all_correction_bisectors(positive, negative, part, max_distance_px):
    """Separate every pair of final regions: +/-, +/+ and -/-."""
    gap = part & ~positive & ~negative
    positive_components, negative_components = component_masks(positive), component_masks(negative)
    pos_neg = np.zeros_like(part, bool)
    pos_pos = np.zeros_like(part, bool)
    neg_neg = np.zeros_like(part, bool)
    for positive_component in positive_components:
        for negative_component in negative_components:
            pos_neg |= pair_bisector(positive_component, negative_component, gap, max_distance_px)
    for index, first in enumerate(positive_components):
        for second in positive_components[index + 1:]:
            pos_pos |= pair_bisector(first, second, gap, max_distance_px)
    for index, first in enumerate(negative_components):
        for second in negative_components[index + 1:]:
            neg_neg |= pair_bisector(first, second, gap, max_distance_px)
    return pos_neg, pos_pos, neg_neg, gap


def anchor_connected_components(raw_line, anchors, max_anchor_distance):
    """Keep full +/- boundary components that begin near an engine anchor."""
    kept, starts = np.zeros_like(raw_line, bool), np.zeros_like(raw_line, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_line.astype(np.uint8), 8)
    rows, used = [], set()
    for anchor in anchors:
        best = None
        for component_id in range(1, count):
            ys, xs = np.where(labels == component_id)
            if not len(xs):
                continue
            distances = np.hypot(xs.astype(float) - anchor.x, ys.astype(float) - anchor.y)
            index = int(np.argmin(distances))
            candidate = (float(distances[index]), component_id, int(xs[index]), int(ys[index]))
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None or best[0] > max_anchor_distance:
            continue
        distance, component_id, x, y = best
        starts[y, x] = True
        if component_id in used:
            continue
        used.add(component_id)
        component = labels == component_id
        kept |= component
        rows.append({"separator_id": len(rows) + 1, "anchor_id": anchor.anchor_id,
                     "anchor_xy": [anchor.x, anchor.y], "line_start_xy": [x, y],
                     "anchor_to_line_distance_px": round(distance, 2),
                     "line_pixels": int(component.sum()),
                     "bbox": [int(v) for v in stats[component_id, :4]]})
    return kept, starts, rows


def build_board(image, positive, negative, anchors, raw_line, kept_line, pos_pos_line, neg_neg_line, starts, subtitle):
    correction = image.copy()
    blend_mask(correction, positive, POS_FILL_RGB, 0.50)
    blend_mask(correction, negative, NEG_FILL_RGB, 0.50)
    anchor_view = correction.copy()
    for anchor in anchors:
        cv2.circle(anchor_view, (anchor.x, anchor.y), 7, ANCHOR_RGB, -1, cv2.LINE_AA)
        cv2.circle(anchor_view, (anchor.x, anchor.y), 7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(anchor_view, f"A{anchor.anchor_id}", (anchor.x + 8, anchor.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    raw_view = correction.copy()
    draw_line(raw_view, raw_line.astype(np.uint8) * 255, RAW_RGB, 3)
    draw_line(raw_view, pos_pos_line.astype(np.uint8) * 255, POS_POS_RGB, 3)
    draw_line(raw_view, neg_neg_line.astype(np.uint8) * 255, NEG_NEG_RGB, 3)
    final_view = image.copy()
    blend_mask(final_view, positive, POS_FILL_RGB, 0.22)
    blend_mask(final_view, negative, NEG_FILL_RGB, 0.22)
    draw_line(final_view, kept_line.astype(np.uint8) * 255, SPLIT_RGB, 4)
    draw_line(final_view, pos_pos_line.astype(np.uint8) * 255, POS_POS_RGB, 4)
    draw_line(final_view, neg_neg_line.astype(np.uint8) * 255, NEG_NEG_RGB, 4)
    draw_line(final_view, starts.astype(np.uint8) * 255, ANCHOR_RGB, 7)
    for anchor in anchors:
        cv2.circle(final_view, (anchor.x, anchor.y), 5, ANCHOR_RGB, 1, cv2.LINE_AA)
    panels = (
        add_title(fit_panel(correction), "1. Final correction regions", "orange: positive / blue: negative"),
        add_title(fit_panel(anchor_view), "2. Existing-engine start anchors", "green: outer-boundary sign transition"),
        add_title(fit_panel(raw_view), "3. All correction-area boundaries", "gray: +/-  magenta: +/+  cyan: -/-"),
        add_title(fit_panel(final_view), "4. Final split lines", subtitle),
    )
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))


def process_one(spec, correction_dir, output_dir, smooth_window, min_anchor_separation,
                max_anchor_distance, max_between_distance):
    source = correction_dir / spec.key
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    image = imread_rgb(Path(source_summary["source_image"]))
    values, valid = map_deviation(image, extract_color_ramp(imread_rgb(Path(source_summary["legend_image"]))), spec.vmin, spec.vmax)
    values = cv2.medianBlur(values, 5)
    part = detect_part_mask(image)
    gray = detect_unmapped_gray(image, part, part & valid)
    values[gray] = 3.01
    positive, negative = read_mask(source / "final_positive_correction_mask.png"), read_mask(source / "final_negative_correction_mask.png")
    anchors = find_boundary_anchors(values, part, smooth_window, min_anchor_separation)
    raw_line, pos_pos_line, neg_neg_line, gap = all_correction_bisectors(
        positive, negative, part, max_between_distance
    )
    final_line, starts, rows = anchor_connected_components(raw_line, anchors, max_anchor_distance)
    raw_count = int(cv2.connectedComponents(raw_line.astype(np.uint8), 8)[0] - 1)
    subtitle = (f"yellow: anchored +/-  magenta: +/+  cyan: -/-; "
                f"{len(rows)} anchored +/- components")
    board = build_board(image, positive, negative, anchors, raw_line, final_line, pos_pos_line, neg_neg_line, starts, subtitle)
    overlay = image.copy()
    blend_mask(overlay, positive, POS_FILL_RGB, 0.20)
    blend_mask(overlay, negative, NEG_FILL_RGB, 0.20)
    draw_line(overlay, final_line.astype(np.uint8) * 255, SPLIT_RGB, 4)
    draw_line(overlay, pos_pos_line.astype(np.uint8) * 255, POS_POS_RGB, 4)
    draw_line(overlay, neg_neg_line.astype(np.uint8) * 255, NEG_NEG_RGB, 4)
    draw_line(overlay, starts.astype(np.uint8) * 255, ANCHOR_RGB, 7)
    item_dir = output_dir / spec.key
    imwrite_rgb(item_dir / "review_board.png", board)
    imwrite_rgb(item_dir / "positive_negative_split_overlay.png", overlay)
    anchor_mask = np.zeros(part.shape, np.uint8)
    for anchor in anchors:
        cv2.circle(anchor_mask, (anchor.x, anchor.y), 5, 255, -1)
    imwrite_gray(item_dir / "boundary_start_anchor_mask.png", anchor_mask)
    imwrite_gray(item_dir / "positive_negative_bisector_mask.png", raw_line.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "anchor_connected_split_line_mask.png", final_line.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "positive_positive_bisector_mask.png", pos_pos_line.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "negative_negative_bisector_mask.png", neg_neg_line.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "selected_line_start_mask.png", starts.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "correction_gap_mask.png", gap.astype(np.uint8) * 255)
    summary = {"source_correction_result": str(source), "source_image": source_summary["source_image"],
               "start_anchor_method": "zero_line_detection.zero_boundary.find_boundary_anchors",
               "separator_method": "equal distance boundaries between every pair of final correction components; restricted to their gap",
               "anchor_smooth_window_px": smooth_window, "anchor_min_separation_px": min_anchor_separation,
               "max_anchor_to_separator_distance_px": max_anchor_distance, "max_distance_to_each_sign_px": max_between_distance,
               "anchor_count": len(anchors), "raw_separator_component_count": raw_count,
               "positive_positive_separator_component_count": int(cv2.connectedComponents(pos_pos_line.astype(np.uint8), 8)[0] - 1),
               "negative_negative_separator_component_count": int(cv2.connectedComponents(neg_neg_line.astype(np.uint8), 8)[0] - 1),
               "selected_separator_count": len(rows), "anchors": [a.to_dict() for a in anchors],
               "selected_separators": rows}
    (item_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smooth-window", type=int, default=230)
    parser.add_argument("--min-anchor-separation", type=float, default=40.0)
    parser.add_argument("--max-anchor-distance", type=float, default=70.0)
    parser.add_argument("--max-between-distance", type=float, default=180.0)
    args = parser.parse_args()
    summaries = [process_one(s, args.correction_dir, args.output_dir, args.smooth_window, args.min_anchor_separation, args.max_anchor_distance, args.max_between_distance) for s in SPECS]
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["image", "anchor_count", "raw_separator_component_count", "selected_separator_count"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for item in summaries:
            writer.writerow({"image": Path(item["source_image"]).name, **{key: item[key] for key in fields[1:]}})
    for item in summaries:
        print(f"{Path(item['source_image']).name}: anchors={item['anchor_count']}, raw={item['raw_separator_component_count']}, selected={item['selected_separator_count']}")


if __name__ == "__main__":
    main()
