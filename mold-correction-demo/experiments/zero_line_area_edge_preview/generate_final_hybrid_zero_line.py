"""Generate the final hybrid zero-line review result.

The common correction masks and the case decision are KDT013's rules:

* remove non-correction components smaller than 1% of the part;
* case 1 when the remaining non-correction area is below 40% and split into
  two or more components;
* case 2 otherwise.

Case 1 uses KDT013's offset-contour intersection polygons. Case 2 uses the
preserved outer-zero-point pair search, collision-free routing, enclosure
validation, and route-complexity scoring. This remains a review-only program
and does not modify the production engine or UI server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import generate_adaptive_zero_line_preview as kdt  # noqa: E402
import generate_correction_only_3pct_preview as correction  # noqa: E402
import case2_route_adapter as case2_adapter  # noqa: E402


DEFAULT_CORRECTION_DIR = HERE / "results_correction_only_2pct"
DEFAULT_OUTPUT_DIR = HERE / "results_final_hybrid_zero_line"
DEFAULT_RAW_INPUT_DIR = DEMO_ROOT / "label_removal" / "input"
FINAL_LINE_WIDTH_PX = 4
FINAL_LINE_RGB = (255, 235, 0)
FINAL_LINE_OUTLINE_RGB = (255, 255, 255)
ZERO_POINT_RGB = (45, 240, 80)
ROUTE_REGION_RGB = (255, 90, 210)
CASE2_FINAL_LINE_RGB = (0, 255, 255)


class ZeroLineDetection(NamedTuple):
    """Normalized result returned by the single hybrid-engine entry point."""

    selected_case: int
    method: str
    final_mask: np.ndarray
    method_data: dict[str, Any]
    summary_details: dict[str, Any]


def json_safe(value: Any) -> Any:
    """Convert NumPy-heavy diagnostic structures to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_correction_results(
    specs: list,
    input_dir: Path,
    correction_dir: Path,
    regenerate: bool,
) -> None:
    """Build the shared correction masks when they do not exist yet."""
    for spec in specs:
        summary_path = correction_dir / spec.key / "summary.json"
        if summary_path.exists() and not regenerate:
            continue
        correction.process_one(
            spec,
            input_dir,
            correction_dir,
            correction.CORRECTION_THRESHOLD_MM,
            0.01,
            0.02,
            24,
        )


def load_common_inputs(spec, correction_dir: Path) -> dict[str, Any]:
    source_dir = correction_dir / spec.key
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    image = kdt.imread_rgb(Path(summary["source_image"]))
    legend = kdt.imread_rgb(Path(summary["legend_image"]))
    values, valid = kdt.map_deviation(
        image,
        kdt.extract_color_ramp(legend),
        spec.vmin,
        spec.vmax,
    )
    values = cv2.medianBlur(values, 5)
    part = kdt.detect_part_mask(image)
    mapped = part & valid
    gray = kdt.detect_unmapped_gray(image, part, mapped)
    effective_values = values.copy()
    positive_gray_path = source_dir / "gray_assigned_positive_mask.png"
    negative_gray_path = source_dir / "gray_assigned_negative_mask.png"
    if positive_gray_path.exists() and negative_gray_path.exists():
        gray_positive = kdt.read_mask(positive_gray_path)
        gray_negative = kdt.read_mask(negative_gray_path)
        effective_values[gray_positive] = float(summary["gray_positive_assigned_mm"])
        effective_values[gray_negative] = float(summary["gray_negative_assigned_mm"])
    else:
        effective_values[gray] = float(summary.get("gray_sentinel_mm", 3.01))

    positive = kdt.read_mask(source_dir / "final_positive_correction_mask.png")
    negative = kdt.read_mask(source_dir / "final_negative_correction_mask.png")
    correction_mask = positive | negative
    part_px = int(part.sum())
    raw_zero = part & ~correction_mask
    zero, zero_labels, zero_rows, raw_zero_rows = kdt.filter_components_by_ratio(
        raw_zero,
        part_px,
        kdt.ZERO_COMPONENT_MIN_RATIO,
    )
    zero_ratio = float(zero.sum() / part_px) if part_px else 0.0
    raw_candidates = sorted(DEFAULT_RAW_INPUT_DIR.glob(f"{spec.key}*"))
    if len(raw_candidates) != 1:
        raise FileNotFoundError(
            f"Expected one raw source image for {spec.key}, found {len(raw_candidates)}"
        )
    return {
        "source_dir": source_dir,
        "source_summary": summary,
        "image": image,
        "values": values,
        "effective_values": effective_values,
        "mapped": mapped,
        "gray": gray,
        "part": part,
        "part_px": part_px,
        "positive": positive,
        "negative": negative,
        "correction": correction_mask,
        "raw_zero": raw_zero,
        "raw_zero_rows": raw_zero_rows,
        "zero": zero,
        "zero_labels": zero_labels,
        "zero_rows": zero_rows,
        "zero_ratio": zero_ratio,
        "zero_count": len(zero_rows),
        "raw_source_image": raw_candidates[0],
        "scale_max_mm": float(max(abs(spec.vmin), abs(spec.vmax))),
    }


def select_case(zero_ratio: float, zero_component_count: int) -> int:
    """Return the shared distribution case number."""
    return int(
        not (
            zero_ratio < kdt.CASE1_LIMIT_RATIO
            and zero_component_count > 1
        )
    ) + 1


def run_case1(common: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the existing offset-contour polygon construction unchanged."""
    split_labels, split_rows, neck_cut_mask, split_events = (
        kdt.split_zero_components_at_narrow_necks(
            common["zero"],
            common["part_px"],
            kdt.ZERO_NECK_MAX_BOUNDARY_GAP_PX,
            kdt.ZERO_NECK_MIN_CHILD_RATIO,
            kdt.ZERO_NECK_MIN_CUT_SPACING_PX,
            kdt.ZERO_NECK_CUT_MARGIN_PX,
            kdt.ZERO_NECK_MIN_PROMINENCE_RATIO,
            kdt.ZERO_NECK_ABSOLUTE_WIDTH_OVERRIDE_PX,
        )
    )
    split_labels, split_rows, removed_small_thin_rows = (
        kdt.filter_small_thin_post_neck_regions(
            split_labels,
            split_rows,
            kdt.POST_NECK_SMALL_THIN_MAX_RATIO,
            kdt.POST_NECK_SMALL_THIN_MAX_WIDTH_PX,
        )
    )
    intersection_rows = [
        row
        for row in split_rows
        if row["ratio_of_part"] >= kdt.ZERO_POST_NECK_MIN_RATIO
    ]
    polygon, points, support, contours, rows = kdt.case1_polygons(
        common["zero"],
        common["positive"],
        common["negative"],
        common["part"],
        split_labels,
        intersection_rows,
        kdt.OFFSET_COUNT,
        common["part_px"],
        kdt.CASE1_MIN_POLYGON_RATIO,
        kdt.CASE1_MAX_CORRECTION_INTRUSION_RATIO,
        kdt.ZERO_COMPONENT_EXPANSION_PX,
    )
    before_corner_squaring = polygon.copy()
    polygon, corner_rows = kdt.square_acute_final_polygon_corners(
        polygon,
        rows,
        common["part"],
        kdt.correction_component_masks(common["positive"], common["negative"]),
        kdt.ACUTE_CORNER_MAX_ANGLE_DEG,
        kdt.ACUTE_CORNER_BEVEL_DISTANCE_PX,
        kdt.CASE1_MIN_POLYGON_RATIO,
        kdt.CASE1_MAX_CORRECTION_INTRUSION_RATIO,
        common["part_px"],
    )
    data = {
        "polygon": polygon,
        "polygon_before_corner_squaring": before_corner_squaring,
        "corner_squaring_rows": corner_rows,
        "points": points,
        "expanded_zero_support": support,
        "neck_cut_mask": neck_cut_mask,
        "split_labels": split_labels,
        "split_events": split_events,
        "split_rows": split_rows,
        "intersection_rows": intersection_rows,
        "post_neck_zero_area": split_labels > 0,
        "minimum_zero_component_ratio": kdt.ZERO_COMPONENT_MIN_RATIO,
        "minimum_post_neck_component_ratio": kdt.ZERO_POST_NECK_MIN_RATIO,
        "neck_min_child_ratio": kdt.ZERO_NECK_MIN_CHILD_RATIO,
        "removed_small_thin_rows": removed_small_thin_rows,
        "contours": contours,
        "rows": rows,
    }
    return polygon, data


def resample_closed_curve(points: np.ndarray, spacing: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return points
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    keep = lengths > 1e-6
    points = points[keep]
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    total = float(lengths.sum())
    sample_count = max(8, int(np.ceil(total / max(spacing, 1.0))))
    targets = np.linspace(0.0, total, sample_count, endpoint=False)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(points) - 1)
    ratios = (targets - cumulative[indices]) / np.maximum(lengths[indices], 1e-6)
    return points[indices] + (following[indices] - points[indices]) * ratios[:, None]


def outer_contour_geometry(part: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(
        part.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("Part outer contour was not detected")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    spacing = max(3.0, min(part.shape) * 0.0045)
    points = resample_closed_curve(contour, spacing)
    silhouette = np.zeros(part.shape, dtype=np.uint8)
    cv2.fillPoly(silhouette, [contour.astype(np.int32)], 255)
    return points, silhouette


def _fallback_sign_crossings(
    contour_points: np.ndarray,
    values: np.ndarray,
    mapped: np.ndarray,
    part: np.ndarray,
) -> list[tuple[int, int, int]]:
    height, width = values.shape
    rounded = np.rint(contour_points).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    samples = np.empty(len(rounded), dtype=np.float64)
    search_radius = max(8, int(round(min(values.shape) * 0.012)))
    for index, (x, y) in enumerate(rounded):
        x0, x1 = max(0, x - search_radius), min(width, x + search_radius + 1)
        y0, y1 = max(0, y - search_radius), min(height, y + search_radius + 1)
        local_mapped = mapped[y0:y1, x0:x1]
        local_part = part[y0:y1, x0:x1]
        candidate_y, candidate_x = np.where(local_mapped)
        if candidate_x.size == 0:
            candidate_y, candidate_x = np.where(local_part)
        if candidate_x.size == 0:
            samples[index] = float(values[y, x])
            continue
        distances = (candidate_x + x0 - x) ** 2 + (candidate_y + y0 - y) ** 2
        nearest = np.argsort(distances)[: min(7, len(distances))]
        samples[index] = float(
            np.median(values[candidate_y[nearest] + y0, candidate_x[nearest] + x0])
        )
    if len(samples) >= 9:
        weights = np.asarray([1, 4, 10, 16, 19, 16, 10, 4, 1], np.float64) / 81.0
        samples = np.convolve(
            np.concatenate((samples[-4:], samples, samples[:4])),
            weights,
            mode="valid",
        )
    candidates: list[tuple[int, int, int]] = []
    signs = np.sign(samples)
    nonzero = np.flatnonzero(signs)
    if nonzero.size:
        for index in np.flatnonzero(signs == 0):
            circular_distance = np.minimum(
                np.abs(nonzero - index), len(signs) - np.abs(nonzero - index)
            )
            signs[index] = signs[nonzero[int(np.argmin(circular_distance))]]
    for index in range(len(samples)):
        following = (index + 1) % len(samples)
        if signs[index] * signs[following] < 0.0:
            x, y = rounded[index]
            if any(np.hypot(x - px, y - py) < 7.0 for px, py, _ in candidates):
                continue
            candidates.append((int(x), int(y), index))
    return candidates


def boundary_zero_points(
    values: np.ndarray,
    part: np.ndarray,
    mapped: np.ndarray,
    contour_points: np.ndarray,
) -> tuple[list[dict], list[Any], str]:
    """Map KDT013 boundary sign-transition anchors to case-2 contour indices."""
    anchors = kdt.find_boundary_anchors(values, part, 230, 40.0)
    method = "kdt_boundary_sign_transition_smooth_230"
    if len(anchors) < 2:
        anchors = kdt.find_boundary_anchors(values, part, 21, 40.0)
        method = "kdt_boundary_sign_transition_smooth_21_fallback"

    selected: list[tuple[int, int, int]] = []
    for anchor in anchors:
        distances = np.linalg.norm(
            contour_points - np.asarray((anchor.x, anchor.y), dtype=np.float64),
            axis=1,
        )
        sample_index = int(np.argmin(distances))
        x, y = np.rint(contour_points[sample_index]).astype(np.int32)
        if any(index == sample_index for _, _, index in selected):
            continue
        selected.append((int(x), int(y), sample_index))

    if len(selected) < 2:
        selected = _fallback_sign_crossings(contour_points, values, mapped, part)
        method = "nearest_interior_mapped_contour_sign_crossing_fallback"
    points = [
        {
            "label": f"Z{index}",
            "point": (float(x), float(y)),
            "type": "outer_boundary_sign_transition",
            "contour": "outer",
            "sample_index": int(sample_index),
        }
        for index, (x, y, sample_index) in enumerate(selected, start=1)
    ]
    if len(points) < 2:
        raise ValueError("Case-2 route selection requires at least two outer zero points")
    return points, anchors, method


def routes_to_mask(selections: list[dict], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for selection in selections:
        points = np.asarray(
            selection["closure_validation"]["route"]["path_points"],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(
                mask,
                [points.reshape(-1, 1, 2)],
                False,
                255,
                FINAL_LINE_WIDTH_PX,
                cv2.LINE_8,
            )
    return mask


def simplified_team_rows(selections: list[dict]) -> list[dict]:
    rows = []
    for selection in selections:
        region = selection["region"]
        route = selection["closure_validation"]["route"]
        rows.append(
            {
                "region_label": region["label"],
                "source_region_labels": region.get("source_region_labels", [region["label"]]),
                "attached_islands": region.get("attached_islands", []),
                "area_px": int(region["area_px"]),
                "centroid": [round(float(value), 3) for value in region["centroid"]],
                "contour_contact_mode": selection["mode"],
                "selected_zero_points": [
                    {
                        "label": item["label"],
                        "point": [round(float(value), 3) for value in item["point"]],
                        "sample_index": int(item["sample_index"]),
                        "direction": item["direction"],
                    }
                    for item in selection["selected"]
                ],
                "route": {
                    "method": route["routing_method"],
                    "path_points": route["path_points"],
                    "path_length_pixels": route["path_length_pixels"],
                    "path_bend_count": route["path_bend_count"],
                    "zero_clearance_fallback_used": route["zero_clearance_fallback_used"],
                },
                "closure_validation": {
                    "valid": selection["closure_validation"]["valid"],
                    "reason": selection["closure_validation"]["reason"],
                    "target_coverage_ratio": selection["closure_validation"][
                        "target_coverage_ratio"
                    ],
                    "product_component_count_after_route": selection[
                        "closure_validation"
                    ]["product_component_count_after_route"],
                },
                "selected_weighted_route_cost": selection[
                    "selected_weighted_route_cost"
                ],
                "pair_attempt_count": selection["pair_attempt_count"],
            }
        )
    return rows


def run_case2(common: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the complete preserved case-2 stages after the shared decision."""
    encoded = np.fromfile(str(common["raw_source_image"]), dtype=np.uint8)
    original_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {common['raw_source_image']}")
    adapted = case2_adapter.run_original_case2_pipeline(
        original_bgr=original_bgr,
        scale_max_mm=common["scale_max_mm"],
    )
    selections = adapted["selections"]
    final_mask = routes_to_mask(selections, common["part"].shape) > 0
    return final_mask, {
        "contour_points": adapted["contour_points"],
        "outer_silhouette": adapted["outer_silhouette"] > 0,
        "zero_points": adapted["zero_points"],
        "anchors": [],
        "zero_point_method": "case2_original_contour_graph",
        "team_cleaned_image": cv2.cvtColor(
            adapted["cleaned_bgr"], cv2.COLOR_BGR2RGB
        ),
        "team_merged_correction": adapted["merged_correction_mask"] > 0,
        "team_tolerance_mm": adapted["tolerance_mm"],
        "team_merge_details": adapted["merge_details"],
        "contact_radius_px": adapted["contact_radius_px"],
        "route_clearance_px": adapted["route_clearance_px"],
        "source_correction_region_count": adapted[
            "source_correction_region_count"
        ],
        "minimum_route_length_px": adapted["minimum_route_length_px"],
        "rejected_routes": adapted["rejected_routes"],
        "selections": selections,
        "rows": simplified_team_rows(selections),
    }


def draw_team_route_view(
    image: np.ndarray,
    case2: dict[str, Any],
    final_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    construction = image.copy()
    contour = np.rint(case2["contour_points"]).astype(np.int32)
    cv2.polylines(
        construction,
        [contour.reshape(-1, 1, 2)],
        True,
        kdt.CONTOUR_RGB,
        2,
        cv2.LINE_AA,
    )
    for item in case2["zero_points"]:
        point = tuple(np.rint(item["point"]).astype(np.int32).tolist())
        cv2.circle(construction, point, 7, ZERO_POINT_RGB, -1, cv2.LINE_AA)
        cv2.circle(construction, point, 7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            construction,
            item["label"],
            (point[0] + 8, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    for selection in case2["selections"]:
        for region_contour in selection["region"]["contours"]:
            cv2.drawContours(
                construction,
                [region_contour],
                -1,
                ROUTE_REGION_RGB,
                2,
                cv2.LINE_AA,
            )
    kdt.draw_line(
        construction,
        final_mask.astype(np.uint8) * 255,
        FINAL_LINE_RGB,
        FINAL_LINE_WIDTH_PX,
    )
    final = image.copy()
    for selection in case2["selections"]:
        points = np.asarray(
            selection["closure_validation"]["route"]["path_points"],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(
            final,
            [points],
            False,
            FINAL_LINE_OUTLINE_RGB,
            FINAL_LINE_WIDTH_PX + 3,
            cv2.LINE_AA,
        )
        cv2.polylines(
            final,
            [points],
            False,
            CASE2_FINAL_LINE_RGB,
            FINAL_LINE_WIDTH_PX,
            cv2.LINE_AA,
        )
    return construction, final


def build_case2_board(
    common: dict[str, Any],
    case2: dict[str, Any],
    final_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    correction_view = case2["team_cleaned_image"].copy()
    kdt.blend_mask(
        correction_view,
        case2["team_merged_correction"],
        ROUTE_REGION_RGB,
        0.52,
    )

    distribution = common["image"].copy()
    kdt.blend_mask(distribution, common["zero"], kdt.ZERO_CANDIDATE_RGB, 0.28)
    kdt.draw_region_numbers(
        distribution,
        common["zero_labels"],
        common["zero_rows"],
        "C",
    )
    construction, final = draw_team_route_view(
        case2["team_cleaned_image"], case2, final_mask
    )
    panels = [
        kdt.add_title(
            kdt.fit_panel(correction_view),
            "1. Case-2 correction regions",
            f"original stage 04-05; threshold=+/-{case2['team_tolerance_mm']:.1f} mm",
        ),
        kdt.add_title(
            kdt.fit_panel(distribution),
            "2. Shared case decision",
            f"zero={common['zero_ratio'] * 100:.2f}% / components={common['zero_count']} / case 2",
        ),
        kdt.add_title(
            kdt.fit_panel(construction),
            "3. Case-2 route selection",
            f"zero points={len(case2['zero_points'])}; validated routes={len(case2['rows'])}",
        ),
        kdt.add_title(
            kdt.fit_panel(final),
            "4. Final hybrid zero lines",
            "best collision-free closed-enclosure route per logical correction region",
        ),
    ]
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))), final


def detect_zero_line(common: dict[str, Any]) -> ZeroLineDetection:
    """Run the common decision and dispatch to case 1 or case 2 internally."""
    selected_case = select_case(common["zero_ratio"], common["zero_count"])
    if selected_case == 1:
        final_mask, case1 = run_case1(common)
        return ZeroLineDetection(
            selected_case=selected_case,
            method="case1_kdt_offset_contour_polygon",
            final_mask=final_mask,
            method_data=case1,
            summary_details={
                "owner": "KDT013",
                "polygon_regions": case1["rows"],
                "neck_split_events": case1["split_events"],
                "corner_squaring": case1["corner_squaring_rows"],
            },
        )

    final_mask, case2 = run_case2(common)
    return ZeroLineDetection(
        selected_case=selected_case,
        method="case2_original_routes_via_adapter",
        final_mask=final_mask,
        method_data=case2,
        summary_details={
            "owner": "case2_original_pipeline",
            "selector_source": "unmodified team stages 01-06",
            "integration": "KDT013 case decision; preserved complete case-2 pipeline",
            "zero_point_method": case2["zero_point_method"],
            "contact_radius_px": case2["contact_radius_px"],
            "route_clearance_px": case2["route_clearance_px"],
            "minimum_route_length_px": case2["minimum_route_length_px"],
            "team_correction_tolerance_mm": case2["team_tolerance_mm"],
            "rejected_routes": case2["rejected_routes"],
            "source_correction_region_count": case2[
                "source_correction_region_count"
            ],
            "routes": case2["rows"],
        },
    )


def process_one(spec, correction_dir: Path, output_dir: Path) -> dict[str, Any]:
    common = load_common_inputs(spec, correction_dir)
    detection = detect_zero_line(common)
    selected_case = detection.selected_case
    final_mask = detection.final_mask
    method = detection.method
    method_details = detection.summary_details
    if selected_case == 1:
        case1 = detection.method_data
        board, overlay = kdt.build_board(
            common["image"],
            common["positive"],
            common["negative"],
            common["zero"],
            "case1_contour_polygon",
            case1,
            None,
            common["zero_ratio"],
            common["zero_count"],
        )
    else:
        case2 = detection.method_data
        board, overlay = build_case2_board(common, case2, final_mask)

    item_dir = output_dir / spec.key
    kdt.imwrite_rgb(item_dir / "review_board.png", board)
    kdt.imwrite_rgb(item_dir / "final_zero_line_overlay.png", overlay)
    kdt.imwrite_gray(
        item_dir / "raw_zero_line_source_area_before_1pct_filter_mask.png",
        common["raw_zero"].astype(np.uint8) * 255,
    )
    kdt.imwrite_gray(
        item_dir / "zero_line_source_area_after_1pct_filter_mask.png",
        common["zero"].astype(np.uint8) * 255,
    )
    kdt.imwrite_gray(
        item_dir / "final_zero_line_mask.png",
        final_mask.astype(np.uint8) * 255,
    )
    summary = {
        "format_version": 1,
        "source_image": common["source_summary"]["source_image"],
        "source_correction_result": str(common["source_dir"]),
        "shared_rules": {
            "correction_threshold_mm": common["source_summary"].get(
                "correction_threshold_mm"
            ),
            "minimum_correction_area_ratio_exclusive": common[
                "source_summary"
            ].get("minimum_correction_area_ratio_exclusive"),
            "minimum_zero_component_ratio": kdt.ZERO_COMPONENT_MIN_RATIO,
            "case1_zero_area_limit_ratio_exclusive": kdt.CASE1_LIMIT_RATIO,
            "case1_requires_multiple_components": True,
        },
        "part_px": common["part_px"],
        "raw_zero_area_px": int(common["raw_zero"].sum()),
        "zero_area_px_after_1pct_filter": int(common["zero"].sum()),
        "zero_area_ratio_of_part": common["zero_ratio"],
        "zero_connected_component_count": common["zero_count"],
        "selected_case": selected_case,
        "method": method,
        "final_zero_line_px": int(final_mask.sum()),
        "method_details": method_details,
        "outputs": {
            "review_board": "review_board.png",
            "final_overlay": "final_zero_line_overlay.png",
            "final_mask": "final_zero_line_mask.png",
        },
    }
    (item_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=correction.DEFAULT_INPUT)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--spec",
        action="append",
        choices=[spec.key for spec in kdt.SPECS],
        help="Process only the selected scan key; may be repeated",
    )
    parser.add_argument(
        "--regenerate-corrections",
        action="store_true",
        help="Rebuild the shared correction masks even when summaries exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [spec for spec in kdt.SPECS if not args.spec or spec.key in args.spec]
    ensure_correction_results(
        specs,
        args.input_dir.resolve(),
        args.correction_dir.resolve(),
        args.regenerate_corrections,
    )
    summaries = [
        process_one(spec, args.correction_dir.resolve(), args.output_dir.resolve())
        for spec in specs
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summaries), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for summary in summaries:
        print(
            f"{Path(summary['source_image']).name}: "
            f"case={summary['selected_case']}, method={summary['method']}, "
            f"zero={summary['zero_area_ratio_of_part'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
