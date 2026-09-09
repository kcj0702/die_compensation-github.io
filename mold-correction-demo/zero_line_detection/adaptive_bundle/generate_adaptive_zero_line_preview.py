"""Generate review-only adaptive zero-line areas/lines from 2% correction masks.

Case 1: the non-correction area is below 40% of the part and disconnected.
Six equally spaced inward part contours (outer, four offsets, innermost) are
intersected with correction boundaries.  Three or more intersection points for
one zero component are joined with straight polygon edges and filled.

Case 2: all other images.  A Voronoi separator between every final correction
component and the background is restricted to the neutral deviation corridor,
prefers nearby structural edges, and only keeps lines that reach the external
part boundary or form a closed loop.

The production engine is not imported or modified.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from generate_preview import (
    SPECS,
    ScanSpec,
    add_title,
    blend_mask,
    detect_part_mask,
    extract_color_ramp,
    fit_panel,
    imread_rgb,
    imwrite_gray,
    imwrite_rgb,
    map_deviation,
)
from generate_between_signs_preview import (
    NEG_RGB,
    POS_RGB,
    ZERO_CANDIDATE_RGB,
    detect_unmapped_gray,
    draw_line,
    mask_boundary,
)
from generate_correction_split_preview import read_mask, skeletonize
from generate_edge_separator_1p5pct_preview import detect_structural_edges


HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))
from zero_line_detection.zero_boundary import find_boundary_anchors  # noqa: E402


DEFAULT_CORRECTION_RESULTS = HERE / "results_correction_only_2pct"
DEFAULT_OUTPUT = HERE / "results_adaptive_zero_line_2pct"
CASE1_LIMIT_RATIO = 0.40
CASE1_MIN_POLYGON_RATIO = 0.005
CASE1_MAX_CORRECTION_INTRUSION_RATIO = 0.10
NEUTRAL_LIMIT_MM = 0.5
NON_NEUTRAL_ALLOWANCE_PX = 30
OFFSET_COUNT = 4
ZERO_COMPONENT_EXPANSION_PX = 0
ZERO_COMPONENT_MIN_RATIO = 0.01
ZERO_POST_NECK_MIN_RATIO = 0.0
ZERO_NECK_MAX_BOUNDARY_GAP_PX = 23.0
ZERO_NECK_MIN_CUT_SPACING_PX = 30
ZERO_NECK_MIN_CHILD_RATIO = 0.0
ZERO_NECK_CUT_MARGIN_PX = 8
ZERO_NECK_MIN_PROMINENCE_RATIO = 0.55
ZERO_NECK_ABSOLUTE_WIDTH_OVERRIDE_PX = 18.0
POST_NECK_SMALL_THIN_MAX_RATIO = 0.01
POST_NECK_SMALL_THIN_MAX_WIDTH_PX = 30.0
ACUTE_CORNER_MAX_ANGLE_DEG = 70.0
ACUTE_CORNER_BEVEL_DISTANCE_PX = 12.0
EDGE_SNAP_RADIUS_PX = 12
CASE2_MIN_LINE_PX = 100
NORMAL_GATEWAY_DEPTH_PX = 12
POLYGON_RGB = (20, 95, 230)
LINE_RGB = (255, 235, 0)
EDGE_LINE_RGB = (255, 115, 215)
ANCHOR_RGB = (45, 240, 80)
CONTOUR_RGB = (245, 45, 45)
POS_FILL_RGB = (255, 115, 65)
NEG_FILL_RGB = (55, 135, 255)


def connected_component_rows(mask: np.ndarray, part_px: int) -> tuple[np.ndarray, list[dict]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    rows = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        rows.append(
            {
                "component_id": component_id,
                "area_px": area,
                "ratio_of_part": float(area / part_px) if part_px else 0.0,
                "centroid": [float(v) for v in centroids[component_id]],
            }
        )
    rows.sort(key=lambda row: row["area_px"], reverse=True)
    return labels, rows


def filter_components_by_ratio(
    mask: np.ndarray,
    part_px: int,
    minimum_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    source_labels, source_rows = connected_component_rows(mask, part_px)
    kept_ids = [
        row["component_id"]
        for row in source_rows
        if row["ratio_of_part"] >= minimum_ratio
    ]
    filtered = np.isin(source_labels, kept_ids)
    filtered_labels, filtered_rows = connected_component_rows(filtered, part_px)
    return filtered, filtered_labels, filtered_rows, source_rows


def split_zero_components_at_narrow_necks(
    zero_area: np.ndarray,
    part_px: int,
    max_boundary_gap_px: float,
    min_child_ratio: float,
    min_cut_spacing_px: int,
    cut_margin_px: int,
    min_prominence_ratio: float = 0.0,
    absolute_width_override_px: float = 0.0,
) -> tuple[np.ndarray, list[dict], np.ndarray, list[dict]]:
    """Partition zero components at short, locally narrow boundary-to-boundary necks."""
    original_labels, original_rows = connected_component_rows(zero_area, part_px)
    output_labels = np.zeros(zero_area.shape, np.int32)
    cut_mask = np.zeros_like(zero_area, dtype=bool)
    split_events: list[dict] = []
    next_component_id = 1
    minimum_child_px = int(np.floor(part_px * min_child_ratio)) + 1
    core_minimum_boundary_distance_px = max_boundary_gap_px / 2.0

    def append_region(region: np.ndarray, original_row: dict, split_index: int) -> None:
        nonlocal next_component_id
        output_labels[region] = next_component_id
        ys, xs = np.where(region)
        area = int(len(xs))
        rows.append(
            {
                "component_id": next_component_id,
                "original_component_id": original_row["component_id"],
                "split_index": split_index,
                "area_px": area,
                "ratio_of_part": float(area / part_px) if part_px else 0.0,
                "centroid": [float(xs.mean()), float(ys.mean())],
            }
        )
        next_component_id += 1

    rows: list[dict] = []
    for original_row in original_rows:
        original_id = original_row["component_id"]
        component = original_labels == original_id
        local_half_width = cv2.distanceTransform(
            component.astype(np.uint8), cv2.DIST_L2, 5
        )
        distance_core = local_half_width > core_minimum_boundary_distance_px
        seed_count, seed_labels, seed_stats, seed_centroids = cv2.connectedComponentsWithStats(
            distance_core.astype(np.uint8), connectivity=8
        )
        seed_ids = [
            seed_id
            for seed_id in range(1, seed_count)
            if int(seed_stats[seed_id, cv2.CC_STAT_AREA]) >= 4
        ]
        seed_ids.sort(key=lambda seed_id: int(seed_stats[seed_id, cv2.CC_STAT_AREA]), reverse=True)

        # Suppress multiple cuts packed into the same local neck neighborhood.
        spaced_seed_ids: list[int] = []
        for seed_id in seed_ids:
            center = seed_centroids[seed_id]
            if all(
                float(np.linalg.norm(center - seed_centroids[kept_id])) >= min_cut_spacing_px
                for kept_id in spaced_seed_ids
            ):
                spaced_seed_ids.append(seed_id)
        seed_ids = spaced_seed_ids
        if len(seed_ids) < 2:
            append_region(component, original_row, 0)
            continue

        seed_mask = np.zeros_like(component, np.uint8)
        for seed_id in seed_ids:
            seed_mask[seed_labels == seed_id] = 1
        _, nearest_seed = cv2.distanceTransformWithLabels(
            (1 - seed_mask).astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_CCOMP,
        )
        nearest_ids = [int(v) for v in np.unique(nearest_seed[component]) if int(v) > 0]
        valid_ids = [
            seed_id
            for seed_id in nearest_ids
            if int((component & (nearest_seed == seed_id)).sum()) >= minimum_child_px
        ]
        if len(valid_ids) < 2:
            append_region(component, original_row, 0)
            continue

        # Recompute the Voronoi partition after rejecting undersized children.
        retained_seeds = np.isin(nearest_seed, valid_ids) & distance_core
        _, partition = cv2.distanceTransformWithLabels(
            (~retained_seeds).astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_CCOMP,
        )
        partition_ids = [int(v) for v in np.unique(partition[component]) if int(v) > 0]
        regions = [component & (partition == partition_id) for partition_id in partition_ids]
        regions = [region for region in regions if int(region.sum()) >= minimum_child_px]
        if len(regions) < 2 or sum(int(region.sum()) for region in regions) != int(component.sum()):
            append_region(component, original_row, 0)
            continue

        seam_center = np.zeros_like(component, dtype=bool)
        for first in range(len(regions)):
            dilated = cv2.dilate(regions[first].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            for second in range(first + 1, len(regions)):
                seam_center |= dilated & regions[second]
        seam_count, seam_labels, _, _ = cv2.connectedComponentsWithStats(
            seam_center.astype(np.uint8), connectivity=8
        )
        component_skeleton = skeletonize(component) if min_prominence_ratio > 0.0 else None
        if component_skeleton is not None:
            skeleton_y, skeleton_x = np.where(component_skeleton)
            skeleton_full_width = 2.0 * local_half_width[skeleton_y, skeleton_x]
        accepted_seam_center = np.zeros_like(component, dtype=bool)
        candidate_gap_rows = []
        for seam_id in range(1, seam_count):
            candidate = seam_labels == seam_id
            estimated_candidate_gap = float(2.0 * local_half_width[candidate].max())
            neck_minimum_width = estimated_candidate_gap
            reference_width = estimated_candidate_gap
            prominence_ratio = 1.0
            salience_passed = True
            if component_skeleton is not None and len(skeleton_x):
                candidate_y, candidate_x = np.where(candidate)
                center_x = float(candidate_x.mean())
                center_y = float(candidate_y.mean())
                radial_distance = np.hypot(
                    skeleton_x.astype(np.float32) - center_x,
                    skeleton_y.astype(np.float32) - center_y,
                )
                near = radial_distance <= 12.0
                annulus = (radial_distance >= 25.0) & (radial_distance <= 55.0)
                if np.any(near):
                    neck_minimum_width = float(skeleton_full_width[near].min())
                if np.any(annulus):
                    reference_width = float(np.percentile(skeleton_full_width[annulus], 75.0))
                prominence_ratio = max(
                    0.0,
                    (reference_width - neck_minimum_width) / max(reference_width, 1e-6),
                )
                salience_passed = (
                    prominence_ratio >= min_prominence_ratio
                    or (
                        absolute_width_override_px > 0.0
                        and neck_minimum_width <= absolute_width_override_px
                    )
                )
            accepted_candidate = (
                estimated_candidate_gap <= max_boundary_gap_px and salience_passed
            )
            candidate_gap_rows.append(
                {
                    "candidate_id": seam_id,
                    "estimated_boundary_gap_px": estimated_candidate_gap,
                    "skeleton_minimum_width_px": neck_minimum_width,
                    "surrounding_reference_width_px": reference_width,
                    "width_prominence_ratio": prominence_ratio,
                    "salience_rule_passed": salience_passed,
                    "accepted": accepted_candidate,
                    "centerline_px": int(candidate.sum()),
                }
            )
            if accepted_candidate:
                accepted_seam_center |= candidate
        if not np.any(accepted_seam_center):
            append_region(component, original_row, 0)
            continue

        cut_size = max(1, cut_margin_px * 2 + 1)
        cut_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cut_size, cut_size))
        seam = cv2.dilate(accepted_seam_center.astype(np.uint8), cut_kernel) > 0
        seam &= component
        remaining = component & ~seam
        child_count, child_labels, child_stats, _ = cv2.connectedComponentsWithStats(
            remaining.astype(np.uint8), connectivity=8
        )
        trimmed_regions = [
            child_labels == child_id
            for child_id in range(1, child_count)
            if int(child_stats[child_id, cv2.CC_STAT_AREA]) >= minimum_child_px
        ]
        if len(trimmed_regions) < 2:
            append_region(component, original_row, 0)
            continue

        cut_mask |= seam
        accepted_gaps = [
            row["estimated_boundary_gap_px"]
            for row in candidate_gap_rows
            if row["accepted"]
        ]
        child_ids = []
        for split_index, region in enumerate(trimmed_regions, start=1):
            child_ids.append(next_component_id)
            append_region(region, original_row, split_index)
        split_events.append(
            {
                "original_component_id": original_id,
                "child_component_ids": child_ids,
                "child_area_px": [int(region.sum()) for region in trimmed_regions],
                "estimated_boundary_gap_px": max(accepted_gaps),
                "neck_candidates": candidate_gap_rows,
                "accepted_neck_candidate_count": sum(
                    row["accepted"] for row in candidate_gap_rows
                ),
                "rejected_neck_candidate_count": sum(
                    not row["accepted"] for row in candidate_gap_rows
                ),
                "distance_core_minimum_boundary_distance_px": (
                    core_minimum_boundary_distance_px
                ),
                "cut_margin_px": cut_margin_px,
                "cut_px": int(seam.sum()),
            }
        )

    rows.sort(key=lambda row: row["area_px"], reverse=True)
    return output_labels, rows, cut_mask, split_events


def filter_small_thin_post_neck_regions(
    labels: np.ndarray,
    rows: list[dict],
    maximum_ratio: float,
    maximum_full_width_px: float,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Trial-only stability filter for fragments that are both small and thin."""
    if maximum_ratio <= 0.0 or maximum_full_width_px <= 0.0:
        return labels, rows, []
    filtered_labels = labels.copy()
    retained_rows: list[dict] = []
    removed_rows: list[dict] = []
    for row in rows:
        component = labels == row["component_id"]
        half_width = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        maximum_width = float(2.0 * half_width.max())
        enriched = {**row, "maximum_internal_width_px": maximum_width}
        remove = (
            row["ratio_of_part"] < maximum_ratio
            and maximum_width < maximum_full_width_px
        )
        enriched["small_thin_filter_removed"] = remove
        if remove:
            filtered_labels[component] = 0
            removed_rows.append(enriched)
        else:
            retained_rows.append(enriched)
    return filtered_labels, retained_rows, removed_rows


def nested_part_contours(part: np.ndarray, offset_count: int) -> tuple[list[np.ndarray], list[float]]:
    """Return outer, two offsets, and dominant-inner contour; ignore small holes."""
    outside, holes = split_outer_and_hole_background(part)
    count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), 8
    )
    if count <= 1:
        # No dominant inner contour: fall back to equally spaced erosion-depth
        # contours of the hole-filled outer silhouette.
        external, _ = cv2.findContours(part.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        working = np.zeros_like(part, np.uint8)
        if external:
            cv2.drawContours(working, external, -1, 1, thickness=cv2.FILLED)
        distance = cv2.distanceTransform(working, cv2.DIST_L2, 5)
        depths = np.linspace(1.0, max(float(distance.max()) * 0.92, 1.0), offset_count + 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        contour_masks = [
            (cv2.morphologyEx((distance >= depth).astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0)
            & (working > 0)
            for depth in depths
        ]
        return contour_masks, [float(v) for v in depths]

    dominant_id = 1 + int(np.argmax(hole_stats[1:, cv2.CC_STAT_AREA]))
    dominant_inner = hole_labels == dominant_id
    external, _ = cv2.findContours(part.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouette = np.zeros_like(part, np.uint8)
    cv2.drawContours(silhouette, external, -1, 1, thickness=cv2.FILLED)
    # Fill all small holes and retain only the dominant inner opening.
    working = (silhouette > 0) & ~dominant_inner
    outer_boundary = working & (cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    inner_boundary = working & (cv2.dilate(dominant_inner.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    distance_outer = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 5)
    distance_inner = cv2.distanceTransform((~dominant_inner).astype(np.uint8), cv2.DIST_L2, 5)
    coordinate = distance_outer / np.maximum(distance_outer + distance_inner, 1e-6)
    contour_masks = [outer_boundary]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ratios = np.linspace(0.0, 1.0, offset_count + 2)
    for ratio in ratios[1:-1]:
        inner_side = ((coordinate >= ratio) & working).astype(np.uint8)
        level = cv2.morphologyEx(inner_side, cv2.MORPH_GRADIENT, kernel) > 0
        level &= working & (distance_outer > 2.0) & (distance_inner > 2.0)
        contour_masks.append(skeletonize(level))
    contour_masks.append(inner_boundary)
    return contour_masks, [float(v) for v in ratios]


def clustered_representatives(points_mask: np.ndarray) -> list[tuple[int, int]]:
    if not np.any(points_mask):
        return []
    joined = cv2.dilate(
        points_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    count, labels = cv2.connectedComponents(joined, connectivity=8)
    points_y, points_x = np.where(points_mask)
    representatives = []
    for component_id in range(1, count):
        selected = labels[points_y, points_x] == component_id
        if not np.any(selected):
            continue
        xs = points_x[selected]
        ys = points_y[selected]
        cx, cy = float(xs.mean()), float(ys.mean())
        nearest = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        representatives.append((int(xs[nearest]), int(ys[nearest])))
    return representatives


def deduplicate_points(points: list[tuple[int, int]], min_distance: float = 10.0) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    for point in points:
        if all(np.hypot(point[0] - other[0], point[1] - other[1]) >= min_distance for other in kept):
            kept.append(point)
    return kept


def correction_component_masks(
    positive: np.ndarray,
    negative: np.ndarray,
) -> list[tuple[str, int, np.ndarray, int]]:
    components = []
    for sign, mask in (("positive", positive), ("negative", negative)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            components.append((sign, component_id, labels == component_id, area))
    return components


def polygon_self_intersects(polygon: np.ndarray) -> bool:
    """Return True when non-adjacent polygon edges intersect."""
    polygon = np.asarray(polygon, np.int64)
    count = len(polygon)
    if count < 4:
        return False

    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
        return int((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
        return (
            min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        )

    def intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        ab_c, ab_d = cross(a, b, c), cross(a, b, d)
        cd_a, cd_b = cross(c, d, a), cross(c, d, b)
        if ((ab_c > 0) != (ab_d > 0)) and ((cd_a > 0) != (cd_b > 0)):
            return True
        return (
            (ab_c == 0 and on_segment(a, b, c))
            or (ab_d == 0 and on_segment(a, b, d))
            or (cd_a == 0 and on_segment(c, d, a))
            or (cd_b == 0 and on_segment(c, d, b))
        )

    for first in range(count):
        a, b = polygon[first], polygon[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first or second == (first + 1) % count:
                continue
            if first == 0 and second == count - 1:
                continue
            c, d = polygon[second], polygon[(second + 1) % count]
            if intersects(a, b, c, d):
                return True
    return False


def rasterize_polygon(polygon: np.ndarray, part: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize an ordered, possibly concave polygon without a convex hull."""
    ordered = []
    seen = set()
    for point in polygon.astype(np.int32):
        key = (int(point[0]), int(point[1]))
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    polygon = np.asarray(ordered, np.int32)
    local = np.zeros_like(part, np.uint8)
    if len(polygon) >= 3:
        polygon = cv2.approxPolyDP(
            polygon.reshape(-1, 1, 2), epsilon=2.0, closed=True
        ).reshape(-1, 2)
        if len(polygon) >= 3:
            cv2.fillPoly(local, [polygon.reshape(-1, 1, 2)], 1)
    return (local > 0) & part, polygon


def offset_interval_pairs(
    contour: np.ndarray,
    component: np.ndarray,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int], int]], float | None]:
    """Return exact offset-line/zero-region-boundary intersection pairs."""
    # Only the part of the offset line that is actually inside this zero
    # component is used. Its segment endpoints are the raster line-line
    # intersections on the zero side; no distance-radius substitute is used.
    on_zero = skeletonize(contour & component)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        on_zero.astype(np.uint8), connectivity=8
    )
    eroded = cv2.erode(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    zero_boundary = component & ~eroded
    boundary_y, boundary_x = np.where(zero_boundary)
    boundary_coords = np.column_stack([boundary_x, boundary_y]).astype(np.int32)
    intervals = []
    for component_id in range(1, count):
        segment = labels == component_id
        segment_px = int(stats[component_id, cv2.CC_STAT_AREA])
        neighbors = cv2.filter2D(segment.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
        ys, xs = np.where(segment & (neighbors <= 2))
        segment_y, segment_x = np.where(segment)
        segment_coords = np.column_stack([segment_x, segment_y]).astype(np.int32)
        if len(xs) >= 2:
            coords = np.column_stack([xs, ys]).astype(np.int32)
        elif len(xs) == 1:
            endpoint = np.asarray([[xs[0], ys[0]]], np.int32)
            if len(segment_coords) >= 2:
                delta = segment_coords - endpoint[0]
                support = segment_coords[int(np.argmax(np.sum(delta ** 2, axis=1)))]
            elif len(boundary_coords) >= 2:
                delta = boundary_coords - endpoint[0]
                distances = np.sum(delta ** 2, axis=1)
                distances[distances == 0] = np.iinfo(np.int32).max
                support = boundary_coords[int(np.argmin(distances))]
            else:
                continue
            coords = np.vstack([endpoint, support])
        else:
            coords = segment_coords
        if len(coords) < 2:
            continue
        delta = coords[:, None, :] - coords[None, :, :]
        distances = np.sum(delta.astype(np.float32) ** 2, axis=2)
        first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
        p0 = tuple(int(v) for v in coords[first])
        p1 = tuple(int(v) for v in coords[second])
        if p0 != p1 and component[p0[1], p0[0]] and component[p1[1], p1[0]]:
            intervals.append((p0, p1, segment_px))
    intervals.sort(key=lambda interval: interval[2], reverse=True)
    return intervals, 0.0 if intervals else None


def orient_interval(
    interval: tuple[tuple[int, int], tuple[int, int], int],
    previous: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], float]:
    p0, p1, _ = interval
    direct = np.hypot(p0[0] - previous[0][0], p0[1] - previous[0][1]) + np.hypot(
        p1[0] - previous[1][0], p1[1] - previous[1][1]
    )
    crossed = np.hypot(p1[0] - previous[0][0], p1[1] - previous[0][1]) + np.hypot(
        p0[0] - previous[1][0], p0[1] - previous[1][1]
    )
    if crossed < direct:
        return p1, p0, float(crossed)
    return p0, p1, float(direct)


def circular_interval_angle(
    interval: tuple[tuple[int, int], tuple[int, int], int],
    center: tuple[float, float],
) -> float:
    angles = [
        np.arctan2(point[1] - center[1], point[0] - center[0])
        for point in interval[:2]
    ]
    return float(np.arctan2(sum(np.sin(angles)), sum(np.cos(angles))))


def circular_angle_distance(first: float, second: float) -> float:
    return float(abs(np.arctan2(np.sin(first - second), np.cos(first - second))))


def boundary_ordered_all_points_polygon(
    points: list[tuple[int, int]],
    component: np.ndarray,
) -> np.ndarray:
    """Order every detected point by arclength around the zero component."""
    if len(points) < 3:
        return np.empty((0, 2), np.int32)
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), np.int32)
    boundary = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
    indexed = []
    for point in points:
        delta = boundary - np.asarray(point, np.int32)
        boundary_index = int(np.argmin(np.sum(delta.astype(np.float32) ** 2, axis=1)))
        indexed.append((boundary_index, point))
    indexed.sort(key=lambda item: item[0])
    ordered = []
    seen = set()
    for _, point in indexed:
        if point not in seen:
            seen.add(point)
            ordered.append(point)
    return np.asarray(ordered, np.int32) if len(ordered) >= 3 else np.empty((0, 2), np.int32)


def interval_polygon_candidates(
    levels: list[list[tuple[tuple[int, int], tuple[int, int], int]]],
    contours: list[np.ndarray],
    contour_center: tuple[float, float],
    component: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    """Connect corresponding interval endpoints across successive offsets."""
    candidates: list[tuple[np.ndarray, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for start_level, intervals in enumerate(levels):
        for start in intervals:
            p0, p1, _ = start
            if p1 < p0:
                p0, p1 = p1, p0
            left = [p0]
            right = [p1]
            previous = (p0, p1)
            previous_angle = circular_interval_angle(start, contour_center)
            for next_level in range(start_level + 1, len(levels)):
                if not levels[next_level]:
                    continue
                matches = []
                for interval in levels[next_level]:
                    next_left, next_right, endpoint_cost = orient_interval(interval, previous)
                    next_angle = circular_interval_angle(interval, contour_center)
                    angle_cost = circular_angle_distance(previous_angle, next_angle)
                    matches.append(
                        (angle_cost * 500.0 + endpoint_cost * 0.05, next_left, next_right, next_angle)
                    )
                _, next_left, next_right, next_angle = min(matches, key=lambda match: match[0])
                left.append(next_left)
                right.append(next_right)
                previous = (next_left, next_right)
                previous_angle = next_angle
                ordered = np.asarray(left + list(reversed(right)), np.int32)
                key = tuple(tuple(int(v) for v in point) for point in ordered)
                if len(np.unique(ordered, axis=0)) >= 3 and key not in seen:
                    seen.add(key)
                    candidates.append((ordered, len(left)))

    # Also create direct strips for every offset pair.  Intermediate detected
    # levels must not force a zigzag path or suppress a wider outer-to-inner
    # quadrilateral.
    for first_level in range(len(levels)):
        for first in levels[first_level]:
            p0, p1, _ = first
            first_angle = circular_interval_angle(first, contour_center)
            for second_level in range(first_level + 1, len(levels)):
                if not levels[second_level]:
                    continue
                ranked = []
                for second in levels[second_level]:
                    q0, q1, endpoint_cost = orient_interval(second, (p0, p1))
                    angle_cost = circular_angle_distance(
                        first_angle, circular_interval_angle(second, contour_center)
                    )
                    ranked.append((angle_cost * 500.0 + endpoint_cost * 0.05, q0, q1))
                _, q0, q1 = min(ranked, key=lambda match: match[0])
                ordered = np.asarray([p0, p1, q1, q0], np.int32)
                key = tuple(tuple(int(v) for v in point) for point in ordered)
                if len(np.unique(ordered, axis=0)) >= 3 and key not in seen:
                    seen.add(key)
                    candidates.append((ordered, 2))

    # A region intersecting only one offset still receives candidates: project
    # its interval endpoints to the nearest adjacent offset and form a strip.
    nonempty_levels = [index for index, intervals in enumerate(levels) if intervals]
    if len(nonempty_levels) == 1:
        level_index = nonempty_levels[0]
        adjacent_levels = [
            index
            for index in (level_index - 1, level_index + 1)
            if 0 <= index < len(contours)
        ]
        for p0, p1, _ in levels[level_index]:
            for adjacent_index in adjacent_levels:
                target_y, target_x = np.where(contours[adjacent_index] & component)
                if not len(target_x):
                    eroded = cv2.erode(
                        component.astype(np.uint8), np.ones((3, 3), np.uint8)
                    ) > 0
                    target_y, target_x = np.where(component & ~eroded)
                if not len(target_x):
                    continue
                target_points = np.column_stack([target_x, target_y]).astype(np.int32)
                projected = []
                for point in (p0, p1):
                    delta = target_points - np.asarray(point, np.int32)
                    nearest = target_points[int(np.argmin(np.sum(delta ** 2, axis=1)))]
                    projected.append((int(nearest[0]), int(nearest[1])))
                q0, q1 = projected
                ordered = np.asarray([p0, p1, q1, q0], np.int32)
                key = tuple(tuple(int(v) for v in point) for point in ordered)
                if len(np.unique(ordered, axis=0)) >= 3 and key not in seen:
                    seen.add(key)
                    candidates.append((ordered, 2))
    return candidates


def correction_intrusion_rows(
    polygon_mask: np.ndarray,
    correction_components: list[tuple[str, int, np.ndarray, int]],
) -> tuple[float, list[dict]]:
    rows = []
    maximum = 0.0
    for sign, component_id, component, area in correction_components:
        overlap = int((polygon_mask & component).sum())
        if overlap == 0:
            continue
        ratio = float(overlap / area) if area else 0.0
        maximum = max(maximum, ratio)
        rows.append(
            {
                "sign": sign,
                "component_id": component_id,
                "correction_area_px": area,
                "overlap_px": overlap,
                "overlap_ratio_of_correction_region": ratio,
            }
        )
    return maximum, rows


def bevel_acute_vertices(
    polygon: np.ndarray,
    maximum_angle_deg: float,
    bevel_distance_px: float,
) -> tuple[np.ndarray, int]:
    """Replace each acute apex with two points on its adjacent edges."""
    points = np.asarray(polygon, np.float32).reshape(-1, 2)
    if len(points) < 3 or maximum_angle_deg <= 0.0 or bevel_distance_px <= 0.0:
        return np.rint(points).astype(np.int32), 0
    output: list[np.ndarray] = []
    bevel_count = 0
    for index, current in enumerate(points):
        previous = points[(index - 1) % len(points)]
        following = points[(index + 1) % len(points)]
        toward_previous = previous - current
        toward_following = following - current
        previous_length = float(np.linalg.norm(toward_previous))
        following_length = float(np.linalg.norm(toward_following))
        if previous_length <= 2.0 or following_length <= 2.0:
            output.append(current)
            continue
        cosine = float(
            np.clip(
                np.dot(toward_previous, toward_following)
                / (previous_length * following_length),
                -1.0,
                1.0,
            )
        )
        angle_deg = float(np.degrees(np.arccos(cosine)))
        if angle_deg >= maximum_angle_deg:
            output.append(current)
            continue
        distance = min(
            bevel_distance_px,
            previous_length * 0.35,
            following_length * 0.35,
        )
        output.append(current + toward_previous / previous_length * distance)
        output.append(current + toward_following / following_length * distance)
        bevel_count += 1
    rounded = np.rint(np.asarray(output)).astype(np.int32)
    deduplicated = []
    for point in rounded:
        if not deduplicated or not np.array_equal(point, deduplicated[-1]):
            deduplicated.append(point)
    if len(deduplicated) > 1 and np.array_equal(deduplicated[0], deduplicated[-1]):
        deduplicated.pop()
    return np.asarray(deduplicated, np.int32), bevel_count


def square_acute_final_polygon_corners(
    polygon_mask: np.ndarray,
    polygon_rows: list[dict],
    part: np.ndarray,
    correction_components: list[tuple[str, int, np.ndarray, int]],
    maximum_angle_deg: float,
    bevel_distance_px: float,
    min_polygon_ratio: float,
    max_intrusion_ratio: float,
    part_px: int,
) -> tuple[np.ndarray, list[dict]]:
    """Bevel sharp final corners without reducing the existing vertex set."""
    if maximum_angle_deg <= 0.0 or bevel_distance_px <= 0.0:
        return polygon_mask, []
    output = np.zeros_like(polygon_mask, dtype=bool)
    result_rows: list[dict] = []
    for row in polygon_rows:
        if not row.get("accepted"):
            continue
        original_u8 = np.zeros_like(polygon_mask, dtype=np.uint8)
        candidate_u8 = np.zeros_like(polygon_mask, dtype=np.uint8)
        original_vertex_count = 0
        squared_vertex_count = 0
        acute_corner_count = 0
        paths_valid = True
        for path_values in row.get("polygon_paths_xy", []):
            path = np.asarray(path_values, np.int32).reshape(-1, 2)
            if len(path) < 3:
                continue
            original_vertex_count += len(path)
            cv2.fillPoly(original_u8, [path], 1)
            squared, bevel_count = bevel_acute_vertices(
                path, maximum_angle_deg, bevel_distance_px
            )
            squared_vertex_count += len(squared)
            acute_corner_count += bevel_count
            if len(squared) < 3 or polygon_self_intersects(squared):
                paths_valid = False
                cv2.fillPoly(candidate_u8, [path], 1)
            else:
                cv2.fillPoly(candidate_u8, [squared], 1)
        original = (original_u8 > 0) & part
        candidate = (candidate_u8 > 0) & part
        intrusion, intrusion_rows = correction_intrusion_rows(
            candidate, correction_components
        )
        candidate_ratio = float(candidate.sum() / part_px) if part_px else 0.0
        accepted = (
            paths_valid
            and acute_corner_count > 0
            and candidate_ratio > min_polygon_ratio
            and intrusion < max_intrusion_ratio
        )
        selected = candidate if accepted else original
        output |= selected
        result_rows.append(
            {
                "component_id": row["component_id"],
                "original_vertex_count": original_vertex_count,
                "squared_vertex_count": squared_vertex_count,
                "acute_corner_count": acute_corner_count,
                "original_area_px": int(original.sum()),
                "squared_area_px": int(candidate.sum()),
                "squared_ratio_of_part": candidate_ratio,
                "maximum_correction_region_intrusion_ratio": intrusion,
                "correction_region_intrusions": intrusion_rows,
                "corner_squaring_accepted": accepted,
            }
        )
    return output if np.any(output) else polygon_mask, result_rows


def reconnect_polygon_to_intrusion_limit(
    polygon: np.ndarray,
    part: np.ndarray,
    correction_components: list[tuple[str, int, np.ndarray, int]],
    max_intrusion_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float, float, list[dict]]:
    """Reconnect neighbouring intersection points instead of scaling the polygon.

    When an ordered strip intrudes too far into a correction component, remove
    the vertex whose neighbour-to-neighbour reconnection most reduces the
    violation.  Only the offending side is changed.
    """
    active = polygon.astype(np.int32)
    reconnect_count = 0
    correction_union = np.zeros_like(part, dtype=bool)
    for _, _, correction_component, _ in correction_components:
        correction_union |= correction_component
    while len(active) >= 3:
        current_mask, current_polygon = rasterize_polygon(active, part)
        current_maximum, current_rows = correction_intrusion_rows(
            current_mask, correction_components
        )
        current_self_intersection = polygon_self_intersects(current_polygon)
        if current_maximum < max_intrusion_ratio and not current_self_intersection:
            return current_mask, current_polygon, float(reconnect_count), current_maximum, current_rows
        if len(current_polygon) <= 3:
            break

        # Restrict reconnection to vertices adjacent to an offending edge.
        # For a self-crossing shape without excessive intrusion, every vertex
        # is evaluated once; only the best local removal is applied.
        local_vertex_indices = set()
        for edge_index in range(len(current_polygon)):
            edge = np.zeros_like(part, np.uint8)
            start = tuple(int(v) for v in current_polygon[edge_index])
            end = tuple(int(v) for v in current_polygon[(edge_index + 1) % len(current_polygon)])
            cv2.line(edge, start, end, 1, thickness=2, lineType=cv2.LINE_8)
            if np.any((edge > 0) & correction_union):
                local_vertex_indices.add(edge_index)
                local_vertex_indices.add((edge_index + 1) % len(current_polygon))
        if current_self_intersection or not local_vertex_indices:
            local_vertex_indices = set(range(len(current_polygon)))

        candidates = []
        for vertex_index in sorted(local_vertex_indices):
            candidate_points = np.delete(current_polygon, vertex_index, axis=0)
            if len(candidate_points) < 3:
                continue
            candidate_mask, candidate_polygon = rasterize_polygon(candidate_points, part)
            maximum, rows = correction_intrusion_rows(candidate_mask, correction_components)
            self_intersection = polygon_self_intersects(candidate_polygon)
            excess_sum = sum(
                max(0.0, row["overlap_ratio_of_correction_region"] - max_intrusion_ratio)
                for row in rows
            )
            score = (
                int(self_intersection),
                max(0.0, maximum - max_intrusion_ratio),
                excess_sum,
                -int(candidate_mask.sum()),
            )
            candidates.append(
                (
                    score,
                    candidate_points,
                    candidate_mask,
                    candidate_polygon,
                    rows,
                    maximum,
                )
            )
        if not candidates:
            break
        best = min(candidates, key=lambda item: item[0])
        # A single removal can be neutral before the next local reconnection
        # produces an improvement. Continue reducing the offending chain until
        # it satisfies the rule or only a triangle remains.
        active = best[1]
        reconnect_count += 1

    failed_mask, failed_polygon = rasterize_polygon(active, part)
    failed_maximum, failed_rows = correction_intrusion_rows(failed_mask, correction_components)
    return failed_mask, failed_polygon, float(reconnect_count), failed_maximum, failed_rows


def case1_polygons(
    zero_area: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    part: np.ndarray,
    labels: np.ndarray,
    rows: list[dict],
    offset_count: int,
    part_px: int,
    min_polygon_ratio: float,
    max_correction_intrusion_ratio: float,
    zero_component_expansion_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[dict]]:
    contour_masks, depths = nested_part_contours(part, offset_count)
    _, holes = split_outer_and_hole_background(part)
    hole_count, _, hole_stats, hole_centroids = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), connectivity=8
    )
    if hole_count > 1:
        dominant_hole = 1 + int(np.argmax(hole_stats[1:, cv2.CC_STAT_AREA]))
        contour_center = tuple(float(v) for v in hole_centroids[dominant_hole])
    else:
        part_y, part_x = np.where(part)
        contour_center = (float(part_x.mean()), float(part_y.mean()))
    correction_components = correction_component_masks(positive, negative)
    polygon_area = np.zeros_like(part, dtype=bool)
    point_mask = np.zeros_like(part, dtype=bool)
    expanded_zero_support = np.zeros_like(part, dtype=bool)
    polygon_rows = []
    for row in rows:
        component_id = row["component_id"]
        component = labels == component_id
        if zero_component_expansion_px > 0:
            expansion_size = zero_component_expansion_px * 2 + 1
            support_component = (
                cv2.dilate(
                    component.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_RECT, (expansion_size, expansion_size)
                    ),
                )
                > 0
            ) & part
        else:
            support_component = component
        expanded_zero_support |= support_component
        points: list[tuple[int, int]] = []
        level_intervals = []
        level_counts = []
        level_search_radii = []
        for contour in contour_masks:
            intervals, used_radius = offset_interval_pairs(contour, support_component)
            level_intervals.append(intervals)
            level_points = [point for interval in intervals for point in interval[:2]]
            points.extend(level_points)
            level_counts.append(len(level_points))
            level_search_radii.append(used_radius)
        points = deduplicate_points(points)
        if points:
            point_coords = np.asarray(points, np.int32)
            point_mask[point_coords[:, 1], point_coords[:, 0]] = True
        all_points_polygon = boundary_ordered_all_points_polygon(points, support_component)
        if len(all_points_polygon) >= 3:
            raw_candidates = [
                (all_points_polygon, sum(bool(intervals) for intervals in level_intervals))
            ]
        else:
            # Keep the single-interval projection fallback when fewer than
            # three actual intersections exist.
            raw_candidates = interval_polygon_candidates(
                level_intervals, contour_masks, contour_center, support_component
            )
        point_rule_passed = bool(raw_candidates)
        accepted = False
        polygon = None
        polygon_paths: list[np.ndarray] = []
        polygon_mask_component_count = 0
        polygon_px = 0
        polygon_ratio = 0.0
        polygon_source_overlap_ratio = 0.0
        polygon_offset_level_count = 0
        polygon_reconnect_count = 0
        max_intrusion = 0.0
        intrusion_rows: list[dict] = []
        if point_rule_passed:
            evaluated = []
            for raw_polygon, used_level_count in raw_candidates:
                local, candidate_polygon, reconnect_count, intrusion, candidate_intrusions = (
                    reconnect_polygon_to_intrusion_limit(
                        raw_polygon,
                        part,
                        correction_components,
                        max_correction_intrusion_ratio,
                    )
                )
                area = int(local.sum())
                ratio = float(area / part_px) if part_px else 0.0
                source_overlap = float((local & component).sum() / max(int(component.sum()), 1))
                valid = (
                    intrusion < max_correction_intrusion_ratio
                    and len(candidate_polygon) >= 3
                    and not polygon_self_intersects(candidate_polygon)
                )
                evaluated.append(
                    {
                        "valid": valid,
                        "local": local,
                        "polygon": candidate_polygon,
                        "reconnect_count": int(reconnect_count),
                        "intrusion": intrusion,
                        "intrusion_rows": candidate_intrusions,
                        "area": area,
                        "ratio": ratio,
                        "source_overlap": source_overlap,
                        "used_level_count": used_level_count,
                    }
                )
            compliant = [candidate for candidate in evaluated if candidate["valid"]]
            if compliant:
                local = np.zeros_like(part, dtype=bool)
                selected_candidates = []
                covered_source = np.zeros_like(part, dtype=bool)
                compliant.sort(
                    key=lambda candidate: (
                        candidate["area"],
                        int((candidate["local"] & component).sum()),
                    ),
                    reverse=True,
                )
                for candidate in compliant:
                    new_source = candidate["local"] & component & ~covered_source
                    if not np.any(new_source) and selected_candidates:
                        continue
                    if selected_candidates:
                        overlap = int((candidate["local"] & local).sum())
                        overlap_ratio = overlap / max(
                            min(candidate["area"], int(local.sum())), 1
                        )
                        if overlap_ratio > 0.65:
                            continue
                    proposed = local | candidate["local"]
                    proposed_intrusion, proposed_rows = correction_intrusion_rows(
                        proposed, correction_components
                    )
                    if proposed_intrusion >= max_correction_intrusion_ratio:
                        continue
                    local = proposed
                    covered_source |= candidate["local"] & component
                    selected_candidates.append(candidate)
                    max_intrusion = proposed_intrusion
                    intrusion_rows = proposed_rows
                polygon_paths = [candidate["polygon"] for candidate in selected_candidates]
                if polygon_paths:
                    polygon = max(polygon_paths, key=lambda path: abs(cv2.contourArea(path)))
                polygon_reconnect_count = sum(
                    candidate["reconnect_count"] for candidate in selected_candidates
                )
                polygon_offset_level_count = max(
                    (candidate["used_level_count"] for candidate in selected_candidates),
                    default=0,
                )
                polygon_px = int(local.sum())
                polygon_ratio = float(polygon_px / part_px) if part_px else 0.0
                polygon_source_overlap_ratio = float(
                    (local & component).sum() / max(int(component.sum()), 1)
                )
                polygon_mask_component_count = max(
                    cv2.connectedComponents(local.astype(np.uint8), connectivity=8)[0] - 1,
                    0,
                )
                accepted = polygon_ratio > min_polygon_ratio and bool(polygon_paths)
            else:
                selected = min(
                    evaluated,
                    key=lambda candidate: (
                        max(0.0, candidate["intrusion"] - max_correction_intrusion_ratio),
                        -candidate["source_overlap"],
                        -candidate["area"],
                    ),
                )
                local = selected["local"]
                polygon = selected["polygon"]
                polygon_paths = [polygon]
                polygon_reconnect_count = selected["reconnect_count"]
                max_intrusion = selected["intrusion"]
                intrusion_rows = selected["intrusion_rows"]
                polygon_px = selected["area"]
                polygon_ratio = selected["ratio"]
                polygon_source_overlap_ratio = selected["source_overlap"]
                polygon_offset_level_count = selected["used_level_count"]
                polygon_mask_component_count = max(
                    cv2.connectedComponents(local.astype(np.uint8), connectivity=8)[0] - 1,
                    0,
                )
            if accepted:
                polygon_area |= local
        if not point_rule_passed:
            skip_reason = "no_offset_interval_or_projectable_candidate"
        elif not accepted:
            skip_reason = "polygon_area_or_correction_intrusion_limit_failed"
        else:
            skip_reason = None
        polygon_rows.append(
            {
                **row,
                "intersection_point_count": len(points),
                "intersection_count_by_contour": level_counts,
                "intersection_search_radius_by_contour_px": level_search_radii,
                "outermost_intersection_count": level_counts[0],
                "innermost_intersection_count": level_counts[-1],
                "point_rule_passed": point_rule_passed,
                "offset_interval_count_by_contour": [len(intervals) for intervals in level_intervals],
                "polygon_candidate_count": len(raw_candidates),
                "selected_polygon_strip_count": len(polygon_paths),
                "polygon_mask_connected_component_count": polygon_mask_component_count,
                "polygon_offset_level_count": polygon_offset_level_count,
                "polygon_area_px": polygon_px,
                "polygon_ratio_of_part": polygon_ratio,
                "polygon_source_component_overlap_ratio": polygon_source_overlap_ratio,
                "polygon_reconnect_count_for_intrusion_limit": polygon_reconnect_count,
                "maximum_correction_region_intrusion_ratio": max_intrusion,
                "correction_region_intrusions": intrusion_rows,
                "accepted": accepted,
                "skip_reason": skip_reason,
                "points_xy": [[int(x), int(y)] for x, y in points],
                "polygon_xy": polygon.tolist() if polygon is not None else [],
                "polygon_paths_xy": [path.tolist() for path in polygon_paths],
            }
        )
    return polygon_area, point_mask, expanded_zero_support, contour_masks, polygon_rows


def source_masks(positive: np.ndarray, negative: np.ndarray, part: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
    masks: list[np.ndarray] = []
    rows: list[dict] = []
    for sign, source in (("positive", positive), ("negative", negative)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(source.astype(np.uint8), 8)
        for component_id in range(1, count):
            component = labels == component_id
            masks.append(component)
            rows.append(
                {
                    "source_id": len(masks) - 1,
                    "kind": sign,
                    "component_id": component_id,
                    "area_px": int(stats[component_id, cv2.CC_STAT_AREA]),
                }
            )
    background, _ = split_outer_and_hole_background(part)
    masks.append(background)
    rows.append(
        {
            "source_id": len(masks) - 1,
            "kind": "external_background",
            "component_id": 1,
            "area_px": int(background.sum()),
        }
    )
    return masks, rows


def voronoi_separator(
    positive: np.ndarray,
    negative: np.ndarray,
    part: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    masks, rows = source_masks(positive, negative, part)
    distances = np.stack(
        [cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) for mask in masks],
        axis=0,
    )
    owner = np.argmin(distances, axis=0).astype(np.int16)
    boundary = np.zeros_like(part, dtype=bool)
    boundary[:, 1:] |= owner[:, 1:] != owner[:, :-1]
    boundary[1:, :] |= owner[1:, :] != owner[:-1, :]
    boundary &= part
    return skeletonize(boundary), owner, rows


def snap_anchor_points(
    values: np.ndarray,
    part: np.ndarray,
    neutral: np.ndarray,
) -> tuple[list, int]:
    anchors = find_boundary_anchors(values, part, 230, 40.0)
    smooth_window = 230
    if len(anchors) < 2:
        anchors = find_boundary_anchors(values, part, 21, 40.0)
        smooth_window = 21
    outer_contours, _ = cv2.findContours(part.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not outer_contours:
        return anchors, smooth_window
    outer = np.zeros_like(part, np.uint8)
    cv2.drawContours(outer, [max(outer_contours, key=cv2.contourArea)], -1, 1, 1)
    candidates_y, candidates_x = np.where((outer > 0) & neutral)
    if not len(candidates_x):
        return anchors, smooth_window
    for anchor in anchors:
        distances = (candidates_x - anchor.x) ** 2 + (candidates_y - anchor.y) ** 2
        index = int(np.argmin(distances))
        if distances[index] <= 35 ** 2:
            anchor.x = int(candidates_x[index])
            anchor.y = int(candidates_y[index])
    return anchors, smooth_window


def split_outer_and_hole_background(part: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split background connected to the image border from enclosed part holes."""
    background = ~part
    count, labels = cv2.connectedComponents(background.astype(np.uint8), 8)
    border_ids = set(np.unique(labels[0, :]).tolist())
    border_ids.update(np.unique(labels[-1, :]).tolist())
    border_ids.update(np.unique(labels[:, 0]).tolist())
    border_ids.update(np.unique(labels[:, -1]).tolist())
    border_ids.discard(0)
    outside = np.isin(labels, list(border_ids)) & background
    holes = background & ~outside
    return outside, holes


def background_boundaries(part: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outside, holes = split_outer_and_hole_background(part)
    outer_boundary = part & (cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    hole_boundary = part & (cv2.dilate(holes.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    return outside, holes, outer_boundary, hole_boundary


def topology_safe_components(
    line: np.ndarray,
    part: np.ndarray,
    anchors: list,
    min_line_px: int = CASE2_MIN_LINE_PX,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    line = skeletonize(line)
    _, _, outer_boundary, hole_boundary = background_boundaries(part)
    termination = cv2.dilate(outer_boundary.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    outer_termination = cv2.dilate(outer_boundary.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    hole_termination = cv2.dilate(hole_boundary.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    anchor_mask = np.zeros_like(part, np.uint8)
    for anchor in anchors:
        cv2.circle(anchor_mask, (anchor.x, anchor.y), 7, 1, -1)
    termination |= anchor_mask > 0
    neighbors = cv2.filter2D(line.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    endpoints_all = line & (neighbors <= 2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), 8)
    kept = np.zeros_like(part, dtype=bool)
    starts = np.zeros_like(part, dtype=bool)
    rows = []
    for component_id in range(1, count):
        component = labels == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        endpoints = component & endpoints_all
        endpoint_count = int(endpoints.sum())
        terminating_count = int((endpoints & termination).sum())
        closed_loop = endpoint_count == 0
        accepted = area >= min_line_px and (closed_loop or terminating_count == endpoint_count)
        touches_anchor = bool(np.any(component & (anchor_mask > 0)))
        touches_background = bool(np.any(component & outer_termination))
        outer_endpoint_count = int((endpoints & outer_termination).sum())
        hole_endpoint_count = int((endpoints & hole_termination).sum())
        if accepted:
            kept |= component
            starts |= component & (anchor_mask > 0)
        rows.append(
            {
                "component_id": component_id,
                "line_px": area,
                "endpoint_count": endpoint_count,
                "terminating_endpoint_count": terminating_count,
                "closed_loop": closed_loop,
                "touches_start_anchor": touches_anchor,
                "touches_background": touches_background,
                "outer_background_endpoint_count": outer_endpoint_count,
                "internal_hole_endpoint_count": hole_endpoint_count,
                "accepted": accepted,
            }
        )
    return kept, starts, rows


def shortest_path_to_target(
    start: tuple[int, int],
    target: np.ndarray,
    allowed: np.ndarray,
    traversal_cost: np.ndarray,
    target_distance: np.ndarray,
    max_visited: int = 300_000,
) -> list[tuple[int, int]]:
    """A* route from one line endpoint to any valid background/anchor target."""
    start_x, start_y = start
    h, w = allowed.shape
    start_index = start_y * w + start_x
    best = {start_index: 0.0}
    previous: dict[int, int] = {}
    queue = [(float(target_distance[start_y, start_x]) * 0.2, 0.0, start_index)]
    visited = 0
    moves = (
        (-1, -1, 1.4142), (0, -1, 1.0), (1, -1, 1.4142),
        (-1, 0, 1.0),                         (1, 0, 1.0),
        (-1, 1, 1.4142),  (0, 1, 1.0),  (1, 1, 1.4142),
    )
    goal_index = None
    while queue and visited < max_visited:
        _, cost_so_far, index = heapq.heappop(queue)
        if cost_so_far != best.get(index):
            continue
        y, x = divmod(index, w)
        visited += 1
        if target[y, x] and index != start_index:
            goal_index = index
            break
        for dx, dy, step in moves:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h or not allowed[ny, nx]:
                continue
            next_index = ny * w + nx
            next_cost = cost_so_far + step * float(traversal_cost[ny, nx])
            if next_cost >= best.get(next_index, float("inf")):
                continue
            best[next_index] = next_cost
            previous[next_index] = index
            heuristic = float(target_distance[ny, nx]) * 0.2
            heapq.heappush(queue, (next_cost + heuristic, next_cost, next_index))
    if goal_index is None:
        return []
    path = []
    index = goal_index
    while True:
        y, x = divmod(index, w)
        path.append((x, y))
        if index == start_index:
            break
        index = previous[index]
    path.reverse()
    return path


def normal_path_to_boundary(
    start: tuple[int, int],
    distance: np.ndarray,
    boundary: np.ndarray,
    allowed: np.ndarray,
    max_steps: int = 40,
) -> list[tuple[int, int]]:
    """Follow the background-distance gradient outward, approximately normally."""
    x, y = start
    path = [(x, y)]
    for _ in range(max_steps):
        if boundary[y, x]:
            return path
        candidates = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < allowed.shape[1] and 0 <= ny < allowed.shape[0] and allowed[ny, nx]:
                    candidates.append((float(distance[ny, nx]), nx, ny))
        if not candidates:
            return []
        next_distance, nx, ny = min(candidates)
        if next_distance >= float(distance[y, x]) - 1e-6:
            return []
        x, y = nx, ny
        path.append((x, y))
    return path if boundary[y, x] else []


def anchor_normal_gateways(
    anchors: list,
    part: np.ndarray,
    allowed: np.ndarray,
    depth_px: int,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Build inward normal rays so lines leave outer anchors orthogonally."""
    part_distance = cv2.distanceTransform(part.astype(np.uint8), cv2.DIST_L2, 5)
    gateway = np.zeros_like(part, dtype=bool)
    rays: list[list[tuple[int, int]]] = []
    for anchor in anchors:
        x, y = anchor.x, anchor.y
        if not allowed[y, x]:
            continue
        ray = [(x, y)]
        for _ in range(depth_px):
            candidates = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < part.shape[1] and 0 <= ny < part.shape[0] and allowed[ny, nx]:
                        candidates.append((float(part_distance[ny, nx]), nx, ny))
            if not candidates:
                break
            next_distance, nx, ny = max(candidates)
            if next_distance <= float(part_distance[y, x]) + 1e-6:
                break
            x, y = nx, ny
            ray.append((x, y))
        if len(ray) >= 4:
            gateway[y, x] = True
            rays.append(ray)
    return gateway, rays


def extend_lines_to_terminations(
    line: np.ndarray,
    allowed: np.ndarray,
    part: np.ndarray,
    values: np.ndarray,
    structural_edges: np.ndarray,
    anchors: list,
    neutral_limit_mm: float,
    edge_preference_radius: int,
) -> tuple[np.ndarray, list[dict]]:
    """Extend open ends to another outer anchor or the external background."""
    line = skeletonize(line)
    outside, _, outer_boundary, _ = background_boundaries(part)
    outer_distance = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 5)
    depth = NORMAL_GATEWAY_DEPTH_PX
    outer_gateway = allowed & part & (outer_distance >= depth - 1.0) & (outer_distance <= depth + 1.0)
    anchor_gateway, anchor_rays = anchor_normal_gateways(anchors, part, allowed, depth)
    targets = {
        "anchor": anchor_gateway if len(anchor_rays) >= 2 else np.zeros_like(part, bool),
        "outer_background": outer_gateway,
    }
    target_distances = {
        kind: cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
        for kind, target in targets.items()
        if np.any(target)
    }
    edge_size = max(3, edge_preference_radius * 2 + 1)
    edge_zone = cv2.dilate(
        (structural_edges > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_size, edge_size)),
    ) > 0
    deviation_cost = 1.0 + 1.5 * np.minimum(np.abs(values) / neutral_limit_mm, 4.0)
    traversal_cost = deviation_cost.astype(np.float32)
    traversal_cost[edge_zone] *= 0.22
    traversal_cost = np.maximum(traversal_cost, 0.2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), 8)
    neighbors = cv2.filter2D(line.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    endpoints_all = line & (neighbors <= 2)
    extended = np.zeros_like(part, dtype=bool)
    rows = []
    for component_id in range(1, count):
        component = labels == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        ys, xs = np.where(component & endpoints_all)
        endpoint_rows = []
        successful = True
        extension = np.zeros_like(part, dtype=bool)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if outer_boundary[y, x]:
                endpoint_rows.append(
                    {"xy": [x, y], "target_kind": "outer_background", "already_terminated": True,
                     "orthogonal_tail_px": 0, "extension_px": 0}
                )
                continue
            options: dict[str, list[tuple[int, int]]] = {}
            for kind in ("anchor", "outer_background"):
                if kind not in target_distances:
                    continue
                path = shortest_path_to_target(
                    (x, y), targets[kind], allowed, traversal_cost, target_distances[kind]
                )
                if path:
                    options[kind] = path
            chosen_kind = None
            path: list[tuple[int, int]] = []
            if "anchor" in options:
                chosen_kind, path = "anchor", options["anchor"]
            elif "outer_background" in options:
                chosen_kind, path = "outer_background", options["outer_background"]
            if chosen_kind is None or not path:
                successful = False
                endpoint_rows.append(
                    {"xy": [x, y], "target_kind": None, "already_terminated": False,
                     "orthogonal_tail_px": 0, "extension_px": 0}
                )
                continue
            tail: list[tuple[int, int]] = []
            goal = path[-1]
            if chosen_kind == "anchor":
                matching = [ray for ray in anchor_rays if ray[-1] == goal]
                tail = list(reversed(matching[0])) if matching else []
            elif chosen_kind == "outer_background":
                tail = normal_path_to_boundary(goal, outer_distance, outer_boundary, allowed)
            if not tail:
                successful = False
                endpoint_rows.append(
                    {"xy": [x, y], "target_kind": chosen_kind, "already_terminated": False,
                     "orthogonal_tail_px": 0, "extension_px": 0}
                )
                continue
            path = path + tail[1:]
            path_array = np.asarray(path, np.int32)
            extension[path_array[:, 1], path_array[:, 0]] = True
            endpoint_rows.append(
                {"xy": [x, y], "target_kind": chosen_kind, "already_terminated": False,
                 "orthogonal_tail_px": len(tail), "extension_px": len(path)}
            )
        closed_loop = len(xs) == 0
        accepted = closed_loop or successful
        if accepted:
            extended |= component | extension
        rows.append(
            {
                "source_component_id": component_id,
                "source_line_px": area,
                "endpoint_count": len(xs),
                "closed_loop": closed_loop,
                "all_endpoints_extended": successful,
                "accepted": accepted,
                "endpoints": endpoint_rows,
            }
        )
    return skeletonize(extended), rows


def case2_separator(
    image: np.ndarray,
    values: np.ndarray,
    mapped: np.ndarray,
    gray: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    part: np.ndarray,
    neutral_limit_mm: float,
    allowance_px: int,
    edge_snap_radius: int,
    min_line_px: int,
) -> dict:
    strict_neutral = part & mapped & ~gray & (np.abs(values) <= neutral_limit_mm)
    if np.any(strict_neutral):
        distance_to_neutral = cv2.distanceTransform((~strict_neutral).astype(np.uint8), cv2.DIST_L2, 5)
        allowed = part & (distance_to_neutral <= allowance_px)
    else:
        distance_to_neutral = np.full(part.shape, np.inf, np.float32)
        allowed = np.zeros_like(part, dtype=bool)
    raw, owner, sources = voronoi_separator(positive, negative, part)
    allowed_raw = raw & allowed
    structural_edges = detect_structural_edges(image, part)
    anchors, smooth_window = snap_anchor_points(values, part, strict_neutral)
    extended, first_extension_rows = extend_lines_to_terminations(
        allowed_raw,
        allowed,
        part,
        values,
        structural_edges,
        anchors,
        neutral_limit_mm,
        edge_snap_radius,
    )
    # A first merge/skeleton pass can expose new open endpoints. Route those
    # ends once more to another outer anchor or the external background before
    # the final topology and length checks.
    extended, second_extension_rows = extend_lines_to_terminations(
        extended,
        allowed,
        part,
        values,
        structural_edges,
        anchors,
        neutral_limit_mm,
        edge_snap_radius,
    )
    final, starts, line_rows = topology_safe_components(
        extended, part, anchors, min_line_px=min_line_px
    )
    edge_size = max(3, edge_snap_radius * 2 + 1)
    edge_zone = cv2.dilate(
        (structural_edges > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_size, edge_size)),
    ) > 0
    edge_following = final & edge_zone
    edge_support = float(edge_following.sum() / final.sum()) if np.any(final) else 0.0
    return {
        "strict_neutral": strict_neutral,
        "allowed": allowed,
        "distance_to_neutral": distance_to_neutral,
        "raw": raw,
        "allowed_raw": allowed_raw,
        "owner": owner,
        "sources": sources,
        "structural_edges": structural_edges,
        "edge_following": edge_following > 0,
        "final": final,
        "starts": starts,
        "anchors": anchors,
        "anchor_smooth_window": smooth_window,
        "edge_support": edge_support,
        "extension_passes": [first_extension_rows, second_extension_rows],
        "line_rows": line_rows,
    }


def draw_points(image: np.ndarray, point_mask: np.ndarray, color: tuple[int, int, int]) -> None:
    count, _, _, centroids = cv2.connectedComponentsWithStats(point_mask.astype(np.uint8), 8)
    for component_id in range(1, count):
        x, y = (int(round(v)) for v in centroids[component_id])
        cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_region_numbers(
    image: np.ndarray,
    labels: np.ndarray,
    rows: list[dict],
    prefix: str,
    minimum_ratio: float = 0.0,
) -> int:
    visible_rows = [row for row in rows if row["ratio_of_part"] >= minimum_ratio]
    visible_rows.sort(key=lambda row: (row["centroid"][1], row["centroid"][0]))
    for display_index, row in enumerate(visible_rows, start=1):
        component = labels == row["component_id"]
        y, x = (int(round(row["centroid"][1])), int(round(row["centroid"][0])))
        if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and component[y, x]):
            ys, xs = np.where(component)
            nearest = int(
                np.argmin((xs - row["centroid"][0]) ** 2 + (ys - row["centroid"][1]) ** 2)
            )
            x, y = int(xs[nearest]), int(ys[nearest])
        text = f"{prefix}{display_index}"
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    return len(visible_rows)


def build_board(
    image: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    zero_area: np.ndarray,
    method: str,
    case1: dict | None,
    case2: dict | None,
    zero_ratio: float,
    zero_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    correction_view = image.copy()
    blend_mask(correction_view, positive, POS_FILL_RGB, 0.52)
    blend_mask(correction_view, negative, NEG_FILL_RGB, 0.52)

    zero_view = image.copy()
    blend_mask(zero_view, zero_area, ZERO_CANDIDATE_RGB, 0.45)
    draw_line(zero_view, mask_boundary(zero_area, 1), LINE_RGB, 2)

    construction = image.copy()
    final = image.copy()
    zero_subtitle = (
        f"{zero_ratio * 100:.2f}% of part / {zero_count} connected components / {method}"
    )
    blend_mask(final, positive, POS_FILL_RGB, 0.16)
    blend_mask(final, negative, NEG_FILL_RGB, 0.16)
    if method == "case1_contour_polygon" and case1 is not None:
        candidate_label_count = draw_region_numbers(
            zero_view,
            case1["split_labels"],
            case1["intersection_rows"],
            "C",
            case1["minimum_post_neck_component_ratio"],
        )
        draw_line(zero_view, case1["neck_cut_mask"].astype(np.uint8) * 255, LINE_RGB, 5)
        neck_labels, neck_rows = connected_component_rows(
            case1["neck_cut_mask"], max(int(zero_area.sum()), 1)
        )
        draw_region_numbers(zero_view, neck_labels, neck_rows, "N")
        zero_subtitle = (
            f"{zero_ratio * 100:.2f}% / {len(case1['split_rows'])} total after split / "
            f"C1-C{candidate_label_count} all post-neck regions / necks N1-N{len(neck_rows)}"
        )
        for contour in case1["contours"]:
            draw_line(construction, contour.astype(np.uint8) * 255, CONTOUR_RGB, 2)
        draw_line(construction, case1["neck_cut_mask"].astype(np.uint8) * 255, LINE_RGB, 4)
        draw_points(construction, case1["points"], ANCHOR_RGB)
        blend_mask(construction, case1["polygon"], POLYGON_RGB, 0.36)
        blend_mask(final, case1["polygon"], POLYGON_RGB, 0.48)
        draw_line(final, mask_boundary(case1["polygon"], 1), POLYGON_RGB, 4)
        final_labels, final_rows = connected_component_rows(case1["polygon"], int(zero_area.sum()))
        final_region_count = draw_region_numbers(final, final_labels, final_rows, "Z")
        construction_title = "3. Six contours + intersections"
        construction_subtitle = (
            f"red: 6 contours; yellow: neck cuts ({len(neck_rows)}); "
            "green: endpoints"
        )
        final_title = "4. Case 1 zero-line polygons"
        final_subtitle = (
            f"blue: Z1-Z{final_region_count}; candidates C1-C{candidate_label_count}; "
            "area >0.5%; intrusion <10%"
        )
    else:
        assert case2 is not None
        blend_mask(construction, case2["allowed"], ZERO_CANDIDATE_RGB, 0.18)
        draw_line(construction, case2["structural_edges"], (90, 90, 90), 1)
        draw_line(construction, case2["allowed_raw"].astype(np.uint8) * 255, CONTOUR_RGB, 2)
        draw_line(construction, case2["edge_following"].astype(np.uint8) * 255, EDGE_LINE_RGB, 3)
        draw_line(final, case2["final"].astype(np.uint8) * 255, LINE_RGB, 4)
        draw_line(final, case2["starts"].astype(np.uint8) * 255, ANCHOR_RGB, 8)
        for anchor in case2["anchors"]:
            cv2.circle(final, (anchor.x, anchor.y), 6, ANCHOR_RGB, 2, cv2.LINE_AA)
        construction_title = "3. Neutral-constrained separators"
        construction_subtitle = "white: Voronoi / magenta: structural-edge preference"
        final_title = "4. Case 2 topology-safe zero lines"
        final_subtitle = "length >=100px; outer-normal exits; internal-hole ends rejected"
    panels = [
        add_title(fit_panel(correction_view), "1. Final 2% correction areas", "orange: positive / blue: negative"),
        add_title(
            fit_panel(zero_view),
            "2. Non-correction zero area",
            zero_subtitle,
        ),
        add_title(fit_panel(construction), construction_title, construction_subtitle),
        add_title(fit_panel(final), final_title, final_subtitle),
    ]
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))), final


def process_one(
    spec: ScanSpec,
    correction_dir: Path,
    output_dir: Path,
    case1_limit_ratio: float,
    neutral_limit_mm: float,
    allowance_px: int,
    offset_count: int,
    zero_component_expansion_px: int,
    zero_component_min_ratio: float,
    zero_post_neck_min_ratio: float,
    zero_neck_max_boundary_gap_px: float,
    zero_neck_min_cut_spacing_px: int,
    zero_neck_cut_margin_px: int,
    zero_neck_min_child_ratio: float,
    zero_neck_min_prominence_ratio: float,
    zero_neck_absolute_width_override_px: float,
    post_neck_small_thin_max_ratio: float,
    post_neck_small_thin_max_width_px: float,
    edge_snap_radius: int,
    min_polygon_ratio: float,
    max_correction_intrusion_ratio: float,
    acute_corner_max_angle_deg: float,
    acute_corner_bevel_distance_px: float,
    min_line_px: int,
) -> dict:
    source_dir = correction_dir / spec.key
    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    image = imread_rgb(Path(source_summary["source_image"]))
    legend = imread_rgb(Path(source_summary["legend_image"]))
    values, valid = map_deviation(image, extract_color_ramp(legend), spec.vmin, spec.vmax)
    values = cv2.medianBlur(values, 5)
    part = detect_part_mask(image)
    mapped = part & valid
    gray = detect_unmapped_gray(image, part, mapped)
    effective_values = values.copy()
    gray_positive_path = source_dir / "gray_assigned_positive_mask.png"
    gray_negative_path = source_dir / "gray_assigned_negative_mask.png"
    if gray_positive_path.exists() and gray_negative_path.exists():
        gray_positive = read_mask(gray_positive_path)
        gray_negative = read_mask(gray_negative_path)
        effective_values[gray_positive] = float(source_summary["gray_positive_assigned_mm"])
        effective_values[gray_negative] = float(source_summary["gray_negative_assigned_mm"])
    else:
        # Compatibility with correction results generated before signed gray
        # assignment was introduced.
        effective_values[gray] = float(source_summary.get("gray_sentinel_mm", 3.01))
    positive = read_mask(source_dir / "final_positive_correction_mask.png")
    negative = read_mask(source_dir / "final_negative_correction_mask.png")
    correction = positive | negative
    part_px = int(part.sum())
    raw_zero_area = part & ~correction
    zero_area, labels, zero_rows, raw_zero_rows = filter_components_by_ratio(
        raw_zero_area, part_px, zero_component_min_ratio
    )
    raw_zero_ratio = float(raw_zero_area.sum() / part_px) if part_px else 0.0
    zero_ratio = float(zero_area.sum() / part_px) if part_px else 0.0
    zero_count = len(zero_rows)
    use_case1 = zero_ratio < case1_limit_ratio and zero_count > 1

    case1_data = None
    case2_data = None
    if use_case1:
        split_labels, split_rows, neck_cut_mask, split_events = (
            split_zero_components_at_narrow_necks(
                zero_area,
                part_px,
                zero_neck_max_boundary_gap_px,
                zero_neck_min_child_ratio,
                zero_neck_min_cut_spacing_px,
                zero_neck_cut_margin_px,
                zero_neck_min_prominence_ratio,
                zero_neck_absolute_width_override_px,
            )
        )
        split_labels, split_rows, removed_small_thin_rows = (
            filter_small_thin_post_neck_regions(
                split_labels,
                split_rows,
                post_neck_small_thin_max_ratio,
                post_neck_small_thin_max_width_px,
            )
        )
        intersection_rows = [
            row for row in split_rows if row["ratio_of_part"] >= zero_post_neck_min_ratio
        ]
        polygon, points, expanded_zero_support, contours, polygon_rows = case1_polygons(
            zero_area,
            positive,
            negative,
            part,
            split_labels,
            intersection_rows,
            offset_count,
            part_px,
            min_polygon_ratio,
            max_correction_intrusion_ratio,
            zero_component_expansion_px,
        )
        polygon_before_corner_squaring = polygon.copy()
        polygon, corner_squaring_rows = square_acute_final_polygon_corners(
            polygon,
            polygon_rows,
            part,
            correction_component_masks(positive, negative),
            acute_corner_max_angle_deg,
            acute_corner_bevel_distance_px,
            min_polygon_ratio,
            max_correction_intrusion_ratio,
            part_px,
        )
        case1_data = {
            "polygon": polygon,
            "polygon_before_corner_squaring": polygon_before_corner_squaring,
            "corner_squaring_rows": corner_squaring_rows,
            "points": points,
            "expanded_zero_support": expanded_zero_support,
            "neck_cut_mask": neck_cut_mask,
            "split_labels": split_labels,
            "split_events": split_events,
            "split_rows": split_rows,
            "intersection_rows": intersection_rows,
            "post_neck_zero_area": split_labels > 0,
            "minimum_zero_component_ratio": zero_component_min_ratio,
            "minimum_post_neck_component_ratio": zero_post_neck_min_ratio,
            "neck_min_child_ratio": zero_neck_min_child_ratio,
            "removed_small_thin_rows": removed_small_thin_rows,
            "contours": contours,
            "rows": polygon_rows,
        }
        method = "case1_contour_polygon"
        final_mask = polygon
    else:
        case2_data = case2_separator(
            image,
            effective_values,
            mapped,
            gray,
            positive,
            negative,
            part,
            neutral_limit_mm,
            allowance_px,
            edge_snap_radius,
            min_line_px,
        )
        method = "case2_topology_safe_separator"
        final_mask = case2_data["final"]

    board, overlay = build_board(
        image, positive, negative, zero_area, method, case1_data, case2_data, zero_ratio, zero_count
    )
    item_dir = output_dir / spec.key
    imwrite_rgb(item_dir / "review_board.png", board)
    imwrite_rgb(item_dir / "final_zero_line_overlay.png", overlay)
    imwrite_gray(
        item_dir / "raw_zero_line_source_area_before_1pct_filter_mask.png",
        raw_zero_area.astype(np.uint8) * 255,
    )
    imwrite_gray(item_dir / "zero_line_source_area_mask.png", zero_area.astype(np.uint8) * 255)
    imwrite_gray(item_dir / "final_zero_line_mask.png", final_mask.astype(np.uint8) * 255)

    summary = {
        "source_image": source_summary["source_image"],
        "source_correction_result": str(source_dir),
        "correction_area_threshold_ratio": source_summary["minimum_correction_area_ratio_exclusive"],
        "case1_zero_area_limit_ratio_exclusive": case1_limit_ratio,
        "part_px": part_px,
        "raw_zero_area_px_before_1pct_filter": int(raw_zero_area.sum()),
        "raw_zero_area_ratio_of_part_before_1pct_filter": raw_zero_ratio,
        "raw_zero_connected_component_count_before_1pct_filter": len(raw_zero_rows),
        "minimum_zero_component_ratio_before_neck_split": zero_component_min_ratio,
        "minimum_post_neck_component_ratio_for_intersections": zero_post_neck_min_ratio,
        "zero_area_px": int(zero_area.sum()),
        "zero_area_ratio_of_part": zero_ratio,
        "zero_connected_component_count": zero_count,
        "selected_method": method,
        "final_zero_line_px": int(final_mask.sum()),
        "zero_components": zero_rows,
    }
    if case1_data is not None:
        summary.update(
            {
                "offset_contour_count": offset_count,
                "total_nested_contour_count": offset_count + 2,
                "offset_internal_small_holes_ignored": True,
                "innermost_contour_method": "boundary of largest enclosed background component",
                "case1_polygon_connection_method": "all intersections ordered by zero-component boundary arclength",
                "case1_convex_hull_disabled": True,
                "case1_intersection_method": "exact offset line and filtered zero-component boundary intersection",
                "zero_component_detection_expansion_px": zero_component_expansion_px,
                "filtered_zero_support_used_for_intersections_and_boundary_order": True,
                "one_percent_filtered_zero_area_used_for_area_and_method_rules": True,
                "case1_vertices_restricted_to_filtered_zero_support": True,
                "zero_neck_split_enabled": True,
                "zero_neck_max_boundary_gap_px": zero_neck_max_boundary_gap_px,
                "zero_neck_detection_method": "L2 distance-transform core partition",
                "zero_neck_distance_core_minimum_boundary_distance_px": (
                    zero_neck_max_boundary_gap_px / 2.0
                ),
                "zero_neck_min_cut_spacing_px": zero_neck_min_cut_spacing_px,
                "zero_neck_cut_margin_px": zero_neck_cut_margin_px,
                "zero_neck_cut_total_nominal_width_px": zero_neck_cut_margin_px * 2 + 1,
                "zero_neck_minimum_child_area_ratio": zero_neck_min_child_ratio,
                "zero_neck_minimum_width_prominence_ratio": zero_neck_min_prominence_ratio,
                "zero_neck_absolute_width_override_px": zero_neck_absolute_width_override_px,
                "post_neck_small_thin_maximum_area_ratio": post_neck_small_thin_max_ratio,
                "post_neck_small_thin_maximum_internal_width_px": post_neck_small_thin_max_width_px,
                "post_neck_small_thin_removed_regions": case1_data["removed_small_thin_rows"],
                "zero_neck_split_event_count": len(case1_data["split_events"]),
                "zero_neck_cut_segment_count": max(
                    cv2.connectedComponents(
                        case1_data["neck_cut_mask"].astype(np.uint8), connectivity=8
                    )[0]
                    - 1,
                    0,
                ),
                "zero_neck_split_events": case1_data["split_events"],
                "zero_components_after_neck_split": case1_data["split_rows"],
                "zero_components_used_for_intersections": case1_data["intersection_rows"],
                "labeled_zero_candidate_count": len(case1_data["intersection_rows"]),
                "case1_intrusion_reconnection_scope": "vertices adjacent to offending edges only",
                "minimum_polygon_area_ratio_exclusive": min_polygon_ratio,
                "maximum_correction_region_intrusion_ratio": max_correction_intrusion_ratio,
                "acute_corner_maximum_angle_deg": acute_corner_max_angle_deg,
                "acute_corner_bevel_distance_px": acute_corner_bevel_distance_px,
                "acute_corner_squaring_regions": case1_data["corner_squaring_rows"],
                "accepted_polygon_count": sum(row["accepted"] for row in case1_data["rows"]),
                "polygon_regions": case1_data["rows"],
            }
        )
        for index, contour in enumerate(case1_data["contours"]):
            imwrite_gray(item_dir / f"part_contour_level_{index}.png", contour.astype(np.uint8) * 255)
        imwrite_gray(item_dir / "contour_correction_intersection_points.png", case1_data["points"].astype(np.uint8) * 255)
        imwrite_gray(
            item_dir / "case1_expanded_zero_support_mask.png",
            case1_data["expanded_zero_support"].astype(np.uint8) * 255,
        )
        imwrite_gray(
            item_dir / "case1_zero_neck_cut_mask.png",
            case1_data["neck_cut_mask"].astype(np.uint8) * 255,
        )
        imwrite_gray(
            item_dir / "case1_post_neck_zero_area_mask.png",
            case1_data["post_neck_zero_area"].astype(np.uint8) * 255,
        )
        imwrite_gray(item_dir / "case1_zero_line_polygon_area_mask.png", case1_data["polygon"].astype(np.uint8) * 255)
        imwrite_gray(
            item_dir / "case1_before_acute_corner_squaring_mask.png",
            case1_data["polygon_before_corner_squaring"].astype(np.uint8) * 255,
        )
    else:
        assert case2_data is not None
        summary.update(
            {
                "neutral_deviation_limit_mm": neutral_limit_mm,
                "non_neutral_path_allowance_px": allowance_px,
                "edge_preference_radius_px": edge_snap_radius,
                "minimum_line_length_px": min_line_px,
                "internal_holes_excluded_from_separator_sources": True,
                "termination_priority": [
                    "other_outer_start_anchor",
                    "external_background",
                ],
                "closed_loops_accepted_without_open_endpoint_extension": True,
                "start_anchor_smooth_window_px": case2_data["anchor_smooth_window"],
                "start_anchor_count": len(case2_data["anchors"]),
                "start_anchors": [anchor.to_dict() for anchor in case2_data["anchors"]],
                "edge_support_ratio": case2_data["edge_support"],
                "source_regions": case2_data["sources"],
                "line_extension_passes": case2_data["extension_passes"],
                "line_components": case2_data["line_rows"],
            }
        )
        imwrite_gray(item_dir / "strict_neutral_deviation_mask.png", case2_data["strict_neutral"].astype(np.uint8) * 255)
        imwrite_gray(
            item_dir / f"neutral_plus_{allowance_px}px_allowed_mask.png",
            case2_data["allowed"].astype(np.uint8) * 255,
        )
        imwrite_gray(item_dir / "raw_region_voronoi_separator_mask.png", case2_data["raw"].astype(np.uint8) * 255)
        imwrite_gray(item_dir / "allowed_voronoi_separator_mask.png", case2_data["allowed_raw"].astype(np.uint8) * 255)
        imwrite_gray(item_dir / "detected_structural_edges.png", case2_data["structural_edges"])
        imwrite_gray(item_dir / "edge_following_segments.png", case2_data["edge_following"].astype(np.uint8) * 255)
        imwrite_gray(item_dir / "selected_start_points.png", case2_data["starts"].astype(np.uint8) * 255)
    (item_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_summary(output_dir: Path, summaries: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "image",
        "zero_area_ratio_of_part",
        "zero_connected_component_count",
        "selected_method",
        "final_zero_line_px",
        "accepted_polygon_count",
        "start_anchor_count",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "image": Path(row["source_image"]).name,
                    "zero_area_ratio_of_part": row["zero_area_ratio_of_part"],
                    "zero_connected_component_count": row["zero_connected_component_count"],
                    "selected_method": row["selected_method"],
                    "final_zero_line_px": row["final_zero_line_px"],
                    "accepted_polygon_count": row.get("accepted_polygon_count", ""),
                    "start_anchor_count": row.get("start_anchor_count", ""),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case1-limit-ratio", type=float, default=CASE1_LIMIT_RATIO)
    parser.add_argument("--neutral-limit-mm", type=float, default=NEUTRAL_LIMIT_MM)
    parser.add_argument("--non-neutral-allowance", type=int, default=NON_NEUTRAL_ALLOWANCE_PX)
    parser.add_argument("--offset-count", type=int, default=OFFSET_COUNT)
    parser.add_argument(
        "--zero-component-expansion-px",
        type=int,
        default=ZERO_COMPONENT_EXPANSION_PX,
    )
    parser.add_argument(
        "--zero-component-min-ratio",
        type=float,
        default=ZERO_COMPONENT_MIN_RATIO,
    )
    parser.add_argument(
        "--zero-post-neck-min-ratio",
        type=float,
        default=ZERO_POST_NECK_MIN_RATIO,
    )
    parser.add_argument(
        "--zero-neck-max-boundary-gap-px",
        type=float,
        default=ZERO_NECK_MAX_BOUNDARY_GAP_PX,
    )
    parser.add_argument(
        "--zero-neck-min-cut-spacing-px",
        type=int,
        default=ZERO_NECK_MIN_CUT_SPACING_PX,
    )
    parser.add_argument(
        "--zero-neck-cut-margin-px",
        type=int,
        default=ZERO_NECK_CUT_MARGIN_PX,
    )
    parser.add_argument(
        "--zero-neck-min-child-ratio",
        type=float,
        default=ZERO_NECK_MIN_CHILD_RATIO,
    )
    parser.add_argument(
        "--zero-neck-min-prominence-ratio",
        type=float,
        default=ZERO_NECK_MIN_PROMINENCE_RATIO,
    )
    parser.add_argument(
        "--zero-neck-absolute-width-override-px",
        type=float,
        default=ZERO_NECK_ABSOLUTE_WIDTH_OVERRIDE_PX,
    )
    parser.add_argument(
        "--post-neck-small-thin-max-ratio",
        type=float,
        default=POST_NECK_SMALL_THIN_MAX_RATIO,
    )
    parser.add_argument(
        "--post-neck-small-thin-max-width-px",
        type=float,
        default=POST_NECK_SMALL_THIN_MAX_WIDTH_PX,
    )
    parser.add_argument("--edge-snap-radius", type=int, default=EDGE_SNAP_RADIUS_PX)
    parser.add_argument("--min-case1-polygon-ratio", type=float, default=CASE1_MIN_POLYGON_RATIO)
    parser.add_argument(
        "--max-case1-correction-intrusion-ratio",
        type=float,
        default=CASE1_MAX_CORRECTION_INTRUSION_RATIO,
    )
    parser.add_argument(
        "--acute-corner-max-angle-deg",
        type=float,
        default=ACUTE_CORNER_MAX_ANGLE_DEG,
    )
    parser.add_argument(
        "--acute-corner-bevel-distance-px",
        type=float,
        default=ACUTE_CORNER_BEVEL_DISTANCE_PX,
    )
    parser.add_argument("--min-case2-line-px", type=int, default=CASE2_MIN_LINE_PX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        process_one(
            spec,
            args.correction_dir,
            args.output_dir,
            args.case1_limit_ratio,
            args.neutral_limit_mm,
            args.non_neutral_allowance,
            args.offset_count,
            args.zero_component_expansion_px,
            args.zero_component_min_ratio,
            args.zero_post_neck_min_ratio,
            args.zero_neck_max_boundary_gap_px,
            args.zero_neck_min_cut_spacing_px,
            args.zero_neck_cut_margin_px,
            args.zero_neck_min_child_ratio,
            args.zero_neck_min_prominence_ratio,
            args.zero_neck_absolute_width_override_px,
            args.post_neck_small_thin_max_ratio,
            args.post_neck_small_thin_max_width_px,
            args.edge_snap_radius,
            args.min_case1_polygon_ratio,
            args.max_case1_correction_intrusion_ratio,
            args.acute_corner_max_angle_deg,
            args.acute_corner_bevel_distance_px,
            args.min_case2_line_px,
        )
        for spec in SPECS
    ]
    write_summary(args.output_dir, summaries)
    for row in summaries:
        print(
            f"{Path(row['source_image']).name}: zero={row['zero_area_ratio_of_part'] * 100:.2f}%, "
            f"components={row['zero_connected_component_count']}, method={row['selected_method']}, "
            f"final={row['final_zero_line_px']}px"
        )


if __name__ == "__main__":
    main()
