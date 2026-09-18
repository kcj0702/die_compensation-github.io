"""Final image-only zero-line rules (no learned model).

Agreed rules
------------
0. Deviation: the colour map is read by hue; drawing lines, whose grey has no hue, take the
   value of the nearest readable colour instead of the red end of the bar
   (build_deviation_from_color_ramp).
1. Zero points: every sampled point of the product's outer contour whose neighbourhood
   median deviation lies in -0.1..+0.1 mm.
2. Correction regions: the product's deviation map is treated as a normal distribution
   centred on 0 mm. Everything outside +-1 sigma is a correction region, with
   sigma = sqrt(mean(deviation^2)) over the product (the spread about 0, not about the mean).
   ``sigma_source="colorbar"`` instead takes the colour bar's largest end as 3 sigma.
3. Small, noise-like regions are removed (opening, closing and a minimum area).
4. Only regions on the outer contour get a zero line. A region that does not touch the
   contour but lies within ``near_outline_mm`` of it counts as touching: the gap between it
   and the contour is filled as correction first, so it really touches and its line goes
   round its inner side. Without the fill the straight line between its two zero points ran
   through the gap, along the outline itself (scan 4's M2: a 165 mm line on the top edge,
   11 mm above the region). Interior regions get no line (little springback influence
   inside the part).
5. Zero points of a region: walking along the contour away from the region on either side,
   the first zero point met on each side (two points, as in contour_route_zero).
   If no route joins those two points without crossing a correction region - e.g. a point
   sits in a narrow gap between two regions on the contour, as scan 4's M1 did between M1
   and M7 - the walk continues to the next zero points on either side, nearest pairs first,
   and only pairs that the free space actually connects are routed.
6. Line: straight between the two points; if that crosses a correction region, the
   shortest detour around correction regions; then parts of the route close to and parallel
   with a fillet, hole or outline curve are snapped onto it (cad_feature_snap).

Steps 1, 5 and 6 call the existing contour_route_zero / case2_route_selector /
cad_feature_snap code unchanged, so the line drawing is exactly the reviewed logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from zero_line_detection import case2_route_selector as router
from zero_line_detection.contour_route_zero import (
    ContourRouteResult,
    _first_zero_on_side,
    _odd,
    _outlined_text,
    _rasterize,
    select_outer_zero_points,
)


@dataclass(frozen=True)
class FinalZeroLineConfig:
    zero_min_mm: float = -0.1
    zero_max_mm: float = 0.1
    sigma_scale: float = 1.0
    sigma_source: str = "data"          # "data" or "colorbar"
    min_region_ratio: float = 0.0005    # of the product area; smaller regions are noise
    near_outline_mm: float = 20.0       # a region this close to the contour counts as touching it
    near_outline_ratio: float = 0.02    # used instead when the image scale (mm per px) is unknown
    snap_to_cad: bool = True
    max_route_attempts: int = 12        # connected zero-point pairs tried when the first pair has no route

    def validate(self) -> None:
        if not self.zero_min_mm < self.zero_max_mm:
            raise ValueError("zero_min_mm must be below zero_max_mm")
        if self.sigma_scale <= 0:
            raise ValueError("sigma_scale must be positive")
        if self.sigma_source not in ("data", "colorbar"):
            raise ValueError("sigma_source must be 'data' or 'colorbar'")
        if self.min_region_ratio < 0 or self.near_outline_mm < 0 or self.near_outline_ratio < 0:
            raise ValueError("region size and distance limits must not be negative")
        if self.max_route_attempts < 1:
            raise ValueError("max_route_attempts must be at least 1")


def _zeros_on_side(
    zero_by_index: dict[int, dict], contact: np.ndarray, start: int, direction: int
) -> list[dict]:
    """Zero points met walking away from a contact run, nearest first (the first is the classic pick)."""
    count = len(contact)
    found: list[dict] = []
    for step in range(1, count + 1):
        index = (start + direction * step) % count
        if contact[index]:
            continue
        if index in zero_by_index:
            found.append(zero_by_index[index])
    return found


class _FreeSpace:
    """Connected pieces of the space a route may use (outside correction regions, inside the part)."""

    def __init__(self, strict: np.ndarray, search_px: int):
        self.labels = cv2.connectedComponents((strict == 0).astype(np.uint8), connectivity=8)[1]
        self.search = search_px

    def piece(self, point) -> int:
        x, y = (int(round(value)) for value in point)
        height, width = self.labels.shape
        for radius in range(self.search + 1):
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            window = self.labels[y0:y1, x0:x1]
            if window.any():
                values, counts = np.unique(window[window > 0], return_counts=True)
                return int(values[np.argmax(counts)])
        return 0


def _bridge_near_regions(
    correction: np.ndarray,
    contour_px: np.ndarray,
    outer_silhouette: np.ndarray,
    touching_px: int,
    near_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(correction with gaps filled, the filled pixels) for regions within near_px of the contour.

    The fill is the strip lying within the gap width of both the region and the contour, so it
    closes the gap where the region faces the contour without spreading along the whole edge.
    Only strip pixels connected to the region are kept (loose pieces along the edge would
    otherwise become new regions), and a band from the region's nearest pixel to the nearest
    contour point guarantees the region really reaches the contour.
    """
    correction = correction.astype(bool)
    silhouette = outer_silhouette > 0
    to_outline = cv2.distanceTransform(silhouette.astype(np.uint8), cv2.DIST_L2, 5)
    bridges = np.zeros(correction.shape, dtype=bool)
    band = max(3, int(touching_px))
    for region in router.extract_regions(correction.astype(np.uint8) * 255):
        component = region["component_mask"] > 0
        to_region = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
        along = to_region[contour_px[:, 1], contour_px[:, 0]]
        nearest = float(along.min())
        if not touching_px < nearest <= near_px:
            continue
        gap = nearest + 2.0
        strip = silhouette & ~component & (to_outline <= gap) & (to_region <= gap)
        target = contour_px[int(np.argmin(along))]
        ys, xs = np.nonzero(component)
        closest = int(np.argmin((xs - target[0]) ** 2 + (ys - target[1]) ** 2))
        connector = np.zeros(correction.shape, dtype=np.uint8)
        cv2.line(connector, (int(xs[closest]), int(ys[closest])), (int(target[0]), int(target[1])), 1, band)
        strip |= (connector > 0) & silhouette & ~component
        count, labels = cv2.connectedComponents((strip | component).astype(np.uint8), connectivity=8)
        attached = np.isin(labels, np.unique(labels[component]))
        bridges |= strip & attached
    return correction | bridges, bridges


def _candidate_pairs(first_side: list[dict], second_side: list[dict], free: _FreeSpace):
    """(first, second, steps skipped) ordered by how far past the first pick they are."""
    pieces_first = [free.piece(item["point"]) for item in first_side]
    pieces_second = [free.piece(item["point"]) for item in second_side]
    pairs = []
    for i, first in enumerate(first_side):
        for j, second in enumerate(second_side):
            if first["sample_index"] == second["sample_index"]:
                continue
            same = pieces_first[i] != 0 and pieces_first[i] == pieces_second[j]
            if i == 0 and j == 0 or same:
                pairs.append((i + j, max(i, j), i, j, first, second))
    pairs.sort(key=lambda item: item[:4])
    return [(first, second, (i, j)) for _, _, i, j, first, second in pairs]


def build_deviation_from_color_ramp(
    image_rgb: np.ndarray,
    ramp_top_to_bottom: np.ndarray,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """(deviation mm per pixel, product mask, report) with drawing lines read as their surroundings.

    The colour map is read by hue. Drawing lines (CAD edges, often two or three side by side at
    hole rims and fillets) are grey: saturation ~0, so their hue is 0, which is the red end of
    the colour bar. build_common_from_color_ramp flags them as unreadable but keeps that value,
    and where lines are thicker than its 5x5 median filter they stay at +vmax and spread onto the
    fringe pixels beside them. On scan 3 that turned 14,616 line pixels into +2.0 mm, drew
    correction bands along a hole rim and a fillet, and inflated sigma.

    Here every unreadable pixel takes the value of the nearest readable colour pixel before any
    filtering. Thick grey *surfaces* (out of the colour bar range, found as in the reviewed code
    by a 7x7 opening) still take the out-of-range value with the sign of their surroundings.
    """
    from zero_line_detection.adaptive_bundle import generate_preview as kdt
    from zero_line_detection.adaptive_bundle.generate_between_signs_preview import detect_unmapped_gray
    from zero_line_detection.adaptive_bundle.generate_correction_only_3pct_preview import (
        CORRECTION_THRESHOLD_MM,
        assign_gray_by_nearest_mapped_sign,
    )

    raw, valid = kdt.map_deviation(image_rgb, ramp_top_to_bottom, vmin, vmax)
    part = kdt.detect_part_mask(image_rgb)
    readable = part & valid
    if not readable.any():
        raise ValueError("no colour-map pixel could be read inside the product")
    gray_surface = detect_unmapped_gray(image_rgb, part, readable)

    _, nearest = cv2.distanceTransformWithLabels(
        (~readable).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    lookup = np.zeros(int(nearest.max()) + 1, dtype=np.float32)
    ys, xs = np.nonzero(readable)
    lookup[nearest[ys, xs]] = raw[ys, xs]
    filled = np.where(readable, raw, lookup[nearest]).astype(np.float32)
    filled = cv2.medianBlur(filled, 5)

    values, gray_positive, gray_negative, _ = assign_gray_by_nearest_mapped_sign(
        filled,
        readable,
        gray_surface,
        max(float(vmax), CORRECTION_THRESHOLD_MM) + 0.01,
        min(float(vmin), -CORRECTION_THRESHOLD_MM) - 0.01,
    )
    values = np.where(part, values, np.nan).astype(np.float32)
    unreadable = part & ~readable & ~gray_surface
    report = {
        "unreadable_line_px": int(unreadable.sum()),
        "unreadable_share_of_product": float(unreadable.sum() / max(1, part.sum())),
        "gray_surface_px": int(gray_surface.sum()),
        "unreadable_rule": "filled from the nearest readable colour pixel before filtering",
    }
    return values, part, report


def deviation_sigma(
    values_mm: np.ndarray,
    part_mask: np.ndarray,
    *,
    source: str = "data",
    colorbar_range: tuple[float, float] | None = None,
) -> float:
    """Spread of the deviation about 0 mm, the centre of the assumed normal distribution."""
    if source == "colorbar":
        if colorbar_range is None:
            raise ValueError("sigma_source='colorbar' needs colorbar_range")
        return max(abs(float(colorbar_range[0])), abs(float(colorbar_range[1]))) / 3.0
    values = np.asarray(values_mm, dtype=np.float64)[part_mask.astype(bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no deviation values inside the product")
    return float(np.sqrt(np.mean(values ** 2)))


def build_sigma_correction_mask(
    values_mm: np.ndarray,
    part_mask: np.ndarray,
    threshold_mm: float,
    min_region_ratio: float,
) -> tuple[np.ndarray, dict]:
    """|deviation| > threshold inside the product, with noise-sized pieces removed."""
    part = part_mask.astype(bool)
    values = np.asarray(values_mm, dtype=np.float64)
    raw = part & np.isfinite(values) & (np.abs(values) > threshold_mm)
    short = min(part.shape)
    opened = cv2.morphologyEx(
        raw.astype(np.uint8), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    close_size = _odd(round(short * 0.005))
    cleaned = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    ).astype(bool) & part
    minimum_area = max(20, int(round(np.count_nonzero(part) * min_region_ratio)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    kept = np.zeros(part.shape, dtype=bool)
    removed = 0
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            kept[labels == label] = True
        else:
            removed += 1
    return kept, {
        "raw_share_of_product": float(raw.sum() / max(1, part.sum())),
        "kept_share_of_product": float(kept.sum() / max(1, part.sum())),
        "noise_regions_removed": removed,
        "minimum_region_area_px": minimum_area,
    }


def construct_final_zero_lines(
    image_rgb: np.ndarray,
    values_mm: np.ndarray,
    part_mask: np.ndarray,
    *,
    cad_feature_lines: list[dict] | None = None,
    config: FinalZeroLineConfig | None = None,
    mm_per_px: float | None = None,
    colorbar_range: tuple[float, float] | None = None,
) -> tuple[ContourRouteResult, dict]:
    """Zero lines by the agreed rules. Returns (result, report)."""
    config = config or FinalZeroLineConfig()
    config.validate()
    part = part_mask.astype(bool)
    sigma = deviation_sigma(values_mm, part, source=config.sigma_source, colorbar_range=colorbar_range)
    threshold = config.sigma_scale * sigma
    correction, mask_report = build_sigma_correction_mask(values_mm, part, threshold, config.min_region_ratio)

    contour, zero_points = select_outer_zero_points(
        values_mm, part, minimum_mm=config.zero_min_mm, maximum_mm=config.zero_max_mm
    )
    short = min(part.shape)
    proximity = max(4, int(round(short * 0.006)))          # "touching", as in contour_route_zero
    if mm_per_px:
        near_px = max(proximity, int(round(config.near_outline_mm / float(mm_per_px))))
    else:
        near_px = max(proximity, int(round(short * config.near_outline_ratio)))

    zero_by_index = {int(item["sample_index"]): item for item in zero_points}
    rounded = np.rint(contour).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, part.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, part.shape[0] - 1)
    outer_silhouette = np.zeros(part.shape, dtype=np.uint8)
    cv2.fillPoly(outer_silhouette, [rounded.astype(np.int32)], 255, cv2.LINE_8)

    # Rule 4: fill the gap between a near-outline region and the contour, then number regions.
    correction, bridges = _bridge_near_regions(correction, rounded, outer_silhouette, proximity, near_px)
    regions = router.extract_regions(correction.astype(np.uint8) * 255)
    planning, strict, route_clearance = router.build_route_obstacles(correction.astype(np.uint8) * 255, outer_silhouette)
    anchor_cache: dict = {}
    free_space = _FreeSpace(strict, route_clearance + 2)

    lines: list[dict] = []
    warnings: list[str] = []
    region_rows: list[dict] = []
    for region in regions:
        component = region["component_mask"] > 0
        distance = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
        contour_distance = distance[rounded[:, 1], rounded[:, 0]]
        nearest = float(contour_distance.min())
        row = {"label": region["label"], "area_px": int(region["area_px"]),
               "centroid": [float(region["centroid"][0]), float(region["centroid"][1])],
               "distance_to_outline_px": round(nearest, 1)}
        if nearest <= proximity:
            bridged = bool((component & bridges).any())
            row["kind"] = "near_outline" if bridged else "touching_outline"
            if bridged:
                row["gap_filled_px"] = int((component & bridges).sum())
            contact = contour_distance <= proximity
        elif nearest <= near_px:
            row["kind"] = "near_outline"
            # The stretch of contour facing the region, as wide as if it touched.
            contact = contour_distance <= nearest + proximity
        else:
            row["kind"] = "interior_no_line"
            region_rows.append(row)
            continue
        runs = router.circular_true_runs(contact)
        contact_run = max(runs, key=len)
        first_side = _zeros_on_side(zero_by_index, contact, contact_run[0], -1)
        second_side = _zeros_on_side(zero_by_index, contact, contact_run[-1], 1)
        first = _first_zero_on_side(zero_by_index, contact, contact_run[0], -1)
        second = _first_zero_on_side(zero_by_index, contact, contact_run[-1], 1)
        if first is None or second is None or first["sample_index"] == second["sample_index"]:
            row["kind"] = "unresolved"
            warnings.append(f"{region['label']}: 보정영역 양쪽에서 서로 다른 제로점을 찾지 못했습니다.")
            region_rows.append(row)
            continue
        found = None
        failures: list[str] = []
        for attempt, (first, second, skipped) in enumerate(_candidate_pairs(first_side, second_side, free_space)):
            if attempt >= config.max_route_attempts:
                break
            first_xy = tuple(np.rint(first["point"]).astype(int).tolist())
            second_xy = tuple(np.rint(second["point"]).astype(int).tolist())
            if router.rasterized_segment_is_clear(strict, first_xy, second_xy):
                found = (first, second, skipped, [list(first_xy), list(second_xy)], "direct_clear_line")
                break
            try:
                route = router.route_pair(first_xy, second_xy, planning, strict, anchor_cache)
            except ValueError as error:
                failures.append(f"{first_xy}-{second_xy}: {error}")
                continue
            found = (first, second, skipped, route["path_points"], route["routing_method"])
            break
        if found is None:
            row["kind"] = "unresolved"
            warnings.append(f"{region['label']}: 보정영역 우회 경로를 찾지 못했습니다 ({'; '.join(failures[:2])}).")
            region_rows.append(row)
            continue
        first, second, skipped, path, method = found
        lines.append({
            "id": len(lines) + 1,
            "kind": "boundary_connection",
            "region": region["label"],
            "regionRelation": row["kind"],
            "closed": False,
            "method": method,
            "zeroPointValuesMm": [first["value_mm"], second["value_mm"]],
            # (0, 0) is the first zero point on each side; more means nearer ones had no route.
            "zeroPointsSkipped": list(skipped),
            "points": path,
        })
        row["line_id"] = len(lines)
        if skipped != (0, 0):
            row["zero_points_skipped"] = list(skipped)
            row["skip_reason"] = failures[0] if failures else "first pair not connected"
        region_rows.append(row)

    snap_records: list[dict] = []
    if cad_feature_lines and config.snap_to_cad:
        from zero_line_detection.cad_feature_snap import snap_boundary_lines_to_cad
        lines, snap_records = snap_boundary_lines_to_cad(lines, cad_feature_lines, part.shape, obstacle_mask=strict)

    result = ContourRouteResult(
        correction_mask=correction,
        outer_contour=contour,
        zero_points=zero_points,
        lines=lines,
        mask=_rasterize(lines, part.shape),
        warnings=warnings,
        snap_records=snap_records,
        proximity_px=proximity,
    )
    report = {
        "config": asdict(config),
        "sigma_mm": sigma,
        "threshold_mm": threshold,
        "near_outline_px": near_px,
        "touching_px": proximity,
        "mm_per_px": mm_per_px,
        **mask_report,
        "gap_filled_px": int(bridges.sum()),
        "regions": region_rows,
    }
    return result, report


def zero_point_labels(lines: list[dict], merge_px: float = 8.0) -> list[dict]:
    """Z1, Z2, ... in order of first appearance, as render_selection_preview numbers them."""
    labels: list[dict] = []
    for line in lines:
        if line.get("kind") != "boundary_connection":
            continue
        points = np.asarray(line["points"], dtype=np.int32)
        values = line.get("zeroPointValuesMm", [float("nan"), float("nan")])
        for point, value in ((points[0], values[0]), (points[-1], values[1])):
            match = next((item for item in labels
                          if np.hypot(point[0] - item["point"][0], point[1] - item["point"][1]) <= merge_px), None)
            if match is None:
                labels.append({"label": f"Z{len(labels) + 1}", "point": [int(point[0]), int(point[1])],
                               "value_mm": float(value), "regions": [line.get("region")]})
            else:
                match["regions"].append(line.get("region"))
    return labels


def render_final_preview(
    image_rgb: np.ndarray,
    part_mask: np.ndarray,
    result: ContourRouteResult,
    report: dict,
) -> np.ndarray:
    """Same layout as render_selection_preview, explaining the final rules."""
    preview = image_rgb.copy()
    correction = result.correction_mask.astype(bool)
    tint = preview.copy()
    tint[correction] = (255, 55, 55)
    preview = cv2.addWeighted(preview, 0.74, tint, 0.26, 0.0)
    contours, _ = cv2.findContours(correction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    layer = preview.copy()
    cv2.drawContours(layer, contours, -1, (255, 65, 65), 1, cv2.LINE_AA)
    outer = np.rint(result.outer_contour).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(layer, [outer], True, (25, 25, 25), 1, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.42, layer, 0.58, 0.0)

    layer = preview.copy()
    for zero in result.zero_points:
        cv2.circle(layer, tuple(np.rint(zero["point"]).astype(int).tolist()), 2, (35, 200, 65), cv2.FILLED, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.42, layer, 0.58, 0.0)

    layer = preview.copy()
    for line in result.lines:
        points = np.asarray(line["points"], dtype=np.int32).reshape(-1, 1, 2)
        if len(points) < 2:
            continue
        colour = (0, 205, 255) if line.get("method") == "direct_clear_line" else (255, 0, 220)
        cv2.polylines(layer, [points], False, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.polylines(layer, [points], False, colour, 2, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.38, layer, 0.62, 0.0)

    zeros = zero_point_labels(result.lines)
    layer = preview.copy()
    for zero in zeros:
        point = tuple(zero["point"])
        cv2.circle(layer, point, 8, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(layer, point, 6, (0, 205, 255), 2, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.34, layer, 0.66, 0.0)
    for zero in zeros:
        _outlined_text(preview, zero["label"], (zero["point"][0] + 7, max(15, zero["point"][1] - 7)), scale=0.37)

    colours = {"touching_outline": (255, 0, 210), "near_outline": (150, 60, 255),
               "interior_no_line": (135, 135, 135), "unresolved": (225, 35, 35)}
    rows = {row["label"]: row for row in report["regions"]}
    layer = preview.copy()
    texts = []
    for region in router.extract_regions(correction.astype(np.uint8) * 255):
        kind = rows.get(region["label"], {}).get("kind", "interior_no_line")
        colour = colours[kind]
        centroid = np.asarray(region["centroid"], dtype=np.float64)
        ys, xs = np.where(region["component_mask"] > 0)
        nearest = int(np.argmin((xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2))
        x, y = int(xs[nearest]), int(ys[nearest])
        diamond = np.asarray([(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(layer, [diamond], True, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.polylines(layer, [diamond], True, colour, 2, cv2.LINE_AA)
        if kind == "unresolved":
            cv2.line(layer, (x - 6, y - 6), (x + 6, y + 6), colour, 2, cv2.LINE_AA)
            cv2.line(layer, (x + 6, y - 6), (x - 6, y + 6), colour, 2, cv2.LINE_AA)
        texts.append((region["label"], (x + 8, y - 7), colour))
    preview = cv2.addWeighted(preview, 0.38, layer, 0.62, 0.0)
    for text, origin, colour in texts:
        _outlined_text(preview, text, origin, scale=0.40, colour=colour)

    ys, xs = np.where(part_mask.astype(bool))
    if xs.size:
        margin = 30
        x0, x1 = max(0, int(xs.min()) - margin), min(preview.shape[1], int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(preview.shape[0], int(ys.max()) + margin + 1)
        preview = preview[y0:y1, x0:x1]

    header = 176
    canvas = np.full((preview.shape[0] + header + 18, max(preview.shape[1] + 36, 1140), 3), 255, dtype=np.uint8)
    canvas[header:header + preview.shape[0], 18:18 + preview.shape[1]] = preview
    cv2.rectangle(canvas, (18, 15), (450, 158), (70, 70, 70), 1)
    cv2.rectangle(canvas, (470, 15), (canvas.shape[1] - 18, 158), (70, 70, 70), 1)
    cv2.putText(canvas, "Final zero line (image rules)", (32, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    legend = [
        ((35, 200, 65), "outer zero candidate (-0.10 to +0.10 mm)"),
        ((255, 0, 210), "M: region on the outline -> zero line"),
        ((150, 60, 255), "M: region near the outline -> zero line"),
        ((135, 135, 135), "M: interior region -> no line"),
        ((255, 0, 220), "route around correction (magenta) / clear (cyan)"),
    ]
    for index, (colour, text) in enumerate(legend):
        y = 64 + index * 19
        cv2.line(canvas, (32, y), (55, y), colour, 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (64, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)

    kinds = [row["kind"] for row in report["regions"]]
    direct = sum(line.get("method") == "direct_clear_line" for line in result.lines)
    snapped = sum(bool(line.get("cadSnapped")) for line in result.lines)
    near_mm = report["near_outline_px"] * report["mm_per_px"] if report.get("mm_per_px") else None
    summary = [
        f"correction = |deviation| > {report['config']['sigma_scale']:g} sigma = {report['threshold_mm']:.2f} mm"
        f"  (sigma about 0 mm, {report['config']['sigma_source']}: {report['sigma_mm']:.2f} mm)",
        f"regions: {len(kinds)}  on outline {kinds.count('touching_outline')}  near outline {kinds.count('near_outline')}"
        f"  interior (no line) {kinds.count('interior_no_line')}  unresolved (red X) {kinds.count('unresolved')}",
        f"noise regions removed: {report['noise_regions_removed']}   near outline: <= {report['near_outline_px']} px"
        + (f" ({near_mm:.0f} mm)" if near_mm else ""),
        f"zero candidates: {len(result.zero_points)}   selected zero points: {len(zeros)}   lines: {len(result.lines)}",
        f"straight: {direct}   detour: {len(result.lines) - direct}   CAD-snapped lines: {snapped} ({len(result.snap_records)} sections)",
    ]
    cv2.putText(canvas, "Selection evidence", (488, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
    for index, text in enumerate(summary):
        cv2.putText(canvas, text, (488, 68 + index * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


__all__ = [
    "FinalZeroLineConfig",
    "build_deviation_from_color_ramp",
    "build_sigma_correction_mask",
    "construct_final_zero_lines",
    "deviation_sigma",
    "render_final_preview",
    "zero_point_labels",
]
