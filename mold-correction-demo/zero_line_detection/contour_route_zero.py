"""Zero-line construction from outer zero points and HSV correction regions.

The algorithm intentionally stops before cosmetic smoothing:

* outer-boundary samples in -0.1..+0.1 mm are zero-point candidates;
* non-green material is a correction region;
* a boundary-attached region uses the first zero point on either side;
* those points stay a straight line when clear, otherwise an obstacle-avoiding
  shortest route is used;
* an interior correction region receives a closed surrounding contour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from zero_line_detection import case2_route_selector as router


ZERO_MIN_MM = -0.1
ZERO_MAX_MM = 0.1
GREEN_HUE_MIN = 35
GREEN_HUE_MAX = 90
GREEN_SATURATION_MIN = 45
GREEN_VALUE_MIN = 35
MIN_CORRECTION_RATIO = 0.0005


@dataclass
class ContourRouteResult:
    correction_mask: np.ndarray
    outer_contour: np.ndarray
    zero_points: list[dict]
    lines: list[dict]
    mask: np.ndarray
    warnings: list[str] = field(default_factory=list)
    snap_records: list[dict] = field(default_factory=list)
    proximity_px: int = 0


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _largest_outer_contour(part_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        part_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("제품 외곽선을 찾지 못했습니다")
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(contour) < 3:
        raise ValueError("제품 외곽선의 점이 부족합니다")
    return contour


def _resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    keep = lengths > 1e-6
    points = points[keep]
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    total = float(lengths.sum())
    if total <= 0:
        return points
    count = max(8, int(np.ceil(total / max(spacing, 1.0))))
    targets = np.linspace(0.0, total, count, endpoint=False)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(points) - 1)
    ratios = (targets - cumulative[indices]) / np.maximum(lengths[indices], 1e-6)
    return points[indices] + (following[indices] - points[indices]) * ratios[:, None]


def build_hsv_correction_mask(image_rgb: np.ndarray, part_mask: np.ndarray) -> np.ndarray:
    """Classify non-green material as correction, filling dark drawing strokes.

    Saturated coloured pixels are classified directly in HSV. Dark/grey pixels
    inside the product are assigned to whichever valid class is spatially
    nearer, preventing thin CAD strokes from becoming false correction bands.
    """
    part = part_mask.astype(bool)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    coloured = part & (saturation >= GREEN_SATURATION_MIN) & (value >= GREEN_VALUE_MIN)
    green_seed = coloured & (hue >= GREEN_HUE_MIN) & (hue <= GREEN_HUE_MAX)
    correction_seed = coloured & ~green_seed

    if not correction_seed.any():
        return np.zeros(part.shape, dtype=bool)
    if not green_seed.any():
        classified = part.copy()
    else:
        distance_green = cv2.distanceTransform((~green_seed).astype(np.uint8), cv2.DIST_L2, 5)
        distance_correction = cv2.distanceTransform((~correction_seed).astype(np.uint8), cv2.DIST_L2, 5)
        classified = part & (distance_correction < distance_green)
        classified[correction_seed] = True
        classified[green_seed] = False

    short = min(part.shape)
    opened = cv2.morphologyEx(
        classified.astype(np.uint8), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    close_size = _odd(round(short * 0.005))
    cleaned = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    ).astype(bool) & part

    minimum_area = max(20, int(round(np.count_nonzero(part) * MIN_CORRECTION_RATIO)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8), connectivity=8
    )
    kept = np.zeros(part.shape, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            kept[labels == label] = True
    return kept


def select_outer_zero_points(
    values_mm: np.ndarray,
    part_mask: np.ndarray,
    *,
    minimum_mm: float = ZERO_MIN_MM,
    maximum_mm: float = ZERO_MAX_MM,
) -> tuple[np.ndarray, list[dict]]:
    """Return a uniformly sampled outer contour and all -0.1..+0.1 mm samples."""
    short = min(part_mask.shape)
    contour = _resample_closed(
        _largest_outer_contour(part_mask), max(2.0, short * 0.003)
    )
    radius = max(2, int(round(short * 0.003)))
    height, width = part_mask.shape
    samples: list[dict] = []
    for index, point in enumerate(contour):
        x, y = np.rint(point).astype(int)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        local_part = part_mask[y0:y1, x0:x1].astype(bool)
        local_values = values_mm[y0:y1, x0:x1]
        finite = local_part & np.isfinite(local_values)
        value_mm = float(np.median(local_values[finite])) if finite.any() else float("nan")
        if np.isfinite(value_mm) and minimum_mm <= value_mm <= maximum_mm:
            samples.append({
                "sample_index": index,
                "point": [float(point[0]), float(point[1])],
                "value_mm": value_mm,
            })
    return contour, samples


def _first_zero_on_side(
    zero_by_index: dict[int, dict], contact: np.ndarray, start: int, direction: int
) -> dict | None:
    count = len(contact)
    for step in range(1, count + 1):
        index = (start + direction * step) % count
        if contact[index]:
            continue
        if index in zero_by_index:
            return zero_by_index[index]
    return None


def _surrounding_line(
    component: np.ndarray, part_mask: np.ndarray, clearance: int
) -> list[list[int]] | None:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * clearance + 1, 2 * clearance + 1)
    )
    expanded = cv2.dilate(component.astype(np.uint8), kernel)
    expanded = cv2.bitwise_and(expanded, part_mask.astype(np.uint8))
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea)[:, 0, :]
    if len(points) < 3:
        return None
    closed = np.vstack((points, points[:1]))
    return closed.astype(int).tolist()


def _rasterize(lines: list[dict], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        points = np.asarray(line["points"], dtype=np.int32).reshape(-1, 1, 2)
        if len(points) >= 2:
            cv2.polylines(mask, [points], bool(line.get("closed")), 255, 4, cv2.LINE_8)
    return mask.astype(bool)


def _outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.48,
    colour: tuple[int, int, int] = (20, 20, 20),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness + 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        colour, thickness, cv2.LINE_AA,
    )


def render_selection_preview(
    image_rgb: np.ndarray,
    part_mask: np.ndarray,
    result: ContourRouteResult,
) -> np.ndarray:
    """Render an audit preview explaining why every zero line was selected."""
    preview = image_rgb.copy()
    correction = result.correction_mask.astype(bool)

    # Red tint and magenta boundary make the HSV decision visible without
    # hiding the original colour map beneath it.
    tint = preview.copy()
    tint[correction] = (255, 55, 55)
    preview = cv2.addWeighted(preview, 0.74, tint, 0.26, 0.0)
    correction_contours, _ = cv2.findContours(
        correction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boundary_overlay = preview.copy()
    cv2.drawContours(boundary_overlay, correction_contours, -1, (255, 65, 65), 1, cv2.LINE_AA)
    outer = np.rint(result.outer_contour).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(boundary_overlay, [outer], True, (25, 25, 25), 1, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.42, boundary_overlay, 0.58, 0.0)

    # All candidates are small green dots. Enlarged ringed candidates are the
    # actual two endpoints chosen for a boundary-attached correction region.
    candidate_overlay = preview.copy()
    for zero in result.zero_points:
        point = tuple(np.rint(zero["point"]).astype(int).tolist())
        cv2.circle(candidate_overlay, point, 2, (35, 200, 65), cv2.FILLED, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.42, candidate_overlay, 0.58, 0.0)

    selected_endpoints: list[tuple[tuple[int, int], float]] = []
    route_colours = {
        "direct": (0, 205, 255),
        "detour": (255, 0, 220),
        "enclosure": (255, 145, 0),
    }
    route_overlay = preview.copy()
    for line in result.lines:
        points = np.asarray(line["points"], dtype=np.int32).reshape(-1, 1, 2)
        if len(points) < 2:
            continue
        if line["kind"] == "interior_enclosure":
            colour = route_colours["enclosure"]
        elif line.get("method") == "direct_clear_line":
            colour = route_colours["direct"]
        else:
            colour = route_colours["detour"]
        cv2.polylines(route_overlay, [points], bool(line.get("closed")), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.polylines(route_overlay, [points], bool(line.get("closed")), colour, 2, cv2.LINE_AA)
        if line["kind"] == "boundary_connection":
            values = line.get("zeroPointValuesMm", [float("nan"), float("nan")])
            selected_endpoints.extend(
                [
                    (tuple(points[0, 0].tolist()), float(values[0])),
                    (tuple(points[-1, 0].tolist()), float(values[1])),
                ]
            )
    preview = cv2.addWeighted(preview, 0.38, route_overlay, 0.62, 0.0)

    unique_endpoints: list[tuple[tuple[int, int], float]] = []
    for point, value in selected_endpoints:
        if all(np.hypot(point[0] - old[0][0], point[1] - old[0][1]) > 8 for old in unique_endpoints):
            unique_endpoints.append((point, value))
    endpoint_overlay = preview.copy()
    for point, _value in unique_endpoints:
        cv2.circle(endpoint_overlay, point, 8, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(endpoint_overlay, point, 6, (0, 205, 255), 2, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.34, endpoint_overlay, 0.66, 0.0)
    for index, (point, _value) in enumerate(unique_endpoints, start=1):
        _outlined_text(
            preview, f"Z{index}",
            (point[0] + 7, max(15, point[1] - 7)), scale=0.37,
        )

    regions = router.extract_regions(correction.astype(np.uint8) * 255)
    failed_labels = {warning.split(":", 1)[0] for warning in result.warnings if ":" in warning}
    region_overlay = preview.copy()
    region_labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for region in regions:
        centroid = np.asarray(region["centroid"], dtype=np.float64)
        component_y, component_x = np.where(region["component_mask"] > 0)
        nearest = int(
            np.argmin((component_x - centroid[0]) ** 2 + (component_y - centroid[1]) ** 2)
        )
        center = (int(component_x[nearest]), int(component_y[nearest]))
        x, y = center
        diamond = np.asarray(
            [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)], dtype=np.int32
        ).reshape(-1, 1, 2)
        colour = (225, 35, 35) if region["label"] in failed_labels else (255, 0, 210)
        cv2.polylines(region_overlay, [diamond], True, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.polylines(region_overlay, [diamond], True, colour, 2, cv2.LINE_AA)
        if region["label"] in failed_labels:
            cv2.line(region_overlay, (x - 6, y - 6), (x + 6, y + 6), colour, 2, cv2.LINE_AA)
            cv2.line(region_overlay, (x + 6, y - 6), (x - 6, y + 6), colour, 2, cv2.LINE_AA)
        region_labels.append((region["label"], (x + 8, y - 7), colour))
    preview = cv2.addWeighted(preview, 0.38, region_overlay, 0.62, 0.0)
    for label, origin, colour in region_labels:
        _outlined_text(preview, label, origin, scale=0.40, colour=colour)

    ys, xs = np.where(part_mask.astype(bool))
    if xs.size:
        margin = 22
        x0, x1 = max(0, int(xs.min()) - margin), min(preview.shape[1], int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(preview.shape[0], int(ys.max()) + margin + 1)
        preview = preview[y0:y1, x0:x1]

    header_height = 156
    canvas = np.full(
        (preview.shape[0] + header_height + 18, preview.shape[1] + 36, 3),
        255,
        dtype=np.uint8,
    )
    canvas[header_height : header_height + preview.shape[0], 18 : 18 + preview.shape[1]] = preview
    cv2.rectangle(canvas, (18, 15), (390, 136), (70, 70, 70), 1)
    cv2.rectangle(canvas, (410, 15), (canvas.shape[1] - 18, 136), (70, 70, 70), 1)
    cv2.putText(
        canvas, "Zero-line selection overview", (32, 43),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA,
    )
    legend_left = [
        ((35, 200, 65), "outer zero candidate  (-0.10 to +0.10 mm)"),
        ((0, 205, 255), "selected endpoint / clear direct line"),
        ((255, 0, 220), "shortest correction-avoiding route"),
        ((255, 145, 0), "closed enclosure for an interior region"),
    ]
    for index, (colour, label) in enumerate(legend_left):
        y = 64 + index * 20
        cv2.line(canvas, (32, y), (55, y), colour, 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (64, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)

    direct_count = sum(line.get("method") == "direct_clear_line" for line in result.lines)
    detour_count = sum(
        line["kind"] == "boundary_connection" and line.get("method") != "direct_clear_line"
        for line in result.lines
    )
    enclosure_count = sum(line["kind"] == "interior_enclosure" for line in result.lines)
    summary_x = 428
    cv2.putText(
        canvas, "Selection evidence", (summary_x, 43),
        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA,
    )
    summary = [
        f"red shade + red edge: HSV non-green correction regions ({len(regions)})",
        f"green candidates: {len(result.zero_points)}   selected endpoints: {len(unique_endpoints)}",
        f"direct: {direct_count}   detour: {detour_count}   enclosure: {enclosure_count}",
        f"unresolved regions (red X): {len(failed_labels)}",
    ]
    for index, label in enumerate(summary):
        cv2.putText(
            canvas, label, (summary_x, 67 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA,
        )
    return canvas


def construct_zero_lines(
    image_rgb: np.ndarray,
    values_mm: np.ndarray,
    part_mask: np.ndarray,
    *,
    cad_feature_lines: list[dict] | None = None,
) -> ContourRouteResult:
    """Construct the requested unsmoothed zero lines."""
    part = part_mask.astype(bool)
    correction = build_hsv_correction_mask(image_rgb, part)
    contour, zero_points = select_outer_zero_points(values_mm, part)
    regions = router.extract_regions(correction.astype(np.uint8) * 255)
    proximity = max(4, int(round(min(part.shape) * 0.006)))
    clearance = max(3, int(round(min(part.shape) * 0.004)))
    warnings: list[str] = []
    lines: list[dict] = []

    if not zero_points and regions:
        warnings.append("외곽선에서 -0.1~+0.1 mm 제로점을 찾지 못했습니다.")

    zero_by_index = {int(item["sample_index"]): item for item in zero_points}
    rounded = np.rint(contour).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, part.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, part.shape[0] - 1)

    # Routing follows the user's straight-line rule inside the product's outer
    # silhouette. Interior openings are therefore traversable; only correction
    # regions and the area outside the outer contour are obstacles.
    outer_silhouette = np.zeros(part.shape, dtype=np.uint8)
    cv2.fillPoly(outer_silhouette, [rounded.astype(np.int32)], 255, cv2.LINE_8)

    planning, strict, _route_clearance = router.build_route_obstacles(
        correction.astype(np.uint8) * 255, outer_silhouette
    )
    anchor_cache: dict = {}

    for region in regions:
        component = region["component_mask"] > 0
        distance = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)
        contour_distance = distance[rounded[:, 1], rounded[:, 0]]
        contact = contour_distance <= proximity
        runs = router.circular_true_runs(contact)

        if not runs:
            points = _surrounding_line(component, part, clearance)
            if points is None:
                warnings.append(f"{region['label']}: 내부 보정영역을 둘러싸는 선을 만들지 못했습니다.")
                continue
            lines.append({
                "id": len(lines) + 1,
                "kind": "interior_enclosure",
                "region": region["label"],
                "closed": True,
                "points": points,
            })
            continue

        contact_run = max(runs, key=len)
        first = _first_zero_on_side(zero_by_index, contact, contact_run[0], -1)
        second = _first_zero_on_side(zero_by_index, contact, contact_run[-1], 1)
        if first is None or second is None or first["sample_index"] == second["sample_index"]:
            warnings.append(
                f"{region['label']}: 보정영역 양쪽에서 서로 다른 제로점을 찾지 못했습니다."
            )
            continue

        first_xy = tuple(np.rint(first["point"]).astype(int).tolist())
        second_xy = tuple(np.rint(second["point"]).astype(int).tolist())
        if router.rasterized_segment_is_clear(strict, first_xy, second_xy):
            path = [list(first_xy), list(second_xy)]
            method = "direct_clear_line"
        else:
            try:
                route = router.route_pair(
                    first_xy, second_xy, planning, strict, anchor_cache
                )
            except ValueError as error:
                warnings.append(f"{region['label']}: 보정영역 우회 경로를 찾지 못했습니다 ({error}).")
                continue
            path = route["path_points"]
            method = route["routing_method"]
        lines.append({
            "id": len(lines) + 1,
            "kind": "boundary_connection",
            "region": region["label"],
            "closed": False,
            "method": method,
            "zeroPointValuesMm": [first["value_mm"], second["value_mm"]],
            "points": path,
        })

    snap_records: list[dict] = []
    if cad_feature_lines:
        from zero_line_detection.cad_feature_snap import snap_boundary_lines_to_cad

        lines, snap_records = snap_boundary_lines_to_cad(
            lines,
            cad_feature_lines,
            part.shape,
            obstacle_mask=strict,
        )

    return ContourRouteResult(
        correction_mask=correction,
        outer_contour=contour,
        zero_points=zero_points,
        lines=lines,
        mask=_rasterize(lines, part.shape),
        warnings=warnings,
        snap_records=snap_records,
        proximity_px=proximity,
    )


__all__ = [
    "ContourRouteResult",
    "build_hsv_correction_mask",
    "construct_zero_lines",
    "render_selection_preview",
    "select_outer_zero_points",
]
