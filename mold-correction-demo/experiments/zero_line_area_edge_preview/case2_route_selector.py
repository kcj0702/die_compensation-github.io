"""Select two zero points by following the product outer contour in both directions."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import cv2
import numpy as np


REGION_COLORS = [
    (255, 0, 255),
    (255, 145, 0),
    (0, 135, 255),
    (180, 0, 255),
    (0, 210, 255),
    (255, 80, 80),
]


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(path)


def load_zero_points(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    points: list[dict] = []
    sequence = 1
    for contour in data.get("contours", []):
        for item in contour.get("zero_points", []):
            x, y = item["point"]
            if "sample_index" in item:
                sample_index = int(item["sample_index"])
            else:
                before, after = item["between_sample_indices"]
                ratio = float(item.get("interpolation_ratio", 0.5))
                sample_index = int(before if ratio < 0.5 else after)
            points.append(
                {
                    "label": f"Z{sequence}",
                    "point": (float(x), float(y)),
                    "type": item.get("type", "unknown"),
                    "contour": contour.get("name", "unknown"),
                    "sample_index": sample_index,
                }
            )
            sequence += 1
    if len(points) < 2:
        raise ValueError(f"At least two zero points are required: {path}")
    return points


def load_contour_geometry(path: Path, image_shape: tuple[int, int]) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    contours = data.get("contours", [])
    product_mask = np.zeros(image_shape, dtype=np.uint8)
    outer_silhouette_mask = np.zeros(image_shape, dtype=np.uint8)
    for contour in contours:
        points = np.rint(
            np.asarray(
                [sample["contour_point"] for sample in contour.get("samples", [])],
                dtype=np.float64,
            )
        ).astype(np.int32)
        if len(points) < 3:
            continue
        if contour.get("kind") == "outer":
            cv2.fillPoly(product_mask, [points], 255, cv2.LINE_8)
            cv2.fillPoly(outer_silhouette_mask, [points], 255, cv2.LINE_8)
        else:
            cv2.fillPoly(product_mask, [points], 0, cv2.LINE_8)

    for contour in contours:
        if contour.get("kind") == "outer":
            points = np.asarray(
                [sample["contour_point"] for sample in contour.get("samples", [])],
                dtype=np.float64,
            )
            if len(points) < 3:
                raise ValueError(f"Outer contour has too few samples: {path}")
            return {
                "name": contour.get("name", "outer"),
                "points": points,
                "product_mask": product_mask,
                "outer_silhouette_mask": outer_silhouette_mask,
            }
    raise ValueError(f"No outer contour found: {path}")


def extract_regions(mask: np.ndarray) -> list[dict]:
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    regions: list[dict] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions.append(
            {
                "area_px": area,
                "centroid": (float(centroids[label, 0]), float(centroids[label, 1])),
                "bounding_box": (x, y, w, h),
                "contours": contours,
                "component_mask": component,
            }
        )
    regions.sort(key=lambda region: region["area_px"], reverse=True)
    for index, region in enumerate(regions, start=1):
        region["label"] = f"M{index}"
    return regions


def circular_path(start: int, end: int, count: int, direction: int) -> list[int]:
    indices = [start]
    cursor = start
    while cursor != end:
        cursor = (cursor + direction) % count
        indices.append(cursor)
        if len(indices) > count + 1:
            raise RuntimeError("Circular contour traversal did not terminate")
    return indices


def circular_true_runs(flags: np.ndarray) -> list[list[int]]:
    count = len(flags)
    if not np.any(flags):
        return []
    if np.all(flags):
        return [list(range(count))]

    false_anchor = int(np.flatnonzero(~flags)[0])
    runs: list[list[int]] = []
    current: list[int] = []
    for offset in range(1, count + 1):
        index = (false_anchor + offset) % count
        if flags[index]:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def contour_path_length(points: np.ndarray, indices: list[int]) -> float:
    if len(indices) < 2:
        return 0.0
    selected = points[np.asarray(indices, dtype=np.int32)]
    return float(np.linalg.norm(np.diff(selected, axis=0), axis=1).sum())


def rasterized_segment_is_clear(
    obstacle_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    delta_x = abs(end[0] - start[0])
    delta_y = abs(end[1] - start[1])
    sample_count = max(delta_x, delta_y) + 1
    xs = np.rint(np.linspace(start[0], end[0], sample_count)).astype(np.int32)
    ys = np.rint(np.linspace(start[1], end[1], sample_count)).astype(np.int32)
    return not np.any(obstacle_mask[ys, xs] > 0)


def octile_distance(x: int, y: int, end_x: int, end_y: int) -> float:
    delta_x = abs(end_x - x)
    delta_y = abs(end_y - y)
    diagonal = min(delta_x, delta_y)
    return float(max(delta_x, delta_y) + (np.sqrt(2.0) - 1.0) * diagonal)


def astar_shortest_path(
    obstacle_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    allow_diagonal_corner_touch: bool = False,
) -> list[tuple[int, int]]:
    height, width = obstacle_mask.shape
    start_x, start_y = start
    end_x, end_y = end
    if obstacle_mask[start_y, start_x] or obstacle_mask[end_y, end_x]:
        raise ValueError("A* endpoint lies inside an obstacle")

    cost = np.full((height, width), np.inf, dtype=np.float64)
    parent = np.full((height, width), -1, dtype=np.int32)
    closed = np.zeros((height, width), dtype=np.uint8)
    cost[start_y, start_x] = 0.0
    parent[start_y, start_x] = start_y * width + start_x
    queue = [(octile_distance(start_x, start_y, end_x, end_y), 0.0, start_x, start_y)]
    neighbours = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, float(np.sqrt(2.0))),
        (1, -1, float(np.sqrt(2.0))),
        (-1, 1, float(np.sqrt(2.0))),
        (1, 1, float(np.sqrt(2.0))),
    )
    found = False
    while queue:
        _estimate, current_cost, x, y = heapq.heappop(queue)
        if closed[y, x] or current_cost > float(cost[y, x]) + 1e-5:
            continue
        closed[y, x] = 1
        if (x, y) == (end_x, end_y):
            found = True
            break
        for delta_x, delta_y, step_cost in neighbours:
            next_x = x + delta_x
            next_y = y + delta_y
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            if obstacle_mask[next_y, next_x] or closed[next_y, next_x]:
                continue
            if delta_x and delta_y and not allow_diagonal_corner_touch:
                if obstacle_mask[y, next_x] or obstacle_mask[next_y, x]:
                    continue
            next_cost = current_cost + step_cost
            if next_cost + 1e-5 >= float(cost[next_y, next_x]):
                continue
            cost[next_y, next_x] = next_cost
            parent[next_y, next_x] = y * width + x
            heapq.heappush(
                queue,
                (next_cost + octile_distance(next_x, next_y, end_x, end_y), next_cost, next_x, next_y),
            )
    if not found:
        raise ValueError(f"No collision-free path from {start} to {end}")

    path: list[tuple[int, int]] = []
    x, y = end_x, end_y
    while True:
        path.append((x, y))
        if (x, y) == (start_x, start_y):
            break
        parent_index = int(parent[y, x])
        if parent_index < 0:
            raise RuntimeError("Broken A* parent chain")
        y, x = divmod(parent_index, width)
    path.reverse()
    return path


def simplify_collision_free_path(
    path: list[tuple[int, int]],
    obstacle_mask: np.ndarray,
) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        visible = len(path) - 1
        while visible > anchor + 1:
            if rasterized_segment_is_clear(obstacle_mask, path[anchor], path[visible]):
                break
            visible -= 1
        simplified.append(path[visible])
        anchor = visible
    return simplified


def path_length(path: list[tuple[int, int]]) -> float:
    return float(
        sum(
            np.hypot(end[0] - start[0], end[1] - start[1])
            for start, end in zip(path, path[1:])
        )
    )


def local_turn_complexity(
    path: list[tuple[int, int]],
    product_mask: np.ndarray,
) -> dict:
    """Measure concentrated bends without penalising a long route by itself."""
    _x, _y, product_width, product_height = cv2.boundingRect(product_mask)
    product_diagonal = float(np.hypot(product_width, product_height))
    window_length = max(80.0, product_diagonal * 0.10)
    minimum_segment_length = max(8.0, window_length * 0.075)
    minimum_turn_angle = 18.0
    maximum_allowed_accumulated_turn = 90.0

    points = np.asarray(path, dtype=np.float64)
    if len(points) < 3:
        return {
            "window_length_pixels": window_length,
            "minimum_segment_length_pixels": minimum_segment_length,
            "minimum_turn_angle_degrees": minimum_turn_angle,
            "maximum_allowed_accumulated_turn_degrees": maximum_allowed_accumulated_turn,
            "significant_turn_count": 0,
            "maximum_turns_in_window": 0,
            "maximum_accumulated_turn_degrees_in_window": 0.0,
            "locally_congested": False,
        }

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    turns: list[tuple[float, float]] = []
    for index in range(1, len(points) - 1):
        before_length = float(segment_lengths[index - 1])
        after_length = float(segment_lengths[index])
        if min(before_length, after_length) < minimum_segment_length:
            continue
        before = segment_vectors[index - 1] / before_length
        after = segment_vectors[index] / after_length
        cosine = float(np.clip(np.dot(before, after), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        if angle >= minimum_turn_angle:
            turns.append((float(cumulative[index]), angle))

    maximum_count = 0
    maximum_angle_sum = 0.0
    end = 0
    running_angle_sum = 0.0
    for start in range(len(turns)):
        if end < start:
            end = start
            running_angle_sum = 0.0
        while end < len(turns) and turns[end][0] - turns[start][0] <= window_length:
            running_angle_sum += turns[end][1]
            end += 1
        maximum_count = max(maximum_count, end - start)
        maximum_angle_sum = max(maximum_angle_sum, running_angle_sum)
        running_angle_sum -= turns[start][1]

    return {
        "window_length_pixels": window_length,
        "minimum_segment_length_pixels": minimum_segment_length,
        "minimum_turn_angle_degrees": minimum_turn_angle,
        "maximum_allowed_accumulated_turn_degrees": maximum_allowed_accumulated_turn,
        "significant_turn_count": len(turns),
        "maximum_turns_in_window": maximum_count,
        "maximum_accumulated_turn_degrees_in_window": maximum_angle_sum,
        "locally_congested": maximum_count >= 3 or maximum_angle_sum > maximum_allowed_accumulated_turn,
    }


def find_endpoint_anchor(
    endpoint: tuple[int, int],
    planning_obstacles: np.ndarray,
    strict_obstacles: np.ndarray,
    maximum_radius: int = 160,
) -> tuple[int, int]:
    x, y = endpoint
    if planning_obstacles[y, x] == 0:
        return endpoint
    height, width = planning_obstacles.shape
    portal_obstacles = strict_obstacles.copy()
    cv2.circle(portal_obstacles, endpoint, 2, 0, cv2.FILLED)
    for radius in range(1, maximum_radius + 1):
        x0, x1 = max(0, x - radius), min(width - 1, x + radius)
        y0, y1 = max(0, y - radius), min(height - 1, y + radius)
        free_y, free_x = np.where(planning_obstacles[y0 : y1 + 1, x0 : x1 + 1] == 0)
        candidates = sorted(
            ((int(local_x + x0), int(local_y + y0)) for local_x, local_y in zip(free_x, free_y)),
            key=lambda point: ((point[0] - x) ** 2 + (point[1] - y) ** 2, point[1], point[0]),
        )
        for candidate in candidates:
            if rasterized_segment_is_clear(portal_obstacles, endpoint, candidate):
                return candidate
    raise ValueError(f"No safe interior anchor found for {endpoint}")


def coarse_astar_path(
    obstacle_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    scale: int = 4,
) -> list[tuple[int, int]]:
    height, width = obstacle_mask.shape
    padded_height = int(np.ceil(height / scale) * scale)
    padded_width = int(np.ceil(width / scale) * scale)
    padded = np.full((padded_height, padded_width), 255, dtype=np.uint8)
    padded[:height, :width] = obstacle_mask
    coarse = padded.reshape(
        padded_height // scale,
        scale,
        padded_width // scale,
        scale,
    ).max(axis=(1, 3))

    def coarse_anchor(point: tuple[int, int]) -> tuple[int, int]:
        base_x = min(coarse.shape[1] - 1, point[0] // scale)
        base_y = min(coarse.shape[0] - 1, point[1] // scale)
        if coarse[base_y, base_x] == 0:
            return base_x, base_y
        for radius in range(1, 16):
            x0, x1 = max(0, base_x - radius), min(coarse.shape[1] - 1, base_x + radius)
            y0, y1 = max(0, base_y - radius), min(coarse.shape[0] - 1, base_y + radius)
            free_y, free_x = np.where(coarse[y0 : y1 + 1, x0 : x1 + 1] == 0)
            if free_x.size:
                candidates = sorted(
                    ((int(x + x0), int(y + y0)) for x, y in zip(free_x, free_y)),
                    key=lambda item: ((item[0] - base_x) ** 2 + (item[1] - base_y) ** 2, item[1], item[0]),
                )
                for candidate in candidates:
                    candidate_point = (
                        min(width - 1, candidate[0] * scale + scale // 2),
                        min(height - 1, candidate[1] * scale + scale // 2),
                    )
                    if rasterized_segment_is_clear(obstacle_mask, point, candidate_point):
                        return candidate
        raise ValueError(f"No coarse-grid anchor for {point}")

    coarse_start = coarse_anchor(start)
    coarse_end = coarse_anchor(end)
    free = np.where(coarse == 0, 255, 0).astype(np.uint8)
    _component_count, component_labels = cv2.connectedComponents(free, connectivity=8)
    if component_labels[coarse_start[1], coarse_start[0]] != component_labels[coarse_end[1], coarse_end[0]]:
        raise ValueError(f"No connected coarse-grid corridor from {start} to {end}")
    coarse_path = astar_shortest_path(coarse, coarse_start, coarse_end)
    full_path = [start]
    for x, y in coarse_path:
        point = (
            min(width - 1, x * scale + scale // 2),
            min(height - 1, y * scale + scale // 2),
        )
        if point != full_path[-1]:
            full_path.append(point)
    if full_path[-1] != end:
        full_path.append(end)
    return full_path


def route_pair(
    first_point: tuple[int, int],
    second_point: tuple[int, int],
    planning_obstacles: np.ndarray,
    strict_obstacles: np.ndarray,
) -> dict:
    def plan(obstacles: np.ndarray, mode: str):
        first_anchor = find_endpoint_anchor(first_point, obstacles, strict_obstacles)
        second_anchor = find_endpoint_anchor(second_point, obstacles, strict_obstacles)
        if rasterized_segment_is_clear(obstacles, first_anchor, second_anchor):
            return first_anchor, second_anchor, [first_anchor, second_anchor], f"direct_{mode}"
        try:
            path = coarse_astar_path(obstacles, first_anchor, second_anchor, scale=4)
            method = f"quarter_scale_astar_{mode}"
        except ValueError:
            try:
                path = coarse_astar_path(obstacles, first_anchor, second_anchor, scale=2)
                method = f"half_scale_astar_{mode}"
            except ValueError:
                free = np.where(obstacles == 0, 255, 0).astype(np.uint8)
                _count, labels = cv2.connectedComponents(free, connectivity=8)
                if labels[first_anchor[1], first_anchor[0]] != labels[second_anchor[1], second_anchor[0]]:
                    raise ValueError(
                        f"No full-resolution corridor from {first_anchor} to {second_anchor}"
                    )
                path = astar_shortest_path(
                    obstacles,
                    first_anchor,
                    second_anchor,
                    allow_diagonal_corner_touch=obstacles is strict_obstacles,
                )
                method = f"full_resolution_astar_{mode}"
        return first_anchor, second_anchor, path, method

    route_obstacles = planning_obstacles
    try:
        first_anchor, second_anchor, grid_path, method = plan(
            planning_obstacles,
            "clearance_detour_full_resolution_validation",
        )
        clearance_fallback_used = False
    except ValueError:
        route_obstacles = strict_obstacles
        first_anchor, second_anchor, grid_path, method = plan(
            strict_obstacles,
            "zero_clearance_detour_full_resolution_validation",
        )
        clearance_fallback_used = True

    simplified = simplify_collision_free_path(grid_path, route_obstacles)
    rendered = [first_point]
    rendered.extend(point for point in simplified if point != rendered[-1])
    if rendered[-1] != second_point:
        rendered.append(second_point)
    portal_obstacles = strict_obstacles.copy()
    cv2.circle(portal_obstacles, first_point, 2, 0, cv2.FILLED)
    cv2.circle(portal_obstacles, second_point, 2, 0, cv2.FILLED)
    if any(
        not rasterized_segment_is_clear(portal_obstacles, start, end)
        for start, end in zip(rendered, rendered[1:])
    ):
        raise ValueError("Full-resolution route validation failed")
    return {
        "routing_method": method,
        "zero_clearance_fallback_used": clearance_fallback_used,
        "product_interior_anchors": [list(first_anchor), list(second_anchor)],
        "path_points": [list(point) for point in rendered],
        "path_length_pixels": path_length(rendered),
        "path_bend_count": max(0, len(rendered) - 2),
    }


def build_route_obstacles(
    correction_mask: np.ndarray,
    product_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    height, width = correction_mask.shape
    clearance = max(2, int(round(min(height, width) * 0.003)))
    raw = np.where(correction_mask > 0, 255, 0).astype(np.uint8)
    strict = raw.copy()
    strict[product_mask == 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clearance * 2 + 1, clearance * 2 + 1))
    planning = cv2.dilate(raw, kernel)
    safe_product = cv2.erode(product_mask, kernel)
    planning[safe_product == 0] = 255
    return planning, strict, clearance


def contour_arc_from_zero_points(
    first: dict,
    second: dict,
    contour_points: np.ndarray,
    direction: int,
) -> list[tuple[int, int]]:
    count = len(contour_points)
    indices = circular_path(first["sample_index"] % count, second["sample_index"] % count, count, direction)
    arc = [tuple(np.rint(first["point"]).astype(int).tolist())]
    arc.extend(tuple(point) for point in np.rint(contour_points[indices]).astype(int).tolist())
    end = tuple(np.rint(second["point"]).astype(int).tolist())
    if arc[-1] != end:
        arc.append(end)
    return arc


def enclosure_mask_from_arc_and_route(
    image_shape: tuple[int, int],
    contour_arc: list[tuple[int, int]],
    route: list[tuple[int, int]],
    product_mask: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    polygon = contour_arc + list(reversed(route))[1:-1]
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255, cv2.LINE_8)
    mask[product_mask == 0] = 0
    return mask, polygon


def product_partition(
    product_mask: np.ndarray,
    route: list[tuple[int, int]],
    barrier_thickness: int,
) -> tuple[int, np.ndarray]:
    barrier = np.zeros_like(product_mask)
    route_array = np.asarray(route, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(barrier, [route_array], False, 255, barrier_thickness, cv2.LINE_8)
    endpoint_radius = barrier_thickness * 2
    cv2.circle(barrier, tuple(route_array[0, 0]), endpoint_radius, 255, cv2.FILLED)
    cv2.circle(barrier, tuple(route_array[-1, 0]), endpoint_radius, 255, cv2.FILLED)
    partition = product_mask.copy()
    partition[barrier > 0] = 0
    count, labels = cv2.connectedComponents(partition, connectivity=8)
    return count - 1, labels


def validate_closed_enclosure(
    first: dict,
    second: dict,
    target_mask: np.ndarray,
    contour_points: np.ndarray,
    product_mask: np.ndarray,
    planning_obstacles: np.ndarray,
    strict_obstacles: np.ndarray,
    route_cache: dict,
    baseline_component_count: int,
    barrier_thickness: int,
) -> dict:
    cache_key = (first["label"], second["label"])
    if cache_key not in route_cache:
        first_point = tuple(np.rint(first["point"]).astype(int).tolist())
        second_point = tuple(np.rint(second["point"]).astype(int).tolist())
        try:
            route_cache[cache_key] = route_pair(
                first_point,
                second_point,
                planning_obstacles,
                strict_obstacles,
            )
        except ValueError as error:
            route_cache[cache_key] = {"routing_error": str(error)}
    route_data = route_cache[cache_key]
    if "routing_error" in route_data:
        return {
            "valid": False,
            "reason": "no_collision_free_route",
            "routing_error": route_data["routing_error"],
        }

    route = [tuple(point) for point in route_data["path_points"]]
    component_count, _labels = product_partition(product_mask, route, barrier_thickness)
    target_area = max(1, int(np.count_nonzero(target_mask)))
    arc_options = []
    for name, direction in (("forward", 1), ("reverse", -1)):
        arc = contour_arc_from_zero_points(first, second, contour_points, direction)
        enclosure_mask, polygon = enclosure_mask_from_arc_and_route(
            target_mask.shape,
            arc,
            route,
            product_mask,
        )
        covered = int(np.count_nonzero((target_mask > 0) & (enclosure_mask > 0)))
        coverage = covered / float(target_area)
        area = int(np.count_nonzero(enclosure_mask))
        arc_options.append((coverage, -area, name, arc, polygon, enclosure_mask))
    coverage, negative_area, arc_name, arc, polygon, enclosure_mask = max(arc_options, key=lambda item: (item[0], item[1]))
    split_product = component_count > baseline_component_count
    valid = coverage >= 0.995 and split_product
    return {
        "valid": valid,
        "reason": "valid_closed_enclosure" if valid else "route_does_not_form_target_enclosing_partition",
        "target_coverage_ratio": coverage,
        "product_component_count_after_route": component_count,
        "baseline_product_component_count": baseline_component_count,
        "contour_arc_name": arc_name,
        "contour_arc_points": [list(point) for point in arc],
        "enclosure_polygon_points": [list(point) for point in polygon],
        "enclosed_area_pixels": -negative_area,
        "route": route_data,
    }


def region_from_mask(
    mask: np.ndarray,
    label: str,
    source_labels: list[str],
    attached_islands: list[dict],
) -> dict:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError(f"Logical region {label} is empty")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {
        "label": label,
        "area_px": int(xs.size),
        "centroid": (float(xs.mean()), float(ys.mean())),
        "bounding_box": (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ),
        "contours": contours,
        "component_mask": mask,
        "source_region_labels": source_labels,
        "attached_islands": attached_islands,
    }


def build_logical_regions(
    regions: list[dict],
    rounded_contour: np.ndarray,
    contact_kernel: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[list[dict], np.ndarray, int]:
    contact_infos = []
    for region in regions:
        dilated = cv2.dilate(region["component_mask"], contact_kernel)
        flags = dilated[rounded_contour[:, 1], rounded_contour[:, 0]] > 0
        indices = np.flatnonzero(flags).astype(int).tolist()
        runs = circular_true_runs(flags)
        contact_infos.append(
            {
                "region": region,
                "contact_indices": indices,
                "contact_runs": runs,
            }
        )

    boundary_infos = [info for info in contact_infos if info["contact_runs"]]
    island_infos = [info for info in contact_infos if not info["contact_runs"]]
    logical_connection_mask = np.zeros(image_shape, dtype=np.uint8)
    bridge_thickness = max(1, int(round(min(image_shape) * 0.003)))
    if not boundary_infos:
        return contact_infos, logical_connection_mask, bridge_thickness

    assignments: dict[str, list[dict]] = {
        info["region"]["label"]: [] for info in boundary_infos
    }
    for island_info in island_infos:
        island = island_info["region"]
        island_y, island_x = np.where(island["component_mask"] > 0)
        candidates = []
        for boundary_info in boundary_infos:
            boundary = boundary_info["region"]
            inverse = np.where(boundary["component_mask"] > 0, 0, 255).astype(np.uint8)
            distance_map = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
            island_distances = distance_map[island_y, island_x]
            nearest_index = int(np.argmin(island_distances))
            island_point = (int(island_x[nearest_index]), int(island_y[nearest_index]))
            boundary_y, boundary_x = np.where(boundary["component_mask"] > 0)
            squared = (boundary_x - island_point[0]) ** 2 + (boundary_y - island_point[1]) ** 2
            boundary_index = int(np.argmin(squared))
            boundary_point = (int(boundary_x[boundary_index]), int(boundary_y[boundary_index]))
            candidates.append(
                (
                    float(island_distances[nearest_index]),
                    -int(boundary["area_px"]),
                    boundary["label"],
                    boundary_info,
                    island_point,
                    boundary_point,
                )
            )
        distance, _negative_area, _label, boundary_info, island_point, boundary_point = min(
            candidates,
            key=lambda item: item[:3],
        )
        assignment = {
            "island_label": island["label"],
            "distance_to_boundary_region_px": distance,
            "island_connection_point": island_point,
            "boundary_region_connection_point": boundary_point,
            "island_region": island,
        }
        assignments[boundary_info["region"]["label"]].append(assignment)

    logical_infos = []
    for boundary_info in boundary_infos:
        boundary = boundary_info["region"]
        attached = assignments[boundary["label"]]
        combined_mask = boundary["component_mask"].copy()
        attachment_records = []
        source_labels = [boundary["label"]]
        for assignment in attached:
            island = assignment["island_region"]
            bridge = np.zeros(image_shape, dtype=np.uint8)
            cv2.line(
                bridge,
                assignment["boundary_region_connection_point"],
                assignment["island_connection_point"],
                255,
                bridge_thickness,
                cv2.LINE_8,
            )
            logical_connection_mask = cv2.bitwise_or(logical_connection_mask, bridge)
            combined_mask = cv2.bitwise_or(combined_mask, island["component_mask"])
            combined_mask = cv2.bitwise_or(combined_mask, bridge)
            source_labels.append(island["label"])
            attachment_records.append(
                {
                    "island_label": assignment["island_label"],
                    "distance_to_boundary_region_px": round(
                        assignment["distance_to_boundary_region_px"], 4
                    ),
                    "island_connection_point": list(assignment["island_connection_point"]),
                    "boundary_region_connection_point": list(
                        assignment["boundary_region_connection_point"]
                    ),
                }
            )
        logical_label = "+".join(source_labels)
        logical_region = region_from_mask(
            combined_mask,
            logical_label,
            source_labels,
            attachment_records,
        )
        logical_infos.append(
            {
                "region": logical_region,
                "contact_indices": boundary_info["contact_indices"],
                "contact_runs": boundary_info["contact_runs"],
            }
        )
    return logical_infos, logical_connection_mask, bridge_thickness


def select_along_outer_contour(
    regions: list[dict],
    zero_points: list[dict],
    contour_points: np.ndarray,
    image_shape: tuple[int, int],
    correction_mask: np.ndarray,
    product_mask: np.ndarray,
) -> tuple[list[dict], int, int]:
    height, width = image_shape
    count = len(contour_points)
    contact_radius = max(2, int(round(min(height, width) * 0.004)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (contact_radius * 2 + 1, contact_radius * 2 + 1),
    )
    rounded_contour = np.rint(contour_points).astype(np.int32)
    rounded_contour[:, 0] = np.clip(rounded_contour[:, 0], 0, width - 1)
    rounded_contour[:, 1] = np.clip(rounded_contour[:, 1], 0, height - 1)
    logical_infos, logical_connection_mask, bridge_thickness = build_logical_regions(
        regions,
        rounded_contour,
        kernel,
        image_shape,
    )
    correction_with_connections = cv2.bitwise_or(
        correction_mask,
        logical_connection_mask,
    )
    planning_obstacles, strict_obstacles, route_clearance = build_route_obstacles(
        correction_with_connections,
        product_mask,
    )
    baseline_component_count = cv2.connectedComponents(product_mask, connectivity=8)[0] - 1
    barrier_thickness = max(3, int(round(min(height, width) * 0.004)))
    route_cache: dict = {}
    directional_candidate_limit = min(4, len(zero_points))

    selections: list[dict] = []
    for logical_info in logical_infos:
        region = logical_info["region"]
        contact_indices = logical_info["contact_indices"]
        contact_runs = logical_info["contact_runs"]
        distance_to_contour = 0.0
        if contact_runs:
            main_run = max(contact_runs, key=len)
            arc_start, arc_end = main_run[0], main_run[-1]
            mode = "red_region_touches_outer_contour"
        else:
            inverse = np.where(region["component_mask"] > 0, 0, 255).astype(np.uint8)
            distance_map = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
            distances = distance_map[rounded_contour[:, 1], rounded_contour[:, 0]]
            anchor = int(np.argmin(distances))
            arc_start = anchor
            arc_end = anchor
            distance_to_contour = float(distances[anchor])
            mode = "nearest_outer_contour_anchor"

        reverse_candidates = sorted(
            zero_points,
            key=lambda point: (
                (arc_start - point["sample_index"]) % count,
                point["label"],
            ),
        )[:directional_candidate_limit]
        forward_candidates = sorted(
            zero_points,
            key=lambda point: (
                (point["sample_index"] - arc_end) % count,
                point["label"],
            ),
        )[:directional_candidate_limit]

        ranked_pairs = []
        seen_pairs = set()
        for reverse_rank, reverse_point in enumerate(reverse_candidates):
            for forward_rank, forward_point in enumerate(forward_candidates):
                if reverse_point["label"] == forward_point["label"]:
                    continue
                pair_key = (reverse_point["label"], forward_point["label"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                ranked_pairs.append(
                    (
                        reverse_rank + forward_rank,
                        max(reverse_rank, forward_rank),
                        reverse_rank,
                        forward_rank,
                        reverse_point,
                        forward_point,
                    )
                )
        # Search by a widening local band. Once a valid pair appears, inspect one
        # additional candidate tier, but do not let a very remote straight line
        # replace locally meaningful contour endpoints.
        ranked_pairs.sort(key=lambda item: (item[1], item[0], item[2], item[3]))

        valid_pairs = []
        pair_attempts = []
        first_valid_max_rank = None
        for _rank_sum, _max_rank, reverse_rank, forward_rank, reverse_point, forward_point in ranked_pairs:
            if first_valid_max_rank is not None and _max_rank > first_valid_max_rank + 1:
                continue
            print(
                f"Checking {region['label']}: {reverse_point['label']} + {forward_point['label']} "
                f"(direction ranks {reverse_rank + 1}, {forward_rank + 1})",
                flush=True,
            )
            validation = validate_closed_enclosure(
                reverse_point,
                forward_point,
                cv2.bitwise_and(region["component_mask"], product_mask),
                contour_points,
                product_mask,
                planning_obstacles,
                strict_obstacles,
                route_cache,
                baseline_component_count,
                barrier_thickness,
            )
            print(
                f"  result={validation['reason']}, "
                f"coverage={validation.get('target_coverage_ratio', 0.0):.4f}, "
                f"components={validation.get('product_component_count_after_route', 0)}",
                flush=True,
            )
            route = validation.get("route", {})
            bend_count = int(route.get("path_bend_count", 10**6))
            route_length = float(route.get("path_length_pixels", float("inf")))
            clearance_fallback = bool(route.get("zero_clearance_fallback_used", True))
            turn_complexity = local_turn_complexity(
                [tuple(point) for point in route.get("path_points", [])],
                product_mask,
            ) if route else {
                "significant_turn_count": 10**6,
                "maximum_turns_in_window": 10**6,
                "maximum_accumulated_turn_degrees_in_window": float("inf"),
                "locally_congested": True,
            }
            local_turn_count = int(turn_complexity["maximum_turns_in_window"])
            local_turn_excess = max(0, local_turn_count - 2)
            maximum_local_turn_angle = float(
                turn_complexity["maximum_accumulated_turn_degrees_in_window"]
            )
            local_angle_excess = max(0.0, maximum_local_turn_angle - 90.0)
            locally_congested = bool(turn_complexity["locally_congested"])
            significant_turn_count = int(turn_complexity["significant_turn_count"])
            rank_sum = reverse_rank + forward_rank
            rank_penalty = 8000.0 * (reverse_rank**2 + forward_rank**2)
            both_sides_advanced_penalty = 5000.0 if reverse_rank > 0 and forward_rank > 0 else 0.0
            # Concentrated bends are expensive, while total bends are only a weak
            # tie-breaker. This keeps long clean routes from being penalised merely
            # because their overall bend count is naturally larger.
            weighted_cost = (
                (40000.0 if locally_congested else 0.0)
                + local_turn_excess * 40000.0
                + local_angle_excess * 50.0
                + local_turn_count * 1000.0
                + significant_turn_count * 300.0
                + bend_count * 100.0
                + (2500.0 if clearance_fallback else 0.0)
                + rank_penalty
                + both_sides_advanced_penalty
                + route_length
            )
            attempt = {
                "zero_point_labels": [reverse_point["label"], forward_point["label"]],
                "directional_candidate_ranks": [reverse_rank + 1, forward_rank + 1],
                "valid": validation["valid"],
                "reason": validation["reason"],
                "routing_error": validation.get("routing_error"),
                "target_coverage_ratio": float(validation.get("target_coverage_ratio", 0.0)),
                "product_component_count_after_route": int(validation.get("product_component_count_after_route", 0)),
                "path_bend_count": None if bend_count >= 10**6 else bend_count,
                "path_length_pixels": None if not np.isfinite(route_length) else route_length,
                "zero_clearance_fallback_used": clearance_fallback if route else None,
                "local_turn_complexity": turn_complexity if route else None,
                "local_congestion_penalty": None if not route else (
                    (40000.0 if locally_congested else 0.0)
                    + local_turn_excess * 40000.0
                    + local_angle_excess * 50.0
                ),
                "directional_rank_penalty": rank_penalty,
                "both_sides_advanced_penalty": both_sides_advanced_penalty,
                "weighted_route_cost": None if not validation["valid"] else weighted_cost,
                "selected": False,
            }
            pair_attempts.append(attempt)
            if validation["valid"]:
                if first_valid_max_rank is None:
                    first_valid_max_rank = _max_rank
                valid_pairs.append(
                    (
                        weighted_cost,
                        bend_count,
                        clearance_fallback,
                        route_length,
                        rank_sum,
                        reverse_rank,
                        forward_rank,
                        reverse_point,
                        forward_point,
                        validation,
                        attempt,
                    )
                )
        if not valid_pairs:
            raise ValueError(f"{region['label']}: no zero-point pair forms a valid closed enclosure")

        selected_pair = min(valid_pairs, key=lambda item: item[:7])
        (
            selected_cost,
            _selected_bends,
            _selected_fallback,
            _selected_length,
            _selected_rank_sum,
            reverse_rank,
            forward_rank,
            reverse_point,
            forward_point,
            closure_validation,
            selected_attempt,
        ) = selected_pair
        selected_attempt["selected"] = True

        reverse_index = reverse_point["sample_index"] % count
        forward_index = forward_point["sample_index"] % count
        reverse_path = circular_path(arc_start, reverse_index, count, -1)
        forward_path = circular_path(arc_end, forward_index, count, 1)
        selections.append(
            {
                "region": region,
                "mode": mode,
                "contact_sample_count": len(contact_indices),
                "contact_run_count": len(contact_runs),
                "used_contact_sample_count": len(main_run) if contact_runs else 0,
                "logical_bridge_thickness_px": bridge_thickness,
                "distance_to_contour_px": distance_to_contour,
                "arc_start": arc_start,
                "arc_end": arc_end,
                "affected_arc": circular_path(arc_start, arc_end, count, 1),
                "selection_status": "best_valid_route_selected",
                "directional_candidate_ranks": [reverse_rank + 1, forward_rank + 1],
                "directional_candidate_limit": directional_candidate_limit,
                "selected_weighted_route_cost": selected_cost,
                "pair_attempt_count": len(pair_attempts),
                "pair_attempts": pair_attempts,
                "closure_validation": closure_validation,
                "selected": [
                    {
                        **reverse_point,
                        "direction": "decreasing_sample_index",
                        "contour_steps": len(reverse_path) - 1,
                        "contour_distance_px": contour_path_length(contour_points, reverse_path),
                        "path_indices": reverse_path,
                    },
                    {
                        **forward_point,
                        "direction": "increasing_sample_index",
                        "contour_steps": len(forward_path) - 1,
                        "contour_distance_px": contour_path_length(contour_points, forward_path),
                        "path_indices": forward_path,
                    },
                ],
            }
        )
    return selections, contact_radius, route_clearance


def outlined_text(
    image: np.ndarray,
    text: str,
    point: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_contour_path(
    image: np.ndarray,
    contour_points: np.ndarray,
    indices: list[int],
    color: tuple[int, int, int],
) -> None:
    if len(indices) < 2:
        return
    points = np.rint(contour_points[np.asarray(indices, dtype=np.int32)]).astype(np.int32)
    cv2.polylines(image, [points], False, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.polylines(image, [points], False, color, 4, cv2.LINE_AA)


def draw_result(
    image: np.ndarray,
    selections: list[dict],
    contour_points: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]

    for index, selection in enumerate(selections):
        region = selection["region"]
        color = REGION_COLORS[index % len(REGION_COLORS)]
        centroid = tuple(np.rint(region["centroid"]).astype(int).tolist())

        for contour in region["contours"]:
            cv2.drawContours(canvas, [contour], -1, color, 2, cv2.LINE_AA)

        draw_contour_path(canvas, contour_points, selection["affected_arc"], color)
        for selected in selection["selected"]:
            draw_contour_path(canvas, contour_points, selected["path_indices"], color)
            point = tuple(np.rint(selected["point"]).astype(int).tolist())
            cv2.circle(canvas, point, 13, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.circle(canvas, point, 13, color, 3, cv2.LINE_AA)

        route = np.asarray(
            selection["closure_validation"]["route"]["path_points"],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(canvas, [route], False, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.polylines(canvas, [route], False, (255, 255, 0), 4, cv2.LINE_AA)

        diamond = np.asarray(
            [
                (centroid[0], centroid[1] - 10),
                (centroid[0] + 10, centroid[1]),
                (centroid[0], centroid[1] + 10),
                (centroid[0] - 10, centroid[1]),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, diamond, (255, 255, 255), cv2.LINE_AA)
        cv2.polylines(canvas, [diamond], True, color, 3, cv2.LINE_AA)
        pair = ",".join(item["label"] for item in selection["selected"])
        label_x = min(max(centroid[0] + 13, 4), width - 150)
        label_y = min(max(centroid[1] - 12, 18), height - 5)
        outlined_text(
            canvas,
            f"{region['label']} -> {pair}",
            (label_x, label_y),
            0.46,
            color,
            2,
        )

    legend_left = max(788, width - 520)
    legend_right = width - 18
    legend_bottom = 139
    cv2.rectangle(canvas, (legend_left, 18), (legend_right, legend_bottom), (255, 255, 255), -1)
    cv2.rectangle(canvas, (legend_left, 18), (legend_right, legend_bottom), (40, 40, 40), 1)
    cv2.rectangle(canvas, (legend_left + 15, 33), (legend_left + 43, 50), (0, 0, 255), -1)
    cv2.putText(canvas, "retained correction regions (+/-0.7 mm)", (legend_left + 55, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (legend_left + 29, 70), (255, 0, 255), cv2.MARKER_DIAMOND, 17, 2, cv2.LINE_AA)
    cv2.putText(canvas, "M#: region centroid", (legend_left + 55, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(canvas, (legend_left + 16, 96), (legend_left + 43, 96), (255, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(canvas, "best simple route among directional candidates", (legend_left + 55, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(canvas, (legend_left + 16, 122), (legend_left + 43, 122), (255, 255, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "validated interior route closing the region", (legend_left + 55, 127), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def process(
    mask_path: Path,
    base_path: Path,
    zero_json_path: Path,
    contour_json_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
    base = read_image(base_path, cv2.IMREAD_COLOR)
    if mask.shape != base.shape[:2]:
        raise ValueError(f"Mask and base image sizes differ: {mask.shape} vs {base.shape[:2]}")

    zero_points = load_zero_points(zero_json_path)
    outer_contour = load_contour_geometry(contour_json_path, mask.shape)
    regions = extract_regions(mask)
    if not regions:
        raise ValueError(f"No correction regions found: {mask_path}")
    selections, contact_radius, route_clearance = select_along_outer_contour(
        regions,
        zero_points,
        outer_contour["points"],
        mask.shape,
        mask,
        outer_contour["outer_silhouette_mask"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "06_nearest_zero_points.png"
    json_path = output_dir / "nearest_zero_points.json"
    write_png(image_path, draw_result(base, selections, outer_contour["points"]))

    result = {
        "source_merged_mask": str(mask_path.resolve()),
        "source_step05_image": str(base_path.resolve()),
        "source_zero_points": str(zero_json_path.resolve()),
        "source_outer_contour": str(contour_json_path.resolve()),
        "selection_rule": {
            "method": "Follow the product outer contour in both directions, route every pair among the first four candidates per direction, and select the best valid closed route",
            "selected_per_region": 2,
            "candidate_zero_points": "all outer-contour zero points from step 03",
            "directional_candidate_limit": 4,
            "contact_radius_px": contact_radius,
            "multiple_contact_runs": "use the longest continuous outer-contour contact run",
            "non_contacting_region_fallback": "use the outer-contour sample geometrically closest to the correction region as a zero-length contact arc",
            "closed_shape_validation": "the candidate route must stay inside the product, avoid all retained correction regions, split the product, and form a contour-arc polygon covering at least 99.5% of the target region",
            "pair_optimization": "search through one tier beyond the nearest valid directional pair; reject hard-risk and non-closing routes, then strongly penalize three or more significant bends inside a sliding window equal to 10% of the product diagonal. Total bend count is only a weak tie-breaker",
            "local_turn_rule": "ignore route segments shorter than 7.5% of the window and direction changes below 18 degrees; mark a window as congested when it contains at least three significant bends or more than 90 degrees of accumulated turning",
            "route_clearance_px": route_clearance,
            "routing_domain": "inside the product outer silhouette; inner openings and holes are traversable",
            "isolated_island_rule": "attach every retained region that does not touch the outer contour to the nearest retained region that does touch it; include the minimum-distance bridge in routing and enclosure validation",
        },
        "source_component_count": len(regions),
        "region_count": len(selections),
        "zero_point_candidate_count": len(zero_points),
        "regions": [],
    }
    for selection in selections:
        region = selection["region"]
        result["regions"].append(
            {
                "region_label": region["label"],
                "source_region_labels": region.get("source_region_labels", [region["label"]]),
                "attached_islands": region.get("attached_islands", []),
                "area_px": region["area_px"],
                "centroid": [round(value, 4) for value in region["centroid"]],
                "bounding_box": list(region["bounding_box"]),
                "contour_contact_mode": selection["mode"],
                "contact_sample_count": selection["contact_sample_count"],
                "contact_run_count": selection["contact_run_count"],
                "used_contact_sample_count": selection["used_contact_sample_count"],
                "distance_to_outer_contour_px": round(selection["distance_to_contour_px"], 4),
                "contact_arc": {
                    "start_sample_index": selection["arc_start"],
                    "end_sample_index": selection["arc_end"],
                },
                "selection_status": selection["selection_status"],
                "directional_candidate_ranks": selection["directional_candidate_ranks"],
                "directional_candidate_limit": selection["directional_candidate_limit"],
                "selected_weighted_route_cost": round(selection["selected_weighted_route_cost"], 4),
                "pair_attempt_count": selection["pair_attempt_count"],
                "pair_attempts": selection["pair_attempts"],
                "closure_validation": selection["closure_validation"],
                "selected_zero_points": [
                    {
                        "label": selected["label"],
                        "point": [round(value, 4) for value in selected["point"]],
                        "direction": selected["direction"],
                        "sample_index": selected["sample_index"],
                        "contour_steps": selected["contour_steps"],
                        "contour_distance_px": round(selected["contour_distance_px"], 4),
                        "type": selected["type"],
                        "contour": selected["contour"],
                    }
                    for selected in selection["selected"]
                ],
            }
        )
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return image_path, json_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=output_root / "05_merged_correction_regions" / "merged_large_regions_mask.png")
    parser.add_argument("--base", type=Path, default=output_root / "05_merged_correction_regions" / "05_merged_large_regions.png")
    parser.add_argument("--zero-json", type=Path, default=output_root / "03_zero_point_selection" / "zero_points.json")
    parser.add_argument("--contour-json", type=Path, default=output_root / "02_contour_graph" / "contour_graph.json")
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    args = parser.parse_args()

    image_path, json_path = process(args.mask, args.base, args.zero_json, args.contour_json, args.output_dir)
    print(f"Created: {image_path}")
    print(f"Created: {json_path}")


if __name__ == "__main__":
    main()
