"""Grow JD_67XX zero-line branches inside non-correction corridors.

The growth domain is bounded by the true outer contour, the main inner
contour, and the retained red correction regions from step 05.  Each raw
my_lab-style zero point is connected to the medial skeleton of its reachable
free-space component, then the reachable skeleton is rendered as branches.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import cv2
import numpy as np


BRANCH_COLOR = (255, 255, 0)
CONNECTOR_COLOR = (255, 120, 255)
EXPANDED_AREA_COLOR = (255, 80, 200)


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(path)


def load_zero_points(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points: list[dict] = []
    sequence = 1
    for contour in payload.get("contours", []):
        for item in contour.get("zero_points", []):
            x, y = item["point"]
            points.append(
                {
                    "label": f"Z{sequence}",
                    "point": (float(x), float(y)),
                    "loop": contour.get("name", "unknown"),
                    "type": item.get("type", "unknown"),
                }
            )
            sequence += 1
    if not points:
        raise ValueError(f"No zero points found: {path}")
    return points


def load_growth_domain(
    contour_json: Path,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    payload = json.loads(contour_json.read_text(encoding="utf-8"))
    outer_mask = np.zeros(image_shape, dtype=np.uint8)
    inner_masks: list[np.ndarray] = []
    boundary_records: list[dict] = []

    for contour in payload.get("contours", []):
        kind = contour.get("kind")
        if kind not in {"outer", "inner"}:
            continue
        points = np.rint(
            np.asarray(
                [sample["contour_point"] for sample in contour.get("samples", [])],
                dtype=np.float64,
            )
        ).astype(np.int32)
        if len(points) < 3:
            continue
        boundary_records.append(
            {
                "name": contour.get("name", kind),
                "kind": kind,
                "points": points,
            }
        )
        if kind == "outer":
            cv2.fillPoly(outer_mask, [points], 255, cv2.LINE_8)
        else:
            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255, cv2.LINE_8)
            inner_masks.append(mask)

    if cv2.countNonZero(outer_mask) == 0:
        raise ValueError(f"No outer contour found: {contour_json}")
    if not inner_masks:
        raise ValueError(f"No inner contour found for JD_67XX: {contour_json}")

    # JD_67XX has one meaningful sunroof opening. If more inner contours are
    # ever produced, treat all of them as growth boundaries.
    domain = outer_mask.copy()
    inner_union = np.zeros(image_shape, dtype=np.uint8)
    for mask in inner_masks:
        inner_union = cv2.bitwise_or(inner_union, mask)
    domain[inner_union > 0] = 0

    boundary_mask = np.zeros(image_shape, dtype=np.uint8)
    for record in boundary_records:
        cv2.polylines(
            boundary_mask,
            [record["points"].reshape((-1, 1, 2))],
            True,
            255,
            2,
            cv2.LINE_8,
        )
    return domain, boundary_mask, boundary_records


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    work = np.where(mask > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        edge = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, edge)
        work = eroded
    return skeleton


def rasterized_line_is_free(
    free_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    count = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    xs = np.rint(np.linspace(start[0], end[0], count)).astype(np.int32)
    ys = np.rint(np.linspace(start[1], end[1], count)).astype(np.int32)
    return bool(np.all(free_mask[ys, xs] > 0))


def astar_path(
    free_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    if start == end:
        return [start]
    if rasterized_line_is_free(free_mask, start, end):
        return [start, end]

    height, width = free_mask.shape
    cost = np.full((height, width), np.inf, dtype=np.float32)
    parent = np.full((height, width), -1, dtype=np.int32)
    closed = np.zeros((height, width), dtype=np.uint8)
    cost[start[1], start[0]] = 0.0
    queue: list[tuple[float, float, int, int]] = []
    heapq.heappush(queue, (0.0, 0.0, start[0], start[1]))
    neighbours = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.41421356),
        (1, -1, 1.41421356),
        (-1, 1, 1.41421356),
        (1, 1, 1.41421356),
    )

    while queue:
        _priority, current_cost, x, y = heapq.heappop(queue)
        if closed[y, x]:
            continue
        closed[y, x] = 1
        if (x, y) == end:
            break
        for dx, dy, step_cost in neighbours:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if not free_mask[ny, nx] or closed[ny, nx]:
                continue
            next_cost = current_cost + step_cost
            if next_cost >= float(cost[ny, nx]):
                continue
            cost[ny, nx] = next_cost
            parent[ny, nx] = y * width + x
            heuristic = max(abs(end[0] - nx), abs(end[1] - ny))
            heapq.heappush(queue, (next_cost + heuristic, next_cost, nx, ny))

    if parent[end[1], end[0]] < 0:
        raise ValueError(f"No free-space path from {start} to {end}")

    path = [end]
    x, y = end
    while (x, y) != start:
        flat = int(parent[y, x])
        y, x = divmod(flat, width)
        path.append((x, y))
    path.reverse()
    return path


def simplify_path(
    path: list[tuple[int, int]],
    free_mask: np.ndarray,
) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        visible = len(path) - 1
        while visible > anchor + 1:
            if rasterized_line_is_free(free_mask, path[anchor], path[visible]):
                break
            visible -= 1
        simplified.append(path[visible])
        anchor = visible
    return simplified


def nearest_coordinate(
    coordinates_yx: np.ndarray,
    point: tuple[float, float],
) -> tuple[int, int]:
    squared = (
        (coordinates_yx[:, 1].astype(np.float64) - point[0]) ** 2
        + (coordinates_yx[:, 0].astype(np.float64) - point[1]) ** 2
    )
    y, x = coordinates_yx[int(np.argmin(squared))]
    return int(x), int(y)


def grow_branches(
    domain_mask: np.ndarray,
    correction_mask: np.ndarray,
    zero_points: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    clearance = max(2, int(round(min(domain_mask.shape) * 0.002)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (clearance * 2 + 1, clearance * 2 + 1),
    )
    safe_domain = cv2.erode(domain_mask, kernel)
    blocked = cv2.dilate(
        np.where(correction_mask > 0, 255, 0).astype(np.uint8),
        kernel,
    )
    free_mask = safe_domain.copy()
    free_mask[blocked > 0] = 0
    free_mask = cv2.morphologyEx(
        free_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    if cv2.countNonZero(free_mask) == 0:
        raise ValueError("No branch-growth space remains after applying boundaries")

    skeleton = morphological_skeleton(free_mask)
    component_count, component_labels = cv2.connectedComponents(free_mask, 8)
    free_coordinates = np.column_stack(np.where(free_mask > 0))
    skeleton_coordinates = np.column_stack(np.where(skeleton > 0))

    skeleton_by_component: dict[int, np.ndarray] = {}
    for component in range(1, component_count):
        coords = skeleton_coordinates[
            component_labels[skeleton_coordinates[:, 0], skeleton_coordinates[:, 1]]
            == component
        ]
        if len(coords):
            skeleton_by_component[component] = coords

    connectors = np.zeros_like(free_mask)
    seed_records: list[dict] = []
    seeded_components: set[int] = set()
    for zero_point in zero_points:
        anchor = nearest_coordinate(free_coordinates, zero_point["point"])
        component = int(component_labels[anchor[1], anchor[0]])
        component_skeleton = skeleton_by_component.get(component)
        if component_skeleton is None:
            target = anchor
            path = [anchor]
        else:
            target = nearest_coordinate(component_skeleton, anchor)
            path = simplify_path(astar_path(free_mask, anchor, target), free_mask)
        if len(path) >= 2:
            cv2.polylines(
                connectors,
                [np.asarray(path, dtype=np.int32).reshape((-1, 1, 2))],
                False,
                255,
                1,
                cv2.LINE_8,
            )
        else:
            connectors[anchor[1], anchor[0]] = 255
        seeded_components.add(component)
        source = zero_point["point"]
        seed_records.append(
            {
                **zero_point,
                "free_space_anchor": list(anchor),
                "skeleton_anchor": list(target),
                "free_space_component": component,
                "distance_to_free_space_px": round(
                    float(np.hypot(source[0] - anchor[0], source[1] - anchor[1])),
                    4,
                ),
                "connector_points": [list(point) for point in path],
            }
        )

    reachable_skeleton = np.zeros_like(skeleton)
    expanded_area = np.zeros_like(free_mask)
    for component in seeded_components:
        reachable_skeleton[
            (skeleton > 0) & (component_labels == component)
        ] = 255
        expanded_area[component_labels == component] = 255
    branch_mask = cv2.bitwise_or(reachable_skeleton, connectors)
    details = {
        "clearance_px": clearance,
        "growth_domain_pixel_count": int(cv2.countNonZero(domain_mask)),
        "blocked_correction_pixel_count": int(cv2.countNonZero(blocked)),
        "free_space_pixel_count": int(cv2.countNonZero(free_mask)),
        "free_space_component_count": component_count - 1,
        "seeded_component_count": len(seeded_components),
        "skeleton_pixel_count": int(cv2.countNonZero(reachable_skeleton)),
        "branch_pixel_count": int(cv2.countNonZero(branch_mask)),
        "expanded_area_pixel_count": int(cv2.countNonZero(expanded_area)),
    }
    return branch_mask, free_mask, expanded_area, seed_records, details


def render_result(
    base: np.ndarray,
    branch_mask: np.ndarray,
    expanded_area: np.ndarray,
    boundary_records: list[dict],
    seeds: list[dict],
) -> np.ndarray:
    canvas = base.copy()

    area_overlay = canvas.copy()
    area_overlay[expanded_area > 0] = EXPANDED_AREA_COLOR
    blended_area = cv2.addWeighted(area_overlay, 0.46, canvas, 0.54, 0.0)
    canvas[expanded_area > 0] = blended_area[expanded_area > 0]

    for record in boundary_records:
        points = record["points"].reshape((-1, 1, 2))
        cv2.polylines(canvas, [points], True, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.polylines(canvas, [points], True, (25, 25, 25), 2, cv2.LINE_AA)

    contours, _ = cv2.findContours(
        branch_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.drawContours(canvas, contours, -1, BRANCH_COLOR, 2, cv2.LINE_AA)

    for seed in seeds:
        source = tuple(np.rint(seed["point"]).astype(np.int32).tolist())
        anchor = tuple(seed["free_space_anchor"])
        if source != anchor:
            cv2.line(canvas, source, anchor, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.line(canvas, source, anchor, CONNECTOR_COLOR, 2, cv2.LINE_AA)
        cv2.circle(canvas, source, 8, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(canvas, source, 5, (0, 230, 255), -1, cv2.LINE_AA)

    width = canvas.shape[1]
    x0 = max(820, width - 515)
    x1 = width - 18
    # Step 05 already has a legend in this corner. Use an opaque panel so the
    # JD_67XX branch-expansion legend replaces it cleanly.
    cv2.rectangle(canvas, (x0, 18), (x1, 112), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0, 18), (x1, 112), (40, 40, 40), 1)
    cv2.rectangle(canvas, (x0 + 16, 31), (x0 + 44, 47), EXPANDED_AREA_COLOR, -1)
    cv2.putText(canvas, "area reached by branch growth", (x0 + 57, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0 + 16, 67), (x0 + 44, 67), BRANCH_COLOR, 4, cv2.LINE_AA)
    cv2.putText(canvas, f"branch centerlines / seeds: {len(seeds)}", (x0 + 57, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0 + 16, 93), (x0 + 44, 93), (25, 25, 25), 3, cv2.LINE_AA)
    cv2.putText(canvas, "stops at outer / inner / red region", (x0 + 57, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def process(
    base_path: Path,
    correction_mask_path: Path,
    zero_json_path: Path,
    contour_json_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    base = read_image(base_path)
    correction_mask = read_image(correction_mask_path, cv2.IMREAD_GRAYSCALE)
    if base.shape[:2] != correction_mask.shape:
        raise ValueError("Base image and correction mask sizes differ")

    domain_mask, _boundary_mask, boundaries = load_growth_domain(
        contour_json_path,
        correction_mask.shape,
    )
    zero_points = load_zero_points(zero_json_path)
    branch_mask, free_mask, expanded_area, seeds, details = grow_branches(
        domain_mask,
        correction_mask,
        zero_points,
    )
    rendered = render_result(base, branch_mask, expanded_area, boundaries, seeds)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "07_zero_line_branch_expansion.png"
    branch_mask_path = output_dir / "branch_expansion_mask.png"
    expanded_area_path = output_dir / "branch_expanded_area_mask.png"
    free_mask_path = output_dir / "branch_growth_free_space.png"
    json_path = output_dir / "branch_expansion.json"
    write_png(image_path, rendered)
    write_png(branch_mask_path, branch_mask)
    write_png(expanded_area_path, expanded_area)
    write_png(free_mask_path, free_mask)
    payload = {
        "product": "JD_67XX6-DR000",
        "source_step05_image": str(base_path.resolve()),
        "source_correction_mask": str(correction_mask_path.resolve()),
        "source_zero_points": str(zero_json_path.resolve()),
        "source_true_contours": str(contour_json_path.resolve()),
        "mechanism": {
            "type": "seeded_medial_skeleton_branch_expansion",
            "seeds": "all my_lab-style raw zero points from step 03",
            "growth_boundaries": [
                "true product outer contour",
                "true product inner contour",
                "retained +/-1.0 mm correction regions from step 05",
            ],
            "small_hole_contours_are_boundaries": False,
            "expansion": "connect each seed to the medial skeleton, retain every reachable skeleton branch, and color the full reachable free-space components",
        },
        "details": details,
        "seed_count": len(seeds),
        "seeds": seeds,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, branch_mask_path, expanded_area_path, free_mask_path, json_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=output_root / "05_merged_correction_regions" / "05_merged_large_regions.png",
    )
    parser.add_argument(
        "--correction-mask",
        type=Path,
        default=output_root / "05_merged_correction_regions" / "merged_large_regions_mask.png",
    )
    parser.add_argument(
        "--zero-json",
        type=Path,
        default=output_root / "03_zero_point_selection" / "zero_points.json",
    )
    parser.add_argument(
        "--contour-json",
        type=Path,
        default=output_root / "02_contour_graph" / "contour_graph.json",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in process(
        args.base.resolve(),
        args.correction_mask.resolve(),
        args.zero_json.resolve(),
        args.contour_json.resolve(),
        args.output_dir.resolve(),
    ):
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
