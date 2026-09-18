"""Snap boundary-connected zero lines to registered CAD feature curves.

Only a nearby, substantially parallel CAD curve may attract a zero line.  The
direction gate prevents an unrelated hole/fillet/outline that merely crosses
the route at right angles from changing it.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.spatial import cKDTree

from zero_line_detection import case2_route_selector as router


FEATURE_KIND_LABELS = {
    "fillet_boundary_1": "fillet edge 1",
    "fillet_center": "fillet centre",
    "fillet_boundary_2": "fillet edge 2",
    "hole": "hole",
    "outer": "outer contour",
}


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (matrix @ np.column_stack((points, np.ones(len(points)))).T).T


def _unpack(data, name: str) -> list[np.ndarray]:
    points = np.asarray(data[name], dtype=np.float64)
    offsets = np.asarray(data[f"{name}_offsets"], dtype=np.int64)
    return [
        points[start:stop]
        for start, stop in zip(offsets[:-1], offsets[1:])
        if stop - start >= 2
    ]


@lru_cache(maxsize=4)
def load_projected_cad_features(
    stage4_dir_text: str,
    stage5_meta_text: str,
    fillet_npz_text: str,
) -> tuple[dict, ...]:
    """Load exact STP curves and project them into registered scan pixels."""
    stage4_dir = Path(stage4_dir_text)
    stage5_meta = json.loads(Path(stage5_meta_text).read_text(encoding="utf-8"))
    matrix = np.asarray(
        stage5_meta["registration"]["cad_xy_to_image_uv_matrix"], dtype=np.float64
    )
    features: list[dict] = []

    with np.load(fillet_npz_text) as fillet_data:
        for key, kind in (
            ("boundary_1", "fillet_boundary_1"),
            ("center", "fillet_center"),
            ("boundary_2", "fillet_boundary_2"),
        ):
            for index, line in enumerate(_unpack(fillet_data, key), start=1):
                features.append({
                    "id": f"{kind}:{index}",
                    "kind": kind,
                    "points": _project(line, matrix),
                })

    feature_data = json.loads((stage4_dir / "features.json").read_text(encoding="utf-8"))
    metadata = json.loads((stage4_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    edge_by_id = {int(edge["edge_id"]): edge for edge in feature_data["edges"]}
    boundary_ids = {
        int(value) for value in metadata["feature_sets"]["boundary"]["ids"]
    }
    hole_candidate_ids = {
        int(value) for value in metadata["feature_sets"]["hole"]["ids"]
    }
    hole_edge_ids: set[int] = set()
    for candidate in feature_data.get("hole_candidates", []):
        if int(candidate["candidate_id"]) in hole_candidate_ids:
            hole_edge_ids.update(int(value) for value in candidate.get("edge_ids", []))
    for candidate in metadata.get("wall_ring_holes", []):
        if int(candidate["candidate_id"]) in hole_candidate_ids:
            hole_edge_ids.update(int(value) for value in candidate.get("edge_ids", []))

    plane = [
        ("x", "y", "z").index(str(column).lower())
        for column in stage5_meta["registration"]["plane"]
    ]
    for kind, edge_ids in (("outer", boundary_ids), ("hole", hole_edge_ids)):
        for edge_id in sorted(edge_ids):
            edge = edge_by_id.get(edge_id)
            xyz = (
                np.asarray(edge.get("polyline_xyz", []), dtype=np.float64)
                if edge else np.empty((0, 3), dtype=np.float64)
            )
            if xyz.ndim == 2 and len(xyz) >= 2:
                features.append({
                    "id": f"{kind}:{edge_id}",
                    "kind": kind,
                    "points": _project(xyz[:, plane], matrix),
                })
    return tuple(features)


def discover_local_ai_features(filename: str, workspace_root: Path) -> list[dict]:
    """Find an exact-part ai_zeroline artifact set; never substitute XX1/XX2."""
    match = re.search(r"(?P<part>\d{2}XX\d)-DR\d+", Path(filename).stem, re.IGNORECASE)
    if not match:
        return []
    part = match.group("part").upper()
    case_id = match.group(0).upper()
    ai_root = workspace_root / "ai_zeroline" / "outputs"
    fillet_path = ai_root / "stage13" / case_id / "fillet_three_lines.npz"
    if not fillet_path.is_file():
        return []

    candidates = sorted((ai_root / "stage5").glob(f"{part}-*"))
    for stage5_dir in candidates:
        meta_path = stage5_dir / "deviation_meta.json"
        stage4_dir = ai_root / "stage4" / stage5_dir.name
        if not meta_path.is_file() or not (stage4_dir / "features.json").is_file():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            cad_name = Path(metadata["cad_source"]["path"]).stem.upper()
        except (KeyError, TypeError, ValueError, OSError):
            continue
        if not cad_name.startswith(part):
            continue
        return [dict(item) for item in load_projected_cad_features(
            str(stage4_dir), str(meta_path), str(fillet_path)
        )]
    return []


def _resample(points: np.ndarray, spacing: float = 1.5) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.copy()
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 1e-6:
        return points[:1].copy()
    targets = np.arange(0.0, total, max(0.5, spacing))
    if len(targets) == 0 or targets[-1] < total:
        targets = np.append(targets, total)
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(segments) - 1)
    ratios = (targets - cumulative[indices]) / np.maximum(lengths[indices], 1e-9)
    return points[indices] + segments[indices] * ratios[:, None]


def _points_and_tangents(lines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    point_groups: list[np.ndarray] = []
    tangent_groups: list[np.ndarray] = []
    for line in lines:
        dense = _resample(line)
        if len(dense) < 2:
            continue
        tangent = np.gradient(dense, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        valid = norm > 1e-8
        if not valid.any():
            continue
        tangent[valid] /= norm[valid, None]
        point_groups.append(dense[valid])
        tangent_groups.append(tangent[valid])
    if not point_groups:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.vstack(point_groups), np.vstack(tangent_groups)


def _feature_indices(features: list[dict], shape: tuple[int, int]) -> dict[str, dict]:
    height, width = shape
    margin = max(20, int(round(min(shape) * 0.03)))
    grouped: dict[str, list[np.ndarray]] = {}
    for feature in features:
        points = np.asarray(feature.get("points", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            continue
        visible = (
            (points[:, 0] >= -margin) & (points[:, 0] < width + margin)
            & (points[:, 1] >= -margin) & (points[:, 1] < height + margin)
        )
        if visible.any():
            grouped.setdefault(str(feature["kind"]), []).append(points)
    result = {}
    for kind, lines in grouped.items():
        points, tangents = _points_and_tangents(lines)
        if len(points):
            result[kind] = {
                "points": points,
                "tangents": tangents,
                "tree": cKDTree(points),
            }
    return result


def _route_tangents(points: np.ndarray) -> np.ndarray:
    tangent = np.gradient(points, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    valid = norm > 1e-8
    tangent[valid] /= norm[valid, None]
    tangent[~valid] = 0.0
    return tangent


def _matching_targets(
    route: np.ndarray,
    indices: dict[str, dict],
    max_distance: float,
    maximum_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tangents = _route_tangents(route)
    count = len(route)
    labels = np.full(count, "", dtype=object)
    targets = route.copy()
    distances = np.full(count, np.inf, dtype=np.float64)
    angles = np.full(count, 180.0, dtype=np.float64)
    minimum_dot = float(np.cos(np.deg2rad(maximum_angle_deg)))

    for kind, index in indices.items():
        k = min(4, len(index["points"]))
        distance, nearest = index["tree"].query(
            route, k=k, distance_upper_bound=max_distance
        )
        if k == 1:
            distance = distance[:, None]
            nearest = nearest[:, None]
        for row in range(count):
            for candidate_distance, candidate_index in zip(distance[row], nearest[row]):
                if not np.isfinite(candidate_distance) or candidate_index >= len(index["points"]):
                    continue
                dot = abs(float(np.dot(tangents[row], index["tangents"][candidate_index])))
                if dot < minimum_dot:
                    continue
                angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
                score = float(candidate_distance) + angle * 0.08
                old_score = distances[row] + angles[row] * 0.08
                if score < old_score:
                    labels[row] = kind
                    targets[row] = index["points"][candidate_index]
                    distances[row] = float(candidate_distance)
                    angles[row] = angle
    return labels, targets, distances, angles


def _runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    start = 0
    while start < len(labels):
        label = str(labels[start])
        end = start + 1
        while end < len(labels) and str(labels[end]) == label:
            end += 1
        if label:
            runs.append((start, end, label))
        start = end
    return runs


def _collision_free(points: np.ndarray, obstacles: np.ndarray | None) -> bool:
    if obstacles is None:
        return True
    portal = obstacles.copy()
    first = tuple(np.rint(points[0]).astype(int).tolist())
    last = tuple(np.rint(points[-1]).astype(int).tolist())
    cv2.circle(portal, first, 2, 0, cv2.FILLED)
    cv2.circle(portal, last, 2, 0, cv2.FILLED)
    rounded = np.rint(points).astype(int)
    return all(
        router.rasterized_segment_is_clear(portal, tuple(a), tuple(b))
        for a, b in zip(rounded, rounded[1:])
    )


def snap_boundary_lines_to_cad(
    lines: list[dict],
    features: list[dict],
    image_shape: tuple[int, int],
    *,
    obstacle_mask: np.ndarray | None = None,
    max_distance_px: float | None = None,
    maximum_angle_deg: float = 28.0,
    minimum_run_px: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Snap parallel, nearby portions of boundary routes to exact CAD curves."""
    if not features:
        return [dict(line) for line in lines], []
    max_distance = float(
        max_distance_px if max_distance_px is not None
        else max(6.0, min(image_shape) * 0.009)
    )
    minimum_run = float(
        minimum_run_px if minimum_run_px is not None
        else max(12.0, min(image_shape) * 0.018)
    )
    indices = _feature_indices(features, image_shape)
    if not indices:
        return [dict(line) for line in lines], []

    output: list[dict] = []
    records: list[dict] = []
    for line in lines:
        copied = dict(line)
        if line.get("kind") != "boundary_connection":
            output.append(copied)
            continue
        original = np.asarray(line.get("points", []), dtype=np.float64)
        route = _resample(original, spacing=1.25)
        if len(route) < 3:
            output.append(copied)
            continue
        labels, targets, distances, angles = _matching_targets(
            route, indices, max_distance, maximum_angle_deg
        )
        candidate = route.copy()
        accepted: list[dict] = []
        for start, end, kind in _runs(labels):
            length = float(np.linalg.norm(np.diff(route[start:end], axis=0), axis=1).sum())
            if end - start < 5 or length < minimum_run:
                continue
            attempt = candidate.copy()
            weights = np.ones(end - start, dtype=np.float64)
            taper = min(5, max(1, (end - start) // 3))
            weights[:taper] = np.linspace(0.25, 1.0, taper)
            weights[-taper:] = np.minimum(weights[-taper:], np.linspace(1.0, 0.25, taper))
            if start == 0:
                weights[0] = 0.0
            if end == len(route):
                weights[-1] = 0.0
            attempt[start:end] = (
                route[start:end] * (1.0 - weights[:, None])
                + targets[start:end] * weights[:, None]
            )
            if not _collision_free(attempt, obstacle_mask):
                continue
            candidate = attempt
            record = {
                "lineId": line.get("id"),
                "region": line.get("region"),
                "featureKind": kind,
                "featureLabel": FEATURE_KIND_LABELS.get(kind, kind),
                "startIndex": int(start),
                "endIndex": int(end - 1),
                "lengthPx": round(length, 2),
                "meanDistancePx": round(float(np.mean(distances[start:end])), 2),
                "meanAngleDeg": round(float(np.mean(angles[start:end])), 2),
                "featurePoints": np.rint(targets[start:end]).astype(int).tolist(),
            }
            accepted.append(record)
            records.append(record)
        if accepted:
            rounded = np.rint(candidate).astype(int)
            keep = np.ones(len(rounded), dtype=bool)
            keep[1:] = np.any(rounded[1:] != rounded[:-1], axis=1)
            copied["preSnapPoints"] = np.rint(original).astype(int).tolist()
            copied["points"] = rounded[keep].tolist()
            copied["cadSnapped"] = True
            copied["snapSegments"] = accepted
        output.append(copied)
    return output, records


def _draw_dashed(
    image: np.ndarray,
    points: np.ndarray,
    colour: tuple[int, int, int],
    thickness: int = 1,
    dash_px: float = 9.0,
) -> None:
    dense = _resample(points, spacing=2.0)
    if len(dense) < 2:
        return
    for index, (start, end) in enumerate(zip(dense, dense[1:])):
        if int(index * 2.0 // dash_px) % 2 == 0:
            cv2.line(
                image,
                tuple(np.rint(start).astype(int)),
                tuple(np.rint(end).astype(int)),
                colour,
                thickness,
                cv2.LINE_AA,
            )


def _outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    colour: tuple[int, int, int],
    scale: float = 0.40,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        colour, 1, cv2.LINE_AA,
    )


def render_cad_snap_preview(
    image_rgb: np.ndarray,
    part_mask: np.ndarray,
    result,
    *,
    max_distance_px: float | None = None,
    maximum_angle_deg: float = 28.0,
    minimum_run_px: float | None = None,
) -> np.ndarray:
    """Show the original route, matched CAD curve and snapped result together."""
    short = min(part_mask.shape)
    max_distance = float(max_distance_px or max(6.0, short * 0.009))
    minimum_run = float(minimum_run_px or max(12.0, short * 0.018))
    preview = image_rgb.copy()
    correction = np.asarray(result.correction_mask, dtype=bool)
    tint = preview.copy()
    tint[correction] = (255, 65, 65)
    preview = cv2.addWeighted(preview, 0.75, tint, 0.25, 0.0)

    correction_contours, _ = cv2.findContours(
        correction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boundary_layer = preview.copy()
    cv2.drawContours(boundary_layer, correction_contours, -1, (235, 55, 55), 1, cv2.LINE_AA)
    outer_contours, _ = cv2.findContours(
        part_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(boundary_layer, outer_contours, -1, (20, 20, 20), 1, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.42, boundary_layer, 0.58, 0.0)

    zero_layer = preview.copy()
    zero_radius = max(2, int(round(short * 0.0025)))
    for zero in result.zero_points:
        point = tuple(np.rint(zero["point"]).astype(int).tolist())
        cv2.circle(zero_layer, point, zero_radius, (35, 200, 65), cv2.FILLED, cv2.LINE_AA)
    preview = cv2.addWeighted(preview, 0.25, zero_layer, 0.75, 0.0)

    region_layer = preview.copy()
    region_labels: list[tuple[str, tuple[int, int]]] = []
    for region in router.extract_regions(correction.astype(np.uint8) * 255):
        centroid = np.asarray(region["centroid"], dtype=np.float64)
        component_y, component_x = np.where(region["component_mask"] > 0)
        nearest = int(
            np.argmin((component_x - centroid[0]) ** 2 + (component_y - centroid[1]) ** 2)
        )
        x, y = int(component_x[nearest]), int(component_y[nearest])
        diamond = np.asarray(
            [(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)], dtype=np.int32
        ).reshape(-1, 1, 2)
        cv2.polylines(region_layer, [diamond], True, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.polylines(region_layer, [diamond], True, (255, 0, 210), 1, cv2.LINE_AA)
        region_labels.append((str(region["label"]), (x + 7, max(14, y - 6))))
    preview = cv2.addWeighted(preview, 0.35, region_layer, 0.65, 0.0)
    for label, origin in region_labels:
        _outlined_text(preview, label, origin, colour=(255, 0, 210))

    route_layer = preview.copy()
    boundary_lines = [line for line in result.lines if line.get("kind") == "boundary_connection"]
    selected_endpoints: list[tuple[int, int]] = []
    selected_endpoint_set: set[tuple[int, int]] = set()
    for line in boundary_lines:
        current = np.asarray(line.get("points", []), dtype=np.float64)
        original = np.asarray(line.get("preSnapPoints", current), dtype=np.float64)
        if len(original) >= 2:
            _draw_dashed(route_layer, original, (70, 70, 70), 1)
        if len(current) >= 2:
            colour = (245, 0, 190) if line.get("cadSnapped") else (0, 190, 235)
            rounded = np.rint(current).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(route_layer, [rounded], False, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.polylines(route_layer, [rounded], False, colour, 2, cv2.LINE_AA)
            for point in (tuple(rounded[0, 0]), tuple(rounded[-1, 0])):
                endpoint = (int(point[0]), int(point[1]))
                if endpoint not in selected_endpoint_set:
                    selected_endpoint_set.add(endpoint)
                    selected_endpoints.append(endpoint)

    for record in result.snap_records:
        feature = np.asarray(record.get("featurePoints", []), dtype=np.int32)
        if len(feature) < 2:
            continue
        cv2.polylines(
            route_layer, [feature.reshape(-1, 1, 2)], False,
            (15, 105, 255), 1, cv2.LINE_AA,
        )

    preview = cv2.addWeighted(preview, 0.34, route_layer, 0.66, 0.0)
    for index, point in enumerate(selected_endpoints, start=1):
        cv2.circle(preview, point, 7, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(preview, point, 6, (0, 205, 255), 2, cv2.LINE_AA)
        _outlined_text(
            preview, f"Z{index}", (point[0] + 7, max(14, point[1] - 7)),
            colour=(0, 145, 210), scale=0.38,
        )

    ys, xs = np.where(part_mask.astype(bool))
    if xs.size:
        # Keep enough whitespace for M/Z labels placed just outside an edge.
        margin = 45
        x0, x1 = max(0, int(xs.min()) - margin), min(preview.shape[1], int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(preview.shape[0], int(ys.max()) + margin + 1)
        preview = preview[y0:y1, x0:x1]

    header_height = 207
    canvas = np.full(
        (preview.shape[0] + header_height + 18, preview.shape[1] + 36, 3),
        255,
        dtype=np.uint8,
    )
    canvas[header_height : header_height + preview.shape[0], 18 : 18 + preview.shape[1]] = preview
    cv2.rectangle(canvas, (18, 15), (510, 187), (70, 70, 70), 1)
    cv2.rectangle(canvas, (530, 15), (canvas.shape[1] - 18, 187), (70, 70, 70), 1)
    cv2.putText(
        canvas, "CAD feature snap - boundary correction regions only", (32, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.57, (20, 20, 20), 2, cv2.LINE_AA,
    )
    legend = [
        ((235, 55, 55), "red shade / edge: HSV correction region"),
        ((35, 200, 65), "green dots: outer zero candidates (-0.10 to +0.10 mm)"),
        ((0, 205, 255), "Z labels / ringed endpoints: selected zero points"),
        ((70, 70, 70), "dashed gray: original shortest route"),
        ((15, 105, 255), "blue: matched exact STP feature curve"),
        ((245, 0, 190), "magenta: final CAD-snapped zero line"),
        ((0, 190, 235), "cyan: route with no valid CAD snap"),
    ]
    for index, (colour, text) in enumerate(legend):
        y = 62 + index * 19
        cv2.line(canvas, (32, y), (55, y), colour, 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (64, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)

    summary_x = 548
    cv2.putText(
        canvas, "Snap decision", (summary_x, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA,
    )
    snapped_lines = sum(bool(line.get("cadSnapped")) for line in boundary_lines)
    feature_counts: dict[str, int] = {}
    for record in result.snap_records:
        label = str(record.get("featureLabel", record.get("featureKind", "feature")))
        feature_counts[label] = feature_counts.get(label, 0) + 1
    details = ", ".join(f"{name} {count}" for name, count in sorted(feature_counts.items())) or "none"
    summary = [
        f"distance <= {max_distance:.1f}px   direction difference <= {maximum_angle_deg:.0f}deg",
        f"parallel run >= {minimum_run:.1f}px   correction collision: rejected",
        f"boundary routes: {len(boundary_lines)}   snapped routes: {snapped_lines}   snap sections: {len(result.snap_records)}",
        f"matched features: {details}",
    ]
    for index, text in enumerate(summary):
        cv2.putText(
            canvas, text, (summary_x, 65 + index * 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA,
        )
    return canvas


__all__ = [
    "FEATURE_KIND_LABELS",
    "discover_local_ai_features",
    "load_projected_cad_features",
    "render_cad_snap_preview",
    "snap_boundary_lines_to_cad",
]
