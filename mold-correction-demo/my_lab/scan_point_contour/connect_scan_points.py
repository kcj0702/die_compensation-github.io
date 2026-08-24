"""Connect label leader endpoints into product-specific closed scan-point loops."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
LAB_DIR = HERE.parent
LABEL_REMOVAL_DIR = LAB_DIR / "label_removal"
if str(LABEL_REMOVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LABEL_REMOVAL_DIR))

from remove_labels import (  # noqa: E402
    build_scan_mask,
    detect_exact_hsv_leader_lines,
    detect_label_boxes,
    read_image,
    write_png,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COLORS = (
    (0, 0, 255),
    (0, 180, 0),
    (255, 80, 0),
    (180, 0, 180),
)


@dataclass
class PointLoop:
    name: str
    points: list[tuple[int, int]]
    color: tuple[int, int, int]
    path: list[tuple[int, int]] | None = None

    @property
    def drawing_path(self) -> list[tuple[int, int]]:
        return self.path if self.path else self.points

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "closed": True,
            "point_count": len(self.points),
            "points": [[x, y] for x, y in self.points],
            "path_point_count": len(self.drawing_path),
            "connection_path": [[x, y] for x, y in self.drawing_path],
        }


def _detected_points(original: np.ndarray) -> list[tuple[int, int]]:
    scan_mask = build_scan_mask(original)
    boxes = detect_label_boxes(original)
    _line_mask, point_specs = detect_exact_hsv_leader_lines(
        original, boxes, scan_mask
    )
    points: list[tuple[int, int]] = []
    for x, y, _radius, _color in point_specs:
        point = (int(x), int(y))
        if point not in points:
            points.append(point)
    return points


def _nearest_contour_position(
    point: tuple[int, int], contour: np.ndarray
) -> tuple[float, int]:
    coordinates = contour.reshape(-1, 2).astype(np.float32)
    delta = coordinates - np.asarray(point, dtype=np.float32)
    distances2 = np.einsum("ij,ij->i", delta, delta)
    index = int(np.argmin(distances2))
    return float(np.sqrt(distances2[index])), index


def _outer_contour(clean: np.ndarray) -> np.ndarray:
    mask = build_scan_mask(clean)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("제품 외곽선을 찾지 못했습니다.")
    return max(contours, key=cv2.contourArea)


def _rounded_rect_openings(clean: np.ndarray) -> list[np.ndarray]:
    """Return the two large, rectangular white openings in JD_64XX."""
    white = np.all(clean >= 245, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    candidates: list[tuple[int, np.ndarray]] = []
    image_area = clean.shape[0] * clean.shape[1]
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if not (image_area * 0.006 <= area <= image_area * 0.08):
            continue
        rectangularity = area / float(max(width * height, 1))
        aspect = width / float(max(height, 1))
        if rectangularity < 0.84 or not (0.75 <= aspect <= 1.55):
            continue
        component_mask = np.where(labels == component, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            candidates.append((area, max(contours, key=cv2.contourArea)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [contour for _area, contour in candidates[:2]]


def _unrepresented_label_boxes(
    original: np.ndarray, points: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    """Find number boxes whose thin HSV leader produced no detected point."""
    boxes = detect_label_boxes(original)
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    exact = (
        (hsv[:, :, 0] == 120)
        & (hsv[:, :, 1] == 255)
        & (hsv[:, :, 2] == 255)
    ).astype(np.uint8)
    local_count = cv2.boxFilter(
        exact,
        ddepth=cv2.CV_16U,
        ksize=(7, 7),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    thin = ((exact > 0) & (local_count <= 14)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if area < 5 or max(width, height) < 6:
            continue
        yy, xx = np.where(labels == component)
        components.append((xx, yy))

    used_components: set[int] = set()
    used_boxes: set[tuple[int, int, int, int]] = set()
    for x, y in points:
        choices: list[tuple[float, int]] = []
        for index, (xx, yy) in enumerate(components):
            if index in used_components:
                continue
            distance2 = (xx - x) ** 2 + (yy - y) ** 2
            choices.append((float(distance2.min()), index))
        if not choices:
            continue
        distance2, component_index = min(choices)
        if distance2 > 15.0 ** 2:
            continue
        used_components.add(component_index)
        xx, yy = components[component_index]

        def box_distance(box: tuple[int, int, int, int]) -> float:
            bx0, by0, bx1, by1 = box
            dx = np.maximum(np.maximum(bx0 - xx, 0), xx - (bx1 - 1))
            dy = np.maximum(np.maximum(by0 - yy, 0), yy - (by1 - 1))
            return float(np.min(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2))

        used_boxes.add(min(boxes, key=box_distance))
    return [box for box in boxes if box not in used_boxes]


def _ordered_near_contour(
    points: list[tuple[int, int]],
    contour: np.ndarray,
    maximum_distance: float,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    selected: list[tuple[int, int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()
    for point in points:
        distance, index = _nearest_contour_position(point, contour)
        if distance <= maximum_distance:
            selected.append((index, point[0], point))
            used.add(point)
    selected.sort(key=lambda item: item[0])
    return [item[2] for item in selected], used


def _filter_contour_distance_outliers(
    points: list[tuple[int, int]], contour: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Reject isolated points far from an otherwise consistent hole contour."""
    if len(points) < 5:
        return points, []
    distances = np.asarray(
        [_nearest_contour_position(point, contour)[0] for point in points],
        dtype=np.float32,
    )
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    robust_sigma = 1.4826 * mad
    threshold = median + max(3.0, 3.5 * robust_sigma)
    accepted = [
        point for point, distance in zip(points, distances) if distance <= threshold
    ]
    rejected = [
        point for point, distance in zip(points, distances) if distance > threshold
    ]
    return accepted, rejected


def _simplified_contour_path(
    contour: np.ndarray, epsilon_ratio: float = 0.003
) -> list[tuple[int, int]]:
    perimeter = float(cv2.arcLength(contour, True))
    approximation = cv2.approxPolyDP(
        contour, max(1.0, perimeter * epsilon_ratio), True
    ).reshape(-1, 2)
    return [tuple(map(int, point)) for point in approximation]


def _jd64_loops(
    points: list[tuple[int, int]], clean: np.ndarray, original: np.ndarray
) -> tuple[list[PointLoop], list[tuple[int, int]], list[tuple[int, int]]]:
    outer = _outer_contour(clean)
    openings = _rounded_rect_openings(clean)
    distance_limit = max(28.0, min(clean.shape[:2]) * 0.055)
    opening_distance_limit = max(18.0, min(clean.shape[:2]) * 0.035)
    supplemented = list(points)

    # A few JD_64XX labels use leaders that are too short for the strict HSV
    # tracer.  Their number boxes sit directly beside an opening, so project
    # only those nearby unused box centers to the nearest target opening.
    inferred: list[tuple[int, int]] = []
    for x0, y0, x1, y1 in _unrepresented_label_boxes(original, points):
        center = ((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0)
        best: tuple[float, tuple[int, int]] | None = None
        for contour in openings:
            coordinates = contour.reshape(-1, 2).astype(np.float32)
            delta = coordinates - np.asarray(center, dtype=np.float32)
            distances2 = np.einsum("ij,ij->i", delta, delta)
            index = int(np.argmin(distances2))
            distance = float(np.sqrt(distances2[index]))
            candidate = tuple(map(int, coordinates[index]))
            if best is None or distance < best[0]:
                best = (distance, candidate)
        if best is None or best[0] > 52.0:
            continue
        candidate = best[1]
        if all(np.hypot(candidate[0] - x, candidate[1] - y) > 14.0 for x, y in supplemented):
            supplemented.append(candidate)
            inferred.append(candidate)

    remaining = supplemented
    loops: list[PointLoop] = []

    # Inner openings are assigned first so nearby points are not consumed by
    # the much larger product contour.
    opening_names = ("large_rounded_opening", "small_rounded_opening")
    for index, contour in enumerate(openings):
        candidates, _candidate_set = _ordered_near_contour(
            remaining, contour, opening_distance_limit
        )
        ordered, _rejected = _filter_contour_distance_outliers(candidates, contour)
        used = set(ordered)
        if len(ordered) >= 3:
            loops.append(
                PointLoop(
                    opening_names[index],
                    ordered,
                    COLORS[index + 1],
                )
            )
            remaining = [point for point in remaining if point not in used]

    ordered_outer, used_outer = _ordered_near_contour(
        remaining, outer, distance_limit
    )
    if len(ordered_outer) >= 3:
        loops.insert(0, PointLoop("product_outer", ordered_outer, COLORS[0]))
    unused = [point for point in remaining if point not in used_outer]
    return loops, unused, inferred


def _jd71_loops(
    points: list[tuple[int, int]], clean: np.ndarray
) -> tuple[list[PointLoop], list[tuple[int, int]]]:
    outer = _outer_contour(clean)
    distance_limit = max(32.0, min(clean.shape[:2]) * 0.06)
    ordered, used = _ordered_near_contour(points, outer, distance_limit)
    loops = [PointLoop("product_outer", ordered, COLORS[0])] if len(ordered) >= 3 else []
    return loops, [point for point in points if point not in used]


def _largest_balanced_gap_threshold(values: np.ndarray) -> float:
    """Split two geometric layers at their largest non-outlier distance gap."""
    ordered = np.sort(values.astype(np.float64))
    if len(ordered) < 6:
        return float(np.median(ordered))
    minimum_side = max(2, int(round(len(ordered) * 0.20)))
    first = minimum_side - 1
    last = len(ordered) - minimum_side - 1
    gaps = np.diff(ordered)
    split = first + int(np.argmax(gaps[first : last + 1]))
    return float((ordered[split] + ordered[split + 1]) / 2.0)


def _nested_onion_labels(
    points: list[tuple[int, int]], outer_contour: np.ndarray
) -> np.ndarray:
    """Peel three measured point shells without product-specific point IDs.

    First separate points lying on the real product boundary. Then build the
    envelope of all remaining points and peel its boundary as the middle
    shell. Points left after the second peel form the innermost shell.
    """
    outer_distance = np.asarray(
        [abs(cv2.pointPolygonTest(outer_contour, point, True)) for point in points],
        dtype=np.float32,
    )
    outer_threshold = _largest_balanced_gap_threshold(outer_distance)
    outer_mask = outer_distance <= outer_threshold
    remaining_indices = np.flatnonzero(~outer_mask)
    if len(remaining_indices) < 6:
        raise RuntimeError("Not enough scan points remain for nested shells")

    remaining_contour = cv2.convexHull(
        np.asarray([points[index] for index in remaining_indices], dtype=np.int32)
        .reshape(-1, 1, 2)
    )
    envelope_distance = np.asarray(
        [
            abs(cv2.pointPolygonTest(remaining_contour, points[index], True))
            for index in remaining_indices
        ],
        dtype=np.float32,
    )
    middle_threshold = _largest_balanced_gap_threshold(envelope_distance)

    labels = np.full(len(points), 2, dtype=np.int32)
    labels[outer_mask] = 0
    labels[remaining_indices[envelope_distance <= middle_threshold]] = 1
    if any(np.count_nonzero(labels == shell) < 3 for shell in range(3)):
        raise RuntimeError("Could not separate three nested scan-point shells")
    return labels


def _jd67_loops(
    points: list[tuple[int, int]], clean: np.ndarray
) -> tuple[list[PointLoop], list[tuple[int, int]]]:
    outer = _outer_contour(clean)
    x, y, width, height = cv2.boundingRect(outer)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    radius_x = max(width / 2.0, 1.0)
    radius_y = max(height / 2.0, 1.0)
    labels = _nested_onion_labels(points, outer)
    point_groups: list[list[tuple[int, int]]] = []
    for shell in range(3):
        selected = [point for point, label in zip(points, labels) if label == shell]
        selected.sort(
            key=lambda point: np.arctan2(
                (point[1] - center_y) / radius_y,
                (point[0] - center_x) / radius_x,
            )
        )
        if len(selected) >= 3:
            point_groups.append(selected)
    loops = [
        PointLoop(
            f"nested_shell_{shell + 1}",
            selected,
            COLORS[shell],
        )
        for shell, selected in enumerate(point_groups)
    ]
    return loops, []


def connect_points(
    original: np.ndarray, clean: np.ndarray, filename: str
) -> tuple[
    list[PointLoop],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    points = _detected_points(original)
    inferred: list[tuple[int, int]] = []
    upper = filename.upper()
    if "JD_64XX2" in upper:
        loops, unused, inferred = _jd64_loops(points, clean, original)
    elif "JD_67XX6" in upper:
        loops, unused = _jd67_loops(points, clean)
    elif "JD_71XX2" in upper:
        loops, unused = _jd71_loops(points, clean)
    else:
        loops, unused = _jd71_loops(points, clean)
    all_points = points + [point for point in inferred if point not in points]
    return loops, all_points, unused, inferred


def render_points(
    clean: np.ndarray,
    points: list[tuple[int, int]],
    loops: list[PointLoop],
    unused: list[tuple[int, int]],
    inferred: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    detected = clean.copy()
    unused_set = set(unused)
    inferred_set = set(inferred)
    for index, point in enumerate(points, start=1):
        if point in unused_set:
            color = (130, 130, 130)
        elif point in inferred_set:
            color = (0, 140, 255)
        else:
            color = (0, 255, 255)
        cv2.circle(detected, point, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            detected,
            str(index),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    connected = clean.copy()
    for loop in loops:
        if len(loop.drawing_path) < 3:
            continue
        contour = np.asarray(loop.drawing_path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(connected, [contour], True, loop.color, 3, cv2.LINE_AA)
        if loop.path:
            for point in loop.points:
                cv2.circle(connected, point, 2, (150, 150, 150), -1, cv2.LINE_AA)
        for point in loop.drawing_path:
            cv2.circle(connected, point, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(connected, point, 5, (0, 0, 0), 1, cv2.LINE_AA)
        label_point = min(loop.points, key=lambda point: (point[1], point[0]))
        cv2.putText(
            connected,
            loop.name,
            (label_point[0] + 8, max(18, label_point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            loop.color,
            2,
            cv2.LINE_AA,
        )
    return detected, connected


def process_image(original_path: Path, clean_path: Path, output_root: Path) -> Path:
    original = read_image(original_path)
    clean = read_image(clean_path)
    loops, points, unused, inferred = connect_points(original, clean, original_path.name)
    detected, connected = render_points(clean, points, loops, unused, inferred)
    output_dir = output_root / original_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    write_png(output_dir / "01_detected_scan_points.png", detected)
    write_png(output_dir / "02_connected_closed_shapes.png", connected)
    payload = {
        "source": str(original_path),
        "detected_point_count": len(points),
        "connected_point_count": sum(len(loop.points) for loop in loops),
        "inferred_short_leader_point_count": len(inferred),
        "inferred_short_leader_points": [[x, y] for x, y in inferred],
        "unused_points": [[x, y] for x, y in unused],
        "loops": [loop.to_dict() for loop in loops],
    }
    (output_dir / "scan_point_loops.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir


def find_clean_image(original: Path, clean_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in clean_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and original.stem.lower() in path.stem.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"라벨 제거 이미지를 찾지 못했습니다: {original.name}")
    return candidates[0]
