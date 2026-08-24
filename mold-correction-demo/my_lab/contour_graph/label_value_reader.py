"""Read printed deviation values and associate them with contour points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
LABEL_REMOVAL_DIR = HERE.parent / "label_removal"
if str(LABEL_REMOVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LABEL_REMOVAL_DIR))

from remove_labels import build_scan_mask, detect_label_boxes  # noqa: E402
from label_numeric_ocr import read_numeric_label  # noqa: E402


Box = tuple[int, int, int, int]


@dataclass
class LabelReading:
    box: Box
    value: float
    ocr_confidence: float
    traced_point: tuple[int, int] | None


def _trace_boxes_to_points(
    image: np.ndarray, boxes: list[Box]
) -> dict[Box, tuple[int, int]]:
    """Trace exact-blue leaders while retaining their associated label box."""
    height, width = image.shape[:2]
    scan_mask = build_scan_mask(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    exact_blue = (
        (hsv[:, :, 0] == 120)
        & (hsv[:, :, 1] == 255)
        & (hsv[:, :, 2] == 255)
    ).astype(np.uint8)
    local_count = cv2.boxFilter(
        exact_blue,
        ddepth=cv2.CV_16U,
        ksize=(7, 7),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    thin = ((exact_blue > 0) & (local_count <= 14)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    point_radius = max(3, int(round(min(height, width) * 0.004)))
    choices: dict[Box, tuple[float, tuple[int, int]]] = {}

    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if area < 5 or max(component_width, component_height) < 6:
            continue
        yy, xx = np.where(labels == component)
        if not len(xx):
            continue

        associated_box: Box | None = None
        nearest_distance = float("inf")
        for box in boxes:
            x0, y0, x1, y1 = box
            dx = np.maximum(np.maximum(x0 - xx, 0), xx - (x1 - 1))
            dy = np.maximum(np.maximum(y0 - yy, 0), yy - (y1 - 1))
            distance = float(np.min(dx.astype(float) ** 2 + dy.astype(float) ** 2))
            if distance < nearest_distance:
                nearest_distance = distance
                associated_box = box
        if associated_box is None or nearest_distance > 14.0**2:
            continue

        inside_scan = scan_mask[yy, xx] > 0
        if np.any(inside_scan):
            xx = xx[inside_scan]
            yy = yy[inside_scan]
        x0, y0, x1, y1 = associated_box
        dx = np.maximum(np.maximum(x0 - xx, 0), xx - (x1 - 1))
        dy = np.maximum(np.maximum(y0 - yy, 0), yy - (y1 - 1))
        distance2 = dx.astype(float) ** 2 + dy.astype(float) ** 2
        endpoint_index = int(np.argmax(distance2))
        endpoint_x = int(xx[endpoint_index])
        endpoint_y = int(yy[endpoint_index])

        endpoint_distance2 = (xx - endpoint_x) ** 2 + (yy - endpoint_y) ** 2
        nearby = (endpoint_distance2 > 0) & (
            endpoint_distance2 <= max(36, (point_radius * 3) ** 2)
        )
        center_x, center_y = endpoint_x, endpoint_y
        if np.any(nearby):
            direction_x = endpoint_x - float(np.mean(xx[nearby]))
            direction_y = endpoint_y - float(np.mean(yy[nearby]))
            length = float(np.hypot(direction_x, direction_y))
            if length > 0:
                center_x = int(round(endpoint_x + point_radius * direction_x / length))
                center_y = int(round(endpoint_y + point_radius * direction_y / length))
        center = (
            int(np.clip(center_x, 0, width - 1)),
            int(np.clip(center_y, 0, height - 1)),
        )
        leader_extent = float(np.sqrt(distance2[endpoint_index]))
        previous = choices.get(associated_box)
        if previous is None or leader_extent > previous[0]:
            choices[associated_box] = (leader_extent, center)
    return {box: point for box, (_extent, point) in choices.items()}


def read_labels(image: np.ndarray) -> list[LabelReading]:
    boxes = detect_label_boxes(image)
    traced = _trace_boxes_to_points(image, boxes)
    readings: list[LabelReading] = []
    for box in boxes:
        x0, y0, x1, y1 = box
        value, confidence = read_numeric_label(image[y0:y1, x0:x1])
        if value is None:
            continue
        readings.append(
            LabelReading(
                box=box,
                value=value,
                ocr_confidence=confidence,
                traced_point=traced.get(box),
            )
        )
    return readings


def _point_to_box_distance(point: tuple[int, int], box: Box) -> float:
    x, y = point
    x0, y0, x1, y1 = box
    dx = max(x0 - x, 0, x - (x1 - 1))
    dy = max(y0 - y, 0, y - (y1 - 1))
    return float(np.hypot(dx, dy))


def match_readings_to_points(
    readings: list[LabelReading], points: list[tuple[int, int]], image_shape: tuple[int, ...]
) -> dict[tuple[int, int], LabelReading]:
    """Create a one-to-one label-to-contour assignment without point IDs."""
    matched: dict[tuple[int, int], LabelReading] = {}
    used_readings: set[int] = set()
    used_points: set[int] = set()
    scale = max(1.0, min(image_shape[:2]) / 680.0)

    direct_candidates: list[tuple[float, int, int]] = []
    for reading_index, reading in enumerate(readings):
        if reading.traced_point is None:
            continue
        for point_index, point in enumerate(points):
            distance = float(np.hypot(
                reading.traced_point[0] - point[0], reading.traced_point[1] - point[1]
            ))
            if distance <= 20.0 * scale:
                direct_candidates.append((distance, reading_index, point_index))
    for _distance, reading_index, point_index in sorted(direct_candidates):
        if reading_index in used_readings or point_index in used_points:
            continue
        matched[points[point_index]] = readings[reading_index]
        used_readings.add(reading_index)
        used_points.add(point_index)

    # Short or faded leaders may not produce an exact-blue component. Assign
    # only the remaining labels and points with a global nearest-pair pass.
    fallback: list[tuple[float, int, int]] = []
    for reading_index, reading in enumerate(readings):
        if reading_index in used_readings:
            continue
        for point_index, point in enumerate(points):
            if point_index in used_points:
                continue
            fallback.append(
                (_point_to_box_distance(point, reading.box), reading_index, point_index)
            )
    for distance, reading_index, point_index in sorted(fallback):
        if reading_index in used_readings or point_index in used_points:
            continue
        if distance > min(image_shape[:2]) * 0.28:
            continue
        matched[points[point_index]] = readings[reading_index]
        used_readings.add(reading_index)
        used_points.add(point_index)
    return matched


__all__ = ["LabelReading", "match_readings_to_points", "read_labels"]
