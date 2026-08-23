"""Remove annotation labels from 3D scan deviation-map images.

The script keeps the largest dense foreground object (the scanned part) and
replaces labels, leader lines, titles, axes, and legends with a white
background. Input files are never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    """Read an image from a path that may contain Korean characters."""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG image to a path that may contain Korean characters."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"이미지를 PNG로 변환할 수 없습니다: {path}")
    encoded.tofile(path)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest connected foreground component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("스캔 형상으로 판단할 전경 영역을 찾지 못했습니다.")

    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == component, 255, 0).astype(np.uint8)


def detect_label_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect red/gray rounded numeric boxes and return expanded rectangles."""
    height, width = image.shape[:2]
    blue, green, red = cv2.split(image)

    red_fill = (
        (red > 235) & (green < 55) & (blue < 55)
    ).astype(np.uint8) * 255

    channel_min = image.min(axis=2)
    channel_max = image.max(axis=2)
    # The export format uses a stable mid-gray outline around every numeric
    # box, including boxes whose translucent fill follows the scan color.
    gray_outline = (
        (channel_min >= 145)
        & (channel_max <= 165)
        & ((channel_max - channel_min) <= 4)
    ).astype(np.uint8) * 255

    rectangles: list[tuple[int, int, int, int]] = []

    def collect(mask: np.ndarray, min_fill: float, max_fill: float) -> None:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for x, y, box_width, box_height, area in stats[1:]:
            fill_ratio = area / float(box_width * box_height)
            if not (24 <= box_width <= 90 and 18 <= box_height <= 58):
                continue
            if not (min_fill <= fill_ratio <= max_fill):
                continue
            if area < 100:
                continue

            rectangles.append(
                (
                    int(x),
                    int(y),
                    int(x + box_width),
                    int(y + box_height),
                )
            )

    # Red fill occupies most of a number box.
    collect(red_fill, min_fill=0.40, max_fill=1.0)

    # Closed gray outlines are detected as contours. RETR_LIST also returns
    # individual boxes when two outlines touch and form one connected component.
    contours, _ = cv2.findContours(
        gray_outline, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not (24 <= box_width <= 90 and 18 <= box_height <= 58):
            continue
        if cv2.contourArea(contour) / float(box_width * box_height) < 0.62:
            continue
        rectangles.append(
            (
                x,
                y,
                x + box_width,
                y + box_height,
            )
        )

    # A gray outline can merge with a similarly colored scan surface. In that
    # case findContours returns the inner edge of the label, which is two
    # pixels smaller than the normal outer box. Expand only candidates that
    # contain dark numeric text and are touched by an exact-blue leader line;
    # this avoids expanding ordinary gray details on the scanned part.
    dark_text = image.max(axis=2) <= 95
    exact_blue = (blue == 255) & (green == 0) & (red == 0)
    normalized_rectangles: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in rectangles:
        box_width = x1 - x0
        box_height = y1 - y0
        text_pixel_count = int(np.count_nonzero(dark_text[y0:y1, x0:x1]))
        red_fill_count = int(np.count_nonzero(red_fill[y0:y1, x0:x1]))

        margin = 6
        nx0 = max(0, x0 - margin)
        ny0 = max(0, y0 - margin)
        nx1 = min(width, x1 + margin)
        ny1 = min(height, y1 + margin)
        leader_nearby = bool(np.any(exact_blue[ny0:ny1, nx0:nx1]))

        inner_gray_box = (
            box_height <= 32
            and box_width <= 55
            and (text_pixel_count >= 8 or red_fill_count >= 100)
            and leader_nearby
        )
        if inner_gray_box:
            x0 = max(0, x0 - 1)
            y0 = max(0, y0 - 1)
            x1 = min(width, x1 + 1)
            y1 = min(height, y1 + 1)

        normalized_rectangles.append((x0, y0, x1, y1))

    rectangles = normalized_rectangles

    # Outer and inner contours of one rounded box produce nearly identical
    # rectangles. Collapse them before building masks.
    unique: list[tuple[int, int, int, int]] = []
    for candidate in sorted(
        rectangles, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]), reverse=True
    ):
        x0, y0, x1, y1 = candidate
        candidate_area = (x1 - x0) * (y1 - y0)
        duplicate = False
        for kept in unique:
            kx0, ky0, kx1, ky1 = kept
            intersection = max(0, min(x1, kx1) - max(x0, kx0)) * max(
                0, min(y1, ky1) - max(y0, ky0)
            )
            kept_area = (kx1 - kx0) * (ky1 - ky0)
            union = candidate_area + kept_area - intersection
            if union and intersection / union > 0.70:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def detect_leader_lines(
    image: np.ndarray,
    label_boxes: list[tuple[int, int, int, int]],
    scan_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int, int, tuple[int, int, int]]],
]:
    """Return thin line masks and compact point-restoration specifications."""
    height, width = image.shape[:2]
    blue, green, red = cv2.split(image)
    pure_blue = (
        (blue > 225) & (green < 70) & (red < 70)
    ).astype(np.uint8) * 255
    line_source = cv2.morphologyEx(
        pure_blue,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    box_neighborhood = np.zeros((height, width), dtype=np.uint8)
    reach = max(8, int(round(min(height, width) * 0.012)))
    for x0, y0, x1, y1 in label_boxes:
        cv2.rectangle(
            box_neighborhood,
            (max(0, x0 - reach), max(0, y0 - reach)),
            (min(width - 1, x1 + reach), min(height - 1, y1 + reach)),
            255,
            thickness=-1,
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_source, connectivity=8
    )
    raw_line_mask = np.zeros_like(pure_blue)
    point_mask = np.zeros_like(pure_blue)
    point_specs: list[tuple[int, int, int, tuple[int, int, int]]] = []
    maximum_area = int(height * width * 0.006)
    point_radius = max(3, int(round(min(height, width) * 0.004)))
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 3 or area > maximum_area:
            continue
        component_mask = labels == component
        if not np.any(box_neighborhood[component_mask] > 0):
            continue

        raw_line_mask[component_mask] = 255

        # The measurement point is the end of the blue component farthest from
        # every label rectangle. Prefer pixels that overlap the scanned part.
        ys, xs = np.where(component_mask)
        inside_scan = scan_mask[ys, xs] > 0
        if np.any(inside_scan):
            xs = xs[inside_scan]
            ys = ys[inside_scan]

        nearest_box_distance = np.full(xs.shape, np.inf, dtype=np.float64)
        for x0, y0, x1, y1 in label_boxes:
            dx = np.maximum(np.maximum(x0 - xs, 0), xs - (x1 - 1))
            dy = np.maximum(np.maximum(y0 - ys, 0), ys - (y1 - 1))
            nearest_box_distance = np.minimum(
                nearest_box_distance, dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2
            )
        point_index = int(np.argmax(nearest_box_distance))
        point_x = int(xs[point_index])
        point_y = int(ys[point_index])
        cv2.circle(
            point_mask,
            (point_x, point_y),
            point_radius,
            255,
            thickness=-1,
        )
        point_color = tuple(int(value) for value in image[point_y, point_x])
        point_specs.append(
            (point_x, point_y, max(2, point_radius - 1), point_color)
        )

    # A 3x3 expansion covers anti-aliasing without erasing a region much wider
    # than the original one- or two-pixel leader line.
    line_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    full_line_mask = cv2.dilate(raw_line_mask, line_kernel, iterations=1)

    # Point-removal versions additionally cover the endpoint marker. Label-only
    # versions remove the full line first and redraw a compact point afterward.
    protected_points = cv2.dilate(
        point_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    line_with_points = cv2.bitwise_or(full_line_mask, protected_points)
    return full_line_mask, line_with_points, point_specs


def detect_exact_hsv_leader_lines(
    image: np.ndarray,
    label_boxes: list[tuple[int, int, int, int]],
    scan_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    list[tuple[int, int, int, tuple[int, int, int]]],
]:
    """Trace label leaders from their exact HSV center color for labels_white."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    # Pure BGR blue (255, 0, 0) maps to HSV (120, 255, 255) in OpenCV.
    exact_center = (
        (hue == 120) & (saturation == 255) & (value == 255)
    ).astype(np.uint8)

    # A true leader stays one or two pixels wide. Exact-blue regions that
    # become locally wider belong to the scan surface and are not followed.
    local_blue_count = cv2.boxFilter(
        exact_center,
        ddepth=cv2.CV_16U,
        ksize=(7, 7),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    thin_center = np.where(
        (exact_center > 0) & (local_blue_count <= 14), 255, 0
    ).astype(np.uint8)

    box_neighborhood = np.zeros((height, width), dtype=np.uint8)
    reach = max(8, int(round(min(height, width) * 0.012)))
    for x0, y0, x1, y1 in label_boxes:
        cv2.rectangle(
            box_neighborhood,
            (max(0, x0 - reach), max(0, y0 - reach)),
            (min(width - 1, x1 + reach), min(height - 1, y1 + reach)),
            255,
            thickness=-1,
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        thin_center, connectivity=8
    )
    selected_center = np.zeros_like(thin_center)
    point_specs: list[tuple[int, int, int, tuple[int, int, int]]] = []
    point_radius = max(3, int(round(min(height, width) * 0.004)))

    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        component_x = int(stats[component, cv2.CC_STAT_LEFT])
        component_y = int(stats[component, cv2.CC_STAT_TOP])
        box_width = int(stats[component, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if area < 5 or max(box_width, box_height) < 6:
            continue

        component_labels = labels[
            component_y : component_y + box_height,
            component_x : component_x + box_width,
        ]
        component_mask = component_labels == component
        neighborhood_roi = box_neighborhood[
            component_y : component_y + box_height,
            component_x : component_x + box_width,
        ]
        if not np.any(neighborhood_roi[component_mask] > 0):
            continue
        selected_roi = selected_center[
            component_y : component_y + box_height,
            component_x : component_x + box_width,
        ]
        selected_roi[component_mask] = 255

        ys, xs = np.where(component_mask)
        xs = xs + component_x
        ys = ys + component_y

        # Associate this leader with the single label rectangle it actually
        # touches. Comparing every point with every nearby label can reverse
        # the endpoint on densely packed annotations such as JD_64XX2.
        associated_box: tuple[int, int, int, int] | None = None
        associated_distance = np.inf
        for candidate_box in label_boxes:
            x0, y0, x1, y1 = candidate_box
            dx = np.maximum(np.maximum(x0 - xs, 0), xs - (x1 - 1))
            dy = np.maximum(np.maximum(y0 - ys, 0), ys - (y1 - 1))
            minimum_distance = float(
                np.min(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)
            )
            if minimum_distance < associated_distance:
                associated_distance = minimum_distance
                associated_box = candidate_box

        inside_scan = scan_mask[ys, xs] > 0
        if np.any(inside_scan):
            xs = xs[inside_scan]
            ys = ys[inside_scan]

        if associated_box is None:
            continue
        x0, y0, x1, y1 = associated_box
        dx = np.maximum(np.maximum(x0 - xs, 0), xs - (x1 - 1))
        dy = np.maximum(np.maximum(y0 - ys, 0), ys - (y1 - 1))
        associated_box_distance = (
            dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2
        )
        point_index = int(np.argmax(associated_box_distance))
        point_x = int(xs[point_index])
        point_y = int(ys[point_index])

        # The exact-blue component stops at the near edge of a differently
        # colored point marker. Estimate the leader direction near that end
        # and move the removal center into the marker instead of centering it
        # on the last blue pixel.
        endpoint_distance = (xs - point_x) ** 2 + (ys - point_y) ** 2
        nearby = (endpoint_distance > 0) & (
            endpoint_distance <= max(36, (point_radius * 3) ** 2)
        )
        center_x = point_x
        center_y = point_y
        if np.any(nearby):
            direction_x = point_x - float(np.mean(xs[nearby]))
            direction_y = point_y - float(np.mean(ys[nearby]))
            direction_length = float(np.hypot(direction_x, direction_y))
            if direction_length > 0:
                center_x = int(
                    round(point_x + point_radius * direction_x / direction_length)
                )
                center_y = int(
                    round(point_y + point_radius * direction_y / direction_length)
                )
        center_x = int(np.clip(center_x, 0, width - 1))
        center_y = int(np.clip(center_y, 0, height - 1))
        point_color = tuple(
            int(channel) for channel in image[center_y, center_x]
        )
        point_specs.append(
            (center_x, center_y, point_radius + 1, point_color)
        )

    # Anti-aliased leader edges are a blend of pure blue and the underlying
    # scan color, so their HSV hue is not always close to 120. Include only
    # pixels next to the exact center whose blue channel is still clearly
    # dominant. This removes the fringe without widening the mask blindly.
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    blended_blue_edge = (
        (blue >= 80)
        & (blue >= green + 4)
        & (blue >= red + 4)
    ).astype(np.uint8) * 255
    center_neighborhood = cv2.dilate(
        selected_center,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    line_mask = cv2.bitwise_and(center_neighborhood, blended_blue_edge)
    return line_mask, point_specs


def extend_scan_colors(
    image: np.ndarray, scan_mask: np.ndarray, annotation_mask: np.ndarray
) -> np.ndarray:
    """Extend nearby scan colors under annotations before smooth inpainting."""
    known_scan = (scan_mask > 0) & (annotation_mask == 0)
    distance_input = np.where(known_scan, 0, 1).astype(np.uint8)
    _, nearest_labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    maximum_label = int(nearest_labels.max())
    colors = np.zeros((maximum_label + 1, 3), dtype=np.uint8)
    colors[nearest_labels[known_scan]] = image[known_scan]

    extended = image.copy()
    unknown = ~known_scan
    extended[unknown] = colors[nearest_labels[unknown]]
    return extended


def build_scan_mask(image: np.ndarray, foreground_threshold: int = 20) -> np.ndarray:
    """Return a mask containing only the main scanned part."""
    height, width = image.shape[:2]

    # The source images use a nearly white background. This also captures gray
    # portions of a scanned part, unlike an HSV-saturation-only mask.
    distance_from_white = 255 - image.min(axis=2)
    foreground = np.where(distance_from_white >= foreground_threshold, 255, 0).astype(
        np.uint8
    )

    # Annotation leaders are only a few pixels thick. Opening disconnects them
    # while leaving the much denser scan body intact. Scale the kernel so the
    # same code works for different image resolutions.
    kernel_size = max(5, int(round(min(height, width) * 0.006)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dense_foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    scan_core = largest_component(dense_foreground)

    # Restore pixels lost around the scan boundary, but do not grow far enough
    # to bring back the disconnected annotation lines.
    restore_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    restored_region = cv2.dilate(scan_core, restore_kernel, iterations=1)
    scan_mask = cv2.bitwise_and(restored_region, foreground)

    # Include anti-aliased edge pixels next to the recovered scan mask.
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scan_mask = cv2.dilate(scan_mask, edge_kernel, iterations=1)

    return scan_mask


def build_annotation_masks(
    image: np.ndarray, scan_mask: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int, int, tuple[int, int, int]]],
]:
    """Build label+line and label+line+point removal masks."""
    height, width = image.shape[:2]
    label_boxes = detect_label_boxes(image)
    full_line_mask, line_with_points, point_specs = detect_leader_lines(
        image, label_boxes, scan_mask
    )

    exact_label_mask = np.zeros((height, width), dtype=np.uint8)
    for x0, y0, x1, y1 in label_boxes:
        cv2.rectangle(
            exact_label_mask, (x0, y0), (x1 - 1, y1 - 1), 255, thickness=-1
        )

    labels_and_lines_mask = cv2.bitwise_or(exact_label_mask, full_line_mask)
    labels_lines_and_points_mask = cv2.bitwise_or(
        exact_label_mask, line_with_points
    )
    return labels_and_lines_mask, labels_lines_and_points_mask, point_specs


def build_labels_white_mask(
    image: np.ndarray, scan_mask: np.ndarray
) -> np.ndarray:
    """Build the labels_white mask using exact-HSV thin-line tracing."""
    height, width = image.shape[:2]
    label_boxes = detect_label_boxes(image)
    line_mask, _ = detect_exact_hsv_leader_lines(
        image, label_boxes, scan_mask
    )

    label_mask = np.zeros((height, width), dtype=np.uint8)
    for x0, y0, x1, y1 in label_boxes:
        cv2.rectangle(label_mask, (x0, y0), (x1 - 1, y1 - 1), 255, thickness=-1)
    return cv2.bitwise_or(label_mask, line_mask)


def build_measurement_point_mask(
    image: np.ndarray, scan_mask: np.ndarray
) -> np.ndarray:
    """Build compact masks around the non-blue point at each leader endpoint."""
    height, width = image.shape[:2]
    label_boxes = detect_label_boxes(image)
    _, point_specs = detect_exact_hsv_leader_lines(
        image, label_boxes, scan_mask
    )

    point_mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, radius, _ in point_specs:
        cv2.circle(point_mask, (x, y), radius, 255, thickness=-1)
    return cv2.bitwise_and(point_mask, scan_mask)


def restore_points(
    image: np.ndarray,
    point_specs: list[tuple[int, int, int, tuple[int, int, int]]],
) -> np.ndarray:
    """Redraw only compact measurement points using their detected colors."""
    restored = image.copy()
    for x, y, radius, color in point_specs:
        cv2.circle(
            restored,
            (x, y),
            radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return restored


def render_white_version(
    image: np.ndarray, scan_mask: np.ndarray, removal_mask: np.ndarray
) -> np.ndarray:
    """Keep the scan and replace removed regions with pure white."""
    result = np.full_like(image, 255)
    keep = (scan_mask > 0) & (removal_mask == 0)
    result[keep] = image[keep]
    return result


def render_inpaint_version(
    image: np.ndarray, scan_mask: np.ndarray, removal_mask: np.ndarray
) -> np.ndarray:
    """Approximately restore removed regions from nearby scan colors."""
    inpaint_region = cv2.bitwise_and(removal_mask, scan_mask)
    extended_image = extend_scan_colors(image, scan_mask, removal_mask)
    restored_image = cv2.inpaint(
        extended_image, inpaint_region, 7, cv2.INPAINT_TELEA
    )

    result = np.full_like(image, 255)
    result[scan_mask > 0] = restored_image[scan_mask > 0]
    return result


def create_labels_inpainted_from_white(
    labels_white: np.ndarray,
    scan_mask: np.ndarray,
    labels_white_mask: np.ndarray,
) -> np.ndarray:
    """Fill only the regions that are blank in the finished labels-white image."""
    blank_pixels = np.all(labels_white == 255, axis=2)
    fill_region = (
        (labels_white_mask > 0) & (scan_mask > 0) & blank_pixels
    )
    fill_mask = np.where(fill_region, 255, 0).astype(np.uint8)
    return render_inpaint_version(labels_white, scan_mask, fill_mask)


def create_labels_points_white_from_white(
    labels_white: np.ndarray,
    scan_mask: np.ndarray,
    point_mask: np.ndarray,
) -> np.ndarray:
    """Remove only measurement points from the finished labels-white image."""
    result = labels_white.copy()
    remove_points = (scan_mask > 0) & (point_mask > 0)
    result[remove_points] = 255
    return result


def create_labels_points_inpainted_from_versions(
    labels_white: np.ndarray,
    labels_inpainted: np.ndarray,
    labels_points_white: np.ndarray,
    scan_mask: np.ndarray,
) -> np.ndarray:
    """Inpaint only point pixels while preserving the finished version 2."""
    newly_removed_points = (
        np.any(labels_points_white != labels_white, axis=2)
        & (scan_mask > 0)
    )
    point_fill_mask = np.where(
        newly_removed_points, 255, 0
    ).astype(np.uint8)

    point_blank_source = labels_inpainted.copy()
    point_blank_source[newly_removed_points] = 255
    return render_inpaint_version(
        point_blank_source, scan_mask, point_fill_mask
    )


def create_versions(image: np.ndarray) -> dict[str, np.ndarray]:
    """Create the four requested label-removal versions."""
    scan_mask = build_scan_mask(image)
    labels_white_mask = build_labels_white_mask(image, scan_mask)
    point_mask = build_measurement_point_mask(image, scan_mask)

    labels_white = render_white_version(image, scan_mask, labels_white_mask)
    labels_inpainted = create_labels_inpainted_from_white(
        labels_white, scan_mask, labels_white_mask
    )
    labels_points_white = create_labels_points_white_from_white(
        labels_white, scan_mask, point_mask
    )
    labels_points_inpainted = create_labels_points_inpainted_from_versions(
        labels_white,
        labels_inpainted,
        labels_points_white,
        scan_mask,
    )

    return {
        "1_labels_white": labels_white,
        "2_labels_inpainted": labels_inpainted,
        "3_labels_points_white": labels_points_white,
        "4_labels_points_inpainted": labels_points_inpainted,
    }


def remove_labels(image: np.ndarray, foreground_threshold: int = 20) -> np.ndarray:
    """Backward-compatible default: approximate restoration without points."""
    scan_mask = build_scan_mask(image, foreground_threshold)
    labels_mask, _, point_specs = build_annotation_masks(image, scan_mask)
    result = render_inpaint_version(image, scan_mask, labels_mask)
    return restore_points(result, point_specs)


def process_directory(input_dir: Path, output_dir: Path) -> int:
    """Process all supported images in input_dir and return the image count."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"처리할 이미지가 없습니다: {input_dir}")

    for source_path in image_paths:
        image = read_image(source_path)
        versions = create_versions(image)
        for version_name, cleaned in versions.items():
            version_dir = output_dir / version_name
            version_dir.mkdir(parents=True, exist_ok=True)
            destination = version_dir / f"{source_path.stem}_{version_name}.png"
            write_png(destination, cleaned)
            print(f"완료: {source_path.name} -> {version_name}/{destination.name}")

    return len(image_paths)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="3D 스캔 편차 이미지에서 라벨과 주석을 제거합니다."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "input",
        help="원본 이미지 폴더 (기본값: label_removal/input)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output",
        help="결과 이미지 폴더 (기본값: label_removal/output)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = process_directory(args.input_dir.resolve(), args.output_dir.resolve())
    print(f"총 {count}개 이미지를 처리했습니다.")


if __name__ == "__main__":
    main()
