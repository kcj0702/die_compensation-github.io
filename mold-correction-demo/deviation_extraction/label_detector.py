"""편차 맵에서 라벨 박스와 실제로 연결된 측정점 좌표를 찾는다."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

if __package__:  # 패키지 import와 직접 스크립트 실행을 모두 지원한다.
    from . import config
else:  # pragma: no cover - 직접 스크립트 실행 경로
    import config


Box = tuple[int, int, int, int]  # x, y, width, height
Rect = tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass
class LabelCandidate:
    """라벨 박스와 해당 라벨이 지시하는 픽셀 좌표 후보."""

    box: Box
    point_xy: tuple[int, int] | None
    label_color: str
    traced: bool


def _resolution_scale(shape: tuple[int, ...]) -> float:
    """고해상도 입력에서 거리와 후보 상한만 완만하게 확대한다."""
    return max(1.0, min(shape[:2]) / float(config.REFERENCE_SHORT_SIDE))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """전경이 없으면 빈 마스크, 있으면 가장 큰 연결 성분만 반환한다."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == component, 255, 0).astype(np.uint8)


def build_scan_mask(bgr: np.ndarray) -> np.ndarray:
    """얇은 주석을 끊은 뒤 가장 큰 조밀 전경을 스캔 본체로 분리한다."""
    height, width = bgr.shape[:2]
    distance_from_white = 255 - bgr.min(axis=2)
    foreground = np.where(
        distance_from_white >= config.FOREGROUND_THRESHOLD, 255, 0
    ).astype(np.uint8)

    kernel_size = max(5, int(round(min(height, width) * 0.006)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dense_foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    scan_core = _largest_component(dense_foreground)
    if np.count_nonzero(scan_core) < height * width * config.SCAN_MIN_AREA_RATIO:
        return np.zeros_like(scan_core)

    restored_region = cv2.dilate(scan_core, kernel, iterations=1)
    scan_mask = cv2.bitwise_and(restored_region, foreground)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(scan_mask, edge_kernel, iterations=1)


def _color_masks(
    bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """빨강, 안정 회색, 넓은 중성 회색, 검은 외곽선 마스크를 만든다."""
    blue, green, red = cv2.split(bgr)
    channel_min = bgr.min(axis=2)
    channel_max = bgr.max(axis=2)
    channel_delta = channel_max.astype(np.int16) - channel_min.astype(np.int16)

    red_fill = (
        (red >= config.LABEL_RED_MIN_RED)
        & (green <= config.LABEL_RED_MAX_GREEN)
        & (blue <= config.LABEL_RED_MAX_BLUE)
    )
    gray_outline = (
        (channel_min >= config.LABEL_GRAY_MIN)
        & (channel_max <= config.LABEL_GRAY_MAX)
        & (channel_delta <= config.LABEL_GRAY_MAX_CHANNEL_DELTA)
    )
    neutral_outline = (
        (channel_min >= config.LABEL_NEUTRAL_MIN)
        & (channel_max <= config.LABEL_NEUTRAL_MAX)
        & (channel_delta <= config.LABEL_NEUTRAL_MAX_CHANNEL_DELTA)
    )
    dark_outline = (
        (channel_max <= config.LABEL_DARK_MAX_VALUE)
        & (channel_delta <= config.LABEL_DARK_MAX_CHANNEL_DELTA)
    )
    return tuple(
        np.where(mask, 255, 0).astype(np.uint8)
        for mask in (red_fill, gray_outline, neutral_outline, dark_outline)
    )


def _box_size_is_valid(width: int, height: int, scale: float) -> bool:
    """라벨 export 크기와 종횡비에 맞는 사각형인지 확인한다."""
    if not (
        config.LABEL_MIN_WIDTH <= width <= round(config.LABEL_MAX_WIDTH * scale)
        and config.LABEL_MIN_HEIGHT
        <= height
        <= round(config.LABEL_MAX_HEIGHT * scale)
    ):
        return False
    aspect = width / float(height)
    return config.LABEL_MIN_ASPECT <= aspect <= config.LABEL_MAX_ASPECT


def _collect_red_components(
    red_fill: np.ndarray, scale: float
) -> list[Rect]:
    """채움률이 높은 빨간 연결 성분을 라벨 후보로 수집한다."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(red_fill, connectivity=8)
    rectangles: list[Rect] = []
    for x, y, width, height, area in stats[1:]:
        if not _box_size_is_valid(int(width), int(height), scale):
            continue
        if int(area) < config.LABEL_MIN_COMPONENT_AREA:
            continue
        fill_ratio = int(area) / float(int(width) * int(height))
        if fill_ratio < config.LABEL_RED_MIN_FILL_RATIO:
            continue
        rectangles.append(
            (int(x), int(y), int(x + width), int(y + height))
        )
    return rectangles


def _collect_outline_contours(
    mask: np.ndarray,
    scale: float,
    minimum_extent: float,
) -> list[Rect]:
    """닫힌 외곽선의 내부/외부 contour를 각각 라벨 후보로 수집한다."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectangles: list[Rect] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not _box_size_is_valid(width, height, scale):
            continue
        rect_area = width * height
        if rect_area == 0 or cv2.contourArea(contour) / rect_area < minimum_extent:
            continue
        rectangles.append((x, y, x + width, y + height))
    return rectangles


def _rect_iou(first: Rect, second: Rect) -> float:
    """두 xyxy 사각형의 IoU를 반환한다."""
    x0, y0, x1, y1 = first
    sx0, sy0, sx1, sy1 = second
    intersection = max(0, min(x1, sx1) - max(x0, sx0)) * max(
        0, min(y1, sy1) - max(y0, sy0)
    )
    if intersection == 0:
        return 0.0
    area = (x1 - x0) * (y1 - y0)
    second_area = (sx1 - sx0) * (sy1 - sy0)
    return intersection / float(area + second_area - intersection)


def _deduplicate_rectangles(rectangles: list[Rect]) -> list[Rect]:
    """동일 라벨의 외부/내부 contour 중 큰 사각형 하나만 남긴다."""
    ordered = sorted(
        rectangles,
        key=lambda item: (item[2] - item[0]) * (item[3] - item[1]),
        reverse=True,
    )
    unique: list[Rect] = []
    for candidate in ordered:
        if any(
            _rect_iou(candidate, kept) > config.LABEL_DUPLICATE_IOU
            for kept in unique
        ):
            continue
        unique.append(candidate)
    return unique


def _find_label_rectangles(
    bgr: np.ndarray,
    scan_mask: np.ndarray | None = None,
) -> list[Rect]:
    """색상·형상·밀도를 결합해 중복 없는 xyxy 라벨 후보를 찾는다."""
    height, width = bgr.shape[:2]
    scale = _resolution_scale(bgr.shape)
    red_fill, gray_outline, neutral_outline, dark_outline = _color_masks(bgr)
    outline_kernel = np.ones((3, 3), dtype=np.uint8)
    gray_outline = cv2.morphologyEx(
        gray_outline, cv2.MORPH_CLOSE, outline_kernel
    )
    neutral_outline = cv2.morphologyEx(
        neutral_outline, cv2.MORPH_CLOSE, outline_kernel
    )
    dark_outline = cv2.morphologyEx(
        dark_outline, cv2.MORPH_CLOSE, outline_kernel
    )

    rectangles = _collect_red_components(red_fill, scale)
    rectangles.extend(
        _collect_outline_contours(
            gray_outline, scale, config.LABEL_OUTLINE_MIN_EXTENT
        )
    )
    # 흰 라벨은 렌더링/압축에 따라 테두리가 145~165 범위를 벗어난다. 넓은
    # 중성 회색 범위에서는 내부의 검은 숫자가 함께 있을 때만 후보로 채택한다.
    neutral_rectangles = _collect_outline_contours(
        neutral_outline,
        scale,
        config.LABEL_NEUTRAL_OUTLINE_MIN_EXTENT,
    )
    for x0, y0, x1, y1 in neutral_rectangles:
        inset = max(3, min(x1 - x0, y1 - y0) // 8)
        interior_text = int(
            np.count_nonzero(
                dark_outline[
                    y0 + inset : y1 - inset,
                    x0 + inset : x1 - inset,
                ]
            )
        )
        if interior_text >= config.LABEL_NEUTRAL_MIN_TEXT_PIXELS:
            rectangles.append((x0, y0, x1, y1))

    # 테두리가 영상 경계에서 잘리거나 몇 픽셀 끊긴 흰 라벨은 폐곡선 contour가
    # 없다. 이 경우 밝은 내부를 찾되 검은 숫자와 실제 얇은 리더까지 모두 확인한다.
    channel_min = bgr.min(axis=2)
    channel_max = bgr.max(axis=2)
    light_fill = np.where(
        (channel_min >= config.LABEL_LIGHT_FILL_MIN)
        & (
            channel_max.astype(np.int16) - channel_min.astype(np.int16)
            <= config.LABEL_LIGHT_FILL_MAX_CHANNEL_DELTA
        ),
        255,
        0,
    ).astype(np.uint8)
    light_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            config.LABEL_LIGHT_FILL_OPEN_KERNEL,
            config.LABEL_LIGHT_FILL_OPEN_KERNEL,
        ),
    )
    light_fill = cv2.morphologyEx(light_fill, cv2.MORPH_OPEN, light_kernel)
    count, _, light_stats, _ = cv2.connectedComponentsWithStats(
        light_fill, connectivity=8
    )
    light_rectangles: list[Rect] = []
    padding = max(2, int(round(config.LABEL_LIGHT_FILL_PADDING * scale)))
    for x, y, component_width, component_height, area in light_stats[1:count]:
        if not (
            config.LABEL_LIGHT_FILL_MIN_WIDTH
            <= component_width
            <= round(config.LABEL_LIGHT_FILL_MAX_WIDTH * scale)
            and config.LABEL_LIGHT_FILL_MIN_HEIGHT
            <= component_height
            <= round(config.LABEL_LIGHT_FILL_MAX_HEIGHT * scale)
        ):
            continue
        if area / float(component_width * component_height) < config.LABEL_LIGHT_FILL_MIN_RATIO:
            continue
        text_count = int(
            np.count_nonzero(
                dark_outline[
                    y : y + component_height,
                    x : x + component_width,
                ]
            )
        )
        if text_count < config.LABEL_LIGHT_FILL_MIN_TEXT_PIXELS:
            continue
        light_rectangles.append(
            (
                max(0, int(x) - padding),
                max(0, int(y) - padding),
                min(width, int(x + component_width) + padding),
                min(height, int(y + component_height) + padding),
            )
        )
    if light_rectangles:
        light_boxes = [
            (x0, y0, x1 - x0, y1 - y0)
            for x0, y0, x1, y1 in light_rectangles
        ]
        connected = _connected_line_components(bgr, light_boxes)
        rectangles.extend(
            rectangle
            for index, rectangle in enumerate(light_rectangles)
            if index in connected
        )
    exact_blue = (
        (bgr[:, :, 0] == 255)
        & (bgr[:, :, 1] == 0)
        & (bgr[:, :, 2] == 0)
    )
    relaxed_blue = (
        (bgr[:, :, 0] >= config.LEADER_BLUE_MIN)
        & (bgr[:, :, 1] <= config.LEADER_OTHER_MAX)
        & (bgr[:, :, 2] <= config.LEADER_OTHER_MAX)
    )
    # 검은 외곽선은 기존 export를 위한 fallback이다. 스캔 본체가 있는 이미지에서는
    # 내부 숫자나 인접 리더가 확인되어야 일반적인 검은 사각 형상을 라벨로 채택한다.
    dark_rectangles = _collect_outline_contours(
        dark_outline,
        scale,
        max(0.75, config.LABEL_OUTLINE_MIN_EXTENT),
    )
    if dark_rectangles:
        if scan_mask is None:
            scan_mask = build_scan_mask(bgr)
        scan_present = bool(np.any(scan_mask))
        for x0, y0, x1, y1 in dark_rectangles:
            inset = max(3, min(x1 - x0, y1 - y0) // 8)
            interior = dark_outline[
                y0 + inset : y1 - inset,
                x0 + inset : x1 - inset,
            ]
            interior_text = int(np.count_nonzero(interior))
            margin = max(
                config.LABEL_NORMALIZE_MARGIN,
                config.LEADER_MAX_BOX_GAP,
            )
            nx0, ny0 = max(0, x0 - margin), max(0, y0 - margin)
            nx1, ny1 = min(width, x1 + margin), min(height, y1 + margin)
            leader_nearby = bool(np.any(relaxed_blue[ny0:ny1, nx0:nx1]))
            if (
                not scan_present
                or interior_text >= config.LABEL_TEXT_MIN_PIXELS
                or leader_nearby
            ):
                rectangles.append((x0, y0, x1, y1))

    dark_text = bgr.max(axis=2) <= config.LABEL_DARK_MAX_VALUE
    normalized: list[Rect] = []
    for x0, y0, x1, y1 in rectangles:
        box_width = x1 - x0
        box_height = y1 - y0
        text_count = int(np.count_nonzero(dark_text[y0:y1, x0:x1]))
        red_count = int(np.count_nonzero(red_fill[y0:y1, x0:x1]))

        margin = config.LABEL_NORMALIZE_MARGIN
        nx0, ny0 = max(0, x0 - margin), max(0, y0 - margin)
        nx1, ny1 = min(width, x1 + margin), min(height, y1 + margin)
        leader_nearby = bool(np.any(exact_blue[ny0:ny1, nx0:nx1]))
        inner_outline = (
            box_width <= round(config.LABEL_INNER_BOX_MAX_WIDTH * scale)
            and box_height <= round(config.LABEL_INNER_BOX_MAX_HEIGHT * scale)
            and (
                text_count >= config.LABEL_TEXT_MIN_PIXELS
                or red_count >= config.LABEL_MIN_COMPONENT_AREA
            )
            and leader_nearby
        )
        if inner_outline:
            x0, y0 = max(0, x0 - 1), max(0, y0 - 1)
            x1, y1 = min(width, x1 + 1), min(height, y1 + 1)
        normalized.append((x0, y0, x1, y1))

    return sorted(_deduplicate_rectangles(normalized), key=lambda r: (r[1], r[0]))


def _find_label_boxes(
    bgr: np.ndarray,
    scan_mask: np.ndarray | None = None,
) -> list[Box]:
    """검출된 xyxy 사각형을 기존 공개 형식인 xywh로 변환한다."""
    return [
        (x0, y0, x1 - x0, y1 - y0)
        for x0, y0, x1, y1 in _find_label_rectangles(bgr, scan_mask=scan_mask)
    ]


def _build_label_mask(bgr: np.ndarray) -> np.ndarray:
    """검출된 라벨 박스만 채운 하위 호환용 마스크를 반환한다."""
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for x, y, width, height in _find_label_boxes(bgr):
        cv2.rectangle(mask, (x, y), (x + width - 1, y + height - 1), 255, -1)
    return mask


def _classify_fill_color(bgr: np.ndarray, box: Box) -> str:
    """단일 패치 대신 박스 내부의 빨간 픽셀 비율로 red/white를 분류한다."""
    x, y, width, height = box
    inset = max(1, min(width, height) // 10)
    roi = bgr[
        y + inset : y + height - inset,
        x + inset : x + width - inset,
    ]
    if roi.size == 0:
        return "white"
    red_pixels = (
        (roi[:, :, 2] >= config.LABEL_RED_MIN_RED)
        & (roi[:, :, 1] <= config.LABEL_RED_MAX_GREEN)
        & (roi[:, :, 0] <= config.LABEL_RED_MAX_BLUE)
    )
    return "red" if np.count_nonzero(red_pixels) / red_pixels.size >= 0.20 else "white"


def _blue_dominant_mask(bgr: np.ndarray) -> np.ndarray:
    """파란 중심선과 혼합된 안티앨리어싱 가장자리 후보를 반환한다."""
    blue = bgr[:, :, 0].astype(np.int16)
    green = bgr[:, :, 1].astype(np.int16)
    red = bgr[:, :, 2].astype(np.int16)
    blue_dominant = (
        (blue >= 80)
        & (blue >= green + 4)
        & (blue >= red + 4)
    )
    return np.where(blue_dominant, 255, 0).astype(np.uint8)


def _build_thin_blue_masks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """파란 히트맵 면은 버리고 1~2px 리더 중심선만 우선 남긴다."""
    blue, green, red = cv2.split(bgr)
    center = (
        (blue >= config.LEADER_BLUE_MIN)
        & (green <= config.LEADER_OTHER_MAX)
        & (red <= config.LEADER_OTHER_MAX)
    ).astype(np.uint8)
    window = config.LEADER_LOCAL_WINDOW
    local_count = cv2.boxFilter(
        center,
        ddepth=cv2.CV_16U,
        ksize=(window, window),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    strict = np.where(
        (center > 0) & (local_count <= config.LEADER_STRICT_LOCAL_PIXELS),
        255,
        0,
    ).astype(np.uint8)
    relaxed = np.where(
        (center > 0) & (local_count <= config.LEADER_RELAXED_LOCAL_PIXELS),
        255,
        0,
    ).astype(np.uint8)
    # 완화 임계값에서는 넓은 파란 면의 테두리가 남을 수 있다. 5x5 erosion에도
    # 살아남는 면을 다시 확장해 빼면 2px 선은 보존하면서 연결된 히트맵 면은 끊긴다.
    wide_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.LEADER_WIDE_REGION_KERNEL, config.LEADER_WIDE_REGION_KERNEL),
    )
    center_mask = center * 255
    wide_region = cv2.erode(center_mask, wide_kernel, iterations=1)
    wide_neighborhood = cv2.dilate(wide_region, wide_kernel, iterations=1)
    relaxed[wide_neighborhood > 0] = 0
    return strict, relaxed


def _distance_to_box_squared(xs: np.ndarray, ys: np.ndarray, box: Box) -> np.ndarray:
    """각 픽셀과 사각형 사이의 제곱거리를 벡터로 계산한다."""
    x, y, width, height = box
    dx = np.maximum(np.maximum(x - xs, 0), xs - (x + width - 1))
    dy = np.maximum(np.maximum(y - ys, 0), ys - (y + height - 1))
    return dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2


def _assign_line_components(
    line_mask: np.ndarray,
    boxes: list[Box],
    allowed_boxes: set[int],
    blocked_mask: np.ndarray | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """라벨 근처의 얇은 성분을 가장 가까운 단일 라벨에 할당한다."""
    height, width = line_mask.shape
    reach = max(
        config.LEADER_BOX_REACH_MIN,
        int(round(min(height, width) * config.LEADER_BOX_REACH_RATIO)),
    )
    maximum_area = max(
        config.LEADER_MIN_COMPONENT_AREA,
        int(height * width * config.LEADER_MAX_AREA_RATIO),
        4 * max(height, width),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_mask, connectivity=8
    )
    assigned: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if not (config.LEADER_MIN_COMPONENT_AREA <= area <= maximum_area):
            continue
        if max(component_width, component_height) < config.LEADER_MIN_COMPONENT_SPAN:
            continue

        ys, xs = np.where(labels == component)
        if blocked_mask is not None and np.any(blocked_mask[ys, xs] > 0):
            continue
        nearby: list[tuple[float, int]] = []
        for index in allowed_boxes:
            x, y, box_width, box_height = boxes[index]
            touches = np.any(
                (xs >= x - reach)
                & (xs < x + box_width + reach)
                & (ys >= y - reach)
                & (ys < y + box_height + reach)
            )
            if not touches:
                continue
            minimum_distance = float(
                np.min(_distance_to_box_squared(xs, ys, boxes[index]))
            )
            if minimum_distance > config.LEADER_MAX_BOX_GAP**2:
                continue
            nearby.append((minimum_distance, index))
        if not nearby:
            continue

        _, associated_index = min(nearby)
        distance = _distance_to_box_squared(xs, ys, boxes[associated_index])
        score = float(np.max(distance))
        previous = assigned.get(associated_index)
        if previous is None or score > previous[2]:
            assigned[associated_index] = (ys, xs, score)

    return {index: (item[0], item[1]) for index, item in assigned.items()}


def _endpoint_from_component(
    xs: np.ndarray,
    ys: np.ndarray,
    box: Box,
    scan_mask: np.ndarray,
) -> tuple[int, int]:
    """라벨에서 가장 먼 스캔 내부 선 끝을 측정점 중심으로 변환한다."""
    inside_scan = scan_mask[ys, xs] > 0
    if np.any(inside_scan):
        candidate_xs = xs[inside_scan]
        candidate_ys = ys[inside_scan]
    else:
        candidate_xs, candidate_ys = xs, ys

    distance = _distance_to_box_squared(candidate_xs, candidate_ys, box)
    endpoint_index = int(np.argmax(distance))
    point_x = int(candidate_xs[endpoint_index])
    point_y = int(candidate_ys[endpoint_index])

    # 실제 export에서는 파란 중심선이 비파란 점 마커의 가까운 가장자리에서 끝난다.
    # 스캔 마스크가 확인되는 경우에만 진행 방향으로 이동해 빈 배경으로의 과보정을 막는다.
    if scan_mask[point_y, point_x] > 0:
        radius = max(
            config.MEASUREMENT_POINT_RADIUS_MIN,
            int(round(min(scan_mask.shape) * config.MEASUREMENT_POINT_RADIUS_RATIO)),
        )
        endpoint_distance = (candidate_xs - point_x) ** 2 + (
            candidate_ys - point_y
        ) ** 2
        nearby = (endpoint_distance > 0) & (endpoint_distance <= (radius * 3) ** 2)
        if np.any(nearby):
            direction_x = point_x - float(np.mean(candidate_xs[nearby]))
            direction_y = point_y - float(np.mean(candidate_ys[nearby]))
            length = float(np.hypot(direction_x, direction_y))
            if length > 0:
                shifted_x = int(round(point_x + radius * direction_x / length))
                shifted_y = int(round(point_y + radius * direction_y / length))
                shifted_x = int(np.clip(shifted_x, 0, scan_mask.shape[1] - 1))
                shifted_y = int(np.clip(shifted_y, 0, scan_mask.shape[0] - 1))
                if scan_mask[shifted_y, shifted_x] > 0:
                    point_x, point_y = shifted_x, shifted_y
    return point_x, point_y


def _trace_connected_leaders(
    bgr: np.ndarray,
    boxes: list[Box],
    scan_mask: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """엄격 마스크를 우선 사용하고 미검출 라벨만 완화 마스크로 재시도한다."""
    components = _connected_line_components(bgr, boxes)
    endpoints: dict[int, tuple[int, int]] = {}
    for index, (ys, xs) in components.items():
        endpoints[index] = _endpoint_from_component(xs, ys, boxes[index], scan_mask)
    return endpoints


def _connected_line_components(
    bgr: np.ndarray,
    boxes: list[Box],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """라벨별로 선택된 엄격/완화 파란 연결 성분을 반환한다."""
    strict_mask, relaxed_mask = _build_thin_blue_masks(bgr)
    unresolved = set(range(len(boxes)))
    components = _assign_line_components(strict_mask, boxes, unresolved)
    unresolved.difference_update(components)
    if unresolved:
        blocked_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for ys, xs in components.values():
            blocked_mask[ys, xs] = 255
        components.update(
            _assign_line_components(
                relaxed_mask,
                boxes,
                unresolved,
                blocked_mask=blocked_mask,
            )
        )
    return components


def build_blue_annotation_mask(
    bgr: np.ndarray,
    boxes: list[Box] | None = None,
    scan_mask: np.ndarray | None = None,
) -> np.ndarray:
    """라벨에 연결된 리더와 종점만 포함하고 파란 히트맵 면은 제외한다."""
    if scan_mask is None:
        scan_mask = build_scan_mask(bgr)
    if boxes is None:
        boxes = _find_label_boxes(bgr, scan_mask=scan_mask)
    if not boxes:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)

    components = _connected_line_components(bgr, boxes)
    selected_center = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for ys, xs in components.values():
        selected_center[ys, xs] = 255
    center_neighborhood = cv2.dilate(
        selected_center,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    annotation_mask = cv2.bitwise_and(
        center_neighborhood, _blue_dominant_mask(bgr)
    )

    point_radius = max(
        config.MEASUREMENT_POINT_RADIUS_MIN,
        int(round(min(bgr.shape[:2]) * config.MEASUREMENT_POINT_RADIUS_RATIO)),
    )
    for index, (ys, xs) in components.items():
        point = _endpoint_from_component(xs, ys, boxes[index], scan_mask)
        cv2.circle(annotation_mask, point, point_radius + 1, 255, thickness=-1)
    return annotation_mask


def _box_anchor_points(box: Box) -> list[tuple[int, int]]:
    """하위 호환 Hough 추적에서 사용할 박스 모서리와 변 중앙 8곳."""
    x, y, width, height = box
    return [
        (x, y),
        (x + width, y),
        (x, y + height),
        (x + width, y + height),
        (x + width // 2, y),
        (x + width // 2, y + height),
        (x, y + height // 2),
        (x + width, y + height // 2),
    ]


def _trace_leader_line(line_mask: np.ndarray, box: Box) -> tuple[int, int] | None:
    """기존 호출자를 위해 가장 긴 유효 Hough 선분 끝점을 반환한다."""
    lines = cv2.HoughLinesP(
        line_mask,
        1,
        np.pi / 180,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP,
    )
    if lines is None:
        return None

    anchors = _box_anchor_points(box)
    best_point: tuple[int, int] | None = None
    best_distance = -1.0
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        for anchor_x, anchor_y in anchors:
            if math.hypot(x1 - anchor_x, y1 - anchor_y) <= config.LEADER_ANCHOR_RADIUS:
                distance = math.hypot(x2 - x1, y2 - y1)
                if distance <= config.MAX_LEADER_LINE_LEN and distance > best_distance:
                    best_point, best_distance = (int(x2), int(y2)), distance
            if math.hypot(x2 - anchor_x, y2 - anchor_y) <= config.LEADER_ANCHOR_RADIUS:
                distance = math.hypot(x1 - x2, y1 - y2)
                if distance <= config.MAX_LEADER_LINE_LEN and distance > best_distance:
                    best_point, best_distance = (int(x1), int(y1)), distance
    return best_point


def detect_labels(bgr: np.ndarray) -> list[LabelCandidate]:
    """라벨별 박스, 연결 성분 기반 측정점, 채움색, 추적 여부를 반환한다."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("BGR 컬러 이미지가 필요합니다.")

    scan_mask = build_scan_mask(bgr)
    boxes = _find_label_boxes(bgr, scan_mask=scan_mask)
    if not boxes:
        return []
    endpoints = _trace_connected_leaders(bgr, boxes, scan_mask)

    return [
        LabelCandidate(
            box=box,
            point_xy=endpoints.get(index),
            label_color=_classify_fill_color(bgr, box),
            traced=index in endpoints,
        )
        for index, box in enumerate(boxes)
    ]
