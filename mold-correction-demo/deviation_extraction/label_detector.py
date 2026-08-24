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


@dataclass(eq=False)
class _LineComponent:
    """여러 라벨이 공유할 수 있는 하나의 연결 리더 성분."""

    ys: np.ndarray
    xs: np.ndarray


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
    """동일 contour 중복과 인접 라벨을 합친 composite 박스를 제거한다."""
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

    # 중성 테두리 closing은 2px 정도로 붙은 두 라벨을 하나의 큰 contour로
    # 합칠 수 있다. 실제 작은 박스 두 개가 서로 겹치지 않은 채 큰 박스 안에
    # 들어 있을 때만 큰 박스를 버려, 단일 라벨의 내·외곽선은 그대로 유지한다.
    atomic: list[Rect] = []
    for candidate in unique:
        x0, y0, x1, y1 = candidate
        candidate_area = (x1 - x0) * (y1 - y0)
        children: list[Rect] = []
        for other in unique:
            if other == candidate:
                continue
            ox0, oy0, ox1, oy1 = other
            other_area = (ox1 - ox0) * (oy1 - oy0)
            center_x = (ox0 + ox1) / 2.0
            center_y = (oy0 + oy1) / 2.0
            if (
                other_area
                <= candidate_area * config.LABEL_COMPOSITE_CHILD_MAX_AREA_RATIO
                and x0 <= center_x <= x1
                and y0 <= center_y <= y1
            ):
                children.append(other)
        merged_pair = any(
            _rect_iou(first, second) <= config.LABEL_COMPOSITE_CHILD_MAX_IOU
            for child_index, first in enumerate(children)
            for second in children[child_index + 1 :]
        )
        if not merged_pair:
            atomic.append(candidate)
    return atomic


def _find_label_rectangles(
    bgr: np.ndarray,
    scan_mask: np.ndarray | None = None,
) -> list[Rect]:
    """색상·형상·밀도를 결합해 중복 없는 xyxy 라벨 후보를 찾는다."""
    height, width = bgr.shape[:2]
    scale = _resolution_scale(bgr.shape)
    red_fill, gray_outline, neutral_outline, dark_outline = _color_masks(bgr)
    outline_kernel = np.ones((3, 3), dtype=np.uint8)
    neutral_outline_kernel = np.ones(
        (
            config.LABEL_NEUTRAL_CLOSE_KERNEL,
            config.LABEL_NEUTRAL_CLOSE_KERNEL,
        ),
        dtype=np.uint8,
    )
    gray_outline = cv2.morphologyEx(
        gray_outline, cv2.MORPH_CLOSE, outline_kernel
    )
    neutral_outline = cv2.morphologyEx(
        neutral_outline, cv2.MORPH_CLOSE, neutral_outline_kernel
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
        if scan_mask is None:
            scan_mask = build_scan_mask(bgr)
        connected = _connected_line_components(bgr, light_boxes, scan_mask)
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


def _scan_snap_distance(shape: tuple[int, int]) -> int:
    """입력 해상도에 맞춘 최대 스캔 경계 스냅 거리를 반환한다."""
    return int(
        np.clip(
            round(min(shape) * config.LEADER_SCAN_SNAP_RATIO),
            config.LEADER_SCAN_SNAP_MIN,
            config.LEADER_SCAN_SNAP_MAX,
        )
    )


def _distance_to_scan(scan_mask: np.ndarray) -> np.ndarray:
    """각 배경 픽셀에서 가장 가까운 실제 스캔 픽셀까지의 거리다."""
    return cv2.distanceTransform(
        np.where(scan_mask > 0, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )


def _scan_boundary_mask(scan_mask: np.ndarray) -> np.ndarray:
    """최초 진입점과 반대편 재이탈을 판별할 스캔 내부 경계 띠다."""
    if not np.any(scan_mask):
        return np.zeros(scan_mask.shape, dtype=bool)
    inset = max(
        config.LEADER_SCAN_BOUNDARY_INSET_MIN,
        int(round(min(scan_mask.shape) * config.MEASUREMENT_POINT_RADIUS_RATIO)),
    )
    kernel_size = inset * 2 + 1
    eroded_scan = cv2.erode(
        scan_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    return (scan_mask > 0) & (eroded_scan == 0)


def _assign_line_components(
    line_mask: np.ndarray,
    boxes: list[Box],
    allowed_boxes: set[int],
    scan_mask: np.ndarray,
    distance_to_scan: np.ndarray | None = None,
    blocked_mask: np.ndarray | None = None,
    maximum_box_gap: int | None = None,
    inside_scan_maximum_box_gap: int | None = None,
) -> dict[int, _LineComponent]:
    """스캔에 닿는 얇은 성분을 인접한 각 라벨에 품질순으로 할당한다."""
    height, width = line_mask.shape
    scan_available = bool(np.any(scan_mask))
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
    if scan_available and distance_to_scan is None:
        distance_to_scan = _distance_to_scan(scan_mask)
    maximum_scan_gap = _scan_snap_distance(scan_mask.shape)
    regular_box_gap = (
        config.LEADER_MAX_BOX_GAP
        if maximum_box_gap is None
        else maximum_box_gap
    )
    assigned: dict[int, tuple[_LineComponent, tuple[int, float, float]]] = {}

    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if not (config.LEADER_MIN_COMPONENT_AREA <= area <= maximum_area):
            continue
        component_span = max(component_width, component_height)
        if component_span < config.LEADER_DIRECT_COMPONENT_MIN_SPAN:
            continue

        ys, xs = np.where(labels == component)
        line_component = _LineComponent(ys=ys, xs=xs)
        if blocked_mask is not None:
            blocked_ratio = float(np.count_nonzero(blocked_mask[ys, xs])) / area
            if blocked_ratio >= config.LEADER_BLOCKED_OVERLAP_RATIO:
                continue
        if distance_to_scan is None:
            minimum_scan_gap = 0.0
            direct_scan_contact = 0
        else:
            minimum_scan_gap = float(np.min(distance_to_scan[ys, xs]))
            if minimum_scan_gap > maximum_scan_gap:
                continue
            direct_scan_contact = int(np.any(scan_mask[ys, xs] > 0))
        direct_short_component = component_span < config.LEADER_MIN_COMPONENT_SPAN
        if direct_short_component:
            if not direct_scan_contact:
                continue
            if (
                min(component_width, component_height)
                < config.LEADER_DIRECT_COMPONENT_MIN_THICKNESS
            ):
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
            allowed_box_gap = (
                config.LEADER_DIRECT_MAX_BOX_GAP
                if direct_short_component
                else regular_box_gap
            )
            if (
                inside_scan_maximum_box_gap is not None
                and not np.any(scan_mask[ys, xs] == 0)
            ):
                allowed_box_gap = min(
                    allowed_box_gap, inside_scan_maximum_box_gap
                )
            if minimum_distance > allowed_box_gap**2:
                continue
            nearby.append((minimum_distance, index))
        if not nearby:
            continue

        # 교차하거나 맞닿은 리더는 하나의 연결 성분이 될 수 있다. 가장 가까운
        # 라벨 하나에만 귀속시키면 나머지 라벨의 포인트가 사라진다.
        for _, associated_index in nearby:
            distance = _distance_to_box_squared(xs, ys, boxes[associated_index])
            span = float(np.max(distance))
            score = (direct_scan_contact, -minimum_scan_gap, span)
            previous = assigned.get(associated_index)
            if previous is None or score > previous[1]:
                assigned[associated_index] = (line_component, score)

    return {index: item[0] for index, item in assigned.items()}


def _top_marker_endpoint(
    bgr: np.ndarray,
    box: Box,
    scan_mask: np.ndarray,
) -> tuple[int, int] | None:
    """흰 라벨 상단 중앙의 짧고 어두운 점 마커를 엄격하게 복구한다."""
    if _classify_fill_color(bgr, box) != "white" or not np.any(scan_mask):
        return None

    x, y, width, _ = box
    scale = _resolution_scale(bgr.shape)
    depth = max(3, int(round(config.LEADER_MARKER_SEARCH_DEPTH * scale)))
    y0 = max(0, y - depth)
    if y0 >= y or width < 3:
        return None

    # 실제 여섯 마커 라벨은 스캔 본체 위에 놓여 세 면이 모두 본체에 둘러싸여
    # 있다. 흰 배경 라벨 근처의 우연한 어두운 무늬를 마커로 받지 않는다.
    height, image_width = scan_mask.shape
    box_height = box[3]
    perimeter_depth = max(
        2, int(round(config.LEADER_MARKER_PERIMETER_DEPTH * scale))
    )
    perimeter_bands = (
        scan_mask[max(0, y - perimeter_depth) : y, x : x + width],
        scan_mask[y : y + box_height, max(0, x - perimeter_depth) : x],
        scan_mask[
            y : y + box_height,
            x + width : min(image_width, x + width + perimeter_depth),
        ],
        scan_mask[
            y + box_height : min(height, y + box_height + perimeter_depth),
            x : x + width,
        ],
    )
    perimeter_ratios = [
        float(np.count_nonzero(band)) / band.size if band.size else 0.0
        for band in perimeter_bands
    ]
    minimum_perimeter_ratio = config.LEADER_MARKER_MIN_PERIMETER_SCAN_RATIO
    if (
        any(ratio < minimum_perimeter_ratio for ratio in perimeter_ratios[:3])
        or sum(ratio >= minimum_perimeter_ratio for ratio in perimeter_ratios) < 3
    ):
        return None

    value = bgr[y0:y, x : x + width].max(axis=2).astype(np.int16)
    scan = scan_mask[y0:y, x : x + width] > 0
    baseline_width = max(
        2, int(round(width * config.LEADER_MARKER_BASELINE_RATIO))
    )
    center_width = max(3, int(round(width * config.LEADER_MARKER_CENTER_RATIO)))
    center_x0 = max(0, (width - center_width) // 2)
    center_x1 = min(width, center_x0 + center_width)
    if baseline_width * 2 >= width or center_x0 >= center_x1:
        return None

    side_values = np.concatenate(
        (value[:, :baseline_width], value[:, -baseline_width:]), axis=1
    )
    baseline = np.median(side_values, axis=1, keepdims=True)
    center_value = value[:, center_x0:center_x1]
    center_scan = scan[:, center_x0:center_x1]
    evidence = (
        (baseline - center_value >= config.LEADER_MARKER_MIN_CONTRAST)
        & center_scan
    )
    minimum_pixels = max(
        config.LEADER_MARKER_MIN_PIXELS,
        int(round(config.LEADER_MARKER_MIN_PIXELS * scale)),
    )
    required_rows = min(config.LEADER_MARKER_REQUIRED_ROWS, evidence.shape[0])
    minimum_row_pixels = max(
        config.LEADER_MARKER_MIN_ROW_PIXELS,
        int(round(config.LEADER_MARKER_MIN_ROW_PIXELS * scale)),
    )
    maximum_row_pixels = max(
        minimum_row_pixels,
        int(round(center_width * config.LEADER_MARKER_MAX_ROW_RATIO)),
    )
    count, marker_labels, marker_stats, _ = cv2.connectedComponentsWithStats(
        evidence.astype(np.uint8), connectivity=8
    )
    selected_component: int | None = None
    selected_area = -1
    for component in range(1, count):
        area = int(marker_stats[component, cv2.CC_STAT_AREA])
        if area < minimum_pixels or not np.any(marker_labels[-1] == component):
            continue
        component_evidence = marker_labels == component
        contact_evidence = component_evidence[-required_rows:]
        row_counts = np.count_nonzero(contact_evidence, axis=1)
        if np.any(row_counts < minimum_row_pixels) or np.any(
            row_counts > maximum_row_pixels
        ):
            continue
        row_centers = np.asarray(
            [np.mean(np.flatnonzero(row)) for row in contact_evidence],
            dtype=np.float64,
        )
        if np.ptp(row_centers) > config.LEADER_MARKER_MAX_CENTER_SHIFT * scale:
            continue
        if area > selected_area:
            selected_component = component
            selected_area = area
    if selected_component is None:
        return None

    selected_evidence = marker_labels == selected_component
    marker_ys, marker_xs = np.where(selected_evidence)
    marker_center_x = float(np.mean(marker_xs)) + center_x0
    if (
        abs(marker_center_x - (width - 1) / 2.0)
        > width * config.LEADER_MARKER_MAX_CENTER_ERROR_RATIO
    ):
        return None
    contrasts = (
        baseline - center_value
    )[marker_ys, marker_xs]
    maximum_contrast = np.max(contrasts)
    strongest = contrasts >= maximum_contrast - config.LEADER_MARKER_MIN_CONTRAST
    point_x = int(round(np.mean(marker_xs[strongest]))) + x + center_x0
    point_y = int(round(np.mean(marker_ys[strongest]))) + y0
    if scan_mask[point_y, point_x] == 0:
        return None
    return point_x, point_y


def _leader_anchor_and_direction(
    xs: np.ndarray,
    ys: np.ndarray,
    box: Box,
    shape: tuple[int, int],
) -> tuple[tuple[float, float], np.ndarray]:
    """라벨을 빠져나가는 리더의 시작점과 국소 진행 방향을 추정한다."""
    box_distance = _distance_to_box_squared(xs, ys, box)
    minimum_distance = float(np.min(box_distance))
    anchor_pixels = box_distance <= minimum_distance + 1.0
    anchor_x = float(np.mean(xs[anchor_pixels]))
    anchor_y = float(np.mean(ys[anchor_pixels]))

    sample_radius = max(
        config.LEADER_DIRECTION_SAMPLE_MIN,
        int(round(min(shape) * config.LEADER_DIRECTION_SAMPLE_RATIO)),
    )
    vectors = np.column_stack((xs - anchor_x, ys - anchor_y)).astype(np.float64)
    squared_radius = np.sum(vectors**2, axis=1)
    local = (
        (squared_radius > 1.0)
        & (squared_radius <= sample_radius**2)
        & (box_distance > minimum_distance + 1.0)
    )
    samples = vectors[local]
    if len(samples) < 2:
        samples = vectors[squared_radius > 1.0]
    if len(samples) == 0:
        return (anchor_x, anchor_y), np.array([1.0, 0.0], dtype=np.float64)

    covariance = samples.T @ samples
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    farthest = samples[int(np.argmax(np.sum(samples**2, axis=1)))]
    if float(np.dot(direction, farthest)) < 0:
        direction = -direction
    length = float(np.hypot(direction[0], direction[1]))
    if length == 0:
        direction = farthest
        length = float(np.hypot(direction[0], direction[1]))
    return (anchor_x, anchor_y), direction / max(length, 1e-9)


def _direction_alignment(
    xs: np.ndarray,
    ys: np.ndarray,
    anchor: tuple[float, float],
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """후보와 라벨 출발 방향의 코사인 유사도 및 거리를 반환한다."""
    vectors = np.column_stack((xs - anchor[0], ys - anchor[1])).astype(np.float64)
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    alignments = np.full(len(xs), -1.0, dtype=np.float64)
    valid = lengths > 0
    alignments[valid] = (vectors[valid] @ direction) / lengths[valid]
    return alignments, lengths


def _best_directional_point(
    xs: np.ndarray,
    ys: np.ndarray,
    anchor: tuple[float, float],
    direction: np.ndarray,
    *,
    prefer_near: bool,
) -> tuple[int, int, float]:
    """진행 방향을 우선하고 동률이면 가까운/먼 후보를 결정적으로 고른다."""
    alignments, lengths = _direction_alignment(xs, ys, anchor, direction)
    if prefer_near:
        # 픽셀 래스터화로 0.999/1.0처럼 미세하게 달라져도 같은 방향 후보로
        # 간주한 뒤 실제 최초 진입점(최단 거리)을 선택한다.
        aligned = (
            alignments
            >= float(np.max(alignments)) - config.LEADER_CONTACT_ALIGNMENT_MARGIN
        )
        indices = np.flatnonzero(aligned)
        index = int(indices[int(np.argmin(lengths[indices]))])
        return int(xs[index]), int(ys[index]), float(alignments[index])

    distance_term = -lengths if prefer_near else lengths
    order = np.lexsort((distance_term, alignments))
    index = int(order[-1])
    return int(xs[index]), int(ys[index]), float(alignments[index])


def _nearest_scan_pixel(
    point_x: int,
    point_y: int,
    scan_mask: np.ndarray,
    maximum_gap: int,
) -> tuple[int, int] | None:
    """짧게 끊긴 리더 끝에서 가장 가까운 실제 스캔 픽셀을 찾는다."""
    height, width = scan_mask.shape
    x0, x1 = max(0, point_x - maximum_gap), min(width, point_x + maximum_gap + 1)
    y0, y1 = max(0, point_y - maximum_gap), min(height, point_y + maximum_gap + 1)
    local_ys, local_xs = np.where(scan_mask[y0:y1, x0:x1] > 0)
    if len(local_xs) == 0:
        return None
    absolute_xs = local_xs + x0
    absolute_ys = local_ys + y0
    distance = (absolute_xs - point_x) ** 2 + (absolute_ys - point_y) ** 2
    index = int(np.argmin(distance))
    if distance[index] > maximum_gap**2:
        return None
    return int(absolute_xs[index]), int(absolute_ys[index])


def _move_point_inside_scan(
    point_x: int,
    point_y: int,
    anchor: tuple[float, float],
    scan_mask: np.ndarray,
) -> tuple[int, int]:
    """스캔 경계 접점을 리더 진행 방향으로 몇 픽셀 안쪽에 놓는다."""
    direction_x = point_x - anchor[0]
    direction_y = point_y - anchor[1]
    length = float(np.hypot(direction_x, direction_y))
    if length == 0:
        return point_x, point_y
    radius = max(
        config.MEASUREMENT_POINT_RADIUS_MIN,
        int(round(min(scan_mask.shape) * config.MEASUREMENT_POINT_RADIUS_RATIO)),
    )
    for step in range(radius, 0, -1):
        shifted_x = int(round(point_x + step * direction_x / length))
        shifted_y = int(round(point_y + step * direction_y / length))
        if (
            0 <= shifted_x < scan_mask.shape[1]
            and 0 <= shifted_y < scan_mask.shape[0]
            and scan_mask[shifted_y, shifted_x] > 0
        ):
            return shifted_x, shifted_y
    return point_x, point_y


def _endpoint_from_component(
    xs: np.ndarray,
    ys: np.ndarray,
    box: Box,
    scan_mask: np.ndarray,
    *,
    directional_only: bool = False,
    distance_to_scan: np.ndarray | None = None,
    boundary_scan: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """스캔 접점만 측정점으로 채택하고 짧은 안티앨리어싱 틈만 복구한다."""
    if not np.any(scan_mask):
        return None

    anchor, direction = _leader_anchor_and_direction(xs, ys, box, scan_mask.shape)
    inside_scan = scan_mask[ys, xs] > 0
    if np.any(inside_scan):
        candidate_xs = xs[inside_scan]
        candidate_ys = ys[inside_scan]
        if directional_only:
            vectors = np.column_stack(
                (candidate_xs - anchor[0], candidate_ys - anchor[1])
            )
            forward = vectors @ direction
            perpendicular = np.abs(
                vectors[:, 0] * direction[1]
                - vectors[:, 1] * direction[0]
            )
            corridor_width = max(3, int(round(min(scan_mask.shape) * 0.01)))
            in_corridor = (forward > 0) & (perpendicular <= corridor_width)
            if np.any(in_corridor):
                candidate_xs = candidate_xs[in_corridor]
                candidate_ys = candidate_ys[in_corridor]
        distance = _distance_to_box_squared(candidate_xs, candidate_ys, box)
        endpoint_index = int(np.argmax(distance))
        point_x = int(candidate_xs[endpoint_index])
        point_y = int(candidate_ys[endpoint_index])
        far_alignment = float(
            _direction_alignment(
                np.asarray([point_x]),
                np.asarray([point_y]),
                anchor,
                direction,
            )[0][0]
        )

        # 스캔 안의 가느다란 파란 형상이 리더와 연결된 경우에는 멀리 따라가지 않고
        # 라벨에서 나온 국소 방향과 가장 잘 맞는 최초 경계 접점을 사용한다.
        if boundary_scan is None:
            boundary_scan = _scan_boundary_mask(scan_mask)
        on_boundary = boundary_scan[candidate_ys, candidate_xs]
        if np.any(on_boundary):
            contact_x, contact_y, contact_alignment = _best_directional_point(
                candidate_xs[on_boundary],
                candidate_ys[on_boundary],
                anchor,
                direction,
                prefer_near=True,
            )
            contact_separation = math.hypot(
                point_x - contact_x, point_y - contact_y
            )
            minimum_exit_span = max(
                _scan_snap_distance(scan_mask.shape),
                config.LEADER_SCAN_BOUNDARY_INSET_MIN * 2,
            )
            reaches_other_boundary = (
                bool(boundary_scan[point_y, point_x])
                and contact_separation > minimum_exit_span
            )
            maximum_leader_length = (
                config.MAX_LEADER_LINE_LEN * _resolution_scale(scan_mask.shape)
            )
            leader_too_long = math.sqrt(
                float(_distance_to_box_squared(
                    np.asarray([point_x]), np.asarray([point_y]), box
                )[0])
            ) > maximum_leader_length
            if (
                reaches_other_boundary
                or leader_too_long
                or far_alignment
                < contact_alignment - config.LEADER_CONTACT_ALIGNMENT_MARGIN
            ):
                point_x, point_y = contact_x, contact_y
    else:
        maximum_gap = _scan_snap_distance(scan_mask.shape)
        if distance_to_scan is None:
            distance_to_scan = _distance_to_scan(scan_mask)
        near_scan = distance_to_scan[ys, xs] <= maximum_gap
        if not np.any(near_scan):
            return None
        line_x, line_y, _ = _best_directional_point(
            xs[near_scan],
            ys[near_scan],
            anchor,
            direction,
            prefer_near=False,
        )
        snapped = _nearest_scan_pixel(line_x, line_y, scan_mask, maximum_gap)
        if snapped is None:
            return None
        point_x, point_y = snapped

    if scan_mask[point_y, point_x] == 0:
        return None
    point_x, point_y = _move_point_inside_scan(
        point_x, point_y, anchor, scan_mask
    )
    if scan_mask[point_y, point_x] == 0:
        return None
    return point_x, point_y


def _shared_components(
    components: dict[int, _LineComponent],
) -> set[_LineComponent]:
    """교차 리더처럼 여러 라벨에 할당된 동일 연결 성분을 반환한다."""
    counts: dict[_LineComponent, int] = {}
    for component in components.values():
        counts[component] = counts.get(component, 0) + 1
    return {component for component, count in counts.items() if count > 1}


def _trace_connected_leaders(
    bgr: np.ndarray,
    boxes: list[Box],
    scan_mask: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """엄격 마스크를 우선 사용하고 미검출 라벨만 완화 마스크로 재시도한다."""
    distance_to_scan = _distance_to_scan(scan_mask) if np.any(scan_mask) else None
    boundary_scan = _scan_boundary_mask(scan_mask)
    components = _connected_line_components(
        bgr, boxes, scan_mask, distance_to_scan
    )
    shared_components = _shared_components(components)
    endpoints: dict[int, tuple[int, int]] = {}
    for index, component in components.items():
        endpoint = _endpoint_from_component(
            component.xs,
            component.ys,
            boxes[index],
            scan_mask,
            directional_only=component in shared_components,
            distance_to_scan=distance_to_scan,
            boundary_scan=boundary_scan,
        )
        if endpoint is not None:
            endpoints[index] = endpoint
    for index, box in enumerate(boxes):
        if index in endpoints:
            continue
        endpoint = _top_marker_endpoint(bgr, box, scan_mask)
        if endpoint is not None:
            endpoints[index] = endpoint
    return endpoints


def _connected_line_components(
    bgr: np.ndarray,
    boxes: list[Box],
    scan_mask: np.ndarray | None = None,
    distance_to_scan: np.ndarray | None = None,
) -> dict[int, _LineComponent]:
    """라벨별로 선택된 엄격/완화 파란 연결 성분을 반환한다."""
    if scan_mask is None:
        scan_mask = build_scan_mask(bgr)
    if distance_to_scan is None and np.any(scan_mask):
        distance_to_scan = _distance_to_scan(scan_mask)
    strict_mask, relaxed_mask = _build_thin_blue_masks(bgr)
    unresolved = set(range(len(boxes)))
    components = _assign_line_components(
        strict_mask, boxes, unresolved, scan_mask, distance_to_scan
    )
    unresolved.difference_update(components)
    if unresolved:
        blocked_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for component in components.values():
            blocked_mask[component.ys, component.xs] = 255
        components.update(
            _assign_line_components(
                relaxed_mask,
                boxes,
                unresolved,
                scan_mask,
                distance_to_scan,
                blocked_mask=blocked_mask,
                maximum_box_gap=config.LEADER_RELAXED_MAX_BOX_GAP,
                inside_scan_maximum_box_gap=(
                    config.LEADER_RELAXED_INSIDE_SCAN_MAX_BOX_GAP
                ),
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

    distance_to_scan = _distance_to_scan(scan_mask) if np.any(scan_mask) else None
    boundary_scan = _scan_boundary_mask(scan_mask)
    components = _connected_line_components(
        bgr, boxes, scan_mask, distance_to_scan
    )
    selected_center = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for component in components.values():
        selected_center[component.ys, component.xs] = 255
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
    shared_components = _shared_components(components)
    for index, component in components.items():
        point = _endpoint_from_component(
            component.xs,
            component.ys,
            boxes[index],
            scan_mask,
            directional_only=component in shared_components,
            distance_to_scan=distance_to_scan,
            boundary_scan=boundary_scan,
        )
        if point is not None:
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
