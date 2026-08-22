"""편차 맵에서 라벨 박스와 리더라인 끝점 후보를 검출한다."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class LabelCandidate:
    """라벨 박스와 그 박스가 가리키는 픽셀 좌표 후보."""

    box: tuple[int, int, int, int]
    point_xy: tuple[int, int] | None
    label_color: str
    # 전체 연결 성공이 아니라 박스 앵커에 닿은 Hough 선분 검출 여부다.
    traced: bool


def _build_label_mask(bgr: np.ndarray) -> np.ndarray:
    """파란 지시선을 제외한 무채색 테두리와 빨간 라벨 영역을 분리한다."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    neutral_dark = (
        (gray <= config.LABEL_BORDER_MAX_GRAY)
        & (hsv[:, :, 1] <= config.LABEL_BORDER_MAX_SATURATION)
    )
    red_hue = (
        (hsv[:, :, 0] <= config.LABEL_RED_HUE_MAX)
        | (hsv[:, :, 0] >= config.LABEL_RED_HUE_MIN)
    )
    red_fill = (
        red_hue
        & (hsv[:, :, 1] >= config.LABEL_RED_MIN_SATURATION)
        & (hsv[:, :, 2] >= config.LABEL_RED_MIN_VALUE)
    )
    mask = np.where(neutral_dark | red_fill, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _find_label_boxes(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """라벨 마스크의 윤곽을 면적·extent·종횡비로 걸러 박스를 찾는다."""
    label_mask = _build_label_mask(bgr)

    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (config.MIN_LABEL_AREA <= area <= config.MAX_LABEL_AREA):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        if rect_area == 0 or area / rect_area < config.MIN_LABEL_EXTENT:
            continue
        aspect = w / h if h else 0
        if not (config.LABEL_MIN_ASPECT <= aspect <= config.LABEL_MAX_ASPECT):
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def _classify_fill_color(bgr: np.ndarray, box: tuple[int, int, int, int]) -> str:
    """숫자 획을 피한 상단 내부 패치의 평균 BGR로 red/white를 분류한다."""
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 4
    patch = bgr[max(cy - 2, 0):cy + 2, max(cx - 2, 0):cx + 2]
    if patch.size == 0:
        return "white"
    b, g, r = patch.reshape(-1, 3).mean(axis=0)
    return "red" if (r > 150 and g < 130 and b < 130) else "white"


def _find_dots(bgr: np.ndarray) -> list[tuple[int, int]]:
    """파란색 마스크에서 설정 면적 안의 블롭 무게중심을 찾는다."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, config.DOT_HSV_LOWER, config.DOT_HSV_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots: list[tuple[int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (config.MIN_DOT_AREA <= area <= config.MAX_DOT_AREA):
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        dots.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return dots


def _box_anchor_points(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """리더 선분 접점을 검사할 박스 모서리와 변 중앙 8곳을 반환한다."""
    x, y, w, h = box
    return [
        (x, y),
        (x + w, y),
        (x, y + h),
        (x + w, y + h),
        (x + w // 2, y),
        (x + w // 2, y + h),
        (x, y + h // 2),
        (x + w, y + h // 2),
    ]


def _trace_leader_line(
    line_mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    """박스 앵커에 닿은 선분 중 가장 긴 유효 선분의 반대 끝점을 반환한다."""
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
    best_dist = -1.0
    # OpenCV 버전에 따른 (N, 1, 4)/(N, 4) 반환 차이를 정규화한다.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        for ax, ay in anchors:
            if math.hypot(x1 - ax, y1 - ay) <= config.LEADER_ANCHOR_RADIUS:
                d = math.hypot(x2 - x1, y2 - y1)
                if d <= config.MAX_LEADER_LINE_LEN and d > best_dist:
                    best_point, best_dist = (int(x2), int(y2)), d
            if math.hypot(x2 - ax, y2 - ay) <= config.LEADER_ANCHOR_RADIUS:
                d = math.hypot(x1 - x2, y1 - y2)
                if d <= config.MAX_LEADER_LINE_LEN and d > best_dist:
                    best_point, best_dist = (int(x1), int(y1)), d
    return best_point


def _snap_to_dot(point: tuple[int, int], dots: list[tuple[int, int]]) -> tuple[int, int]:
    """선분 끝점을 반경 안의 가장 가까운 파란 점 중심으로 보정한다."""
    if not dots:
        return point
    nearest = min(dots, key=lambda d: math.hypot(d[0] - point[0], d[1] - point[1]))
    if math.hypot(nearest[0] - point[0], nearest[1] - point[1]) <= config.DOT_SNAP_RADIUS:
        return nearest
    return point


def detect_labels(bgr: np.ndarray) -> list[LabelCandidate]:
    """라벨 박스별 지시 좌표, 배경색, 선분 검출 여부를 반환한다."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    line_mask = cv2.inRange(hsv, config.LEADER_LINE_HSV_LOWER, config.LEADER_LINE_HSV_UPPER)

    boxes = _find_label_boxes(bgr)
    dots = _find_dots(bgr)

    candidates: list[LabelCandidate] = []
    for box in boxes:
        endpoint = _trace_leader_line(line_mask, box)
        traced = endpoint is not None
        point_xy = _snap_to_dot(endpoint, dots) if endpoint is not None else None
        candidates.append(
            LabelCandidate(
                box=box,
                point_xy=point_xy,
                label_color=_classify_fill_color(bgr, box),
                traced=traced,
            )
        )
    return candidates
