"""사람이 지정하는 보정 경계에 가까운 단일 제로라인 실험 로직.

기존 ``zero_line_detection`` 코드는 변경하지 않는다. 기존 모듈에서는 검증된
컬러바 판독기만 가져오고, 후보 밴드와 단일 경로 생성은 이 파일에서 독립적으로
수행한다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import heapq
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from zero_line_detection.colorbar import detect_colorbar  # noqa: E402
from label_removal.remove_labels import (  # noqa: E402
    build_scan_mask,
    detect_exact_hsv_leader_lines,
    detect_label_boxes,
)


@dataclass
class AdvanceConfig:
    band_low: float = -0.5
    band_high: float = 0.5
    color_max_dist: float = 14.0
    smooth_ksize: int = 5
    morph_open: int = 1
    morph_close: int = 4
    min_band_area: int = 100
    min_part_area: int = 500
    part_area_ratio: float = 0.03
    anchor_ring_inner: int = 6
    anchor_ring_outer: int = 14
    anchor_snap_radius: float = 80.0
    simplify_epsilon: float = 8.0
    max_vertices: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Anchor:
    x: int
    y: int
    estimated_value: float | None
    source: str
    linearity: float | None = None
    evidence: str | None = None
    group: str | None = None
    contour_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdvanceResult:
    values: np.ndarray
    part_mask: np.ndarray
    candidate_mask: np.ndarray
    selected_skeleton: np.ndarray
    raw_path: list[tuple[int, int]]
    smooth_path: list[tuple[float, float]]
    point_only_path: list[tuple[float, float]]
    detected_points: list[Anchor]
    candidate_anchors: list[Anchor]
    zero_candidates: list[Anchor]
    zero_waypoints: list[Anchor]
    zero_points: list[Anchor]
    line_anchors: list[Anchor]
    snapped_anchors: list[tuple[int, int]]
    colorbar: dict
    warnings: list[str]
    config: AdvanceConfig


def read_image(path: Path) -> np.ndarray:
    """한글 Windows 경로에서도 안전하게 BGR 이미지를 읽는다."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """한글 Windows 경로에서도 안전하게 이미지를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"이미지를 인코딩할 수 없습니다: {path}")
    encoded.tofile(str(path))


def _clean_binary(mask: np.ndarray, open_radius: int, close_radius: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if open_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_radius * 2 + 1, open_radius * 2 + 1)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out.astype(bool)


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label] = True
    return keep


def _keep_main_parts(mask: np.ndarray, min_area: int, ratio: float) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = max(min_area, int(areas.max() * ratio))
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= threshold:
            keep[labels == label] = True
    return keep


def detect_measurement_points(
    bgr: np.ndarray,
) -> list[tuple[int, int, int, tuple[int, int, int]]]:
    """remove_labels와 같은 사각형-연결선-끝점 관계로 측정 포인트를 찾는다.

    단순히 작은 파란 덩어리를 찾지 않는다. 숫자 라벨 사각형을 먼저 검출하고,
    그 사각형에 실제로 닿은 정확한 HSV 파란 연결선만 추적한 뒤 사각형에서 가장
    먼 끝을 측정 포인트로 사용한다. 연결선이 다른 색의 포인트 가장자리에서
    끝나는 경우에는 진행 방향으로 포인트 중심까지 이동시킨 좌표가 반환된다.
    """
    scan_mask = build_scan_mask(bgr)
    label_boxes = detect_label_boxes(bgr)
    _, point_specs = detect_exact_hsv_leader_lines(bgr, label_boxes, scan_mask)
    return point_specs


def _zero_label_boxes(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find neutral numeric boxes whose rendered text is exactly ``0.0``.

    The exported labels use the same glyph bitmap for both zero characters.
    After removing the decimal dot, a true ``0.0`` therefore has two tall glyph
    components with almost identical normalized masks. This distinguishes it
    from ``0.1`` through ``0.5`` without OCR or a network model.
    """
    matches: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in detect_label_boxes(bgr):
        roi = bgr[y0 + 3 : y1 - 3, x0 + 3 : x1 - 3]
        if roi.size == 0:
            continue
        red_fill_ratio = float(
            ((roi[:, :, 2] > 200) & (roi[:, :, 1] < 100) & (roi[:, :, 0] < 100)).mean()
        )
        if red_fill_ratio >= 0.10:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark = ((gray < 110) & (hsv[:, :, 1] < 90)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        glyphs: list[tuple[int, int, int, int, int, int]] = []
        small_components: list[tuple[int, int, int, int, int]] = []
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            if height >= 10 and area >= 20:
                glyphs.append((x, y, width, height, area, component))
            elif area >= 3:
                small_components.append((x, y, width, height, area))
        glyphs.sort(key=lambda item: item[0])
        if len(glyphs) != 2:
            continue

        # Reject a leading minus sign; this prevents a hypothetical -0.0 label
        # from being treated as the zero anchor requested by the operator.
        first_x = glyphs[0][0]
        has_minus = any(
            x < first_x and width >= 4 and height <= 3
            for x, _y, width, height, _area in small_components
        )
        if has_minus:
            continue

        normalized = []
        for x, y, width, height, _area, component in glyphs:
            glyph = (labels[y : y + height, x : x + width] == component).astype(np.uint8)
            normalized.append(
                cv2.resize(glyph, (20, 28), interpolation=cv2.INTER_NEAREST).astype(bool)
            )
        intersection = int(np.logical_and(normalized[0], normalized[1]).sum())
        union = int(np.logical_or(normalized[0], normalized[1]).sum())
        similarity = intersection / union if union else 0.0
        if similarity >= 0.82:
            matches.append((x0, y0, x1, y1))
    return matches


def detect_numeric_zero_points(bgr: np.ndarray) -> list[Anchor]:
    """Return leader endpoints belonging to labels whose displayed value is 0.0."""
    point_specs = detect_measurement_points(bgr)
    available = list(range(len(point_specs)))
    zero_points: list[Anchor] = []
    for x0, y0, x1, y1 in _zero_label_boxes(bgr):
        if not available:
            break

        def rectangle_distance(index: int) -> float:
            x, y, _radius, _color = point_specs[index]
            dx = max(x0 - x, 0, x - (x1 - 1))
            dy = max(y0 - y, 0, y - (y1 - 1))
            return float(np.hypot(dx, dy))

        selected = min(available, key=rectangle_distance)
        if rectangle_distance(selected) > max(bgr.shape[:2]) * 0.25:
            continue
        available.remove(selected)
        x, y, _radius, _color = point_specs[selected]
        zero_points.append(Anchor(x, y, 0.0, "numeric_zero_label"))
    return zero_points


def _numeric_text_mask(
    bgr: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray | None:
    """Extract only the numeric glyphs from a gray or red measurement label."""
    x0, y0, x1, y1 = box
    roi = bgr[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    red_background = float(
        ((roi[:, :, 2] > 170) & (roi[:, :, 1] < 120) & (roi[:, :, 0] < 120)).mean()
    ) > 0.20
    if red_background:
        mask = (gray > 175) & (hsv[:, :, 1] < 125)
    else:
        mask = (gray < 145) & (hsv[:, :, 1] < 115)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    kept = np.zeros(mask.shape, dtype=np.uint8)
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if area < 2 or width > roi.shape[1] * 0.65 or height > roi.shape[0] * 0.95:
            continue
        kept[labels == component] = 1
    yy, xx = np.where(kept > 0)
    if not len(xx):
        return None
    return kept[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1].astype(bool)


@lru_cache(maxsize=1)
def _numeric_templates() -> dict[float, np.ndarray]:
    """Render the fixed one-decimal label vocabulary using its DejaVu font."""
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    templates: dict[float, np.ndarray] = {}
    for tenth in range(-35, 36):
        value = tenth / 10.0
        text = f"{value:.1f}"
        canvas = Image.new("L", (100, 40), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text((3, 1), text, font=font, fill=255)
        mask = np.asarray(canvas) > 100
        yy, xx = np.where(mask)
        templates[value] = mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    return templates


def _normalized_glyph_mask(mask: np.ndarray, height: int = 30) -> np.ndarray:
    source_height, source_width = mask.shape
    width = max(1, int(round(source_width * height / max(source_height, 1))))
    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    canvas = np.zeros((height + 8, 150), dtype=bool)
    x0 = max(0, (canvas.shape[1] - width) // 2)
    copy_width = min(width, canvas.shape[1])
    canvas[4 : 4 + height, x0 : x0 + copy_width] = resized[:, :copy_width]
    return canvas


def _glyph_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric chamfer distance tolerant of antialiasing and one-pixel shifts."""
    first = _normalized_glyph_mask(first)
    second = _normalized_glyph_mask(second)
    distance_to_first = cv2.distanceTransform(
        (~first).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_second = cv2.distanceTransform(
        (~second).astype(np.uint8), cv2.DIST_L2, 3
    )
    if not first.any() or not second.any():
        return float("inf")
    return float(distance_to_second[first].mean() + distance_to_first[second].mean())


def _read_label_value(
    bgr: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[float | None, float]:
    """Read a one-decimal measurement label by local font-template matching."""
    observed = _numeric_text_mask(bgr, box)
    if observed is None:
        return None, float("inf")
    ranked = sorted(
        ((_glyph_distance(observed, template), value) for value, template in _numeric_templates().items()),
        key=lambda item: item[0],
    )
    best_score, best_value = ranked[0]
    separation = ranked[1][0] - best_score if len(ranked) > 1 else 0.0
    # A low absolute distance and a clear lead over the next digit are both
    # required. Unreadable labels are excluded instead of inventing a value.
    if best_score > 4.2 or separation < 0.08:
        return None, best_score
    return float(best_value), best_score


def _trace_box_leader_endpoint(
    exact_blue: np.ndarray,
    box: tuple[int, int, int, int],
    max_length: int = 230,
) -> tuple[int, int] | None:
    """Trace a short straight exact-blue leader directly from its label box."""
    x0, y0, x1, y1 = box
    height, width = exact_blue.shape
    reach = 12
    rx0, rx1 = max(0, x0 - reach), min(width, x1 + reach)
    ry0, ry1 = max(0, y0 - reach), min(height, y1 + reach)
    yy, xx = np.where(exact_blue[ry0:ry1, rx0:rx1] > 0)
    xx = xx + rx0
    yy = yy + ry0
    outside = ~((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1))
    xx, yy = xx[outside], yy[outside]
    if not len(xx):
        return None

    center = np.asarray(((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0))
    all_y, all_x = np.where(exact_blue > 0)
    all_points = np.column_stack((all_x, all_y)).astype(np.float64)
    delta_all = all_points - center
    best: tuple[float, np.ndarray, float] | None = None
    seen_angles: set[int] = set()
    for px, py in zip(xx, yy):
        direction = np.asarray((px, py), dtype=np.float64) - center
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            continue
        direction /= length
        angle_key = int(round(np.degrees(np.arctan2(direction[1], direction[0])) / 2.0))
        if angle_key in seen_angles:
            continue
        seen_angles.add(angle_key)
        projection = delta_all @ direction
        perpendicular = np.abs(delta_all[:, 0] * direction[1] - delta_all[:, 1] * direction[0])
        aligned = np.sort(projection[(projection > 4.0) & (projection <= max_length) & (perpendicular <= 1.65)])
        if len(aligned) < 5:
            continue
        # Keep the continuous run beginning at the box edge. A gap prevents
        # the ray from jumping to a separate collinear annotation or scan edge.
        start_position = float(aligned[0])
        end_position = start_position
        for position in aligned[1:]:
            if float(position) - end_position > 4.2:
                break
            end_position = float(position)
        span = end_position - start_position
        if span < 7.0:
            continue
        score = span + min(len(aligned), 60) * 0.08
        if best is None or score > best[0]:
            best = (score, direction.copy(), end_position)
    if best is None:
        return None
    _score, direction, end_position = best
    endpoint = center + direction * (end_position + 4.0)
    return (
        int(np.clip(round(endpoint[0]), 0, width - 1)),
        int(np.clip(round(endpoint[1]), 0, height - 1)),
    )


def _project_box_to_scan_contour(
    box: tuple[int, int, int, int], contours: list[np.ndarray], max_distance: float = 90.0
) -> tuple[int, int] | None:
    """Project a line-less inner numeric box onto the nearest scan contour."""
    x0, y0, x1, y1 = box
    center = np.asarray(((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0), dtype=np.float32)
    best: tuple[float, tuple[int, int]] | None = None
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float32)
        distances2 = np.square(points - center).sum(axis=1)
        index = int(np.argmin(distances2))
        distance = float(np.sqrt(distances2[index]))
        point = tuple(map(int, points[index]))
        if best is None or distance < best[0]:
            best = (distance, point)
    if best is None or best[0] > max_distance:
        return None
    return best[1]


def _measurement_records(
    bgr: np.ndarray,
) -> list[dict]:
    """Associate each detected leader endpoint with its own numeric label box."""
    boxes = detect_label_boxes(bgr)
    point_specs = detect_measurement_points(bgr)
    if not boxes or not point_specs:
        return []

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
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
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if area < 5 or max(width, height) < 6:
            continue
        yy, xx = np.where(labels == component)
        components.append((xx, yy))

    records: list[dict] = []
    used_components: set[int] = set()
    used_boxes: set[tuple[int, int, int, int]] = set()
    for x, y, radius, _color in point_specs:
        component_choices: list[tuple[float, int]] = []
        for index, (xx, yy) in enumerate(components):
            if index in used_components:
                continue
            distance2 = (xx - x) ** 2 + (yy - y) ** 2
            component_choices.append((float(distance2.min()), index))
        if not component_choices:
            continue
        distance2, component_index = min(component_choices)
        if distance2 > float((radius + 10) ** 2):
            continue
        used_components.add(component_index)
        xx, yy = components[component_index]

        def box_distance(box: tuple[int, int, int, int]) -> float:
            bx0, by0, bx1, by1 = box
            dx = np.maximum(np.maximum(bx0 - xx, 0), xx - (bx1 - 1))
            dy = np.maximum(np.maximum(by0 - yy, 0), yy - (by1 - 1))
            return float(np.min(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2))

        box = min(boxes, key=box_distance)
        used_boxes.add(box)
        value, ocr_score = _read_label_value(bgr, box)
        bx0, by0, bx1, by1 = box
        label_center = ((bx0 + bx1 - 1) / 2.0, (by0 + by1 - 1) / 2.0)
        direction = np.asarray((label_center[0] - x, label_center[1] - y), dtype=np.float64)
        length = float(np.hypot(direction[0], direction[1]))
        if length > 0:
            direction /= length
        if abs(direction[1]) >= abs(direction[0]) * 0.70:
            group = "bottom" if direction[1] >= 0 else "top"
        else:
            group = "right" if direction[0] >= 0 else "left"
        records.append(
            {
                "point": (int(x), int(y)),
                "box": box,
                "value": value,
                "ocr_score": ocr_score,
                "leader_direction": direction,
                "label_center": np.asarray(label_center, dtype=np.float64),
                "group": group,
            }
        )

    # Strict thinning can reject very short leaders beside inner rounded
    # openings. Trace the exact-blue ray directly from every unused number box.
    scan_mask = build_scan_mask(bgr)
    scan_contours, _ = cv2.findContours(
        scan_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    scan_contours = [
        contour
        for contour in scan_contours
        if len(contour) >= 35 and abs(cv2.contourArea(contour)) >= 220.0
    ]
    for box in (item for item in boxes if item not in used_boxes):
        value, ocr_score = _read_label_value(bgr, box)
        if value is None:
            continue
        endpoint = _trace_box_leader_endpoint(exact, box)
        if endpoint is None:
            endpoint = _project_box_to_scan_contour(box, scan_contours)
        if endpoint is None:
            continue
        x, y = endpoint
        if any(np.hypot(x - item["point"][0], y - item["point"][1]) <= 9.0 for item in records):
            continue
        used_boxes.add(box)
        bx0, by0, bx1, by1 = box
        label_center = ((bx0 + bx1 - 1) / 2.0, (by0 + by1 - 1) / 2.0)
        direction = np.asarray((label_center[0] - x, label_center[1] - y), dtype=np.float64)
        length = float(np.hypot(direction[0], direction[1]))
        if length > 0:
            direction /= length
        if abs(direction[1]) >= abs(direction[0]) * 0.70:
            group = "bottom" if direction[1] >= 0 else "top"
        else:
            group = "right" if direction[0] >= 0 else "left"
        records.append(
            {
                "point": (int(x), int(y)),
                "box": box,
                "value": value,
                "ocr_score": ocr_score,
                "leader_direction": direction,
                "label_center": np.asarray(label_center, dtype=np.float64),
                "group": group,
                "recovered_short_leader": True,
            }
        )
    return records


def _deduplicate_anchors(anchors: list[Anchor], radius: float = 14.0) -> list[Anchor]:
    kept: list[Anchor] = []
    ordered = sorted(anchors, key=lambda item: item.source != "displayed_zero")
    for anchor in ordered:
        if all(np.hypot(anchor.x - other.x, anchor.y - other.y) > radius for other in kept):
            kept.append(anchor)
    return kept


def _contour_arc(
    contour: np.ndarray, start: int, end: int
) -> np.ndarray:
    points = contour.reshape(-1, 2)
    if end >= start:
        return points[start : end + 1]
    return np.vstack((points[start:], points[: end + 1]))


def _point_on_arc(arc: np.ndarray, ratio: float) -> tuple[int, int]:
    if len(arc) < 2:
        return tuple(map(int, arc[0]))
    lengths = np.linalg.norm(np.diff(arc.astype(np.float64), axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-6:
        return tuple(map(int, arc[0]))
    target = float(np.clip(ratio, 0.0, 1.0)) * total
    cumulative = 0.0
    for index, length in enumerate(lengths):
        if cumulative + length >= target:
            local = (target - cumulative) / max(float(length), 1e-6)
            point = arc[index] + local * (arc[index + 1] - arc[index])
            return tuple(map(int, np.rint(point)))
        cumulative += float(length)
    return tuple(map(int, arc[-1]))


def infer_zero_candidates(records: list[dict], contour_mask: np.ndarray) -> list[Anchor]:
    """Interpolate zeros only between adjacent numeric points on one contour."""
    readable = [record for record in records if record["value"] is not None]
    contours, _hierarchy = cv2.findContours(
        contour_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        contour
        for contour in contours
        if len(contour) >= 40 and abs(cv2.contourArea(contour)) >= 250.0
    ]
    assignments: list[dict] = []
    maximum_projection_distance = max(18.0, min(contour_mask.shape) * 0.07)
    for record in readable:
        point = np.asarray(record["point"], dtype=np.float32)
        best: tuple[float, int, int] | None = None
        for contour_id, contour in enumerate(contours):
            contour_points = contour.reshape(-1, 2).astype(np.float32)
            distances2 = np.square(contour_points - point).sum(axis=1)
            contour_index = int(np.argmin(distances2))
            distance = float(np.sqrt(distances2[contour_index]))
            if best is None or distance < best[0]:
                best = (distance, contour_id, contour_index)
        if best is not None and best[0] <= maximum_projection_distance:
            assignments.append(
                {
                    "record": record,
                    "contour_id": best[1],
                    "contour_index": best[2],
                    "distance": best[0],
                }
            )

    anchors: list[Anchor] = []
    for record in readable:
        value = float(record["value"])
        if abs(value) < 0.05:
            x, y = record["point"]
            assignment = next(
                (item for item in assignments if item["record"] is record), None
            )
            anchors.append(
                Anchor(
                    x,
                    y,
                    0.0,
                    "displayed_zero",
                    evidence="표시값 0.0",
                    group=record["group"],
                    contour_id=(assignment["contour_id"] if assignment is not None else None),
                )
            )

    maximum_arc_gap = max(170.0, min(contour_mask.shape) * 0.52)
    for contour_id, contour in enumerate(contours):
        # OpenCV traces inner openings in the opposite direction to the outer
        # part boundary.  Normalize every contour to the same (negative
        # oriented-area) direction before reading a sequence of measurements.
        # Otherwise the same physical trend is interpreted backwards on a
        # rounded opening, and the turning zero is placed on the wrong side of
        # the minimum value.
        working_contour = contour
        contour_items = [
            dict(item)
            for item in assignments
            if item["contour_id"] == contour_id
        ]
        if cv2.contourArea(contour, oriented=True) > 0.0:
            working_contour = contour[::-1].copy()
            last_index = len(contour) - 1
            for item in contour_items:
                item["contour_index"] = last_index - item["contour_index"]
        items = sorted(
            contour_items,
            key=lambda item: item["contour_index"],
        )
        if len(items) < 2:
            continue
        pairs = list(zip(items, items[1:]))
        if len(items) >= 3:
            pairs.append((items[-1], items[0]))
        for first_item, second_item in pairs:
            first = first_item["record"]
            second = second_item["record"]
            first_value = float(first["value"])
            second_value = float(second["value"])
            if first_value * second_value >= 0.0:
                continue
            arc = _contour_arc(
                working_contour,
                first_item["contour_index"],
                second_item["contour_index"],
            )
            arc_length = float(
                np.linalg.norm(np.diff(arc.astype(np.float64), axis=0), axis=1).sum()
            )
            if arc_length > maximum_arc_gap:
                continue
            ratio = -first_value / (second_value - first_value)
            x, y = _point_on_arc(arc, ratio)
            anchors.append(
                Anchor(
                    x,
                    y,
                    0.0,
                    "contour_interpolated_zero",
                    evidence=f"{first_value:+.1f} → {second_value:+.1f}",
                    group=(
                        first["group"]
                        if first["group"] == second["group"]
                        else f"contour_{contour_id}"
                    ),
                    contour_id=contour_id,
                )
            )

        # Same-sign extrema are deliberately not treated as zeros.  The final
        # operator line may start/end only at an explicit 0 or a measured
        # minus/plus sign crossing.
    return _deduplicate_anchors(anchors)


def _local_linearity(
    values: np.ndarray,
    valid: np.ndarray,
    point: tuple[int, int],
    radius: int = 24,
) -> float:
    """Score how closely the local deviation field follows a linear plane."""
    x, y = point
    height, width = values.shape
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch_valid = valid[y0:y1, x0:x1]
    yy, xx = np.where(patch_valid)
    if len(xx) < 30:
        return 0.0
    samples = values[y0:y1, x0:x1][patch_valid].astype(np.float64)
    design = np.column_stack(
        (xx.astype(np.float64) - (x - x0), yy.astype(np.float64) - (y - y0), np.ones(len(xx)))
    )
    coefficients, *_ = np.linalg.lstsq(design, samples, rcond=None)
    predicted = design @ coefficients
    residual = float(np.square(samples - predicted).sum())
    total = float(np.square(samples - samples.mean()).sum())
    r_squared = max(0.0, 1.0 - residual / total) if total > 1e-9 else 0.0
    gradient = float(np.hypot(coefficients[0], coefficients[1]))
    gradient_factor = min(1.0, gradient / 0.004)
    return float(np.clip(r_squared * gradient_factor, 0.0, 1.0))


def _ring_value(
    values: np.ndarray,
    valid: np.ndarray,
    point: tuple[int, int],
    inner: int,
    outer: int,
) -> float | None:
    x, y = point
    h, w = values.shape
    x0, x1 = max(0, x - outer), min(w, x + outer + 1)
    y0, y1 = max(0, y - outer), min(h, y + outer + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance2 = (xx - x) ** 2 + (yy - y) ** 2
    ring = (
        (distance2 >= inner * inner)
        & (distance2 <= outer * outer)
        & valid[y0:y1, x0:x1]
    )
    samples = values[y0:y1, x0:x1][ring]
    if samples.size < 12:
        return None
    return float(np.median(samples))


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 세선화로 후보 밴드를 한 픽셀 중심선으로 만든다."""
    image = mask.astype(np.uint8).copy()
    for _ in range(240):
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            count = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9).astype(np.int16)
            sequence = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = np.zeros(image.shape, dtype=np.int16)
            for index in range(8):
                transitions += (
                    (sequence[index] == 0) & (sequence[index + 1] == 1)
                ).astype(np.int16)
            if step == 0:
                c1, c2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                c1, c2 = p2 * p4 * p8, p2 * p6 * p8
            remove = (
                (image == 1)
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
                & (c1 == 0)
                & (c2 == 0)
            )
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image.astype(bool)


NEIGHBORS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),            (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)

# Clockwise order is important for measuring steering-angle changes in A*.
DRIVING_NEIGHBORS = (
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
)


def _bfs(mask: np.ndarray, start: tuple[int, int]):
    h, w = mask.shape
    start_index = start[1] * w + start[0]
    distances = np.full(h * w, -1, dtype=np.int32)
    parents = np.full(h * w, -1, dtype=np.int32)
    distances[start_index] = 0
    queue: deque[int] = deque([start_index])
    farthest = start_index
    while queue:
        index = queue.popleft()
        y, x = divmod(index, w)
        if distances[index] > distances[farthest]:
            farthest = index
        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h) or not mask[ny, nx]:
                continue
            neighbor = ny * w + nx
            if distances[neighbor] >= 0:
                continue
            distances[neighbor] = distances[index] + 1
            parents[neighbor] = index
            queue.append(neighbor)
    return distances, parents, farthest


def _reconstruct(
    parents: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
) -> list[tuple[int, int]]:
    start_index = start[1] * width + start[0]
    index = end[1] * width + end[0]
    path: list[tuple[int, int]] = []
    while index >= 0:
        y, x = divmod(int(index), width)
        path.append((x, y))
        if index == start_index:
            path.reverse()
            return path
        index = int(parents[index])
    return []


def _nearest(point: tuple[int, int], coordinates_xy: np.ndarray):
    if not len(coordinates_xy):
        return None, float("inf")
    delta = coordinates_xy.astype(np.float32) - np.asarray(point, dtype=np.float32)
    distances2 = np.einsum("ij,ij->i", delta, delta)
    index = int(np.argmin(distances2))
    return tuple(map(int, coordinates_xy[index])), float(np.sqrt(distances2[index]))


def _select_skeleton_component(
    skeleton: np.ndarray,
    anchors: list[Anchor],
    snap_radius: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return skeleton, []

    best_label = 1
    best_score = (-1, -1)
    best_snapped: list[tuple[int, int]] = []
    for label in range(1, n):
        yy, xx = np.where(labels == label)
        coordinates = np.column_stack((xx, yy))
        snapped: list[tuple[int, int]] = []
        for anchor in anchors:
            point, distance = _nearest((anchor.x, anchor.y), coordinates)
            if point is not None and distance <= snap_radius and point not in snapped:
                snapped.append(point)
        score = (len(snapped), int(stats[label, cv2.CC_STAT_AREA]))
        if score > best_score:
            best_label, best_score, best_snapped = label, score, snapped
    return labels == best_label, best_snapped


def _single_path(
    skeleton: np.ndarray,
    snapped_anchors: list[tuple[int, int]],
    preserve_anchor_order: bool,
) -> list[tuple[int, int]]:
    yy, xx = np.where(skeleton)
    if not len(xx):
        return []
    h, w = skeleton.shape

    if preserve_anchor_order and len(snapped_anchors) >= 2:
        combined: list[tuple[int, int]] = []
        for start, end in zip(snapped_anchors, snapped_anchors[1:]):
            distances, parents, _ = _bfs(skeleton, start)
            if distances[end[1] * w + end[0]] < 0:
                continue
            segment = _reconstruct(parents, start, end, w)
            combined.extend(segment if not combined else segment[1:])
        if combined:
            return combined

    if len(snapped_anchors) >= 2:
        best_distance = -1
        best_path: list[tuple[int, int]] = []
        for index, start in enumerate(snapped_anchors[:-1]):
            distances, parents, _ = _bfs(skeleton, start)
            for end in snapped_anchors[index + 1:]:
                distance = int(distances[end[1] * w + end[0]])
                if distance > best_distance:
                    best_distance = distance
                    best_path = _reconstruct(parents, start, end, w)
        if best_path:
            return best_path

    if len(snapped_anchors) == 1:
        start = snapped_anchors[0]
        _, parents, farthest_index = _bfs(skeleton, start)
        ey, ex = divmod(int(farthest_index), w)
        return _reconstruct(parents, start, (ex, ey), w)

    seed = (int(xx[0]), int(yy[0]))
    _, _, farthest_index = _bfs(skeleton, seed)
    fy, fx = divmod(int(farthest_index), w)
    start = (fx, fy)
    _, parents, farthest_index = _bfs(skeleton, start)
    ey, ex = divmod(int(farthest_index), w)
    return _reconstruct(parents, start, (ex, ey), w)


def _work_path(
    path: list[tuple[int, int]],
    epsilon: float,
    max_vertices: int,
    allowed_mask: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """원시 경로를 작업자가 옮겨 그리기 쉬운 직선형 폴리라인으로 줄인다.

    곡선 보간은 사용하지 않는다. RDP 허용 오차를 자동으로 올려 꼭짓점 수를
    제한하므로 작은 요철을 따라 생기는 잦은 꺾임은 사라지고 큰 방향 전환만
    남는다.
    """
    if len(path) < 2:
        return [(float(x), float(y)) for x, y in path]
    contour = np.asarray(path, dtype=np.float32).reshape(-1, 1, 2)
    epsilon = max(float(epsilon), 0.0)
    effective_epsilon = epsilon
    points = cv2.approxPolyDP(contour, epsilon, False).reshape(-1, 2)

    if max_vertices >= 2 and len(points) > max_vertices:
        low = epsilon
        high = float(np.hypot(*np.ptp(contour.reshape(-1, 2), axis=0)))
        best = points
        for _ in range(28):
            middle = (low + high) / 2.0
            candidate = cv2.approxPolyDP(contour, middle, False).reshape(-1, 2)
            if len(candidate) <= max_vertices:
                best = candidate
                high = middle
            else:
                low = middle
        points = best
        effective_epsilon = high

    if allowed_mask is not None and len(contour) > 2:
        corridor = cv2.dilate(
            allowed_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        ).astype(bool)
        source = contour.reshape(-1, 2)

        def segment_is_allowed(first: np.ndarray, last: np.ndarray) -> bool:
            line = np.zeros(corridor.shape, dtype=np.uint8)
            cv2.line(
                line,
                tuple(np.rint(first).astype(int)),
                tuple(np.rint(last).astype(int)),
                255,
                1,
                cv2.LINE_8,
            )
            samples = corridor[line > 0]
            return bool(samples.size and samples.mean() >= 0.88)

        def constrained(points_in: np.ndarray) -> list[np.ndarray]:
            if len(points_in) <= 2:
                return [points_in[0], points_in[-1]]
            start, end = points_in[0], points_in[-1]
            vector = end - start
            length = float(np.hypot(vector[0], vector[1]))
            if length <= 1e-6:
                distances = np.linalg.norm(points_in - start, axis=1)
            else:
                distances = np.abs(
                    vector[0] * (start[1] - points_in[:, 1])
                    - (start[0] - points_in[:, 0]) * vector[1]
                ) / length
            split = int(np.argmax(distances))
            maximum = float(distances[split])
            if maximum <= effective_epsilon and segment_is_allowed(start, end):
                return [start, end]
            if split <= 0 or split >= len(points_in) - 1:
                split = len(points_in) // 2
            left = constrained(points_in[: split + 1])
            right = constrained(points_in[split:])
            return left[:-1] + right

        points = np.asarray(constrained(source), dtype=np.float32)

    points = points.astype(np.float32)
    return [(float(x), float(y)) for x, y in points]


def _smooth_waypoint_route(
    skeleton: np.ndarray,
    waypoints: list[Anchor],
    allowed_mask: np.ndarray,
    epsilon: float,
    max_vertices: int,
) -> list[tuple[float, float]]:
    """Join ordered zero waypoints without drawing through internal holes."""
    yy, xx = np.where(skeleton)
    if len(xx) == 0 or len(waypoints) < 2:
        return [(float(item.x), float(item.y)) for item in waypoints]
    coordinates = np.column_stack((xx, yy))
    snapped: list[tuple[int, int]] = []
    for waypoint in waypoints:
        point, _distance = _nearest((waypoint.x, waypoint.y), coordinates)
        if point is None:
            return [(float(item.x), float(item.y)) for item in waypoints]
        snapped.append(point)

    combined: list[tuple[float, float]] = []
    segment_vertex_limit = max(3, int(np.ceil(max_vertices / max(len(waypoints) - 1, 1))) + 1)
    width = skeleton.shape[1]
    for index, (start, end) in enumerate(zip(snapped, snapped[1:])):
        distances, parents, _ = _bfs(skeleton, start)
        if distances[end[1] * width + end[0]] < 0:
            segment = [
                (waypoints[index].x, waypoints[index].y),
                (waypoints[index + 1].x, waypoints[index + 1].y),
            ]
        else:
            segment = _reconstruct(parents, start, end, width)
            segment.insert(0, (waypoints[index].x, waypoints[index].y))
            segment.append((waypoints[index + 1].x, waypoints[index + 1].y))
        simplified = _work_path(
            segment,
            epsilon,
            segment_vertex_limit,
            allowed_mask=allowed_mask,
        )
        combined.extend(simplified if not combined else simplified[1:])
    return combined


def _autonomous_cost_path(
    allowed_mask: np.ndarray,
    values: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    turn_penalty: float,
    obstacle_clearance_weight: float,
    margin: int = 150,
) -> list[tuple[int, int]]:
    """Direction-aware A* through the drivable deviation band.

    Pixels outside ``allowed_mask`` are hard obstacles.  Among drivable pixels,
    paths close to zero deviation and paths with fewer direction changes are
    cheaper.  Clearance cost also keeps the route from scraping obstacle edges.
    """
    scale = 3 if min(allowed_mask.shape) >= 450 else 2
    search_width = max(1, int(round(allowed_mask.shape[1] / scale)))
    search_height = max(1, int(round(allowed_mask.shape[0] / scale)))
    search_allowed = cv2.resize(
        allowed_mask.astype(np.uint8),
        (search_width, search_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    search_values = cv2.resize(
        values.astype(np.float32),
        (search_width, search_height),
        interpolation=cv2.INTER_AREA,
    )
    scaled_start = (int(round(start[0] / scale)), int(round(start[1] / scale)))
    scaled_end = (int(round(end[0] / scale)), int(round(end[1] / scale)))
    scaled_margin = max(20, int(round(margin / scale)))
    height, width = search_allowed.shape
    x0 = max(0, min(scaled_start[0], scaled_end[0]) - scaled_margin)
    x1 = min(width, max(scaled_start[0], scaled_end[0]) + scaled_margin + 1)
    y0 = max(0, min(scaled_start[1], scaled_end[1]) - scaled_margin)
    y1 = min(height, max(scaled_start[1], scaled_end[1]) + scaled_margin + 1)
    walkable = search_allowed[y0:y1, x0:x1]
    yy, xx = np.where(walkable)
    if not len(xx):
        return []
    coordinates = np.column_stack((xx, yy))
    local_start, _ = _nearest((scaled_start[0] - x0, scaled_start[1] - y0), coordinates)
    local_end, _ = _nearest((scaled_end[0] - x0, scaled_end[1] - y0), coordinates)
    if local_start is None or local_end is None:
        return []

    local_values = np.abs(search_values[y0:y1, x0:x1]).astype(np.float32)
    normalized_deviation = np.clip(local_values / 0.5, 0.0, 2.0)
    clearance = cv2.distanceTransform(walkable.astype(np.uint8), cv2.DIST_L2, 3)
    travel_cost = (
        1.0
        + 10.0 * normalized_deviation
        + float(obstacle_clearance_weight) / (clearance + 1.0)
    )
    crop_height, crop_width = walkable.shape
    start_index = local_start[1] * crop_width + local_start[0]
    end_index = local_end[1] * crop_width + local_end[0]
    direction_count = len(DRIVING_NEIGHBORS)
    state_count = crop_height * crop_width * direction_count
    distances = np.full(state_count, np.inf, dtype=np.float32)
    parents = np.full(state_count, -1, dtype=np.int32)
    queue: list[tuple[float, float, int]] = []
    for direction in range(direction_count):
        state = start_index * direction_count + direction
        distances[state] = 0.0
        heapq.heappush(
            queue,
            (
                float(np.hypot(local_start[0] - local_end[0], local_start[1] - local_end[1])),
                0.0,
                state,
            ),
        )
    final_state = -1
    while queue:
        _priority, current_distance, state = heapq.heappop(queue)
        if current_distance > float(distances[state]) + 1e-6:
            continue
        index, previous_direction = divmod(state, direction_count)
        if index == end_index:
            final_state = state
            break
        y, x = divmod(index, crop_width)
        for direction, (dx, dy) in enumerate(DRIVING_NEIGHBORS):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < crop_width and 0 <= ny < crop_height) or not walkable[ny, nx]:
                continue
            neighbor = ny * crop_width + nx
            step = 1.41421356 if dx and dy else 1.0
            edge_cost = step * float((travel_cost[y, x] + travel_cost[ny, nx]) * 0.5)
            direction_delta = abs(direction - previous_direction)
            direction_delta = min(direction_delta, direction_count - direction_delta)
            if direction_delta:
                edge_cost += float(turn_penalty) * (direction_delta ** 1.35)
            candidate_distance = current_distance + edge_cost
            neighbor_state = neighbor * direction_count + direction
            if candidate_distance + 1e-6 >= float(distances[neighbor_state]):
                continue
            distances[neighbor_state] = candidate_distance
            parents[neighbor_state] = state
            heuristic = float(np.hypot(nx - local_end[0], ny - local_end[1]))
            heapq.heappush(
                queue,
                (candidate_distance + heuristic, candidate_distance, neighbor_state),
            )
    if final_state < 0:
        return []
    path: list[tuple[int, int]] = []
    state = final_state
    while state >= 0:
        index, _direction = divmod(state, direction_count)
        y, x = divmod(int(index), crop_width)
        path.append(
            (
                int(np.clip((x + x0) * scale, 0, allowed_mask.shape[1] - 1)),
                int(np.clip((y + y0) * scale, 0, allowed_mask.shape[0] - 1)),
            )
        )
        if index == start_index:
            path.reverse()
            return path
        state = int(parents[state])
    return []


def _operator_style_route(
    clean_bgr: np.ndarray,
    part_mask: np.ndarray,
    values: np.ndarray,
    endpoints: list[Anchor],
    epsilon: float,
    max_vertices: int,
) -> list[tuple[float, float]]:
    """Build a sparse correction-sheet polyline between two certain zeros.

    When both zeros lie on the same side of the part, the route goes around the
    opposite side of the major openings.  This produces the long, intentional
    straight segments used on an operator correction sheet.  For mixed-side
    endpoints, a high-turn-penalty shape route is used as a conservative
    fallback and then reduced to at most six vertices.
    """
    if len(endpoints) < 2:
        return [(float(item.x), float(item.y)) for item in endpoints]

    first, second = endpoints[:2]
    if first.x > second.x:
        first, second = second, first
    start = (first.x, first.y)
    end = (second.x, second.y)

    yy, xx = np.where(part_mask)
    if not len(xx):
        return [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
    part_x0, part_x1 = int(xx.min()), int(xx.max())
    part_y0, part_y1 = int(yy.min()), int(yy.max())
    part_width = part_x1 - part_x0 + 1
    part_height = part_y1 - part_y0 + 1
    part_area_box = float(part_width * part_height)
    center_y = (part_y0 + part_y1) / 2.0

    white = np.all(clean_bgr >= 245, axis=2).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        white, connectivity=8
    )
    openings: list[tuple[int, int, int, int, int]] = []
    minimum_opening_area = max(1400, int(part_area_box * 0.0035))
    maximum_opening_area = int(part_area_box * 0.18)
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        center_x = x + width / 2.0
        center_opening_y = y + height / 2.0
        if not (minimum_opening_area <= area <= maximum_opening_area):
            continue
        if not (first.x < center_x < second.x):
            continue
        if not (part_y0 < center_opening_y < part_y1):
            continue
        openings.append((x, y, width, height, area))

    side_margin = max(18, int(round(part_height * 0.055)))
    same_lower_side = first.y > center_y + part_height * 0.12 and second.y > center_y + part_height * 0.12
    same_upper_side = first.y < center_y - part_height * 0.12 and second.y < center_y - part_height * 0.12
    major_area = max(5500, int(part_area_box * 0.014))
    major_openings = [item for item in openings if item[4] >= major_area]

    if (same_lower_side or same_upper_side) and major_openings:
        horizontal_clearance = max(24, int(round(part_width * 0.028)))
        left_edge = min(item[0] for item in major_openings)
        right_edge = max(item[0] + item[2] for item in major_openings)
        left_knee = int(np.clip(left_edge - horizontal_clearance, first.x + 20, second.x - 60))
        right_knee = int(np.clip(right_edge + horizontal_clearance, left_knee + 60, second.x - 20))
        upper_base = min(item[1] for item in major_openings) - side_margin
        lower_base = max(item[1] + item[3] for item in major_openings) + side_margin
        upper_base = int(np.clip(upper_base, part_y0 + side_margin, center_y - side_margin))
        lower_base = int(np.clip(lower_base, center_y + side_margin, part_y1 - side_margin))

        guide_rows = {
            int(np.clip(upper_base + offset, part_y0 + side_margin, center_y - side_margin))
            for offset in (-14, 0, 14)
        }
        guide_rows.update(
            int(np.clip(lower_base + offset, center_y + side_margin, part_y1 - side_margin))
            for offset in (-14, 0, 14)
        )
        candidates: list[list[tuple[float, float]]] = []
        for guide_y in sorted(guide_rows):
            candidates.append(
                [
                    (float(start[0]), float(start[1])),
                    (float(left_knee), float(guide_y)),
                    (float(right_knee), float(guide_y)),
                    (float(end[0]), float(end[1])),
                ]
            )

        def route_score(route: list[tuple[float, float]]) -> float:
            line = np.zeros(part_mask.shape, dtype=np.uint8)
            points = np.rint(route).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(line, [points], False, 1, 3, cv2.LINE_8)
            selected = line > 0
            if not selected.any():
                return float("inf")
            inside = part_mask[selected]
            absolute = np.abs(values[selected]).astype(np.float64)
            outside_part_ratio = 1.0 - float(inside.mean())
            outside_band_ratio = float(((absolute > 0.5) | ~inside).mean())
            mean_deviation = float(absolute[inside].mean()) if inside.any() else 10.0
            length = sum(
                float(np.hypot(b[0] - a[0], b[1] - a[1]))
                for a, b in zip(route, route[1:])
            )
            return (
                250.0 * outside_part_ratio
                + 45.0 * outside_band_ratio
                + 6.0 * mean_deviation
                + 0.002 * length
            )

        return min(candidates, key=route_score)

    raw = _autonomous_cost_path(
        part_mask,
        values,
        start,
        end,
        turn_penalty=10.0,
        obstacle_clearance_weight=14.0,
        margin=max(part_width, part_height),
    )
    if not raw:
        raw = [start, end]
    if raw[0] != start:
        raw.insert(0, start)
    if raw[-1] != end:
        raw.append(end)
    return _work_path(
        raw,
        max(14.0, epsilon * 1.8),
        min(max(4, max_vertices), 6),
        allowed_mask=part_mask,
    )


def _order_anchors_along_path(
    raw_path: list[tuple[int, int]],
    anchors: list[Anchor],
    max_distance: float,
    cluster_gap: int = 18,
) -> list[Anchor]:
    """실제 측정 포인트를 후보 중심 경로를 따라 나타나는 순서로 정렬한다.

    후보 중심 경로 좌표는 순서를 정할 때만 사용하며 최종선의 꼭짓점으로 사용하지
    않는다. 경로의 거의 같은 위치에 여러 포인트가 투영되면 가장 가까운 실제
    측정 포인트 하나를 대표로 선택한다.
    """
    if not raw_path:
        return []
    path_xy = np.asarray(raw_path, dtype=np.float32)
    records: list[tuple[int, float, Anchor]] = []
    for anchor in anchors:
        delta = path_xy - np.asarray((anchor.x, anchor.y), dtype=np.float32)
        distances2 = np.einsum("ij,ij->i", delta, delta)
        index = int(np.argmin(distances2))
        distance = float(np.sqrt(distances2[index]))
        if distance <= max_distance:
            records.append((index, distance, anchor))
    records.sort(key=lambda item: item[0])
    if not records:
        return []

    groups: list[list[tuple[int, float, Anchor]]] = [[records[0]]]
    for record in records[1:]:
        if record[0] - groups[-1][-1][0] <= cluster_gap:
            groups[-1].append(record)
        else:
            groups.append([record])
    return [min(group, key=lambda item: item[1])[2] for group in groups]


def _attach_zero_endpoints(
    raw_path: list[tuple[int, int]], zero_points: list[Anchor]
) -> list[tuple[int, int]]:
    """Make displayed numeric-zero points exact endpoints of the selected path."""
    if not raw_path or not zero_points:
        return raw_path
    path = list(raw_path)
    if len(zero_points) == 1:
        zero = (zero_points[0].x, zero_points[0].y)
        if np.hypot(zero[0] - path[-1][0], zero[1] - path[-1][1]) < np.hypot(
            zero[0] - path[0][0], zero[1] - path[0][1]
        ):
            path.reverse()
        if path[0] != zero:
            path.insert(0, zero)
        return path

    first = (zero_points[0].x, zero_points[0].y)
    second = (zero_points[1].x, zero_points[1].y)
    direct = np.hypot(first[0] - path[0][0], first[1] - path[0][1]) + np.hypot(
        second[0] - path[-1][0], second[1] - path[-1][1]
    )
    reversed_cost = np.hypot(second[0] - path[0][0], second[1] - path[0][1]) + np.hypot(
        first[0] - path[-1][0], first[1] - path[-1][1]
    )
    if reversed_cost < direct:
        first, second = second, first
    if path[0] != first:
        path.insert(0, first)
    if path[-1] != second:
        path.append(second)
    return path


def _select_zero_endpoints(
    raw_path: list[tuple[int, int]],
    candidates: list[Anchor],
    max_distance: float,
) -> list[Anchor]:
    """Recover the zero candidates that produced the two ends of a route."""
    if not raw_path or not candidates:
        return []
    endpoints = (raw_path[0], raw_path[-1])
    selected: list[Anchor] = []
    for endpoint in endpoints:
        available = [candidate for candidate in candidates if candidate not in selected]
        if not available:
            break
        candidate = min(
            available,
            key=lambda item: np.hypot(item.x - endpoint[0], item.y - endpoint[1]),
        )
        distance = float(np.hypot(candidate.x - endpoint[0], candidate.y - endpoint[1]))
        if distance <= max_distance:
            selected.append(candidate)
    if not selected:
        selected.append(
            min(
                candidates,
                key=lambda item: min(
                    np.hypot(item.x - endpoints[0][0], item.y - endpoints[0][1]),
                    np.hypot(item.x - endpoints[1][0], item.y - endpoints[1][1]),
                ),
            )
        )
    return selected


def _preferred_zero_endpoints(candidates: list[Anchor]) -> list[Anchor]:
    """Choose route endpoints while always honoring explicitly displayed zeros."""
    if len(candidates) <= 2:
        return list(candidates)
    displayed = [item for item in candidates if item.source == "displayed_zero"]
    if len(displayed) >= 2:
        best_pair = (displayed[0], displayed[1])
        best_distance = -1.0
        for index, first in enumerate(displayed[:-1]):
            for second in displayed[index + 1 :]:
                distance = float(np.hypot(first.x - second.x, first.y - second.y))
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (first, second)
        return list(best_pair)
    if len(displayed) == 1:
        first = displayed[0]
        same_row = [
            item
            for item in candidates
            if item is not first and item.group == first.group
        ]
        sign_crossings = [item for item in same_row if item.source != "turning_zero"]
        if sign_crossings:
            same_row = sign_crossings
        pool = same_row or [item for item in candidates if item is not first]
        second = max(
            pool,
            key=lambda item: np.hypot(first.x - item.x, first.y - item.y),
        )
        return [first, second]

    # With no displayed zero, prefer the longest pair belonging to one
    # measurement row; this avoids joining unrelated top/right annotations.
    preferred_candidates = [
        item for item in candidates if item.source != "turning_zero"
    ] or candidates
    best_pair: tuple[Anchor, Anchor] | None = None
    best_distance = -1.0
    for index, first in enumerate(preferred_candidates[:-1]):
        for second in preferred_candidates[index + 1 :]:
            if first.group != second.group:
                continue
            distance = float(np.hypot(first.x - second.x, first.y - second.y))
            if distance > best_distance:
                best_distance = distance
                best_pair = (first, second)
    return list(best_pair) if best_pair else preferred_candidates[:2]


def _zero_route_waypoints(
    candidates: list[Anchor], endpoints: list[Anchor]
) -> list[Anchor]:
    """Add opposite-row zero crossings when two endpoints lie on one edge.

    A route whose two zero endpoints are both on the bottom edge would otherwise
    take the shortest path straight along the bottom. Passing through the left
    and right zero crossings on the opposite row produces the operator-style
    boundary around the correction region.
    """
    if len(endpoints) != 2 or endpoints[0].group != endpoints[1].group:
        return endpoints
    opposite = {
        "bottom": "top",
        "top": "bottom",
        "left": "right",
        "right": "left",
    }.get(endpoints[0].group)
    if opposite is None:
        return endpoints
    pool = [item for item in candidates if item.group == opposite]
    if len(pool) < 2:
        return endpoints

    if endpoints[0].group in {"top", "bottom"}:
        coordinate = lambda item: item.x
    else:
        coordinate = lambda item: item.y
    first, second = sorted(endpoints, key=coordinate)
    perpendicular = (lambda item: item.y) if endpoints[0].group in {"top", "bottom"} else (lambda item: item.x)
    best_middle: tuple[Anchor, Anchor] | None = None
    best_cost = float("inf")
    for index, candidate_a in enumerate(pool[:-1]):
        for candidate_b in pool[index + 1 :]:
            left, right = sorted((candidate_a, candidate_b), key=coordinate)
            cost = (
                abs(coordinate(left) - coordinate(first))
                + abs(coordinate(right) - coordinate(second))
                + 4.0 * abs(perpendicular(left) - perpendicular(right))
            )
            if cost < best_cost:
                best_cost = cost
                best_middle = (left, right)
    if best_middle is None:
        return endpoints
    middle = list(best_middle)
    return [first, *middle, second]


def detect_advanced_zero_line(
    original_bgr: np.ndarray,
    clean_bgr: np.ndarray | None,
    *,
    vmin: float,
    vmax: float,
    config: AdvanceConfig | None = None,
    manual_anchors: list[tuple[int, int]] | None = None,
) -> AdvanceResult:
    cfg = config or AdvanceConfig()
    warnings: list[str] = []
    if cfg.band_low >= cfg.band_high:
        raise ValueError("band_low는 band_high보다 작아야 합니다.")

    clean_bgr = original_bgr if clean_bgr is None else clean_bgr
    if clean_bgr.shape[:2] != original_bgr.shape[:2]:
        raise ValueError("원본 이미지와 라벨 복원 이미지의 크기가 다릅니다.")

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    colorbar = detect_colorbar(original_rgb, vmin=vmin, vmax=vmax)
    values, color_valid = colorbar.map_image(
        clean_rgb, max_dist=cfg.color_max_dist, method="hue"
    )
    if cfg.smooth_ksize > 1:
        values = cv2.medianBlur(values, cfg.smooth_ksize | 1)

    h, w = values.shape
    part = color_valid.copy()
    x0, x1 = colorbar.info.x0, colorbar.info.x1
    pad = max(int(w * 0.045), 40)
    if x0 < w / 2:
        part[:, : min(x1 + pad, w)] = False
    else:
        part[:, max(x0 - pad, 0):] = False
    part = _clean_binary(part, 1, 2)
    part = _keep_main_parts(part, cfg.min_part_area, cfg.part_area_ratio)

    candidate = part & (values >= cfg.band_low) & (values <= cfg.band_high)
    candidate = _clean_binary(candidate, cfg.morph_open, cfg.morph_close)
    candidate &= part
    candidate = _drop_small(candidate, cfg.min_band_area)
    if not candidate.any():
        raise RuntimeError("지정한 편차 범위에서 후보 영역을 찾지 못했습니다.")

    detected_points: list[Anchor] = []
    candidate_anchors: list[Anchor] = []
    zero_candidates: list[Anchor] = []
    zero_points: list[Anchor] = []
    if manual_anchors:
        for x, y in manual_anchors:
            if 0 <= x < w and 0 <= y < h:
                anchor = Anchor(x, y, None, "manual")
                detected_points.append(anchor)
                candidate_anchors.append(anchor)
    else:
        for x, y, _radius, _color in detect_measurement_points(original_bgr):
            estimated = _ring_value(
                values,
                part,
                (x, y),
                cfg.anchor_ring_inner,
                cfg.anchor_ring_outer,
            )
            anchor = Anchor(x, y, estimated, "label_leader_endpoint")
            detected_points.append(anchor)
            if estimated is not None and cfg.band_low <= estimated <= cfg.band_high:
                candidate_anchors.append(anchor)

        measurement_records = _measurement_records(original_bgr)
        zero_candidates = infer_zero_candidates(measurement_records, part)
        if not zero_candidates:
            # Keep the previous exact-0 detector as a conservative fallback if
            # a new label font prevents the numeric template reader from working.
            zero_candidates = detect_numeric_zero_points(original_bgr)
            for anchor in zero_candidates:
                anchor.linearity = _local_linearity(values, part, (anchor.x, anchor.y))

    skeleton = _skeletonize(candidate)
    preferred_zero_points = _preferred_zero_endpoints(zero_candidates)
    # Autonomous mode starts with the numeric start/end zeros.  Extra zeros are
    # guidance points only when a later route-selection stage proves they are
    # necessary; forcing unrelated zeros can create an impossible segment whose
    # only solution crosses the out-of-band obstacle region.
    zero_waypoints = list(preferred_zero_points)
    route_anchors = zero_waypoints if zero_waypoints else candidate_anchors
    selected_skeleton, snapped = _select_skeleton_component(
        skeleton, route_anchors, cfg.anchor_snap_radius
    )
    if not selected_skeleton.any():
        raise RuntimeError("후보 영역의 중심선을 만들지 못했습니다.")
    if len(snapped) < 2:
        if zero_candidates and len(snapped) == 1:
            warnings.append(
                "표시된 0점이 1개이므로 해당 점을 한쪽 끝점으로 고정하고 "
                "반대쪽 끝점을 자동 선택했습니다."
            )
        else:
            warnings.append(
                "후보 영역에 붙은 기준점이 2개 미만이어서 가장 긴 중심 경로를 사용했습니다."
            )

    raw_path = _single_path(
        selected_skeleton,
        snapped,
        preserve_anchor_order=bool(manual_anchors or len(zero_waypoints) > 2),
    )
    if len(raw_path) < 2:
        raise RuntimeError("연속된 단일 경로를 만들지 못했습니다.")
    zero_points = _select_zero_endpoints(
        raw_path, preferred_zero_points, cfg.anchor_snap_radius * 1.35
    )
    raw_path = _attach_zero_endpoints(raw_path, zero_points)
    ordered_anchors = _order_anchors_along_path(
        raw_path, candidate_anchors, cfg.anchor_snap_radius
    )
    if len(ordered_anchors) < 2:
        raise RuntimeError(
            "선택된 후보 경로를 따라 연결할 실제 라벨 포인트가 2개 미만입니다."
        )

    # RDP는 입력 좌표 중 일부만 남긴다. 따라서 단순화 후에도 모든 꼭짓점은
    # 반드시 실제 라벨 측정 포인트다.
    anchor_path = [(anchor.x, anchor.y) for anchor in ordered_anchors]
    point_only_path = _work_path(
        anchor_path, cfg.simplify_epsilon, cfg.max_vertices
    )
    anchor_lookup = {(anchor.x, anchor.y): anchor for anchor in ordered_anchors}
    line_anchors = [
        anchor_lookup[(int(round(x)), int(round(y)))] for x, y in point_only_path
    ]

    # 최종 권장선: 측정 포인트는 경로 선택에만 사용하고, 후보 밴드 중심 경로를
    # 작업자가 옮겨 그리기 쉬운 직선형 폴리라인으로 단순화한다. 후보 영역이나
    # 구멍을 가로지르는 지름길은 constrained RDP가 다시 분할한다.
    if len(zero_waypoints) >= 2:
        smooth_path = _operator_style_route(
            clean_bgr,
            part,
            values,
            zero_waypoints,
            cfg.simplify_epsilon,
            cfg.max_vertices,
        )
    else:
        smooth_path = _work_path(
            raw_path,
            cfg.simplify_epsilon,
            cfg.max_vertices,
            allowed_mask=candidate & part,
        )

    return AdvanceResult(
        values=values,
        part_mask=part,
        candidate_mask=candidate,
        selected_skeleton=selected_skeleton,
        raw_path=raw_path,
        smooth_path=smooth_path,
        point_only_path=point_only_path,
        detected_points=detected_points,
        candidate_anchors=candidate_anchors,
        zero_candidates=zero_candidates,
        zero_waypoints=zero_waypoints,
        zero_points=zero_points,
        line_anchors=line_anchors,
        snapped_anchors=snapped,
        colorbar=colorbar.to_dict(),
        warnings=warnings,
        config=cfg,
    )


def _alpha_mask(
    bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> np.ndarray:
    out = bgr.copy()
    tint = np.empty_like(out)
    tint[:] = color
    selected = mask.astype(bool)
    out[selected] = (
        out[selected].astype(np.float32) * (1.0 - alpha)
        + tint[selected].astype(np.float32) * alpha
    ).astype(np.uint8)
    return out


def _text_origin(
    image: np.ndarray,
    point: tuple[int, int],
    text: str,
    font_scale: float,
    thickness: int = 1,
    gap: int = 14,
) -> tuple[int, int]:
    """Place a point label inside the image even near an edge."""
    height, width = image.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = point
    text_x = x + gap
    if text_x + text_width > width - 5:
        text_x = x - gap - text_width
    text_x = max(5, min(text_x, width - text_width - 5))

    text_y = y - gap
    if text_y - text_height < 5:
        text_y = y + gap + text_height
    text_y = max(text_height + 5, min(text_y, height - baseline - 5))
    return int(text_x), int(text_y)


def _draw_zero_label(
    image: np.ndarray,
    point: tuple[int, int],
    text: str,
    radius: int,
    font_scale: float = 0.55,
) -> None:
    cv2.circle(image, point, radius + 3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(image, point, radius, (0, 255, 255), 3, cv2.LINE_AA)
    origin = _text_origin(image, point, text, font_scale)
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def render_debug_images(
    clean_bgr: np.ndarray, result: AdvanceResult
) -> dict[str, np.ndarray]:
    """후보 영역부터 최종선까지 단계별 검증 이미지를 생성한다."""
    candidate_view = _alpha_mask(
        clean_bgr, result.candidate_mask, (255, 220, 40), 0.42
    )

    point_view = candidate_view.copy()
    candidate_lookup = {(a.x, a.y) for a in result.candidate_anchors}
    for anchor in result.detected_points:
        chosen = (anchor.x, anchor.y) in candidate_lookup
        color = (0, 210, 255) if chosen else (150, 150, 150)
        cv2.circle(point_view, (anchor.x, anchor.y), 5 if chosen else 3, color, -1, cv2.LINE_AA)
        if chosen and anchor.estimated_value is not None:
            cv2.putText(
                point_view,
                f"{anchor.estimated_value:+.2f}",
                (anchor.x + 6, anchor.y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    for zero in result.zero_points:
        _draw_zero_label(point_view, (zero.x, zero.y), "0 POINT", radius=8)

    raw_view = clean_bgr.copy()
    raw_view[result.selected_skeleton] = (255, 255, 255)
    raw_points = np.asarray(result.raw_path, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(raw_view, [raw_points], False, (0, 0, 255), 2, cv2.LINE_AA)
    for point in result.snapped_anchors:
        cv2.circle(raw_view, point, 5, (0, 210, 255), -1, cv2.LINE_AA)

    final_view = _alpha_mask(
        clean_bgr, result.candidate_mask, (255, 220, 40), 0.18
    )
    smooth_points = np.rint(result.smooth_path).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(final_view, [smooth_points], False, (0, 0, 255), 3, cv2.LINE_AA)
    endpoint_lookup = {(zero.x, zero.y) for zero in result.zero_points}
    for zero in result.zero_waypoints:
        is_endpoint = (zero.x, zero.y) in endpoint_lookup
        _draw_zero_label(
            final_view,
            (zero.x, zero.y),
            "0 POINT" if is_endpoint else "0 CROSS",
            radius=7 if is_endpoint else 5,
            font_scale=0.55 if is_endpoint else 0.45,
        )
    zero_view = clean_bgr.copy()
    selected_lookup = {(zero.x, zero.y) for zero in result.zero_points}
    route_lookup = {(zero.x, zero.y) for zero in result.zero_waypoints}
    for index, zero in enumerate(result.zero_candidates, start=1):
        selected = (zero.x, zero.y) in selected_lookup
        on_route = (zero.x, zero.y) in route_lookup
        label = "END 0" if selected else ("LINE 0" if on_route else f"Z{index}")
        _draw_zero_label(
            zero_view,
            (zero.x, zero.y),
            label,
            radius=9 if selected else 6,
            font_scale=0.48,
        )

    operator_view = clean_bgr.copy()
    cv2.polylines(
        operator_view, [smooth_points], False, (255, 255, 255), 7, cv2.LINE_AA
    )
    cv2.polylines(
        operator_view, [smooth_points], False, (0, 0, 255), 3, cv2.LINE_AA
    )
    for index, point in enumerate(smooth_points[:, 0], start=1):
        location = tuple(map(int, point))
        cv2.circle(operator_view, location, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(operator_view, location, 6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            operator_view,
            f"V{index}",
            _text_origin(operator_view, location, f"V{index}", 0.45, gap=9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        operator_view,
        "OPERATOR POLYLINE VERTICES",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        operator_view,
        "OPERATOR POLYLINE VERTICES",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    point_only_view = clean_bgr.copy()
    point_only_points = np.rint(result.point_only_path).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        point_only_view, [point_only_points], False, (0, 0, 255), 3, cv2.LINE_AA
    )
    for point in point_only_points[:, 0]:
        cv2.circle(point_only_view, tuple(point), 5, (0, 0, 255), -1, cv2.LINE_AA)

    line_mask = np.zeros(clean_bgr.shape[:2], dtype=np.uint8)
    cv2.polylines(line_mask, [smooth_points], False, 255, 2, cv2.LINE_AA)
    return {
        "01_candidate_band.png": candidate_view,
        "02_detected_points.png": point_view,
        "03_raw_path.png": raw_view,
        "04_work_zero_line.png": final_view,
        "04_smooth_zero_line.png": final_view,
        "05_points_only_line.png": point_only_view,
        "06_numeric_zero_points.png": zero_view,
        "07_operator_vertices.png": operator_view,
        "zero_line_mask.png": line_mask,
    }


__all__ = [
    "AdvanceConfig",
    "AdvanceResult",
    "Anchor",
    "detect_advanced_zero_line",
    "read_image",
    "render_debug_images",
    "write_image",
]
