"""영라인을 시트처럼 깔끔한 직선으로 긋는다.

[왜 필요했나]
지금까지는 두 0포인트 사이를 편차 기준 픽셀 최단경로로 이었다. 현업
지적이 두 가지였다 —

    "정답지처럼 좀 깔끔한 직선으로. 너는 지금 곡선이 많아서 흐물흐물거림"
    "저 구간엔 포인트도 없고 수치값도 없는데 왜 선을 넣는지"

[근거 — 현업 my_lab/zero_line_drawing (2026-08-26 수령)]
그쪽 README 의 "단순 제로라인 기준" 이 답을 그대로 준다.

    - 시작점과 끝점은 반드시 zero_point_selection 의 주요 0포인트를 쓴다.
    - 주요 0포인트 중 **직선 정렬도가 높은 점들을 자동으로 묶는다.**
    - **곡선은 사용하지 않는다.**
    - 직선을 우선하고, 허용범위 통과율이 낮을 때만 중간 꺾임을 최대 1개.

그리고 "허용범위" 는 편차 -0.5~+0.5mm 이진화 영역이다. 즉 선이 좋은지
나쁜지를 **그 선이 허용범위를 얼마나 지나가는가**로 판정한다. 경로를
최적화하는 게 아니라 후보 직선을 채점하는 방식이라 결과가 직선이다.

상수는 현업 코드(draw_simple_zero_lines.py, binarize_products.py)의
값을 그대로 옮겼다.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# 허용범위 — 현업 코드 그대로 -0.5~+0.5mm
TOLERANCE_MM = 0.5
# 이진화 후 정리: 7x7 타원 커널로 1회 침식, 팽창은 하지 않는다.
MORPH_KERNEL_SIZE = 7
EROSION_ITERATIONS = 1
# 선이 허용범위를 "지나간다" 고 볼 여유 반경
ROUTE_TOLERANCE_RADIUS_PX = 18
# 이 통과율을 넘으면 꺾지 않고 직선으로 끝낸다
MIN_STRAIGHT_COVERAGE = 0.55
# 꺾임 1개를 허용할 때의 제한
MAX_BEND_ANGLE_DEG = 72.0
MAX_ROUTE_LENGTH_RATIO = 1.55
# 꺾어서 이만큼은 좋아져야 꺾는다(애매하면 직선을 남긴다)
BEND_IMPROVEMENT = 0.08
# 0포인트를 한 직선으로 묶을 때의 정렬 허용 오차
KEY_POINT_ALIGNMENT_FACTOR = 0.65
MIN_ALIGNMENT_TOLERANCE_PX = 28.0
# 통과율 = 허용범위 통과율 * 0.78 + 부품 안 통과율 * 0.22
TOLERANCE_WEIGHT = 0.78
PRODUCT_WEIGHT = 0.22

# 현업이 알려준 품번별 컬러바 범위(위 mm, 아래 mm). 우리 검출기는
# 컬러바 눈금 숫자를 못 읽으면 편차를 +-1 로 정규화해서 내보내는데,
# 그때 이 값으로 되돌려야 "0.5mm" 가 실제 0.5mm 가 된다.
PRODUCT_COLORBAR_MM = {
    "64XX2": (2.0, -1.6),
    "67XX6": (3.0, -3.0),
    "71XX2": (2.0, -2.0),
}


@dataclass
class SimpleZeroLine:
    """0포인트 두 개를 잇는 직선(필요하면 꺾임 1개) 영라인."""

    line_id: int
    points: list                 # [[x, y], ...] — 2개 또는 3개
    route_type: str              # straight | one_bend | straight_preferred_over_weak_bend
    bend_count: int
    combined_coverage: float     # 종합 통과율
    tolerance_coverage: float    # 허용범위(-0.5~0.5mm) 통과율
    product_coverage: float      # 부품 안 통과율
    support_count: int           # 이 직선에 정렬된 0포인트 수
    length_px: float

    def to_dict(self) -> dict:
        return asdict(self)


def colorbar_span_for(part_no):
    """품번의 컬러바 범위(위, 아래 mm). 모르는 품번이면 None."""
    if not part_no:
        return None
    folded = part_no.upper().replace("-", "_").replace("_", "")
    return next(
        (span for key, span in PRODUCT_COLORBAR_MM.items() if key in folded), None
    )


def to_millimetres(values: np.ndarray, part_mask, part_no) -> np.ndarray:
    """정규화된 편차(+-1)를 품번 컬러바 기준 mm 로 되돌린다.

    이미 mm 눈금이면(범위가 +-1 을 넘으면) 그대로 둔다.
    """
    span = colorbar_span_for(part_no)
    if span is None:
        return values
    inside = values[np.asarray(part_mask) > 0]
    if inside.size == 0 or float(np.abs(inside).max()) > 1.05:
        return values
    top_mm, bottom_mm = span
    # 정규화값 +1 = 컬러바 위, -1 = 컬러바 아래
    centre = (top_mm + bottom_mm) / 2.0
    half = (top_mm - bottom_mm) / 2.0
    return values * half + centre


def build_tolerance_mask(values: np.ndarray, part_mask,
                         tolerance_mm: float = TOLERANCE_MM) -> np.ndarray:
    """|편차| <= tolerance_mm 인 곳을 이진화하고 현업 방식으로 정리한다.

    현업 README 3~5단계: 이진화 -> 7x7 1회 침식 -> 팽창 없이, 배경과
    연결되지 않은 검은 구멍만 채우기.
    """
    mask = np.asarray(part_mask) > 0
    binary = ((np.abs(values) <= tolerance_mm) & mask).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    binary = cv2.erode(binary, kernel, iterations=EROSION_ITERATIONS)

    # 바깥에서 물을 부어 배경을 표시하면, 남은 검은 곳이 내부 구멍이다.
    height, width = binary.shape
    flood = binary.copy()
    cv2.floodFill(flood, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def build_route_mask(tolerance_mask: np.ndarray, part_mask,
                     radius_px: int = ROUTE_TOLERANCE_RADIUS_PX) -> np.ndarray:
    """선이 허용범위를 "지나간다" 고 볼 여유를 준 마스크."""
    size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    grown = cv2.dilate(tolerance_mask, kernel, iterations=1)
    return cv2.bitwise_and(grown, (np.asarray(part_mask) > 0).astype(np.uint8) * 255)


def _point_line_distance(point, start, end) -> float:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return float(np.linalg.norm(point - start))
    offset = point - start
    cross = abs(float(direction[0] * offset[1] - direction[1] * offset[0]))
    return cross / length


def _sample_segment(start, end):
    length = max(int(round(float(np.linalg.norm(end - start)))), 1)
    steps = np.linspace(0.0, 1.0, length + 1)
    samples = start[None, :] + steps[:, None] * (end - start)[None, :]
    return (np.rint(samples[:, 0]).astype(np.int32),
            np.rint(samples[:, 1]).astype(np.int32))


def _segment_coverage(start, end, route_mask, product_mask):
    """이 선분이 허용범위와 부품 안을 얼마나 지나는가."""
    xs, ys = _sample_segment(start, end)
    xs = np.clip(xs, 0, route_mask.shape[1] - 1)
    ys = np.clip(ys, 0, route_mask.shape[0] - 1)
    tolerance = float(np.mean(route_mask[ys, xs] > 0))
    product = float(np.mean(product_mask[ys, xs] > 0))
    combined = TOLERANCE_WEIGHT * tolerance + PRODUCT_WEIGHT * product
    return combined, tolerance, product


def _group_aligned_points(points, route_mask, product_mask) -> list:
    """가장 강한 "거의 한 직선 위" 그룹을 반복해서 골라낸다."""
    remaining = set(range(len(points)))
    groups: list = []
    median_radius = float(np.median([p["radius_px"] for p in points]))
    alignment_tolerance = max(
        MIN_ALIGNMENT_TOLERANCE_PX, median_radius * KEY_POINT_ALIGNMENT_FACTOR
    )
    diag = math.hypot(route_mask.shape[1], route_mask.shape[0])

    while len(remaining) >= 2:
        best = None
        indexes = sorted(remaining)
        for left_pos, left in enumerate(indexes[:-1]):
            for right in indexes[left_pos + 1:]:
                start, end = points[left]["point"], points[right]["point"]
                span = float(np.linalg.norm(end - start))
                if span < 1.0:
                    continue
                group = [
                    i for i in indexes
                    if _point_line_distance(points[i]["point"], start, end)
                    <= alignment_tolerance
                ]
                direction = (end - start) / span
                projections = [
                    float(np.dot(points[i]["point"] - start, direction)) for i in group
                ]
                first = group[int(np.argmin(projections))]
                last = group[int(np.argmax(projections))]
                coverage, _, product = _segment_coverage(
                    points[first]["point"], points[last]["point"],
                    route_mask, product_mask,
                )
                group_span = float(
                    np.linalg.norm(points[last]["point"] - points[first]["point"])
                )
                score = (float(len(group)), round(coverage, 6), round(product, 6),
                         round(group_span / max(diag, 1.0), 6))
                if best is None or score > best[0]:
                    best = (score, group)
        if best is None:
            break
        groups.append(best[1])
        remaining.difference_update(best[1])

    # 홀로 남은 점은 가장 가까운 점에 붙인다.
    for singleton in sorted(remaining):
        other = min(
            (i for i in range(len(points)) if i != singleton),
            key=lambda i: float(
                np.linalg.norm(points[i]["point"] - points[singleton]["point"])
            ),
        )
        groups.append([singleton, other])
    return groups


def _group_endpoints(group, points):
    if len(group) == 2:
        return group[0], group[1]
    coordinates = np.array([points[i]["point"] for i in group])
    centered = coordinates - coordinates.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projections = centered @ axes[0]
    return group[int(np.argmin(projections))], group[int(np.argmax(projections))]


def _bend_angle_degrees(start, bend, end) -> float:
    first, second = bend - start, end - bend
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    if first_length < 1e-6 or second_length < 1e-6:
        return 180.0
    cosine = float(np.dot(first, second) / (first_length * second_length))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _straight_result(straight, tolerance, product, route_type):
    return {
        "route_type": route_type, "bend_count": 0,
        "combined_coverage": round(straight, 6),
        "tolerance_coverage": round(tolerance, 6),
        "product_coverage": round(product, 6),
    }


def _choose_simple_route(start, end, route_mask, product_mask):
    """직선을 먼저 보고, 통과율이 낮을 때만 꺾임 1개를 찾는다."""
    straight, tolerance, product = _segment_coverage(
        start, end, route_mask, product_mask)
    direct_length = max(float(np.linalg.norm(end - start)), 1.0)
    if straight >= MIN_STRAIGHT_COVERAGE:
        return [start, end], _straight_result(
            straight, tolerance, product, "straight")

    ys, xs = np.nonzero((route_mask > 0) & (product_mask > 0))
    if len(xs) == 0:
        return [start, end], _straight_result(
            straight, tolerance, product, "straight")

    stride = max(len(xs) // 4500, 1)
    candidates = np.column_stack((xs[::stride], ys[::stride])).astype(np.float64)
    margin = direct_length * 0.30
    candidates = candidates[
        (candidates[:, 0] >= min(start[0], end[0]) - margin)
        & (candidates[:, 0] <= max(start[0], end[0]) + margin)
        & (candidates[:, 1] >= min(start[1], end[1]) - margin)
        & (candidates[:, 1] <= max(start[1], end[1]) + margin)
    ]

    best = None
    for bend in candidates:
        first_length = float(np.linalg.norm(bend - start))
        second_length = float(np.linalg.norm(end - bend))
        length_ratio = (first_length + second_length) / direct_length
        if length_ratio > MAX_ROUTE_LENGTH_RATIO:
            continue
        angle = _bend_angle_degrees(start, bend, end)
        if angle > MAX_BEND_ANGLE_DEG:
            continue
        one = _segment_coverage(start, bend, route_mask, product_mask)
        two = _segment_coverage(bend, end, route_mask, product_mask)
        total = max(first_length + second_length, 1.0)
        weighted = [
            (one[i] * first_length + two[i] * second_length) / total for i in range(3)
        ]
        score = weighted[0] - 0.28 * (length_ratio - 1.0) - 0.0015 * angle
        if best is None or score > best[0]:
            best = (score, bend.copy(), {
                "route_type": "one_bend", "bend_count": 1,
                "combined_coverage": round(weighted[0], 6),
                "tolerance_coverage": round(weighted[1], 6),
                "product_coverage": round(weighted[2], 6),
            })

    if best is not None and best[2]["combined_coverage"] >= straight + BEND_IMPROVEMENT:
        return [start, best[1], end], best[2]
    return [start, end], _straight_result(
        straight, tolerance, product, "straight_preferred_over_weak_bend")


def find_simple_zero_lines(
    values: np.ndarray,
    part_mask,
    key_points,
    part_no=None,
    tolerance_mm: float = TOLERANCE_MM,
) -> list:
    """0포인트들을 정렬도로 묶어 직선 영라인을 낸다.

    Args:
        values: 편차값 맵. 정규화된 +-1 이면 품번 컬러바로 mm 로 되돌린다.
        part_mask: 부품 영역.
        key_points: [(x, y, radius_px), ...] 주요 0포인트.
        part_no: 컬러바 범위를 아는 품번(64XX2 / 67XX6 / 71XX2).
        tolerance_mm: 허용범위 반폭(mm).
    """
    points = [
        {"point": np.array([float(x), float(y)]), "radius_px": float(radius)}
        for x, y, radius in key_points
    ]
    if len(points) < 2:
        return []

    millimetres = to_millimetres(values, part_mask, part_no)
    tolerance_mask = build_tolerance_mask(millimetres, part_mask, tolerance_mm)
    product_mask = (np.asarray(part_mask) > 0).astype(np.uint8) * 255
    route_mask = build_route_mask(tolerance_mask, part_mask)

    lines: list = []
    seen: set = set()
    for group in _group_aligned_points(points, route_mask, product_mask):
        start_index, end_index = _group_endpoints(group, points)
        pair = tuple(sorted((start_index, end_index)))
        if start_index == end_index or pair in seen:
            continue
        seen.add(pair)
        vertices, details = _choose_simple_route(
            points[start_index]["point"], points[end_index]["point"],
            route_mask, product_mask,
        )
        length = float(sum(
            np.linalg.norm(b - a) for a, b in zip(vertices[:-1], vertices[1:])
        ))
        lines.append(SimpleZeroLine(
            line_id=0,
            points=[[round(float(x), 1), round(float(y), 1)] for x, y in vertices],
            support_count=len(group),
            length_px=round(length, 1),
            **details,
        ))

    lines.sort(key=lambda line: (-line.support_count, -line.combined_coverage))
    for index, line in enumerate(lines, start=1):
        line.line_id = index
    return lines


__all__ = [
    "TOLERANCE_MM", "PRODUCT_COLORBAR_MM",
    "SimpleZeroLine", "find_simple_zero_lines",
    "build_tolerance_mask", "build_route_mask",
    "colorbar_span_for", "to_millimetres",
]
