"""라벨 실측값으로 찾은 0포인트를 묶어서 제로라인(점/존)으로 만든다.

[입력 — my_lab 파이프라인]
scan_point_contour 가 스캔포인트를 제품별 닫힌 윤곽선으로 잇고,
contour_graph 가 윤곽선을 따라 라벨 편차값을 배열하면,
zero_point_selection 이 부호가 바뀌는 구간을 선형 보간해 0포인트를 찍는다.

이 방식이 픽셀 색을 보는 것보다 낫다. 0포인트가 **작업자가 실제로 측정한
값** 사이에서만 나오므로, 리브·나사구멍 음영 같은 색 잡음에 흔들리지 않는다.

[여기서 하는 일 — 왜 필요한가]
현업 확인(2026-08-25): JD_64(대시보드)는 0포인트가 너무 많이 찍힌다.
가까이 붙은 점들은 편차가 +,- 로 계속 왔다갔다해서 생긴 것이라 하나로
봐야 한다. 다만 JD_67(썬루프)에서 가까운 점들은 **서로 다른 윤곽선**
위의 점이라 절대 합치면 안 된다.

그래서 거리로 묶지 않고 **같은 윤곽선 위의 위치(sample index)** 로 묶는다.
윤곽선이 다르면 좌표가 아무리 가까워도 별개로 남는다.

점이 여러 개 뭉친 자리는 점 하나로 찍기보다 **존(면)** 으로 잡는다.
실제 보정시트도 그렇게 표기한다 — JD_67XX6 시트는 우상단 범례에
`"0" 라인 = 빨간 점선 + 살몬 채움` 이라 적고 부품 둘레를 9개 존으로
나눠 칠해뒀다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np


# 라벨에 0.0 으로 찍힌 점은 작업자가 명시한 것이라 가장 확실하다.
EXACT_ZERO_STRENGTH = 9.9


@dataclass
class ZeroPoint:
    """윤곽선 위에서 편차가 0이 되는 지점 하나."""

    loop: str
    position: float          # 윤곽선 위 위치 (sample index, 보간 포함)
    x: float
    y: float
    kind: str                # "exact" | "interpolated"
    strength: float          # 부호 전환의 크기. 작으면 측정 노이즈일 수 있다
    values: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ZeroCluster:
    """같은 윤곽선에서 인접한 0포인트들을 하나로 묶은 것."""

    cluster_id: int
    loop: str
    kind: str                # "point"(단독) | "zone"(여러 개 뭉침)
    center: list             # [x, y]
    members: list            # [[x, y], ...]
    contour: list            # zone 일 때 채울 폴리곤. point 면 빈 리스트
    strength: float          # 구성원 중 가장 확실한 전환 크기
    span: float              # 윤곽선 위에서 차지한 길이 (sample 단위)

    def to_dict(self) -> dict:
        return asdict(self)


def load_loop_paths(loops_json: str | Path) -> dict:
    """scan_point_contour 가 낸 scan_point_loops.json 에서 윤곽선 좌표를 읽는다.

    zero_points.json 의 sample_index 는 이 connection_path 의 인덱스다.
    존을 그릴 때 이 경로를 따라가야 시트처럼 테두리를 감싸는 모양이 된다
    (점들을 부풀린 타원 덩어리로 그리면 엉뚱한 자리에 얼룩이 생긴다).
    """
    data = json.loads(Path(loops_json).read_text(encoding="utf-8"))
    paths: dict = {}
    for index, loop in enumerate(data.get("loops", [])):
        name = loop.get("name") or f"loop_{index}"
        path = loop.get("connection_path") or loop.get("points") or []
        if path:
            paths[name] = np.asarray(path, dtype=np.float32)
    return paths


def load_zero_points(json_path: str | Path) -> list:
    """zero_point_selection 이 낸 zero_points.json 을 읽는다."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    points: list = []
    for index, loop in enumerate(data.get("loops", [])):
        name = loop.get("name") or f"loop_{index}"
        for entry in loop.get("zero_points", []):
            x, y = entry["point"]
            if entry["type"] == "exact_label_zero":
                points.append(ZeroPoint(
                    loop=name, position=float(entry["sample_index"]),
                    x=float(x), y=float(y), kind="exact",
                    strength=EXACT_ZERO_STRENGTH,
                    values=[float(entry.get("value_mm", 0.0))],
                ))
            else:
                first, _second = entry["between_sample_indices"]
                ratio = float(entry.get("interpolation_ratio", 0.5))
                values = [float(v) for v in entry.get("between_values_mm", [])]
                # 양쪽 값이 둘 다 뚜렷해야 진짜 전환이다. 한쪽이 0 근처면
                # 측정 오차만으로도 부호가 뒤집힐 수 있다.
                strength = float(min(abs(v) for v in values)) if values else 0.0
                points.append(ZeroPoint(
                    loop=name, position=float(first) + ratio,
                    x=float(x), y=float(y), kind="interpolated",
                    strength=strength, values=values,
                ))
    return points


def _contour_segment(
    path: np.ndarray, start: float, end: float, margin: float, closed: bool = True
) -> np.ndarray:
    """윤곽선 경로에서 [start-margin, end+margin] 구간을 잘라낸다.

    인덱스는 실수라 양 끝은 이웃 점 사이를 선형 보간해 정확히 맞춘다.
    닫힌 경로면 끝을 넘어가는 구간이 처음으로 이어진다.
    """
    n = len(path)
    if n < 2:
        return path
    lo, hi = start - margin, end + margin

    def at(position: float) -> np.ndarray:
        base = int(np.floor(position)) % n
        nxt = (base + 1) % n
        frac = float(position - np.floor(position))
        return path[base] * (1.0 - frac) + path[nxt] * frac

    span = hi - lo
    if not closed:
        lo, hi = max(lo, 0.0), min(hi, n - 1.0)
        span = hi - lo
    if span <= 0:
        return np.asarray([at(lo)], dtype=np.float32)

    steps = [lo]
    cursor = float(np.floor(lo)) + 1.0
    while cursor < hi:
        steps.append(cursor)
        cursor += 1.0
    steps.append(hi)
    return np.asarray([at(v) for v in steps], dtype=np.float32)


def cluster_zero_points(
    points: list,
    loop_paths: dict | None = None,
    merge_gap: float = 2.5,
    min_strength: float = 0.15,
    max_pixel_gap: float = 120.0,
    zone_margin: float = 0.6,
    zone_thickness: int = 26,
) -> list:
    """같은 윤곽선에서 인접한 0포인트를 묶는다. 윤곽선이 다르면 절대 안 묶는다.

    Args:
        merge_gap: 윤곽선 위에서 이만큼(샘플 수) 이내면 같은 자리로 본다.
            편차가 +,- 로 흔들려 연달아 찍힌 점들을 하나로 만든다.
        min_strength: 부호 전환 크기가 이보다 작으면 버린다. 0 이면 안 버린다.
            양쪽 값이 둘 다 0 근처면(예: [0.2, -0.1]) 측정 오차만으로도
            부호가 뒤집히므로 진짜 전환으로 보기 어렵다.
        max_pixel_gap: 윤곽선 위에서 붙어 있어도 화면상 이보다 멀면 안 합친다.
        zone_margin: 존을 윤곽선 방향으로 앞뒤 이만큼(샘플) 더 넓힌다.
        zone_thickness: 윤곽선 구간을 이 두께(px)로 부풀려 면으로 만든다.

    기본값 merge_gap=2.5, min_strength=0.15 는 JD_64XX2 실측 보정시트에
    맞춰 고른 값이다(군집->정답 3.4%, 정답->군집 5.6% 로 양방향 최적).
    """
    kept = [p for p in points if p.strength >= min_strength]
    clusters: list = []

    by_loop: dict = {}
    for point in kept:
        by_loop.setdefault(point.loop, []).append(point)

    def close_enough(first: ZeroPoint, second: ZeroPoint) -> bool:
        """윤곽선 위에서도 붙어 있고, 화면상으로도 가까워야 같은 자리로 본다.

        윤곽선 위치만 보면 위험하다 — 폐곡선의 끝과 처음은 인덱스가 멀어도
        이어져 있고, 반대로 이어져 있다고 해서 같은 자리인 것도 아니다.
        실측(JD_71XX2): 부품 양 끝의 "0.0" 라벨 두 개가 220px 떨어져
        있는데 폐곡선 처리 때문에 하나로 합쳐졌었다. 그 둘은 제로라인의
        시작점과 끝점이라 절대 합치면 안 된다.
        """
        if abs(second.position - first.position) > merge_gap:
            return False
        return float(np.hypot(second.x - first.x, second.y - first.y)) <= max_pixel_gap

    for loop_name, group in by_loop.items():
        group.sort(key=lambda p: p.position)
        runs: list = [[group[0]]] if group else []
        for point in group[1:]:
            if close_enough(runs[-1][-1], point):
                runs[-1].append(point)
            else:
                runs.append([point])

        # 닫힌 윤곽선이라 마지막과 처음이 이어질 수 있다. 다만 화면상으로도
        # 가까울 때만 합친다 (위 close_enough 의 이유와 같다).
        if len(runs) > 2 and float(np.hypot(
            runs[0][0].x - runs[-1][-1].x, runs[0][0].y - runs[-1][-1].y
        )) <= max_pixel_gap:
            runs[0] = runs[-1] + runs[0]
            runs.pop()

        for members in runs:
            xy = np.array([[m.x, m.y] for m in members], dtype=np.float32)
            center = xy.mean(axis=0)
            span = float(members[-1].position - members[0].position)
            strength = float(max(m.strength for m in members))
            if len(members) == 1:
                clusters.append(ZeroCluster(
                    cluster_id=0, loop=loop_name, kind="point",
                    center=[round(float(center[0]), 1), round(float(center[1]), 1)],
                    members=xy.round(1).tolist(), contour=[],
                    strength=round(strength, 3), span=round(span, 2),
                ))
            else:
                # 뭉친 자리는 면으로 잡는다 (현업 제안). 점들을 부풀리지
                # 않고 **윤곽선을 따라가는 구간**을 두껍게 만든다 — 그래야
                # 시트처럼 테두리를 감싸는 모양이 되고, 엉뚱한 자리에
                # 타원 얼룩이 생기지 않는다.
                path = (loop_paths or {}).get(loop_name)
                outline = None
                if path is not None and len(path) >= 2:
                    segment = _contour_segment(
                        path, members[0].position, members[-1].position,
                        margin=zone_margin,
                    )
                    if len(segment) >= 2:
                        band = np.zeros((2, 2), np.uint8)
                        pad = zone_thickness + 4
                        x0 = int(segment[:, 0].min()) - pad
                        y0 = int(segment[:, 1].min()) - pad
                        x1 = int(segment[:, 0].max()) + pad
                        y1 = int(segment[:, 1].max()) + pad
                        band = np.zeros(
                            (max(y1 - y0, 1), max(x1 - x0, 1)), np.uint8)
                        cv2.polylines(
                            band,
                            [(segment - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)],
                            False, 255, zone_thickness, cv2.LINE_8,
                        )
                        found, _ = cv2.findContours(
                            band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if found:
                            outline = (
                                max(found, key=cv2.contourArea).reshape(-1, 2)
                                + [x0, y0]
                            )
                if outline is None:
                    # 경로가 없으면 최소한 구성원을 감싸는 볼록껍질로 대체
                    outline = cv2.convexHull(
                        xy.reshape(-1, 1, 2)).reshape(-1, 2)
                clusters.append(ZeroCluster(
                    cluster_id=0, loop=loop_name, kind="zone",
                    center=[round(float(center[0]), 1), round(float(center[1]), 1)],
                    members=xy.round(1).tolist(),
                    contour=np.asarray(outline).astype(int).tolist(),
                    strength=round(strength, 3), span=round(span, 2),
                ))

    clusters.sort(key=lambda c: -c.strength)
    for index, cluster in enumerate(clusters, start=1):
        cluster.cluster_id = index
    return clusters


def draw_zero_clusters(
    rgb: np.ndarray,
    clusters: list,
    point_color: tuple = (0, 200, 0),
    zone_edge: tuple = (220, 20, 20),
    zone_fill: tuple = (255, 120, 120),
    fill_alpha: float = 0.30,
) -> np.ndarray:
    """존은 살몬색 면으로, 단독 점은 초록 점으로 그린다(시트 표기와 맞춤)."""
    out = rgb.copy()
    zones = [c for c in clusters if c.kind == "zone" and len(c.contour) >= 3]
    if zones:
        overlay = out.copy()
        for cluster in zones:
            cv2.fillPoly(overlay, [np.asarray(cluster.contour, np.int32)], zone_fill)
        out = cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0)
        for cluster in zones:
            cv2.polylines(
                out, [np.asarray(cluster.contour, np.int32)], True, zone_edge, 2,
                cv2.LINE_AA)
    for cluster in clusters:
        if cluster.kind != "point":
            continue
        centre = (int(cluster.center[0]), int(cluster.center[1]))
        cv2.circle(out, centre, 7, point_color, -1, cv2.LINE_AA)
        cv2.circle(out, centre, 7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def snap_into_mask(mask: np.ndarray, x: float, y: float) -> tuple:
    """좌표를 부품 마스크 안의 가장 가까운 픽셀로 옮긴다.

    0포인트는 라벨 지시선 끝이라 부품 테두리에 딱 걸치거나 살짝 밖에
    있는 경우가 있다. 경로 탐색은 마스크 안에서만 되므로 먼저 넣어준다.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return int(x), int(y), float("inf")
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    k = int(np.argmin(d2))
    return int(xs[k]), int(ys[k]), float(np.sqrt(d2[k]))


def connect_strongest_pair(
    clusters: list,
    values: np.ndarray,
    part_mask: np.ndarray,
    tolerance: float,
    max_snap_px: float = 60.0,
):
    """가장 확실한 0포인트 두 개를 편차가 낮은 경로로 잇는다.

    선으로 표기하는 부품(예: DASH UPR)의 제로라인을 만든다. 끝점은
    컬러바에서 추정한 것이 아니라 **작업자가 실제로 측정한 라벨**에서
    나온 0포인트라, 색 잡음에 흔들리지 않는다.

    양 끝이 마스크에서 max_snap_px 보다 멀면 신뢰할 수 없으므로 만들지
    않는다 — 억지로 이으면 엉뚱한 자리를 지나간다.
    """
    from zero_line_detection.zero_valley import find_valley_lines

    ranked = sorted(clusters, key=lambda c: -c.strength)[:2]
    if len(ranked) < 2:
        return None

    class _Anchor:
        def __init__(self, anchor_id, x, y):
            self.anchor_id, self.x, self.y = anchor_id, x, y

    anchors = []
    for index, cluster in enumerate(ranked, start=1):
        sx, sy, moved = snap_into_mask(part_mask, cluster.center[0], cluster.center[1])
        if moved > max_snap_px:
            return None
        anchors.append(_Anchor(index, sx, sy))

    lines = find_valley_lines(
        values, part_mask, anchors, tolerance,
        max_quality_ratio=100.0, min_length_px=0.0, max_uses_per_anchor=2,
    )
    return lines[0] if lines else None


__all__ = [
    "ZeroPoint", "ZeroCluster",
    "snap_into_mask", "connect_strongest_pair", "load_loop_paths",
    "load_zero_points", "cluster_zero_points", "draw_zero_clusters",
]
