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
    key_score: float | None = None  # key_zero_point_engine의 컬러바 실측
                                     # |편차| 평균(mm). 작을수록 진짜 0에
                                     # 가깝다 — 있으면 strength보다 우선
                                     # 신뢰한다(끝점 선택 순위에 사용).
    # 윤곽선 위 시작/끝 위치(sample index). 나중에 이 군집을 존으로
    # 넓힐 때 윤곽선을 다시 잘라내려면 필요하다(expand_clusters_to_zones).
    start_position: float = 0.0
    end_position: float = 0.0

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


def load_key_zero_points(json_path: str | Path) -> set:
    """key_zero_point_engine 이 낸 key_zero_points.json 에서 '핵심'으로
    확인된 점들의 좌표를 읽는다.

    현업 팀이 만든 후처리다(2026-08-25 공유): 0포인트 후보 하나하나를
    반경 원으로 잘라, **컬러바 HSV로 읽은 실제 편차값**의 평균이
    threshold_mm 미만인지 다시 확인한다. 우리 strength(부호 전환
    크기)와는 독립적인 신호다 — strength는 "라벨 두 값이 부호가
    바뀌었는가"만 보는데, 그 자리 실제 색이 여전히 짙으면(진짜 편차가
    크면) 노이즈였다는 뜻이다. JD_64XX2 기준 12개 후보 중 5개만
    "핵심"으로 남았다 — 현업이 지적한 "0포인트가 너무 많이 찍힌다"는
    문제를 색으로 다시 검증해 줄인 것이다.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    keys: set = set()
    for loop in data.get("loops", []):
        for item in loop.get("key_zero_points", []):
            x, y = item["point"]
            keys.add((round(float(x), 1), round(float(y), 1)))
    return keys


def load_key_scores(json_path: str | Path) -> dict:
    """key_zero_point_engine 결과에서 좌표별 컬러바 실측 |편차| 평균(mm)을 읽는다.

    selected 여부와 무관하게 candidates 전체에서 읽는다 — 이미
    filter_to_key_points 로 걸렀으면 살아남은 점만 매칭되고, 안 걸렀으면
    후보 전체가 매칭된다. connect_strongest_pair 에서 strength 대신(혹은
    같이) 끝점 순위를 매기는 데 쓴다 — strength는 "라벨 부호가 바뀌었나"
    만 보는 간접 신호고, 이건 그 자리 실제 색을 다시 잰 직접 신호다.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    scores: dict = {}
    for loop in data.get("loops", []):
        for item in loop.get("candidates", []):
            x, y = item["point"]
            scores[(round(float(x), 1), round(float(y), 1))] = float(
                item["mean_abs_deviation_mm"])
    return scores


def filter_to_key_points(points: list, key_json_path: str | Path) -> list:
    """0포인트 후보를 key_zero_point_engine 이 '핵심'으로 남긴 것만 남긴다.

    좌표로 매칭한다 — key_zero_points.json 에는 sample_index 가 없지만,
    같은 zero_points.json 후보를 그대로 넘겨 계산했으므로 좌표가
    소수점까지 정확히 일치한다.
    """
    keys = load_key_zero_points(key_json_path)
    return [p for p in points if (round(p.x, 1), round(p.y, 1)) in keys]


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


def build_zone_outline(
    path,
    start_position: float,
    end_position: float,
    margin: float,
    thickness: int,
    fallback_xy=None,
    snap_mask=None,
    snap_boundary=None,
):
    """윤곽선 [start-margin, end+margin] 구간을 두껍게 부풀려 존 폴리곤을 만든다.

    점들을 부풀리는 대신 **실제 측정 윤곽선을 따라가는 띠**를 만든다 —
    그래야 보정시트처럼 부품 테두리를 감싸는 모양이 나온다. 점을
    부풀리면 엉뚱한 자리에 타원 얼룩이 생긴다(실측으로 확인).

    snap_mask 를 주면 부품 밖으로 나간 구간을 테두리로 당겨 붙인다
    (snap_segment_to_mask 참고).

    path 가 없으면 fallback_xy 의 볼록껍질로 대체한다.
    """
    outline = None
    if path is not None and len(path) >= 2:
        segment = _contour_segment(
            path, start_position, end_position, margin=margin)
        if snap_mask is not None:
            segment = snap_segment_to_mask(segment, snap_mask, snap_boundary)
        if len(segment) >= 2:
            pad = thickness + 4
            x0 = int(segment[:, 0].min()) - pad
            y0 = int(segment[:, 1].min()) - pad
            x1 = int(segment[:, 0].max()) + pad
            y1 = int(segment[:, 1].max()) + pad
            band = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), np.uint8)
            cv2.polylines(
                band,
                [(segment - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)],
                False, 255, thickness, cv2.LINE_8,
            )
            found, _ = cv2.findContours(
                band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if found:
                outline = max(found, key=cv2.contourArea).reshape(-1, 2) + [x0, y0]
    if outline is None and fallback_xy is not None and len(fallback_xy) > 0:
        outline = cv2.convexHull(
            np.asarray(fallback_xy, np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
    return outline


def mask_boundary_points(mask) -> np.ndarray:
    """부품 마스크의 테두리 픽셀 좌표. 면을 부품에 붙일 때 쓴다."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    found, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not found:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack([c.reshape(-1, 2) for c in found]).astype(np.float32)


def snap_segment_to_mask(segment, mask, boundary=None) -> np.ndarray:
    """윤곽선 구간에서 부품 밖으로 나간 점을 가장 가까운 테두리로 당긴다.

    [왜 필요한가]
    실측(JD_64XX2): my_lab 의 connection_path 가 부품 아래쪽에서 최대
    y=539 까지 처지는데 부품은 y=513 에서 끝난다. 그대로 띠를 그리면
    측정점이 실제로는 테두리 위(y≈507)에 있는데도 흰 배경에 면이
    둥둥 떠 보였다("빨간 면이 부품 밖에 있다"는 피드백).

    안에 있는 점은 그대로 두고, 밖에 있는 점만 당긴다 — 부품 안쪽을
    지나는 구간의 모양은 건드리지 않는다.
    """
    segment = np.asarray(segment, dtype=np.float32)
    if segment.size == 0:
        return segment
    binary = np.asarray(mask) > 0
    height, width = binary.shape[:2]
    xs = np.clip(segment[:, 0].round().astype(int), 0, width - 1)
    ys = np.clip(segment[:, 1].round().astype(int), 0, height - 1)
    outside = ~binary[ys, xs]
    if not outside.any():
        return segment
    if boundary is None:
        boundary = mask_boundary_points(binary)
    if len(boundary) == 0:
        return segment
    snapped = segment.copy()
    targets = segment[outside]
    # 점 수십 개 x 테두리 수천 개라 브로드캐스팅으로 충분히 빠르다
    d2 = ((boundary[None, :, 0] - targets[:, None, 0]) ** 2
          + (boundary[None, :, 1] - targets[:, None, 1]) ** 2)
    snapped[outside] = boundary[np.argmin(d2, axis=1)]
    return snapped


def _rank_key(cluster: "ZeroCluster") -> tuple:
    """군집 순위를 매기는 공통 기준. 낮을수록 더 믿을 만한 끝점 후보.

    실측(JD_64XX2)으로 드러난 문제를 고쳤다: key_score(컬러바 실측)만
    보고 정렬하면, 작업자가 라벨에 **직접 "0.0"이라고 적은 exact 점**
    (strength=EXACT_ZERO_STRENGTH)이 key_score 0.03mm 차이로 밀려나
    엉뚱한 점이 끝점으로 뽑혔다. exact 라벨은 컬러바 재검증보다 더
    직접적인 증거라 항상 최우선이어야 한다.

    우선순위: 1) exact 라벨 2) key_score 낮은 순(0에 가까움)
    3) strength 높은 순.
    """
    is_exact = cluster.strength >= EXACT_ZERO_STRENGTH - 0.5
    return (
        0 if is_exact else (1 if cluster.key_score is not None else 2),
        cluster.key_score if cluster.key_score is not None else 0.0,
        -cluster.strength,
    )


def cluster_zero_points(
    points: list,
    loop_paths: dict | None = None,
    merge_gap: float = 2.5,
    min_strength: float = 0.15,
    max_pixel_gap: float = 120.0,
    zone_margin: float = 0.6,
    zone_thickness: int = 26,
    key_scores: dict | None = None,
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

        key_scores: {(x,y): mean_abs_deviation_mm} — key_zero_point_engine
            이 컬러바로 다시 잰 실측값(load_key_scores). 주면 각 군집에
            key_score(구성원 중 최솟값 = 가장 0에 가까움)를 매겨서
            connect_strongest_pair 가 strength 대신 이걸로 순위를 매길
            수 있게 한다.
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
            member_scores = [
                key_scores[(round(m.x, 1), round(m.y, 1))]
                for m in members
                if key_scores is not None and (round(m.x, 1), round(m.y, 1)) in key_scores
            ]
            key_score = min(member_scores) if member_scores else None
            if len(members) == 1:
                clusters.append(ZeroCluster(
                    cluster_id=0, loop=loop_name, kind="point",
                    center=[round(float(center[0]), 1), round(float(center[1]), 1)],
                    members=xy.round(1).tolist(), contour=[],
                    strength=round(strength, 3), span=round(span, 2),
                    key_score=round(key_score, 5) if key_score is not None else None,
                    start_position=round(float(members[0].position), 3),
                    end_position=round(float(members[-1].position), 3),
                ))
            else:
                # 뭉친 자리는 면으로 잡는다 (현업 제안). 점들을 부풀리지
                # 않고 **윤곽선을 따라가는 구간**을 두껍게 만든다 — 그래야
                # 시트처럼 테두리를 감싸는 모양이 되고, 엉뚱한 자리에
                # 타원 얼룩이 생기지 않는다.
                outline = build_zone_outline(
                    (loop_paths or {}).get(loop_name),
                    members[0].position, members[-1].position,
                    margin=zone_margin, thickness=zone_thickness,
                    fallback_xy=xy,
                )
                clusters.append(ZeroCluster(
                    cluster_id=0, loop=loop_name, kind="zone",
                    center=[round(float(center[0]), 1), round(float(center[1]), 1)],
                    members=xy.round(1).tolist(),
                    contour=np.asarray(outline).astype(int).tolist(),
                    strength=round(strength, 3), span=round(span, 2),
                    key_score=round(key_score, 5) if key_score is not None else None,
                    start_position=round(float(members[0].position), 3),
                    end_position=round(float(members[-1].position), 3),
                ))

    # key_score(컬러바 실측)가 있으면 그게 strength(부호전환 크기)보다
    # 신뢰도 높은 신호다 — 있는 군집을 먼저, 그 안에서는 0에 가까운
    # (key_score 낮은) 순으로. key_score 없는 군집은 기존처럼 strength
    # 로만 뒤에 정렬한다.
    clusters.sort(key=_rank_key)
    for index, cluster in enumerate(clusters, start=1):
        cluster.cluster_id = index
    return clusters


def _mean_sample_spacing(path) -> float:
    """윤곽선 이웃 샘플 사이 평균 픽셀 거리."""
    if path is None or len(path) < 2:
        return 0.0
    pts = np.asarray(path, dtype=np.float64)
    return float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).mean())


def expand_clusters_to_zones(
    clusters: list,
    loop_paths: dict | None = None,
    point_margin_px: float = 70.0,
    zone_thickness: int = 20,
    part_mask=None,
) -> list:
    """모든 군집을 존(면)으로 만든다 — 단독 점도 윤곽선 구간으로 넓힌다.

    [왜 필요한가]
    보정시트를 보면 부품에 따라 제로를 **선 하나**가 아니라 **여러 존**
    으로 표기한다(JD_67XX6 = 존 9개, JD_71XX2 = 존 5개). 그런데 지금
    파이프라인은 형태와 무관하게 항상 "점 2개를 이은 선 하나"만 낸다.
    그래서 존 부품의 정답 커버리지가 나빴다(실측: 67XX6 19.8%,
    71XX2 42.4% — 선 부품 64XX2도 5.3%).

    이 함수는 군집 하나하나를 그 자리 윤곽선 구간을 두껍게 한 존으로
    바꾼다. 단독 점은 앞뒤로 point_margin_px 만큼 넓혀 존을 만든다 —
    시트도 점 하나짜리 자리를 면으로 칠하기 때문이다.

    [왜 픽셀 단위인가]
    처음엔 sample index 단위로 넓혔는데, 윤곽선 점 간격이 부품마다
    크게 달라(실측: 64XX2 48.5px / 67XX6 80.2px / 71XX2 65.1px) 같은
    설정이 부품마다 2배 가까이 다른 크기의 존을 만들었다. 픽셀로
    지정하고 loop 별 평균 간격으로 환산하면 물리적으로 같은 크기가 된다.

    [기본값 근거 — 정답 면적에 맞춤]
    존을 크게 그리면 커버리지는 공짜로 좋아지지만 실무에서 쓸모없는
    안내가 된다. 그래서 **보정시트 정답이 실제로 차지하는 면적**을
    목표로 잡았다(실측: 67XX6 = 부품의 12.32%, 71XX2 = 8.85%).
    margin 70px / 두께 20px 이 그 목표에 가장 가깝다(각 1.06x, 0.88x).

    이 설정에서 선 출력 대비 실측 변화(커버리지 = 정답점→예측 거리
    중앙값, 대각선 대비 %):

        64XX2  커버리지 5.34% -> 1.71%   정밀도 4.81% -> 4.07%  (둘 다 개선)
        67XX6  커버리지 19.82% -> 5.60%  정밀도 3.48% -> 8.90%
        71XX2  커버리지 42.43% -> 18.56% 정밀도 3.90% -> 5.94%

    존 부품 두 개는 정밀도를 내주고 커버리지를 크게 얻는다 — 존이
    실제로 면이라 당연하고, 정답 면적에 맞춘 상태의 값이라 과도하게
    칠해서 얻은 수치가 아니다. 64XX2(선 부품)는 양쪽 다 좋아졌다.

    Args:
        point_margin_px: 단독 점을 존으로 넓힐 때 윤곽선 앞뒤 여유(px).
        zone_thickness: 윤곽선 구간을 부풀릴 두께(px).
        part_mask: 주면 부품 밖으로 나간 구간을 테두리로 당겨 붙인다 —
            my_lab 윤곽선이 부품 아래로 처지는 구간이 있어 그대로 두면
            면이 흰 배경에 떠 보인다(snap_segment_to_mask 참고).
    """
    # 테두리는 한 번만 뽑아 모든 면에서 재사용한다(면마다 다시 뽑으면 느리다)
    boundary = mask_boundary_points(part_mask) if part_mask is not None else None
    expanded: list = []
    for cluster in clusters:
        path = (loop_paths or {}).get(cluster.loop)
        # 픽셀 여유를 이 loop 의 sample 간격으로 나눠 sample 단위로 환산
        spacing = _mean_sample_spacing(path)
        margin_samples = (point_margin_px / spacing) if spacing > 1e-6 else 1.0
        # 단독 점이면 앞뒤로 넓혀서 구간을 만든다. 이미 존이면 원래
        # 구간을 그대로 쓰되 두께만 다시 적용한다.
        if cluster.kind == "point":
            start = cluster.start_position - margin_samples
            end = cluster.end_position + margin_samples
        else:
            start, end = cluster.start_position, cluster.end_position
        outline = build_zone_outline(
            path, start, end, margin=0.0, thickness=zone_thickness,
            fallback_xy=np.asarray(cluster.members, dtype=np.float32),
            snap_mask=part_mask, snap_boundary=boundary,
        )
        if outline is None:
            expanded.append(cluster)
            continue
        expanded.append(ZeroCluster(
            cluster_id=cluster.cluster_id, loop=cluster.loop, kind="zone",
            center=cluster.center, members=cluster.members,
            contour=np.asarray(outline).astype(int).tolist(),
            strength=cluster.strength, span=cluster.span,
            key_score=cluster.key_score,
            start_position=cluster.start_position,
            end_position=cluster.end_position,
        ))
    return expanded


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

    # 부품 마스크 위에 제대로 얹히는 군집만 끝점 후보로 쓴다.
    # 실측(JD_64XX2): 0포인트 6개 중 2개가 부품에서 90~140px 밖이었다
    # (좌하단 곡면이 마스크에서 빠졌거나 스캔포인트 오검출). 그런 점을
    # 억지로 끌어와 이으면 엉뚱한 자리를 지나간다.
    class _Anchor:
        def __init__(self, anchor_id, x, y):
            self.anchor_id, self.x, self.y = anchor_id, x, y

    # 순위는 cluster_zero_points 와 같은 기준(_rank_key)을 쓴다 — exact
    # 라벨 최우선, 그다음 key_score(컬러바 실측), 마지막 strength.
    on_part = []
    for cluster in clusters:
        sx, sy, moved = snap_into_mask(part_mask, cluster.center[0], cluster.center[1])
        if moved <= max_snap_px:
            on_part.append((_rank_key(cluster), sx, sy))
    if len(on_part) < 2:
        return None
    on_part.sort(key=lambda item: item[0])
    anchors = [_Anchor(index, x, y) for index, (_k, x, y) in enumerate(on_part[:2], start=1)]

    lines = find_valley_lines(
        values, part_mask, anchors, tolerance,
        max_quality_ratio=100.0, min_length_px=0.0, max_uses_per_anchor=2,
    )
    return lines[0] if lines else None


__all__ = [
    "ZeroPoint", "ZeroCluster",
    "snap_into_mask", "connect_strongest_pair", "load_loop_paths",
    "load_zero_points", "cluster_zero_points", "draw_zero_clusters",
    "load_key_zero_points", "filter_to_key_points", "load_key_scores",
    "build_zone_outline", "expand_clusters_to_zones",
]
