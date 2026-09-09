"""제로라인을 **실제 측정점 위로만** 잇는다.

[왜 이게 필요한가 — 실측으로 드러난 문제]
기존 경로 탐색(zero_valley.find_valley_lines)은 픽셀 단위 그래프에서
`|편차|` 가 낮은 길을 따라간다. 그런데 그 `|편차|` 는 컬러바 색을 읽은
값이고, **측정점이 없는 자리의 색은 렌더링 보간일 뿐 측정 결과가 아니다.**

실측(JD_64XX2, 측정점 77개):

    우리 선  : 측정점까지 중앙값 43px, **14.2% 구간이 100px 넘게 떨어짐**
    시트 정답: 측정점까지 중앙값 34px, 8.0%

현업 지적도 같았다 — "저 구간엔 포인트도 없고 수치값도 없는데 왜 선을
넣는지". 부품 왼쪽 중앙처럼 측정점이 비어 있는 자리를 우리 선이 지나고
있었다. 색은 초록(편차 낮음)으로 보이지만 그걸 뒷받침하는 측정이 없다.

[여기서 하는 일]
노드를 픽셀이 아니라 **스캔 측정점**으로 놓고 최단경로를 찾는다.

- 노드  : my_lab 이 잡은 스캔포인트(윤곽선 샘플) 전부
- 간선  : 서로 가까운 점끼리. 단, 두 점을 잇는 직선이 부품 안에 있어야
          한다(개구부를 가로지르지 않게).
- 비용  : 두 점 사이 거리. 거리로만 재므로 결과가 자연히 곧은 선분이 된다.

이렇게 하면 선이 지나는 자리는 전부 실측이 있는 곳이고, 시트처럼
꺾인 점이 적은 직선 구간이 나온다.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# 이 거리 안의 측정점끼리만 이을 수 있다. 스캔포인트 평균 간격이
# 부품마다 48~84px 이라(실측) 그 2~3배면 이웃을 넉넉히 잡는다.
DEFAULT_MAX_EDGE_PX = 220.0
# 간선이 부품 안에 있는지 확인할 때 몇 점을 찍어볼지
EDGE_SAMPLES = 12
# 간선의 이 비율 이상이 부품 안에 있어야 쓴다(테두리를 스치는 건 허용)
EDGE_MIN_INSIDE = 0.85
# 부품 안/밖 판정을 할 때 마스크를 이만큼 넓힌다. 스캔 측정점은 부품
# 테두리에 걸치거나 살짝 밖에 찍혀서(실측 JD_64XX2: 77개 중 36%만
# 마스크 안) 원본 마스크로 재면 멀쩡한 간선이 거의 다 걸러진다.
# 개구부는 수백 px 크기라 이 정도 확장으로는 안 뚫린다.
MASK_DILATE_PX = 30


def collect_measurement_points(loop_paths: dict) -> np.ndarray:
    """모든 윤곽선의 스캔포인트를 한 배열로 모은다."""
    if not loop_paths:
        return np.empty((0, 2), dtype=float)
    return np.vstack([np.asarray(p, dtype=float) for p in loop_paths.values()])


def _segment_inside_ratio(a, b, part_mask) -> float:
    """두 점을 잇는 직선이 부품 마스크 안에 있는 비율."""
    height, width = part_mask.shape[:2]
    t = np.linspace(0.0, 1.0, EDGE_SAMPLES)
    pts = a[None, :] * (1 - t)[:, None] + b[None, :] * t[:, None]
    xs = np.clip(pts[:, 0].round().astype(int), 0, width - 1)
    ys = np.clip(pts[:, 1].round().astype(int), 0, height - 1)
    return float((part_mask[ys, xs] > 0).mean())


def dilated_mask(part_mask, dilate_px: int = MASK_DILATE_PX):
    """측정점이 테두리에 걸치는 걸 감안해 넓힌 부품 마스크."""
    import cv2

    binary = (np.asarray(part_mask) > 0).astype(np.uint8)
    if dilate_px <= 0:
        return binary > 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
    return cv2.dilate(binary, kernel) > 0


def build_measurement_graph(
    points: np.ndarray,
    part_mask,
    max_edge_px: float = DEFAULT_MAX_EDGE_PX,
):
    """측정점 사이 간선을 만든다. 개구부를 가로지르는 간선은 뺀다."""
    n = len(points)
    if n < 2:
        return csr_matrix((n, n))
    part_mask = dilated_mask(part_mask)
    d = np.hypot(
        points[:, None, 0] - points[None, :, 0],
        points[:, None, 1] - points[None, :, 1],
    )
    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            dist = d[i, j]
            if dist <= 1e-6 or dist > max_edge_px:
                continue
            if _segment_inside_ratio(points[i], points[j], part_mask) < EDGE_MIN_INSIDE:
                continue  # 개구부/부품 밖을 가로지르는 연결
            rows.append(i); cols.append(j); data.append(dist)
            rows.append(j); cols.append(i); data.append(dist)
    if not rows:
        return csr_matrix((n, n))
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _nearest_index(points: np.ndarray, xy) -> int:
    return int(np.argmin(np.hypot(points[:, 0] - xy[0], points[:, 1] - xy[1])))


def connect_along_measurements(
    start_xy,
    end_xy,
    loop_paths: dict,
    part_mask,
    max_edge_px: float = DEFAULT_MAX_EDGE_PX,
):
    """두 0포인트를 실제 측정점만 거쳐 잇는다. 못 이으면 None.

    Returns:
        (points, info) — points 는 [[x, y], ...] 폴리라인,
        info 는 경로 길이/경유 측정점 수 등.
    """
    points = collect_measurement_points(loop_paths)
    if len(points) < 2:
        return None

    graph = build_measurement_graph(points, part_mask, max_edge_px)
    si = _nearest_index(points, start_xy)
    ei = _nearest_index(points, end_xy)
    if si == ei:
        return None

    dist, pred = dijkstra(graph, indices=si, return_predecessors=True)
    if not np.isfinite(dist[ei]):
        return None

    chain = [ei]
    while chain[-1] != si:
        prev = pred[chain[-1]]
        if prev < 0:
            return None
        chain.append(int(prev))
    chain.reverse()

    path = points[chain]
    length = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    return path, {
        "n_measurement_points": int(len(chain)),
        "length_px": round(length, 1),
        "start_snap_px": round(float(np.hypot(*(points[si] - np.asarray(start_xy, float)))), 1),
        "end_snap_px": round(float(np.hypot(*(points[ei] - np.asarray(end_xy, float)))), 1),
    }


__all__ = [
    "DEFAULT_MAX_EDGE_PX",
    "collect_measurement_points",
    "build_measurement_graph",
    "connect_along_measurements",
]
