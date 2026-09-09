"""제로라인을 '선'으로 표현하는 부품을 위한 두 번째 방식.

[배경]
find_boundary_anchors() 로 찾은 앵커(부품 테두리에서 부호가 바뀌는 지점)는
RING SUNROOF, DASH UPR 두 부품 모두에서 실측 보정시트와 잘 맞는 것으로
검증됐다. 문제는 그다음이었다 — 앵커와 앵커 사이를 어떻게 실제 시트처럼
하나의 선으로 이어줄 것인가.

처음 시도한 방법(zero_polyline.py 로 스캔 전체에서 미리 뽑아둔 조각들을
앵커 근처에서 필터링)은 실패했다. DASH UPR 에서 조각 12개 중 양쪽 끝이
모두 앵커 30px 이내인 것이 하나도 없었다 — 스켈레톤이 51개 조각으로
끊겨 있어서, 조각을 이어붙여도(모폴로지 closing 21px 까지 시도) 앵커
2개가 걸리는 조각이 없었다.

[이번 방식 — 앵커 사이 최단경로]
조각을 미리 뽑아서 필터링하는 대신, 앵커 A 에서 앵커 B 까지 **직접**
최단경로를 찾는다. 비용을 |편차| 로 두면(0 에 가까울수록 싸다) 다익스트라
최단경로가 자연히 "편차가 0 에 가까운 골짜기" 를 따라간다.

DASH UPR(JD_64XX2) 좌하단 앵커 → 우측 앵커로 이 방식을 적용했더니:
  - 경로가 좌측 테두리를 타고 올라가 상단을 가로지르고 우측으로
    대각선으로 내려오는, 실제 보정시트의 "0" LINE 모양과 위상이 일치
  - 정량 비교(실제 시트의 빨간 곡선 픽셀을 그대로 뽑아 부품 bbox 기준
    정규화 좌표로 변환 후 대칭 최근접 거리 계산): 평균 오차 대각선의
    3.68%, 최대 11.42% — 패치 방식(RING SUNROOF, 평균 7.2%)보다도 나음.

[한계]
- 검증은 DASH UPR 한 건뿐이다. RING SUNROOF 는 애초에 면(패치) 방식이라
  이 선 방식으로 검증할 대상이 없다.
- 어떤 앵커 쌍을 이어야 하는지는 여전히 휴리스틱(경로의 평균 |편차| 가
  낮고, 충분히 긴 것만 채택)이다. 사람이 최종 확인해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class ZeroValleyLine:
    """두 앵커 사이를, |편차| 가 0 에 가까운 경로로 이은 선."""

    line_id: int
    anchor_start_id: int
    anchor_end_id: int
    points: list              # [[x, y], ...] 단순화된 경로
    length_px: float
    mean_abs_deviation: float

    def to_dict(self) -> dict:
        return asdict(self)


def _build_graph(values: np.ndarray, part_mask: np.ndarray, smooth_sigma: float,
                 length_cost: float = 0.02):
    vs = cv2.GaussianBlur(values, (0, 0), smooth_sigma)
    absv = np.abs(vs)
    h, w = values.shape
    idx = -np.ones((h, w), dtype=np.int64)
    ys, xs = np.nonzero(part_mask)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)
    cost = absv[ys, xs].astype(np.float64)

    # 픽셀 한 걸음의 고정 비용. |편차|(보통 0.2~1.5mm)에 비해 너무 작으면
    # "편차 낮은 곳으로 한참 돌아가는" 경로가 싸게 먹혀서, 정답선이 개구부
    # 사이를 곧장 가로지르는데 우리 선은 부품 가장자리를 크게 우회했다
    # (JD_64XX2 실측). 이 값을 올리면 곧게 가고 선도 덜 흐물거린다.
    eps = float(length_cost)
    rows, cols, data = [], [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ny, nx = ys + dy, xs + dx
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ny2, nx2 = ny[valid], nx[valid]
        src_idx = idx[ys[valid], xs[valid]]
        dst_idx = idx[ny2, nx2]
        ok = dst_idx >= 0
        src_idx, dst_idx = src_idx[ok], dst_idx[ok]
        weight = (cost[src_idx] + cost[dst_idx]) / 2 + eps
        rows.append(src_idx); cols.append(dst_idx); data.append(weight)
        rows.append(dst_idx); cols.append(src_idx); data.append(weight)

    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    return graph, idx, xs, ys, vs


@dataclass
class ZeroLineCandidate:
    """앵커 쌍 하나로 만든 제로라인 후보와 그 근거 점수."""

    rank: int
    anchor_start_id: int
    anchor_end_id: int
    points: list
    length_px: float
    mean_abs_deviation: float
    separation: float      # 선 양쪽 영역의 평균 편차 차이 (클수록 진짜 경계다움)
    balance: float         # 양쪽 영역 크기 비 (작으면 얇게 잘라낸 가짜)
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def rank_zero_line_candidates(
    values: np.ndarray,
    part_mask: np.ndarray,
    anchors: list,
    top_n: int = 4,
    min_balance: float = 0.30,
    smooth_sigma: float = 15.0,
    simplify_eps: float = 2.0,
) -> list:
    """모든 앵커 쌍의 경로를 만들고 "진짜 보정 경계다움"으로 순위를 매긴다.

    [왜 이렇게 하나 — 2026-08-24 실측으로 알아낸 것]
    실제 보정시트의 정답선 위에서 스캔 편차값을 직접 재봤다:
      - |편차| 평균 0.273 (허용오차 0.15가 아니다), 90%가 ±0.5 안
      - 선 위 양수 비율 47.7% -> 부호가 균형을 이루는 경계가 맞다
      - 최단경로가 아니다. 개구부 위로 일부러 크게 우회한다
    즉 "편차가 정확히 0인 등고선"이 아니라 "±0.5 밴드 안에서 부품을
    의미 있게 둘로 가르는 경계"다. 그래서 경로 자체의 |편차|만 보지 않고,
    그 선으로 부품을 잘랐을 때 양쪽이 실제로 갈리는지(separation)를 본다.

    [점수 — balance 는 곱하지 말고 걸러내는 데만 쓴다]
    separation(양쪽 평균 편차 차이)에 balance(양쪽 크기 비)를 곱해봤더니
    오히려 나빠졌다(실측: 정답이 16개 중 9위). balance 를 곱하면 "부품을
    반반으로 가르기만 하면 점수가 오르는" 쏠림이 생긴다. balance 는
    min_balance 로 얇은 가짜만 걸러내고, 순위는 separation 으로만 매긴다
    (같은 실측에서 정답이 3·4위로 올라옴).

    [한계 — 정직하게]
    이 점수로도 정답 쌍이 1등은 아니다. 실측(JD_64XX2, 앵커 8개 = 28쌍):
    얇은 가짜를 거른 16개 중 정답 후보 두 개가 3위·4위였다. 스캔만으로는
    여러 후보가 물리적으로 똑같이 그럴듯하다 — 실제로 그 부품 시트의 한쪽
    "0" 표시는 작업자가 판단으로 추가한 것이라 스캔에 아예 없는 정보다.
    그래서 하나로 확정하지 않고 상위 후보를 순위대로 내보내 사람이 고르게
    한다(회의록 "AI 제안 -> 작업자 수정" 방향).
    """
    if len(anchors) < 2:
        return []

    graph, idx, xs, ys, vs = _build_graph(values, part_mask, smooth_sigma)
    h, w = values.shape
    part_area = float(part_mask.sum())
    scored: list = []

    for i, a in enumerate(anchors):
        start = idx[a.y, a.x]
        if start < 0:
            continue
        dist, pred = dijkstra(graph, indices=start, return_predecessors=True)
        for b in anchors[i + 1:]:
            end = idx[b.y, b.x]
            if end < 0 or not np.isfinite(dist[end]):
                continue
            path = []
            cur = end
            while cur != -9999 and cur != start:
                path.append(cur)
                cur = pred[cur]
            if cur != start:
                continue
            path.append(start)
            path.reverse()
            pts = np.array([(int(xs[k]), int(ys[k])) for k in path])
            if len(pts) < 2:
                continue

            # 이 선으로 부품을 자르면 양쪽이 실제로 갈리는가
            cut = np.zeros((h, w), np.uint8)
            cv2.polylines(cut, [pts.reshape(-1, 1, 2)], False, 255, 7)
            split = part_mask.copy()
            split[cut > 0] = False
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                split.astype(np.uint8), connectivity=8
            )
            biggest = sorted(range(1, count), key=lambda k: -stats[k, cv2.CC_STAT_AREA])[:2]
            if len(biggest) < 2:
                continue
            area_a = float(stats[biggest[0], cv2.CC_STAT_AREA])
            area_b = float(stats[biggest[1], cv2.CC_STAT_AREA])
            if area_b < part_area * 0.03:
                continue
            balance = area_b / area_a
            if balance < min_balance:
                # 얇게 귀퉁이만 잘라낸 가짜 경계
                continue
            side_a = values[labels == biggest[0]]
            side_b = values[labels == biggest[1]]
            separation = abs(float(side_a.mean()) - float(side_b.mean()))

            path_vals = np.abs(vs[pts[:, 1], pts[:, 0]])
            length_px = float(np.hypot(*np.diff(pts, axis=0).T).sum())
            simplified = cv2.approxPolyDP(
                pts.reshape(-1, 1, 2).astype(np.int32), simplify_eps, False
            ).reshape(-1, 2)

            scored.append(ZeroLineCandidate(
                rank=0,
                anchor_start_id=a.anchor_id,
                anchor_end_id=b.anchor_id,
                points=simplified.tolist(),
                length_px=round(length_px, 1),
                mean_abs_deviation=round(float(path_vals.mean()), 4),
                separation=round(separation, 4),
                balance=round(balance, 3),
                score=round(separation, 4),
            ))

    scored.sort(key=lambda c: -c.score)
    top = scored[:top_n]
    for rank, c in enumerate(top, start=1):
        c.rank = rank
    return top


def find_valley_lines(
    values: np.ndarray,
    part_mask: np.ndarray,
    anchors: list,
    tolerance: float,
    max_quality_ratio: float = 2.0,
    min_length_px: float = 150.0,
    max_uses_per_anchor: int = 2,
    smooth_sigma: float = 15.0,
    simplify_eps: float = 12.0,
    length_cost: float = 0.02,
) -> list:
    """앵커들 사이를 |편차|가 낮은 경로(다익스트라 최단경로)로 잇는다.

    Args:
        max_quality_ratio: 경로의 평균 |편차| 가 tolerance 의 이 배수를
            넘으면 버린다 (너무 색이 진한 곳을 가로지르는 억지 경로 제외).
        min_length_px: 이보다 짧은 경로는 버린다 (인접 앵커끼리의 사소한
            연결 제외).
        max_uses_per_anchor: 한 앵커가 몇 개의 선에 양 끝으로 쓰일 수
            있는지 (조합 폭증 방지 — 앵커 하나는 보통 이웃 앵커 2개와만
            이어진다).
        length_cost: 픽셀 한 걸음의 고정 비용. 올리면 경로가 곧아지지만
            정확도는 나빠진다 — 실측(JD_64XX2)으로 0.08 이상에서
            정답->선 3.46% -> 12.54% 로 악화돼 0.02 를 유지한다.
            "곧은 선"은 simplify_eps 로 얻는 게 맞다.
        simplify_eps: 결과 폴리라인 단순화 강도(px). 클수록 꺾인 점이
            줄어 시트처럼 깔끔한 직선에 가까워진다. 실측: 2.0 -> 12.0 에서
            꼭짓점이 53->19(64XX2), 68->17(67XX6)로 줄었는데 정확도는
            그대로거나 오히려 소폭 좋아졌다.
    """
    if len(anchors) < 2:
        return []

    graph, idx, xs, ys, vs = _build_graph(
        values, part_mask, smooth_sigma, length_cost)

    candidates = []
    n_anchors = len(anchors)
    for si in range(n_anchors):
        a = anchors[si]
        s_idx = idx[a.y, a.x]
        if s_idx < 0:
            continue
        dist, pred = dijkstra(graph, indices=s_idx, return_predecessors=True)
        for ti in range(si + 1, n_anchors):
            b = anchors[ti]
            t_idx = idx[b.y, b.x]
            if t_idx < 0 or not np.isfinite(dist[t_idx]):
                continue
            path = []
            cur = t_idx
            while cur != -9999 and cur != s_idx:
                path.append(cur)
                cur = pred[cur]
            if cur != s_idx:
                continue
            path.append(s_idx)
            path.reverse()
            pts = np.array([(int(xs[i]), int(ys[i])) for i in path])
            length_px = float(np.hypot(*np.diff(pts, axis=0).T).sum())
            if length_px < min_length_px:
                continue
            vals = np.abs(vs[pts[:, 1], pts[:, 0]])
            mad = float(vals.mean())
            if mad > tolerance * max_quality_ratio:
                continue
            candidates.append((mad, a, b, pts, length_px))

    candidates.sort(key=lambda c: c[0])

    uses = {}
    lines = []
    for mad, a, b, pts, length_px in candidates:
        if uses.get(a.anchor_id, 0) >= max_uses_per_anchor:
            continue
        if uses.get(b.anchor_id, 0) >= max_uses_per_anchor:
            continue
        simplified = cv2.approxPolyDP(
            pts.reshape(-1, 1, 2).astype(np.int32), simplify_eps, False
        ).reshape(-1, 2)
        lines.append(ZeroValleyLine(
            line_id=0,
            anchor_start_id=a.anchor_id,
            anchor_end_id=b.anchor_id,
            points=simplified.tolist(),
            length_px=round(length_px, 1),
            mean_abs_deviation=round(mad, 4),
        ))
        uses[a.anchor_id] = uses.get(a.anchor_id, 0) + 1
        uses[b.anchor_id] = uses.get(b.anchor_id, 0) + 1

    for i, l in enumerate(lines, start=1):
        l.line_id = i
    return lines


def draw_zero_valley(
    rgb: np.ndarray,
    lines: list,
    line_color: tuple = (220, 20, 20),
    endpoint_color: tuple = (0, 200, 0),
    thickness: int = 3,
) -> np.ndarray:
    """선(빨강)과 양 끝점(초록)을 그린다. 시트의 '0 LINE' 표기 스타일."""
    out = rgb.copy()
    for l in lines:
        pts = np.asarray(l.points, dtype=np.int32)
        if len(pts) < 2:
            continue
        cv2.polylines(out, [pts], False, line_color, thickness, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[0]), 7, endpoint_color, -1, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[0]), 7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[-1]), 7, endpoint_color, -1, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[-1]), 7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


__all__ = [
    "ZeroValleyLine", "ZeroLineCandidate",
    "find_valley_lines", "rank_zero_line_candidates", "draw_zero_valley",
]
