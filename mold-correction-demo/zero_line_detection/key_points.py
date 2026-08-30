"""편차 포인트 중 보정시트에 적을 것만 고른다.

[왜 골라야 하나]
스캔에는 편차 포인트가 수십~백여 개 찍힌다(실측 JD_67XX6 은 130개).
그런데 현업 보정시트에 적히는 것은 열몇 개뿐이다. 전부 적으면 시트가
읽히지 않고, 3D 에 올리면 형상이 콜아웃에 덮인다.

향후 계획 02번 "핵심 포인트 선별 기준 확정" 이 이것이고, 아직 현업에서
기준을 받지 못했다. 그래서 **설명 가능한 규칙**으로 첫 판을 만든다.
기준이 오면 갈아 끼우면 된다.

[무엇을 고르나 — 시트를 보고 세운 규칙]
현업 시트("보정 적용 내용")를 보면 콜아웃이 이렇게 찍혀 있다.

    - 편차가 큰 자리 (-2.0, +2.0 처럼 양 끝값)
    - 그 주변에서 가장 큰 자리 하나 (옆에 비슷한 값이 여럿 있어도 하나만)
    - 서로 떨어뜨려 찍는다 (한 곳에 몰려 있지 않다)
    - 0 인 자리도 몇 군데 적는다 (기준을 보여주려고)

그래서 세 가지를 본다.
    1. 크기      |편차| 가 클수록 손볼 값이 크다
    2. 도드라짐  주변보다 얼마나 튀는가 — 옆도 같이 크면 대표 하나면 된다
    3. 간격      이미 고른 것과 너무 가까우면 건너뛴다

셋을 합쳐 점수를 매기고 높은 것부터 간격을 지켜 가며 뽑는다
(비최대 억제, non-maximum suppression).

[실측 — JD_67XX6, 편차 포인트 130개, 컬러바 +-3.0mm]
    먼저 범위 밖 판독값 33개(25%)를 버린다. 남은 97개에서 —

        목표   고른 수   값 범위            서로 최소거리
         10      10개    -3.1 ~ +2.9mm        173px
         15      15개    -3.1 ~ +2.9mm        168px
         20      16개    -3.1 ~ +2.9mm        168px

    원본 130개는 서로 최소거리가 0px 였다(같은 자리에 겹친 포인트가
    있다). 목표를 20 으로 올려도 16개에서 멈추는데, 간격 조건을 지키며
    더 넣을 자리가 없어서다 — 억지로 채우지 않는다.

    범위 밖 값을 안 버리면 상위 3개가 전부 +9.0mm 오독이었다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from zero_line_detection.simple_zero_line import colorbar_span_for

# 이보다 작으면 손볼 값이 아니다(mm).
NOISE_FLOOR_MM = 0.2
# 이미 고른 포인트와 최소한 이만큼 떨어져야 한다. 이미지 대각선 대비.
MIN_GAP_RATIO = 0.08
# "주변" 을 볼 반경. 이 안의 포인트와 견줘 도드라짐을 잰다.
NEIGHBOUR_RATIO = 0.12
# 목표 개수 — 현업 시트가 보통 이 정도다.
DEFAULT_TARGET = 15


@dataclass
class KeyPoint:
    """보정시트에 적을 포인트."""

    point_id: str
    x_px: float
    y_px: float
    value: float
    score: float
    reason: str        # 왜 골랐는지 — 사람이 납득해야 한다

    def to_dict(self) -> dict:
        return asdict(self)


def select(points: list, width: int, height: int,
           target: int = DEFAULT_TARGET,
           part_no: str | None = None,
           noise_floor: float = NOISE_FLOOR_MM,
           min_gap_ratio: float = MIN_GAP_RATIO) -> tuple:
    """편차 포인트에서 보정시트에 적을 것을 고른다.

    Args:
        points: [{id, xPx, yPx, value}, ...]
        width, height: 스캔 크기(px).
        target: 몇 개를 고를지.
        part_no: 컬러바 범위를 아는 품번. 주면 범위 밖 판독값을 버린다.

    Returns:
        (KeyPoint 목록, 버린 판독값 목록). 점수가 높은 순이다.
    """
    # 컬러바 범위 밖 값은 판독 오류다. 이걸 안 거르면 **오류가 1등으로
    # 뽑힌다** — 실측 JD_67XX6(컬러바 +-3.0mm)에서 상위 3개가 전부
    # +9.0mm 오독이었다. 크기로 점수를 매기는 이상 반드시 먼저 버려야
    # 한다. cad_overlay_for 와 같은 기준(5% 여유)을 쓴다.
    span = colorbar_span_for(part_no) if part_no else None
    limit = max(abs(span[0]), abs(span[1])) * 1.05 if span else None

    usable, rejected = [], []
    for point in points:
        if not point.get("id"):
            continue
        value = float(point.get("value", 0.0))
        if limit is not None and abs(value) > limit:
            rejected.append({"id": point["id"], "value": round(value, 3)})
            continue
        if abs(value) >= noise_floor:
            usable.append(point)
    if not usable:
        return [], rejected

    spots = np.array([[float(p["xPx"]), float(p["yPx"])] for p in usable])
    values = np.array([float(p["value"]) for p in usable])
    diagonal = float(np.hypot(width, height))
    neighbourhood = diagonal * NEIGHBOUR_RATIO
    gap = diagonal * min_gap_ratio

    # 도드라짐 — 주변 포인트들의 |편차| 중앙값보다 얼마나 큰가.
    # 옆도 다 같이 크면 그중 하나만 대표로 남기려는 것이다.
    magnitude = np.abs(values)
    standout = np.zeros(len(usable))
    for i, spot in enumerate(spots):
        near = np.hypot(*(spots - spot).T) <= neighbourhood
        near[i] = False
        standout[i] = (magnitude[i] - float(np.median(magnitude[near]))
                       if near.any() else magnitude[i])

    peak = max(float(magnitude.max()), 1e-6)
    score = magnitude / peak + 0.6 * np.clip(standout / peak, 0, None)

    chosen: list = []
    taken: list = []
    for index in np.argsort(-score):
        spot = spots[index]
        if taken and min(float(np.hypot(*(spot - other))) for other in taken) < gap:
            continue        # 이미 고른 것과 너무 가깝다
        taken.append(spot)
        point = usable[index]
        if magnitude[index] >= peak * 0.75:
            reason = "편차가 가장 큰 자리"
        elif standout[index] > 0:
            reason = "주변보다 도드라진 자리"
        else:
            reason = "간격을 두고 고른 자리"
        chosen.append(KeyPoint(
            point_id=str(point["id"]),
            x_px=round(float(point["xPx"]), 1),
            y_px=round(float(point["yPx"]), 1),
            value=round(float(point["value"]), 3),
            score=round(float(score[index]), 4),
            reason=reason,
        ))
        if len(chosen) >= target:
            break
    return chosen, rejected


__all__ = ["DEFAULT_TARGET", "MIN_GAP_RATIO", "NOISE_FLOOR_MM",
           "KeyPoint", "select"]
