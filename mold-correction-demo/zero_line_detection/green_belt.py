"""현업이 준 '영라인 선정 방법' 을 그대로 구현한다.

[근거 — 현업 제공 자료 2026-08-25]
    1단계: 녹색 영역(Zero Zone) 확인
        "녹색은 도면 대비 오차가 0에 가까운 구간. 길게 형성된 **녹색 벨트
         구역**이 영라인 설정의 최우선 후보."
    3단계: 플러스(+)와 마이너스(-)의 '전환점'
        "노란·빨간색(플러스) 영역에서 파란색(마이너스) 영역으로 색상이
         넘어가는 **경계선의 녹색 부위**를 찾습니다. 이 경계선들을
         연결하면 뒤틀림 중심 축(영라인)을 도출할 수 있습니다."

즉 영라인 후보 = (녹색) AND (부호 전환대) 중 **길쭉한 것**.

[왜 이 방식으로 바꿨나]
이전에는 두 0포인트 사이를 픽셀 최단경로로 이었다. 그런데 그 경로가
측정점이 없는 자리를 지나가는 문제가 있었다 — 현업 지적: "저 구간엔
포인트도 없고 수치값도 없는데 왜 선을 넣는지". 실측하니 경로의 14.2%가
측정점에서 100px 넘게 떨어져 있었다(JD_64XX2).

측정점만 경유하게 바꿔봤지만 정확도가 반토막 났다(선->정답 5.52% ->
10.70%). 끝점이 둘 다 부품 아래쪽이라 최단경로가 아래 테두리로 직진해
버리는데, 정답선은 같은 두 점 사이를 위로 크게 돌아간다.

이 방식은 아예 "잇지 않는다". 근거(녹색 + 부호전환)가 있는 자리만
벨트로 내놓고, 없는 자리는 비워둔다. 끊긴 채로 나오지만 **그리는 것은
전부 실측이 뒷받침한다.**

[실측 성능 — JD_64XX2, 벨트 3개, 정답선 1136px / 대각선 1472px]
    벨트->정답 (그린 것이 맞나): 중앙값 3.00%
    정답->벨트 (놓친 것은 없나): 25분위 2.00%  중앙값 4.39%
                                75분위 9.25%  90분위 19.80%
    정답선 커버리지:  50px 이내 41%   100px 이내 65%   200px 이내 81%

    즉 **정밀하지만 불완전하다**. 그린 자리는 정답선 위에 있고, 대신
    오른쪽 하강 구간처럼 근거가 약한 곳은 아예 비운다(최대 388px).
    기존 픽셀 최단경로는 선->정답 5.52% 로 이보다 나빴고, 그러면서
    14.2% 구간이 측정점에서 100px 넘게 떨어져 있었다.

[검증 범위 — 한 부품뿐이다]
    선 형태의 정답이 있는 것은 JD_64XX2 하나다. JD_67XX6 의 시트는
    면(area) 으로 등록돼 있는데 그 폴리곤이 선루프 빈 구멍을 대각선으로
    가로지른다 — 폴리곤 자체가 잘못돼 거리 비교에 쓸 수 없다.
    JD_71XX2 는 앞서 정답이 무효 판정됐다.
    따라서 아래 상수(GREEN_QUANTILE, MIN_BELT_LENGTH_RATIO)는 사실상
    **부품 한 개에 맞춰진 값**이다. 사례가 더 들어오면 다시 재야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

# 녹색(오차 0 근처) 판정 기준 — **부품 면적의 분위수**로 잡는다.
#
# 처음엔 절대값(mm)이었는데 프로덕션에서 벨트가 3개->1개로 줄었다.
# 원인: 서버가 컬러바 눈금을 못 읽으면 편차가 +-1 로 정규화돼 들어온다
# (실측 JD_64XX2: 실제 -1.49~+2.00 인데 기본 config 로는 -0.99~+1.00).
# 같은 "0.2" 가 스케일에 따라 다른 뜻이 되므로 상대 기준이 필요하다.
#
# 다음으로 "편차 범위 대비 비율"을 썼지만 두 부품이 서로 다른 값을 원했다
# (JD_64XX2 는 0.04, JD_67XX6 은 0.08~0.12). 범위는 이상치 한 점에
# 끌려다니기 때문이다. 면적 분위수는 그 영향을 받지 않는다 —
# "부품에서 편차가 가장 작은 쪽 28%" 가 녹색이다. 두 부품 모두에서
# q=0.25~0.30 이 안정적으로 동작한다(0.28 과 0.30 결과가 동일).
GREEN_QUANTILE = 0.28
# 분위수 대신 절대값(mm)을 쓰고 싶을 때만 지정한다.
GREEN_THRESHOLD_MM = None
# 부호 전환을 볼 때의 스무딩. 작을수록 국소적인 전환까지 잡는다.
TRANSITION_SIGMA = 10.0
# 전환'대' 폭 — 경계 픽셀만 보면 너무 얇아 녹색과 겹치지 않는다.
TRANSITION_DILATE = 9
# 벨트로 인정할 최소 길이(대각선 대비)와 최소 길쭉함(긴변/짧은변).
# 0.08 은 정답과 겹치는 짧은 벨트까지 잘라냈다 — 0.06 으로 낮추니
# 두 부품 모두 좋아졌다(JD_64XX2 합산 11.0 -> 7.4, JD_67XX6 12.7 -> 2.9).
MIN_BELT_LENGTH_RATIO = 0.06
MIN_BELT_ELONGATION = 2.0


@dataclass
class GreenBelt:
    """녹색이면서 부호가 전환되는 길쭉한 구간 — 영라인 후보."""

    belt_id: int
    contour: list          # [[x, y], ...] 폴리곤
    center: list           # [x, y]
    length_px: float       # 긴 변 길이
    area_px: int
    mean_abs_deviation: float

    def to_dict(self) -> dict:
        return asdict(self)


def green_threshold_for(values: np.ndarray, part_mask,
                        quantile: float = GREEN_QUANTILE) -> float:
    """|편차| 가 작은 쪽 quantile 만큼의 면적을 '녹색' 으로 삼는다.

    컬러바 눈금을 못 읽어 값이 +-1 로 정규화돼 들어와도, 이상치 한 점이
    범위를 늘려놔도 같은 뜻이 되도록 면적 기준으로 잡는다.
    """
    inside = values[np.asarray(part_mask) > 0]
    if inside.size == 0:
        return 0.0
    return max(float(np.quantile(np.abs(inside), float(quantile))), 1e-6)


def find_green_belts(
    values: np.ndarray,
    part_mask,
    green_threshold: float | None = GREEN_THRESHOLD_MM,
    green_quantile: float = GREEN_QUANTILE,
    transition_sigma: float = TRANSITION_SIGMA,
    min_length_ratio: float = MIN_BELT_LENGTH_RATIO,
    min_elongation: float = MIN_BELT_ELONGATION,
    rezero: bool = True,
) -> list:
    """녹색 + 부호전환대의 교집합에서 길쭉한 벨트를 찾는다.

    Args:
        values: 편차값 맵(mm).
        part_mask: 부품 영역.
        green_threshold: |편차| 가 이 값 이하면 '녹색'(절대값, mm).
            None 이면 green_quantile 로 자동 계산한다.
        green_quantile: 부품 면적의 이 비율만큼을 녹색으로 본다.
        transition_sigma: 부호 전환을 볼 때 쓰는 스무딩.
        min_length_ratio: 이미지 대각선 대비 최소 벨트 길이.
        min_elongation: 긴 변 / 짧은 변 최소 비율(덩어리 제외).
        rezero: 부품 편차의 중앙값을 0으로 다시 맞춘다(자료 4단계).

    [4단계 — 평탄도 평균을 0으로 리셋]
        "제품 전체가 한쪽으로 쏠려 녹색이 거의 없다면, 가장 넓은 평면
         구간의 평균값을 0으로 리셋(Alignment 재설정)해야 합니다."

    실측(JD_64XX2, 라벨로 보정한 편차): 중앙값이 -0.142 라 '0' 이 0 이
    아니었다. 리셋 전후로 결과가 크게 갈린다 —

        리셋 없음   정답->벨트 13.53%  벨트->정답 14.04%
        중앙값 리셋  정답->벨트  5.34%  벨트->정답  5.69%
    """
    mask = np.asarray(part_mask) > 0
    height, width = values.shape
    diag = float(np.hypot(width, height))
    if rezero:
        inside = values[mask]
        if inside.size:
            values = values - float(np.median(inside))
    if green_threshold is None:
        green_threshold = green_threshold_for(values, mask, green_quantile)

    # 1단계 — 녹색(오차 0 근처). 스무딩하지 않은 실제 값으로 본다.
    green = (np.abs(values) <= green_threshold) & mask

    # 3단계 — 플러스 영역과 마이너스 영역이 만나는 '전환대'
    smoothed = cv2.GaussianBlur(values, (0, 0), float(transition_sigma))
    positive = ((smoothed > 0) & mask).astype(np.uint8)
    negative = ((smoothed < 0) & mask).astype(np.uint8)
    widen = np.ones((TRANSITION_DILATE, TRANSITION_DILATE), np.uint8)
    transition = (
        (cv2.dilate(positive, widen) > 0)
        & (cv2.dilate(negative, widen) > 0)
        & mask
    )

    belt_mask = (green & transition).astype(np.uint8)
    # 잔가지를 정리하고 살짝 끊긴 벨트를 잇는다
    belt_mask = cv2.morphologyEx(belt_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(belt_mask, 8)
    belts: list = []
    for i in range(1, count):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        longest, shortest = max(w, h), max(min(w, h), 1)
        if longest < diag * min_length_ratio:
            continue                      # 짧은 조각 — 벨트가 아니다
        if longest < min_elongation * shortest:
            continue                      # 덩어리 — "길게 형성된" 것이 아니다
        piece = (labels == i).astype(np.uint8)
        found, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            continue
        outline = max(found, key=cv2.contourArea).reshape(-1, 2)
        region = piece > 0
        belts.append(GreenBelt(
            belt_id=0,
            contour=outline.astype(int).tolist(),
            center=[round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)],
            length_px=float(longest),
            area_px=int(stats[i, cv2.CC_STAT_AREA]),
            mean_abs_deviation=round(float(np.abs(values[region]).mean()), 4),
        ))

    belts.sort(key=lambda b: -b.length_px)   # 긴 벨트가 더 중요한 후보
    for index, belt in enumerate(belts, start=1):
        belt.belt_id = index
    return belts


__all__ = [
    "GREEN_THRESHOLD_MM", "GREEN_QUANTILE", "TRANSITION_SIGMA",
    "green_threshold_for",
    "GreenBelt", "find_green_belts",
]
