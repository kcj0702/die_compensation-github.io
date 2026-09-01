"""제로 영역을 사람이 읽을 수 있는 **도형**으로 바꾼다.

[왜 필요한가]
받은 파이프라인이 67XX6 에 내놓는 제로 영역은 링을 따라 구불구불 흐르는
띠 하나다(그쪽 최종 그림 08_zero_region_on_label_removed_scan.png 이
그렇게 생겼다). 그 윤곽을 그대로 쓰면 —

  · 2D 시트에서는 실오라기 같은 경계가 되어 어디까지가 영역인지 안 읽힌다
  · 3D 에서는 정점 단위로 칠해져 삼각망을 따라 조각조각 갈라진다

ADC 보정시트는 영역을 **네모**로 표기한다. 그래서 여기서 띠를 네모
몇 개로 바꾼다. 하나로 감싸면 안 된다 — ㄱ 자로 꺾인 띠를 한 네모로
감싸면 빈 데까지 덮는다(실측 67XX6 의 한 영역이 625x274 네모에 채움
18% 였고, 그걸 칠하면 그림의 10% 를 잘못 칠한다).

[어떻게]
최소 외접 사각형에 실하게 들어차면 그대로 쓰고, 성기면 **긴 축으로 반
갈라** 각각 다시 본다. 꺾인 데서 저절로 갈라져 네모 몇 개가 띠를 따라
이어진다. 조각 수에는 상한을 둔다 — 무한정 쪼개면 다시 지저분해진다.
"""
from __future__ import annotations

import cv2
import numpy as np

# 네모 넓이의 이만큼은 실제로 차 있어야 그 네모를 쓴다.
#
# [값을 어떻게 정했나 — 실측 67XX6, 제로 영역이 그림의 5.6%]
#   채움  조각  네모수  진짜 영역 중 덮음  네모 중 헛덮음
#   0.50    6      17            98%            43%
#   0.50    8      19            98%            41%
#   0.62    6      23            92%            41%
#   0.72    8      37            76%            39%
# 잘게 쪼개도 헛덮음이 40% 아래로는 안 내려간다 — 굽은 띠를 네모로
# 싸는 이상 어쩔 수 없다. 그러면 **네모 수가 적고 빠뜨리지 않는** 쪽이
# 낫다. 17개로 98% 를 덮는 (0.50, 6) 을 쓴다.
MIN_FILL = 0.50
# 한 영역을 이보다 잘게 쪼개지 않는다.
MAX_PIECES = 6
# 이보다 작은 조각은 버린다(픽셀).
MIN_AREA_PX = 120


def _box(points: np.ndarray) -> np.ndarray:
    return np.rint(cv2.boxPoints(cv2.minAreaRect(points))).astype(np.int32)


def _fill(points: np.ndarray) -> float:
    (_mid, (width, height), _angle) = cv2.minAreaRect(points)
    span = float(width) * float(height)
    return (cv2.contourArea(points) / span) if span > 0 else 0.0


def _split(points: np.ndarray) -> list:
    """긴 축의 한가운데에서 두 쪽으로 가른다."""
    (mid, (width, height), angle) = cv2.minAreaRect(points)
    turn = np.deg2rad(angle)
    along = (np.array([np.cos(turn), np.sin(turn)]) if width >= height
             else np.array([-np.sin(turn), np.cos(turn)]))
    reach = (points - np.asarray(mid)) @ along
    near = points[reach <= 0]
    far = points[reach > 0]
    return [part for part in (near, far) if len(part) >= 3]


def boxes_of(contour, budget: int = MAX_PIECES) -> list:
    """윤곽 하나를 네모 몇 개로 바꾼다.

    Args:
        contour: [[x, y], ...] 픽셀 좌표.
        budget: 이 윤곽에 쓸 수 있는 조각 수.

    Returns:
        [[[x, y] x4], ...] — 네모마다 꼭짓점 4개.
    """
    points = np.rint(np.asarray(contour, dtype=float)).astype(np.int32)
    if len(points) < 3 or cv2.contourArea(points) < MIN_AREA_PX:
        return []
    if budget <= 1 or _fill(points) >= MIN_FILL:
        return [_box(points).tolist()]

    halves = _split(points)
    if len(halves) < 2:
        return [_box(points).tolist()]

    # 남은 조각 수를 넓이에 비례해 나눠 준다
    sizes = [max(cv2.contourArea(h), 1.0) for h in halves]
    total = sum(sizes)
    out: list = []
    for half, size in zip(halves, sizes):
        share = max(1, int(round(budget * size / total)))
        out.extend(boxes_of(half, min(share, budget - 1)))
    return out


def clean(contours: list, budget: int = MAX_PIECES) -> list:
    """영역 목록을 통째로 네모 목록으로 바꾼다."""
    out: list = []
    for contour in contours or []:
        out.extend(boxes_of(contour, budget))
    return out


__all__ = ["MAX_PIECES", "MIN_AREA_PX", "MIN_FILL", "boxes_of", "clean"]
