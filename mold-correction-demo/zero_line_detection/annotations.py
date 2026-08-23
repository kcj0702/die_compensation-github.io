"""주석(라벨·지시선·제목) 영역 검출.

3D 스캔 이미지에는 히트맵 위에 다음이 얹혀 있다.

    빨간 라벨 박스 (+ 흰 글씨)   예: -1.7, 2.8
    흰/회색 라벨 박스 (+ 검은 글씨)  예: -0.3
    파란 지시선
    빨간 제목 텍스트            예: JG SUNROOF 26.06.09

이 중 **빨간 라벨 박스와 파란 지시선이 특히 위험하다.**
둘 다 컬러바에 실제로 존재하는 색(빨강=최댓값, 파랑=음수)이라
색상만으로는 히트맵 데이터와 구분되지 않는다.
지우지 않으면 라벨이 통째로 "편차 +3.0 영역"으로 오인된다.

흰 배경·회색 미측정면·검은 글씨는 컬러바에 없는 색이므로
colorbar.map_image() 의 유효성 판정에서 자동으로 걸러진다.
따라서 이 모듈은 자동으로 안 걸러지는 두 가지만 처리한다.

파트 4(라벨 제거)의 clean_deviation_map.png 가 준비되면 그쪽이 우선이며,
이 모듈은 파트 4 산출물이 없을 때를 위한 대비책이다.
"""

from __future__ import annotations

import cv2
import numpy as np


def _components(mask: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return n, labels, stats, centroids


def detect_label_boxes(
    rgb: np.ndarray,
    min_area: int = 120,
    max_area: int = 12000,
    min_fill: float = 0.55,
) -> np.ndarray:
    """빨간 라벨 박스를 찾는다.

    히트맵의 빨간 영역과 라벨 박스를 가르는 근거는 형태다.
      · 라벨 박스 : 작고, 사각형에 가까우며(채움비 높음), 안에 흰 글씨가 있다
      · 히트맵 빨강 : 크고 불규칙하며, 주황색 그라데이션과 이어져 있다
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    pure_red = (r >= 190) & (g <= 95) & (b <= 95)

    # 글씨 구멍을 메워야 박스 전체가 한 덩어리로 잡힌다
    filled = cv2.morphologyEx(
        pure_red.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )

    near_white = (r >= 200) & (g >= 200) & (b >= 200)
    out = np.zeros(rgb.shape[:2], dtype=bool)

    n, labels, stats, _ = _components(filled)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_area <= area <= max_area):
            continue
        if w * h == 0 or area / (w * h) < min_fill:
            continue
        if not (0.6 <= w / max(h, 1) <= 6.0):
            continue
        # 박스 안에 흰 글씨가 있어야 라벨이다
        inside = near_white[y:y + h, x:x + w]
        if inside.mean() < 0.03:
            continue
        out[labels == i] = True

    return out


def detect_leader_lines(rgb: np.ndarray, max_thickness: int = 5) -> np.ndarray:
    """파란 지시선을 찾는다.

    지시선은 얇다. 굵기 이상의 구조 요소로 열림 연산을 하면 사라지므로,
    '열어서 사라지는 파란 픽셀' 을 지시선으로 본다.
    부품의 넓은 파란 면은 열림 후에도 남는다.
    """
    r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
    blueish = (b >= 110) & (b - r >= 45) & (b - g >= 30)

    k = np.ones((max_thickness * 2 + 1, max_thickness * 2 + 1), np.uint8)
    thick = cv2.morphologyEx(blueish.astype(np.uint8), cv2.MORPH_OPEN, k)
    thin = blueish & (thick == 0)

    # 선 형태만 남기고 흩어진 점은 버린다
    thin = cv2.morphologyEx(
        thin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)

    n, labels, stats, _ = _components(thin)
    out = np.zeros(rgb.shape[:2], dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 25:
            continue
        if max(w, h) < 12:            # 선이라기엔 너무 짧다
            continue
        if area / max(w * h, 1) > 0.85 and min(w, h) > max_thickness * 2:
            continue                  # 꽉 찬 덩어리는 선이 아니다
        out[labels == i] = True
    return out


def build_annotation_mask(
    rgb: np.ndarray,
    colorbar_bbox: tuple[int, int, int, int] | None = None,
    dilate: int = 3,
) -> np.ndarray:
    """주석 전체 마스크. True = 히트맵 데이터가 아닌 픽셀.

    Args:
        colorbar_bbox: (x0, y0, x1, y1). 컬러바와 그 눈금 라벨 영역을 제외한다.
    """
    mask = detect_label_boxes(rgb) | detect_leader_lines(rgb)

    if dilate > 0:
        k = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)

    if colorbar_bbox is not None:
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = colorbar_bbox
        # 컬러바 바깥쪽(이미지 가장자리 방향)에는 눈금 숫자가 붙어 있다.
        # 안쪽으로도 약간 여유를 둔다.
        pad = max(int(w * 0.045), 40)
        if x0 < w / 2:                       # 좌측 컬러바
            mask[:, : min(x1 + pad, w)] = True
        else:                                # 우측 컬러바
            mask[:, max(x0 - pad, 0):] = True

    return mask


__all__ = ["detect_label_boxes", "detect_leader_lines", "build_annotation_mask"]
