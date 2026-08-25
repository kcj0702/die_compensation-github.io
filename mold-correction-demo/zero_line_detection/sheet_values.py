"""보정시트에 찍힌 보정치 콜아웃(예: "-0.4")을 읽어 좌표와 값을 뽑는다.

[왜 필요한가]
지금까지 시트에서는 빨간 제로라인/제로존만 읽었다(sheet_reference.py).
그런데 시트에는 그 외에도 부품 곳곳에 **실제 보정치 숫자**가 콜아웃
박스로 찍혀 있다 — 하형 기준 "-0.4", "-0.7mm" 식으로. 이건 작업자가
그 위치에서 최종 결정한 보정량이라, 같은 위치의 스캔 실측 편차값과
비교하면 "측정값 그대로 보정한 게 아니라 뭘 더 반영했는지"를 배울 수
있는 신호가 된다.

[구조]
콜아웃은 흰 바탕 사각 박스(옅은 회색 테두리) 안에 숫자가 있고, 빨간
가는 선으로 부품 위 빨간 점(측정 지점)까지 이어진다. 박스 테두리가
너무 옅어서(실측: 임계값 130에서도 대부분 안 잡힘) 테두리로 박스를
찾는 방법은 실패했다. 대신 **글자 자체**(뚜렷이 어두움)를 찾아 서로
가까운 글자를 묶어 콜아웃 영역을 만든다.

[한계]
- 어느 콜아웃이 어느 점을 가리키는지는 "가장 가까운 빨간 점"으로
  추정한다. 리더선을 끝까지 추적하지 않는다 — 시트마다 선이 여러 점에서
  모이기도 해서(사진 확인: "-0.7" 박스에 점 3개가 모임) 추적보다 근접
  매칭이 더 안정적이었다.
- "0" LINE 처럼 두 줄짜리 라벨도 콜아웃으로 잡힌다. 값 판독은 VLM에
  맡기고, 숫자가 안 나오면 버린다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


HEADER_CUTOFF_Y = 200  # 상단 정보 테이블(관리번호 등) 제외
GLYPH_DARK_THRESHOLD = 110
GLYPH_MAX_SIZE = 60          # 글자 하나의 최대 폭/높이(px)
GLYPH_MERGE_DIST = 14        # 이 거리 이내의 글자는 같은 콜아웃으로 묶는다
CALLOUT_MIN_SIZE = (8, 6)    # 너무 작은 잡음 성분 제외
CALLOUT_MAX_WIDTH = 70       # 이보다 넓으면 콜아웃이 아니라 캡션 문장이다
CALLOUT_PADDING = 8          # VLM 에 넘길 crop 여백
MAX_DOT_DISTANCE = 60.0      # 실측: 진짜 매칭은 55px 이내, 오매칭은 80px+ 부터 시작


@dataclass
class SheetCallout:
    """콜아웃 박스 하나 — 값과 그 값이 가리키는 점."""

    value: float
    point: list          # [x, y] 점 좌표 (스캔이 아니라 시트 픽셀)
    box: list            # [x, y, w, h] 콜아웃 텍스트 영역
    dot_distance: float  # 콜아웃에서 매칭된 점까지 거리(px)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_red_dots(sheet_bgr: np.ndarray) -> list:
    """빨간 점(측정 지점 마커)을 찾는다.

    점과 리더선이 픽셀상 맞닿아 있어(실측 확인) 단순히 연결성분의
    bounding-box 채움비로 나누면 점+선이 하나의 길쭉한 성분으로 잡혀
    점을 놓친다. 대신 **모폴로지 열기**로 가는 선(1~2px 두께)만 지우고
    통통한 점(지름 6~10px)은 남긴다 — 침식 후 팽창이라 선은 사라지고
    점은 살아남는다.
    """
    b, g, r = (sheet_bgr[..., i].astype(int) for i in range(3))
    red_mask = ((r > 150) & (r - g > 60) & (r - b > 60)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(opened, 8)

    dots = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 15 or w > 20 or h > 20:
            continue
        dots.append((float(centroids[i][0]), float(centroids[i][1])))
    return dots


def detect_callout_regions(sheet_bgr: np.ndarray) -> list:
    """흰 바탕 위의 어두운 글자를 찾아 인접한 것끼리 묶는다.

    콜아웃 테두리가 아니라 **글자**를 기준으로 찾는 이유는 테두리가
    실측상 너무 옅어(그레이스케일 130 이하로도 안 잡힘) 안정적으로
    닫힌 사각형을 못 만들었기 때문이다.
    """
    height, width = sheet_bgr.shape[:2]
    gray = cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < GLYPH_DARK_THRESHOLD).astype(np.uint8)
    white_bg = np.all(sheet_bgr > 190, axis=2).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    ring_kernel = np.ones((7, 7), np.uint8)

    glyphs: list = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if y < HEADER_CUTOFF_Y or area < 2 or w > GLYPH_MAX_SIZE or h > GLYPH_MAX_SIZE:
            continue
        y0, y1 = max(0, y - 3), min(height, y + h + 3)
        x0, x1 = max(0, x - 3), min(width, x + w + 3)
        component = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(component, ring_kernel) - component
        if ring.sum() == 0:
            continue
        white_around = float((ring & white_bg[y0:y1, x0:x1]).sum()) / float(ring.sum())
        if white_around < 0.5:
            continue  # 부품 렌더(파랑) 위의 어두운 획은 글자가 아니다
        glyphs.append((x, y, w, h))

    # 가까운 글자를 하나의 콜아웃으로 병합한다 (여러 자리 숫자, 두 줄 라벨)
    groups: list = []
    for gx, gy, gw, gh in sorted(glyphs, key=lambda g: (g[1], g[0])):
        gx0, gy0, gx1, gy1 = gx, gy, gx + gw, gy + gh
        merged = False
        for grp in groups:
            if (gx0 <= grp[2] + GLYPH_MERGE_DIST and gx1 >= grp[0] - GLYPH_MERGE_DIST
                    and gy0 <= grp[3] + GLYPH_MERGE_DIST and gy1 >= grp[1] - GLYPH_MERGE_DIST):
                grp[0] = min(grp[0], gx0); grp[1] = min(grp[1], gy0)
                grp[2] = max(grp[2], gx1); grp[3] = max(grp[3], gy1)
                merged = True
                break
        if not merged:
            groups.append([gx0, gy0, gx1, gy1])
        _merge_overlapping_groups(groups)

    min_w, min_h = CALLOUT_MIN_SIZE
    return [
        (x0, y0, x1 - x0, y1 - y0)
        for x0, y0, x1, y1 in groups
        if min_w <= (x1 - x0) <= CALLOUT_MAX_WIDTH and (y1 - y0) >= min_h
    ]


def _merge_overlapping_groups(groups: list) -> None:
    """서로 겹치거나 맞닿은 그룹을 한 번 더 합친다 (제자리 수정)."""
    changed = True
    while changed:
        changed = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ax0, ay0, ax1, ay1 = groups[a]
                bx0, by0, bx1, by1 = groups[b]
                if (ax0 <= bx1 + GLYPH_MERGE_DIST and ax1 >= bx0 - GLYPH_MERGE_DIST
                        and ay0 <= by1 + GLYPH_MERGE_DIST and ay1 >= by0 - GLYPH_MERGE_DIST):
                    groups[a] = [min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)]
                    groups.pop(b)
                    changed = True
                    break
            if changed:
                break


def match_nearest_dot(box: tuple, dots: list, max_distance: float = MAX_DOT_DISTANCE):
    """콜아웃 박스에서 가장 가까운 빨간 점을 찾는다."""
    if not dots:
        return None, float("inf")
    x, y, w, h = box
    center = np.array([x + w / 2.0, y + h / 2.0])
    pts = np.array(dots, dtype=float)
    dists = np.hypot(*(pts - center).T)
    idx = int(np.argmin(dists))
    distance = float(dists[idx])
    if distance > max_distance:
        return None, distance
    return tuple(pts[idx]), distance


def build_callout_crops(sheet_bgr: np.ndarray, boxes: list) -> list:
    """콜아웃 박스마다 VLM에 넘길 crop(PIL 이미지)을 만든다.

    실제 VLM 호출은 여기서 하지 않는다 — server.py 의 `_read_qwen_values`
    가 배치 실패시 개별 재판독·집중 재판독까지 해주는 걸 그대로 재사용
    하기 위해서다(작은 crop 일수록 배치에서 한 번에 놓치기 쉬웠다).
    """
    from PIL import Image as PILImage

    height, width = sheet_bgr.shape[:2]
    crops = []
    for x, y, w, h in boxes:
        x0 = max(0, x - CALLOUT_PADDING)
        y0 = max(0, y - CALLOUT_PADDING)
        x1 = min(width, x + w + CALLOUT_PADDING)
        y1 = min(height, y + h + CALLOUT_PADDING)
        crop = sheet_bgr[y0:y1, x0:x1]
        crops.append(PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    return crops


def assemble_callouts(boxes: list, dots: list, values: list) -> list:
    """검출된 박스·점·판독값을 하나의 콜아웃 목록으로 합친다."""
    callouts = []
    for box, value in zip(boxes, values):
        if value is None:
            continue
        dot, distance = match_nearest_dot(box, dots)
        if dot is None:
            continue
        callouts.append(SheetCallout(
            value=round(float(value), 3),
            point=[round(dot[0], 1), round(dot[1], 1)],
            box=list(box),
            dot_distance=round(distance, 1),
        ))
    return callouts


__all__ = [
    "SheetCallout", "detect_red_dots", "detect_callout_regions",
    "match_nearest_dot", "build_callout_crops", "assemble_callouts",
]
