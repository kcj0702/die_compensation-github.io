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
LIGHT_RING_MIN_RATIO = 0.6   # 글자 주변에서 "밝고 파랑 아님" 픽셀이 이 비율 이상이어야 콜아웃
CALLOUT_MIN_SIZE = (8, 6)    # 너무 작은 잡음 성분 제외
CALLOUT_MAX_WIDTH = 70       # 이보다 넓으면 콜아웃이 아니라 캡션 문장이다
CALLOUT_PADDING = 8          # VLM 에 넘길 crop 여백
MAX_DOT_DISTANCE = 60.0      # 실측: 진짜 매칭은 55px 이내, 오매칭은 80px+ 부터 시작

# "절대높이 유지" 같은 기준점 지시문 — 숫자 콜아웃과 다른 스타일이다
# (실측: JD_67XX6 "폼 좌면 6ea 절대높이 유지" — 빨간 글자, 노란 바탕).
# 이 지시문이 정확히 어느 점(들)을 가리키는지 자동으로 풀어내는 건
# 아직 안 한다 — 실측해보니 빨간 글자 검출이 리더선·점까지 같이 걸려
# 후보가 20개 넘게 나오고, 어느 게 진짜 지시문인지 안정적으로 못
# 걸러냈다(아래 detect_instruction_notes 문서 참고). 그래서 지금은
# "여기 숫자 아닌 지시문이 있다"는 존재만 표시하고 대상 매칭은 사람이
# 확인하게 한다.
INSTRUCTION_RED_TEXT_MIN_RATIO = 0.15
INSTRUCTION_MIN_SIZE = (60, 10)
INSTRUCTION_MAX_SIZE = (350, 45)


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
    """단색 배경 위의 어두운 글자를 찾아 인접한 것끼리 묶는다.

    콜아웃 테두리가 아니라 **글자**를 기준으로 찾는 이유는 테두리가
    실측상 너무 옅어(그레이스케일 130 이하로도 안 잡힘) 안정적으로
    닫힌 사각형을 못 만들었기 때문이다.

    박스 배경색은 시트마다 다르다 — JD_64XX2는 흰 박스, JD_67XX6는
    노란 박스를 쓴다(둘 다 실측 확인). 색을 미리 정해두면 새 시트가
    또 다른 색을 쓸 때 또 놓친다. 그래서 **"부품 렌더(파랑)가 아니고
    밝다"** 로만 판정한다 — 흰색·노랑·분홍 등 콜아웃 배경이 뭐든 통하고,
    파란 렌더 위의 글자(부품명 등)만 제외한다.

    [한계] JD_67XX6처럼 콜아웃이 촘촘히 몰린 시트에서는 이웃 콜아웃·
    빨간선이 링에 섞여 여전히 상당수를 놓친다(실측: 콜아웃 약 57개 중
    24개 정도만 검출). 임계값을 더 조여도 개선 폭이 작아 여기서 멈췄다
    — 근본적으로 고치려면 색/모양 휴리스틱이 아니라 VLM으로 영역 자체를
    제안받는 방식이 필요해 보인다.
    """
    height, width = sheet_bgr.shape[:2]
    gray = cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < GLYPH_DARK_THRESHOLD).astype(np.uint8)
    b, g, r = (sheet_bgr[..., i].astype(int) for i in range(3))
    brightness = (r + g + b) / 3.0
    is_blueish = (b - r) > 25  # 부품 렌더는 파랑이 빨강보다 뚜렷이 크다
    light_bg = (brightness > 140) & ~is_blueish
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)

    glyphs: list = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if y < HEADER_CUTOFF_Y or area < 2 or w > GLYPH_MAX_SIZE or h > GLYPH_MAX_SIZE:
            continue
        glyphs.append((x, y, w, h))

    # 먼저 가까운 글자 조각을 하나의 콜아웃으로 합친다(자릿수 여러 개,
    # 두 줄 라벨). 배경색 판정은 이 다음, **합쳐진 덩어리** 기준으로
    # 한다 — 자릿수 하나(예: "-0.4"의 "4")만 놓고 주변을 보면 바로 옆
    # 자릿수("0", ".")가 링 안에 섞여 들어와 단색처럼 안 보인다.
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
    boxes: list = []
    ring_pad = 6
    for x0, y0, x1, y1 in groups:
        w, h = x1 - x0, y1 - y0
        if not (min_w <= w <= CALLOUT_MAX_WIDTH and h >= min_h):
            continue
        ry0, ry1 = max(0, y0 - ring_pad), min(height, y1 + ring_pad)
        rx0, rx1 = max(0, x0 - ring_pad), min(width, x1 + ring_pad)
        ring_light = light_bg[ry0:ry1, rx0:rx1]
        glyph_local = np.zeros(ring_light.shape, bool)
        gy0, gy1 = y0 - ry0, y1 - ry0
        gx0, gx1 = x0 - rx0, x1 - rx0
        glyph_local[max(0, gy0):gy1, max(0, gx0):gx1] = True
        ring_only = ring_light & ~glyph_local
        denom = int((~glyph_local).sum())
        if denom < 10:
            continue
        light_ratio = float(ring_only.sum()) / denom
        if light_ratio < LIGHT_RING_MIN_RATIO:
            continue  # 주변 대부분이 부품 렌더(파랑)다 — 글자가 아니다
        boxes.append((x0, y0, w, h))
    return boxes


def detect_instruction_notes(sheet_bgr: np.ndarray) -> list:
    """빨간 글자로 된 기준점 지시문의 존재만 찾는다(예: "N ea 절대높이 유지").

    [한계 — 왜 대상 점까지 자동으로 안 푸는가]
    실측(JD_67XX6): 이 방식으로 후보를 뽑으면 20개 넘게 나온다 — 빨간
    글자만이 아니라 빨간 리더선·점이 모폴로지 닫기에서 같이 뭉쳐 문장
    처럼 보이는 덩어리가 많이 생긴다. 진짜 지시문과 오탐을 안정적으로
    구분할 방법을 못 찾았고, 설령 진짜 박스를 찾아도 "가장 가까운 점
    N개"로 대상을 정하면 틀릴 위험이 있다(개수 표기를 못 읽으므로).
    그래서 여기서는 **위치만** 알려주고, 그 지시문이 어느 점에 적용
    되는지는 사람이 시트를 보고 확인하게 한다 — 틀린 자동 매칭을
    학습 데이터에 섞는 것보다 안전하다.
    """
    height, _width = sheet_bgr.shape[:2]
    b, g, r = (sheet_bgr[..., i].astype(int) for i in range(3))
    red_text = ((r > 120) & (g < 110) & (b < 110)).astype(np.uint8)
    red_text[:HEADER_CUTOFF_Y, :] = 0
    closed = cv2.morphologyEx(
        red_text * 255, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    min_w, min_h = INSTRUCTION_MIN_SIZE
    max_w, max_h = INSTRUCTION_MAX_SIZE
    notes: list = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        red_ratio = area / max(w * h, 1)
        if red_ratio < INSTRUCTION_RED_TEXT_MIN_RATIO:
            continue
        notes.append((x, y, w, h))
    return notes


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
    "detect_instruction_notes",
]
