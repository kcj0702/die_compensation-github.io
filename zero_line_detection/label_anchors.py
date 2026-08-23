"""라벨 박스와 지시선 끝점(앵커) 검출.

[무엇을 하는가]
스캔 이미지의 라벨 박스를 찾고, 거기서 뻗어 나온 파란 지시선을 따라가
**실제로 가리키는 부품 위의 한 점**을 찾는다.

    [-1.7] ──────────────● (앵커)
     라벨 박스   지시선    이 지점의 편차가 -1.7 이라는 뜻

[왜 필요한가]
파트 2의 색->편차값 변환이 실제 스캔에서 맞는지 확인할 방법이 없었다.
합성 데이터 IoU 는 내가 만든 정답에 대한 점수라 자기 채점에 가깝다.

그런데 스캔 이미지에는 이미 사람이 적어 둔 정답(라벨 값)이 들어 있다.
앵커 지점의 색을 편차값으로 되읽어 라벨 값과 비교하면
**실제 데이터에 대한 오차를 mm 단위로 낼 수 있다.**

[파트 3 과의 관계]
파트 3(편차값·좌표 추출)이 하려는 일의 절반이 여기 들어 있다.
숫자를 읽는 OCR 만 붙이면 그대로 deviation_points.csv 가 된다.
담당자가 가져다 쓸 수 있도록 CSV 로 내보낸다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zero_line_detection.annotations import detect_leader_lines  # noqa: E402


@dataclass
class LabelAnchor:
    """라벨 박스 1개와 그것이 가리키는 지점."""

    label_id: int
    kind: str                 # "red" (강조) 또는 "light" (일반)
    box_x: int
    box_y: int
    box_w: int
    box_h: int
    box_cx: float
    box_cy: float
    anchor_x: int             # 지시선이 가리키는 지점
    anchor_y: int
    leader_len: float         # 지시선 길이 (px). 0 이면 지시선을 못 찾음
    value: float | None = None    # OCR 또는 수기 입력으로 채운다

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────
def detect_red_boxes(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """빨간 라벨 박스. 안에 흰 글씨가 있는 작은 사각형."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red = (r >= 190) & (g <= 95) & (b <= 95)
    filled = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((7, 7), np.uint8))
    near_white = (r >= 200) & (g >= 200) & (b >= 200)

    out = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (120 <= area <= 12000):
            continue
        if area / max(w * h, 1) < 0.55:
            continue
        if not (0.8 <= w / max(h, 1) <= 6.0):
            continue
        if near_white[y:y + h, x:x + w].mean() < 0.03:
            continue
        out.append((int(x), int(y), int(w), int(h)))
    return out


def detect_light_boxes(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """흰/회색 라벨 박스. 어두운 테두리로 둘러싸인 밝은 사각형.

    흰 배경 위에 놓이는 경우가 많아 색만으로는 배경과 구분되지 않는다.
    테두리(어두운 닫힌 사각형)를 찾아 안쪽이 밝고 글씨가 있는지로 판정한다.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 165).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if not (250 <= area <= 9000):
            continue
        if not (1.1 <= w / max(h, 1) <= 6.0):
            continue
        if not (9 <= h <= 40):
            continue
        # 테두리 안쪽이 밝아야 한다
        pad = 3
        inner = gray[y + pad:y + h - pad, x + pad:x + w - pad]
        if inner.size < 20 or inner.mean() < 175:
            continue
        # 안쪽에 글씨(어두운 픽셀)가 있어야 한다
        if (inner < 130).mean() < 0.04:
            continue
        # 채도가 낮아야 한다 (빨간 박스 제외)
        patch = rgb[y:y + h, x:x + w].astype(np.int16)
        sat = patch.max(axis=2) - patch.min(axis=2)
        if sat.mean() > 45:
            continue
        out.append((int(x), int(y), int(w), int(h)))

    return _dedup(out)


def _dedup(boxes: list, iou_thr: float = 0.5) -> list:
    """겹치는 박스를 하나로 합친다 (테두리 안팎이 따로 잡히는 경우)."""
    boxes = sorted(boxes, key=lambda b: -b[2] * b[3])
    kept: list = []
    for b in boxes:
        bx, by, bw, bh = b
        dup = False
        for k in kept:
            kx, ky, kw, kh = k
            ix = max(0, min(bx + bw, kx + kw) - max(bx, kx))
            iy = max(0, min(by + bh, ky + kh) - max(by, ky))
            inter = ix * iy
            if inter / min(bw * bh, kw * kh) > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


# ─────────────────────────────────────────────────────────────────
def _seed_points(
    lead: np.ndarray, box: tuple, pad: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """박스 바로 바깥에서 지시선이 시작하는 지점들."""
    bx, by, bw, bh = box
    h, w = lead.shape
    x0, y0 = max(bx - pad, 0), max(by - pad, 0)
    x1, y1 = min(bx + bw + pad, w), min(by + bh + pad, h)

    sub = lead[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return None
    xs = xs + x0
    ys = ys + y0
    # 박스 안쪽 픽셀은 뺀다 (글씨나 테두리에 걸린 것)
    outside = ~((xs >= bx) & (xs < bx + bw) & (ys >= by) & (ys < by + bh))
    if outside.sum() == 0:
        return None
    return xs[outside], ys[outside]


def _walk_leader(
    lead: np.ndarray,
    box: tuple,
    pad: int = 14,
    step: float = 2.0,
    snap: float = 3.5,
    max_steps: int = 900,
) -> tuple[tuple[int, int], float] | None:
    """지시선을 박스에서 바깥쪽으로 한 걸음씩 따라간다.

    [왜 이렇게 하는가]
    처음에는 '지시선 덩어리에서 박스로부터 가장 먼 픽셀' 을 끝점으로 삼았다.
    그런데 지시선끼리 교차하면 두 선이 하나의 덩어리로 붙어버려,
    엉뚱한 선의 끝을 잡는다. 실제로 라벨 3·4·8 이 모두 같은 지점을
    가리키는 문제가 있었다.

    지시선은 거의 직선이므로, 시작 방향으로 계속 걸어가면서
    지시선 픽셀에만 붙어 있으면 교차점에서도 원래 가던 방향을 유지한다.
    """
    seeds = _seed_points(lead, box, pad)
    if seeds is None:
        return None
    sx, sy = seeds

    bx, by, bw, bh = box
    cx, cy = bx + bw / 2.0, by + bh / 2.0

    # 박스에서 가장 멀리 있는 시작점 무리를 쓴다
    d = np.hypot(sx - cx, sy - cy)
    far = d >= max(d.max() - 4.0, 0.0)
    px, py = float(sx[far].mean()), float(sy[far].mean())

    vx, vy = px - cx, py - cy
    norm = float(np.hypot(vx, vy))
    if norm < 1e-6:
        return None
    vx, vy = vx / norm, vy / norm

    ys_all, xs_all = np.nonzero(lead)
    if len(xs_all) == 0:
        return None
    pts = np.stack([xs_all, ys_all], axis=1).astype(np.float32)

    h, w = lead.shape
    last = (px, py)
    for _ in range(max_steps):
        nx, ny = last[0] + vx * step, last[1] + vy * step
        if not (0 <= nx < w and 0 <= ny < h):
            break
        dist = np.hypot(pts[:, 0] - nx, pts[:, 1] - ny)
        j = int(np.argmin(dist))
        if dist[j] > snap:
            break
        newp = (float(pts[j, 0]), float(pts[j, 1]))
        mx, my = newp[0] - last[0], newp[1] - last[1]
        mnorm = float(np.hypot(mx, my))
        if mnorm > 1e-6:
            # 방향은 거의 유지하고 아주 조금만 보정한다.
            # 교차점에서 다른 선으로 갈아타지 않게 하려는 것이다.
            vx, vy = 0.88 * vx + 0.12 * (mx / mnorm), 0.88 * vy + 0.12 * (my / mnorm)
            n2 = float(np.hypot(vx, vy))
            vx, vy = vx / n2, vy / n2
        last = newp

    return (int(round(last[0])), int(round(last[1]))), float(np.hypot(
        last[0] - cx, last[1] - cy))


def find_anchors(rgb: np.ndarray, search_pad: int = 14) -> list[LabelAnchor]:
    """라벨 박스를 찾고 각각의 지시선 끝점을 추적한다."""
    lead = detect_leader_lines(rgb)

    boxes = [(b, "red") for b in detect_red_boxes(rgb)]
    boxes += [(b, "light") for b in detect_light_boxes(rgb)]

    anchors: list[LabelAnchor] = []
    for i, (box, kind) in enumerate(boxes, start=1):
        bx, by, bw, bh = box
        cx, cy = bx + bw / 2.0, by + bh / 2.0

        walked = _walk_leader(lead, box, pad=search_pad)
        if walked is None:
            pt, length = (int(round(cx)), int(round(cy))), 0.0
        else:
            pt, length = walked

        anchors.append(LabelAnchor(
            label_id=i, kind=kind,
            box_x=bx, box_y=by, box_w=bw, box_h=bh,
            box_cx=round(cx, 1), box_cy=round(cy, 1),
            anchor_x=pt[0], anchor_y=pt[1],
            leader_len=round(length, 1),
        ))
    return anchors


def draw_anchors(rgb: np.ndarray, anchors: list, font_scale: float = 0.5) -> np.ndarray:
    """라벨에 번호를 찍은 검증용 이미지. 값을 사람이 읽어 넣을 때 쓴다."""
    out = rgb.copy()
    for a in anchors:
        color = (200, 0, 0) if a.kind == "red" else (0, 110, 0)
        cv2.rectangle(out, (a.box_x, a.box_y),
                      (a.box_x + a.box_w, a.box_y + a.box_h), color, 2)
        cv2.circle(out, (a.anchor_x, a.anchor_y), 4, (255, 0, 255), -1)
        cv2.line(out, (int(a.box_cx), int(a.box_cy)),
                 (a.anchor_x, a.anchor_y), (255, 0, 255), 1)
        txt = str(a.label_id)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        tx, ty = a.box_x, max(a.box_y - 4, th + 2)
        cv2.rectangle(out, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (255, 255, 0), -1)
        cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    return out


__all__ = [
    "LabelAnchor", "detect_red_boxes", "detect_light_boxes",
    "find_anchors", "draw_anchors",
]
