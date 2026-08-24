"""제로라인을 '선'으로 뽑는다 — 보정시트와 같은 형태.

[왜 다시 만드는가]
보정시트를 보면 제로라인은 **면이 아니라 선**이다.
DASH UPR 시트에는 빨간 곡선 하나가 부품을 가로지르고, 양 끝에
`"0" LINE` 이 붙어 있다. 그 선 안쪽에만 보정값(-0.4, -0.7 ...)이 적히고
바깥쪽은 손대지 않는다. 즉 제로라인은 **보정 영역의 경계선**이다.

그런데 지금까지는 |편차| <= 허용오차 인 픽셀을 모두 모아 '면' 으로 냈다.
그래서 조각이 수십 개로 흩어지고 알아보기 어려웠다.

[선을 잡는 기준]
편차의 **부호가 바뀌는 곳**이다. + 에서 - 로, 또는 - 에서 + 로 넘어가는
경계가 곧 편차 0 인 자리다. 허용오차 같은 임의의 값이 필요 없고,
누가 계산해도 같은 자리가 나온다.

[평활화가 필요한 이유]
원본 편차장을 그대로 쓰면 부호 경계가 30조각으로 부서진다. 스캔 이미지의
색 양자화와 국소 잡음 때문에 0 근처에서 부호가 잘게 요동치기 때문이다.
가우시안으로 넓게 흐린 뒤 부호를 보면 큰 흐름만 남아 시트처럼 몇 가닥의
연속선이 된다. sigma 는 "이 정도 크기 미만의 요철은 제로라인으로 보지
않는다" 는 뜻이라, 값 자체가 판단 기준이 된다.

    sigma  5 → 조각 30개
    sigma 15 → 조각  4개   (기본값)
    sigma 30 → 선이 거의 사라짐
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class ZeroPolyline:
    """제로라인 한 가닥."""

    line_id: int
    points: list              # [[x, y], ...] 순서대로 이어진 점
    length_px: float
    start_xy: list            # 시작점 — 시트의 `"0" LINE` 라벨 위치
    end_xy: list              # 끝점
    mean_abs_deviation: float # 선 위 평균 |편차|. 0에 가까울수록 신뢰도 높음
    is_closed: bool           # 고리 모양이면 True (시작=끝)

    def to_dict(self) -> dict:
        return asdict(self)


def _trace_skeleton(skel: np.ndarray) -> list:
    """1픽셀 뼈대를 순서 있는 점 목록으로 잇는다.

    끝점(이웃이 1개인 픽셀)에서 출발해 따라간다. 끝점이 없으면
    고리 모양이므로 아무 점에서나 시작한다.
    """
    pts = set(map(tuple, np.argwhere(skel > 0)))   # (y, x)
    if not pts:
        return []

    def neighbors(p):
        y, x = p
        return [(y + dy, x + dx)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in pts]

    paths: list = []
    while pts:
        ends = [p for p in pts if len(neighbors(p)) == 1]
        start = ends[0] if ends else next(iter(pts))

        path = [start]
        pts.discard(start)
        cur = start
        while True:
            nxt = neighbors(cur)
            if not nxt:
                break
            # 갈래가 여러 개면 방향이 가장 안 꺾이는 쪽으로 간다
            if len(nxt) > 1 and len(path) >= 2:
                py, px = path[-2]
                cy, cx = cur
                vy, vx = cy - py, cx - px
                nxt.sort(key=lambda q: -((q[0] - cy) * vy + (q[1] - cx) * vx))
            cur = nxt[0]
            path.append(cur)
            pts.discard(cur)
        if len(path) >= 5:
            paths.append(path)
    return paths


def _thin(mask: np.ndarray) -> np.ndarray:
    """부호 경계 띠를 1픽셀 선으로 얇게 만든다."""
    try:
        return cv2.ximgproc.thinning(mask.astype(np.uint8) * 255)
    except AttributeError:
        pass
    # opencv-contrib 이 없을 때를 위한 Zhang-Suen 대체 구현
    img = (mask > 0).astype(np.uint8)
    for _ in range(120):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            B = (P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9).astype(np.int16)
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros(img.shape, np.int16)
            for i in range(8):
                A += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16)
            c1 = P2 * P4 * P6 if step == 0 else P2 * P4 * P8
            c2 = P4 * P6 * P8 if step == 0 else P2 * P6 * P8
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    return img * 255


def extract_zero_polylines(
    values: np.ndarray,
    part_mask: np.ndarray,
    smooth_sigma: float = 15.0,
    min_length_px: int = 60,
    simplify_eps: float = 2.0,
    max_mean_abs_deviation: float | None = 0.6,
) -> list:
    """편차 부호가 바뀌는 경계를 순서 있는 폴리라인으로 뽑는다.

    Args:
        values:        편차값 배열
        part_mask:     부품 영역
        smooth_sigma:  이 크기 미만의 요철은 제로라인으로 보지 않는다
        min_length_px: 이보다 짧은 선은 버린다
        simplify_eps:  폴리라인 단순화 강도 (px)
        max_mean_abs_deviation:
            선 위 평균 |편차| 가 이 값을 넘으면 버린다. 제목 글자나
            마스킹 박스처럼 히트맵이 아닌 영역이 부품으로 잘못 잡히면
            그 테두리가 부호 경계처럼 보이는데, 실제 편차 0 선이 아니므로
            평균 |편차| 가 크게 나온다. 이것으로 걸러낸다.
            기본값 0.6 은 정규화 단위 기준이며, 실제 스캔 3장에서
            제목 글자·마스킹 박스로 생긴 가짜 선을 모두 걸러냈다.
    """
    smooth = cv2.GaussianBlur(values.astype(np.float32), (0, 0), smooth_sigma)
    pos = ((smooth > 0) & part_mask).astype(np.uint8)
    neg = ((smooth < 0) & part_mask).astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    band = (cv2.dilate(pos, k) > 0) & (cv2.dilate(neg, k) > 0) & part_mask
    if not band.any():
        return []

    skel = _thin(band)

    lines: list = []
    for path in _trace_skeleton(skel):
        pts = np.array([[x, y] for y, x in path], dtype=np.int32)
        if len(pts) < 3:
            continue
        # 꺾임점만 남겨 매끈하게
        simplified = cv2.approxPolyDP(pts.reshape(-1, 1, 2), simplify_eps, False)
        simplified = simplified.reshape(-1, 2)
        if len(simplified) < 2:
            continue

        seg = np.diff(simplified.astype(np.float64), axis=0)
        length = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        if length < min_length_px:
            continue

        vals = values[pts[:, 1], pts[:, 0]]
        mean_abs = float(np.abs(vals).mean())
        if max_mean_abs_deviation is not None and mean_abs > max_mean_abs_deviation:
            continue
        start, end = simplified[0], simplified[-1]
        lines.append(ZeroPolyline(
            line_id=0,
            points=simplified.tolist(),
            length_px=round(length, 1),
            start_xy=[int(start[0]), int(start[1])],
            end_xy=[int(end[0]), int(end[1])],
            mean_abs_deviation=round(mean_abs, 4),
            is_closed=bool(np.hypot(*(start - end)) < 8),
        ))

    lines.sort(key=lambda l: l.length_px, reverse=True)
    for i, l in enumerate(lines, start=1):
        l.line_id = i
    return lines


def draw_zero_polylines(
    rgb: np.ndarray,
    lines: list,
    color: tuple = (220, 20, 20),
    thickness: int = 3,
    label_ends: bool = True,
    max_labeled: int = 3,
) -> np.ndarray:
    """보정시트처럼 그린다. 빨간 선 + 양 끝에 `"0" LINE` 표기.

    선이 여러 가닥이면 라벨을 다 붙이는 순간 화면이 글자로 덮인다.
    그래서 긴 것부터 max_labeled 가닥까지만 라벨을 붙이고,
    나머지는 선만 그린다.
    """
    out = rgb.copy()
    for line in lines:
        pts = np.asarray(line.points, dtype=np.int32)
        cv2.polylines(out, [pts], line.is_closed, color, thickness, cv2.LINE_AA)

    if not label_ends:
        return out

    placed: list = []          # 이미 라벨을 놓은 자리 (겹침 방지)

    def put(x: int, y: int) -> None:
        text = '"0" LINE'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        for dx, dy in ((14, -14), (14, 20), (-tw - 20, -14), (-tw - 20, 20)):
            bx, by = x + dx, y + dy
            if not (0 <= bx and bx + tw < out.shape[1] and th < by < out.shape[0]):
                continue
            if any(abs(bx - px) < tw + 10 and abs(by - py) < th + 10
                   for px, py in placed):
                continue
            cv2.line(out, (x, y), (bx, by), color, 1, cv2.LINE_AA)
            cv2.rectangle(out, (bx - 4, by - th - 6), (bx + tw + 4, by + 6),
                          (255, 255, 255), -1)
            cv2.rectangle(out, (bx - 4, by - th - 6), (bx + tw + 4, by + 6),
                          color, 1)
            cv2.putText(out, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
            placed.append((bx, by))
            break
        cv2.circle(out, (x, y), 4, color, -1, cv2.LINE_AA)

    for line in lines[:max_labeled]:
        if line.is_closed:
            continue
        put(int(line.start_xy[0]), int(line.start_xy[1]))
        put(int(line.end_xy[0]), int(line.end_xy[1]))
    return out


__all__ = ["ZeroPolyline", "extract_zero_polylines", "draw_zero_polylines"]
