"""제로라인을 부품 가장자리의 부호 전환 지점에서 시작해 찾는다.

[현장에서 확인한 기준]
2026-08-25 아진산업 방문에서 확인:
  1. 제로라인의 시작/끝점은 편차 부호가 바뀌는 지점에서 출발한다.
  2. 부품마다 제로라인이 '선' 으로 표현되기도, '면' 으로 표현되기도 한다.
  3. 현재 유일한 정답지는 실제 보정시트뿐이므로, 시트에 그려진 형태에
     최대한 맞춰야 한다.

[왜 부품 가장자리에서 찾는가]
이전에는 스캔 전체에서 부호가 바뀌는 곳을 다 찾았다. 그런데 복잡한
부품(DASH UPR)에서는 실제로 여러 번 부호가 바뀌어 후보가 12개나 나왔고,
그중 어느 것이 진짜 제로라인인지 알 방법이 없었다.

RING SUNROOF 실측 보정시트를 보면 '0' 패치는 전부 **부품 테두리를 따라**
붙어 있다. 그래서 스캔 전체가 아니라 **부품의 바깥 테두리를 따라가며**
부호가 바뀌는 지점만 찾으면, 이 지점들이 실제 시트의 '0' 패치 위치와
거의 겹친다 (검증 완료 — RING SUNROOF 기준 6개 지점 모두 시트의 패치
위치와 일치).

[처리 흐름]
    1. 부품 외곽 테두리를 한 바퀴 추출한다
    2. 테두리를 따라 편차값을 채취하고, 테두리 진행 방향으로 스무딩한다
       (리브·나사 구멍 같은 국소 디테일의 잡음을 없애기 위해)
    3. 스무딩된 값의 부호가 바뀌는 지점 = 제로라인 시작점(앵커)
    4. 각 앵커를 중심으로 |편차| 가 작은 영역을 넓혀 시트의 '0' 패치와
       같은 작은 구획을 만든다
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class ZeroAnchor:
    """부품 테두리에서 편차 부호가 바뀌는 지점."""

    anchor_id: int
    x: int
    y: int
    boundary_arclen: float    # 테두리 시작점 기준 호 길이 (px). 인접성 판단용

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ZeroPatch:
    """앵커를 중심으로 자란 0 영역 패치. 시트의 '0' 표기에 대응."""

    patch_id: int
    anchor: ZeroAnchor
    contour: list             # [[x, y], ...] 패치 외곽선
    area_px: int
    mean_abs_deviation: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["anchor"] = self.anchor.to_dict()
        return d


def find_boundary_anchors(
    values: np.ndarray,
    part_mask: np.ndarray,
    smooth_window: int = 230,
    min_separation_px: float = 40.0,
) -> list:
    """부품 테두리를 따라가며 편차 부호가 바뀌는 지점을 찾는다.

    Args:
        smooth_window: 테두리 진행 방향 스무딩 폭 (px). 리브·나사 구멍
            같은 국소 디테일을 무시하는 정도.
        min_separation_px: 이보다 가까운 전환점은 하나로 합친다.
    """
    contours, _ = cv2.findContours(
        (part_mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return []
    outer = max(contours, key=cv2.contourArea).reshape(-1, 2)
    n = len(outer)
    h, w = values.shape

    raw = np.array([
        values[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))]
        for x, y in outer
    ])

    # 원형 신호이므로 양 끝을 이어 붙여 스무딩한 뒤 가운데만 취한다
    win = max(smooth_window, 3)
    kernel = np.ones(win) / win
    padded = np.concatenate([raw[-win:], raw, raw[:win]])
    smooth = np.convolve(padded, kernel, mode="same")[win:win + n]
    signs = np.sign(smooth)

    arclen = np.concatenate([[0.0], np.cumsum(np.hypot(
        np.diff(outer[:, 0].astype(float), append=outer[0, 0]),
        np.diff(outer[:, 1].astype(float), append=outer[0, 1]),
    ))])[:n]

    anchors: list = []
    last_idx = -10**9
    for i in range(n):
        a, b = signs[i], signs[(i + 1) % n]
        if a == 0 or b == 0 or a == b:
            continue
        if i - last_idx < min_separation_px:
            continue
        x, y = outer[i]
        anchors.append(ZeroAnchor(
            anchor_id=0, x=int(x), y=int(y), boundary_arclen=float(arclen[i]),
        ))
        last_idx = i

    # 마지막 지점이 처음 지점과 너무 가까우면(원 둘레라서) 병합
    if len(anchors) >= 2:
        total = arclen[-1] if n else 0.0
        if total - anchors[-1].boundary_arclen + anchors[0].boundary_arclen < min_separation_px:
            anchors.pop()

    for i, a in enumerate(anchors, start=1):
        a.anchor_id = i
    return anchors


def grow_patches(
    values: np.ndarray,
    part_mask: np.ndarray,
    anchors: list,
    tolerance: float,
    max_radius_px: int = 55,
    min_patch_area: int = 300,
) -> list:
    """각 앵커에서 |편차| <= tolerance 인 영역을 지역적으로 넓힌다.

    전체 이미지에서 0 근처를 다 모으는 것과 다르다. 앵커 주변
    max_radius_px 안에서만 넓혀서, 시트처럼 작고 독립된 패치가 되게 한다.
    """
    h, w = values.shape
    near_zero = (np.abs(values) <= tolerance) & part_mask

    patches: list = []
    for a in anchors:
        y0, y1 = max(a.y - max_radius_px, 0), min(a.y + max_radius_px, h)
        x0, x1 = max(a.x - max_radius_px, 0), min(a.x + max_radius_px, w)
        local = near_zero[y0:y1, x0:x1].astype(np.uint8)
        if local.sum() == 0:
            continue

        n, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
        # 앵커에 가장 가까운 성분을 고른다
        ay, ax = a.y - y0, a.x - x0
        best, best_d = None, 1e18
        for i in range(1, n):
            ys, xs = np.nonzero(labels == i)
            d = np.min((ys - ay) ** 2 + (xs - ax) ** 2)
            if d < best_d:
                best_d, best = d, i
        if best is None or best_d > (max_radius_px * 0.6) ** 2:
            continue

        comp = (labels == best).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea) + [x0, y0]
        eps = 1.5
        simplified = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(simplified) < 3:
            continue

        if int(comp.sum()) < min_patch_area:
            continue

        vals = values[y0:y1, x0:x1][comp > 0]
        patches.append(ZeroPatch(
            patch_id=0, anchor=a,
            contour=simplified.tolist(),
            area_px=int(comp.sum()),
            mean_abs_deviation=round(float(np.abs(vals).mean()), 4),
        ))

    patches.sort(key=lambda p: p.anchor.anchor_id)
    for i, p in enumerate(patches, start=1):
        p.patch_id = i
    return patches


def draw_zero_boundary(
    rgb: np.ndarray,
    anchors: list,
    patches: list,
    anchor_color: tuple = (0, 200, 0),
    patch_fill: tuple = (255, 0, 200),
    patch_edge: tuple = (220, 20, 20),
    fill_alpha: float = 0.35,
) -> np.ndarray:
    """앵커(초록 점)와 패치(빨간 테두리 + 분홍 채움)를 그린다."""
    out = rgb.copy()
    if patches:
        overlay = out.copy()
        for p in patches:
            pts = np.asarray(p.contour, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], patch_fill)
        out = cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0)
        for p in patches:
            pts = np.asarray(p.contour, dtype=np.int32)
            cv2.polylines(out, [pts], True, patch_edge, 2, cv2.LINE_AA)

    for a in anchors:
        cv2.circle(out, (a.x, a.y), 7, anchor_color, -1, cv2.LINE_AA)
        cv2.circle(out, (a.x, a.y), 7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


__all__ = [
    "ZeroAnchor", "ZeroPatch",
    "find_boundary_anchors", "grow_patches", "draw_zero_boundary",
]
