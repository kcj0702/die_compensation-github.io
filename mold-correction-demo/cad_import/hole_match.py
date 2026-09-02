"""스캔 이미지의 구멍과 CAD 조립 홀을 짝지어 자세를 잡는다.

[왜 만들었나]
지금까지 CAD 와 스캔을 **바깥 윤곽(실루엣)** 으로만 맞췄다. 그런데 윤곽은
정보가 얇다. 실측 71XX2(센터 필러)는 -90~+90도를 5도 간격으로 전부 훑어도
겹침이 **최대 57.5%** 에서 막혔다 — 알고리즘을 더 만져도 안 넘는 벽이다.

구멍은 다르다. 위치가 콕 집히는 특징점이라 세 개만 제대로 짝지어도 자세가
정해진다. 그리고 우리는 양쪽 다 이미 갖고 있다 —

    스캔 이미지 속 구멍   64XX2 42개 · 67XX6 32개 · 71XX2 97개
    CAD 조립 홀           64XX2 43개 · 67XX6 180개 · 71XX2 44개

[개수가 안 맞아도 된다]
64XX2 는 42 대 43 으로 거의 일대일이지만 나머지는 어긋난다. 67XX6 은 작은
홀이 스캔 해상도에서 안 뚫려 보이고, 71XX2 는 스캔 결손이 구멍처럼 잡혀
CAD 보다 많다.

그래도 상관없다. **세 개만 맞으면 되고 나머지는 틀려도 된다.** 무작위로
짝을 지어 보고 가장 많은 구멍이 들어맞는 조합을 고른다(RANSAC). 가짜가
섞여 있어도 진짜가 몇 개 있으면 걸린다.

[왜 닮음변환인가]
스캔 그림이 정투영이라고 보면 CAD 를 평면에 투영한 것과 스캔은
**회전 + 균일 배율 + 평행이동** 으로 이어진다. 원근이라면 이걸로는 안
맞는데, 그때는 들어맞는 구멍 수가 적게 나와서 결과로 드러난다 —
정확도를 주장하지 말고 재서 돌려주는 이유다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

# 이보다 작은 구멍은 잡음으로 본다. 부품 넓이 대비 비율이다.
MIN_HOLE_AREA_RATIO = 0.00015
# 짝을 지어 볼 구멍 수 상한. 전부 쓰면 조합이 폭발한다.
MAX_SCAN_HOLES = 18
MAX_CAD_HOLES = 24
# 들어맞았다고 볼 거리. 부품 대각선 대비 비율이다.
INLIER_RATIO = 0.02
# 이만큼은 들어맞아야 자세를 인정한다.
MIN_INLIERS = 3


@dataclass
class HoleFit:
    """구멍으로 잡은 자세."""

    axis: int                  # 어느 축에서 내려다봤나
    mirrored: bool
    scale: float               # mm/px
    angle: float               # 라디안
    offset: list               # 화면 픽셀
    inliers: int
    rmse: float                # 들어맞은 구멍들의 잔차 (px)
    scan_holes: int
    cad_holes: int
    pairs: list = field(default_factory=list)   # (스캔 번호, CAD 번호)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_holes(part_mask) -> tuple:
    """부품 마스크 안의 구멍을 찾는다.

    바깥을 물로 채우면 남는 빈 곳이 곧 부품 안의 구멍이다.

    Returns:
        (중심 (N,2) 픽셀, 넓이 (N,))  — 넓은 순.
    """
    solid = (np.asarray(part_mask) > 0).astype(np.uint8)
    area_part = int(solid.sum())
    if not area_part:
        return np.zeros((0, 2)), np.zeros(0)

    padded = cv2.copyMakeBorder(solid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    scratch = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, scratch, (0, 0), 1)
    inner = (padded[1:-1, 1:-1] == 0).astype(np.uint8)

    count, _labels, stats, centres = cv2.connectedComponentsWithStats(
        inner, connectivity=8)
    keep = [i for i in range(1, count)
            if stats[i][4] >= area_part * MIN_HOLE_AREA_RATIO]
    keep.sort(key=lambda i: -stats[i][4])
    if not keep:
        return np.zeros((0, 2)), np.zeros(0)
    return (np.asarray([centres[i] for i in keep], dtype=float),
            np.asarray([stats[i][4] for i in keep], dtype=float))


def _similarity(src: np.ndarray, dst: np.ndarray) -> tuple:
    """두 점 무리를 겹치는 2D 닮음변환 (회전 + 균일 배율 + 평행이동).

    두 점이면 정확히 하나로 정해지고, 그보다 많으면 최소제곱이다.
    """
    src_mid, dst_mid = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - src_mid, dst - dst_mid
    denominator = float((a ** 2).sum())
    if denominator < 1e-12:
        return None
    # 복소수로 보면 닮음변환이 곱셈 하나다
    za = a[:, 0] + 1j * a[:, 1]
    zb = b[:, 0] + 1j * b[:, 1]
    factor = complex(np.vdot(za, zb) / denominator)
    if abs(factor) < 1e-12:
        return None
    scale = float(abs(factor))
    angle = float(np.angle(factor))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    offset = dst_mid - scale * (rotation @ src_mid)
    return scale, angle, offset


def _apply(points: np.ndarray, scale: float, angle: float,
           offset: np.ndarray) -> np.ndarray:
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return scale * (points @ rotation.T) + offset


def _unique_pairs(moved: np.ndarray, scan: np.ndarray, limit: float) -> list:
    """CAD 홀과 스캔 구멍을 **일대일**로 짝짓는다.

    가까운 것부터 탐욕적으로 배정하고, 이미 쓴 구멍은 다시 안 쓴다.
    이걸 안 하면 배율이 무너진 가짜 자세가 이긴다 — 실측에서 배율이
    0 에 가까워지자 CAD 홀 전부가 스캔 구멍 하나로 뭉쳐 "24개 들어맞음
    (잔차 0.6px)" 이 나왔는데 실제 겹침은 0% 였다. 다대일을 허용하면
    지표가 속는다.
    """
    gaps = ((moved[:, None, :] - scan[None, :, :]) ** 2).sum(axis=2)
    order = np.dstack(np.unravel_index(np.argsort(gaps, axis=None), gaps.shape))[0]
    used_cad: set = set()
    used_scan: set = set()
    pairs: list = []
    for cad_i, scan_i in order:
        if gaps[cad_i, scan_i] > limit:
            break
        if int(cad_i) in used_cad or int(scan_i) in used_scan:
            continue
        used_cad.add(int(cad_i)); used_scan.add(int(scan_i))
        pairs.append((int(scan_i), int(cad_i), float(gaps[cad_i, scan_i])))
    return pairs


def match(scan_xy: np.ndarray, cad_xy: np.ndarray,
          tolerance: float, scale_hint: float | None = None) -> tuple:
    """구멍 두 쌍씩 짝지어 보고 가장 많이 들어맞는 자세를 고른다.

    Args:
        scale_hint: 기대 배율(px/mm). 부품 크기와 마스크 크기에서 나온다.
            이것과 크게 다른 배율은 버린다 — 배율이 자유면 무너진 자세가
            이긴다(위 _unique_pairs 참고).

    Returns:
        (들어맞은 수, rmse, scale, angle, offset, 짝 목록) 또는 None.
    """
    scan = np.asarray(scan_xy, dtype=float)[:MAX_SCAN_HOLES]
    cad = np.asarray(cad_xy, dtype=float)[:MAX_CAD_HOLES]
    if len(scan) < 2 or len(cad) < 2:
        return None

    best = None
    limit = tolerance * tolerance
    for i in range(len(cad)):
        for j in range(i + 1, len(cad)):
            base = cad[[i, j]]
            base_gap = float(np.linalg.norm(cad[i] - cad[j]))
            if base_gap < 1e-6:
                continue
            for p in range(len(scan)):
                for q in range(len(scan)):
                    if p == q:
                        continue
                    pair_gap = float(np.linalg.norm(scan[p] - scan[q]))
                    if pair_gap < 1e-6:
                        continue
                    guess = _similarity(base, scan[[p, q]])
                    if guess is None:
                        continue
                    scale, angle, offset = guess
                    if scale_hint is not None and not (
                            scale_hint * 0.6 <= scale <= scale_hint * 1.7):
                        continue      # 무너진 배율
                    moved = _apply(cad, scale, angle, offset)
                    matched = _unique_pairs(moved, scan, limit)
                    count = len(matched)
                    if count < MIN_INLIERS:
                        continue
                    rmse = float(np.sqrt(
                        np.mean([g for _s, _c, g in matched])))
                    key = (count, -rmse)
                    if best is None or key > best[0]:
                        picks = [(s_i, c_i) for s_i, c_i, _g in matched]
                        best = (key, count, rmse, scale, angle, offset, picks)
    if best is None:
        return None
    _key, count, rmse, scale, angle, offset, picks = best

    # 들어맞은 것 전부로 다시 맞춘다 — 두 점만으로 잡은 자세를 다듬는다
    if len(picks) >= 3:
        cad_pts = np.asarray([cad[c] for _s, c in picks], dtype=float)
        scan_pts = np.asarray([scan[s] for s, _c in picks], dtype=float)
        better = _similarity(cad_pts, scan_pts)
        if better is not None:
            scale, angle, offset = better
            moved = _apply(cad_pts, scale, angle, offset)
            rmse = float(np.sqrt(((moved - scan_pts) ** 2).sum(axis=1).mean()))
    return count, rmse, scale, angle, offset, picks


def fit_by_holes(part_mask, hole_centres_3d) -> HoleFit | None:
    """스캔 마스크와 CAD 홀 좌표로 자세를 잡는다.

    Args:
        part_mask: 스캔의 부품 마스크.
        hole_centres_3d: CAD 조립 홀 중심 (M,3) — step_reader 가 뽑은 것.

    Returns:
        HoleFit 또는 None(짝을 못 지었을 때).
    """
    scan_xy, _areas = detect_holes(part_mask)
    holes = np.asarray(hole_centres_3d, dtype=float).reshape(-1, 3)
    if len(scan_xy) < MIN_INLIERS or len(holes) < MIN_INLIERS:
        return None

    height, width = np.asarray(part_mask).shape
    tolerance = float(np.hypot(width, height)) * INLIER_RATIO

    best: HoleFit | None = None
    for axis in (0, 1, 2):
        plane = [k for k in (0, 1, 2) if k != axis]
        for mirrored in (False, True):
            flat = holes[:, plane].copy()
            if mirrored:
                flat[:, 0] = -flat[:, 0]
            # 기대 배율: 마스크 대각선(px) / CAD 투영 대각선(mm).
            # 부품 전체가 화면에 들어와 있다는 가정이고, 0.6~1.7배의
            # 여유를 둔다(match 안에서 거른다).
            span = flat.max(axis=0) - flat.min(axis=0)
            cad_diag = float(np.hypot(*span)) or 1.0
            hint = float(np.hypot(width, height)) / cad_diag
            got = match(scan_xy, flat, tolerance, scale_hint=hint)
            if got is None:
                continue
            count, rmse, scale, angle, offset, picks = got
            if best is not None and (count, -rmse) <= (best.inliers, -best.rmse):
                continue
            best = HoleFit(
                axis=axis, mirrored=mirrored,
                scale=round(float(scale), 6), angle=round(float(angle), 6),
                offset=[round(float(v), 3) for v in offset],
                inliers=count, rmse=round(rmse, 3),
                scan_holes=len(scan_xy), cad_holes=len(holes),
                pairs=[[int(s), int(c)] for s, c in picks],
            )
    return best


__all__ = ["HoleFit", "MIN_INLIERS", "detect_holes", "fit_by_holes", "match"]
