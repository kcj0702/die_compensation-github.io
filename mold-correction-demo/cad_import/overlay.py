"""스캔 화면의 2D 좌표를 CAD 표면 위의 3D 점으로 옮긴다.

[무엇을 하는가]
제로라인과 보정 포인트는 스캔 히트맵의 **픽셀 좌표**로 나온다. CAD 는
**부품 좌표(mm)** 다. 둘을 잇지 않으면 3D 화면에 아무것도 못 얹는다.

[어떻게 잇는가 — 실루엣 맞추기]
스캔 히트맵은 부품을 한 방향에서 내려다본 그림이다. 판재는 한 축으로
얇으니 그 축이 보는 방향일 가능성이 높다. 그래서

  1. 축 6가지(+-X, +-Y, +-Z)로 CAD 를 눌러 실루엣을 만든다
  2. 좌우/상하 뒤집기 4가지를 곱해 24가지 후보를 만든다
     (보정시트가 좌우반전으로 그려진 사례가 실제로 있었다)
  3. 각 후보를 스캔 마스크에 크기·위치를 맞춘 뒤 겹침(IoU)을 잰다
  4. 가장 잘 겹치는 것을 고른다
  5. 2D 점마다 그 방향으로 광선을 쏴 표면에 닿는 3D 점을 얻는다

[한계 — 이것은 계측이 아니다]
데이텀(조립 홀·기준면)으로 맞춘 정합이 아니라 **겉모양으로 맞춘
근사**다. 화면에서 "대략 이 자리" 를 보여주는 용도이고, 보정량을
숫자로 다시 재는 데 쓰면 안 된다. 제대로 하려면 같은 부품의 3D 스캔
원본이 있어야 한다(그래야 데이텀 정렬이 가능하다).

fit.iou 로 얼마나 믿을 만한지 함께 내보내니, 화면에서도 그 값을
보여주고 낮으면 경고하는 쪽이 맞다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

# 실루엣을 맞출 때 쓰는 격자 크기. 원본 해상도로 하면 느리고,
# 이 정도면 어느 축인지 고르는 데 충분하다.
FIT_GRID = 256
# 이보다 겹침이 낮으면 맞췄다고 보기 어렵다.
# 바깥 윤곽(구멍을 메운 실루엣) 기준이다 — 아래 설명 참고.
MIN_IOU = 0.75


@dataclass
class ViewFit:
    """스캔 화면 -> 부품 좌표 변환."""

    axis: int              # 0=X, 1=Y, 2=Z — 이 축을 따라 내려다본다
    sign: int              # +1 이면 축의 양방향에서 본다
    flip_u: bool           # 화면 가로를 뒤집었나
    flip_v: bool           # 화면 세로를 뒤집었나
    mm_per_px: float
    origin_u: float        # 화면 (0,0) 에 대응하는 부품 좌표
    origin_v: float
    iou: float             # 바깥 윤곽 겹침 — 자리를 맞췄는지
    detail_iou: float = 0.0  # 구멍까지 포함한 겹침 — 형상이 같은지
    reliable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _hull(binary: np.ndarray) -> np.ndarray:
    """가장 큰 덩어리의 볼록 껍질을 채운다 — 부품이 차지한 자리.

    [왜 껍질인가 — 실측 JD_67XX6]
    선루프 프레임처럼 가운데가 뚫린 부품은 안쪽 테두리가 조금만 어긋나도
    IoU 가 급락한다. 스캔(DR000)과 CAD(DR050)는 공정 단계가 달라 안쪽
    테두리가 다른데, 바깥 윤곽은 잘 맞는다. 원본끼리 재면 0.41 이라
    "못 맞췄다" 가 되어 버린다.

    구멍만 메우는 것으로는 부족했다. 스캔 마스크의 링이 끊겨 있어서
    바깥 윤곽을 따도 띠 모양 그대로였다(면적이 격자의 13.5%). 모폴로지
    닫기로도 16~19% 밖에 안 찼다 — 틈이 크다.

    볼록 껍질은 끊긴 링에도 흔들리지 않는다. 실측 —

        원본끼리        Z축 0.413   (X축 0.277 과 아슬아슬)
        구멍 메우기     Z축 0.219   (오히려 X축에 짐)
        볼록 껍질       Z축 0.946   (2위와 확실히 벌어짐)

    다만 껍질만 보면 좌우·상하 뒤집기 4가지가 전부 같은 점수가 나온다.
    방향은 원본 겹침으로 따로 가른다(fit_view 참고).
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [cv2.convexHull(biggest)], -1, 255, cv2.FILLED)
    return filled


def _plane_axes(axis: int) -> tuple:
    """보는 축을 뺀 나머지 두 축. (가로, 세로) 순서."""
    return {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]


def _rasterize(points_2d: np.ndarray, faces: np.ndarray, grid: int) -> tuple:
    """투영된 삼각형을 격자에 채워 실루엣을 만든다."""
    lo = points_2d.min(axis=0)
    hi = points_2d.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = (grid - 1) / span.max()
    pixels = (points_2d - lo) * scale

    canvas = np.zeros((grid, grid), np.uint8)
    triangles = pixels[faces].astype(np.int32)
    cv2.fillPoly(canvas, triangles, 255)
    return canvas, lo, scale


def _mask_to_grid(mask: np.ndarray, grid: int) -> tuple:
    """스캔 마스크를 같은 격자에 올린다 — 바운딩 박스를 꽉 채워서."""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if not len(xs):
        raise ValueError("부품 마스크가 비어 있습니다.")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cropped = (np.asarray(mask)[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8) * 255

    height, width = cropped.shape
    scale = (grid - 1) / max(height, width)
    resized = cv2.resize(cropped, (max(int(width * scale), 1),
                                   max(int(height * scale), 1)),
                         interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((grid, grid), np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return canvas, (x0, y0, x1, y1)


def fit_view(vertices: np.ndarray, faces: np.ndarray, part_mask) -> ViewFit:
    """CAD 실루엣을 스캔 마스크에 맞춰 어느 방향에서 본 그림인지 찾는다."""
    mask_grid, (mx0, my0, mx1, my1) = _mask_to_grid(part_mask, FIT_GRID)
    mask_bool = mask_grid > 0
    mask_solid = _hull(mask_grid) > 0

    best: ViewFit | None = None
    best_key = (-1.0, -1.0)

    for axis in (0, 1, 2):
        u_axis, v_axis = _plane_axes(axis)
        for sign in (1, -1):
            for flip_u in (False, True):
                for flip_v in (False, True):
                    u = vertices[:, u_axis] * (sign if not flip_u else -sign)
                    v = vertices[:, v_axis] * (-1 if flip_v else 1)
                    projected = np.stack([u, v], axis=1)
                    canvas, lo, scale = _rasterize(projected, faces, FIT_GRID)
                    shape_solid = _hull(canvas) > 0
                    union = int((shape_solid | mask_solid).sum())
                    if union == 0:
                        continue
                    # 자리는 껍질로, 방향은 원본 겹침으로 가른다.
                    hull_iou = float((shape_solid & mask_solid).sum()) / union
                    shape_bool = canvas > 0
                    detail_union = int((shape_bool | mask_bool).sum())
                    detail_iou = (float((shape_bool & mask_bool).sum())
                                  / detail_union) if detail_union else 0.0
                    key = (round(hull_iou, 3), detail_iou)
                    if key <= best_key:
                        continue
                    best_key = key

                    # 화면 픽셀 -> 부품 좌표. 마스크 바운딩 박스를 실루엣
                    # 바운딩 박스에 맞춘 것이므로 배율은 두 폭의 비다.
                    mask_width = max(mx1 - mx0, my1 - my0) + 1
                    span = float(np.maximum(
                        projected.max(axis=0) - projected.min(axis=0), 1e-9).max())
                    best = ViewFit(
                        axis=axis, sign=sign, flip_u=flip_u, flip_v=flip_v,
                        mm_per_px=span / mask_width,
                        origin_u=float(lo[0] - mx0 * span / mask_width),
                        origin_v=float(lo[1] - my0 * span / mask_width),
                        iou=round(hull_iou, 4),
                        detail_iou=round(detail_iou, 4),
                    )
    if best is None:
        raise ValueError("CAD 실루엣을 스캔에 맞추지 못했습니다.")
    best.reliable = best.iou >= MIN_IOU
    return best


def unproject(points_px, vertices: np.ndarray, faces: np.ndarray,
              fit: ViewFit, mesh=None) -> list:
    """화면 좌표를 표면 위의 3D 점으로 바꾼다.

    보는 축을 따라 광선을 쏴서 처음 닿는 면을 쓴다. 못 맞으면 가장
    가까운 정점으로 대신한다 — 형상 밖으로 조금 벗어난 점도 화면에
    남겨야 사용자가 "왜 여기가 비었지" 를 알 수 있다.
    """
    points_px = np.asarray(points_px, dtype=float).reshape(-1, 2)
    if not len(points_px):
        return []

    u_axis, v_axis = _plane_axes(fit.axis)
    u = fit.origin_u + points_px[:, 0] * fit.mm_per_px
    v = fit.origin_v + points_px[:, 1] * fit.mm_per_px

    # 투영 평면 좌표를 부품 좌표로 되돌린다 (fit_view 의 부호 규칙을 뒤집는다)
    u_part = u * (fit.sign if not fit.flip_u else -fit.sign)
    v_part = v * (-1 if fit.flip_v else 1)

    depth_lo = float(vertices[:, fit.axis].min())
    depth_hi = float(vertices[:, fit.axis].max())
    start = depth_hi + (depth_hi - depth_lo) * 0.1

    origins = np.zeros((len(points_px), 3), dtype=float)
    origins[:, u_axis] = u_part
    origins[:, v_axis] = v_part
    origins[:, fit.axis] = start
    directions = np.zeros_like(origins)
    directions[:, fit.axis] = -1.0

    hits = [None] * len(points_px)
    if mesh is not None:
        try:
            locations, ray_index, _tri = mesh.ray.intersects_location(
                ray_origins=origins, ray_directions=directions,
                multiple_hits=False)
            for location, index in zip(locations, ray_index):
                hits[int(index)] = [round(float(c), 3) for c in location]
        except Exception:
            pass

    # 광선이 빗나간 점은 가장 가까운 정점으로 채운다
    missing = [i for i, hit in enumerate(hits) if hit is None]
    if missing:
        flat = vertices[:, [u_axis, v_axis]]
        wanted = np.stack([u_part[missing], v_part[missing]], axis=1)
        for slot, target in zip(missing, wanted):
            nearest = int(np.argmin(((flat - target) ** 2).sum(axis=1)))
            hits[slot] = [round(float(c), 3) for c in vertices[nearest]]
    return hits


__all__ = ["ViewFit", "MIN_IOU", "fit_view", "unproject"]
