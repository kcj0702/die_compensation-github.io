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
    swap: bool = False     # 화면에서 90도 돌렸나 (가로세로 맞바꿈)
    angle: float = 0.0     # 평면에서 더 돌린 각(라디안). 90도 단위가 아니다.
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


def _fit_axes(fit) -> tuple:
    """이 맞춤이 쓴 (가로, 세로) 축. swap 이면 둘을 맞바꾼다.

    fit_view · unproject · sample_deviation 이 **같은 순서**를 써야 한다.
    한 군데라도 어긋나면 정합은 맞다고 나오는데 좌표가 틀어진다.
    """
    u_axis, v_axis = _plane_axes(fit.axis)
    return (v_axis, u_axis) if getattr(fit, "swap", False) else (u_axis, v_axis)


def _principal_angle(xs: np.ndarray, ys: np.ndarray) -> float:
    """점 무리가 가장 길게 뻗은 방향(라디안).

    2차 중심 모멘트로 구한다. 180도 뒤집힌 답이 같이 나오는데, 그건
    이미 돌고 있는 뒤집기(flip_u/flip_v)가 흡수한다.
    """
    x = xs - xs.mean()
    y = ys - ys.mean()
    xx = float((x * x).mean())
    yy = float((y * y).mean())
    xy = float((x * y).mean())
    return 0.5 * float(np.arctan2(2.0 * xy, xx - yy))


def _rotate(points_2d: np.ndarray, angle: float) -> np.ndarray:
    """평면 위에서 돌린다."""
    if not angle:
        return points_2d
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.stack([
        points_2d[:, 0] * cos - points_2d[:, 1] * sin,
        points_2d[:, 0] * sin + points_2d[:, 1] * cos,
    ], axis=1)


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

    # 스캔 그림은 검사 소프트웨어에서 작업자가 놓은 각도 그대로다. 90도
    # 단위로 맞아떨어질 이유가 없다. 실측 71XX2 는 스캔과 CAD 투영의
    # 주축이 20도 넘게 어긋나 있어서, 축·부호·뒤집기·90도회전을 다 훑어도
    # 겹침이 33% 에서 멈췄다 — 실루엣 두 개가 같은 모양인데 서로 기울어
    # 가운데 대각선 띠만 겹쳤다.
    #
    # 그래서 두 실루엣의 **주축 각도 차이**만큼 돌려 놓고 잰다. 각도를
    # 훑지 않고 모멘트로 한 번에 구하므로 비용이 늘지 않는다.
    mys, mxs = np.nonzero(mask_solid)
    mask_angle = _principal_angle(mxs.astype(float), mys.astype(float))

    best: ViewFit | None = None
    best_key = (-1.0, -1.0)

    # swap = 화면에서 90도 돌려 보기.
    #
    # 이게 없어서 71XX2(센터 필러)가 25% 밖에 안 맞았다. 스캔은 필러를
    # **눕혀서** 찍었는데(983 x 568) CAD 의 Y 투영은 **서 있다**(545 x 1230).
    # 축 3개 x 부호 2 x 뒤집기 2 x 2 = 24가지를 다 봐도 가로세로를 맞바꾸는
    # 경우가 없어서, 어떤 조합으로도 맞출 수가 없었다. 뒤집기는 거울일 뿐
    # 회전이 아니다.
    for axis in (0, 1, 2):
        # 가로세로 맞바꾸기(swap)를 따로 훑을 필요가 없다 — 주축 정렬이
        # 90도 회전을 이미 포함한다. 필드는 예전 결과를 읽으려고 남겨 둔다.
        u_axis, v_axis = _plane_axes(axis)
        if True:
            swap = False
            for sign in (1, -1):
                for flip_u in (False, True):
                    for flip_v in (False, True):
                        u = vertices[:, u_axis] * (sign if not flip_u else -sign)
                        v = vertices[:, v_axis] * (-1 if flip_v else 1)
                        flat = np.stack([u, v], axis=1)
                        # 각도는 **채워진 실루엣**에서 잰다. 정점 구름으로
                        # 재면 작은 피처가 몰린 쪽으로 주축이 끌려가
                        # 스캔 마스크(픽셀은 고르게 채워진다)와 기준이
                        # 달라진다 — 그렇게 했다가 71XX2 가 30% 에
                        # 머물렀다. 그래서 한 번 그려 각도를 얻고,
                        # 돌린 뒤 다시 그린다.
                        rough, _lo0, _s0 = _rasterize(flat, faces, FIT_GRID)
                        rys, rxs = np.nonzero(_hull(rough) > 0)
                        angle = 0.0
                        if len(rxs):
                            angle = mask_angle - _principal_angle(
                                rxs.astype(float), rys.astype(float))
                        projected = _rotate(flat, angle)
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
                            swap=swap, angle=float(angle),
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

    u_axis, v_axis = _fit_axes(fit)
    turned = np.stack([
        fit.origin_u + points_px[:, 0] * fit.mm_per_px,
        fit.origin_v + points_px[:, 1] * fit.mm_per_px,
    ], axis=1)
    # fit_view 가 돌려 놓은 만큼 되돌린 뒤 부호 규칙을 뒤집는다
    flat = _rotate(turned, -getattr(fit, "angle", 0.0))
    u_part = flat[:, 0] * (fit.sign if not fit.flip_u else -fit.sign)
    v_part = flat[:, 1] * (-1 if fit.flip_v else 1)

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

    # 광선이 빗나간 점은 **비운다**(None).
    #
    # 예전에는 "가장 가까운 정점" 으로 채웠는데, 그 비교를 부호·뒤집기·
    # 회전을 적용하지 않은 원본 좌표로 하고 있었다. 변환한 좌표와 변환하지
    # 않은 좌표를 견주니 아무 정점이나 잡혔고, 제로라인이 부품 밖으로
    # 길게 뻗고 보정량 콜아웃이 허공에 떴다.
    #
    # 공간을 맞춰 고칠 수도 있지만 그러지 않는다. 광선이 빗나갔다는 것은
    # **그 자리에 부품이 없다**는 뜻이다. 없는 자리를 지어내면 화면에는
    # 그럴듯하게 나오지만 틀린 값이다. 비워서 호출한 쪽이 빼게 한다.
    return hits


def to_pixels(vertices: np.ndarray, fit: ViewFit) -> tuple:
    """부품 좌표 -> 스캔 화면 픽셀. unproject 의 반대 방향이다.

    sample_deviation 과 sample_flags 가 같은 식을 써야 해서 따로 뺐다.
    """
    u_axis, v_axis = _fit_axes(fit)
    flat = np.stack([
        vertices[:, u_axis] * (fit.sign if not fit.flip_u else -fit.sign),
        vertices[:, v_axis] * (-1 if fit.flip_v else 1),
    ], axis=1)
    turned = _rotate(flat, getattr(fit, "angle", 0.0))
    xs = np.rint((turned[:, 0] - fit.origin_u) / fit.mm_per_px).astype(int)
    ys = np.rint((turned[:, 1] - fit.origin_v) / fit.mm_per_px).astype(int)
    return xs, ys


def sample_flags(vertices: np.ndarray, fit: ViewFit, stencil: np.ndarray) -> list:
    """정점마다 스캔 화면의 도장(stencil) 값을 읽는다.

    제로라인을 **표면에 칠하려고** 만들었다. 예전에는 제로라인을 3D 공간에
    관(tube)으로 띄웠는데, 곡면 위를 지나가면 형상에서 떠서 "선을 얹은
    느낌" 이 났다. 표면 자체를 칠하면 굴곡을 그대로 따라간다 — 칠하는
    대상이 곧 그 곡면이기 때문이다.
    """
    xs, ys = to_pixels(vertices, fit)
    height, width = stencil.shape
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    out = [0] * len(vertices)
    if inside.any():
        picked = stencil[ys[inside], xs[inside]]
        for slot, value in zip(np.nonzero(inside)[0], picked):
            if value:
                out[int(slot)] = int(value)
    return out


def sample_deviation(vertices: np.ndarray, fit: ViewFit,
                    values: np.ndarray, part_mask) -> list:
    """정점마다 스캔 편차값을 찍어 준다 — 3D 표면에 히트맵을 입히려고.

    unproject 의 반대 방향이다. 부품 좌표를 화면 픽셀로 되돌린 뒤
    그 자리의 편차를 읽는다. 부품 밖이면 None 을 넣어 화면에서
    회색으로 남긴다.
    """
    mask = np.asarray(part_mask) > 0
    height, width = mask.shape
    xs, ys = to_pixels(vertices, fit)

    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    inside[inside] &= mask[ys[inside], xs[inside]]

    out: list = [None] * len(vertices)
    if inside.any():
        picked = np.asarray(values)[ys[inside], xs[inside]]
        for slot, value in zip(np.nonzero(inside)[0], picked):
            out[int(slot)] = round(float(value), 3)
    return out


__all__ = ["ViewFit", "MIN_IOU", "fit_view", "unproject",
           "sample_deviation", "sample_flags", "to_pixels"]
