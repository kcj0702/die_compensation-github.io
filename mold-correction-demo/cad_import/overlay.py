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
# 겹침 기준. 이건 "자리를 잡았나" 의 대리 지표일 뿐이다.
MIN_IOU = 0.75
# 실제 기준: 스캔 부품 위의 점을 쏘았을 때 형상에 맞는 비율.
#
# 겹침만 보다가 두 번 속았다.
#   - 껍질 겹침은 구멍과 오목한 곳을 메우고 재서 후하다. 실측 64XX2 는
#     껍질 96.9% 인데 실루엣은 42.2% 다. 화면에 97% 라고 띄우면 사용자는
#     "거의 완벽" 으로 읽는데 안쪽은 절반도 안 맞는다.
#   - 그렇다고 실루엣으로 판정하면 멀쩡한 부품을 버린다. 판금을 비스듬히
#     보면 투영 넓이가 원래 다르다 — 42% 가 정상이다.
#
# 오버레이가 하는 일은 "스캔의 한 점을 형상 어디에 얹느냐" 다. 그러니
# 그것을 직접 재는 게 맞다. 실측 —
#
#     부품     껍질    실루엣   광선명중
#     64XX2   96.9%   42.2%    91.0%
#     67XX6   94.7%   39.8%    75.5%
#     71XX2   28.1%   12.7%    29.8%
#
# 명중률만 세 부품을 깨끗이 가른다. 0.60 이면 앞 둘은 통과, 71XX2 는
# 걸린다.
MIN_HIT_RATE = 0.60
# 명중률을 잴 때 쏴 보는 점 수. 많이 쏠 이유가 없다.
HIT_SAMPLE = 300


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
    hit_rate: float = 0.0    # 스캔 위의 점이 형상에 얹히는 비율 — 실제 기준
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


# 자세를 찾을 때 쓸 삼각형 수 상한. 격자가 FIT_GRID 라 이보다 촘촘해도
# 실루엣이 달라지지 않는다.
FIT_MAX_FACES = 12_000

# 주축으로 잡은 각도 둘레를 이만큼 더 훑는다(도).
#
# [왜 필요한가 — 실측 71XX2]
# 주축 각도만 믿고 한 번에 돌렸더니 CAD 실루엣이 스캔 위에 **15~20도
# 비스듬히** 얹혔다. 모양은 같은데 각도가 어긋나 겹침이 0.28 에서
# 막혔고, 그래서 "이 부품은 스캔으로 못 맞춘다" 로 보였다. 주축은
# 끝이 벌어진 부품에서 쉽게 흔들린다 — 둘레를 훑어야 한다.
FIT_ANGLE_SPAN = 30.0
FIT_ANGLE_STEP = 2.5
# 각도를 훑어 볼 후보 수(1차로 추린 것 중 위에서부터).
FIT_REFINE_TOP = 4

# 위치·배율 다듬기.
#
# [왜 필요한가]
# 자리와 배율을 **바운딩 상자끼리** 맞춰서 정한다. CAD 에만 있는 살
# (스캔 각도에서는 안 보이는 플랜지)이 상자를 키우면 전체가 작아지고
# 밀린다. 실측 71XX2 는 각도를 다 훑어도 겹침 0.59 에서 멈췄는데,
# 그림을 보면 CAD 가 스캔 위에 대체로 얹혀 있으면서 왼쪽 위로 삐져
# 나가 있다 — 상자에 끌려간 것이다.
# 성기게 훑고 이긴 자리 둘레를 다시 촘촘히 훑는다. 한 번에 촘촘히
# 훑으면(5x5x5 = 125칸) 자세 찾기가 60초가 된다.
FIT_PLACE_SCALES = (0.88, 1.0, 1.12)
FIT_PLACE_SHIFTS = (-0.06, 0.0, 0.06)
FIT_PLACE_FINE = 0.5          # 2차에서 간격을 절반으로 좁힌다


def fit_view(vertices: np.ndarray, faces: np.ndarray, part_mask,
             top_k: int = 1):
    """CAD 실루엣을 스캔 마스크에 맞춰 어느 방향에서 본 그림인지 찾는다.

    Args:
        top_k: 1 이면 제일 나은 것 하나(ViewFit). 2 이상이면 그만큼을
            목록으로 준다.

    [왜 여러 개를 주나]
    여기서 쓰는 점수는 실루엣 겹침인데, 오버레이가 쓸 만한지를 가르는
    건 **점이 형상에 얹히는 비율**이다. 둘이 어긋날까 봐 후보를 남겨
    두고 부르는 쪽이 광선을 쏴서 다시 고를 수 있게 했다.

    다만 재보니 세 부품 모두 **겹침 1등이 곧 얹힘 1등**이었다. 그래서
    지금은 다시 고를 필요가 없지만, 다른 부품에서 어긋날 수 있으니 길은
    열어 둔다.

    [자세를 어떻게 좁히나 — 세 단계]
      1. 축 3 x 부호 2 x 뒤집기 4 = 24 가지를 주축 각도로 한 번씩
      2. 위 몇 개만 각도 둘레를 훑는다 (FIT_ANGLE_SPAN)
      3. 이긴 것의 자리와 배율을 다듬는다 (FIT_PLACE_*)

    단계마다 얹힘이 얼마나 오르는지 (실측) —

        부품     1단계만   +각도 훑기   +자리 다듬기   +LH/RH 가르기
        64XX2     99.7%      99.7%        99.7%          99.7%
        67XX6     73.3%      73.3%        91.7%          91.7%
        71XX2     31.3%      55.0%        61.0%          67.7%

    71XX2 가 오래 30% 대에 묶여 있었는데, 그게 "이 부품은 스캔으로 못
    맞춘다" 로 보였던 이유다. 실제로는 각도 · 자리 · 좌우 세 가지가
    한꺼번에 어긋나 있었다.
    """
    # 자세를 찾는 데는 **실루엣**만 있으면 된다. 격자가 FIT_GRID 라
    # 삼각형을 다 그릴 필요가 없다.
    #
    # 실측: 64XX1 은 삼각형이 369,082 개라 후보마다 두 번씩 그리면
    # 자세 찾기에만 150초가 걸렸다(71XX2 41초 · 67XX6 60초). 이게
    # "3D 에 보정시트 얹는 게 느리다" 의 대부분이다.
    faces = np.asarray(faces)
    if len(faces) > FIT_MAX_FACES:
        step = int(np.ceil(len(faces) / FIT_MAX_FACES))
        faces = faces[::step]

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

    mask_width = max(mx1 - mx0, my1 - my0) + 1

    def evaluate(flat, turn, axis, sign, flip_u, flip_v):
        """이 자세로 그려 보고 점수와 ViewFit 을 준다."""
        projected = _rotate(flat, turn)
        canvas, lo, _scale = _rasterize(projected, faces, FIT_GRID)
        shape_solid = _hull(canvas) > 0
        union = int((shape_solid | mask_solid).sum())
        if union == 0:
            return None
        # 자리는 껍질로, 방향은 원본 겹침으로 가른다.
        hull_iou = float((shape_solid & mask_solid).sum()) / union
        shape_bool = canvas > 0
        detail_union = int((shape_bool | mask_bool).sum())
        detail_iou = (float((shape_bool & mask_bool).sum())
                      / detail_union) if detail_union else 0.0
        # 화면 픽셀 -> 부품 좌표. 마스크 바운딩 박스를 실루엣 바운딩
        # 박스에 맞춘 것이므로 배율은 두 폭의 비다.
        span = float(np.maximum(
            projected.max(axis=0) - projected.min(axis=0), 1e-9).max())
        return ((round(hull_iou, 3), detail_iou), ViewFit(
            axis=axis, sign=sign, flip_u=flip_u, flip_v=flip_v,
            swap=False, angle=float(turn),
            mm_per_px=span / mask_width,
            origin_u=float(lo[0] - mx0 * span / mask_width),
            origin_v=float(lo[1] - my0 * span / mask_width),
            iou=round(hull_iou, 4), detail_iou=round(detail_iou, 4),
        ))

    def place(projected, base_lo, base_scale, factor, dx, dy):
        """배율과 자리를 조금 바꿔 그려 보고 (점수, 배율, 원점)을 준다."""
        scale = base_scale * factor
        shift = np.array([dx, dy], dtype=float) * FIT_GRID
        pixels = (projected - base_lo) * scale + shift
        canvas = np.zeros((FIT_GRID, FIT_GRID), np.uint8)
        cv2.fillPoly(canvas, pixels[faces].astype(np.int32), 255)

        shape_bool = canvas > 0
        union = int((shape_bool | mask_bool).sum())
        detail = (float((shape_bool & mask_bool).sum()) / union) if union else 0.0
        # 자리를 고르는 점수는 **껍질 먼저, 세부 나중**이다. 1차와 같은
        # 기준이어야 한다.
        #
        # 세부 겹침만으로 골라 봤더니 얇은 형상에서 무너졌다 — 합성
        # 시험(막대 하나)에서 세부가 0.058 밖에 안 나오는데 그 0.058 을
        # 좇느라 껍질이 0.85 -> 0.79 로 떨어졌다. 세부는 리브와 구멍이
        # 많은 실제 부품에서나 의미가 있다.
        solid = _hull(canvas) > 0
        hull_union = int((solid | mask_solid).sum())
        hull = ((float((solid & mask_solid).sum()) / hull_union)
                if hull_union else 0.0)

        # 격자 좌표 g = scale*(mm - base_lo) + shift 를 ViewFit 의
        # mm_per_px·원점으로 되돌린다.
        step = (FIT_GRID - 1)
        mm_per_px = step / (scale * mask_width)
        offset = shift - scale * base_lo
        origin_u = -mm_per_px * mx0 - offset[0] / scale
        origin_v = -mm_per_px * my0 - offset[1] / scale
        return detail, hull, mm_per_px, origin_u, origin_v

    # ── 1차: 24가지 조합을 주축 각도 하나로 훑는다 ───────────
    combos: list = []
    for axis in (0, 1, 2):
        u_axis, v_axis = _plane_axes(axis)
        for sign in (1, -1):
            for flip_u in (False, True):
                for flip_v in (False, True):
                    u = vertices[:, u_axis] * (sign if not flip_u else -sign)
                    v = vertices[:, v_axis] * (-1 if flip_v else 1)
                    flat = np.stack([u, v], axis=1)
                    # 각도는 **채워진 실루엣**에서 잰다. 정점 구름으로
                    # 재면 작은 피처가 몰린 쪽으로 주축이 끌려가 스캔
                    # 마스크(픽셀이 고르게 찬다)와 기준이 달라진다.
                    rough, _lo, _s = _rasterize(flat, faces, FIT_GRID)
                    rys, rxs = np.nonzero(_hull(rough) > 0)
                    base = 0.0
                    if len(rxs):
                        base = mask_angle - _principal_angle(
                            rxs.astype(float), rys.astype(float))
                    got = evaluate(flat, base, axis, sign, flip_u, flip_v)
                    if got is not None:
                        combos.append((got[0], flat, base, axis, sign,
                                       flip_u, flip_v, got[1]))

    if not combos:
        raise ValueError("CAD 실루엣을 스캔에 맞추지 못했습니다.")
    combos.sort(key=lambda item: item[0], reverse=True)

    # ── 2차: 위 몇 개만 각도 둘레를 훑는다 ───────────────────
    found: list = [(item[0], item[7]) for item in combos]
    steps = int(FIT_ANGLE_SPAN / FIT_ANGLE_STEP)
    for _key, flat, base, axis, sign, flip_u, flip_v, _fit in             combos[:FIT_REFINE_TOP]:
        for k in range(-steps, steps + 1):
            if k == 0:
                continue
            got = evaluate(flat, base + np.deg2rad(FIT_ANGLE_STEP * k),
                           axis, sign, flip_u, flip_v)
            if got is not None:
                found.append(got)

    found.sort(key=lambda item: item[0], reverse=True)

    # ── 3차: 위에서 몇 개의 자리와 배율을 다듬는다 ───────────
    # 상자 맞춤이 만든 치우침을 여기서 걷어낸다. 점수는 **세부 겹침**으로
    # 본다 — 껍질 겹침은 CAD 에만 있는 살에 끌려간다.
    refined: list = []
    for key, fit in found[:max(1, min(top_k, FIT_REFINE_TOP))]:
        u_axis, v_axis = _plane_axes(fit.axis)
        u = vertices[:, u_axis] * (fit.sign if not fit.flip_u else -fit.sign)
        v = vertices[:, v_axis] * (-1 if fit.flip_v else 1)
        projected = _rotate(np.stack([u, v], axis=1), fit.angle)
        base_lo = projected.min(axis=0)
        base_scale = (FIT_GRID - 1) / np.maximum(
            projected.max(axis=0) - base_lo, 1e-9).max()
        def scan_around(scales, shifts_x, shifts_y):
            here = []
            for factor in scales:
                for dx in shifts_x:
                    for dy in shifts_y:
                        detail, hull, mm_per_px, origin_u, origin_v = place(
                            projected, base_lo, base_scale, factor, dx, dy)
                        moved = ViewFit(
                            axis=fit.axis, sign=fit.sign, flip_u=fit.flip_u,
                            flip_v=fit.flip_v, swap=False, angle=fit.angle,
                            mm_per_px=float(mm_per_px),
                            origin_u=float(origin_u), origin_v=float(origin_v),
                            iou=round(hull, 4), detail_iou=round(detail, 4))
                        # 1차와 같은 기준 — 껍질 먼저, 세부 나중.
                        # 세부만으로 골라 봤더니 얇은 형상에서 무너졌다.
                        here.append(((round(hull, 3), detail), moved,
                                     factor, dx, dy))
            here.sort(key=lambda item: item[0], reverse=True)
            return here

        coarse_best = scan_around(
            FIT_PLACE_SCALES, FIT_PLACE_SHIFTS, FIT_PLACE_SHIFTS)

        # 2차 — 이긴 자리 둘레를 절반 간격으로 다시 본다
        _key, _moved, factor, dx, dy = coarse_best[0]
        gap_s = (FIT_PLACE_SCALES[-1] - FIT_PLACE_SCALES[0]) / 2 * FIT_PLACE_FINE
        gap_d = (FIT_PLACE_SHIFTS[-1] - FIT_PLACE_SHIFTS[0]) / 2 * FIT_PLACE_FINE
        fine_best = scan_around(
            (factor - gap_s, factor, factor + gap_s),
            (dx - gap_d, dx, dx + gap_d),
            (dy - gap_d, dy, dy + gap_d))

        refined.extend((key, moved)
                       for key, moved, *_rest in fine_best + coarse_best)

    refined.sort(key=lambda item: item[0], reverse=True)
    picked = [fit for _key, fit in (refined or found)[:max(1, top_k)]]
    for fit in picked:
        fit.reliable = fit.iou >= MIN_IOU
    return picked[0] if top_k <= 1 else picked


def measure_hit_rate(fit: ViewFit, vertices: np.ndarray, faces: np.ndarray,
                     part_mask, mesh=None, seed: int = 0) -> float:
    """스캔 부품 위의 점을 쏘았을 때 형상에 맞는 비율.

    오버레이가 실제로 하는 일을 그대로 재는 것이라, 겹침 넓이보다
    정직한 기준이다(MIN_HIT_RATE 주석 참고).
    """
    mask = np.asarray(part_mask) > 0
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    rng = np.random.default_rng(seed)     # 같은 입력이면 같은 값이 나오게
    pick = rng.choice(len(xs), size=min(HIT_SAMPLE, len(xs)), replace=False)
    points = [[int(xs[i]), int(ys[i])] for i in pick]
    placed = unproject(points, vertices, faces, fit, mesh)
    if not placed:
        return 0.0
    return float(sum(1 for spot in placed if spot is not None) / len(placed))


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



# 한 파일에 두 짝이 들어 있는지 볼 때, 가운데 이만큼의 띠가 비어 있으면
# 갈라진 것으로 본다(전체 폭 대비).
SPLIT_BAND = 0.03
# 두 짝의 크기가 이보다 많이 차이 나면 대칭 한 쌍이 아니다.
SPLIT_BALANCE = 0.35


def split_sides(vertices, faces):
    """LH·RH 가 한 파일에 들어 있으면 갈라 준다.

    [왜 필요한가 — 실측 71XX1-DR000_HDCT0458]
    스캔은 LH **한 짝**인데 CAD 에는 LH 와 RH 가 같이 들어 있다. 두 짝을
    합친 실루엣을 한 짝짜리 스캔에 맞출 수는 없다 — 얹힘이 30% 에서
    막혔고, 그래서 제로라인·보정량을 아예 안 그렸다.

    갈라진 것을 어떻게 아나. 이 파일은 Y 한가운데 3% 띠에 정점이
    **0 개** 이고 좌우가 91,895 대 91,926 이다. 붙어 있는 부품이라면
    가운데에 살이 있어야 한다. 실측 64XX1 은 그 띠에 6,240 개(2.1%),
    67XX6 은 18,934 개(6.3%) 라 갈라지지 않는다 — 이 셋을 한 규칙으로
    가릴 수 있다.

    Returns:
        [(정점, 면, 자르는 축, 가운데, 어느 쪽), ...].
        갈라지지 않으면 원본 하나에 축이 -1 이다. 축·가운데·쪽을 같이
        주는 이유는, 화면용으로 따로 솎아 낸 메시에도 **같은 기준**을
        적용해야 하기 때문이다 — 안 그러면 고른 쪽에 맞춘 자세로 반대쪽
        살까지 색을 칠한다.
    """
    points = np.asarray(vertices, dtype=float)
    cells = np.asarray(faces)
    if not len(points) or not len(cells):
        return [(points, cells, -1, 0.0, 0)]

    low, high = points.min(axis=0), points.max(axis=0)
    for axis in (0, 1, 2):
        span = float(high[axis] - low[axis])
        if span <= 0:
            continue
        middle = (low[axis] + high[axis]) / 2.0
        if np.any(np.abs(points[:, axis] - middle) < span * SPLIT_BAND):
            continue      # 가운데에 살이 있다 — 한 덩어리다
        left = points[:, axis] < middle
        share = float(left.sum()) / len(points)
        if not (SPLIT_BALANCE <= share <= 1.0 - SPLIT_BALANCE):
            continue      # 한쪽이 부스러기다 — 두 짝이 아니다

        halves = []
        for side, keep in ((-1, left), (1, ~left)):
            # 면은 세 꼭짓점이 모두 그 쪽일 때만 가져간다
            wanted = keep[cells].all(axis=1)
            if not wanted.any():
                continue
            index = np.full(len(points), -1, dtype=np.int64)
            index[keep] = np.arange(int(keep.sum()))
            halves.append((points[keep], index[cells[wanted]],
                           axis, middle, side))
        if len(halves) == 2:
            return halves
    return [(points, cells, -1, 0.0, 0)]

__all__ = ["ViewFit", "MIN_IOU", "MIN_HIT_RATE",
           "fit_view", "measure_hit_rate", "split_sides", "unproject",
           "sample_deviation", "sample_flags", "to_pixels"]
