"""승인된 영라인 도형 — 검증 기준선.

[출처]
feat/product-zero-line-profiles 브랜치(kcj0702, 2026-08-26)의
product_profiles.py 에 들어 있던 좌표다. 승인된 JD64/JD67/JD71 도면에서
뽑아 정규화한 것이라 해상도와 무관하게 쓸 수 있다.

[왜 그 브랜치를 그대로 병합하지 않았나]
그쪽 server.py 는 이 좌표를 검출 결과 자리에 그대로 덮어쓴다 —

    if product_profile is not None:
        zero_overlay, zero_lines = product_profile      # 검출을 통째로 대체
    elif zero_patches:
        ...

우리 세 부품이 전부 해당하므로 검출이 하나도 쓰이지 않게 된다. 이 파일은
같은 좌표를 **비교 대상**으로만 내놓는다. 검출 결과(zeroLines, greenBelts,
simpleZeroLines)는 건드리지 않는다.

[왜 그래도 가져왔나 — 실측]
내가 보정시트에서 유도한 정답보다 이쪽이 더 믿을 만한 기준이었다.
JD_64XX2 에서 셋을 서로 대조하면(대각선 대비 거리 중앙값):

    우리 직선  <-> 승인 도형      2.04% / 1.82%
    우리 직선  <-> 시트 유도 정답  5.34% / 3.23%
    승인 도형  <-> 시트 유도 정답  6.12% / 5.00%

우리 검출이 승인 도형에 붙고, 어긋나는 쪽은 시트 유도 정답이다. 시트에서
패널을 찾아 좌표를 투영하는 과정에 오차가 끼기 때문이다(JD_71XX2 는 아예
무효 판정됐다). 승인 도형은 그 과정을 건너뛴다.

JD64 만 선(open_lines) 이고 JD67 은 면(closed_loops) 이다. JD71 은 선
3개인데 서로 떨어져 있다 — 시트 표기 형태가 부품마다 다르다는 근거다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ApprovedProfile:
    """승인 도면에서 뽑은 영라인 도형(정규화 좌표)."""

    part_no: str
    open_lines: tuple = ()      # 끝이 열린 폴리라인들
    closed_loops: tuple = ()    # 닫힌 면들


JD64 = ApprovedProfile(
    part_no="64XX2",
    open_lines=(
        ((0.26513410, 0.25478645), (0.34712644, 0.70986745),
         (0.70344828, 0.70986745), (0.70498084, 0.25478645)),
    ),
)

# jd67_zero_areas.json 의 사각형들. 1688 x 1016 캔버스 기준으로 정규화됨.
JD67 = ApprovedProfile(
    part_no="67XX6",
    closed_loops=tuple(
        tuple((x / 1687, y / 1015) for x, y in rect)
        for rect in (
            ((494, 199), (563, 199), (563, 231), (494, 231)),
            ((989, 199), (1058, 199), (1058, 233), (989, 233)),
            ((1286, 199), (1355, 199), (1355, 231), (1286, 231)),
            ((1403, 412), (1455, 412), (1455, 584), (1403, 584)),
            ((315, 774), (413, 774), (413, 852), (315, 852)),
            ((924, 781), (1129, 781), (1129, 814), (924, 814)),
        )
    ),
)

JD71 = ApprovedProfile(
    part_no="71XX2",
    open_lines=(
        ((129.90909 / 1271, 412.18182 / 767), (590 / 1271, 321 / 767)),
        ((129.90909 / 1271, 412.18182 / 767), (342.75 / 1271, 508.75 / 767)),
        ((853.33333 / 1271, 370.33333 / 767), (849 / 1271, 418 / 767),
         (1031 / 1271, 449 / 767), (1112 / 1271, 449 / 767)),
    ),
)

PROFILES = (JD64, JD67, JD71)


def profile_for(part_no) -> ApprovedProfile | None:
    """품번에 해당하는 승인 도형. 없으면 None."""
    if not part_no:
        return None
    compact = str(part_no).upper().replace("-", "").replace("_", "")
    return next((p for p in PROFILES if p.part_no in compact), None)


def to_pixels(profile: ApprovedProfile, width: int, height: int) -> list:
    """정규화 좌표를 이미지 픽셀 좌표로 편다.

    Returns:
        [{"shape_id": 1, "points": [[x, y], ...], "is_closed": bool}, ...]
    """
    def spread(points):
        return [[round(x * (width - 1)), round(y * (height - 1))] for x, y in points]

    shapes = []
    for normalized in profile.closed_loops:
        shapes.append({"points": spread(normalized), "is_closed": True})
    for normalized in profile.open_lines:
        shapes.append({"points": spread(normalized), "is_closed": False})
    for index, shape in enumerate(shapes, start=1):
        shape["shape_id"] = index
    return shapes


def approved_shapes_for(part_no, width: int, height: int) -> list:
    """품번 + 이미지 크기로 바로 비교용 도형을 얻는다."""
    profile = profile_for(part_no)
    return to_pixels(profile, width, height) if profile is not None else []


def densify(points, step: float = 3.0) -> np.ndarray:
    """거리 비교를 하려면 꼭짓점이 아니라 선 위를 촘촘히 봐야 한다.

    꼭짓점만으로 재면 꼭짓점이 많은 쪽이 유리해진다 — 실제로 그 함정에
    두 번 빠져서 반대 결론을 냈었다.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        count = max(int(np.hypot(*(b - a)) / step), 1)
        for k in range(1, count + 1):
            out.append(a + (b - a) * (k / count))
    return np.asarray(out)


def distance_report(predicted, part_no, width: int, height: int) -> dict | None:
    """검출 결과가 승인 도형에서 얼마나 떨어져 있는지 양방향으로 잰다.

    Args:
        predicted: [[x, y], ...] 또는 그 목록(선 여러 개).

    Returns:
        {"to_approved_pct", "to_predicted_pct", "diagonal_px"} — 이미지
        대각선 대비 중앙값 %. 승인 도형이 없는 품번이면 None.
    """
    shapes = approved_shapes_for(part_no, width, height)
    if not shapes:
        return None
    predicted = np.asarray(predicted, dtype=object)
    groups = predicted if predicted.ndim == 1 and isinstance(
        predicted[0], (list, np.ndarray)) and np.ndim(predicted[0]) == 2 else [predicted]

    ours = np.vstack([densify(np.asarray(g, float)) for g in groups])
    theirs = np.vstack([densify(s["points"]) for s in shapes])
    if not len(ours) or not len(theirs):
        return None

    diagonal = float(np.hypot(width, height))
    to_approved = float(np.median(
        [np.hypot(*(theirs - p).T).min() for p in ours]))
    to_predicted = float(np.median(
        [np.hypot(*(ours - p).T).min() for p in theirs]))
    return {
        "to_approved_pct": round(to_approved / diagonal * 100, 2),
        "to_predicted_pct": round(to_predicted / diagonal * 100, 2),
        "diagonal_px": round(diagonal, 1),
    }


__all__ = [
    "ApprovedProfile", "PROFILES",
    "profile_for", "to_pixels", "approved_shapes_for", "distance_report",
]
