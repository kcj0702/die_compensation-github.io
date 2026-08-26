"""my_lab 파이프라인이 그린 영라인 — 대조용 기준선.

[출처 — 처음에 잘못 적었던 것을 바로잡는다]
feat/product-zero-line-profiles 브랜치의 product_profiles.py 는 이 좌표를
"승인된 JD64/JD67/JD71 도면에서 나왔다" 고 적어 두었다. 그 말을 믿고
이 파일도 approved_profile.py 라는 이름으로 시작했는데, 실제 출처를
따라가 보니 **승인 도면이 아니라 my_lab 스크립트의 출력**이었다.

    my_lab/zero_line_drawing/draw_jd67_zero_areas.py
        "Create six rectangular JD67 zero-area primitives around key zero points"
        -> output/JD_67XX6.../jd67_zero_areas.json
           areas[0].rectangle = [494, 199, 563, 231]
           product_profiles.py 의 첫 사각형과 정확히 일치한다.

    my_lab/zero_line_drawing/draw_jd64_base_profile.py
        -> jd64_base_profile.json
           P1.normalized = [0.2651341, 0.25478645]  <- PR 의 첫 점과 일치
           fully_fitted_to_key_zero_points: False
           "P1/P4 fitted to key zero points; P2/P3 fitted to image landmarks"

즉 사람이 승인한 도면이 아니라 **다른 알고리즘의 검출 결과**다.

[그래서 '일치'가 무슨 뜻인지도 달라진다]
전에 "우리 직선이 승인 도형과 2.04% / 1.82% 로 일치한다" 고 적었는데,
그 도형과 우리 검출은 **같은 key_zero_points.json 을 끝점으로 쓴다.**
시작점이 같으니 붙는 것이 당연하다 — 독립적인 검증이 아니었다.

[사람이 그린 정답은 따로 있다]
보정시트에서 읽은 0라인(zero_line_library.json)이 유일한 사람 기준이다.
그것으로 다시 재보니 색과의 관계가 이렇게 나온다 —

    JD_64XX2  |편차|<=0.5mm  선 위 88.3%  부품 66.2%  (+22.1%p)
              |편차|<=0.1mm  선 위 15.0%  부품 18.3%  (-3.3%p)
    JD_67XX6  |편차|<=0.5mm  선 위 48.4%  부품 29.5%  (+18.9%p)
              |편차|<=0.1mm  선 위 15.2%  부품  7.3%  (+7.9%p)

0.5mm 로 보면 확실히 편차가 작은 쪽에 있다. 그런데 그 구간이 부품의
66% 를 덮는다 — 색은 범위만 알려주고 그 안 어디에 그을지는 정하지 않는다.

[이 파일의 용도]
검출을 대체하지 않는다. my_lab 결과와 우리 결과가 얼마나 다른지 보는
대조용이다. 데모 화면에서는 이 도형을 영라인으로 표시하도록 되어 있다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabProfile:
    """my_lab 스크립트가 그린 영라인 도형(정규화 좌표)."""

    part_no: str
    open_lines: tuple = ()      # 끝이 열린 폴리라인들
    closed_loops: tuple = ()    # 닫힌 면들


JD64 = LabProfile(
    part_no="64XX2",
    open_lines=(
        ((0.26513410, 0.25478645), (0.34712644, 0.70986745),
         (0.70344828, 0.70986745), (0.70498084, 0.25478645)),
    ),
)

# jd67_zero_areas.json 의 사각형들. 1688 x 1016 캔버스 기준으로 정규화됨.
JD67 = LabProfile(
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

JD71 = LabProfile(
    part_no="71XX2",
    open_lines=(
        ((129.90909 / 1271, 412.18182 / 767), (590 / 1271, 321 / 767)),
        ((129.90909 / 1271, 412.18182 / 767), (342.75 / 1271, 508.75 / 767)),
        ((853.33333 / 1271, 370.33333 / 767), (849 / 1271, 418 / 767),
         (1031 / 1271, 449 / 767), (1112 / 1271, 449 / 767)),
    ),
)

PROFILES = (JD64, JD67, JD71)


def profile_for(part_no) -> LabProfile | None:
    """품번에 해당하는 my_lab 도형. 없으면 None."""
    if not part_no:
        return None
    compact = str(part_no).upper().replace("-", "").replace("_", "")
    return next((p for p in PROFILES if p.part_no in compact), None)


def to_pixels(profile: LabProfile, width: int, height: int) -> list:
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


def lab_shapes_for(part_no, width: int, height: int) -> list:
    """품번 + 이미지 크기로 바로 대조용 도형을 얻는다."""
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
    """검출 결과가 my_lab 도형에서 얼마나 떨어져 있는지 양방향으로 잰다.

    Args:
        predicted: [[x, y], ...] 또는 그 목록(선 여러 개).

    Returns:
        {"to_lab_pct", "to_predicted_pct", "diagonal_px"} — 이미지
        대각선 대비 중앙값 %. 도형이 없는 품번이면 None.
    """
    shapes = lab_shapes_for(part_no, width, height)
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
    to_lab = float(np.median(
        [np.hypot(*(theirs - p).T).min() for p in ours]))
    to_predicted = float(np.median(
        [np.hypot(*(ours - p).T).min() for p in theirs]))
    return {
        "to_lab_pct": round(to_lab / diagonal * 100, 2),
        "to_predicted_pct": round(to_predicted / diagonal * 100, 2),
        "diagonal_px": round(diagonal, 1),
    }


__all__ = [
    "LabProfile", "PROFILES",
    "profile_for", "to_pixels", "lab_shapes_for", "distance_report",
]
