"""보정 포인트 **주변의 형상**을 읽는다.

[무엇을 하려는 것인가]
지금 보정량은 좌표로 찾는다 — "이 자리의 편차는 -1.2mm". 그런데 좌표는
부품이 바뀌면 아무 뜻이 없다. 새 부품의 (473, -739, 340) 이 어떤 자리인지
알 길이 없기 때문이다.

대신 **그 자리가 어떻게 생겼는지**로 말하면 부품을 건너 옮겨진다 —
"⌀30 홀에서 12mm 떨어진, 가장자리에서 40mm 안쪽의 평평한 면". 이런 기술이
쌓이면 새 CAD 를 열었을 때 비슷하게 생긴 자리를 찾아 보정량을 미리
말할 수 있다.

[먼저 재 보고 알게 된 것 — 64XX2 검사 포인트 79개]
형상 특징이 좌표보다 낫다는 것은 사실이었다. 하나를 빼고 나머지로 맞히는
식으로 재니 —

    쓰는 것              설명력 R2    평균오차
    좌표만 (X,Y,Z)          0.01      0.64mm
    모양 특징만              0.08      0.59mm
    둘 다                  0.04      0.61mm
    아무것도 안 씀(평균)       0.00      0.63mm

개별 관계는 이렇다(실측 79개) —

    홀까지 -0.423 · 홀지름 +0.372 · 굽힘R까지 +0.283
    가장자리까지 -0.116 · 국부굴곡 -0.015 · 법선z +0.007

`홀까지 거리` 가 어떤 좌표보다 셌다. 홀 근처는 소재가 물려 있어 스프링백이
덜하다는 뜻이라 물리적으로도 말이 된다.

**다만 예측까지는 못 간다.** 그냥 평균을 찍는 것이 0.63mm 인데 최선이
0.59mm 다. 부품이 하나뿐이라 그렇다 — 스프링백을 실제로 지배하는 소재·
판두께·금형·드로우 깊이는 한 부품 안에서 전부 상수라 학습할 것이 없고,
포인트 79개도 같은 판 위에 붙어 있어 독립 표본이 아니다.

그래서 이 파일은 **예측을 하지 않는다.** 특징을 뽑아 표로 낼 뿐이다.
부품이 여럿 모이면 그 표가 그대로 학습 입력이 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# 국부 굴곡을 잴 반경(mm). 판넬의 굽힘 R 이 보통 5~15mm 라, 그보다 넓게
# 잡아야 "이 자리가 평평한가 휘었나" 가 드러난다. 너무 넓히면 부품 전체가
# 휜 정도에 묻힌다.
CURVE_REACH_MM = 20.0
# 굴곡을 재려면 법선이 이만큼은 있어야 한다. 반경 안에 모자라면 가까운
# 정점을 이 개수만큼 끌어와 잰다.
CURVE_MIN_POINTS = 12


@dataclass
class PointFeature:
    """보정 포인트 하나의 주변 형상."""

    point_id: str
    position: tuple            # 부품 좌표(mm)
    deviation: float | None    # 검사 원본이 준 편차(mm). 없으면 None

    hole_mm: float             # 가장 가까운 조립 홀까지
    hole_dia: float            # 그 홀의 지름
    rim_mm: float              # 부품 가장자리(자유 모서리)까지
    bend_mm: float             # 가장 가까운 굽힘 R 까지
    curve: float               # 국부 굴곡 0(평평) ~ 1(심하게 휨)
    normal: tuple              # 그 자리 면의 법선

    def as_row(self) -> dict:
        row = asdict(self)
        row["position"] = [round(float(v), 2) for v in self.position]
        row["normal"] = [round(float(v), 4) for v in self.normal]
        return row


def _rim_vertices(mesh) -> np.ndarray:
    """부품의 자유 모서리에 붙은 정점들.

    한 번만 나오는 모서리가 경계다 — 안쪽 모서리는 이웃한 면 둘이 나눠
    갖는다. 판금은 껍데기라 바깥 테두리와 홀 둘레가 여기 걸린다.
    """
    edges = np.asarray(mesh.edges_sorted).reshape(-1, 2)
    pairs, count = np.unique(edges, axis=0, return_counts=True)
    lone = pairs[count == 1]
    if not len(lone):
        return np.asarray(mesh.vertices, dtype=float)
    return np.asarray(mesh.vertices, dtype=float)[np.unique(lone)]


def features_for(points, mesh, cylinders=None) -> list:
    """포인트마다 주변 형상을 읽는다.

    Args:
        points: [{"id", "position", "deviation"}, ...] 또는 그 셋을 가진 객체.
        mesh: trimesh.Trimesh — CAD 삼각망(부품 좌표).
        cylinders: step_reader.read_step_full 이 준 원통 목록. 홀과 굽힘 R 을
            여기서 가른다. 없으면 그 두 값은 무한대로 둔다.

    Returns:
        PointFeature 목록.
    """
    from scipy.spatial import cKDTree

    spots = []
    for item in points:
        if isinstance(item, dict):
            spots.append((str(item.get("id") or item.get("name") or ""),
                          np.asarray(item["position"], dtype=float),
                          item.get("deviation")))
        else:
            spots.append((str(getattr(item, "name", "")),
                          np.asarray(item.position, dtype=float),
                          getattr(item, "deviation", None)))
    if not spots:
        return []
    P = np.array([s[1] for s in spots])

    holes = [c for c in (cylinders or []) if c.get("kind") == "hole"]
    bends = [c for c in (cylinders or []) if c.get("kind") == "fillet"]
    hole_at = np.array([h["center"] for h in holes], dtype=float) if holes else None
    hole_d = np.array([h["diameter"] for h in holes], dtype=float) if holes else None
    bend_at = np.array([b["center"] for b in bends], dtype=float) if bends else None

    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    near_v = cKDTree(vertices)
    near_rim = cKDTree(_rim_vertices(mesh))

    out: list = []
    for (point_id, spot, deviation), row in zip(spots, P):
        if hole_at is not None:
            gap = np.linalg.norm(hole_at - spot, axis=1)
            pick = int(np.argmin(gap))
            hole_mm, hole_dia = float(gap[pick]), float(hole_d[pick])
        else:
            hole_mm, hole_dia = float("inf"), 0.0
        if bend_at is not None:
            bend_mm = float(np.min(np.linalg.norm(bend_at - spot, axis=1)))
        else:
            bend_mm = float("inf")

        # 국부 굴곡 — 둘레 정점들의 법선이 흩어진 정도. 평평하면 법선이
        # 모두 같은 쪽을 보므로 평균 길이가 1 에 가깝고, 휘었으면 짧아진다.
        near = near_v.query_ball_point(spot, CURVE_REACH_MM)
        if len(near) < CURVE_MIN_POINTS:
            # 반경 안에 정점이 모자라면 가까운 것 몇 개로 넓혀 잰다.
            #
            # 그냥 "모른다" 로 두면 **평평한 데만 골라 빠진다.** STEP 은
            # 평평한 면을 삼각형 몇 개로만 쪼개기 때문이다 — 실측 64XX1 은
            # 정점 간격이 90% 는 1.3mm 인데 성긴 데는 32mm 까지 벌어진다.
            # 그 자리를 빼고 세면 "굴곡과 편차의 관계" 가 휜 자리 쪽으로
            # 기울어진다(실측에서 79개 중 22개가 이렇게 빠졌다).
            near = near_v.query(spot, k=min(CURVE_MIN_POINTS, len(vertices)))[1]
            near = np.atleast_1d(near)
        curve = float(1.0 - np.linalg.norm(normals[near].mean(axis=0)))

        out.append(PointFeature(
            point_id=point_id,
            position=tuple(float(v) for v in spot),
            deviation=(None if deviation is None else float(deviation)),
            hole_mm=round(hole_mm, 2),
            hole_dia=round(hole_dia, 2),
            rim_mm=round(float(near_rim.query(spot)[0]), 2),
            bend_mm=round(bend_mm, 2),
            curve=round(curve, 4),
            normal=tuple(float(v) for v in normals[int(near_v.query(spot)[1])]),
        ))
    return out


def to_csv(rows: list, target) -> "Path":  # noqa: F821
    """표로 내보낸다. 부품이 여럿 모이면 그대로 학습 입력이 된다."""
    import csv
    from pathlib import Path

    target = Path(target)
    head = ["부품", "포인트", "X", "Y", "Z", "편차mm",
            "홀까지mm", "홀지름mm", "가장자리까지mm", "굽힘R까지mm",
            "국부굴곡", "법선X", "법선Y", "법선Z"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(head)
        for part, feature in rows:
            writer.writerow([
                part, feature.point_id,
                *[round(v, 2) for v in feature.position],
                "" if feature.deviation is None else round(feature.deviation, 3),
                feature.hole_mm, feature.hole_dia, feature.rim_mm,
                ("" if feature.bend_mm == float("inf") else feature.bend_mm),
                feature.curve,
                *[round(v, 4) for v in feature.normal],
            ])
    return target


def relate(rows: list) -> dict:
    """특징과 편차가 얼마나 함께 움직이는지 잰다.

    예측이 아니라 **관계**만 본다. 부품 하나로는 예측이 안 된다는 것을
    위 주석에서 재 뒀다 — 그 사실을 숨기지 않으려고 이름을 이렇게 뒀다.
    """
    have = [f for f in rows if f.deviation is not None]
    if len(have) < 3:
        return {}
    dev = np.array([f.deviation for f in have])
    columns = {
        "홀까지": np.array([f.hole_mm for f in have]),
        "홀지름": np.array([f.hole_dia for f in have]),
        "가장자리까지": np.array([f.rim_mm for f in have]),
        "굽힘R까지": np.array([f.bend_mm for f in have]),
        "국부굴곡": np.array([f.curve for f in have]),
        "법선z": np.array([f.normal[2] for f in have]),
    }
    out: dict = {}
    for name, values in columns.items():
        good = np.isfinite(values)
        if good.sum() < 3 or values[good].std() < 1e-9:
            continue
        out[name] = round(float(np.corrcoef(values[good], dev[good])[0, 1]), 3)
    return out


__all__ = ["CURVE_MIN_POINTS", "CURVE_REACH_MM", "PointFeature", "features_for", "relate", "to_csv"]
