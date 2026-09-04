"""보정시트가 적어 둔 단면 위치로 제로라인을 **계산**한다.

[무엇을 알아냈나]
71XX2(PILLAR-CTR INR) 보정시트의 제로 표기를 오래 못 읽었다. 다른 두
부품처럼 `"0" LINE` 이 그려져 있지 않고, 노란 `0` 콜아웃 옆에 이런
글자만 붙어 있었다 —

    H : 300      H : 250      T : 1700

이게 **부품 좌표(mm)** 이고, 어느 축인지는 CAD 좌표 범위로 소거된다.
실측 71XX1-DR000_HDCT0458.stp —

    X: 1445.0 ~ 1990.0    <- 1700 이 여기만 들어간다
    Y: -795.5 ~  795.5
    Z:    30.5 ~ 1260.2   <- 250, 300 이 여기 들어간다

    따라서  H = Z(높이),  T = X(차량 전후)

CATIA 에서 같은 부품의 한 점을 찍어 좌표계가 일치하는 것도 확인했다 —
CATIA 가 (X 1450, Y -786.2, Z 30.5), 우리 파싱의 최솟값이
(1445.0, -795.5, 30.5) 로 같은 자리다.

[그래서 제로라인은 점이 아니라 단면이다]
`H : 300` 은 "여기 한 점이 0" 이 아니라 **"Z=300 평면에서 0"** 이라는
뜻이다. 그 평면이 부품과 만나는 곡선 전체가 제로라인이다.

이게 왜 중요하냐면 — 이 방식에는 **추정이 하나도 없다.** 실루엣을
맞추지도, 색을 읽지도 않는다. 시트가 적어 준 숫자로 CAD 를 자르면
끝이다. 지금까지 만든 어떤 제로라인보다 근거가 강하다.

[한계]
어느 쪽이 LH 인지는 CAD 만으로 확정하지 못한다. 이 CAD 는 LH(71412DC000)
와 RH(71422DC000)가 한 파일에 들어 있고 Y=0 을 사이에 두고 대칭이다.
`side` 로 골라 쓰되, 기본은 양쪽 다 준다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np

# 시트 표기 -> 부품 좌표축. 위 문서의 소거법으로 정했다.
AXIS_OF = {"H": 2, "T": 0}          # H = Z(높이), T = X(전후)
AXIS_NAME = {0: "X", 1: "Y", 2: "Z"}

# "H : 300" · "T:1700" · "H : 250" 을 모두 잡는다
NOTE = re.compile(r"\b([HT])\s*[:：]\s*(-?\d+(?:\.\d+)?)", re.I)


@dataclass
class SectionZeroLine:
    """단면 하나가 만들어 낸 제로라인."""

    label: str                  # "H:300"
    axis: int                   # 0=X, 1=Y, 2=Z
    value: float                # mm
    polylines: list             # [[[x,y,z], ...], ...] 부품 좌표
    point_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def parse_notes(text: str) -> list:
    """시트에서 읽은 글자에서 단면 표기를 뽑는다.

    Returns:
        [("H", 300.0), ("T", 1700.0), ...] — 나온 순서대로, 중복 제거.
    """
    found: list = []
    for kind, value in NOTE.findall(text or ""):
        item = (kind.upper(), float(value))
        if item not in found:
            found.append(item)
    return found


def cut(mesh, axis: int, value: float, side: str = "both") -> list:
    """평면 하나로 잘라 폴리라인 목록을 얻는다.

    Args:
        mesh: trimesh.Trimesh (부품 좌표 그대로, 원점 이동 전).
        axis: 0=X, 1=Y, 2=Z.
        value: 그 축의 위치(mm).
        side: "both" · "lh"(Y<0) · "rh"(Y>0).

    Returns:
        [[[x,y,z], ...], ...] — 끊긴 곡선마다 하나씩.
    """
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    origin = [0.0, 0.0, 0.0]
    origin[axis] = float(value)

    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return []

    lines: list = []
    vertices = np.asarray(section.vertices, dtype=float)
    for entity in section.entities:
        idx = np.asarray(entity.points, dtype=int)
        if len(idx) < 2:
            continue
        pts = vertices[idx]
        if side == "lh":
            pts = pts[pts[:, 1] < 0]
        elif side == "rh":
            pts = pts[pts[:, 1] > 0]
        if len(pts) >= 2:
            lines.append([[round(float(c), 3) for c in p] for p in pts])
    return lines


def zero_lines_from_notes(mesh, notes: list, side: str = "both") -> list:
    """시트 표기 목록을 그대로 제로라인으로 바꾼다.

    Args:
        mesh: trimesh.Trimesh (부품 좌표).
        notes: parse_notes 결과, 또는 [("H", 300.0), ...].

    Returns:
        SectionZeroLine 목록. 자르지 못한 표기는 건너뛴다.
    """
    out: list = []
    for kind, value in notes:
        axis = AXIS_OF.get(kind.upper())
        if axis is None:
            continue
        polylines = cut(mesh, axis, value, side=side)
        if not polylines:
            continue
        out.append(SectionZeroLine(
            label=f"{kind.upper()}:{value:g}",
            axis=axis,
            value=float(value),
            polylines=polylines,
            point_count=sum(len(p) for p in polylines),
        ))
    return out


__all__ = ["AXIS_OF", "AXIS_NAME", "SectionZeroLine",
           "cut", "parse_notes", "zero_lines_from_notes"]
