"""STEP(B-Rep)을 읽어 삼각망과 **RPS 기준 후보**를 뽑는다.

[왜 STL 이 아니라 STEP 인가]
현업 자료(2026-08-25)의 부품별 우선순위를 보면 제로라인 기준은
"조립 기준"에서 나온다 —

    선루프  : 1순위 가이드레일 장착 중심선, 2순위 섀시 조립 홀(Datum Hole)
    대시보드: 1순위 차량 센터 Y0, 3순위 크로스멤버 조립 마운트(보스/홀)

즉 **홀 중심과 기준면 좌표**를 알아야 RPS 정렬이 된다. STL 로 내보내면
원통면이 이미 삼각형으로 쪼개져 "이게 지름 12mm 홀이다"라는 정보가
사라진다. STEP 은 B-Rep 을 유지하므로 원통면을 원통면으로 읽을 수 있다.

그리고 같은 자료의 경고 — *"Best Fit 으로 정렬하면 조립 부위가 다 틀어져
금형을 망친다"*. 그래서 정렬 기준을 홀·기준면으로 잡는 게 중요하다.

[한계]
어느 홀이 실제 RPS 점인지는 도면에 지정돼 있다. 여기서는 기하학적
후보(원통면·큰 평면)를 뽑아줄 뿐이고, 최종 지정은 사람이 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

STEP_SUFFIXES = {".step", ".stp"}

# 테셀레이션 정밀도(mm). 자동차 패널 기준 0.5mm 면 화면 표시엔 충분하고
# 삼각형 수도 감당된다. 정밀 계산이 필요하면 낮춰 부른다.
DEFAULT_DEFLECTION = 0.5

# 이보다 작은 원통은 라운드/모따기일 가능성이 높아 홀 후보에서 뺀다.
MIN_HOLE_RADIUS_MM = 1.5
# 기준면 후보로 볼 최소 평면 넓이(mm^2). 작은 면턱을 걸러낸다.
MIN_PLANE_AREA_MM2 = 400.0


@dataclass
class Cylinder:
    """원통면 하나. 홀이면 조립 기준(Datum Hole) 후보가 된다."""

    kind: str              # "hole"(안쪽) | "boss"(바깥쪽) | "unknown"
    radius: float
    diameter: float
    center: list           # 원통 축 위 중심점 [x,y,z]
    axis: list             # 축 방향 단위벡터
    height: float
    area: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlaneFace:
    """평면 하나. 넓은 평면은 기준면(Datum Plane) 후보가 된다."""

    center: list
    normal: list
    area: float

    def to_dict(self) -> dict:
        return asdict(self)


def is_step_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in STEP_SUFFIXES


def load_step(path: str | Path):
    """STEP 파일을 읽어 OCCT shape 로 돌려준다."""
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    path = Path(path)
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"STEP 을 읽지 못했습니다: {path.name}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"STEP 에 형상이 없습니다: {path.name}")
    return shape


def tessellate(shape, deflection: float = DEFAULT_DEFLECTION):
    """B-Rep 을 삼각망으로 바꾼다. (vertices Nx3, faces Mx3) 을 준다."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_Orientation, TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)

    all_v: list = []
    all_f: list = []
    offset = 0
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            n_nodes = triangulation.NbNodes()
            verts = np.empty((n_nodes, 3), dtype=np.float64)
            for i in range(1, n_nodes + 1):
                p = triangulation.Node(i).Transformed(transform)
                verts[i - 1] = (p.X(), p.Y(), p.Z())

            reversed_face = face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED
            n_tri = triangulation.NbTriangles()
            tris = np.empty((n_tri, 3), dtype=np.int64)
            for i in range(1, n_tri + 1):
                a, b, c = triangulation.Triangle(i).Get()
                # 뒤집힌 면은 정점 순서를 바꿔야 법선이 바깥을 향한다
                tris[i - 1] = (a - 1, c - 1, b - 1) if reversed_face else (a - 1, b - 1, c - 1)

            all_v.append(verts)
            all_f.append(tris + offset)
            offset += n_nodes
        explorer.Next()

    if not all_v:
        raise ValueError("테셀레이션 결과가 비었습니다.")
    return np.vstack(all_v), np.vstack(all_f)


def _face_area(face) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return float(props.Mass())


def find_cylinders(shape, min_radius: float = MIN_HOLE_RADIUS_MM) -> list:
    """원통면을 찾아 홀/보스로 분류한다 — 조립 기준(RPS) 후보.

    안쪽(홀)인지 바깥쪽(보스)인지는 면의 방향으로 판정한다. 원통면의
    바깥 법선이 축을 향하면 재료가 바깥에 있다는 뜻이라 홀이다.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_Orientation, TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    found: list = []
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        try:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() != GeomAbs_SurfaceType.GeomAbs_Cylinder:
                explorer.Next()
                continue

            cylinder = surface.Cylinder()
            radius = float(cylinder.Radius())
            if radius < min_radius:
                explorer.Next()
                continue

            axis = cylinder.Axis()
            direction = axis.Direction()
            location = axis.Location()

            # v 파라미터 범위가 원통 높이
            v0, v1 = surface.FirstVParameter(), surface.LastVParameter()
            height = float(abs(v1 - v0))
            # 원통 중심을 실제 구간 중앙으로 옮긴다
            mid = (v0 + v1) / 2.0
            centre = np.array([location.X(), location.Y(), location.Z()], dtype=float)
            axis_v = np.array([direction.X(), direction.Y(), direction.Z()], dtype=float)
            centre = centre + axis_v * mid

            # 홀/보스 판정: 면이 REVERSED 면 법선이 안쪽(축 방향)을 향한다
            reversed_face = face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED
            kind = "hole" if reversed_face else "boss"

            found.append(Cylinder(
                kind=kind,
                radius=round(radius, 3),
                diameter=round(radius * 2.0, 3),
                center=[round(float(v), 3) for v in centre],
                axis=[round(float(v), 4) for v in axis_v],
                height=round(height, 3),
                area=round(_face_area(face), 2),
            ))
        except Exception:
            # 한 면이 이상해도 전체가 멈추면 안 된다
            pass
        explorer.Next()

    found.sort(key=lambda c: -c.diameter)
    return found


def find_planes(shape, min_area: float = MIN_PLANE_AREA_MM2) -> list:
    """넓은 평면을 찾는다 — 기준면(Datum Plane)·매칭면 후보."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_Orientation, TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    found: list = []
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        try:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() != GeomAbs_SurfaceType.GeomAbs_Plane:
                explorer.Next()
                continue
            area = _face_area(face)
            if area < min_area:
                explorer.Next()
                continue

            plane = surface.Plane()
            position = plane.Location()
            normal = plane.Axis().Direction()
            normal_v = np.array([normal.X(), normal.Y(), normal.Z()], dtype=float)
            if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                normal_v = -normal_v

            found.append(PlaneFace(
                center=[round(float(v), 3) for v in
                        (position.X(), position.Y(), position.Z())],
                normal=[round(float(v), 4) for v in normal_v],
                area=round(float(area), 2),
            ))
        except Exception:
            pass
        explorer.Next()

    found.sort(key=lambda p: -p.area)
    return found


def read_step_full(
    path: str | Path,
    deflection: float = DEFAULT_DEFLECTION,
    max_features: int = 200,
) -> dict:
    """STEP 하나를 읽어 메시 + RPS 후보를 한 번에 준다."""
    import trimesh

    shape = load_step(path)
    vertices, faces = tessellate(shape, deflection)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    cylinders = find_cylinders(shape)
    planes = find_planes(shape)
    holes = [c for c in cylinders if c.kind == "hole"]

    return {
        "mesh": mesh,
        "cylinders": [c.to_dict() for c in cylinders[:max_features]],
        "holes": [c.to_dict() for c in holes[:max_features]],
        "planes": [p.to_dict() for p in planes[:max_features]],
        "counts": {
            "cylinders": len(cylinders),
            "holes": len(holes),
            "planes": len(planes),
        },
    }


__all__ = [
    "STEP_SUFFIXES", "DEFAULT_DEFLECTION",
    "Cylinder", "PlaneFace",
    "is_step_file", "load_step", "tessellate",
    "find_cylinders", "find_planes", "read_step_full",
]
