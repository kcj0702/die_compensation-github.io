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

import math
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

    kind: str              # "hole"(안쪽) | "boss"(바깥쪽) | "fillet"(굽힘 R)
    radius: float
    diameter: float
    center: list           # 원통 축 위 중심점 [x,y,z]
    axis: list             # 축 방향 단위벡터
    height: float
    area: float
    wrap: float = 1.0      # 원통을 몇 바퀴 감았나 (1.0 이면 360도)
    faces: int = 1         # 이 원통을 이루는 면 개수

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


def _face_props(face) -> tuple:
    """면의 넓이와 무게중심을 함께 준다.

    무게중심이 필요한 이유 — 평면의 gp_Pln.Location() 은 그 **무한 평면**의
    파라미터 원점이지 면이 실제로 놓인 자리가 아니다. 실측(001 REINF SIDE
    OTR.stp)에서 부품 X 범위가 1337~1830 인데 Location() 이 x=1000 을
    돌려줬다 — 데이텀 후보 좌표로 쓰면 엉뚱한 곳을 가리킨다.
    """
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    centre = props.CentreOfMass()
    return float(props.Mass()), (centre.X(), centre.Y(), centre.Z())


def _face_area(face) -> float:
    return _face_props(face)[0]


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

    merged = _merge_cylinder_faces(found)
    merged.sort(key=lambda c: -c.diameter)
    return merged


def _merge_cylinder_faces(faces: list, closed_wrap: float = 0.8) -> list:
    """쪼개진 원통면을 하나로 합치고, 감긴 정도로 다시 분류한다.

    [왜 필요한가 — 실측 001 REINF SIDE OTR.stp]
    CAD 는 원통면을 이음매에서 반으로 자른다. 면 하나만 보면 감김이
    50% 라 굽힘 R 과 구분이 안 된다. 실제로 이 부품에서 원통면 220개가
    나왔는데 감김이 85% 를 넘는 면이 **하나도 없었다.**

    같은 축·중심·지름끼리 묶어 넓이를 더하면 그림이 완전히 달라진다 —

        묶음 110개 중 감김 80% 이상  15개  전부 Ø6.00mm, 높이 3.00mm
                    감김 80% 미만  95개  높이 중앙값 15mm, 최대 437mm

    15개가 진짜 홀이다. 그중 6개는 Z=-32.0 에 X 좌표 50mm 간격으로
    늘어서 있다 — 볼트홀 열이다. 95개는 굽힘 R 과 모서리다(높이 437mm
    짜리 원통이 홀일 리 없다).

    합치기 전에는 "홀 58개" 라고 내놨는데, 대부분 굽힘 R 이었다.
    """
    groups: dict = {}
    for face in faces:
        key = (
            tuple(round(v, 1) for v in face.center),
            round(face.diameter, 1),
            tuple(round(abs(v), 2) for v in face.axis),   # 축 부호는 무시
        )
        groups.setdefault(key, []).append(face)

    merged: list = []
    for members in groups.values():
        head = members[0]
        height = max(m.height for m in members)
        area = sum(m.area for m in members)
        full = math.pi * head.diameter * height
        wrap = (area / full) if full > 0 else 0.0

        if wrap >= closed_wrap:
            # 닫힌 원통 — 안쪽을 보면 홀, 바깥쪽을 보면 보스
            kind = "hole" if any(m.kind == "hole" for m in members) else "boss"
        else:
            kind = "fillet"     # 굽힘 R·모서리. 조립 기준이 아니다

        merged.append(Cylinder(
            kind=kind,
            radius=head.radius,
            diameter=head.diameter,
            center=head.center,
            axis=head.axis,
            height=round(height, 3),
            area=round(area, 2),
            wrap=round(min(wrap, 1.0), 3),
            faces=len(members),
        ))
    return merged


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
            area, centroid = _face_props(face)
            if area < min_area:
                explorer.Next()
                continue

            plane = surface.Plane()
            normal = plane.Axis().Direction()
            normal_v = np.array([normal.X(), normal.Y(), normal.Z()], dtype=float)
            if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                normal_v = -normal_v

            found.append(PlaneFace(
                center=[round(float(v), 3) for v in centroid],
                normal=[round(float(v), 4) for v in normal_v],
                area=round(float(area), 2),
            ))
        except Exception:
            pass
        explorer.Next()

    found.sort(key=lambda p: -p.area)
    return found


def _dedupe(features: list, *keys) -> list:
    """같은 자리에 겹쳐 나온 항목을 하나로 줄인다.

    실제 파일(001 REINF SIDE OTR.stp, CATIA V5 내보내기)에서 지름과
    중심이 완전히 같은 원통이 반복해 나왔다 — 솔리드와 셸이 함께
    들어 있어서다. 걸러내지 않으면 개수가 부풀고, max_features 컷에
    진짜 홀이 밀려난다(원통 220개 중 절반가량이 중복이었다).
    """
    seen = set()
    unique = []
    for item in features:
        signature = []
        for key in keys:
            value = getattr(item, key)
            if isinstance(value, (list, tuple)):
                signature.extend(round(float(v), 2) for v in value)
            else:
                signature.append(round(float(value), 2))
        token = tuple(signature)
        if token in seen:
            continue
        seen.add(token)
        unique.append(item)
    return unique


CACHE_DIR = Path(__file__).resolve().parent / "_parsed"
CACHE_VERSION = 2      # 판정 규칙이 바뀌면 올린다 (예전 캐시를 버리려고)


def _cache_key(path: Path, deflection: float) -> str:
    """파일 내용이 같으면 같은 키. 앞뒤 조각과 크기만 본다 —
    200MB 를 통째로 해시하면 그것만으로 몇 초가 걸린다."""
    import hashlib

    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"{CACHE_VERSION}|{size}|{deflection}".encode())
    with path.open("rb") as handle:
        digest.update(handle.read(262144))
        if size > 524288:
            handle.seek(-262144, 2)
            digest.update(handle.read(262144))
    return digest.hexdigest()[:16]


def read_step_full(
    path: str | Path,
    deflection: float = DEFAULT_DEFLECTION,
    max_features: int = 400,
    use_cache: bool = True,
) -> dict:
    """STEP 하나를 읽어 메시 + RPS 후보를 한 번에 준다.

    [디스크 캐시를 두는 이유]
    실측 파싱 시간 — 12.4MB 11초, 119MB 32초, 163MB 43초, 215MB 48초.
    파일이 큰 이유는 자유곡면이 많아서다(163MB 파일에서 B-spline 면
    5,461개, CARTESIAN_POINT 160만개로 전체 줄의 87%). 프레스 판넬은
    전체가 곡면이라 정상이고 줄일 방법이 없다.

    대신 한 번 읽은 결과를 디스크에 남긴다. 서버를 다시 켜도 다시
    읽지 않는다.
    """
    import json
    import trimesh

    path = Path(path)
    cached = None
    if use_cache:
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            cached = CACHE_DIR / f"{path.stem}_{_cache_key(path, deflection)}.npz"
            if cached.exists():
                blob = np.load(cached, allow_pickle=False)
                meta = json.loads(str(blob["meta"]))
                return {
                    "mesh": trimesh.Trimesh(vertices=blob["vertices"],
                                            faces=blob["faces"], process=False),
                    "cylinders": meta["cylinders"],
                    "holes": meta["holes"],
                    "planes": meta["planes"],
                    "counts": meta["counts"],
                }
        except Exception:
            cached = None      # 캐시가 깨져도 그냥 다시 읽으면 된다

    shape = load_step(path)
    vertices, faces = tessellate(shape, deflection)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 원통은 find_cylinders 안에서 이미 면을 합쳐 놓았다(_merge_cylinder_faces).
    cylinders = find_cylinders(shape)
    planes = _dedupe(find_planes(shape), "center", "normal")
    holes = [c for c in cylinders if c.kind == "hole"]

    result = {
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
    if cached is not None:
        try:
            meta = {k: v for k, v in result.items() if k != "mesh"}
            np.savez_compressed(
                cached, vertices=vertices, faces=faces,
                meta=np.array(json.dumps(meta, ensure_ascii=False)))
        except Exception:
            pass       # 캐시를 못 써도 결과는 그대로 돌려준다
    return result


__all__ = [
    "STEP_SUFFIXES", "DEFAULT_DEFLECTION",
    "Cylinder", "PlaneFace",
    "is_step_file", "load_step", "tessellate",
    "find_cylinders", "find_planes", "read_step_full",
]
