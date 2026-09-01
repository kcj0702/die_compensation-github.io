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
from dataclasses import dataclass, asdict, field
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

    # "hole"(관통) | "boss"(바깥쪽) | "fillet"(굽힘 R)
    # | "step"(더 큰 홀 안의 턱 — 따로 센 홀이 아니다)
    kind: str
    radius: float
    diameter: float
    center: list           # 원통 축 위 중심점 [x,y,z]
    axis: list             # 축 방향 단위벡터
    height: float
    area: float
    wrap: float = 1.0      # 원통을 몇 바퀴 감았나 (1.0 이면 360도)
    faces: int = 1         # 이 원통을 이루는 면 개수
    # 아래 둘은 쪼개진 면을 합칠 때만 쓴다. 축선 위의 기준점과, 그
    # 기준점에서 축 방향으로 이 면이 차지하는 구간이다. 화면에 쓸 값이
    # 아니라 to_dict 에서 뺀다.
    origin: list = field(default_factory=list)
    span: list = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("origin", None)
        data.pop("span", None)
        return data


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


def _cylinder_side(surface, face, origin: np.ndarray,
                   direction: np.ndarray) -> str:
    """이 원통면이 홀(안쪽)인지 보스(바깥쪽)인지 본다.

    면 위 한 점에서 **실제 바깥 법선**을 구해 축 쪽을 향하는지 본다.
    축을 향하면 재료가 원통 바깥에 있다는 뜻이니 홀이다.

    예전에는 `face.Orientation() == REVERSED` 하나로 갈랐다. 그건
    원통면의 파라미터 방향이 늘 바깥을 향한다는 가정인데, 내보낸
    시스템에 따라 뒤집혀 있을 수 있다. 접선 두 개를 외적해 법선을
    직접 구하면 그 가정이 필요 없다.
    """
    from OCP.TopAbs import TopAbs_Orientation
    from OCP.gp import gp_Pnt, gp_Vec

    u = (surface.FirstUParameter() + surface.LastUParameter()) / 2.0
    v = (surface.FirstVParameter() + surface.LastVParameter()) / 2.0
    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    surface.D1(u, v, point, du, dv)

    normal = np.cross([du.X(), du.Y(), du.Z()], [dv.X(), dv.Y(), dv.Z()])
    size = float(np.linalg.norm(normal))
    if size < 1e-12:
        return "boss"
    normal = normal / size
    if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
        normal = -normal

    along = np.array([point.X(), point.Y(), point.Z()], dtype=float) - origin
    radial = along - direction * float(along @ direction)
    reach = float(np.linalg.norm(radial))
    if reach < 1e-9:
        return "boss"
    return "hole" if float(normal @ (radial / reach)) < 0 else "boss"


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

            base = np.array([location.X(), location.Y(), location.Z()],
                            dtype=float)
            kind = _cylinder_side(surface, face, base, axis_v)

            found.append(Cylinder(
                kind=kind,
                radius=round(radius, 3),
                diameter=round(radius * 2.0, 3),
                center=[round(float(v), 3) for v in centre],
                axis=[round(float(v), 4) for v in axis_v],
                height=round(height, 3),
                area=round(_face_area(face), 2),
                origin=[float(v) for v in base],
                span=[float(v0), float(v1)],
            ))
        except Exception:
            # 한 면이 이상해도 전체가 멈추면 안 된다
            pass
        explorer.Next()

    merged = _mark_inner_steps(_merge_cylinder_faces(found))
    merged.sort(key=lambda c: -c.diameter)
    return merged


# 안쪽 턱으로 볼 조건 — 축이 나란한 정도와, 축 방향으로 맞닿았다고 볼 여유.
STEP_AXIS_DOT = 0.99
STEP_TOUCH_MM = 0.2


def _mark_inner_steps(cylinders: list) -> list:
    """더 큰 홀 안에 들어앉은 원통을 홀에서 뺀다.

    [무엇이 문제였나 — 실측 67XX6-DR050]
    이 부품에서 홀이 **180개** 나왔다. 그런데 중심이 3mm 안에 겹친 홀
    쌍만 104 개였고, 높이도 0.4 · 0.8 · 0.9 · 1.1 · 1.3 · … · 15.7mm 로
    제각각이었다. 판 하나를 뚫은 구멍이라면 높이는 판 두께 하나여야 한다.

    겹친 것들을 들여다보면 Ø12 홀 안에 Ø6.3 · Ø6.4 원통이 1~3mm 비껴
    앉아 있다. 두 개가 나란히 뚫린 구멍일 수 없다 — 작은 원이 큰 원
    안에 통째로 들어가기 때문이다. 같은 구멍의 **안쪽 턱**이다.

    그래서 지운다 —
      · 축이 나란하고
      · 작은 원이 큰 원 안에 통째로 들어가고 (비낀 거리 + 작은 반지름
        <= 큰 반지름)
      · 축 방향으로 서로 맞닿아 있다 (떨어져 있으면 다른 판의 구멍이다)

    세 번째 조건이 중요하다. 이게 없으면 판금에서 위아래 플랜지에 각각
    뚫린 멀쩡한 볼트홀이 "큰 홀 안에 있다" 는 이유로 지워진다 — 실측
    64XX1 에서 Ø4.5 · Ø5.2 · Ø6.0 · Ø21 이 통째로 사라졌다(43 -> 23개).

    [실측 결과]
        64XX1-DR000   43 -> 43 개   (건드리지 않는다)
        71XX1-DR000   44 -> 44 개   (건드리지 않는다)
        67XX6-DR050  180 -> 152 개  (Ø6.4 11개 · Ø6.3 17개)
    """
    holes = [c for c in cylinders if c.kind == "hole"]
    inner: set = set()
    for outer in holes:
        way = np.asarray(outer.axis, dtype=float)
        for other in holes:
            if other is outer or id(other) in inner or id(outer) in inner:
                continue
            if other.radius >= outer.radius:
                continue
            if abs(float(way @ np.asarray(other.axis, dtype=float))) < STEP_AXIS_DOT:
                continue
            gap = (np.asarray(other.center, dtype=float)
                   - np.asarray(outer.center, dtype=float))
            deep = abs(float(gap @ way))
            side = float(np.linalg.norm(gap - way * float(gap @ way)))
            if side + other.radius > outer.radius:
                continue
            if deep > (outer.height + other.height) / 2.0 + STEP_TOUCH_MM:
                continue      # 축을 따라 떨어져 있다 — 다른 판의 구멍이다
            inner.add(id(other))

    for cylinder in cylinders:
        if id(cylinder) in inner:
            cylinder.kind = "step"
    return cylinders


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
    groups: list = []
    for face in faces:
        for members in groups:
            if _same_cylinder(members[0], face):
                members.append(face)
                break
        else:
            groups.append([face])

    merged: list = []
    for members in groups:
        head = members[0]
        direction = np.asarray(head.axis, dtype=float)
        base = np.asarray(head.origin, dtype=float)

        # 축선 위에서 면마다 차지하는 구간을 구한다.
        #
        # 예전에는 `max(height)` 를 통짜 원통 높이로 썼다. 축을 따라
        # 위아래로 쪼개진 면들은 그러면 높이가 절반만 잡혀 감김이
        # 어긋난다. 구간을 직접 재서 합친다.
        spread: list = []
        for member in members:
            start = np.asarray(member.origin, dtype=float)
            way = np.asarray(member.axis, dtype=float)
            edges = [float((start + way * float(edge) - base) @ direction)
                     for edge in member.span]
            spread.append((min(edges), max(edges), member))
        spread.sort(key=lambda item: item[0])

        # 축은 같은데 **떨어져 있는** 것은 서로 다른 홀이다.
        #
        # 판금은 플랜지가 겹치는 자리가 많아 같은 축에 홀이 위아래로
        # 두 개씩 뚫린다. 그걸 한 덩어리로 보면 사이의 빈 구간까지
        # 높이에 들어가 감김이 무너지고, 둘 다 굽힘 R 로 밀려난다 —
        # 실측 71XX1 에서 이걸 안 갈랐더니 홀이 44 -> 28 개로 줄고
        # Ø8.4 짜리 12 개가 통째로 사라졌다.
        runs: list = []
        for low, high, member in spread:
            if runs and low <= runs[-1][1] + SAME_RUN_GAP_MM:
                runs[-1][1] = max(runs[-1][1], high)
                runs[-1][2].append(member)
            else:
                runs.append([low, high, [member]])

        for low, high, group in runs:
            extent = high - low
            area = sum(m.area for m in group)
            full = math.pi * head.diameter * extent
            wrap = (area / full) if full > 0 else 0.0

            if wrap >= closed_wrap:
                # 닫힌 원통 — 안쪽을 보면 홀, 바깥쪽을 보면 보스
                kind = "hole" if any(m.kind == "hole" for m in group) else "boss"
            else:
                kind = "fillet"     # 굽힘 R·모서리. 조립 기준이 아니다

            centre = base + direction * ((low + high) / 2.0)
            merged.append(Cylinder(
                kind=kind,
                radius=head.radius,
                diameter=head.diameter,
                center=[round(float(v), 3) for v in centre],
                axis=head.axis,
                height=round(float(extent), 3),
                area=round(area, 2),
                wrap=round(min(wrap, 1.0), 3),
                faces=len(group),
            ))
    return merged


# 같은 원통으로 볼 기준. 자리 맞춤 오차와 내보내기 정밀도를 감안한 값이다.
SAME_AXIS_DOT = 0.9995        # 축이 나란한 정도 (약 1.8도)
SAME_AXIS_GAP_MM = 0.05       # 두 축선 사이 거리
SAME_DIAMETER_MM = 0.02
# 축을 따라 이만큼 넘게 떨어져 있으면 서로 다른 원통으로 본다.
SAME_RUN_GAP_MM = 0.05


def _same_cylinder(a: "Cylinder", b: "Cylinder") -> bool:
    """두 원통면이 같은 원통에서 쪼개져 나온 것인가.

    예전에는 중심·지름·축을 소수점 한 자리로 **반올림해 열쇠**를 만들어
    묶었다. 두 가지가 깨진다 —

      (1) 중심을 v 구간 한가운데로 잡아 놨는데, 축을 따라 위아래로
          쪼개진 면들은 그 한가운데가 서로 다르다. 같은 홀인데 열쇠가
          갈린다.
      (2) 반올림은 경계에서 갈린다. 12.349 와 12.351 은 0.002mm 차이인데
          12.3 과 12.4 로 나뉜다.

    실측 67XX6-DR050 에서 반올림 열쇠로 묶으면 감김이 모자란 조각이
    남는다 — 거리로 묶으니 굽힘 R 묶음이 160 -> 146 개로 줄었다(홀
    개수는 그대로다. 이건 조각을 없앤 것이지 홀을 더 찾은 게 아니다).

    그래서 반올림 대신 **거리로** 잰다. 축이 나란하고, 두 축선이 겹치고,
    지름이 같으면 같은 원통이다. 축을 따라 떨어져 있어도 상관없다 —
    구간은 합칠 때 다시 구한다.
    """
    if abs(a.diameter - b.diameter) > SAME_DIAMETER_MM:
        return False

    ua = np.asarray(a.axis, dtype=float)
    ub = np.asarray(b.axis, dtype=float)
    if abs(float(ua @ ub)) < SAME_AXIS_DOT:
        return False

    # 두 축선 사이 거리 — 나란하므로 한 점을 축 방향으로 지운 나머지다
    gap = np.asarray(b.origin, dtype=float) - np.asarray(a.origin, dtype=float)
    return float(np.linalg.norm(gap - ua * float(gap @ ua))) <= SAME_AXIS_GAP_MM


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
CACHE_VERSION = 3      # 판정 규칙이 바뀌면 올린다 (예전 캐시를 버리려고)


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
