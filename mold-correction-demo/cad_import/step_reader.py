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


def _to_hex(tone) -> str:
    """OCCT 색을 화면에 쓸 sRGB hex 로 바꾼다.

    [왜 그냥 못 쓰나 — 색이 어둡게 나왔다]
    OCCT 의 Quantity_Color 는 **선형 RGB** 를 들고 있고 STEP 파일은 sRGB 로
    적혀 있다. 읽을 때 OCCT 가 sRGB -> 선형으로 바꿔 두므로, Red() 를 그대로
    255 배 하면 파일에 적힌 색보다 어두운 값이 나온다. 실측 67XX6 에서 —

        파일에 적힌 색      그냥 쓴 값     되돌린 값
        #FF8000 (주황)      #FF3700        #FF8000
        #E800E8 (자홍)      #CE00CE        #E800E8
        #C1C4C0 (회색)      #888D86        #C1C4C0
        #969B95 (회색)      #4E544D        #969B95

    선형 -> sRGB 로 되돌려야 CATIA 에서 보던 색과 같아진다.
    """
    def back(v: float) -> float:
        v = max(0.0, min(1.0, float(v)))
        return v * 12.92 if v <= 0.0031308 else 1.055 * (v ** (1 / 2.4)) - 0.055

    return "#{:02X}{:02X}{:02X}".format(
        int(round(back(tone.Red()) * 255)),
        int(round(back(tone.Green()) * 255)),
        int(round(back(tone.Blue()) * 255)))


def load_step_coloured(path: str | Path):
    """STEP 을 한 번만 읽어 형상과 CATIA 면 색을 함께 준다.

    [왜 한 번인가 — 두 번 읽으면 두 배 걸린다]
    색은 XCAF 문서에 딸려 오므로 STEPControl_Reader 로는 못 읽는다. 그렇다고
    형상은 STEPControl 로, 색은 STEPCAFControl 로 따로 읽으면 큰 파일을 두
    번 파싱한다 — 실측 64XX1(206MB)이 243초, 71XX1(57MB)이 77초였다.
    CAF 리더가 형상도 주므로 그것 하나만 쓴다.

    Returns:
        (shape, {"dominant": "#RRGGBB" | None, "palette": {색: 넓이}})
    """
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_LabelSequence
    from OCP.Quantity import Quantity_Color
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopoDS import TopoDS, TopoDS_Compound
    from OCP.BRep import BRep_Builder

    path = Path(path)
    doc = TDocStd_Document(TCollection_ExtendedString("step"))
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.ReadFile(str(path))
    reader.Transfer(doc)

    colours = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    shapes.GetFreeShapes(roots)
    if roots.Length() == 0:
        raise ValueError(f"STEP 에 형상이 없습니다: {path.name}")

    # 최상위가 여럿이면 하나로 묶는다. 아래 단계(테셀레이션·원통 찾기)가
    # shape 하나를 받는다.
    builder = BRep_Builder()
    bundle = TopoDS_Compound()
    builder.MakeCompound(bundle)
    for i in range(1, roots.Length() + 1):
        builder.Add(bundle, shapes.GetShape_s(roots.Value(i)))

    """색을 어느 단계에서 찾나 — 면이 아니라 **껍질(shell)** 이다.

    처음에는 면에만 물었는데 71XX1 이 죄다 흰색으로 나왔다. CATIA 화면은
    회색과 분홍 두 가지인데 말이다. 단계별로 세어 보니 —

        SOLID    1개   #C1C4C0
        SHELL   11개   #D9D9D9 6 · #FF00FF 2 · #FF99CC 1 · #857489 1 · 없음 1
        FACE  3,682개  #FFFFFF 2,359 · 없음 1,323

    세 파일을 다 세어 보니 색이 어느 단계에 있는지가 파일마다 다르다 —

        64XX1  SOLID #C1C4C0 · SHELL #C1C4C0 48 · FACE #00FF00 6,126
        67XX6  SOLID #83AAD6 3 · #0080FF 2 · SHELL #0080FF 25 · FACE 없음 6,899
        71XX1  SOLID #C1C4C0 · SHELL #FF99CC 1 · #D9D9D9 6 · FACE #FFFFFF 2,359

    CATIA 화면과 맞는 것은 **껍질** 이다. 71XX1 은 회색 몸통에 아랫부분만
    분홍인데 그 분홍이 껍질에 있고, 67XX6 의 파랑도 껍질·솔리드에만 있다.
    면의 흰색(71XX1)은 물어보면 나오지만 화면에 그 색으로 보이지 않는다 —
    OCCT 가 물려받은 기본값을 돌려주는 것이라 믿을 값이 아니다.

    그래서 껍질 -> 솔리드 -> 면 순으로 본다. 어느 단계에도 없으면 None 을
    주고, 화면이 기본 회색으로 칠한다.
    """
    from OCP.TopExp import TopExp
    from OCP.TopTools import (TopTools_IndexedDataMapOfShapeListOfShape,
                              TopTools_IndexedMapOfShape)

    def tone_of(shape_item):
        tone = Quantity_Color()
        for kind in (XCAFDoc_ColorType.XCAFDoc_ColorSurf,
                     XCAFDoc_ColorType.XCAFDoc_ColorGen):
            if colours.GetColor(shape_item, kind, tone):
                return _to_hex(tone)
        return None

    # 면이 어느 껍질에 속하는지 미리 훑어 둔다.
    parents = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(bundle, TopAbs_ShapeEnum.TopAbs_FACE,
                                   TopAbs_ShapeEnum.TopAbs_SHELL, parents)
    solid_tone = None
    solids = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(bundle, TopAbs_ShapeEnum.TopAbs_SOLID, solids)
    if solids.Extent():
        solid_tone = tone_of(solids.FindKey(1))

    shell_tone: dict = {}

    def colour_of(face):
        """(색, 껍질에 **직접** 칠해진 것인가) 를 준다.

        솔리드 색은 "칠하지 않은 자리" 를 메우는 값일 뿐이다. 그런데 실측
        71XX1 을 재 보니 그 회색 껍질이 부품 전체를 덮고 있고, 색이 칠해진
        껍질(분홍 등)이 **그 위에 0.1mm 안으로 겹쳐** 있다 — 분홍 삼각형
        중심 3,861개 중 98%가 회색 표면에서 0.1mm 안이었다. 그대로 그리면
        깊이 싸움이 나고 회색이 이겨 색이 하나도 안 보인다. 어느 쪽이
        진짜 칠해진 색인지 알려 줘야 화면이 그것을 앞으로 당길 수 있다.
        """
        # 껍질 색이 있으면 그것이 CATIA 가 보여 주는 색이다.
        try:
            index = parents.FindIndex(face)
        except Exception:
            index = 0
        if index:
            shells = parents.FindFromIndex(index)
            if shells.Extent():
                shell = shells.First()
                # OCP 판에 따라 HashCode 가 없다. 파이썬 해시로 갈음한다 —
                # 같은 껍질이면 같은 값이 나오면 그만이다.
                key = hash(shell)
                if key not in shell_tone:
                    shell_tone[key] = tone_of(shell)
                if shell_tone[key]:
                    return shell_tone[key], True
        return (solid_tone or tone_of(face)), False

    spread: dict = {}
    walker = TopExp_Explorer(bundle, TopAbs_ShapeEnum.TopAbs_FACE)
    while walker.More():
        face = TopoDS.Face_s(walker.Current())
        got, _painted = colour_of(face)
        if got:
            spread[got] = spread.get(got, 0.0) + _face_area(face)
        walker.Next()

    top = max(spread, key=spread.get) if spread else None
    return bundle, {"dominant": top,
                    "palette": {k: round(v, 1) for k, v in sorted(
                        spread.items(), key=lambda kv: -kv[1])},
                    "of_face": colour_of}


def face_colours(path: str | Path) -> dict:
    """STEP 에 CATIA 가 넣어 둔 면 색만 읽는다(형상은 버린다).

    [실측 — 카티아 파일 세 개, sRGB 로 되돌린 값]
        64XX1-DR000_HDCT1860   #00FF00
        67XX6-DR050_HDCT1750   #969B95 · #E800E8 · #555A55 · #C1C4C0 · #FF8000
        71XX1-DR000_HDCT0458   #FFFFFF

    67XX6 은 팔레트에 파랑 #0080FF 이 등록돼 있지만 **어느 면에도 칠해져
    있지 않다**(면 6,899개는 아예 색이 없다). CATIA 에서 파랗게 보인다면
    그 색은 CATPart 쪽에 있고 STEP 으로 나오지 않은 것이라, 화면에서
    손으로 정하는 수밖에 없다.

    64XX1 의 #00FF00 은 COLOUR_RGB 가 아니라 DRAUGHTING_PRE_DEFINED_COLOUR
    ('green') 에서 온다. 파일을 글자로 훑어 COLOUR_RGB 만 찾으면 엉뚱하게
    #C1C4C0 이 나온다 — 그래서 OCCT 로 실제 적용된 색을 물어야 한다.

    한 부품이 여러 색을 쓰기도 한다. 화면에는 **넓이가 가장 넓은 색**을
    대표로 쓴다 — 면 개수로 세면 작은 모따기 수백 개가 큰 판 하나를 이긴다.
    """
    return load_step_coloured(path)[1]


def tessellate(shape, deflection: float = DEFAULT_DEFLECTION,
               colour_of=None):
    """B-Rep 을 삼각망으로 바꾼다. (vertices Nx3, faces Mx3) 을 준다.

    colour_of 를 주면 색깔이 같은 삼각형끼리 이어 붙이고 구간 목록을
    함께 돌려준다 — (vertices, faces, [(색, 시작삼각형, 개수), ...]).
    CATIA 는 한 부품을 여러 색으로 칠한다(실측 71XX1 은 회색 몸통에
    아랫부분만 분홍이다). 정점마다 색을 실어 보내면 30만 개가 넘어 무거우니
    구간으로 준다 — three.js 의 geometry group 과 그대로 맞는다.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_Orientation, TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)

    # 색깔별 주머니. 색을 안 쓰면 주머니 하나에 다 담긴다.
    buckets: dict = {}
    order: list = []
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        tone = colour_of(face) if colour_of is not None else None
        if tone not in buckets:
            buckets[tone] = {"v": [], "f": [], "n": 0}
            order.append(tone)
        bucket = buckets[tone]
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

            bucket["v"].append(verts)
            bucket["f"].append(tris + bucket["n"])
            bucket["n"] += n_nodes
        explorer.Next()

    if not any(b["v"] for b in buckets.values()):
        raise ValueError("테셀레이션 결과가 비었습니다.")

    all_v: list = []
    all_f: list = []
    groups: list = []
    start = 0
    shift = 0
    # 직접 칠한 껍질을 **나중에** 쌓는다. 겹쳐 있을 때 화면이 앞으로
    # 당기기 쉽게 순서를 맞춰 둔다.
    for tone in sorted(order, key=lambda t: bool(t[1]) if isinstance(t, tuple) else False):
        bucket = buckets[tone]
        if not bucket["v"]:
            continue
        faces_here = np.vstack(bucket["f"]) + shift
        all_v.append(np.vstack(bucket["v"]))
        all_f.append(faces_here)
        hex_tone, direct = tone if isinstance(tone, tuple) else (tone, False)
        groups.append((hex_tone, start, len(faces_here), direct))
        start += len(faces_here)
        shift += bucket["n"]

    vertices, faces = np.vstack(all_v), np.vstack(all_f)
    if colour_of is None:
        return vertices, faces
    return vertices, faces, groups


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
CACHE_VERSION = 7      # 판정 규칙이 바뀌면 올린다 (예전 캐시를 버리려고)
                       # 4: CATIA 면 색(colour)을 함께 담는다
                       # 5: 그 색을 선형에서 sRGB 로 되돌린다
                       # 6: 껍질 단위 색과 삼각형 구간을 담는다
                       # 7: 구간마다 직접 칠한 색인지 표시한다


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
                    "colour": meta.get("colour"),
                    "colour_groups": meta.get("colour_groups") or [],
                }
        except Exception:
            cached = None      # 캐시가 깨져도 그냥 다시 읽으면 된다

    # 형상과 색을 한 번에 읽는다. 색을 못 읽는 파일이면 형상만 다시 읽는다.
    groups: list = []
    try:
        shape, colour = load_step_coloured(path)
        vertices, faces, groups = tessellate(
            shape, deflection, colour_of=colour.pop("of_face"))
    except Exception:
        shape, colour = load_step(path), None
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
        "colour": colour,
        # 색이 같은 삼각형 구간. three.js 의 geometry group 과 그대로 맞는다.
        "colour_groups": [[t, int(a), int(n), bool(d)]
                          for t, a, n, d in groups if t],
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
    "is_step_file", "load_step", "load_step_coloured", "tessellate",
    "face_colours",
    "find_cylinders", "find_planes", "read_step_full",
]
