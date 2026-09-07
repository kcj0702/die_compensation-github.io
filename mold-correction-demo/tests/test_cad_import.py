"""cad_import 검증 — 정답을 아는 STEP 을 만들어 되읽는다.

실제 부품 STEP 이 아직 없으므로, 치수와 홀 위치를 **우리가 지정해서**
STEP 을 생성하고 리더가 그걸 그대로 복원하는지 본다. 이렇게 해야
"돌아가는 것 같다"가 아니라 "정확히 맞다"를 말할 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import mesh_io, step_reader  # noqa: E402

# 검증용 판재: 200 x 100 x 10 mm, 지름 12/12/20 홀 3개
PLATE = (200.0, 100.0, 10.0)
HOLES = [
    # (x, y, 지름)  — 판재 좌하단이 원점
    (40.0, 50.0, 12.0),
    (160.0, 50.0, 12.0),
    (100.0, 25.0, 20.0),
]


def _make_plate_step(path: Path) -> None:
    """정답을 아는 판재 STEP 을 만든다 (OCCT 로 직접 생성)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_Reader, STEPControl_StepModelType, STEPControl_Writer

    width, depth, thickness = PLATE
    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), width, depth, thickness).Shape()

    for x, y, diameter in HOLES:
        axis = gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1))
        drill = BRepPrimAPI_MakeCylinder(axis, diameter / 2.0, thickness + 2.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, drill).Shape()

    Interface_Static.SetCVal_s("write.step.schema", "AP214")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
    assert writer.Write(str(path)) == 1, "STEP 쓰기 실패"


@pytest.fixture(scope="module")
def plate_step(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cad") / "plate.step"
    _make_plate_step(path)
    return path


def test_step_tessellation_matches_plate_size(plate_step: Path) -> None:
    """테셀레이션한 삼각망의 크기가 지정한 판재 치수와 같아야 한다."""
    shape = step_reader.load_step(plate_step)
    vertices, faces = step_reader.tessellate(shape)

    assert len(faces) > 0
    size = vertices.max(axis=0) - vertices.min(axis=0)
    assert np.allclose(size, PLATE, atol=0.6), f"치수 불일치: {size} != {PLATE}"


def test_step_finds_every_hole_with_right_diameter(plate_step: Path) -> None:
    """뚫은 홀 3개를 전부, 지정한 지름 그대로 찾아야 한다."""
    shape = step_reader.load_step(plate_step)
    holes = step_reader.find_cylinders(shape)

    assert len(holes) == len(HOLES), f"홀 개수 불일치: {len(holes)}"
    assert all(h.kind == "hole" for h in holes), [h.kind for h in holes]

    found = sorted(h.diameter for h in holes)
    expected = sorted(d for _x, _y, d in HOLES)
    assert np.allclose(found, expected, atol=0.05), f"{found} != {expected}"


def test_step_hole_centres_match(plate_step: Path) -> None:
    """홀 중심 XY 가 지정한 좌표와 맞아야 한다 (RPS 정렬의 근거)."""
    shape = step_reader.load_step(plate_step)
    holes = step_reader.find_cylinders(shape)

    for x, y, diameter in HOLES:
        match = [h for h in holes if abs(h.diameter - diameter) < 0.05
                 and abs(h.center[0] - x) < 0.05 and abs(h.center[1] - y) < 0.05]
        assert match, f"홀 (x={x}, y={y}, ø{diameter}) 을 못 찾음"
        # 홀 축은 Z 방향이어야 한다
        assert abs(abs(match[0].axis[2]) - 1.0) < 1e-3, match[0].axis


def test_step_finds_plate_faces(plate_step: Path) -> None:
    """판재 윗면·아랫면(각 넓이 약 200x100)을 기준면 후보로 찾아야 한다."""
    shape = step_reader.load_step(plate_step)
    planes = step_reader.find_planes(shape)

    width, depth, _t = PLATE
    hole_area = sum(np.pi * (d / 2.0) ** 2 for _x, _y, d in HOLES)
    expected_face = width * depth - hole_area

    big = [p for p in planes if abs(p.area - expected_face) < 5.0]
    assert len(big) >= 2, f"큰 평면 2개를 못 찾음: {[p.area for p in planes[:5]]}"


def test_read_step_full_shape(plate_step: Path) -> None:
    """통합 함수가 메시와 후보를 한 번에 주는지."""
    result = step_reader.read_step_full(plate_step)

    assert result["counts"]["holes"] == len(HOLES)
    assert result["mesh"].faces.shape[0] > 0
    assert len(result["planes"]) >= 2


def test_web_mesh_is_json_safe_and_recentred(plate_step: Path) -> None:
    """웹으로 내보낸 결과가 JSON 직렬화 가능하고 원점 근처로 옮겨졌는지."""
    import json

    result = step_reader.read_step_full(plate_step)
    web = mesh_io.to_web_mesh(result["mesh"], name="plate", source_format="step")

    json.dumps(web)  # 직렬화 안 되면 여기서 터진다

    assert web["summary"]["n_faces"] > 0
    assert np.allclose(web["summary"]["bounds"]["size"], PLATE, atol=0.6)

    positions = np.asarray(web["positions"], dtype=float).reshape(-1, 3)
    assert np.abs(positions.mean(axis=0)).max() < max(PLATE), "원점 근처로 안 옮겨짐"


def test_mesh_roundtrip_via_stl(plate_step: Path, tmp_path: Path) -> None:
    """STEP -> STL 로 내보냈다 다시 읽어도 치수가 유지되는지.

    스캔 데이터는 보통 STL 로 오므로 이 경로도 확인한다.
    """
    result = step_reader.read_step_full(plate_step)
    stl_path = tmp_path / "plate.stl"
    result["mesh"].export(stl_path)

    assert mesh_io.is_mesh_file(stl_path)
    mesh = mesh_io.load_mesh(stl_path)
    bounds = mesh_io.mesh_bounds(mesh)
    assert np.allclose(bounds.size, PLATE, atol=0.6), bounds.size


def test_더_큰_홀_안의_턱은_홀로_안_센다(tmp_path: Path) -> None:
    """같은 자리에 큰 원통과 작은 원통이 겹쳐 있으면 하나다.

    실측 67XX6-DR050 에서 홀이 180개 나왔는데, 중심이 3mm 안에 겹친
    쌍만 104개였다. Ø12 홀 안에 Ø6.3 원통이 비껴 앉아 있는 식이다 —
    나란히 뚫린 두 구멍일 수 없다.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    plate = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 80.0, 80.0, 6.0).Shape()
    wide = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(40, 40, -1), gp_Dir(0, 0, 1)), 6.0, 4.0).Shape()
    narrow = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(41, 40, 2.5), gp_Dir(0, 0, 1)), 3.0, 5.0).Shape()
    shape = BRepAlgoAPI_Cut(
        BRepAlgoAPI_Cut(plate, wide).Shape(), narrow).Shape()

    kinds = [c.kind for c in step_reader.find_cylinders(shape)]
    assert kinds.count("hole") == 1, f"턱까지 홀로 셌다: {kinds}"
    assert "step" in kinds, "안쪽 턱을 표시하지 않았다"


def test_떨어져_있는_같은_축_홀은_따로_센다() -> None:
    """위아래 플랜지에 각각 뚫린 볼트홀.

    축이 같다고 한 덩어리로 보면 사이의 빈 구간까지 높이에 들어가
    감김이 무너져 둘 다 굽힘 R 로 밀려난다 — 실측 71XX1 에서 홀이
    44 -> 28 개로 줄고 Ø8.4 짜리 12개가 통째로 사라졌다.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    lower = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 60.0, 60.0, 4.0).Shape()
    upper = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 30), 60.0, 60.0, 4.0).Shape()
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    plates = BRepAlgoAPI_Fuse(lower, upper).Shape()
    drill = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(30, 30, -5), gp_Dir(0, 0, 1)), 5.0, 50.0).Shape()
    shape = BRepAlgoAPI_Cut(plates, drill).Shape()

    holes = [c for c in step_reader.find_cylinders(shape) if c.kind == "hole"]
    assert len(holes) == 2, f"떨어진 두 홀을 하나로 봤다: {len(holes)}개"
    for hole in holes:
        assert hole.height < 10.0, f"빈 구간까지 높이에 넣었다: {hole.height}"


def test_색을_판_두께_건너까지_옮긴다():
    """CATIA 가 칠한 껍질은 판의 한쪽 면에만 얹혀 있다.

    실측 71XX1 은 분홍 껍질이 회색 솔리드 표면과 0.1mm 안으로 겹치는데,
    그 솔리드는 닫힌 껍데기라 반대쪽 면이 따로 있다. 그대로 두면 한쪽에서만
    분홍이고 돌리면 회색이 나온다.
    """
    import numpy as np
    from cad_import.step_reader import spread_through_thickness

    # 두께 2mm 판 흉내 — 앞면 두 장(칠함) 과 뒷면 두 장(안 칠함).
    vertices = np.array([
        [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],       # 앞면
        [0, 0, 2], [10, 0, 2], [0, 10, 2], [10, 10, 2],       # 뒷면
        [0, 0, 90], [10, 0, 90], [0, 10, 90],                 # 멀리 떨어진 것
    ], dtype=float)
    faces = np.array([
        [4, 5, 6], [5, 7, 6],      # 뒷면 — 기본색
        [8, 9, 10],                # 멀리 — 기본색, 그대로 남아야 한다
        [0, 1, 2], [1, 3, 2],      # 앞면 — 분홍
    ])
    groups = [("#C1C4C0", 0, 3, False), ("#FF99CC", 3, 2, True)]

    out_faces, out_groups = spread_through_thickness(vertices, faces, groups)
    assert len(out_faces) == len(faces)
    got = {tone: count for tone, _start, count, _direct in out_groups}
    # 뒷면 두 장이 분홍을 따라온다. 멀리 있는 한 장은 그대로다.
    assert got["#FF99CC"] == 4
    assert got["#C1C4C0"] == 1


def test_칠한_색이_없으면_그대로_둔다():
    import numpy as np
    from cad_import.step_reader import spread_through_thickness

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    groups = [("#C1C4C0", 0, 1, False)]
    out_faces, out_groups = spread_through_thickness(vertices, faces, groups)
    assert out_groups == groups
    assert np.array_equal(out_faces, faces)


def test_시트_그림의_빈_바탕을_잘라낸다():
    """3D 화면은 가로로 넓고 부품은 가운데만 차지한다.

    그대로 시트에 실으면 부품이 작게 떠 있고 둘레가 허옇게 남는다.
    """
    import numpy as np
    from zero_line_detection.sheet_excel import trim_border

    canvas = np.full((940, 2000, 3), 255, np.uint8)
    canvas[300:640, 700:1300] = (120, 130, 140)      # 부품 340 x 600
    cut = trim_border(canvas, slack=12)
    assert cut.shape[0] == 340 + 24
    assert cut.shape[1] == 600 + 24


def test_어두운_바탕도_같은_방법으로_잘린다():
    import numpy as np
    from zero_line_detection.sheet_excel import trim_border

    canvas = np.full((400, 800, 3), 22, np.uint8)
    canvas[100:300, 200:600] = (200, 200, 200)
    cut = trim_border(canvas, slack=0)
    assert cut.shape[:2] == (200, 400)


def test_잘라낼_것이_없으면_그대로_둔다():
    import numpy as np
    from zero_line_detection.sheet_excel import trim_border

    plain = np.full((50, 60, 3), 255, np.uint8)
    assert trim_border(plain).shape == plain.shape


def test_쪽마다_그림_자리가_같다():
    """시트는 40행 묶음이 되풀이되는 물건이라 쪽마다 그림이 같은 자리여야 한다.

    예전에는 비율을 지키느라 폭이나 높이 하나만 맞춰서, 납작한 부품은
    가로로 늘어지고 길쭉한 부품은 구석에 작게 박혔다.
    """
    import numpy as np
    from zero_line_detection.sheet_excel import (
        IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, fit_on_page,
    )

    납작 = np.full((180, 1900, 3), 200, np.uint8)
    길쭉 = np.full((880, 260, 3), 200, np.uint8)
    for 원본 in (납작, 길쭉):
        page = fit_on_page(원본)
        assert page.shape[:2] == (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX)


def test_부품을_비율_그대로_키운다():
    import numpy as np
    from zero_line_detection.sheet_excel import fit_on_page

    원본 = np.full((100, 200, 3), 0, np.uint8)     # 2:1
    page = fit_on_page(원본, width=800, height=400, pad=0)
    # 검은 부분의 가로세로 비가 그대로여야 한다.
    dark = np.argwhere((page < 50).all(axis=2))
    high = dark[:, 0].max() - dark[:, 0].min() + 1
    wide = dark[:, 1].max() - dark[:, 1].min() + 1
    assert abs(wide / high - 2.0) < 0.05


def test_빈_그림도_한_장으로_돌려준다():
    import numpy as np
    from zero_line_detection.sheet_excel import (
        IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, fit_on_page,
    )

    page = fit_on_page(np.zeros((0, 0, 3), np.uint8))
    assert page.shape[:2] == (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX)
