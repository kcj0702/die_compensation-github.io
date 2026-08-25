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
