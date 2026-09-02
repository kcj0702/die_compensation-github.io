"""보정량의 부호가 곧 공정이다 — + 용접, - 가공.

두 일은 작업자도 견적도 다르므로 물량이 따로 나와야 한다. 부호가
뒤바뀌거나 물량이 합쳐지면 현장에서 엉뚱한 지시가 나간다.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import morph as mp  # noqa: E402


def _plate(side: float = 100.0):
    """한 변 side 인 정사각 판 한 장(위쪽 면만)."""
    mesh = trimesh.creation.box(extents=[side, side, 2.0])
    return np.asarray(mesh.vertices, float), np.asarray(mesh.faces)


def test_부호대로_용접과_가공으로_갈린다():
    vertices, faces = _plate()
    shift = np.where(vertices[:, 0] > 0, 1.0, -1.0)
    got = {w.kind: w for w in mp.work_volumes(vertices, faces, shift)}
    assert set(got) == {"weld", "cut"}
    assert got["weld"].max_mm > 0 and got["cut"].max_mm > 0
    # 반씩 갈랐으니 넓이가 비슷해야 한다
    assert abs(got["weld"].area_mm2 - got["cut"].area_mm2) < got["weld"].area_mm2 * 0.2


def test_부피가_넓이_곱하기_두께다():
    """평평한 판을 1mm 균일하게 밀면 부피 = 겉넓이 x 1mm."""
    side = 100.0
    vertices, faces = _plate(side)
    shift = np.ones(len(vertices))
    got = mp.work_volumes(vertices, faces, shift)
    assert len(got) == 1 and got[0].kind == "weld"
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert abs(got[0].area_mm2 - mesh.area) < mesh.area * 0.01
    assert abs(got[0].volume_mm3 - mesh.area * 1.0) < mesh.area * 0.02


def test_얇은_꼬리는_물량에서_뺀다():
    """보간 꼬리가 넓은 면적에 얇게 깔려 부피를 부풀린다."""
    vertices, faces = _plate()
    shift = np.full(len(vertices), 0.01)      # 문턱(0.05mm) 아래
    assert mp.work_volumes(vertices, faces, shift) == []


def test_안_민_형상은_물량이_없다():
    vertices, faces = _plate()
    assert mp.work_volumes(vertices, faces, np.zeros(len(vertices))) == []


def test_공정별로_삼각형을_갈라_준다():
    vertices, faces = _plate()
    shift = np.where(vertices[:, 0] > 0, 1.0, -1.0)
    split = mp.split_by_process(faces, shift)
    assert len(split["weld"]) > 0 and len(split["cut"]) > 0
    assert len(split["weld"]) + len(split["cut"]) <= len(faces)
    # 갈라 낸 면이 원래 면 안에 있어야 내보낼 수 있다
    for part in split.values():
        if len(part):
            assert part.max() < len(vertices)


def test_빈_입력에도_안_터진다():
    empty = np.zeros((0, 3))
    assert mp.work_volumes(empty, np.zeros((0, 3), int), np.zeros(0)) == []
    split = mp.split_by_process(np.zeros((0, 3), int), np.zeros(0))
    assert len(split["weld"]) == 0 and len(split["cut"]) == 0
