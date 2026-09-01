"""보정 후 형상 — 표면을 따라 민다."""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import morph as mp  # noqa: E402


def _folded():
    """ㄱ 자로 접힌 판.

    두 판이 직선거리로는 가깝지만 판을 따라가면 멀다 — 판금이 딱
    이렇다. 직선거리로 밀면 건너편까지 같이 움직인다.
    """
    flat = trimesh.creation.box(extents=[200, 100, 4])
    stand = trimesh.creation.box(extents=[4, 100, 200])
    stand.apply_translation([98, 0, 100])
    return trimesh.util.concatenate([flat, stand])


def test_표면을_따라_잰_거리가_직선보다_멀다():
    mesh = _folded()
    mesh = mesh.subdivide().subdivide()
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces)
    spot = np.array([[98.0, 0.0, 40.0]])      # 세운 판 아래쪽

    along = mp.surface_distances(vertices, faces, spot, reach=120.0)
    assert along is not None, "scipy 가 없으면 이 시험은 못 한다"
    straight = np.linalg.norm(vertices - spot[0], axis=1)
    near_straight = int((straight < 120.0).sum())
    near_along = int((along[0] < 120.0).sum())
    assert near_along < near_straight, (
        f"표면 거리가 직선보다 좁아야 한다: 직선 {near_straight} · "
        f"표면 {near_along}")


def test_겹친_정점을_합치지_않으면_거리가_끊긴다():
    """면마다 따로 삼각형을 만든 메시(STEP 이 그렇다)."""
    mesh = _folded()
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces)
    # 정점을 일부러 쪼갠다 — 면마다 자기 정점을 갖게
    split_vertices = vertices[faces].reshape(-1, 3)
    split_faces = np.arange(len(split_vertices)).reshape(-1, 3)

    spot = np.array([[0.0, 0.0, 2.0]])
    along = mp.surface_distances(split_vertices, split_faces, spot, reach=150.0)
    assert along is not None
    reached = int((along[0] < 150.0).sum())
    assert reached > len(split_vertices) * 0.05, (
        f"정점을 합치지 않아 한 면에서 멈췄다 ({reached}개)")


def test_보정량이_포인트_자리에서_그대로_나온다():
    mesh = _folded().subdivide()
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals, float)
    spot = vertices[int(np.argmin(np.linalg.norm(
        vertices - np.array([0.0, 0.0, 2.0]), axis=1)))]

    _moved, shift, stats = mp.morph(
        vertices, faces, normals, [spot], [2.0], reach_ratio=0.12)
    top = float(np.abs(shift).max())
    assert 1.6 <= top <= 2.0, f"지정한 보정량이 안 나온다 ({top})"
    assert stats.moved > 0


def test_보정_포인트가_없으면_형상이_안_변한다():
    mesh = _folded()
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals, float)
    moved, shift, stats = mp.morph(vertices, faces, normals, [], [])
    assert np.allclose(moved, vertices)
    assert stats.moved == 0
    assert float(np.abs(shift).max()) == 0.0


def test_가중치가_포인트_자리에서_매끄럽다():
    """(1-r^2)^2 은 r=0 에서 기울기가 0 이라 꼭짓점이 안 생긴다."""
    vertices = np.stack([np.linspace(0, 10, 201),
                         np.zeros(201), np.zeros(201)], axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(vertices), 1))
    shift = mp.displacement_field(
        vertices, normals, np.array([[5.0, 0.0, 0.0]]), np.array([1.0]),
        reach=4.0)
    peak = int(np.argmax(shift))
    curve = np.diff(shift[peak - 6:peak + 7])
    assert abs(curve[5]) < 0.01 and abs(curve[6]) < 0.01, (
        "포인트 자리에서 꺾인다 — 원뿔처럼 보인다")
