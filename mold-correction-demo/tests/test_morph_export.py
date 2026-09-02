"""보정 후 형상을 STL 로 내보낼 때 **실제 값**이 나가는지.

화면은 변형을 수십 배 부풀려 보여준다 — 보정량이 부품 크기의 0.13~0.16%
라 1픽셀 안팎이어서 그러지 않으면 눈으로 못 본다. 그 부풀린 값이 STL 에
섞여 나가면 금형을 그만큼 잘못 판다. 화면과 파일이 갈라지는 자리라
시험으로 못 박아 둔다.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import morph as mp  # noqa: E402


def _plate():
    mesh = trimesh.creation.box(extents=[200, 120, 4]).subdivide().subdivide()
    return (np.asarray(mesh.vertices, float), np.asarray(mesh.faces),
            np.asarray(mesh.vertex_normals, float))


def test_내보낼_값에는_과장이_없다():
    vertices, faces, normals = _plate()
    spot = vertices[int(np.argmax(vertices[:, 2]))]
    moved, shift, stats = mp.morph(
        vertices, faces, normals, [spot], [2.0], reach_ratio=0.15)

    walked = np.linalg.norm(moved - vertices, axis=1)
    assert walked.max() <= 2.0 + 1e-6, (
        f"실제 보정량보다 크게 움직였다 ({walked.max():.3f}mm)")
    assert abs(stats.max_shift - float(np.abs(shift).max())) < 1e-6


def test_STL_로_내보내도_값이_유지된다(tmp_path: Path):
    """STL 은 정점 순서를 보존하지 않는 **삼각형 뭉치**다.

    그래서 정점 번호로 견줄 수 없다(그렇게 짰다가 234mm 어긋난다는
    엉뚱한 결과가 나왔다). 경계 상자와 삼각형 수로 견준다.
    """
    vertices, faces, normals = _plate()
    spot = vertices[int(np.argmax(vertices[:, 2]))]
    moved, _shift, _stats = mp.morph(
        vertices, faces, normals, [spot], [1.5], reach_ratio=0.15)

    out = tmp_path / "after.stl"
    made = trimesh.Trimesh(vertices=moved, faces=faces, process=False)
    made.export(out)
    back = trimesh.load(out, process=False)

    assert len(back.faces) == len(faces), "삼각형이 늘거나 줄었다"
    # STL 은 float32 라 소수점 아래가 조금 깎인다
    assert np.allclose(back.bounds, made.bounds, atol=0.01), (
        "경계가 달라졌다: 내보낸 것 %s · 읽은 것 %s"
        % (made.bounds, back.bounds))

    # 부풀린 값이 섞여 나가지 않았는지 — 원본 경계에서 1.5mm 넘게
    # 벗어난 자리가 없어야 한다
    grew = np.abs(made.bounds - trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False).bounds).max()
    assert grew <= 1.5 + 0.01, f"실제 보정량보다 크게 커졌다 ({grew:.3f}mm)"


def test_보정량이_0_이면_형상이_그대로다():
    vertices, faces, normals = _plate()
    moved, shift, stats = mp.morph(
        vertices, faces, normals, [vertices[0]], [0.0], reach_ratio=0.15)
    assert np.allclose(moved, vertices)
    assert stats.max_shift == 0.0


def test_부호가_지켜진다():
    """살을 붙이는 쪽(+)과 깎는 쪽(-)이 뒤바뀌면 안 된다."""
    vertices, faces, normals = _plate()
    top = int(np.argmax(vertices[:, 2]))
    _moved, plus, _s = mp.morph(
        vertices, faces, normals, [vertices[top]], [1.0], reach_ratio=0.15)
    _moved2, minus, _s2 = mp.morph(
        vertices, faces, normals, [vertices[top]], [-1.0], reach_ratio=0.15)
    assert plus[top] > 0 and minus[top] < 0
    assert abs(plus[top] + minus[top]) < 1e-6
