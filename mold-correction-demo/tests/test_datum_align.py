"""스캔을 CAD 데이텀에 맞춰 세우기 (RPS 정렬)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import.datum_align import (  # noqa: E402
    align, datum_candidates, rigid_from_points,
)


def _pose(rx, ry, rz, tx, ty, tz):
    """알려진 6자유도 자세를 4x4 로."""
    a, b, c = np.deg2rad([rx, ry, rz])
    rot_x = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rot_y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rot_z = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = rot_z @ rot_y @ rot_x
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def _datums():
    """넓게 벌어진 데이텀 세 점."""
    return np.array([[0.0, 0.0, 0.0], [300.0, 10.0, 5.0], [40.0, 220.0, -8.0]])


def test_알려진_자세를_되찾는다():
    """정답을 아는 시험 — 스캔을 일부러 틀어 놓고 되돌린다."""
    cad = _datums()
    pose = _pose(12, -7, 33, 150.0, -80.0, 42.0)
    scan = cad @ pose[:3, :3].T + pose[:3, 3]

    got = align(scan, cad, scan)
    assert got.datum_rmse < 1e-6, f"되돌리지 못했다 (RMSE {got.datum_rmse})"
    back = got.apply(scan)
    assert np.allclose(back, cad, atol=1e-6)


def test_측정_오차가_있으면_잔차가_남는다():
    """데이텀에 잡음이 섞이면 완벽히 겹칠 수 없다.

    "소수점 4자리까지 일치" 를 약속할 수 없는 이유다. 재서 돌려줄 뿐이다.
    """
    rng = np.random.default_rng(7)
    cad = _datums()
    pose = _pose(5, 5, 5, 10.0, 20.0, 30.0)
    scan = cad @ pose[:3, :3].T + pose[:3, 3] + rng.normal(0, 0.05, cad.shape)

    got = align(scan, cad, scan)
    assert got.datum_rmse > 0, "잡음이 있는데 잔차가 0 일 수 없다"
    assert got.datum_rmse < 0.2, f"잡음보다 훨씬 크다 ({got.datum_rmse})"


def test_거울상으로_맞추지_않는다():
    """부품을 뒤집어 맞추면 안 된다 — det = +1 을 지킨다."""
    cad = _datums()
    flipped = cad * np.array([1.0, 1.0, -1.0])
    matrix = np.asarray(rigid_from_points(flipped, cad), dtype=float)
    assert np.linalg.det(matrix[:3, :3]) > 0


def test_점이_모자라면_거부한다():
    with pytest.raises(ValueError):
        rigid_from_points(np.zeros((2, 3)), np.zeros((2, 3)))


def test_일직선_데이텀은_경고한다():
    """한 줄로 놓인 세 점은 그 축 둘레로 자세가 안 정해진다."""
    line = np.array([[0.0, 0, 0], [100.0, 0, 0], [200.0, 0, 0]])
    got = align(line, line, line)
    assert any("일직선" in note for note in got.notes)


def test_표면까지_맞춘다():
    """ICP 다듬기 — 메시를 주면 표면 오차도 잰다."""
    box = trimesh.creation.box(extents=[200, 120, 60])
    pose = _pose(8, -4, 15, 60.0, -30.0, 25.0)
    moved = box.copy()
    moved.apply_transform(np.linalg.inv(pose))

    cad_datums = np.asarray(box.vertices[[0, 3, 5]], dtype=float)
    scan_datums = cad_datums @ np.linalg.inv(pose)[:3, :3].T + np.linalg.inv(pose)[:3, 3]

    got = align(moved.vertices, cad_datums, scan_datums,
                scan_mesh=moved, cad_mesh=box)
    assert got.datum_rmse < 1e-5
    back = got.apply(moved.vertices)
    assert np.abs(back - np.asarray(box.vertices)).max() < 1e-3


def test_데이텀_후보는_서로_멀리_고른다():
    holes = [{"center": c} for c in
             [[0, 0, 0], [1, 1, 0], [300, 0, 0], [2, 2, 0], [0, 250, 0]]]
    picked = datum_candidates(holes, want=3)
    assert len(picked) == 3
    chosen = np.array([holes[i]["center"] for i in picked], dtype=float)
    gaps = np.linalg.norm(chosen[:, None] - chosen[None, :], axis=2)
    assert gaps.max() > 200, "가까이 붙은 홀만 골랐다"
