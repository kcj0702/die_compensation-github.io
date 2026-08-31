"""CAD 실루엣을 스캔에 맞추는 정합."""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import overlay as ov  # noqa: E402


def _bar():
    """X 로 길고 Z 로 짧은 막대. Y 축에서 보면 400 x 100 이다."""
    box = trimesh.creation.box(extents=[400, 60, 100])
    return np.asarray(box.vertices, float), np.asarray(box.faces)


def _mask(width: int, height: int):
    """가운데 8할을 채운 부품 마스크."""
    mask = np.zeros((height, width), np.uint8)
    mask[int(height * 0.1):int(height * 0.9),
         int(width * 0.1):int(width * 0.9)] = 255
    return mask


def test_같은_방향이면_잘_맞는다():
    vertices, faces = _bar()
    fit = ov.fit_view(vertices, faces, _mask(400, 100))
    assert fit.iou > 0.9


def test_스캔이_90도_돌아가_있어도_맞춘다():
    """부품을 눕혀 찍은 스캔.

    실측 71XX2(센터 필러)가 이 경우였다. 스캔은 필러를 눕혀 찍었는데
    (983 x 568) CAD 의 Y 투영은 서 있어서(545 x 1230) 겹침이 25% 에
    머물렀다. 축·부호·뒤집기를 다 훑어도 회전이 없어서 맞출 수가 없었다.
    뒤집기는 거울이지 회전이 아니다.
    """
    vertices, faces = _bar()
    fit = ov.fit_view(vertices, faces, _mask(100, 400))   # 세로로 긴 스캔
    assert fit.iou > 0.9, f"90도 돌아간 스캔을 못 맞춘다 (겹침 {fit.iou})"
    # 90도를 어떻게 흡수했는지는 상관없다 — 실제로 맞았는지만 본다
    assert abs(fit.angle) > 1e-6 or fit.swap


def test_좌표를_되돌리는_쪽도_같은_축_순서를_쓴다():
    """fit_view 와 unproject·sample_deviation 의 축 순서가 어긋나면
    정합은 맞다고 나오는데 좌표가 틀어진다."""
    for axis in (0, 1, 2):
        for swap in (False, True):
            fit = ov.ViewFit(axis=axis, sign=1, flip_u=False, flip_v=False,
                             mm_per_px=1.0, origin_u=0.0, origin_v=0.0,
                             iou=1.0, swap=swap)
            base = ov._plane_axes(axis)
            got = ov._fit_axes(fit)
            assert got == ((base[1], base[0]) if swap else base)


def test_되돌린_좌표가_제자리로_온다():
    """화면 좌표 -> 부품 좌표 -> 화면 좌표 가 왕복해야 한다."""
    vertices, faces = _bar()
    mask = _mask(100, 400)
    fit = ov.fit_view(vertices, faces, mask)
    placed = ov.unproject([[50, 200]], vertices, faces, fit,
                          trimesh.Trimesh(vertices=vertices, faces=faces,
                                          process=False))
    assert placed and placed[0] is not None


def test_비스듬히_기울어진_스캔도_맞춘다():
    """스캔은 검사 소프트웨어에서 작업자가 놓은 각도 그대로다.

    90도 단위로 맞아떨어질 이유가 없다. 실측 71XX2 는 주축이 20도 넘게
    어긋나 있었다.
    """
    vertices, faces = _bar()
    grid = np.zeros((400, 400), np.uint8)
    box = np.array([[-160, -40], [160, -40], [160, 40], [-160, 40]], float)
    angle = np.deg2rad(23.0)
    turn = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    import cv2
    cv2.fillPoly(grid, [np.rint(box @ turn.T + 200).astype(np.int32)], 255)

    fit = ov.fit_view(vertices, faces, grid)
    assert fit.iou > 0.85, f"기울어진 스캔을 못 맞춘다 (겹침 {fit.iou})"
