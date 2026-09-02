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


def test_얹힘_비율을_잰다():
    """오버레이가 쓸 만한지는 겹침 넓이가 아니라 얹힘 비율로 가른다.

    실측에서 껍질 겹침은 후하고(64XX2 96.9%) 실루엣은 박하다(42.2%).
    둘 다 세 부품을 못 가르는데 얹힘 비율은 가른다
    (91.0 / 75.5 / 29.8%).
    """
    vertices, faces = _bar()
    mask = _mask(400, 100)
    fit = ov.fit_view(vertices, faces, mask)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rate = ov.measure_hit_rate(fit, vertices, faces, mask, mesh)
    assert rate > 0.9, f"잘 맞은 자세인데 얹힘이 낮다 ({rate})"


def test_얹힘_비율은_같은_입력에_같은_값():
    """표본을 무작위로 뽑되 씨앗을 고정한다 — 볼 때마다 숫자가 바뀌면
    사용자가 못 믿는다."""
    vertices, faces = _bar()
    mask = _mask(400, 100)
    fit = ov.fit_view(vertices, faces, mask)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    first = ov.measure_hit_rate(fit, vertices, faces, mask, mesh)
    second = ov.measure_hit_rate(fit, vertices, faces, mask, mesh)
    assert first == second


def test_한_파일에_두_짝이면_갈라_준다():
    """LH·RH 가 같이 든 CAD.

    실측 71XX1-DR000 이 그렇다 — Y 한가운데 3% 띠에 정점이 0개이고
    좌우가 91,895 대 91,926 이다. 스캔은 한 짝뿐이라 합친 실루엣과는
    맞지 않는다(얹힘 31.3%).
    """
    left = trimesh.creation.box(extents=[400, 60, 100])
    left.apply_translation([0, -200, 0])
    right = trimesh.creation.box(extents=[400, 60, 100])
    right.apply_translation([0, 200, 0])
    both = trimesh.util.concatenate([left, right])

    halves = ov.split_sides(np.asarray(both.vertices, float),
                            np.asarray(both.faces))
    assert len(halves) == 2, "두 짝을 못 갈랐다"
    for points, cells, axis, _mid, _side in halves:
        assert axis == 1, "Y 로 갈려야 한다"
        assert len(cells) > 0
        assert len(points) == len(both.vertices) // 2


def test_붙어_있는_부품은_안_가른다():
    vertices, faces = _bar()
    halves = ov.split_sides(vertices, faces)
    assert len(halves) == 1
    assert halves[0][2] == -1, "자르는 축이 없어야 한다"


def test_갈라도_면이_안_깨진다():
    """가른 뒤 면 번호가 새 정점 번호를 가리켜야 한다."""
    left = trimesh.creation.box(extents=[100, 40, 40])
    left.apply_translation([0, 0, -120])
    right = trimesh.creation.box(extents=[100, 40, 40])
    right.apply_translation([0, 0, 120])
    both = trimesh.util.concatenate([left, right])
    for points, cells, *_rest in ov.split_sides(
            np.asarray(both.vertices, float), np.asarray(both.faces)):
        assert cells.min() >= 0
        assert cells.max() < len(points)


def test_자리와_배율을_다듬어_더_잘_맞춘다():
    """CAD 에만 있는 살이 바운딩 상자를 키우면 전체가 밀린다.

    실측 71XX2 가 이것 때문에 각도를 다 훑고도 57.7% 에서 멈췄다.
    """
    import cv2

    vertices, faces = _bar()
    # 스캔에는 막대 본체만 보이고, CAD 에는 위로 뻗은 살이 더 있다
    tab = trimesh.creation.box(extents=[60, 60, 40])
    tab.apply_translation([-160, 0, 65])
    grown = trimesh.util.concatenate([
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False), tab])

    mask = np.zeros((300, 500), np.uint8)
    cv2.rectangle(mask, (60, 110), (440, 190), 255, -1)
    fit = ov.fit_view(np.asarray(grown.vertices, float),
                      np.asarray(grown.faces), mask)
    # 살이 붙어 있어도 본체가 마스크에 얹혀야 한다
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rate = ov.measure_hit_rate(fit, vertices, faces, mask, mesh)
    assert rate > 0.6, f"살이 붙으니 본체가 밀렸다 (얹힘 {rate:.2f})"


def test_손으로_옮기면_그만큼_옮겨진다():
    """작업자가 화면에서 밀면 딱 그 픽셀만큼 움직여야 한다."""
    vertices, faces = _bar()
    mask = _mask(400, 100)
    fit = ov.fit_view(vertices, faces, mask)
    before = np.stack(ov.to_pixels(vertices, fit), axis=1).astype(float)

    moved = ov.nudge_fit(fit, mask.shape, dx=12.0, dy=-7.0)
    after = np.stack(ov.to_pixels(vertices, moved), axis=1).astype(float)
    gap = after - before
    assert abs(gap[:, 0].mean() - 12.0) < 1.0, f"가로가 안 맞다 {gap[:, 0].mean()}"
    assert abs(gap[:, 1].mean() + 7.0) < 1.0, f"세로가 안 맞다 {gap[:, 1].mean()}"


def test_돌릴_때_그림_밖으로_날아가지_않는다():
    """원점을 축으로 돌리면 조금만 돌려도 부품이 사라진다."""
    vertices, faces = _bar()
    mask = _mask(400, 100)
    fit = ov.fit_view(vertices, faces, mask)
    before = np.stack(ov.to_pixels(vertices, fit), axis=1).astype(float)
    moved = ov.nudge_fit(fit, mask.shape, angle_deg=8.0)
    after = np.stack(ov.to_pixels(vertices, moved), axis=1).astype(float)
    assert np.linalg.norm(after.mean(axis=0) - before.mean(axis=0)) < 20.0, (
        "가운데가 크게 밀렸다 — 그림 한가운데를 축으로 돌리지 않았다")


def test_배율도_가운데를_축으로_한다():
    vertices, faces = _bar()
    mask = _mask(400, 100)
    fit = ov.fit_view(vertices, faces, mask)
    before = np.stack(ov.to_pixels(vertices, fit), axis=1).astype(float)
    moved = ov.nudge_fit(fit, mask.shape, scale=1.2)
    after = np.stack(ov.to_pixels(vertices, moved), axis=1).astype(float)

    span_before = before.max(axis=0) - before.min(axis=0)
    span_after = after.max(axis=0) - after.min(axis=0)
    assert np.allclose(span_after / span_before, 1.2, atol=0.05), (
        f"1.2배가 안 됐다 {span_after / span_before}")
    assert np.linalg.norm(after.mean(axis=0) - before.mean(axis=0)) < 25.0


def test_손대지_않으면_그대로다():
    vertices, faces = _bar()
    mask = _mask(400, 100)
    fit = ov.fit_view(vertices, faces, mask)
    same = ov.nudge_fit(fit, mask.shape)
    assert same.to_dict() == fit.to_dict()


def test_딱_맞는_부품은_부풀리지_않는다():
    """자리를 고를 때 재현율만 좇으면 형상을 키워 마스크를 덮는 쪽이 이긴다.

    껍질 겹침이 뚜렷하게 좋은 자세가 있으면 그것을 지켜야 한다.
    """
    vertices, faces = _bar()
    mask = _mask(400, 100)
    fit = ov.fit_view(vertices, faces, mask)
    xs, ys = ov.to_pixels(vertices, fit)
    inside = ((xs >= 0) & (xs < mask.shape[1])
              & (ys >= 0) & (ys < mask.shape[0]))
    assert inside.mean() > 0.95, (
        f"형상이 그림 밖으로 부풀었다 ({inside.mean():.2f})")


def test_자세를_여러_개_받을_수_있다():
    vertices, faces = _bar()
    got = ov.fit_view(vertices, faces, _mask(400, 100), top_k=4)
    assert isinstance(got, list) and len(got) == 4
    assert all(hasattr(f, "mm_per_px") for f in got)
    # 겹침이 좋은 순이어야 한다
    assert got[0].iou >= got[-1].iou


def test_얹힘으로_다듬으면_안_나빠진다():
    """실루엣 겹침으로 고른 자세를 얹힘 기준으로 더 다듬는다."""
    import cv2

    vertices, faces = _bar()
    tab = trimesh.creation.box(extents=[60, 60, 40])
    tab.apply_translation([-160, 0, 65])
    grown = trimesh.util.concatenate([
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False), tab])
    grown_v = np.asarray(grown.vertices, float)
    grown_f = np.asarray(grown.faces)

    mask = np.zeros((300, 500), np.uint8)
    cv2.rectangle(mask, (60, 110), (440, 190), 255, -1)
    mesh = trimesh.Trimesh(vertices=grown_v, faces=grown_f, process=False)

    fit = ov.fit_view(grown_v, grown_f, mask)
    before = ov.measure_hit_rate(fit, grown_v, grown_f, mask, mesh)
    after_fit = ov.polish_by_hit_rate(fit, grown_v, grown_f, mask, mesh)
    assert after_fit.hit_rate >= before - 1e-6, "다듬어서 되레 나빠졌다"
    assert after_fit.hit_rate == round(
        ov.measure_hit_rate(after_fit, grown_v, grown_f, mask, mesh), 4)


def test_이미_완벽하면_그대로_둔다():
    vertices, faces = _bar()
    mask = _mask(400, 100)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    fit = ov.fit_view(vertices, faces, mask)
    polished = ov.polish_by_hit_rate(fit, vertices, faces, mask, mesh)
    assert polished.hit_rate >= ov.measure_hit_rate(
        fit, vertices, faces, mask, mesh) - 1e-6
