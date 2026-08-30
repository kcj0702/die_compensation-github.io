"""보정시트에 적을 포인트 선별 규칙."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_line_detection.key_points import select  # noqa: E402


def _point(pid, x, y, value):
    return {"id": pid, "xPx": x, "yPx": y, "value": value}


def test_큰_값을_먼저_고른다():
    points = [_point("A", 100, 100, 0.3), _point("B", 900, 900, 2.8)]
    chosen, _ = select(points, 1000, 1000, target=2)
    assert [c.point_id for c in chosen] == ["B", "A"]


def test_가까이_몰린_포인트는_하나만_남는다():
    # 같은 자리에 겹친 포인트 다섯 — 실측 스캔에 실제로 있다(최소거리 0px)
    points = [_point(f"P{i}", 500, 500, 2.0) for i in range(5)]
    points.append(_point("먼곳", 100, 100, 1.5))
    chosen, _ = select(points, 1000, 1000, target=6)
    assert len(chosen) == 2
    assert {c.point_id for c in chosen} == {"P0", "먼곳"}


def test_컬러바_범위_밖_판독값은_버린다():
    # 67XX6 은 +-3.0mm 다. +9.0 은 VLM 오독이며 이걸 안 버리면 1등이 된다.
    points = [_point("오독", 100, 100, 9.0), _point("진짜", 900, 900, 2.5)]
    chosen, rejected = select(points, 1000, 1000, target=2, part_no="67XX6")
    assert [c.point_id for c in chosen] == ["진짜"]
    assert [r["id"] for r in rejected] == ["오독"]


def test_품번을_모르면_거르지_않는다():
    points = [_point("큰값", 100, 100, 9.0)]
    chosen, rejected = select(points, 1000, 1000, target=2)
    assert [c.point_id for c in chosen] == ["큰값"]
    assert rejected == []


def test_잡음은_제외한다():
    points = [_point("잡음", 100, 100, 0.05), _point("진짜", 900, 900, 1.0)]
    chosen, _ = select(points, 1000, 1000, target=5)
    assert [c.point_id for c in chosen] == ["진짜"]


def test_목표보다_적게_나올_수_있다():
    """간격을 지키며 더 넣을 자리가 없으면 억지로 채우지 않는다."""
    points = [_point(f"P{i}", 500 + i, 500, 2.0) for i in range(20)]
    chosen, _ = select(points, 1000, 1000, target=15)
    assert len(chosen) == 1


def test_빈_입력():
    assert select([], 1000, 1000) == ([], [])


@pytest.mark.parametrize("target", [1, 5, 10])
def test_목표_개수를_넘지_않는다(target):
    points = [_point(f"P{i}", (i * 137) % 1000, (i * 311) % 1000, 1.0 + i * 0.1)
              for i in range(40)]
    chosen, _ = select(points, 1000, 1000, target=target)
    assert len(chosen) <= target
