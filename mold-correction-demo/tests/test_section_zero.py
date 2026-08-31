"""보정시트 단면 표기로 제로라인 계산하기."""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_line_detection.section_zero import (  # noqa: E402
    AXIS_OF, cut, parse_notes, zero_lines_from_notes,
)


@pytest.mark.parametrize("text,want", [
    ("H : 300", [("H", 300.0)]),
    ("T:1700", [("T", 1700.0)]),
    ("H : 300   H : 250   T : 1700", [("H", 300.0), ("H", 250.0), ("T", 1700.0)]),
    # 같은 표기가 두 번 나와도 한 번만
    ("H : 300 ... H : 300", [("H", 300.0)]),
    ("소재 SPFC980Y 1.6T + SPFC590DP 1.4T", []),   # 판 두께는 단면이 아니다
    ("", []),
])
def test_단면_표기를_읽는다(text, want):
    assert parse_notes(text) == want


def test_축은_H가_높이_T가_전후다():
    """71XX1 CAD 좌표 범위로 소거해 정했다 —
    Z 는 30~1260 이라 1700 이 안 들어가고, X 는 1445~1990 이라 들어간다."""
    assert AXIS_OF["H"] == 2      # Z
    assert AXIS_OF["T"] == 0      # X


def _slab():
    """Z 0~10, X 0~20, Y -5~5 인 상자."""
    box = trimesh.creation.box(extents=[20, 10, 10])
    box.apply_translation([10, 0, 5])
    return box


def test_평면으로_자르면_곡선이_나온다():
    lines = cut(_slab(), axis=2, value=5.0)
    assert lines
    zs = [p[2] for poly in lines for p in poly]
    assert all(abs(z - 5.0) < 1e-6 for z in zs)


def test_부품_밖에서_자르면_비어_있다():
    assert cut(_slab(), axis=2, value=999.0) == []


def test_좌우를_가른다():
    both = cut(_slab(), axis=2, value=5.0, side="both")
    lh = cut(_slab(), axis=2, value=5.0, side="lh")
    assert lh, "LH 쪽에도 형상이 있어야 한다"
    assert all(p[1] < 0 for poly in lh for p in poly)
    assert sum(len(p) for p in lh) < sum(len(p) for p in both)


def test_표기를_그대로_제로라인으로():
    lines = zero_lines_from_notes(_slab(), [("H", 5.0), ("T", 10.0)])
    assert [l.label for l in lines] == ["H:5", "T:10"]
    assert [l.axis for l in lines] == [2, 0]
    assert all(l.point_count > 0 for l in lines)


def test_잘리지_않는_표기는_건너뛴다():
    lines = zero_lines_from_notes(_slab(), [("H", 5.0), ("H", 999.0)])
    assert [l.label for l in lines] == ["H:5"]
