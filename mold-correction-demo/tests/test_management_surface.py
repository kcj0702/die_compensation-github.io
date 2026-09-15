"""CATIA 관리면 트리 해석과 스캔 좌표 투영."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import.management_surface import (  # noqa: E402
    ManagementGroup,
    apply_management_value,
    calculate_management_result,
    discover_management_groups,
    parse_management_value,
    project_boundary,
)
from cad_import.overlay import ViewFit  # noqa: E402


class Collection:
    def __init__(self, *items):
        self.items = list(items)
        self.Count = len(items)

    def Item(self, index):
        return self.items[index - 1]


class Node:
    def __init__(self, name, shapes=(), groups=()):
        self.Name = name
        self.HybridShapes = Collection(*shapes)
        self.HybridBodies = Collection(*groups)


def test_상위_이름에서만_관리치를_읽는다():
    assert parse_management_value('+0.5관리 기준면') == 0.5
    assert parse_management_value('-0.7 관리 기준면') == -0.7
    assert parse_management_value('"0"기준 관리면') == 0.0
    assert parse_management_value('A) -0.5 방향 적용면') is None


def test_적용면_이름의_잘못된_부호를_목표값으로_쓰지_않는다():
    group = Node('+0.5관리 기준면', shapes=[
        Node('SPEC SURFACE-01'),
        Node('01) -0.5 방향 적용면'),
        Node('BOUNDARY'),
    ])
    part = Node('Part', groups=[group])

    groups = discover_management_groups(part)

    assert len(groups) == 1
    assert groups[0].target_mm == 0.5
    assert groups[0].surfaces[0].surface_id == '01'
    assert groups[0].surfaces[0].applied_name == '01) -0.5 방향 적용면'


def test_A부터_D까지_ID로_짝짓는다():
    group = Node('-0.5관리 기준면', shapes=[
        Node('SPEC SURFACE-A'), Node('A) -0.5 방향 적용면'),
        Node('SPEC SURFACE-B'), Node('B) -0.5 방향 적용면'),
    ], groups=[Node('BOUNDARY')])
    groups = discover_management_groups(Node('Part', groups=[group]))
    assert [surface.surface_id for surface in groups[0].surfaces] == ['A', 'B']
    assert groups[0].boundary_names == ['BOUNDARY']


def test_CAD_뷰어_fit으로_boundary를_PNG에_투영한다():
    # Z축에서 본 XY 평면. 뷰어 좌표는 원본 CATIA 좌표에서 offset을 뺀 값이다.
    fit = ViewFit(axis=2, sign=1, flip_u=False, flip_v=False,
                  mm_per_px=1.0, origin_u=0.0, origin_v=0.0, iou=1.0)
    boundary = [[10, 20, 5], [30, 20, 5], [30, 40, 5], [10, 40, 5]]
    mask, polygon, ratio = project_boundary(
        boundary, fit, (100, 100), cad_offset=(10, 20, 0))
    assert ratio == 1.0
    assert polygon[0] == [0.0, 0.0]
    assert mask[10, 10] == 255


def test_영역_편차에서_관리치를_뺀다():
    group = ManagementGroup('+0.5관리 기준면', 0.5, [], ['BOUNDARY'])
    values = np.full((20, 20), np.nan)
    values[5:15, 5:15] = 0.2
    values[6, 6] = 9.0  # OCR/라벨 이상치가 있어도 중앙값은 흔들리지 않는다.
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 5:15] = 255

    result = calculate_management_result(group, values, mask)

    assert result.status == 'VISIBLE'
    assert result.measured_mm == 0.2
    assert result.delta_mm == -0.3
    assert result.correction_mm == -0.3


def test_관리영역의_픽셀별_편차도_수정한다():
    group = ManagementGroup('+0.5관리 기준면', 0.5, [], ['BOUNDARY'])
    values = np.full((5, 5), 1.0)
    mask = np.zeros((5, 5), np.uint8)
    mask[1:4, 1:4] = 255

    corrected = apply_management_value(values, group, mask)

    assert np.all(corrected[1:4, 1:4] == 0.5)
    assert corrected[0, 0] == 1.0
    assert np.all(values == 1.0), "원본 편차 배열을 직접 변경하면 안 됩니다."


def test_화면에_거의_없는_옆면은_자동_계산하지_않는다():
    group = ManagementGroup('-0.5관리 기준면', -0.5, [], ['BOUNDARY'])
    values = np.zeros((20, 20), float)
    mask = np.zeros((20, 20), np.uint8)

    result = calculate_management_result(
        group, values, mask, visible_ratio=0.0)

    assert result.status == 'HIDDEN'
    assert result.measured_mm is None
    assert result.correction_mm is None
