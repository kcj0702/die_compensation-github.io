"""현업이 준 두 가지 영라인 판정 방식이 실제로 그 규칙을 지키는지 본다.

  green_belt.py        녹색 x 부호전환대 교집합 (2026-08-25 자료)
  simple_zero_line.py  주요 0포인트를 잇는 직선, 꺾임 최대 1개 (2026-08-26)

합성 편차장을 써서 정답을 알고 검사한다. 실제 스캔에 대한 정확도는
각 모듈 문서에 실측으로 적어 두었다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_line_detection.green_belt import find_green_belts, green_threshold_for
from zero_line_detection.simple_zero_line import (
    build_tolerance_mask, colorbar_span_for, find_simple_zero_lines, to_millimetres,
)


@pytest.fixture
def sloped_field():
    """왼쪽이 -1mm, 오른쪽이 +1mm 로 기우는 판 — 영라인은 가운데 세로선."""
    height, width = 200, 400
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    values = np.tile(xs, (height, 1))
    mask = np.zeros((height, width), np.uint8)
    mask[20:180, 20:380] = 255
    return values, mask


def test_green_belt_finds_the_sign_transition(sloped_field):
    values, mask = sloped_field
    belts = find_green_belts(values, mask)
    assert belts, "부호가 뒤바뀌는 판에서는 벨트가 나와야 한다"
    # 가장 긴 벨트는 화면 가운데(부호가 바뀌는 x) 에 서 있어야 한다
    longest = belts[0]
    assert abs(longest.center[0] - 200) < 25
    assert longest.length_px > longest.area_px / longest.length_px, "길쭉해야 한다"


def test_green_belt_skips_a_field_that_never_changes_sign():
    """전부 +쪽이면 '전환점' 이 없으므로 아무것도 내면 안 된다."""
    values = np.full((200, 400), 0.6, np.float32)
    mask = np.zeros((200, 400), np.uint8)
    mask[20:180, 20:380] = 255
    assert find_green_belts(values, mask) == []


def test_green_threshold_follows_the_area_not_the_range(sloped_field):
    """이상치 한 점이 범위를 늘려도 기준이 흔들리면 안 된다."""
    values, mask = sloped_field
    before = green_threshold_for(values, mask, 0.28)
    spiked = values.copy()
    spiked[100, 200] = 50.0            # 범위를 25배로 늘리는 한 점
    after = green_threshold_for(spiked, mask, 0.28)
    assert after == pytest.approx(before, rel=0.05)


def test_datum_reset_recovers_a_shifted_field(sloped_field):
    """전체가 한쪽으로 쏠려도 4단계 리셋이 있으면 찾아낸다.

    자료 4단계: "제품 전체가 한쪽으로 쏠려 녹색이 거의 없다면, 가장 넓은
    평면 구간의 평균값을 0으로 리셋(Alignment 재설정)해야 합니다."
    편차를 전부 양수로 밀어 두면 리셋 없이는 부호 전환 자체가 없다.
    """
    values, mask = sloped_field
    shifted = values + 1.2
    assert float(shifted[mask > 0].min()) > 0, "전부 양수인 상황을 만든다"
    assert find_green_belts(shifted, mask, rezero=False) == []
    assert find_green_belts(shifted, mask, rezero=True)


def test_simple_zero_line_stays_straight(sloped_field):
    """0포인트가 한 직선 위에 있으면 꺾지 않는다."""
    values, mask = sloped_field
    key_points = [(200, 30, 30.0), (200, 100, 30.0), (200, 170, 30.0)]
    lines = find_simple_zero_lines(values, mask, key_points)
    assert len(lines) == 1
    line = lines[0]
    assert line.bend_count == 0
    assert len(line.points) == 2
    assert line.support_count == 3, "세 점이 한 직선으로 묶여야 한다"
    assert line.tolerance_coverage > 0.9


def test_simple_zero_line_never_exceeds_one_bend(sloped_field):
    """어떤 배치에서도 꼭짓점은 최대 3개다 — 곡선을 쓰지 않는다."""
    values, mask = sloped_field
    key_points = [(60, 40, 30.0), (200, 100, 30.0), (340, 160, 30.0), (80, 165, 30.0)]
    for line in find_simple_zero_lines(values, mask, key_points):
        assert line.bend_count <= 1
        assert 2 <= len(line.points) <= 3


def test_simple_zero_line_needs_two_points(sloped_field):
    values, mask = sloped_field
    assert find_simple_zero_lines(values, mask, [(200, 100, 30.0)]) == []


def test_tolerance_mask_is_the_half_millimetre_band(sloped_field):
    values, mask = sloped_field
    band = build_tolerance_mask(values, mask, tolerance_mm=0.5)
    inside = band > 0
    assert inside.any()
    # 침식 때문에 경계가 조금 깎이므로 여유를 두고 확인한다
    assert np.abs(values[inside]).max() <= 0.55
    assert not inside[mask == 0].any(), "부품 밖으로 나가면 안 된다"


@pytest.mark.parametrize("part_no, span", [
    ("JD_64XX2-DR000", (2.0, -1.6)),
    ("67XX6", (3.0, -3.0)),
    ("JD_71XX2", (2.0, -2.0)),
    ("모르는품번", None),
])
def test_colorbar_span_lookup(part_no, span):
    assert colorbar_span_for(part_no) == span


def test_normalised_values_are_restored_to_millimetres(sloped_field):
    """+-1 로 정규화돼 들어온 편차를 품번 컬러바 눈금으로 되돌린다."""
    values, mask = sloped_field
    normalised = values[mask > 0]
    millimetres = to_millimetres(values, mask, "64XX2")
    inside = millimetres[mask > 0]
    # 컬러바 2.0 ~ -1.6mm 이므로 +1 -> 2.0, -1 -> -1.6, 0 -> 가운데 0.2
    assert inside.max() == pytest.approx(normalised.max() * 1.8 + 0.2, abs=0.02)
    assert inside.min() == pytest.approx(normalised.min() * 1.8 + 0.2, abs=0.02)
    assert to_millimetres(np.zeros_like(values), mask, "64XX2")[100, 200] == pytest.approx(0.2)


def test_real_millimetre_values_are_left_alone(sloped_field):
    """이미 mm 눈금이면 두 번 환산하지 않는다."""
    values, mask = sloped_field
    already = values * 3.0
    assert np.array_equal(to_millimetres(already, mask, "67XX6"), already)
