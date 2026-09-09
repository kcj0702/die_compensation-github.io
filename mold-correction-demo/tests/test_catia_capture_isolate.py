"""CATIA 캡처에서 부품만 잘라내는 부분 시험.

실제 캡처는 2500x1200 짜리 JPEG 라 저장소에 넣지 않는다. 대신 실측에서
드러난 **모양**을 손으로 지어 낸다 — 빗금으로 그려진 부품 하나와, 화면
왼쪽 위의 작고 꽉 찬 스펙 트리 아이콘 하나.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("cv2")

from cad_import.catia_capture import _isolate_part  # noqa: E402


def 캡처만들기(wide=2499, high=1209, 빗금간격=6, 아이콘=(46, 246, 40, 44)):
    """흰 화면에 빗금으로 그린 부품과 꽉 찬 UI 아이콘을 얹는다.

    빗금 간격 6px 은 실측 캡처에서 나온 값에 맞췄다 — 부품 상자 안이
    어두운 선 41.5% 와 흰 배경 57.8% 로 갈리고 중간 밝기가 없었다.
    """
    shot = np.full((high, wide, 3), 255, dtype=np.uint8)
    # 부품: 화면 한가운데 1100 x 470 자리에 세로 빗금만 긋는다.
    left, top, w, h = 700, 380, 1100, 470
    for x in range(left, left + w, 빗금간격):
        shot[top:top + h, x:x + 2] = (110, 198, 110)   # CATIA 초록
    # UI 아이콘: 작지만 **꽉 찬** 덩어리. 오프닝에 안 지워진다.
    ix, iy, iw, ih = 아이콘
    shot[iy:iy + ih, ix:ix + iw] = (200, 220, 40)
    return shot, (left, top, w, h)


def test_빗금으로_그려진_부품을_아이콘_대신_잡는다():
    """실측에서 이게 뒤집혀 46x50 짜리 아이콘이 시트 배경으로 나갔다."""
    shot, (left, top, w, h) = 캡처만들기()
    잘린것, 왼쪽, 위쪽 = _isolate_part(shot)

    높이, 너비 = 잘린것.shape[:2]
    # 여백 6% 를 붙이므로 딱 맞지는 않는다. 부품 크기의 언저리면 된다.
    assert 너비 > w * 0.9, f"부품 너비 {w} 를 잡아야 하는데 {너비} 만 나왔다"
    assert 높이 > h * 0.9, f"부품 높이 {h} 를 잡아야 하는데 {높이} 만 나왔다"
    # 아이콘(40x44)을 잡았다면 100px 도 안 된다.
    assert 너비 > 200 and 높이 > 200
    # 자른 자리도 부품 쪽이어야 한다 — 아이콘은 (46, 246) 에 있다.
    assert 왼쪽 > 400, f"화면 왼쪽 위 아이콘을 잡았다 (left={왼쪽})"


def test_통칠된_부품은_예전처럼_그대로_잡는다():
    """빗금이 아니라 통칠로 오는 캡처도 있다. 그쪽을 망치면 안 된다."""
    shot = np.full((1209, 2499, 3), 255, dtype=np.uint8)
    shot[380:850, 700:1800] = (110, 198, 110)
    shot[246:290, 46:86] = (200, 220, 40)          # 같은 UI 아이콘
    잘린것, 왼쪽, _ = _isolate_part(shot)
    assert 잘린것.shape[1] > 1000 and 잘린것.shape[0] > 400
    assert 왼쪽 > 400


def test_배경만_있으면_원본을_그대로_돌려준다():
    민화면 = np.full((600, 800, 3), 255, dtype=np.uint8)
    잘린것, 왼쪽, 위쪽 = _isolate_part(민화면)
    assert 잘린것.shape == 민화면.shape
    assert (왼쪽, 위쪽) == (0, 0)
