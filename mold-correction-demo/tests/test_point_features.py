"""보정 포인트 주변 형상 읽기 시험.

실제 CAD 는 크고 느려서 저장소에 넣지 않는다. 대신 손으로 만든 작은
판으로 각 특징이 **뜻대로** 나오는지 본다 — 홀에 가까우면 값이 작고,
가장자리에 가까우면 그것도 작고, 평평한 데는 굴곡이 0 이어야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import point_features as pf  # noqa: E402

trimesh = pytest.importorskip("trimesh")


def 판만들기(wide=200.0, high=100.0, step=4.0, z=1.0):
    """촘촘한 **열린** 판을 짓는다.

    상자를 쓰면 두 가지가 안 맞는다 — 정점이 여덟 개뿐이라 국부 굴곡을
    잴 이웃이 없고, 닫힌 껍데기라 자유 모서리가 아예 없어 "가장자리까지"
    가 뜻을 잃는다. 실제 판금은 두께 1~2mm 의 열린 껍데기이고 삼각망
    간격이 1~2mm 다(실측 64XX1 이 369,082 삼각형). 그와 닮게 만든다.
    """
    xs = np.arange(-wide / 2, wide / 2 + step, step)
    ys = np.arange(-high / 2, high / 2 + step, step)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.column_stack([gx.ravel(), gy.ravel(),
                             np.full(gx.size, z, dtype=float)])
    faces = []
    wideN = len(xs)
    for r in range(len(ys) - 1):
        for c in range(wideN - 1):
            a = r * wideN + c
            faces.append([a, a + 1, a + wideN])
            faces.append([a + 1, a + wideN + 1, a + wideN])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)


@pytest.fixture()
def 평판():
    """200 x 100 평평한 판. 삼각망 간격 4mm."""
    return 판만들기()


def 점(x, y, z=1.0, dev=None, name="P"):
    return {"id": name, "position": [x, y, z], "deviation": dev}


def test_홀에_가까울수록_홀까지_거리가_작다(평판):
    구멍 = [{"kind": "hole", "center": [50.0, 0.0, 0.0],
            "diameter": 30.0, "radius": 15.0}]
    가까이, 멀리 = 점(45, 0, name="가까이"), 점(-60, 0, name="멀리")
    쌍 = pf.features_for([가까이, 멀리], 평판, 구멍)
    assert 쌍[0].hole_mm < 10
    assert 쌍[1].hole_mm > 100
    # 지름은 그 홀의 것을 그대로 들고 온다.
    assert 쌍[0].hole_dia == 30.0


def test_홀이_없으면_무한대로_둔다(평판):
    하나 = pf.features_for([점(0, 0)], 평판, [])[0]
    assert hasattr(하나, "hole_mm") and 하나.hole_mm == float("inf")
    assert 하나.hole_dia == 0.0


def test_가장자리에_가까우면_그_거리가_작다(평판):
    # 판이 200 x 100 이라 x=95 는 끝(x=100)에서 5mm, 한가운데는 50mm 다.
    끝, 가운데 = 점(95, 0, name="끝"), 점(0, 0, name="가운데")
    쌍 = pf.features_for([끝, 가운데], 평판, [])
    assert 쌍[0].rim_mm < 10
    assert 쌍[1].rim_mm > 40


def test_평평한_데는_굴곡이_0_에_가깝다(평판):
    하나 = pf.features_for([점(0, 0)], 평판, [])[0]
    assert 하나.curve < 0.05


def test_휜_데는_굴곡이_커진다():
    # 정점이 촘촘한 구를 쓴다. trimesh 의 원통은 끝 링에만 정점이 있어
    # 옆면 한가운데서는 반경 안에 이웃이 잡히지 않는다.
    구 = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    하나 = pf.features_for([점(20, 0, 0)], 구, [])[0]
    assert 하나.curve > 0.1


def test_성긴_자리도_넓혀서_잰다(평판):
    """반경 안에 정점이 모자라면 가까운 것을 끌어와 잰다.

    그냥 "모른다" 로 두면 **평평한 데만 골라 빠진다** — STEP 은 평평한
    면을 삼각형 몇 개로만 쪼개기 때문이다. 실측 64XX1 에서 79개 중 22개가
    그렇게 빠졌고, 그 상태로 세니 굴곡과 편차의 상관이 -0.30 으로 나왔다.
    빠짐을 없애자 -0.02 로 내려앉았다 — 앞의 값은 착시였다.
    """
    import math

    # 판에서 멀리 떨어져 반경 20mm 안에는 정점이 하나도 없는 자리.
    하나 = pf.features_for([점(0, 0, 500)], 평판, [])[0]
    assert not math.isnan(하나.curve)
    assert 하나.curve < 0.05          # 판이 평평하니 값도 작아야 한다


def test_굽힘R_은_홀과_따로_센다(평판):
    원통 = [
        {"kind": "hole", "center": [90.0, 0.0, 0.0], "diameter": 12.0, "radius": 6.0},
        {"kind": "fillet", "center": [5.0, 0.0, 0.0], "diameter": 8.0, "radius": 4.0},
    ]
    하나 = pf.features_for([점(0, 0)], 평판, 원통)[0]
    assert 하나.bend_mm < 10          # 굽힘 R 이 가깝고
    assert 하나.hole_mm > 80          # 홀은 멀다


def test_편차가_없어도_특징은_나온다(평판):
    하나 = pf.features_for([점(0, 0, dev=None)], 평판, [])[0]
    assert 하나.deviation is None
    assert 하나.rim_mm > 0


def test_관계는_편차가_있어야_잰다(평판):
    없음 = pf.features_for([점(0, 0), 점(10, 0)], 평판, [])
    assert pf.relate(없음) == {}

    구멍 = [{"kind": "hole", "center": [0.0, 0.0, 0.0],
            "diameter": 20.0, "radius": 10.0}]
    # 홀에서 멀수록 편차가 큰 자료를 지어 넣는다 — 상관이 +1 이어야 한다.
    지어냄 = [점(x, 0, dev=x * 0.01, name=f"P{x}") for x in (10, 30, 50, 70, 90)]
    관계 = pf.relate(pf.features_for(지어냄, 평판, 구멍))
    assert 관계["홀까지"] > 0.99


def test_표로_내보낸다(tmp_path, 평판):
    구멍 = [{"kind": "hole", "center": [0.0, 0.0, 0.0],
            "diameter": 20.0, "radius": 10.0}]
    rows = pf.features_for([점(30, 0, dev=-1.2, name="surf pt 1")], 평판, 구멍)
    target = pf.to_csv([("64XX2", rows[0])], tmp_path / "특징.csv")
    글자 = target.read_text(encoding="utf-8-sig")
    assert "부품" in 글자 and "홀까지mm" in 글자
    assert "64XX2" in 글자 and "surf pt 1" in 글자
    assert "-1.2" in 글자


def test_포인트가_없으면_빈_목록(평판):
    assert pf.features_for([], 평판, []) == []
