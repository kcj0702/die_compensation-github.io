"""현업 파일명 규칙 읽기.

예시는 전부 `추가 자료/OOO/파일명 예시.xlsx` 와 실제 받은 파일 이름에서
그대로 가져왔다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_line_detection.file_naming import parse  # noqa: E402
from zero_line_detection.register_sheet import part_no_from_name  # noqa: E402


@pytest.mark.parametrize("name,part_no", [
    ("JM_67312-DZ000_LAYOUT_260827.zip", "67312-DZ000"),
    # 품번의 하이픈이 언더바로 바뀌어도 되살린다
    ("JM_67312_DZ000_DASH LWR_OP10_260825.zip", "67312-DZ000"),
    ("JM 67312-DZ000_보정적용_260803..xlsx", "67312-DZ000"),
    ("JD_64XX2-DR000 3D 스캔.png", "64XX2-DR000"),
    ("67XX6-DR050_HDCT1750.CATPart", "67XX6-DR050"),
    ("CD8 71XX2_22 보정내용.xlsx", "71XX2"),
])
def test_품번을_읽는다(name, part_no):
    assert parse(name).part_no == part_no


def test_날짜를_품번으로_착각하지_않는다():
    """NC 데이터는 날짜가 맨 앞에 온다.

    예전 정규식은 260825 를 품번으로 집어냈다. 품번은 컬러바 범위를
    고르는 열쇠라 이게 틀리면 편차값이 통째로 어긋난다.
    """
    got = parse("260825_JDZ_DASH LWR_OP10_형상_UPRDIE_NC DATA.ZIP")
    assert got.part_no is None
    assert got.applied_at == "2026-08-25"
    assert got.process == "OP10"


def test_공정과_날짜와_차종():
    got = parse("JM_67312_DZ000_DASH LWR_OP10_260825.zip")
    assert got.process == "OP10"
    assert got.applied_at == "2026-08-25"
    assert got.maker == "JM"
    assert got.part_name == "DASH LWR"


def test_관리번호는_차종_품번_순번():
    """실제 시트의 관리 NO 가 'JM 67312-DZ000-13' 형태다."""
    assert parse("JM_67312-DZ000_DASH LWR_OP50_260825.zip").control_no \
        == "JM 67312-DZ000-01"


def test_날짜가_아닌_여섯자리는_넘긴다():
    """13월 45일 같은 건 날짜가 아니다."""
    assert parse("JD_64XX2-DR000_261345.png").applied_at is None


def test_규칙에_안_맞는_이름():
    got = parse("_boundary_anchors.png")
    assert got.part_no is None
    assert got.process is None


@pytest.mark.parametrize("name,key", [
    ("JD_64XX2-DR000 3D 스캔.png", "64XX2"),
    ("JD_67XX6-DR000 3D 스캔.png", "67XX6"),
    ("JD_71XX2-DR000 3D 스캔.png", "71XX2"),
    ("CD8 71XX2_22 보정내용.xlsx", "71XX2"),
])
def test_조회_열쇠는_짧은_형태를_지킨다(name, key):
    """컬러바 표와 제로라인 라이브러리가 이 형태를 열쇠로 쓴다.

    라이브러리는 정확 일치 조회라 '64XX2-DR000' 으로 바뀌면 못 찾는다.
    """
    assert part_no_from_name(name) == key


def test_품번이_없으면_파일명을_그대로_쓴다():
    """날짜(260825)를 품번이라고 우기지 않는다."""
    got = part_no_from_name("260825_JDZ_DASH LWR_OP10_NC DATA.ZIP")
    assert got != "260825"
    assert "260825" in got          # 파일명 그대로라 날짜는 남아 있다
