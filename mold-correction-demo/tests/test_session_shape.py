"""작업 저장 형식 — 시트 머리말까지 담기는지.

TypeScript 쪽 형식이라 파이썬 시험으로는 **모양만** 지킨다. 저장에
빠지면 작업자가 채운 관리 NO·공정·원소재·적용일자가 새로고침 한 번에
날아간다 — 현업 시트는 그 칸이 비면 결재가 안 된다.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
STORE = (ROOT / "ui" / "app" / "session-store.ts").read_text(encoding="utf-8")
PAGE = (ROOT / "ui" / "app" / "page.tsx").read_text(encoding="utf-8")


def test_저장_형식에_머리말이_있다():
    assert re.search(r"head\?\s*:\s*Record<string, string>", STORE), (
        "session-store 형식에 head 가 없다")


def test_저장할_때_담는다():
    assert "head: hasHead ? { ...head } : undefined," in PAGE


def test_불러올_때_되살린다():
    assert PAGE.count("setSheetHeadByScan") >= 3, (
        "저장·자동복원·파일불러오기 세 곳에서 써야 한다")


def test_비우기가_머리말도_지운다():
    where = PAGE.index("clearSession();")
    assert "setSheetHeadByScan({})" in PAGE[where:where + 600]


def test_머리말_칸이_전부_입력칸이다():
    for key in ("controlNo", "partName", "process", "partNo",
                "material", "appliedAt"):
        assert f"field('{key}'" in PAGE, f"{key} 가 입력칸이 아니다"


def test_표준_공정_구역이_저장_형식에_있다():
    """보정시트는 "① 하형 용접" 처럼 구역을 고정된 자리에 표기한다.

    같은 부품이면 매번 같은 자리이므로 부품 좌표로 한 번 등록해 두면
    그 부품의 CAD 를 열 때마다 저절로 떠야 한다. 손으로 그린 구역과
    달리 **품번**에 묶는다 — 파일이 바뀌어도 부품이 같으면 같은 자리다.
    """
    assert re.search(r"zones\?\s*:\s*unknown\[\]", STORE), (
        "session-store 형식에 zones 가 없다")


def test_표준_구역을_저장하고_되살린다():
    assert "zones: zones?.length ? zones : undefined," in PAGE
    assert PAGE.count("setZonesByPart") >= 4, (
        "추가·저장·자동복원·파일불러오기에서 다 써야 한다")


def test_비우기가_표준_구역도_지운다():
    where = PAGE.index("clearSession();")
    assert "setZonesByPart(() => ({}))" in PAGE[where:where + 700]


def test_손_그림_저장소에_표준_구역이_섞이지_않는다():
    """표준 구역은 품번에 묶인다. 손 그림 저장소에 같이 넣으면
    파일마다 사본이 생겨 지워도 되살아난다."""
    assert "next.filter((r) => !r.standard)" in PAGE


def test_구역이_부품_좌표로_그려진다():
    viewer = (ROOT / "ui" / "app" / "cad-viewer.tsx").read_text(encoding="utf-8")
    assert "box?: { min:" in viewer, "좌표 상자 형식이 없다"
    # 화면은 원점을 옮겨 놓았으므로 되돌려야 제자리에 뜬다
    assert "mesh.summary.bounds.center" in viewer
