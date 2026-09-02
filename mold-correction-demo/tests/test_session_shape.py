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
