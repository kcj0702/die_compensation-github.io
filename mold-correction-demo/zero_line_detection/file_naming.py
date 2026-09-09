"""현업 파일명 규칙에서 차종·품번·품명·공정·날짜를 읽는다.

[근거]
현업이 준 `추가 자료/OOO/파일명 예시.xlsx` 가 규칙을 정해 두었다.

    차종_품번_품명_공정_(종류)_날짜.확장자

    JM_67312-DZ000_LAYOUT_260827.zip
    JM_67312_DZ000_DASH LWR_OP10_260825.zip
    JM_67312-DZ000_DASH LWR_OP20_패턴도_260825.zip
    JM 67312-DZ000_보정적용_260803..xlsx
    260825_JDZ_DASH LWR_OP10_형상_UPRDIE_NC DATA.ZIP   <- 날짜가 앞에 온다

구분자가 `_` 와 공백을 섞어 쓰고, 품번의 하이픈이 언더바로 바뀌기도 한다
(`67312-DZ000` / `67312_DZ000`). 그래서 정규식 하나로 훑지 않고 토큰으로
쪼갠 뒤 하나씩 무엇인지 가린다.

[왜 필요한가 — 두 가지]

1. 날짜를 품번으로 착각하고 있었다.
   기존 규칙은 `[0-9]{2}[A-Z0-9]{2,4}` 라 NC 데이터 파일에서 맨 앞의
   `260825`(날짜)를 품번으로 집어냈다. 품번은 컬러바 범위를 고르는
   열쇠라 이게 틀리면 제로라인이 통째로 어긋난다.

2. 보정시트 머리말을 채울 수 있다.
   실제 시트(JM 67312-DZ000_보정적용.xlsx) 머리말은 이렇게 생겼다 —

       관리 NO    JM 67312-DZ000-13      (차종 품번-순번)
       공   정    OP50
       원소재     A6451P-T4S 1.8t        (재질 + 판 두께)
       PART NAME  DASH UPR LHD
       PART NO    67312-DZ000
       적용일자    2025-07-15

   원소재만 파일명에 없다. 나머지는 파일명에서 나온다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

# 공정 — OP10 ~ OP50
PROCESS = re.compile(r"^OP\d{2}$", re.I)
# 날짜 — YYMMDD 여섯 자리
DATED = re.compile(r"^(\d{2})(\d{2})(\d{2})$")
# 품번 앞부분 — 숫자 두 자리로 시작하는 4~6자리
CORE = re.compile(r"^\d{2}[A-Z0-9]{2,4}$", re.I)
# 품번 뒷부분 — DR000 / DZ000 / XB000 처럼 영문 두 자 + 숫자 세 자
TAIL = re.compile(r"^[A-Z]{2}\d{3}$", re.I)
# 차종 — JM · JD · CD8 · JDZ 처럼 짧은 영문(숫자 한 자리까지)
MAKER = re.compile(r"^[A-Z]{2,3}\d?$", re.I)

# 품번으로 보지 않을 것들. 파일명에 자주 섞이는 잡음이다.
NOT_A_PART = {"3D", "NC", "OP", "LH", "RH", "LHD", "RHD"}


@dataclass
class NamedFile:
    """파일명에서 읽어낸 것들. 못 읽은 칸은 None 이다."""

    part_no: str | None = None       # 67312-DZ000
    maker: str | None = None         # JM · JD · CD8
    part_name: str | None = None     # DASH LWR
    process: str | None = None       # OP10
    applied_at: str | None = None    # 2026-08-25
    control_no: str | None = None    # JM 67312-DZ000-01

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(stem: str) -> list:
    """구분자로 쪼갠다. `_` 와 공백을 같게 본다."""
    return [t for t in re.split(r"[_\s]+", stem) if t]


def _looks_like_part(token: str) -> bool:
    """품번인가.

    핵심은 **날짜를 걸러내는 것**이다. `260825` 도 "숫자 두 자리 + 4자"
    라서 모양만으로는 품번과 구별되지 않는다. 진짜 품번은 글자를 품고
    있거나(71XX2) 하이픈 꼬리를 달고 있다(67312-DZ000). 둘 다 없는
    순수 숫자 여섯 자리는 날짜로 본다.
    """
    head = token.split("-")[0].split("/")[0]
    if not CORE.match(head):
        return False
    if head.upper() in NOT_A_PART:
        return False
    if "-" in token or "/" in token:
        return True
    return not head.isdigit()          # 글자가 섞여 있어야 품번


def parse(filename: str) -> NamedFile:
    """파일명 하나를 읽는다.

    Args:
        filename: 확장자를 붙여도 되고 안 붙여도 된다.
    """
    stem = Path(filename).stem
    # `..xlsx` 처럼 점이 겹친 실제 파일이 있어 한 번 더 턴다
    stem = stem.rstrip(".")
    tokens = _tokens(stem)
    found = NamedFile()

    rest: list = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        # 품번이 `67312_DZ000` 처럼 갈라진 경우 뒤 토큰을 붙여 되살린다
        if (found.part_no is None and CORE.match(token.split("-")[0])
                and index + 1 < len(tokens) and TAIL.match(tokens[index + 1])):
            found.part_no = f"{token.split('-')[0]}-{tokens[index + 1]}".upper()
            index += 2
            continue
        if found.part_no is None and _looks_like_part(token):
            found.part_no = token.upper()
            index += 1
            continue
        if found.process is None and PROCESS.match(token):
            found.process = token.upper()
            index += 1
            continue
        if found.applied_at is None and DATED.match(token):
            year, month, day = DATED.match(token).groups()
            try:                      # 26xxxx -> 2026
                found.applied_at = date(
                    2000 + int(year), int(month), int(day)).isoformat()
            except ValueError:
                pass                  # 날짜 모양이지만 날짜가 아니면 넘긴다
            index += 1
            continue
        if (found.maker is None and found.part_no is None
                and MAKER.match(token) and token.upper() not in NOT_A_PART):
            found.maker = token.upper()
            index += 1
            continue
        rest.append(token)
        index += 1

    # 남은 영문 토큰이 품명이다. 한글 낱말(패턴도·보정적용 등)은 문서
    # 종류를 가리키는 말이라 품명에서 뺀다.
    words = [t for t in rest
             if re.fullmatch(r"[A-Za-z][A-Za-z0-9()\-]*", t)
             and t.upper() not in NOT_A_PART]
    if words:
        found.part_name = " ".join(words).upper()

    if found.part_no:
        head = f"{found.maker} " if found.maker else ""
        found.control_no = f"{head}{found.part_no}-01"
    return found


def part_no_of(filename: str) -> str | None:
    """품번만 필요할 때. 못 찾으면 None."""
    return parse(filename).part_no


__all__ = ["NamedFile", "parse", "part_no_of"]
