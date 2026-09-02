"""파일명 태그를 읽어 품번 계열/자료유형 폴더로 자동 분류하고 실행하는 엔진.

분류 규칙은 ``rules.json`` 에 있고, 이 모듈은 그 규칙을 파일명에 적용하는 로직만
담당한다. 회사 NAS 경로나 카테고리 키워드가 바뀌어도 이 파일은 건드릴 필요가
없도록 규칙과 로직을 분리했다.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# 아진산업 품번 표기 예시(파일명 예시.xlsx, 품번별 폴더 정리 자료_예시 기준):
#   "64XX2-DR000", "67312-DZ000", "71XX2_22" 처럼 숫자로 시작하는 계열 코드와
#   숫자로 끝나는 상세 코드가 '-' 또는 '_' 로 붙는다. "DASH", "LWR" 같은 순수
#   문자 토큰과 구분하기 위해 두 토큰 모두 숫자를 하나 이상 포함해야 매칭한다.
_ITEM_NO_RE = re.compile(
    r"(?<![0-9A-Za-z])(\d[0-9A-Za-z]{2,6})[-_]([0-9A-Za-z]{1,6}\d)(?![0-9A-Za-z])"
)
# 상세 코드 없이 계열 코드만 적힌 파일명("ADC-64XX2 보정내용.xlsx")을 위한 보조 패턴.
_FAMILY_ONLY_RE = re.compile(r"(?<![0-9A-Za-z])(\d[0-9A-Za-z]{3,6})(?![0-9A-Za-z])")
_PROCESS_RE = re.compile(r"OP\s?-?\d{2,3}", re.IGNORECASE)
_DATE_TOKEN_RE = re.compile(r"^\d{6}$")


def _decode_product_segment(segment: str) -> dict[str, str]:
    """차종/상세품번 축 폴더 한 칸에서 차종·품명·품번·공정을 이름 모양으로
    뽑아낸다(위치가 아니라 모양으로 구분해야 축에 없는 칸이 섞여 있어도
    안전하다). 예전 방식으로 품명과 품번이 "PNL CTR FLR 65XX2-DR000"처럼
    한 칸에 뭉쳐 있어도 품번 패턴만 뽑아내고 나머지는 품명으로 살린다."""
    text = segment.strip()
    if _PROCESS_RE.fullmatch(text):
        return {"process": text}
    if re.fullmatch(r"[A-Za-z]{1,4}", text):
        return {"customer": text}
    if _FAMILY_ONLY_RE.fullmatch(text) and not _DATE_TOKEN_RE.match(text):
        return {"item_no": text}
    match = _ITEM_NO_RE.search(text)
    if match is None:
        return {"product_name": text}
    if match.group(0) == text:
        return {"item_no": text}
    before = text[: match.start()].strip(" _-")
    after = text[match.end():].strip(" _-")
    remainder = " ".join(part for part in (before, after) if part)
    result = {"item_no": match.group(0)}
    if remainder:
        result["product_name"] = remainder
    return result


def _normalize(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text).strip().lower()


# 폴더 계층을 이루는 세 축. 순서(folder_order)를 바꾸면 아래 세 이름의 순열로
# 폴더 경로를 다시 조립한다 — "품번 계열/자료유형/차종" 이든 "차종/자료유형/품번 계열"
# 이든 이 세 축을 어떤 순서로 쌓을지의 문제일 뿐이라, 축 자체는 그대로 두고
# classify() 안에서 순서만 따라가면 된다.
AXIS_FAMILY = "family"
AXIS_CATEGORY = "category"
AXIS_PRODUCT = "product"
AXES: tuple[str, ...] = (AXIS_FAMILY, AXIS_CATEGORY, AXIS_PRODUCT)
AXIS_LABELS: dict[str, str] = {
    AXIS_FAMILY: "품번 계열",
    AXIS_CATEGORY: "자료유형(카테고리)",
    AXIS_PRODUCT: "차종·상세품번",
}
DEFAULT_FOLDER_ORDER: tuple[str, ...] = (AXIS_FAMILY, AXIS_CATEGORY, AXIS_PRODUCT)
_FOLDER_ORDER_FILENAME = ".folder_order.json"


def is_valid_folder_order(order: Any) -> bool:
    """folder_order 는 품번 계열/카테고리/차종 세 축 중 겹치지 않게 고른 목록이다.

    차종·상세품번과 자료유형 축은 파일의 실제 소속을 구분하므로 항상 있어야 한다.
    품번 계열 축만 품번 자체와 중복될 수 있어 선택적으로 뺄 수 있다."""
    return (
        isinstance(order, list)
        and 1 <= len(order) <= len(AXES)
        and len(set(order)) == len(order)
        and all(axis in AXES for axis in order)
        and AXIS_PRODUCT in order
        and AXIS_CATEGORY in order
    )


def load_folder_order(base_dir: Path, default_order: list[str] | None = None) -> list[str]:
    """저장된 폴더 순서 설정을 읽는다. 없으면 규칙 기본값(또는 코드 기본값)을 쓴다."""
    path = base_dir / _FOLDER_ORDER_FILENAME
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if is_valid_folder_order(data):
            return data
    return list(default_order) if default_order else list(DEFAULT_FOLDER_ORDER)


def save_folder_order(base_dir: Path, order: list[str]) -> None:
    if not is_valid_folder_order(order):
        raise ValueError(
            "folder_order는 product와 category를 반드시 포함하고, "
            + ", ".join(AXES)
            + " 값을 겹치지 않게 사용해야 합니다."
        )
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / _FOLDER_ORDER_FILENAME).write_text(
        json.dumps(order, ensure_ascii=False), encoding="utf-8"
    )


@dataclass(frozen=True)
class FolderMigrationResult:
    moved: int
    skipped: int
    errors: list[str]


def migrate_folder_structure(
    folder_root: Path,
    rules: dict[str, Any],
    previous_order: list[str],
    new_order: list[str],
) -> FolderMigrationResult:
    """자료유형 단위로 실제 폴더를 새 축 순서로 재배치한다.

    예전 구현은 가장 아래의 빈 폴더를 한 건으로 취급했다. OP10 같은 표준 하위
    폴더가 생긴 뒤에는 그 이름까지 차종/품명으로 오해해 루트에 폴더가 계속
    누적될 수 있었다. 여기서는 정확한 자료유형 폴더와 품번 폴더를 먼저 찾고,
    그 둘 사이의 내용 전체를 하나의 묶음으로 임시 보관한 다음 새 경로에 합친다.
    따라서 구조도/패턴도/OP 하위 트리는 이동 중에도 그대로 유지된다.
    """
    if previous_order == new_order or not folder_root.is_dir():
        return FolderMigrationResult(moved=0, skipped=0, errors=[])

    if not is_valid_folder_order(previous_order) or not is_valid_folder_order(new_order):
        return FolderMigrationResult(
            moved=0,
            skipped=0,
            errors=["자료유형과 차종·상세품번 축이 포함된 폴더 순서만 이동할 수 있습니다."],
        )

    category_names = {
        f"{category['key']}. {category['label']}"
        for category in rules.get("categories", [])
        if category.get("key") and category.get("label")
    }
    customers = {str(value).casefold() for value in rules.get("customers", [])}

    # 옛 순서에 품번 계열 축이 따로 있었다면, 이름 전체가 "64XX2"처럼 계열
    # 코드 하나뿐인 폴더는 그 축의 껍데기일 뿐 품번 폴더가 아니다 — 그때만
    # 품번으로 인정하지 않는다. "JD PNL DASH 64XX2"처럼 차종·품명과 합쳐진
    # 이름은 축이 남아있어도 실제 품번 폴더이므로 계속 인식해야 한다 —
    # 안 그러면 그 폴더가 옮길 대상에서 통째로 빠져 자기 축만 남고
    # 나머지(차종·품명)가 안 옮겨지는 사고가 난다.
    family_axis_was_separate = AXIS_FAMILY in previous_order

    def _item_no_from_name(name: str) -> str:
        match = _ITEM_NO_RE.search(name)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        if family_axis_was_separate and _FAMILY_ONLY_RE.fullmatch(name):
            return ""
        # 새 명명 규칙은 "JD PNL DASH 64XX2" 처럼 차종·품명·계열을 한 폴더 이름에
        # 공백으로 이어붙이므로, 계열 코드가 이름 전체가 아니라 일부(보통 맨 끝)일
        # 수도 있다 — 그래서 뒤에서부터 찾아 날짜 등 다른 숫자와 헷갈리지 않는다.
        for family_match in reversed(list(_FAMILY_ONLY_RE.finditer(name))):
            token = family_match.group(1)
            if not _DATE_TOKEN_RE.match(token):
                return token
        return ""

    def _product_values(item_dir: Path, category_dir: Path) -> dict[str, str] | None:
        parts = item_dir.relative_to(folder_root).parts
        item_index = next(
            (index for index, part in enumerate(parts) if _item_no_from_name(part)),
            -1,
        )
        if item_index < 0:
            return None
        item_no = _item_no_from_name(parts[item_index])
        family = item_no.split("-", 1)[0]
        before_item = list(parts[:item_index])
        customer = next(
            (part for part in before_item if part.casefold() in customers),
            "",
        )
        ignored = {family.casefold(), category_dir.name.casefold()}
        if customer:
            ignored.add(customer.casefold())
        product_parts = [
            part for part in before_item
            if part.casefold() not in ignored
            and part not in category_names
            and not _PROCESS_RE.fullmatch(part)
        ]
        if not customer and not product_parts:
            # "JD PNL DASH 64XX2-DR000" 처럼 차종·품명·계열이 폴더 하나에 공백으로
            # 합쳐진 새 명명 규칙 — 그 폴더 이름 자체에서 이미 알아낸 품번/계열
            # 토큰만 정확히 빼고 나머지로 차종·품명을 되살린다(토큰 하나만 놓고
            # 다시 _item_no_from_name 을 부르면, 계열 코드 단독 토큰이 "품번 계열
            # 축의 껍데기"로 오인되어 못 걸러진다 — 위에서 이미 확정한 item_no/
            # family 값과 직접 비교해야 정확하다). 위쪽 경로에 차종/품명이 이미
            # 별도 폴더로 있었다면(구버전 구조) 거긴 손대지 않는다.
            exclude = {item_no.casefold(), family.casefold()}
            remainder = [token for token in parts[item_index].split() if token.casefold() not in exclude]
            if remainder and remainder[0].casefold() in customers:
                customer = remainder[0]
                remainder = remainder[1:]
            product_parts = remainder
        return {
            "customer": customer,
            "product_name": " ".join(product_parts),
            "item_no": item_no,
            "family": family,
        }

    def _unwrap_family_layer(path: Path, family: str) -> Path:
        """옛 순서에서 품번 계열이 카테고리/품번 폴더 바로 밑에 중첩되어 있었다면
        그 껍데기 폴더 안까지 들어간다. 그대로 두면 새 순서에 품번 계열 축이
        빠졌을 때 그 폴더 이름만 빈 채로 계속 남는다."""
        if not family:
            return path
        try:
            children = list(path.iterdir())
        except OSError:
            return path
        if len(children) == 1 and children[0].is_dir() and children[0].name.casefold() == family.casefold():
            return _unwrap_family_layer(children[0], family)
        return path

    records: list[tuple[Path, str, dict[str, str]]] = []
    category_dirs = sorted(
        (
            path for path in folder_root.rglob("*")
            if path.is_dir() and path.name in category_names
        ),
        key=lambda path: len(path.parts),
    )
    for category_dir in category_dirs:
        ancestors = list(category_dir.relative_to(folder_root).parents)
        item_ancestor = next(
            (
                folder_root / ancestor
                for ancestor in ancestors
                if ancestor != Path(".") and _item_no_from_name(ancestor.name)
            ),
            None,
        )
        if item_ancestor is not None:
            values = _product_values(item_ancestor, category_dir)
            if values:
                source = _unwrap_family_layer(category_dir, values["family"])
                records.append((source, category_dir.name, values))
            continue

        item_dirs = [
            path for path in category_dir.rglob("*")
            if path.is_dir() and _item_no_from_name(path.name)
        ]
        for item_dir in item_dirs:
            values = _product_values(item_dir, category_dir)
            if values:
                source = _unwrap_family_layer(item_dir, values["family"])
                records.append((source, category_dir.name, values))

    if not records:
        return FolderMigrationResult(moved=0, skipped=0, errors=[])

    staging_root = folder_root / f".folder-migration-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True)
    moved = 0
    skipped = 0
    errors: list[str] = []

    def _merge_path(source: Path, destination: Path) -> None:
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            for child in list(source.iterdir()):
                _merge_path(child, destination / child.name)
            source.rmdir()
            return
        if destination.exists():
            if source.name == ".gitkeep":
                source.unlink()
                return
            index = 1
            while destination.with_name(
                f"{destination.stem} ({index}){destination.suffix}"
            ).exists():
                index += 1
            destination = destination.with_name(
                f"{destination.stem} ({index}){destination.suffix}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    staged: list[tuple[Path, Path, Path]] = []
    for index, (source_root, category_name, values) in enumerate(records):
        destination_parts: list[str] = []
        for axis in new_order:
            if axis == AXIS_FAMILY:
                destination_parts.append(values["family"])
            elif axis == AXIS_CATEGORY:
                destination_parts.append(category_name)
            elif axis == AXIS_PRODUCT:
                # 차종 폴더 안에 품명 폴더, 그 안에 품번(계열) 폴더 — classify() 의
                # 새 폴더 생성 규칙과 동일하게 각각 별도 하위 폴더로 중첩한다.
                destination_parts.extend(
                    part for part in (
                        values["customer"], values["product_name"], values["family"]
                    ) if part
                )
        destination_root = folder_root.joinpath(*destination_parts)
        stage = staging_root / str(index)
        stage.mkdir()
        try:
            for child in list(source_root.iterdir()):
                _merge_path(child, stage / child.name)
            staged.append((stage, destination_root, source_root))
        except OSError as exc:
            skipped += 1
            errors.append(f"{source_root.relative_to(folder_root)}: {exc}")

    for directory in sorted(
        (
            path for path in folder_root.rglob("*")
            if path.is_dir() and staging_root not in path.parents and path != staging_root
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    for stage, destination_root, source_root in staged:
        try:
            destination_root.mkdir(parents=True, exist_ok=True)
            for child in list(stage.iterdir()):
                _merge_path(child, destination_root / child.name)
            stage.rmdir()
            if source_root.resolve() != destination_root.resolve():
                moved += 1
        except OSError as exc:
            errors.append(f"{destination_root.relative_to(folder_root)}: {exc}")

    try:
        staging_root.rmdir()
    except OSError:
        errors.append(f"임시 이동 폴더를 정리하지 못했습니다: {staging_root.name}")

    return FolderMigrationResult(moved=moved, skipped=skipped, errors=errors)


@dataclass(frozen=True)
class Classification:
    customer: str
    item_no: str
    family: str
    product_name: str
    process: str
    category_key: str
    category_label: str
    confidence: int
    reasons: list[str]
    target_dir: Path | None
    matched_product_folder: str
    detail_path: str = ""


@dataclass(frozen=True)
class OperationResult:
    source: str
    destination: str
    operation: str
    status: str  # "success" | "skipped" | "error"
    message: str | None = None


@dataclass(frozen=True)
class _Category:
    key: str
    label: str
    keywords: tuple[str, ...]
    extensions: tuple[str, ...]

    @property
    def folder_name(self) -> str:
        return f"{self.key}. {self.label}"


def load_rules(path: Path) -> dict[str, Any]:
    """rules.json 을 읽고, 상대 경로는 rules.json 위치 기준 절대경로로 바꿔 돌려준다."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent

    def _resolve_root(value: str) -> str:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return str(candidate)

    raw["destination_root"] = _resolve_root(raw.get("destination_root", "data/organized"))
    raw["source_root"] = _resolve_root(raw.get("source_root", "data/incoming"))
    raw.setdefault("unclassified_folder", "_미분류")
    raw.setdefault("ignored_names", [])
    raw.setdefault("categories", [])
    if not is_valid_folder_order(raw.get("folder_order")):
        raw["folder_order"] = list(DEFAULT_FOLDER_ORDER)
    return raw


class FilenameClassifier:
    """파일명에서 품번/자료유형 태그를 읽어 정리 대상 폴더를 추정한다."""

    def __init__(
        self,
        rules: dict[str, Any],
        folder_root: Path,
        folder_order: list[str] | None = None,
    ) -> None:
        self._folder_root = folder_root
        order = folder_order or rules.get("folder_order")
        self._folder_order = list(order) if is_valid_folder_order(order) else list(DEFAULT_FOLDER_ORDER)
        self._categories = [
            _Category(
                key=str(item["key"]),
                label=str(item["label"]),
                keywords=tuple(_normalize(keyword) for keyword in item.get("keywords", [])),
                extensions=tuple(ext.lower() for ext in item.get("extensions", [])),
            )
            for item in rules.get("categories", [])
        ]
        self._fallback_extensions = {
            str(extension).lower(): str(category_key)
            for extension, category_key in rules.get("fallback_extensions", {}).items()
        }
        self._detail_rules = rules.get("detail_rules", {})
        self._customers = {str(value).casefold() for value in rules.get("customers", [])}
        self._known_tag_keywords: set[str] = {
            keyword for category in self._categories for keyword in category.keywords
        }
        for rule_list in self._detail_rules.values():
            if not isinstance(rule_list, list):
                continue
            for rule in rule_list:
                if not isinstance(rule, dict):
                    continue
                for keyword in rule.get("keywords", []):
                    self._known_tag_keywords.add(_normalize(str(keyword)))

    def _match_item_no(self, stem: str) -> tuple[str, str, str, int]:
        """(품번, 계열, 앞쪽 차종 토큰, 품번 뒤 남은 텍스트의 시작 위치)를 돌려준다.

        못 찾으면 빈 문자열들과 -1. 마지막 값은 품명/공정을 다시 정규식으로
        찾지 않고 이 매칭 위치 그대로 잘라 쓰기 위한 것이다 — 품번을 "-"로
        표준화해 돌려주다 보니(원문이 "_"였을 수도 있음) 문자열로 다시
        찾으면 실패할 수 있다.
        """
        match = _ITEM_NO_RE.search(stem)
        if match:
            family, suffix = match.group(1), match.group(2)
            item_no = f"{family}-{suffix}"
            prefix = stem[: match.start()].strip(" _-")
            return item_no, family, prefix, match.end()
        for candidate in _FAMILY_ONLY_RE.finditer(stem):
            token = candidate.group(1)
            if _DATE_TOKEN_RE.match(token):
                continue  # "260825" 같은 YYMMDD 날짜 토큰은 품번이 아니다.
            prefix = stem[: candidate.start()].strip(" _-")
            return token, token, prefix, candidate.end()
        return "", "", "", -1

    def _customer_from_prefix(self, prefix: str) -> str:
        token = prefix.split()[-1] if prefix.split() else prefix
        token = token.strip(" _-")
        if token and len(token) <= 8 and re.fullmatch(r"[0-9A-Za-z]+", token):
            return token.upper()
        return ""

    def _detect_customer_match(self, stem: str) -> re.Match | None:
        """품번이 없어 위치를 못 잡을 때, "_"·공백으로 나뉜 토큰 중 차종처럼
        생긴 것을 찾아 그 위치(문자열 안 시작·끝)를 돌려준다 — 품번 없는
        파일에서도 품명을 잘라낼 기준점으로 쓴다. 등록된 차종 목록
        (rules.json)에 있는 토큰을 먼저 찾고, 없으면 순수 영문 토큰을 새
        차종 후보로 본다 — 아직 목록에도 없고 폴더도 없는 진짜 새 차종
        ("JDZ" 등)도 놓치지 않기 위해서다."""
        fallback: re.Match | None = None
        for token_match in re.finditer(r"[^_\-\s]+", stem):
            token = token_match.group(0)
            if _DATE_TOKEN_RE.match(token) or _PROCESS_RE.fullmatch(token):
                continue
            if token.casefold() in self._customers:
                return token_match
            if fallback is None and re.fullmatch(r"[A-Za-z]{2,8}", token):
                fallback = token_match
        return fallback

    def _product_name_and_process(self, stem: str, tail_start: int) -> tuple[str, str]:
        process_match = _PROCESS_RE.search(stem)
        process = process_match.group(0).upper().replace(" ", "").replace("-", "") if process_match else ""

        tail_end = process_match.start() if process_match and process_match.start() >= tail_start else len(stem)
        tail = stem[tail_start:tail_end].strip(" _-")

        tokens = [
            cleaned for token in re.split(r"[_\-]+", tail)
            # "260803..xlsx"처럼 파일명에 마침표가 겹쳐 붙어 있으면 날짜
            # 토큰 끝에 "."이 남아 "260803."처럼 되어 날짜로 인식되지 못하고
            # 그대로 품명에 섞여 들어간다 — 앞뒤 마침표를 떼고 나서 판단한다.
            # "보정적용"·"LAYOUT"처럼 이미 자료유형 키워드로 등록된 말은
            # 진짜 품명이 아니라 문서 종류를 나타내는 말이므로 품명에서 뺀다.
            if (cleaned := token.strip(" .")) and not _DATE_TOKEN_RE.match(cleaned)
            and _normalize(cleaned) not in self._known_tag_keywords
        ]
        product_name = " ".join(tokens).strip()
        return product_name, process

    def _fallback_product_name(self, stem: str, tail_start: int, tail_end: int) -> str:
        """품번이 없어 "_" 구분자 기준으로 품명 구간을 못 자를 때 쓴다(예:
        "JM DASH LWR 성형해석 리포트 260825.ppt"처럼 공백만으로 이어진
        경우). 이미 자료유형 키워드로 등록된 말("성형해석", "리포트" 등)과
        날짜는 설명용 텍스트로 보고 빼고, 남는 말만 품명으로 삼는다."""
        tail = stem[tail_start:tail_end].strip(" .")
        tokens = [
            cleaned for token in re.split(r"[_\-\s]+", tail)
            if (cleaned := token.strip(" ."))
            and not _DATE_TOKEN_RE.match(cleaned)
            and _normalize(cleaned) not in self._known_tag_keywords
        ]
        return " ".join(tokens).strip()

    def _match_category(self, filename: str) -> tuple[_Category | None, str, str]:
        """(카테고리, 매칭방식, 근거키워드) 를 돌려준다. 매칭방식은 keyword|extension|""."""
        normalized_name = _normalize(filename)
        for category in self._categories:
            for keyword in category.keywords:
                if keyword and keyword in normalized_name:
                    return category, "keyword", keyword
        extension = Path(filename).suffix.lower()
        if extension:
            for category in self._categories:
                if extension in category.extensions:
                    return category, "extension", extension
            fallback_key = self._fallback_extensions.get(extension)
            if fallback_key:
                for category in self._categories:
                    if category.key == fallback_key:
                        return category, "extension", extension
        return None, "", ""

    def _append_detail_target(
        self,
        target_dir: Path | None,
        *,
        filename: str,
        category: _Category | None,
        process: str,
    ) -> tuple[Path | None, str]:
        """엑셀의 세부 구분(구조도/패턴도/완성도/보정이력/OP)을 최종 경로에 붙인다."""
        if target_dir is None or category is None:
            return target_dir, ""
        rules = self._detail_rules.get(category.key, [])
        if not isinstance(rules, list):
            return target_dir, ""
        normalized_name = _normalize(filename)
        selected: dict[str, Any] | None = None
        default_rule: dict[str, Any] | None = None
        for raw_rule in rules:
            if not isinstance(raw_rule, dict):
                continue
            if raw_rule.get("default_when_process"):
                default_rule = raw_rule
            keywords = raw_rule.get("keywords", [])
            if any(_normalize(str(keyword)) in normalized_name for keyword in keywords):
                selected = raw_rule
                break
        if selected is None and process:
            default_rule = default_rule or next(
                (rule for rule in rules if isinstance(rule, dict) and rule.get("folder") == "{process}"),
                None,
            )
            selected = default_rule
        if selected is None:
            return target_dir, ""

        folder = str(selected.get("folder", "")).strip()
        if folder == "{process}":
            if not process:
                return target_dir, ""
            parts = [process]
        else:
            parts = [folder] if folder else []
            if selected.get("use_process") and process:
                parts.append(process)
        if not parts:
            return target_dir, ""
        return target_dir.joinpath(*parts), "/".join(parts)

    def _find_by_item_no(self, parent: Path, item_no: str, family: str = "", max_depth: int = 4) -> Path | None:
        """품번이 포함된 폴더를 찾는다. 차종/품명/품번/공정처럼 몇 단계로
        묶여 있어도(예: "JD/PNL CTR FLR/65XX2-DR000/OP10") 그 안까지 들여다
        본다. 얕은 단계부터 확인해 가장 가까운 일치를 우선한다.

        새로 만드는 폴더는 상세 코드 없이 계열만 쓰므로("JD PNL DASH 64XX2"),
        전체 품번이 안 걸리면 계열만으로도 찾는다 — 안 그러면 같은 제품의
        서로 다른 파일마다 매번 새 폴더가 생긴다."""
        if not parent.is_dir():
            return None
        needle = item_no.replace("_", "-").lower()
        family_needle = family.lower() if family else ""
        frontier = [parent]
        for _ in range(max_depth):
            next_frontier: list[Path] = []
            for directory in frontier:
                try:
                    entries = [entry for entry in directory.iterdir() if entry.is_dir()]
                except OSError:
                    continue
                for entry in entries:
                    normalized = entry.name.replace("_", "-").lower()
                    if needle in normalized or (family_needle and family_needle in normalized):
                        return entry
                next_frontier.extend(entries)
            if not next_frontier:
                break
            frontier = next_frontier
        return None

    def _resolve_target(
        self,
        *,
        family: str,
        item_no: str,
        category: _Category | None,
        customer: str,
        product_name: str,
        process: str,
    ) -> tuple[Path | None, str, list[str]]:
        """folder_order 를 따라가며 대상 폴더를 조립한다.

        폴더 구조를 자유롭게(품번→카테고리→차종, 차종→카테고리→품번 등) 바꿀 수
        있어야 하므로 세 축을 순서대로 밟되, 각 축의 폴더 이름은 이미 알고 있는
        값(계열명, "01. 자료유형")으로 만들거나 — 차종/상세품번 축만은 정확한
        폴더명을 모르므로 기존 폴더 중 품번이 포함된 것을 찾고, 없으면 새로
        만들 이름을 제안한다.

        어느 축이든 해당 폴더가 아직 없으면 새로 만들 대상으로 제안한다 —
        "JDZ" 처럼 아직 없던 차종이 와도 그 폴더까지 자동으로 만들어지도록.
        차종/상세품번 축은 정식 품번 패턴(계열+상세코드)이 파일명에서 실제로
        읽혔을 때만 새 폴더를 제안하므로(다른 축으로 못 건너뛴 상태에서만
        여기까지 온다), 날짜 등을 품번으로 잘못 읽어 엉뚱한 폴더가 생기는
        사고는 여전히 막힌다.
        """
        current = self._folder_root
        matched_product_folder = ""
        reasons: list[str] = []
        descended = False

        for axis in self._folder_order:
            candidate: Path | None = None
            if axis == AXIS_FAMILY:
                if not family:
                    continue
                candidate = current / family
            elif axis == AXIS_CATEGORY:
                if category is None:
                    continue
                candidate = current / category.folder_name
            elif axis == AXIS_PRODUCT:
                if item_no:
                    match = self._find_by_item_no(current, item_no, family)
                    if match is not None:
                        matched_product_folder = match.name
                        current = match
                        reasons.append(f"기존 폴더 '{match.name}'과 품번 일치")
                        descended = True
                        continue
                    if item_no == family:
                        # 계열 코드만 애매하게 읽힌 경우, 기존 폴더가 없으면
                        # 새로 만들지 않는다 — 잘못 읽은 숫자로 엉뚱한 폴더가
                        # 생기는 사고를 막기 위해서다.
                        continue
                elif not customer and not product_name:
                    # 품번은 물론 차종·품명조차 못 읽었으면 이 축 자체를
                    # 건너뛴다 — 지어낼 근거가 아무것도 없다.
                    continue
                detailed_category = category is not None and category.key in self._detail_rules
                # 차종 폴더 안에 품명 폴더, 그 안에 품번(계열) 폴더 — 각각 별도
                # 하위 폴더로 중첩한다(기존 JD/MD/ND/SD 폴더와 같은 방식).
                # 품번을 모르면 그 칸은 비운다 — 없는 값을 지어내지 않는다.
                segments = [part for part in (customer, product_name, family) if part]
                if process and not detailed_category:
                    segments.append(process)
                if not segments:
                    continue
                candidate = current.joinpath(*segments)

            if candidate is None:
                continue
            if not candidate.is_dir():
                reasons.append(f"'{candidate.name}' 폴더가 없어 새로 만듭니다")
            current = candidate
            descended = True

        if not descended:
            return None, "", reasons
        return current, matched_product_folder, reasons

    def classify(self, path: Path) -> Classification:
        filename = path.name
        stem = path.stem
        reasons: list[str] = []
        score = 0

        item_no, family, prefix, tail_start = self._match_item_no(stem)
        customer = self._customer_from_prefix(prefix) if prefix else ""

        if item_no:
            product_name, process = self._product_name_and_process(stem, tail_start)
        else:
            # 품번 패턴이 아예 없는 파일(NC데이터·성형해석 리포트 등)은 품번을
            # 지어내지 않는다 — 차종·품명·공정처럼 파일명에 실제로 적혀 있는
            # 것만 읽고, 없는 값(품번)은 없는 채로 둔다.
            customer_match = self._detect_customer_match(stem)
            if customer_match and "_" in stem:
                product_name, process = self._product_name_and_process(stem, customer_match.end())
            else:
                process_match = _PROCESS_RE.search(stem)
                process = (
                    process_match.group(0).upper().replace(" ", "").replace("-", "")
                    if process_match else ""
                )
                if customer_match:
                    tail_start = customer_match.end()
                    tail_end = (
                        process_match.start()
                        if process_match and process_match.start() >= tail_start
                        else len(stem)
                    )
                    product_name = self._fallback_product_name(stem, tail_start, tail_end)
                else:
                    product_name = ""
            if customer_match:
                customer = customer_match.group(0).upper()

        if item_no:
            score += 45
            reasons.append(f"파일명에서 품번 '{item_no}' 인식")
        else:
            reasons.append("파일명에서 품번 패턴을 찾지 못했습니다 — 품번 폴더 없이 분류합니다")

        category, match_kind, matched_keyword = self._match_category(filename)
        if category and match_kind == "keyword":
            score += 35
            reasons.append(f"'{matched_keyword}' 키워드로 '{category.label}' 자료유형 판별")
        elif category and match_kind == "extension":
            score += 20
            reasons.append(f"확장자 '{matched_keyword}' 기준으로 '{category.label}' 자료유형 추정")
        else:
            reasons.append("자료유형 키워드/확장자를 인식하지 못했습니다")

        target_dir, matched_product_folder, target_reasons = self._resolve_target(
            family=family, item_no=item_no, category=category,
            customer=customer, product_name=product_name, process=process,
        )
        target_dir, detail_path = self._append_detail_target(
            target_dir,
            filename=filename,
            category=category,
            process=process,
        )
        if target_dir is not None:
            if matched_product_folder:
                score += 20
            order_label = " → ".join(AXIS_LABELS[axis] for axis in self._folder_order)
            reasons.append(f"폴더 순서 '{order_label}' 기준으로 위치 결정")
            reasons.extend(target_reasons)
            if detail_path:
                score += 5
                reasons.append(f"세부 폴더 '{detail_path}'까지 자동 선택")
        elif family:
            reasons.append("기준 폴더가 아직 없어 미분류로 이동합니다")

        return Classification(
            customer=customer,
            item_no=item_no,
            family=family,
            product_name=product_name,
            process=process,
            category_key=category.key if category else "",
            category_label=category.label if category else "",
            confidence=min(100, score),
            reasons=reasons,
            target_dir=target_dir,
            matched_product_folder=matched_product_folder,
            detail_path=detail_path,
        )


def classify_batch(classifier: FilenameClassifier, paths: list[Path]) -> list[Classification]:
    """같은 classifier 하나로 여러 파일을 분류한다."""
    return [classifier.classify(path) for path in paths]


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    index = 1
    while True:
        candidate = parent / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def execute_batch(
    pairs: list[tuple[Path, Path]],
    *,
    operation: str,
    conflict: str,
) -> list[OperationResult]:
    """(원본, 목적지) 쌍을 순서대로 복사/이동한다. 한 파일의 실패가 나머지를 막지 않는다."""
    results: list[OperationResult] = []
    for source, destination in pairs:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            final_destination = destination
            if destination.exists():
                if conflict == "skip":
                    results.append(
                        OperationResult(
                            source=str(source), destination=str(destination),
                            operation=operation, status="skipped",
                            message="이미 존재하는 파일이라 건너뛰었습니다.",
                        )
                    )
                    continue
                if conflict == "rename":
                    final_destination = _next_available_path(destination)
                # conflict == "overwrite" 는 final_destination 을 그대로 두고 덮어쓴다.

            if operation == "move":
                shutil.move(str(source), str(final_destination))
            else:
                shutil.copy2(str(source), str(final_destination))

            results.append(
                OperationResult(
                    source=str(source), destination=str(final_destination),
                    operation=operation, status="success", message=None,
                )
            )
        except OSError as exc:
            results.append(
                OperationResult(
                    source=str(source), destination=str(destination),
                    operation=operation, status="error", message=str(exc),
                )
            )
    return results


def write_history(results: list[OperationResult], log_root: Path) -> None:
    """실행 결과를 로컬 감사 로그(JSONL)에 남긴다. MariaDB 미설정 시에도 항상 남는다."""
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()
    log_path = log_root / f"{timestamp.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.jsonl"
    with log_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(
                json.dumps(
                    {
                        "source": result.source,
                        "destination": result.destination,
                        "operation": result.operation,
                        "status": result.status,
                        "message": result.message,
                        "timestamp": timestamp.isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
