"""파일명 태그를 읽어 품번/차종/카테고리 폴더로 자동 분류하고 실행하는 엔진.

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


def _normalize(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text).strip().lower()


# 고객사 기준 폴더 구조. 전체 품번을 첫 단계에 두어 차종을 몰라도 원하는
# 품번의 모든 자료를 바로 찾을 수 있게 한다. 세부 하위폴더는 카테고리별
# 규칙(구조도/패턴도/완성도/보정이력/OP 등)이 만드는 마지막 단계다.
AXIS_ITEM = "item"
AXIS_VEHICLE = "vehicle"
AXIS_CATEGORY = "category"
AXIS_DETAIL = "detail"
AXES: tuple[str, ...] = (AXIS_ITEM, AXIS_VEHICLE, AXIS_CATEGORY, AXIS_DETAIL)
AXIS_LABELS: dict[str, str] = {
    AXIS_ITEM: "품번",
    AXIS_VEHICLE: "차종",
    AXIS_CATEGORY: "카테고리",
    AXIS_DETAIL: "세부 하위폴더",
}
DEFAULT_FOLDER_ORDER: tuple[str, ...] = AXES
_FOLDER_ORDER_FILENAME = ".folder_order.json"
# 파일명에서 차종을 못 읽었을 때 쓰는 자리. 이름을 여기 하나로 두어야
# 분류가 만든 폴더를 순서 변경(migrate_folder_structure)도 알아본다.
UNKNOWN_VEHICLE_FOLDER = "_차종미확인"
# 빈 폴더를 형상관리에 남기기 위한 표시 파일. 사용자가 넣은 자료가 아니다.
PLACEHOLDER_NAMES = {".gitkeep"}


def is_valid_folder_order(order: Any) -> bool:
    """네 폴더 축을 빠짐없이 한 번씩 사용한 순열인지 확인한다."""
    return (
        isinstance(order, list)
        and len(order) == len(AXES)
        and len(set(order)) == len(AXES)
        and set(order) == set(AXES)
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
    if default_order and is_valid_folder_order(default_order):
        return list(default_order)
    return list(DEFAULT_FOLDER_ORDER)


def save_folder_order(base_dir: Path, order: list[str]) -> None:
    if not is_valid_folder_order(order):
        raise ValueError(
            "folder_order는 품번, 차종, 카테고리, 세부 하위폴더를 "
            "각각 한 번씩 포함해야 합니다."
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
    new_order: list[str],
) -> FolderMigrationResult:
    """파일을 임시 보관한 뒤 네 축의 새 순서로 안전하게 재배치한다.

    경로에서 품번·차종·카테고리를 이름 규칙으로 찾아내므로 세부 하위폴더가
    카테고리 앞이나 뒤 어느 위치에 있어도 나머지 경로를 그대로 보존한다.
    이미 제자리인 파일은 건드리지 않으므로, 순서를 그대로 두고 다시 저장하면
    이름 규칙만 어긋난 폴더(상세코드까지 갈라진 품번 폴더 등)를 정리하는
    용도로도 쓸 수 있다.
    """
    if not folder_root.is_dir():
        return FolderMigrationResult(moved=0, skipped=0, errors=[])
    if not is_valid_folder_order(new_order):
        return FolderMigrationResult(
            moved=0,
            skipped=0,
            errors=["네 폴더 항목이 모두 포함된 순서만 적용할 수 있습니다."],
        )

    category_names = {
        f"{category['key']}. {category['label']}"
        for category in rules.get("categories", [])
        if category.get("key") and category.get("label")
    }
    # 분류가 만든 '_차종미확인' 폴더도 차종 칸으로 본다 — 그러지 않으면 차종을
    # 못 읽은 파일만 옛 자리에 남아 두 구조가 섞인다.
    customers = {str(value).casefold() for value in rules.get("customers", [])}
    customers.add(UNKNOWN_VEHICLE_FOLDER.casefold())
    # 빈 폴더 표시(.gitkeep)는 무시하지 않고 구조를 따라 같이 옮긴다 — 그러지
    # 않으면 옛 자리의 빈 폴더가 지워지지 않아 두 구조가 섞인다. 다만 사용자가
    # 옮겼다고 셀 파일은 아니므로 moved 수에는 넣지 않는다.
    ignored_names = {
        str(value).casefold() for value in rules.get("ignored_names", [])
    } - PLACEHOLDER_NAMES

    def _is_vehicle(part: str) -> bool:
        """차종 폴더로 볼 이름인지 판단한다.

        분류기는 ``rules.json`` 에 없는 새 차종("XM" 등)도 스스로 만들므로,
        등록된 목록에만 의존하면 그렇게 만들어진 폴더가 순서 변경에서 통째로
        빠진다. 등록된 차종을 먼저 찾고, 없으면 차종처럼 생긴 영문 토큰을 쓴다
        (카테고리·세부 폴더 이름은 모두 숫자 접두사나 OP 번호가 붙어 걸리지 않는다).
        """
        return bool(re.fullmatch(r"[A-Za-z]{2,8}", part))

    def _item_value(part: str) -> str:
        """폴더 한 칸에서 품번 폴더 이름을 뽑는다.

        앞자리 계열 코드('64XX2')가 같으면 같은 품번으로 보므로 상세 코드
        ('-DR000')는 떼어 낸다 — 그래야 같은 제품의 자료가 한 폴더에 모인다.
        """
        match = _ITEM_NO_RE.fullmatch(part)
        if match:
            return match.group(1)
        if _FAMILY_ONLY_RE.fullmatch(part) and not _DATE_TOKEN_RE.match(part):
            return part
        return ""

    records: list[tuple[Path, dict[str, list[str]]]] = []
    skipped = 0
    for source in folder_root.rglob("*"):
        if not source.is_file() or source.name.casefold() in ignored_names:
            continue
        relative_parts = list(source.relative_to(folder_root).parts)
        folder_parts = relative_parts[:-1]
        item_index = next(
            (index for index, part in enumerate(folder_parts) if _item_value(part)),
            -1,
        )
        category_index = next(
            (index for index, part in enumerate(folder_parts) if part in category_names),
            -1,
        )
        vehicle_index = next(
            (index for index, part in enumerate(folder_parts) if part.casefold() in customers),
            -1,
        )
        if vehicle_index < 0:
            vehicle_index = next(
                (
                    index for index, part in enumerate(folder_parts)
                    if index != item_index and index != category_index and _is_vehicle(part)
                ),
                -1,
            )
        if min(item_index, category_index, vehicle_index) < 0:
            skipped += 1
            continue
        used = {item_index, category_index, vehicle_index}
        detail_parts = [part for index, part in enumerate(folder_parts) if index not in used]
        records.append(
            (
                source,
                {
                    AXIS_ITEM: [_item_value(folder_parts[item_index])],
                    AXIS_VEHICLE: [folder_parts[vehicle_index]],
                    AXIS_CATEGORY: [folder_parts[category_index]],
                    AXIS_DETAIL: detail_parts,
                },
            )
        )

    pending: list[tuple[Path, Path]] = []
    for source, axis_parts in records:
        destination_parts = [
            part
            for axis in new_order
            for part in axis_parts[axis]
        ]
        destination = folder_root.joinpath(*destination_parts, source.name)
        if destination != source:
            pending.append((source, destination))

    if not pending:
        return FolderMigrationResult(moved=0, skipped=skipped, errors=[])

    # 옮길 파일을 먼저 임시 폴더로 빼 둔다 — 어떤 파일의 새 자리가 다른 파일의
    # 옛 자리인 경우에도 서로 덮어쓰지 않게 하기 위해서다.
    staging_root = folder_root / f".folder-migration-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True)
    staged: list[tuple[Path, Path]] = []
    errors: list[str] = []
    for index, (source, destination) in enumerate(pending):
        stage = staging_root / f"{index}{source.suffix}"
        try:
            shutil.move(str(source), str(stage))
            staged.append((stage, destination))
        except OSError as exc:
            errors.append(f"{source.relative_to(folder_root)}: {exc}")

    for directory in sorted(
        (
            path for path in folder_root.rglob("*")
            if path.is_dir() and path != staging_root and staging_root not in path.parents
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    moved = 0
    for stage, destination in staged:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = _next_available_path(destination)
            shutil.move(str(stage), str(destination))
            if destination.name.casefold() not in PLACEHOLDER_NAMES:
                moved += 1
        except OSError as exc:
            errors.append(f"{destination.relative_to(folder_root)}: {exc}")
    try:
        staging_root.rmdir()
    except OSError:
        if staging_root.exists():
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
        tokens = re.findall(r"[0-9A-Za-z]+", prefix)
        for token in reversed(tokens):
            if token.casefold() in self._customers:
                return token.upper()
        for token in reversed(tokens):
            if 2 <= len(token) <= 8 and re.fullmatch(r"[A-Za-z]+", token):
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

    def _detail_target_parts(
        self,
        *,
        filename: str,
        category: _Category | None,
        process: str,
    ) -> tuple[list[str], str]:
        """카테고리 규칙에서 세부 하위폴더 경로 조각을 만든다."""
        if category is None:
            return [], ""
        rules = self._detail_rules.get(category.key, [])
        if not isinstance(rules, list):
            return [], ""
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
            return [], ""

        folder = str(selected.get("folder", "")).strip()
        if folder == "{process}":
            if not process:
                return [], ""
            parts = [process]
        else:
            parts = [folder] if folder else []
            if selected.get("use_process") and process:
                parts.append(process)
        if not parts:
            return [], ""
        return parts, "/".join(parts)

    def _vehicle_folder_name(self, parent: Path, customer: str) -> tuple[str, str]:
        """차종 폴더 이름과 그 근거를 돌려준다.

        파일명에서 차종을 읽었으면 그대로 쓴다. 못 읽었을 때만, 차종 칸이 놓일
        자리(``parent``)에 이미 폴더가 딱 하나 있으면 그것을 쓴다 — 같은 품번의
        다른 파일이 만들어 둔 차종 폴더일 가능성이 높다. 후보가 여럿이면 고르지
        않는다(엉뚱한 차종에 섞이는 것보다 미확인 폴더가 낫다).
        """
        if customer.strip():
            return customer.strip(), ""
        category_names = {entry.folder_name for entry in self._categories}
        try:
            candidates = [
                entry.name for entry in parent.iterdir()
                if entry.is_dir() and entry.name not in category_names
            ] if parent.is_dir() else []
        except OSError:
            candidates = []
        if len(candidates) == 1:
            return candidates[0], f"기존 차종 폴더 '{candidates[0]}'를 사용합니다"
        return (
            UNKNOWN_VEHICLE_FOLDER,
            f"차종을 읽지 못해 '{UNKNOWN_VEHICLE_FOLDER}' 폴더를 사용합니다",
        )

    def _resolve_target(
        self,
        *,
        family: str,
        item_no: str,
        category: _Category | None,
        customer: str,
        detail_parts: list[str],
    ) -> tuple[Path | None, str, list[str]]:
        """선택된 네 축의 순서대로 대상 경로를 만든다.

        품번 폴더는 계열 코드까지만 쓴다("64XX2-DR000"이 아니라 "64XX2") —
        앞자리가 같으면 같은 품번이라, 상세 코드까지 폴더를 가르면 같은 제품의
        자료가 흩어진다. 차종 폴더에는 엑셀의 차종 값만 쓴다(예: ``JM``).
        품명은 파일 태그와 카탈로그에 보존하되, 고객사가 지정한 네 단계 밖의
        폴더를 추가하지 않는다. 어느 축이 몇 번째로 오는지는 화면에서 정한
        순서만 따르므로, 기존 폴더를 찾는 것도 그 순서대로 내려가며 확인한다.
        """
        if not item_no:
            return None, "", ["품번이 없어 폴더 구조를 만들 수 없습니다"]
        if category is None:
            return None, "", ["카테고리가 없어 폴더 구조를 만들 수 없습니다"]

        reasons: list[str] = []
        matched_product_folder = ""
        axis_parts: dict[str, list[str]] = {
            AXIS_ITEM: [family or item_no],
            AXIS_VEHICLE: [],  # 차종 칸에 도착했을 때 그 자리를 보고 정한다.
            AXIS_CATEGORY: [category.folder_name],
            AXIS_DETAIL: detail_parts,
        }
        current = self._folder_root
        for axis in self._folder_order:
            if axis == AXIS_VEHICLE:
                vehicle_name, vehicle_reason = self._vehicle_folder_name(current, customer)
                axis_parts[AXIS_VEHICLE] = [vehicle_name]
                if vehicle_reason:
                    reasons.append(vehicle_reason)
            for part in axis_parts[axis]:
                candidate = current / part
                if not candidate.is_dir():
                    reasons.append(f"{AXIS_LABELS[axis]} 폴더 '{part}'를 새로 만듭니다")
                elif axis == AXIS_ITEM:
                    matched_product_folder = candidate.name
                    reasons.append(f"기존 품번 폴더 '{candidate.name}'와 일치")
                current = candidate
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

        detail_parts, detail_path = self._detail_target_parts(
            filename=filename,
            category=category,
            process=process,
        )
        target_dir, matched_product_folder, target_reasons = self._resolve_target(
            family=family, item_no=item_no, category=category,
            customer=customer, detail_parts=detail_parts,
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
