"""Look up and store the product-data image and alignment for a part number.

A deviation scan is produced for every trial, but the product-data render is one
fixed image per part. Registering it once keeps the existing "drop the scan in"
flow intact for every later scan of the same part.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config
from .alignment import Alignment


PART_NUMBER_RE = re.compile(config.PART_NUMBER_PATTERN)


def part_number_from_name(name: str) -> str | None:
    """Extract the part number from a file name.

    Both `JD_64XX2-DR000 3D 스캔.png` and `64XX2-DR000 제품데이터.png` yield
    `64XX2-DR000`, which is the key the company file-naming rule already uses.
    """
    match = PART_NUMBER_RE.search(Path(name).stem.upper())
    return match.group(0) if match else None


def base_number(part_number: str) -> str:
    """Return the part number without its trailing variant code.

    Real pairs are not always exact: the `67XX6-DR000` scan belongs with the
    `67XX6-DR050` product data. The shared prefix is what links them.
    """
    return part_number.split("-", 1)[0]


def read_image(path: Path) -> np.ndarray:
    """Read an image from a path that may contain Korean characters."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG to a path that may contain Korean characters."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"이미지를 PNG로 변환할 수 없습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


@dataclass(frozen=True)
class ProductMatch:
    """A registered product image found for a requested part number."""

    part_number: str
    path: Path
    exact: bool


class ProductLibrary:
    """Product-data images stored one PNG per part number."""

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory or config.PRODUCT_DIR)

    def path_for(self, part_number: str) -> Path:
        return self.directory / f"{part_number}.png"

    def registered(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(path.stem for path in self.directory.glob("*.png"))

    def find(self, part_number: str) -> ProductMatch | None:
        """Return the registered image for a part number, exact match first."""
        exact = self.path_for(part_number)
        if exact.is_file():
            return ProductMatch(part_number, exact, exact=True)

        prefix = base_number(part_number)
        for candidate in self.registered():
            if base_number(candidate) == prefix:
                return ProductMatch(candidate, self.path_for(candidate), exact=False)
        return None

    def register(self, part_number: str, image: np.ndarray) -> Path:
        """Store the product image for a part number, replacing any previous one."""
        destination = self.path_for(part_number)
        write_png(destination, image)
        return destination

    def entries(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for part_number in self.registered():
            path = self.path_for(part_number)
            stat = path.stat()
            result.append(
                {
                    "partNumber": part_number,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                        timespec="seconds"
                    ),
                }
            )
        return result


class MeshLibrary:
    """CATIA 에서 export 한 STEP/STL 을 품번당 한 파일로 보관한다.

    ProductLibrary 의 PNG 와 같은 규칙(품번을 파일명, 접두어 매칭 폴백)을
    쓴다. 파일 확장자는 원본을 그대로 살려서, 서버가 STEP 인지 STL 인지
    구분해 알맞은 파서를 부르게 한다.
    """

    # 파일 크기가 크고 파서가 다양해 여기서 지원 확장자를 강하게 잡아 둔다.
    # .catpart / .catproduct 는 이 PC 의 CATIA 로 STEP 을 뽑아 캐시한 뒤 쓴다.
    # 앞쪽 확장자가 우선순위가 높다 — 같은 품번에 여러 확장자가 있으면 STEP 이
    # CATPart 보다 먼저 선택돼 CATIA COM 호출을 아예 건너뛴다.
    SUPPORTED_SUFFIXES = (".step", ".stp", ".stl", ".ply", ".obj", ".off",
                          ".glb", ".gltf", ".3mf", ".catpart", ".catproduct")
    _SUFFIX_PRIORITY = {suffix: rank for rank, suffix in enumerate(SUPPORTED_SUFFIXES)}

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory or config.MESH_DIR)

    def _stored_paths(self) -> list[Path]:
        """저장된 파일 목록. 확장자 우선순위(STEP > STL > CATPart) 순으로 정렬한다.

        같은 품번에 .CATPart 와 .step 이 함께 있으면 STEP 을 먼저 반환해 CATIA
        COM 호출을 건너뛰게 한다. 확장자가 같은 것끼리는 이름 알파벳 순.
        """
        if not self.directory.is_dir():
            return []
        entries = [
            path for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_SUFFIXES
        ]
        entries.sort(key=lambda p: (self._SUFFIX_PRIORITY.get(p.suffix.lower(), 99), p.name.lower()))
        return entries

    def registered(self) -> list[str]:
        # 같은 stem 에 여러 확장자가 있으면 우선순위가 높은 하나만 노출.
        seen: set[str] = set()
        result: list[str] = []
        for path in self._stored_paths():
            if path.stem in seen:
                continue
            seen.add(path.stem)
            result.append(path.stem)
        return result

    def find(self, part_number: str) -> ProductMatch | None:
        """등록된 mesh 를 찾는다. 정확한 품번 우선, 접두어 매칭 폴백."""
        for path in self._stored_paths():
            if path.stem == part_number:
                return ProductMatch(part_number, path, exact=True)
        prefix = base_number(part_number)
        for path in self._stored_paths():
            if base_number(path.stem) == prefix:
                return ProductMatch(path.stem, path, exact=False)
        return None

    def register(self, part_number: str, suffix: str, data: bytes) -> Path:
        """Mesh 데이터를 저장한다. 같은 품번의 기존 파일(다른 확장자 포함)은 지운다."""
        normalized = suffix.lower()
        if normalized not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"지원하지 않는 mesh 확장자입니다: {suffix}")
        self.directory.mkdir(parents=True, exist_ok=True)
        # 같은 품번이 다른 확장자로 남아 있으면 매칭이 갈리므로 먼저 제거.
        for existing in self.directory.glob(f"{part_number}.*"):
            if existing.suffix.lower() in self.SUPPORTED_SUFFIXES:
                existing.unlink()
        destination = self.directory / f"{part_number}{normalized}"
        destination.write_bytes(data)
        return destination

    def forget(self, part_number: str) -> bool:
        removed = False
        for existing in self.directory.glob(f"{part_number}.*"):
            if existing.suffix.lower() in self.SUPPORTED_SUFFIXES:
                existing.unlink()
                removed = True
        return removed

    def entries(self) -> list[dict[str, object]]:
        # 같은 품번에 여러 확장자가 있으면 우선순위가 가장 높은 하나만.
        seen: set[str] = set()
        result: list[dict[str, object]] = []
        for path in self._stored_paths():
            if path.stem in seen:
                continue
            seen.add(path.stem)
            stat = path.stat()
            result.append({
                "partNumber": path.stem,
                "format": path.suffix.lower().lstrip("."),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
            })
        return result


class AlignmentStore:
    """Confirmed scan-to-product orientations, one JSON file per part number."""

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory or config.ALIGNMENT_DIR)

    def path_for(self, part_number: str) -> Path:
        return self.directory / f"{part_number}.json"

    def load(self, part_number: str) -> Alignment | None:
        path = self.path_for(part_number)
        if not path.is_file():
            return None
        try:
            return Alignment.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def save(self, part_number: str, alignment: Alignment) -> Path:
        path = self.path_for(part_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = alignment.to_dict()
        payload["confirmedAt"] = datetime.now().isoformat(timespec="seconds")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def forget(self, part_number: str) -> bool:
        path = self.path_for(part_number)
        if not path.is_file():
            return False
        path.unlink()
        return True
