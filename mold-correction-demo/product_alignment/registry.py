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
