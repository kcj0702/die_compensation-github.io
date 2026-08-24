"""OpenCV가 유니코드 경로에서도 안정적으로 이미지를 읽고 쓰게 한다."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """`cv2.imread`의 Windows 유니코드 경로 제약 없이 이미지를 읽는다."""
    path = Path(path)
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"이미지를 열 수 없음: {path}") from exc

    try:
        image = cv2.imdecode(encoded, flags)
    except cv2.error as exc:
        raise FileNotFoundError(f"이미지를 열 수 없음: {path}") from exc
    if image is None:
        raise FileNotFoundError(f"이미지를 열 수 없음: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """경로의 확장자로 인코딩해 유니코드 경로에 이미지를 저장한다."""
    path = Path(path)
    extension = path.suffix.lower() or ".png"
    if extension == ".jpg":
        extension = ".jpeg"
    try:
        ok, encoded = cv2.imencode(extension, image)
    except cv2.error as exc:
        raise OSError(f"지원하지 않는 이미지 확장자입니다: {path}") from exc
    if not ok:
        raise OSError(f"이미지를 인코딩할 수 없음: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise OSError(f"이미지를 저장할 수 없음: {path}") from exc
