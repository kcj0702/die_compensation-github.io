"""공통 유틸 — 이미지 입출력, 로깅, JSON 저장.

한글 경로에서 cv2.imread / cv2.imwrite 가 실패하는 문제가 있어
numpy 버퍼를 경유하는 래퍼를 제공한다. Windows 환경 필수.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ── 로깅 ─────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ── 이미지 IO (한글 경로 대응) ───────────────────────────────────
def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread 대체. 한글·공백 경로에서도 동작한다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ValueError(f"이미지를 디코딩할 수 없습니다: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> Path:
    """cv2.imwrite 대체. 한글·공백 경로에서도 동작한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"이미지를 인코딩할 수 없습니다: {path}")
    buf.tofile(str(path))
    return path


def read_rgb(path: str | Path) -> np.ndarray:
    """RGB 배열로 읽는다. 알파 채널은 흰 배경에 합성한다."""
    img = imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3:4].astype(np.float32)) / 255.0
        white = np.full_like(bgr, 255.0)
        bgr = bgr * alpha + white * (1.0 - alpha)
        img = bgr.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, rgb: np.ndarray) -> Path:
    return imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# ── JSON ─────────────────────────────────────────────────────────
def save_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"JSON 직렬화 불가: {type(o)}")


__all__ = [
    "get_logger", "imread", "imwrite", "read_rgb", "write_rgb",
    "save_json", "load_json",
]
