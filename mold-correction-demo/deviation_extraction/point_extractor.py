"""라벨 후보를 편차 포인트 레코드로 변환하고 결과 파일을 저장한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage

import config
from colormap_reader import build_lut
from label_detector import LabelCandidate, detect_labels

CROP_PADDING = 4
CSV_COLUMNS = (
    "point_id",
    "x_px",
    "y_px",
    "x_norm",
    "y_norm",
    "value_mm",
    "label_color",
    "in_zero_line",
    "confidence",
)


class ValueReader(Protocol):
    """라벨 crop을 숫자 하나로 변환하는 판독기 계약."""

    def read_value(self, crop: PILImage.Image) -> float | None: ...


@dataclass
class DeviationPoint:
    """편차값과 이미지 기준 좌표를 묶은 CSV 출력 레코드."""

    point_id: str
    x_px: int | None
    y_px: int | None
    x_norm: float | None
    y_norm: float | None
    value_mm: float | None
    label_color: str
    in_zero_line: bool
    confidence: str
    label_box: tuple[int, int, int, int] = field(repr=False)


def _crop_label(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """이미지 경계를 넘지 않도록 여백을 더해 라벨을 자른다."""
    x, y, w, h = box
    h_img, w_img = bgr.shape[:2]
    x0, y0 = max(x - CROP_PADDING, 0), max(y - CROP_PADDING, 0)
    x1, y1 = min(x + w + CROP_PADDING, w_img), min(y + h + CROP_PADDING, h_img)
    return bgr[y0:y1, x0:x1]


def _load_zero_line_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """선택 마스크를 읽고 원본 크기와 다르면 최근접 보간으로 맞춘다."""
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def extract_points(
    image_path: Path = config.DEVIATION_MAP_PATH,
    zero_line_mask_path: Path = config.ZERO_LINE_MASK_PATH,
    reader: ValueReader | None = None,
    reader_factory: Callable[[], ValueReader] | None = None,
    cross_check: bool = False,
) -> list[DeviationPoint]:
    """라벨 후보를 좌상단 원점의 픽셀·정규화 좌표와 판독값으로 변환한다."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"편차 이미지를 열 수 없음: {image_path}")

    h_img, w_img = bgr.shape[:2]
    candidates: list[LabelCandidate] = detect_labels(bgr)
    zero_mask = _load_zero_line_mask(zero_line_mask_path, (h_img, w_img))
    lut = build_lut(bgr) if cross_check else None

    if not candidates:
        return []
    if reader is None:
        if reader_factory is not None:
            reader = reader_factory()
        else:
            from vlm_reader import LabelValueReader

            reader = LabelValueReader()

    points: list[DeviationPoint] = []

    for i, cand in enumerate(candidates, start=1):
        crop_bgr = _crop_label(bgr, cand.box)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        value_mm = reader.read_value(PILImage.fromarray(crop_rgb))

        status: list[str] = []
        if value_mm is None:
            status.append("value_not_read")
        if not cand.traced or cand.point_xy is None:
            status.append("leader_line_not_traced")

        if cand.point_xy is None:
            x = y = None
            x_norm = y_norm = None
            in_zero_line = False
        else:
            x, y = cand.point_xy
            x_norm = round(x / w_img, 4)
            y_norm = round(y / h_img, 4)
            # 이미지 배열은 [y, x] 순서로 접근한다.
            in_zero_line = bool(zero_mask is not None and zero_mask[y, x] > 0)

        if (
            cross_check
            and lut is not None
            and value_mm is not None
            and x is not None
            and y is not None
        ):
            # 컬러맵은 판독값을 바꾸지 않고 불일치 상태만 기록한다.
            color_value = lut.to_value(bgr[y, x])
            if abs(color_value - value_mm) > config.CROSS_CHECK_MISMATCH_MM:
                status.append("color_mismatch")

        points.append(
            DeviationPoint(
                point_id=f"P{i:03d}",
                x_px=x,
                y_px=y,
                x_norm=x_norm,
                y_norm=y_norm,
                value_mm=value_mm,
                label_color=cand.label_color,
                in_zero_line=in_zero_line,
                confidence="|".join(status) if status else "ok",
                label_box=cand.box,
            )
        )
    return points


def save_csv(points: list[DeviationPoint], out_path: Path = config.OUTPUT_CSV_PATH) -> None:
    """포인트 목록을 UTF-8 BOM이 있는 CSV로 저장한다."""
    records = [{column: getattr(point, column) for column in CSV_COLUMNS} for point in points]
    df = pd.DataFrame(records, columns=CSV_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_debug_image(
    image_path: Path, points: list[DeviationPoint], out_path: Path = config.DEBUG_IMAGE_PATH
) -> None:
    """원본 위에 좌표와 판독값을 겹쳐 검출 상태를 시각화한다."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"편차 이미지를 열 수 없음: {image_path}")
    for p in points:
        color = (0, 0, 255) if p.confidence != "ok" else (0, 200, 0)
        label_x, label_y, label_w, label_h = p.label_box
        cv2.rectangle(
            bgr,
            (label_x, label_y),
            (label_x + label_w, label_y + label_h),
            color,
            1,
        )
        text_origin = (label_x, max(label_y - 4, 10))
        if p.x_px is not None and p.y_px is not None:
            label_center = (label_x + label_w // 2, label_y + label_h // 2)
            cv2.line(bgr, label_center, (p.x_px, p.y_px), color, 1)
            cv2.circle(bgr, (p.x_px, p.y_px), 4, color, -1)
            text_origin = (p.x_px + 6, max(p.y_px - 6, 10))
        cv2.putText(
            bgr,
            f"{p.point_id}:{p.value_mm}",
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), bgr):
        raise OSError(f"디버그 이미지를 저장할 수 없음: {out_path}")
