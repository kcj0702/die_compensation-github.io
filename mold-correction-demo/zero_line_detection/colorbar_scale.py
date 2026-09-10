"""Read the physical deviation range printed beside a detected colour bar."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def _endpoint_crop(
    image_bgr: np.ndarray,
    info: Any,
    endpoint: str,
) -> Image.Image:
    """Crop one colour-bar endpoint together with its printed numeric label."""
    height, width = image_bgr.shape[:2]
    bar_height = max(1, int(info.y1) - int(info.y0))
    # Crop only the endpoint label row.  The old 5.5%-of-bar crop included
    # the adjacent tick (for example both -3.00 and -2.75), allowing OCR to
    # return the inner tick and changing the zero-line mask.
    half_height = max(10, min(32, int(round(bar_height * 0.018))))
    horizontal_margin = max(48, min(140, int(round(width * 0.05))))
    endpoint_at_top = (endpoint == "min") == (info.vmin_at == "top")
    center_y = int(info.y0) if endpoint_at_top else int(info.y1) - 1
    if getattr(info, "side", "right") == "left":
        x0 = max(0, int(info.x0) - 12)
        x1 = min(width, int(info.x1) + horizontal_margin)
    else:
        x0 = max(0, int(info.x0) - horizontal_margin)
        x1 = min(width, int(info.x1) + 12)
    y0 = max(0, center_y - half_height)
    y1 = min(height, center_y + half_height + 1)
    crop_rgb = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def read_colorbar_range_mm(
    image_bgr: np.ndarray,
    colorbar_info: Any,
    reader: Any,
) -> tuple[float, float]:
    """OCR min/max labels from the detected bar and validate the physical range."""
    crops = [
        _endpoint_crop(image_bgr, colorbar_info, "min"),
        _endpoint_crop(image_bgr, colorbar_info, "max"),
    ]
    values = reader.read_values(crops, batch_size=2)
    focused = getattr(reader, "read_value_focused", None)
    if callable(focused):
        values = [
            focused(crop) if value is None else value
            for crop, value in zip(crops, values)
        ]
    if len(values) != 2 or values[0] is None or values[1] is None:
        raise RuntimeError(
            "편차 컬러바의 최소·최대 숫자를 읽지 못했습니다. "
            "품번 기본값으로 대체하지 않습니다."
        )
    vmin, vmax = float(values[0]), float(values[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise RuntimeError(
            f"편차 컬러바 범위가 유효하지 않습니다: {vmin:g}~{vmax:g} mm"
        )
    if not (vmin <= 0.0 <= vmax):
        raise RuntimeError(
            f"편차 컬러바에 0 mm가 포함되지 않습니다: {vmin:g}~{vmax:g} mm"
        )
    return vmin, vmax


__all__ = ["read_colorbar_range_mm"]
