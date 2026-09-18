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
    # Endpoint text is centred slightly inside the bar (64XX2's bottom
    # ``-1.50`` sits about 18 px above y1).  The previous 1.8% crop clipped
    # the minus sign/text and OCR returned 0.0.  3.5% keeps that label while
    # still excluding the adjacent 0.25-step tick roughly 45 px away.
    half_height = max(18, min(32, int(round(bar_height * 0.035))))
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

    def implausible(index: int, value: Any) -> bool:
        """The minimum end must be negative and the maximum end positive."""
        if value is None:
            return True
        try:
            number = float(value)
        except (TypeError, ValueError):
            return True
        return (index == 0 and number >= 0.0) or (index == 1 and number <= 0.0)

    if callable(focused):
        values = [
            focused(crop) if implausible(index, value) else value
            for index, (crop, value) in enumerate(zip(crops, values))
        ]
    # Two things make an endpoint unreadable at first: the export can be saved upside down (scan,
    # colour bar and numbers together), and a small export prints the number only a few pixels tall
    # (a 578 x 338 export gives 6 px digits). Both are recoverable, so an endpoint that still makes
    # no sense is read again from the crop turned round, enlarged, and both at once - the same
    # picture, never a guessed number.
    if len(values) == 2 and any(implausible(index, value) for index, value in enumerate(values)):
        values = list(values)
        readers = [lambda picture: reader.read_values([picture], batch_size=1)[0]]
        if callable(focused):
            readers.append(focused)
        for index, value in enumerate(values):
            if not implausible(index, value):
                continue
            crop = crops[index]
            enlarged = crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS)
            seen: list[float] = []
            for variant in (crop.rotate(180), enlarged, enlarged.rotate(180)):
                for read in readers:
                    try:
                        again = read(variant)
                    except Exception:               # a reader that cannot take one crop at a time
                        continue
                    if not implausible(index, again):
                        seen.append(round(float(again), 4))
            # Two looks must agree. A number that is genuinely there only turned or too small reads
            # the same every time; one that is clipped by the image edge reads differently each
            # time, and a wrong range would silently rescale every deviation value.
            for candidate in seen:
                if seen.count(candidate) >= 2:
                    values[index] = candidate
                    break
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
    if not (vmin < 0.0 < vmax):
        raise RuntimeError(
            f"편차 컬러바는 음수 최솟값과 양수 최댓값을 모두 포함해야 합니다: "
            f"{vmin:g}~{vmax:g} mm"
        )
    return vmin, vmax


__all__ = ["read_colorbar_range_mm"]
