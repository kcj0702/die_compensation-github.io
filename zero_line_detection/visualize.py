"""검출 결과 시각화.

숫자만으로는 검출이 제대로 됐는지 알 수 없다.
원본 위에 0-Line 을 얹어 눈으로 확인할 수 있게 만든다.
데모 발표에서도 이 이미지가 그대로 쓰인다.
"""

from __future__ import annotations

import cv2
import numpy as np


ZERO_COLOR = (255, 0, 200)      # 0 영역 채움 (마젠타 — 히트맵에 없는 색)
LINE_COLOR = (0, 0, 0)          # 0-Line 외곽선
CENTER_COLOR = (255, 255, 255)  # 중심선


def make_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    centerline: np.ndarray | None = None,
    alpha: float = 0.45,
    draw_contour: bool = True,
) -> np.ndarray:
    """원본 + 0-Line 오버레이 이미지를 만든다."""
    out = rgb.copy()
    m = mask > 0

    tint = np.zeros_like(out)
    tint[:] = ZERO_COLOR
    out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)

    if draw_contour:
        cnts, _ = cv2.findContours(
            (m.astype(np.uint8)), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, cnts, -1, LINE_COLOR, 2)

    if centerline is not None:
        out[centerline > 0] = CENTER_COLOR

    return out


def make_value_map(
    values: np.ndarray,
    part_mask: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """색에서 되읽은 편차값을 다시 히트맵으로 그린다.

    원본과 나란히 놓고 보면 색->값 변환이 맞는지 한눈에 검증된다.
    변환이 틀렸다면 이 이미지가 원본과 다른 모양이 된다.
    """
    norm = np.clip((values - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat[~part_mask] = 255
    return heat


def make_panel(
    rgb: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    titles: tuple = ("원본 스캔", "0-Line 오버레이", "0-Line 마스크"),
) -> np.ndarray:
    """원본 / 오버레이 / 마스크를 가로로 붙인 검증용 패널."""
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    panels = [rgb, overlay, mask_rgb]

    h = min(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        scale = h / p.shape[0]
        resized.append(cv2.resize(p, (int(p.shape[1] * scale), h)))

    gap = 12
    total_w = sum(p.shape[1] for p in resized) + gap * (len(resized) - 1)
    canvas = np.full((h, total_w, 3), 245, dtype=np.uint8)
    x = 0
    for p in resized:
        canvas[:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    return canvas


def draw_regions(
    rgb: np.ndarray,
    regions,
    top_n: int = 12,
    color: tuple = (0, 0, 255),
) -> np.ndarray:
    """면적 상위 영역에 번호와 외접 사각형을 그린다."""
    out = rgb.copy()
    for i, r in enumerate(regions[:top_n], start=1):
        cv2.rectangle(
            out, (r.bbox_x, r.bbox_y),
            (r.bbox_x + r.bbox_w, r.bbox_y + r.bbox_h), color, 2
        )
        cv2.putText(
            out, str(i), (r.bbox_x, max(r.bbox_y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
    return out


__all__ = ["make_overlay", "make_value_map", "make_panel", "draw_regions"]
