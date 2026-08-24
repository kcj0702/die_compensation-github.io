"""컬러바 또는 기준 컬러맵의 BGR 색상을 편차값에 대응한다."""

from __future__ import annotations

import matplotlib
import numpy as np

if __package__:  # 패키지 import와 직접 스크립트 실행을 모두 지원한다.
    from . import config
else:  # pragma: no cover - 직접 스크립트 실행 경로
    import config


class ColorToValueLUT:
    """색상 표본과 mm 값이 일대일로 대응된 최근접 탐색 LUT."""

    def __init__(self, colors_bgr: np.ndarray, values_mm: np.ndarray):
        colors = np.asarray(colors_bgr)
        values = np.asarray(values_mm)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("컬러 LUT는 (N, 3) BGR 배열이어야 합니다.")
        if len(colors) == 0 or len(colors) != len(values):
            raise ValueError("컬러와 편차값 표본 수가 같고 1개 이상이어야 합니다.")
        self._colors = colors.astype(np.float32)
        self._values = values.astype(np.float32)

    def to_value(self, bgr_pixel: np.ndarray) -> float:
        """BGR 제곱거리로 가장 가까운 표본의 값을 반환한다."""
        dists = np.sum((self._colors - bgr_pixel.astype(np.float32)) ** 2, axis=1)
        return float(self._values[int(np.argmin(dists))])


def _from_roi(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    min_mm: float,
    max_mm: float,
) -> ColorToValueLUT:
    """ROI 중앙 띠의 중앙값을 샘플링해 실제 컬러바 LUT를 만든다."""
    x, y, w, h = roi
    image_height, image_width = bgr.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError(f"컬러바 ROI의 폭과 높이는 양수여야 합니다: {roi}")
    if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
        raise ValueError(
            f"컬러바 ROI가 이미지 범위를 벗어났습니다: {roi}, "
            f"image=({image_width}, {image_height})"
        )
    strip = bgr[y:y + h, x:x + w]
    if h >= w:
        # 세로 범례는 위에서 아래로 max → min이다.
        center = w // 2
        half_band = max(0, w // 6)
        band = strip[:, max(0, center - half_band):min(w, center + half_band + 1)]
        samples = np.rint(np.median(band, axis=1)).astype(np.uint8)
        values = np.linspace(max_mm, min_mm, len(samples))
    else:
        # 가로 범례는 왼쪽에서 오른쪽으로 min → max다.
        center = h // 2
        half_band = max(0, h // 6)
        band = strip[max(0, center - half_band):min(h, center + half_band + 1), :]
        samples = np.rint(np.median(band, axis=0)).astype(np.uint8)
        values = np.linspace(min_mm, max_mm, len(samples))
    return ColorToValueLUT(samples, values)


def _from_matplotlib_colormap(
    name: str,
    min_mm: float,
    max_mm: float,
    steps: int = 256,
) -> ColorToValueLUT:
    """Matplotlib 컬러맵을 균일하게 샘플링해 대체 LUT를 만든다."""
    cmap = matplotlib.colormaps[name].resampled(steps)
    rgba = (cmap(np.linspace(0, 1, steps)) * 255).astype(np.uint8)
    bgr = rgba[:, [2, 1, 0]]
    values = np.linspace(min_mm, max_mm, steps)
    return ColorToValueLUT(bgr, values)


def build_lut(bgr: np.ndarray) -> ColorToValueLUT:
    """설정된 ROI를 우선 사용하고, 없으면 기준 컬러맵 LUT를 반환한다."""
    if config.COLORBAR_ROI is not None:
        return _from_roi(
            bgr,
            config.COLORBAR_ROI,
            config.COLORBAR_MIN_MM,
            config.COLORBAR_MAX_MM,
        )
    return _from_matplotlib_colormap(
        config.FALLBACK_COLORMAP,
        config.COLORBAR_MIN_MM,
        config.COLORBAR_MAX_MM,
    )
