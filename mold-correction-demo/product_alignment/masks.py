"""Build the silhouette, hole, and boundary masks used to align two views."""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest connected foreground component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("부품으로 판단할 전경 영역을 찾지 못했습니다.")

    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == component, 255, 0).astype(np.uint8)


def build_product_mask(
    image: np.ndarray, foreground_threshold: int | None = None
) -> np.ndarray:
    """Return a mask containing only the panel in a product-data render."""
    threshold = (
        config.PRODUCT_FOREGROUND_THRESHOLD
        if foreground_threshold is None
        else foreground_threshold
    )
    distance_from_white = 255 - image.min(axis=2)
    foreground = np.where(distance_from_white >= threshold, 255, 0).astype(np.uint8)

    size = config.PRODUCT_CLOSE_KERNEL
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    return largest_component(closed)


def build_scan_mask(image: np.ndarray) -> np.ndarray:
    """Return the scanned-part mask using the existing label_removal rule.

    The deviation map also contains a colorbar, and keeping the largest dense
    component is what already excludes it everywhere else in this project.
    Importing lazily keeps this package usable without the label_removal
    dependency when the caller already holds a mask.
    """
    from label_removal.remove_labels import build_scan_mask as _build_scan_mask

    return _build_scan_mask(image)


def fill_silhouette(mask: np.ndarray) -> np.ndarray:
    """Return the outer silhouette with every interior hole filled in."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled


def hole_mask(mask: np.ndarray, filled: np.ndarray | None = None) -> np.ndarray:
    """Return only the interior holes of a part mask.

    Holes carry the left-right information the outer silhouette loses on a
    symmetric panel, so they are scored separately from the silhouette.
    """
    silhouette = fill_silhouette(mask) if filled is None else filled
    return cv2.bitwise_and(silhouette, cv2.bitwise_not(mask))


def boundary_band(mask: np.ndarray, width: int | None = None) -> np.ndarray:
    """Return a thin band that follows every edge of the mask.

    Small notches decide the orientation of an otherwise symmetric part but
    contribute almost nothing to a whole-area IoU. Restricting the comparison
    to the edges gives those features their own weight.
    """
    size = config.BOUNDARY_BAND_WIDTH if width is None else width
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
    return cv2.subtract(cv2.dilate(mask, kernel), cv2.erode(mask, kernel))


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the (x, y, width, height) box of the foreground."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("마스크에 전경 픽셀이 없습니다.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    """Return the IoU of two binary masks, or 0.0 when both are empty."""
    intersection = int(np.count_nonzero((first > 0) & (second > 0)))
    union = int(np.count_nonzero((first > 0) | (second > 0)))
    return intersection / union if union else 0.0
