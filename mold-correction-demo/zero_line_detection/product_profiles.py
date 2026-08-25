"""Product-specific zero-line profiles used on correction sheets.

The profiles originate from the approved JD64/JD67/JD71 lab drawings.  Their
coordinates are normalized so that a standard scan can be displayed at any
resolution without changing the intended geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


RED = (235, 55, 55)  # RGB
FILL_RED = (255, 45, 45)  # RGB


@dataclass(frozen=True)
class ProductProfile:
    name: str
    open_lines: tuple[tuple[tuple[float, float], ...], ...] = ()
    closed_loops: tuple[tuple[tuple[float, float], ...], ...] = ()


JD64 = ProductProfile(
    name="JD_64XX2",
    open_lines=(((0.26513410, 0.25478645), (0.34712644, 0.70986745),
                 (0.70344828, 0.70986745), (0.70498084, 0.25478645)),),
)

# Rectangles from jd67_zero_areas.json, normalized from its 1688 x 1016 canvas.
JD67 = ProductProfile(
    name="JD_67XX6",
    closed_loops=(
        ((494 / 1687, 199 / 1015), (563 / 1687, 199 / 1015), (563 / 1687, 231 / 1015), (494 / 1687, 231 / 1015)),
        ((989 / 1687, 199 / 1015), (1058 / 1687, 199 / 1015), (1058 / 1687, 233 / 1015), (989 / 1687, 233 / 1015)),
        ((1286 / 1687, 199 / 1015), (1355 / 1687, 199 / 1015), (1355 / 1687, 231 / 1015), (1286 / 1687, 231 / 1015)),
        ((1403 / 1687, 412 / 1015), (1455 / 1687, 412 / 1015), (1455 / 1687, 584 / 1015), (1403 / 1687, 584 / 1015)),
        ((315 / 1687, 774 / 1015), (413 / 1687, 774 / 1015), (413 / 1687, 852 / 1015), (315 / 1687, 852 / 1015)),
        ((924 / 1687, 781 / 1015), (1129 / 1687, 781 / 1015), (1129 / 1687, 814 / 1015), (924 / 1687, 814 / 1015)),
    ),
)

JD71 = ProductProfile(
    name="JD_71XX2",
    open_lines=(
        ((129.90909 / 1271, 412.18182 / 767), (590 / 1271, 321 / 767)),
        ((129.90909 / 1271, 412.18182 / 767), (342.75 / 1271, 508.75 / 767)),
        ((853.33333 / 1271, 370.33333 / 767), (849 / 1271, 418 / 767),
         (1031 / 1271, 449 / 767), (1112 / 1271, 449 / 767)),
    ),
)

PROFILES = (JD64, JD67, JD71)


def profile_for_filename(filename: str) -> ProductProfile | None:
    folded = filename.upper().replace("-", "_")
    compact = folded.replace("_", "")
    return next(
        (
            profile for profile in PROFILES
            if profile.name in folded or profile.name.replace("_", "") in compact
        ),
        None,
    )


def _pixels(points: tuple[tuple[float, float], ...], width: int, height: int) -> np.ndarray:
    return np.asarray(
        [[round(x * (width - 1)), round(y * (height - 1))] for x, y in points],
        dtype=np.int32,
    )


def _draw_dashed_polyline(canvas: np.ndarray, points: np.ndarray, closed: bool) -> None:
    """Draw a 3 px red dashed polyline, preserving visual dash length on scans."""
    sequence = np.vstack((points, points[:1])) if closed else points
    dash, gap = 10.0, 7.0
    for start, end in zip(sequence[:-1], sequence[1:]):
        delta = end.astype(np.float64) - start
        length = float(np.hypot(*delta))
        if length == 0:
            continue
        direction = delta / length
        distance = 0.0
        while distance < length:
            stop = min(distance + dash, length)
            a = tuple(np.rint(start + direction * distance).astype(int))
            b = tuple(np.rint(start + direction * stop).astype(int))
            cv2.line(canvas, a, b, RED, 3, cv2.LINE_AA)
            distance += dash + gap


def draw_product_profile(rgb: np.ndarray, filename: str, fill_alpha: float = 0.20) -> tuple[np.ndarray, list[dict]] | None:
    """Overlay the approved product profile, or return ``None`` for other parts."""
    profile = profile_for_filename(filename)
    if profile is None:
        return None
    height, width = rgb.shape[:2]
    result = rgb.copy()
    lines: list[dict] = []

    for index, normalized in enumerate(profile.closed_loops, start=1):
        points = _pixels(normalized, width, height)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        tint = np.empty_like(result)
        tint[:] = FILL_RED
        inside = mask > 0
        result[inside] = (result[inside] * (1.0 - fill_alpha) + tint[inside] * fill_alpha).astype(np.uint8)
        _draw_dashed_polyline(result, points, closed=True)
        lines.append({"line_id": index, "points": points.tolist(), "is_closed": True})

    offset = len(lines)
    for index, normalized in enumerate(profile.open_lines, start=1):
        points = _pixels(normalized, width, height)
        _draw_dashed_polyline(result, points, closed=False)
        lines.append({"line_id": offset + index, "points": points.tolist(), "is_closed": False})
    return result, lines


__all__ = ["draw_product_profile", "profile_for_filename"]
