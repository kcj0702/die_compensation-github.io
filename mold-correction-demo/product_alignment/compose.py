"""Draw transferred measurement points onto the clean product-data image."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class SheetPoint:
    """One measurement point already expressed in product-image pixels."""

    point_id: str
    x: float
    y: float
    value: float | None = None
    label_color: str = "white"


def compose_scale(width: int) -> int:
    """Return the integer upscale that makes labels readable."""
    if width <= 0:
        return 1
    needed = -(-config.COMPOSE_MIN_WIDTH // width)  # ceil division
    return int(max(1, min(config.COMPOSE_MAX_SCALE, needed)))


def _format_value(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:+.1f}"


def _place_label(
    text_size: tuple[int, int],
    anchor: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
    canvas_size: tuple[int, int],
    offset: int,
) -> tuple[int, int, int, int]:
    """Pick a non-overlapping box near the anchor, preferring straight up."""
    text_width, text_height = text_size
    box_width = text_width + 10
    box_height = text_height + 8
    width, height = canvas_size
    anchor_x, anchor_y = anchor

    directions = (
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
    )
    for distance in (offset, offset * 2, offset * 3):
        for dx, dy in directions:
            x0 = int(anchor_x + dx * distance - box_width / 2)
            y0 = int(anchor_y + dy * distance - box_height / 2)
            x0 = int(np.clip(x0, 0, max(0, width - box_width)))
            y0 = int(np.clip(y0, 0, max(0, height - box_height)))
            box = (x0, y0, x0 + box_width, y0 + box_height)
            if not any(
                box[0] < other[2] and other[0] < box[2]
                and box[1] < other[3] and other[1] < box[3]
                for other in placed
            ):
                return box

    x0 = int(np.clip(anchor_x - box_width / 2, 0, max(0, width - box_width)))
    y0 = int(np.clip(anchor_y - offset - box_height / 2, 0, max(0, height - box_height)))
    return x0, y0, x0 + box_width, y0 + box_height


def render_points(
    product_image: np.ndarray,
    points: list[SheetPoint],
    *,
    scale: int | None = None,
    show_values: bool = True,
) -> np.ndarray:
    """Return the product image with every given point drawn on top.

    Values are drawn exactly as passed in. This step transfers coordinates and
    does not convert a deviation into a correction value.
    """
    factor = compose_scale(product_image.shape[1]) if scale is None else max(1, scale)
    canvas = cv2.resize(
        product_image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
    )
    height, width = canvas.shape[:2]
    radius = max(
        config.COMPOSE_MARKER_RADIUS_MIN,
        int(round(width * config.COMPOSE_MARKER_RADIUS_RATIO)),
    )
    offset = max(
        config.COMPOSE_LABEL_OFFSET_MIN,
        int(round(width * config.COMPOSE_LABEL_OFFSET_RATIO)),
    )
    font_scale = max(
        config.COMPOSE_FONT_SCALE_MIN, width * config.COMPOSE_FONT_SCALE_RATIO
    )
    font_thickness = max(1, int(round(width * config.COMPOSE_FONT_THICKNESS_RATIO)))
    placed: list[tuple[int, int, int, int]] = []

    ordered = sorted(points, key=lambda item: (item.y, item.x))
    for point in ordered:
        x = int(round(point.x * factor))
        y = int(round(point.y * factor))
        if not (0 <= x < width and 0 <= y < height):
            continue

        marker_color = (0, 0, 200) if point.label_color == "red" else (70, 70, 70)
        text = _format_value(point.value) if show_values else None
        if text is not None:
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            box = _place_label(
                (text_width, text_height), (x, y), placed, (width, height), offset
            )
            placed.append(box)
            box_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            cv2.line(canvas, box_center, (x, y), (60, 60, 60), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (60, 60, 60), 1)
            cv2.putText(
                canvas,
                text,
                (box[0] + 5, box[3] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (20, 20, 20),
                font_thickness,
                cv2.LINE_AA,
            )

        cv2.circle(canvas, (x, y), radius, marker_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def render_alignment_overlay(
    product_image: np.ndarray, warped_scan_mask: np.ndarray
) -> np.ndarray:
    """Draw the warped scan outline on the product image for confirmation.

    A filled two-colour overlay is misleading here: the scan mask closes small
    holes that the CAD render leaves open, so even a correct alignment paints
    large areas as mismatched. Outlines put the real question on screen --
    do the scanned edges and holes land on the product's edges and holes?
    """
    overlay = product_image.copy()
    height, width = overlay.shape[:2]
    # RETR_CCOMP keeps the interior hole boundaries, which are what actually
    # show a left-right flip on a panel with a symmetric outline.
    contours, _ = cv2.findContours(
        warped_scan_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    thickness = max(1, int(round(min(height, width) * 0.006)))
    cv2.drawContours(overlay, contours, -1, (0, 0, 220), thickness, cv2.LINE_AA)
    return overlay
