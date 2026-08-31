"""Decide where the views and their value labels sit on the sheet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import config


@dataclass(frozen=True)
class SheetPoint:
    """One value to print, positioned as a ratio of its view image."""

    point_id: str
    text: str
    x_ratio: float
    y_ratio: float


@dataclass
class SheetView:
    """One picture on the sheet with the points that belong to it."""

    image: np.ndarray
    points: list[SheetPoint] = field(default_factory=list)
    title: str = ""
    box: tuple[float, float, float, float] | None = None  # x, y, width, height


@dataclass(frozen=True)
class PlacedLabel:
    """A label box and the point its leader must reach, in sheet pixels."""

    text: str
    label_x: float
    label_y: float
    point_x: float
    point_y: float
    edge: str


def fit_box(image: np.ndarray, x: float, y: float,
            width: float, height: float) -> tuple[float, float, float, float]:
    """Fit the image inside the box, keeping its aspect and centring it."""
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("이미지 크기가 올바르지 않습니다.")
    scale = min(width / source_width, height / source_height)
    drawn_width = source_width * scale
    drawn_height = source_height * scale
    return (
        x + (width - drawn_width) / 2,
        y + (height - drawn_height) / 2,
        drawn_width,
        drawn_height,
    )


def default_layout(views: Sequence[SheetView]) -> None:
    """Place the front view on top and any detail views in a row below.

    Boxes already set by the caller are left alone, so a hand-tuned layout from
    the UI can be passed straight through.
    """
    if not views:
        return
    area_top = config.DRAWING_TOP + config.VIEW_MARGIN
    area_bottom = config.DRAWING_BOTTOM - config.VIEW_MARGIN
    area_left = config.VIEW_MARGIN
    area_right = config.SHEET_WIDTH - config.VIEW_MARGIN
    area_width = area_right - area_left
    area_height = area_bottom - area_top

    front, details = views[0], list(views[1:])
    front_height = area_height * (config.FRONT_HEIGHT_RATIO if details else 1.0)

    # 라벨이 이미지 바깥 여백에 놓이므로 그만큼 안쪽으로 들여 배치한다.
    inset_x = config.LABEL_GUTTER + config.LABEL_WIDTH
    inset_y = config.LABEL_GUTTER + config.LABEL_HEIGHT
    if front.box is None:
        front.box = fit_box(
            front.image,
            area_left + inset_x,
            area_top + inset_y,
            area_width - 2 * inset_x,
            front_height - 2 * inset_y,
        )

    if not details:
        return
    detail_top = area_top + front_height + config.DETAIL_GAP
    detail_height = area_bottom - detail_top - config.DETAIL_TITLE_HEIGHT
    slot_width = (area_width - config.DETAIL_GAP * (len(details) - 1)) / len(details)
    for index, view in enumerate(details):
        if view.box is not None:
            continue
        slot_x = area_left + index * (slot_width + config.DETAIL_GAP)
        view.box = fit_box(
            view.image,
            slot_x + config.LABEL_GUTTER + config.LABEL_WIDTH,
            detail_top + config.DETAIL_TITLE_HEIGHT,
            slot_width - 2 * (config.LABEL_GUTTER + config.LABEL_WIDTH),
            detail_height,
        )


def place_labels(view: SheetView) -> list[PlacedLabel]:
    """Push every label into the nearest margin and spread out collisions.

    The real sheets keep values off the part with a leader pointing in, which
    is what makes a crowded panel readable.
    """
    if view.box is None:
        raise ValueError("뷰 배치가 정해지지 않았습니다.")
    box_x, box_y, box_width, box_height = view.box

    entries = []
    for point in view.points:
        px = box_x + point.x_ratio * box_width
        py = box_y + point.y_ratio * box_height
        distances = (
            (px - box_x, "left"),
            (box_x + box_width - px, "right"),
            (py - box_y, "top"),
            (box_y + box_height - py, "bottom"),
        )
        entries.append({"point": point, "px": px, "py": py, "edge": min(distances)[1]})

    placed: list[PlacedLabel] = []
    for edge in ("top", "bottom", "left", "right"):
        group = [item for item in entries if item["edge"] == edge]
        horizontal = edge in ("top", "bottom")
        group.sort(key=lambda item: item["px"] if horizontal else item["py"])
        previous: float | None = None
        for item in group:
            if horizontal:
                label_x = item["px"] - config.LABEL_WIDTH / 2
                label_y = (
                    box_y - config.LABEL_GUTTER - config.LABEL_HEIGHT
                    if edge == "top"
                    else box_y + box_height + config.LABEL_GUTTER
                )
                if previous is not None:
                    label_x = max(
                        label_x, previous + config.LABEL_WIDTH + config.LABEL_MIN_GAP
                    )
                previous = label_x
            else:
                label_y = item["py"] - config.LABEL_HEIGHT / 2
                label_x = (
                    box_x - config.LABEL_GUTTER - config.LABEL_WIDTH
                    if edge == "left"
                    else box_x + box_width + config.LABEL_GUTTER
                )
                if previous is not None:
                    label_y = max(
                        label_y, previous + config.LABEL_HEIGHT + config.LABEL_MIN_GAP
                    )
                previous = label_y
            placed.append(
                PlacedLabel(
                    text=item["point"].text,
                    label_x=label_x,
                    label_y=label_y,
                    point_x=item["px"],
                    point_y=item["py"],
                    edge=edge,
                )
            )
    return placed


def crop_view(
    image: np.ndarray,
    points: Sequence[SheetPoint],
    region: tuple[float, float, float, float],
    title: str = "",
) -> SheetView:
    """Cut a detail view out of an image and re-express its points inside it.

    The region is given as ratios of the source image, the same form the UI
    uses for its detail boxes.
    """
    height, width = image.shape[:2]
    rx, ry, rw, rh = region
    x0 = int(round(max(0.0, min(1.0, rx)) * width))
    y0 = int(round(max(0.0, min(1.0, ry)) * height))
    x1 = int(round(max(0.0, min(1.0, rx + rw)) * width))
    y1 = int(round(max(0.0, min(1.0, ry + rh)) * height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("Detail 영역이 너무 작습니다.")

    inside = []
    for point in points:
        if rx <= point.x_ratio <= rx + rw and ry <= point.y_ratio <= ry + rh:
            inside.append(
                SheetPoint(
                    point_id=point.point_id,
                    text=point.text,
                    x_ratio=(point.x_ratio - rx) / rw,
                    y_ratio=(point.y_ratio - ry) / rh,
                )
            )
    return SheetView(image=image[y0:y1, x0:x1].copy(), points=inside, title=title)
