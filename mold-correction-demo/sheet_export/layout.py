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


@dataclass(frozen=True)
class SheetAnnotation:
    """A free-form annotation the operator drew on the UI preview.

    Coordinates are ratios (0..1) of the containing view's image, so the
    same annotation shifts with the picture when the layout resizes it.
    ``kind`` mirrors the four UI tools: ``rect``, ``ellipse``, ``text``,
    ``arrow``. ``w``/``h`` may be negative for ``arrow`` to encode
    direction from (x, y) toward (x+w, y+h).
    """

    kind: str
    x_ratio: float
    y_ratio: float
    w_ratio: float
    h_ratio: float
    color: str = "#e8802f"
    text: str = ""
    font_size_px: float | None = None
    font_family: str | None = None


@dataclass
class SheetView:
    """One picture on the sheet with the points that belong to it.

    ``label_positions`` mirrors what the UI actually drew for each point on
    this specific view -- the label box top-left as a ratio of the picture
    (0..1 inside, negative or > 1 when the label sits outside in the sheet
    margin). Keyed by ``SheetPoint.point_id``. When present ``place_labels``
    honours these positions verbatim (so a manually dragged label survives
    the round trip to Excel); missing entries fall back to auto-placement.
    """

    image: np.ndarray
    points: list[SheetPoint] = field(default_factory=list)
    annotations: list[SheetAnnotation] = field(default_factory=list)
    title: str = ""
    box: tuple[float, float, float, float] | None = None  # x, y, width, height
    label_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Reference zero-curve polylines, each a list of (x_ratio, y_ratio) in
    # this view's own image (same 0..1 convention as SheetPoint). Exported as
    # its own vector shape rather than baked into the picture, so the sheet
    # keeps the part image and the zero-line as two independently selectable
    # objects.
    zero_lines: list[list[tuple[float, float]]] = field(default_factory=list)


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


def _resolve_collisions(natural: list[float], size: float, gap: float) -> list[float]:
    """Push overlapping labels apart without leaning the whole group one way.

    A forward-only pass (each label pushed past the previous one) anchors the
    whole chain on the *first* label: when the group is too crowded to fit
    without overlap, every later label gets shoved further and further past
    its own point, and the group's leaders all slant the same direction
    instead of the row looking centered on the points it labels -- a crowded
    edge visibly runs off the side of the sheet instead of fanning out
    around its points.

    Anchoring a second chain on the *last* label (pulling each one left of
    its right neighbour) gives the mirror-image layout -- now the first
    label is the one shoved away from its point. Averaging the two chains
    centers the unavoidable overlap instead of dumping it on either end.
    Averaging can reintroduce a hair of overlap between adjacent averaged
    positions (each was only guaranteed non-overlapping against its own
    anchor direction), so a final forward pass over the averaged values
    re-enforces the minimum gap -- by this point the values are already
    balanced, so that pass rarely needs to move anything.
    """
    if not natural:
        return []
    n = len(natural)
    forward = [natural[0]] + [0.0] * (n - 1)
    for i in range(1, n):
        forward[i] = max(natural[i], forward[i - 1] + size + gap)
    backward = [0.0] * (n - 1) + [natural[-1]]
    for i in range(n - 2, -1, -1):
        backward[i] = min(natural[i], backward[i + 1] - size - gap)
    centered = [(f + b) / 2 for f, b in zip(forward, backward)]
    resolved = [centered[0]]
    for value in centered[1:]:
        resolved.append(max(value, resolved[-1] + size + gap))
    return resolved


def place_labels(view: SheetView) -> list[PlacedLabel]:
    """Push every label into the nearest margin and spread out collisions.

    The real sheets keep values off the part with a leader pointing in, which
    is what makes a crowded panel readable.

    When a point carries ``label_x_ratio`` / ``label_y_ratio`` the UI has
    already decided (and possibly the operator has hand-dragged) where the
    label sits; that position is honoured verbatim so the exported sheet
    matches what the operator saw on screen. Only points without provided
    positions fall through to the automatic placement below.
    """
    if view.box is None:
        raise ValueError("뷰 배치가 정해지지 않았습니다.")
    box_x, box_y, box_width, box_height = view.box

    placed_from_ui: list[PlacedLabel] = []
    entries = []
    label_positions = getattr(view, "label_positions", {}) or {}
    for point in view.points:
        px = box_x + point.x_ratio * box_width
        py = box_y + point.y_ratio * box_height
        provided = label_positions.get(point.point_id)
        if provided is not None:
            lx_ratio, ly_ratio = provided
            label_x = box_x + lx_ratio * box_width
            label_y = box_y + ly_ratio * box_height
            # Which side of the point the label sits on decides which side of
            # the picture the leader enters -- Excel uses that to pin the
            # connector when the label is dragged.
            label_cx = label_x + config.LABEL_WIDTH / 2
            label_cy = label_y + config.LABEL_HEIGHT / 2
            if abs(label_cx - px) >= abs(label_cy - py):
                edge = "right" if label_cx > px else "left"
            else:
                edge = "bottom" if label_cy > py else "top"
            placed_from_ui.append(
                PlacedLabel(
                    text=point.text, label_x=label_x, label_y=label_y,
                    point_x=px, point_y=py, edge=edge,
                )
            )
            continue
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
        if horizontal:
            natural = [item["px"] - config.LABEL_WIDTH / 2 for item in group]
            resolved = _resolve_collisions(
                natural, config.LABEL_WIDTH, config.LABEL_MIN_GAP
            )
            fixed_y = (
                box_y - config.LABEL_GUTTER - config.LABEL_HEIGHT
                if edge == "top"
                else box_y + box_height + config.LABEL_GUTTER
            )
        else:
            natural = [item["py"] - config.LABEL_HEIGHT / 2 for item in group]
            resolved = _resolve_collisions(
                natural, config.LABEL_HEIGHT, config.LABEL_MIN_GAP
            )
            fixed_x = (
                box_x - config.LABEL_GUTTER - config.LABEL_WIDTH
                if edge == "left"
                else box_x + box_width + config.LABEL_GUTTER
            )
        for item, coord in zip(group, resolved):
            label_x = coord if horizontal else fixed_x
            label_y = fixed_y if horizontal else coord
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
    return placed_from_ui + placed


def _clip_segment_to_rect(
    x0: float, y0: float, x1: float, y1: float,
    rx: float, ry: float, rw: float, rh: float,
) -> tuple[float, float, float, float] | None:
    """Liang-Barsky clip of one segment to the rect; ``None`` if it misses."""
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0 - rx), (dx, rx + rw - x0),
        (-dy, y0 - ry), (dy, ry + rh - y0),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


def _clip_polyline_to_rect(
    points: Sequence[tuple[float, float]],
    rx: float, ry: float, rw: float, rh: float,
) -> list[list[tuple[float, float]]]:
    """Split a polyline into the pieces that fall inside the rect.

    A zero-line that only partly crosses a Detail region must not be
    silently dropped (missing) or snapped whole into the crop (drawn outside
    the picture) -- each crossing becomes its own shorter piece, the same
    way the picture itself is cropped to the region.
    """
    pieces: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        clipped = _clip_segment_to_rect(x0, y0, x1, y1, rx, ry, rw, rh)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(current)
            current = []
            continue
        cx0, cy0, cx1, cy1 = clipped
        if not current or current[-1] != (cx0, cy0):
            if len(current) >= 2:
                pieces.append(current)
            current = [(cx0, cy0)]
        current.append((cx1, cy1))
    if len(current) >= 2:
        pieces.append(current)
    return pieces


def crop_view(
    image: np.ndarray,
    points: Sequence[SheetPoint],
    region: tuple[float, float, float, float],
    title: str = "",
    zero_lines: Sequence[list[tuple[float, float]]] = (),
) -> SheetView:
    """Cut a detail view out of an image and re-express its points inside it.

    The region is given as ratios of the source image, the same form the UI
    uses for its detail boxes. ``zero_lines`` (also in that ratio space) are
    clipped to the region and re-expressed the same way -- a line that only
    partly crosses the crop is split rather than dropped or left un-cropped.
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

    cropped_lines: list[list[tuple[float, float]]] = []
    for polyline in zero_lines:
        if len(polyline) < 2:
            continue
        for piece in _clip_polyline_to_rect(polyline, rx, ry, rw, rh):
            cropped_lines.append([
                ((px - rx) / rw, (py - ry) / rh) for px, py in piece
            ])

    return SheetView(
        image=image[y0:y1, x0:x1].copy(), points=inside, title=title,
        zero_lines=cropped_lines,
    )
