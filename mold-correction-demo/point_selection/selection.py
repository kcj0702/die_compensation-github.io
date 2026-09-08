"""Pick the measurement points that belong on a correction sheet.

An engineer marks a handful of points, not every label the scanner printed.
The two that carry the springback story are the local extremes of deviation and
the places where its sign flips, so those are what this selects. Everything
else stays available; this only says what to show by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import config


@dataclass(frozen=True)
class KeyPoint:
    """One selected point and why it was kept."""

    index: int
    point_id: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.point_id, "reasons": list(self.reasons)}


@dataclass
class Selection:
    """The chosen subset together with the counts behind it."""

    keys: list[KeyPoint] = field(default_factory=list)
    total: int = 0

    @property
    def ids(self) -> list[str]:
        return [key.point_id for key in self.keys]

    def count(self, reason: str) -> int:
        return sum(1 for key in self.keys if reason in key.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": self.ids,
            "total": self.total,
            "selected": len(self.keys),
            "peaks": self.count("peak"),
            "signChanges": self.count("sign_change"),
            "extremes": self.count("extreme"),
            "points": [key.to_dict() for key in self.keys],
        }


def _coordinate(point: Any, axis: str) -> float:
    """Read a pixel coordinate from either an object or a mapping."""
    key = "xPx" if axis == "x" else "yPx"
    if isinstance(point, dict):
        return float(point[key])
    return float(getattr(point, key))


def _value(point: Any) -> float:
    if isinstance(point, dict):
        return float(point["value"])
    return float(point.value)


def _identifier(point: Any, index: int) -> str:
    if isinstance(point, dict):
        return str(point.get("id", index))
    return str(getattr(point, "id", index))


def _nearest(
    coordinates: list[tuple[float, float]], count: int
) -> list[list[int]]:
    """Return the indices of the nearest `count` points for each point."""
    result: list[list[int]] = []
    for index, origin in enumerate(coordinates):
        order = sorted(
            (other for other in range(len(coordinates)) if other != index),
            key=lambda other: math.dist(origin, coordinates[other]),
        )
        result.append(order[:count])
    return result


def select_key_points(
    points: Sequence[Any],
    *,
    peak_neighbours: int | None = None,
    peak_min_abs: float | None = None,
    sign_neighbours: int | None = None,
    sign_min_abs: float | None = None,
    sign_merge_radius: float | None = None,
    keep_extremes: bool | None = None,
) -> Selection:
    """Return the points worth putting on the sheet, with a reason for each.

    Points may be dicts (the UI backend's shape) or objects exposing xPx, yPx
    and value. Distances use scan pixels so the neighbourhood is isotropic.
    """
    peak_neighbours = peak_neighbours or config.PEAK_NEIGHBOURS
    sign_neighbours = sign_neighbours or config.SIGN_NEIGHBOURS
    peak_min_abs = (
        config.PEAK_MIN_ABS_MM if peak_min_abs is None else peak_min_abs
    )
    sign_min_abs = (
        config.SIGN_MIN_ABS_MM if sign_min_abs is None else sign_min_abs
    )
    keep_extremes = (
        config.KEEP_GLOBAL_EXTREMES if keep_extremes is None else keep_extremes
    )

    if not points:
        return Selection(total=0)

    coordinates = [
        (_coordinate(point, "x"), _coordinate(point, "y")) for point in points
    ]
    values = [_value(point) for point in points]

    if sign_merge_radius is None:
        span_x = max(x for x, _ in coordinates) - min(x for x, _ in coordinates)
        span_y = max(y for _, y in coordinates) - min(y for _, y in coordinates)
        diagonal = math.hypot(span_x, span_y)
        sign_merge_radius = diagonal * config.SIGN_MERGE_RADIUS_RATIO

    reasons: dict[int, list[str]] = {}

    # 피크: 이웃들 사이에서 signed 값이 국소 극값인 지점.
    peak_neighbourhood = _nearest(coordinates, peak_neighbours)
    for index, value in enumerate(values):
        neighbours = [values[other] for other in peak_neighbourhood[index]]
        if not neighbours or abs(value) < peak_min_abs:
            continue
        if all(value > other for other in neighbours) or all(
            value < other for other in neighbours
        ):
            reasons.setdefault(index, []).append("peak")

    # 부호 변화: 양쪽 모두 유의미한 크기일 때만 인정하고, 제로 크로싱은 선이라
    # 가까운 후보를 묶어 대표점 하나만 남긴다.
    sign_neighbourhood = _nearest(coordinates, sign_neighbours)
    candidates: list[int] = []
    for index, value in enumerate(values):
        for other in sign_neighbourhood[index]:
            neighbour = values[other]
            if value * neighbour < 0 and min(abs(value), abs(neighbour)) >= sign_min_abs:
                candidates.append(index)
                break

    kept: list[int] = []
    for index in sorted(candidates, key=lambda item: -abs(values[item])):
        if all(
            math.dist(coordinates[index], coordinates[other]) > sign_merge_radius
            for other in kept
        ):
            kept.append(index)
    for index in kept:
        reasons.setdefault(index, []).append("sign_change")

    if keep_extremes:
        for index in (values.index(max(values)), values.index(min(values))):
            reasons.setdefault(index, []).append("extreme")

    keys = [
        KeyPoint(
            index=index,
            point_id=_identifier(points[index], index),
            reasons=tuple(reasons[index]),
        )
        for index in sorted(reasons)
    ]
    return Selection(keys=keys, total=len(points))
