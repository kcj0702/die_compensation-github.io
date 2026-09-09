"""검출된 편차 포인트 중 형상의 변화를 대표하는 포인트를 선별한다."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import config


@dataclass(frozen=True)
class KeyPoint:
    index: int
    point_id: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.point_id, "reasons": list(self.reasons)}


@dataclass
class Selection:
    keys: list[KeyPoint] = field(default_factory=list)
    total: int = 0

    @property
    def ids(self) -> list[str]:
        return [key.point_id for key in self.keys]

    def count(self, reason: str) -> int:
        return sum(1 for key in self.keys if reason in key.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {"ids": self.ids, "total": self.total, "selected": len(self.keys),
                "peaks": self.count("peak"), "signChanges": self.count("sign_change"),
                "extremes": self.count("extreme"),
                "points": [key.to_dict() for key in self.keys]}


def _coordinate(point: Any, axis: str) -> float:
    key = "xPx" if axis == "x" else "yPx"
    return float(point[key] if isinstance(point, dict) else getattr(point, key))


def _value(point: Any) -> float:
    return float(point["value"] if isinstance(point, dict) else point.value)


def _identifier(point: Any, index: int) -> str:
    return str(point.get("id", index) if isinstance(point, dict) else getattr(point, "id", index))


def _nearest(coordinates: list[tuple[float, float]], count: int) -> list[list[int]]:
    return [sorted((other for other in range(len(coordinates)) if other != index),
                   key=lambda other: math.dist(origin, coordinates[other]))[:count]
            for index, origin in enumerate(coordinates)]


def select_key_points(points: Sequence[Any], *, peak_neighbours: int | None = None,
                      peak_min_abs: float | None = None,
                      sign_neighbours: int | None = None,
                      sign_min_abs: float | None = None,
                      sign_merge_radius: float | None = None,
                      keep_extremes: bool | None = None) -> Selection:
    """국소 극값, 유효한 부호 변화, 전체 최대·최소 포인트를 반환한다."""
    peak_neighbours = peak_neighbours or config.PEAK_NEIGHBOURS
    sign_neighbours = sign_neighbours or config.SIGN_NEIGHBOURS
    peak_min_abs = config.PEAK_MIN_ABS_MM if peak_min_abs is None else peak_min_abs
    sign_min_abs = config.SIGN_MIN_ABS_MM if sign_min_abs is None else sign_min_abs
    keep_extremes = config.KEEP_GLOBAL_EXTREMES if keep_extremes is None else keep_extremes
    if not points:
        return Selection(total=0)

    coordinates = [(_coordinate(point, "x"), _coordinate(point, "y")) for point in points]
    values = [_value(point) for point in points]
    if sign_merge_radius is None:
        span_x = max(x for x, _ in coordinates) - min(x for x, _ in coordinates)
        span_y = max(y for _, y in coordinates) - min(y for _, y in coordinates)
        sign_merge_radius = math.hypot(span_x, span_y) * config.SIGN_MERGE_RADIUS_RATIO

    reasons: dict[int, list[str]] = {}
    peak_neighbourhood = _nearest(coordinates, peak_neighbours)
    for index, value in enumerate(values):
        neighbours = [values[other] for other in peak_neighbourhood[index]]
        if neighbours and abs(value) >= peak_min_abs and (
                all(value > other for other in neighbours)
                or all(value < other for other in neighbours)):
            reasons.setdefault(index, []).append("peak")

    sign_neighbourhood = _nearest(coordinates, sign_neighbours)
    candidates = [index for index, value in enumerate(values)
                  if any(value * values[other] < 0
                         and min(abs(value), abs(values[other])) >= sign_min_abs
                         for other in sign_neighbourhood[index])]
    kept: list[int] = []
    for index in sorted(candidates, key=lambda item: -abs(values[item])):
        if all(math.dist(coordinates[index], coordinates[other]) > sign_merge_radius
               for other in kept):
            kept.append(index)
    for index in kept:
        reasons.setdefault(index, []).append("sign_change")

    if keep_extremes:
        for index in (values.index(max(values)), values.index(min(values))):
            reasons.setdefault(index, []).append("extreme")

    keys = [KeyPoint(index, _identifier(points[index], index), tuple(reasons[index]))
            for index in sorted(reasons)]
    return Selection(keys=keys, total=len(points))
