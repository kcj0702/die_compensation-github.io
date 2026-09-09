"""Data structures owned by the zero-line selection package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class ZeroLineRegion:
    """One connected region classified as zero-line material."""

    region_id: int
    area_px: int
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    perimeter_px: float
    mean_value: float
    unit: Literal["normalized", "mm"] = "normalized"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ZeroLineResult:
    """Complete result of the base zero-line detector."""

    source_image: str
    image_width: int
    image_height: int
    regions: list[ZeroLineRegion] = field(default_factory=list)
    total_zero_px: int = 0
    part_px: int = 0
    zero_ratio: float = 0.0
    tolerance: float = 0.0
    tolerance_unit: Literal["normalized", "mm"] = "normalized"
    colorbar: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["regions"] = [region.to_dict() for region in self.regions]
        return result


@dataclass
class ColorbarInfo:
    """Location and scale metadata for a detected colour bar."""

    side: Literal["left", "right"]
    x0: int
    x1: int
    y0: int
    y1: int
    n_samples: int
    vmin_at: Literal["top", "bottom"]
    vmin: float | None = None
    vmax: float | None = None
    symmetric: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ColorbarInfo", "ZeroLineRegion", "ZeroLineResult"]
