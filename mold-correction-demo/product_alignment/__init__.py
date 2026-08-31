"""3D 스캔의 측정점을 깨끗한 제품데이터 이미지 위로 옮기는 패키지."""

from .alignment import (
    Alignment,
    OrientationScore,
    estimate_alignment,
    is_inside,
    map_point,
    map_points,
    score_orientations,
    warp_scan_mask,
)
from .compose import SheetPoint, render_alignment_overlay, render_points
from .masks import build_product_mask, build_scan_mask
from .registry import (
    AlignmentStore,
    ProductLibrary,
    ProductMatch,
    part_number_from_name,
)

__all__ = [
    "Alignment",
    "AlignmentStore",
    "OrientationScore",
    "ProductLibrary",
    "ProductMatch",
    "SheetPoint",
    "build_product_mask",
    "build_scan_mask",
    "estimate_alignment",
    "is_inside",
    "map_point",
    "map_points",
    "part_number_from_name",
    "render_alignment_overlay",
    "render_points",
    "score_orientations",
    "warp_scan_mask",
]
