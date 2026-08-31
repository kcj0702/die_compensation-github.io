"""Estimate the transform that puts scan coordinates onto a product-data image.

Both inputs are axis-aligned orthographic renders of the same panel, so the
transform is a per-axis scale and translation plus one of four flips. The flip
is the only part that cannot be read off the bounding boxes, and on a symmetric
panel it cannot be decided from the masks at all -- that case is reported
instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from . import config
from .masks import (
    boundary_band,
    bounding_box,
    fill_silhouette,
    hole_mask,
    intersection_over_union,
)


@dataclass(frozen=True)
class OrientationScore:
    """How well one of the four flip candidates matches the product data."""

    flip_x: bool
    flip_y: bool
    outline_iou: float
    hole_iou: float
    band_iou: float

    @property
    def score(self) -> float:
        return (
            self.outline_iou
            + self.hole_iou
            + config.BOUNDARY_BAND_WEIGHT * self.band_iou
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "flipX": self.flip_x,
            "flipY": self.flip_y,
            "outlineIou": round(self.outline_iou, 4),
            "holeIou": round(self.hole_iou, 4),
            "bandIou": round(self.band_iou, 4),
            "score": round(self.score, 4),
        }


@dataclass
class Alignment:
    """A scan-to-product transform together with the evidence behind it."""

    matrix: tuple[float, float, float, float, float, float]
    flip_x: bool
    flip_y: bool
    outline_iou: float
    hole_iou: float
    band_iou: float
    margin: float
    scan_size: tuple[int, int]
    product_size: tuple[int, int]
    overridden: bool = False
    candidates: list[OrientationScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (
            self.outline_iou
            + self.hole_iou
            + config.BOUNDARY_BAND_WEIGHT * self.band_iou
        )

    @property
    def confident(self) -> bool:
        """True only when the shapes agree and one orientation clearly won.

        A person can confirm the orientation, but nobody can confirm away a
        shape mismatch: a confirmed flip replayed against the wrong product
        image must still report itself as untrustworthy.
        """
        if self.outline_iou < config.MIN_OUTLINE_IOU:
            return False
        return self.overridden or self.margin >= config.MIN_DECISION_MARGIN

    def as_array(self) -> np.ndarray:
        return np.array(self.matrix, dtype=np.float64).reshape(2, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": [round(value, 6) for value in self.matrix],
            "flipX": self.flip_x,
            "flipY": self.flip_y,
            "outlineIou": round(self.outline_iou, 4),
            "holeIou": round(self.hole_iou, 4),
            "bandIou": round(self.band_iou, 4),
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "confident": self.confident,
            "overridden": self.overridden,
            "scanSize": list(self.scan_size),
            "productSize": list(self.product_size),
            "candidates": [item.to_dict() for item in self.candidates],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Alignment":
        matrix = tuple(float(value) for value in payload["matrix"])
        if len(matrix) != 6:
            raise ValueError("저장된 정렬 행렬의 값 개수가 6개가 아닙니다.")
        return cls(
            matrix=matrix,  # type: ignore[arg-type]
            flip_x=bool(payload["flipX"]),
            flip_y=bool(payload["flipY"]),
            outline_iou=float(payload.get("outlineIou", 0.0)),
            hole_iou=float(payload.get("holeIou", 0.0)),
            band_iou=float(payload.get("bandIou", 0.0)),
            margin=float(payload.get("margin", 0.0)),
            scan_size=tuple(payload.get("scanSize", (0, 0))),  # type: ignore[arg-type]
            product_size=tuple(payload.get("productSize", (0, 0))),  # type: ignore[arg-type]
            overridden=bool(payload.get("overridden", False)),
            warnings=list(payload.get("warnings", [])),
        )


def _flip_matrix(
    scan_box: tuple[int, int, int, int],
    product_box: tuple[int, int, int, int],
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    """Map the scan bounding box onto the product bounding box.

    A box is treated as the continuous span from x0 - 0.5 to x0 + width - 0.5
    rather than as pixel indices. With indices the flipped branch lands one
    pixel off the unflipped one, which quietly penalises every flipped
    orientation and skews the decision margin.
    """
    sx, sy, sw, sh = scan_box
    px, py, pw, ph = product_box
    scale_x = pw / sw
    scale_y = ph / sh

    if flip_x:
        a, tx = -scale_x, px + pw - 0.5 + scale_x * (sx - 0.5)
    else:
        a, tx = scale_x, px - 0.5 + scale_x * (0.5 - sx)
    if flip_y:
        d, ty = -scale_y, py + ph - 0.5 + scale_y * (sy - 0.5)
    else:
        d, ty = scale_y, py - 0.5 + scale_y * (0.5 - sy)

    return np.array([[a, 0.0, tx], [0.0, d, ty]], dtype=np.float64)


def _adjusted(
    matrix: np.ndarray,
    product_box: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> np.ndarray:
    """Scale about the product box centre, then translate."""
    px, py, pw, ph = product_box
    centre_x = px + pw / 2.0
    centre_y = py + ph / 2.0
    a, _, tx = matrix[0]
    _, d, ty = matrix[1]
    return np.array(
        [
            [a * scale_x, 0.0, scale_x * (tx - centre_x) + centre_x + shift_x],
            [0.0, d * scale_y, scale_y * (ty - centre_y) + centre_y + shift_y],
        ],
        dtype=np.float64,
    )


def _overlap(
    matrix: np.ndarray,
    scan_solid: np.ndarray,
    scan_holes: np.ndarray,
    product_solid: np.ndarray,
    product_holes: np.ndarray,
) -> float:
    height, width = product_solid.shape[:2]
    warped_solid = cv2.warpAffine(scan_solid, matrix, (width, height))
    warped_holes = cv2.warpAffine(scan_holes, matrix, (width, height))
    return intersection_over_union(
        warped_solid, product_solid
    ) + intersection_over_union(warped_holes, product_holes)


def _refine(
    matrix: np.ndarray,
    product_box: tuple[int, int, int, int],
    scan_solid: np.ndarray,
    scan_holes: np.ndarray,
    product_solid: np.ndarray,
    product_holes: np.ndarray,
) -> np.ndarray:
    """Nudge scale and offset until the two masks overlap as much as possible.

    Matching bounding boxes inherits whatever each mask includes at its edge --
    the scan mask keeps a dilated anti-aliasing rim, the CAD render keeps its
    outline stroke -- and that difference shows up as a slight shrink that
    pulls every transferred point inwards. Optimising the overlap removes it.
    """
    params = [1.0, 1.0, 0.0, 0.0]
    best = _overlap(matrix, scan_solid, scan_holes, product_solid, product_holes)

    for scale_step, shift_step in config.REFINE_STEPS:
        steps = (scale_step, scale_step, shift_step, shift_step)
        for _ in range(config.REFINE_MAX_ROUNDS):
            improved = False
            for index in range(4):
                for sign in (1.0, -1.0):
                    trial = list(params)
                    trial[index] += sign * steps[index]
                    score = _overlap(
                        _adjusted(matrix, product_box, *trial),
                        scan_solid,
                        scan_holes,
                        product_solid,
                        product_holes,
                    )
                    if score > best + config.REFINE_MIN_GAIN:
                        best, params, improved = score, trial, True
            if not improved:
                break

    return _adjusted(matrix, product_box, *params)


def score_orientations(
    scan_mask: np.ndarray, product_mask: np.ndarray
) -> list[OrientationScore]:
    """Score all four flips of the scan mask against the product mask."""
    scan_solid = fill_silhouette(scan_mask)
    product_solid = fill_silhouette(product_mask)
    scan_holes = hole_mask(scan_mask, scan_solid)
    product_holes = hole_mask(product_mask, product_solid)

    scan_box = bounding_box(scan_solid)
    product_box = bounding_box(product_solid)
    height, width = product_mask.shape[:2]
    product_band = cv2.bitwise_or(
        boundary_band(product_solid), boundary_band(product_holes)
    )

    scores: list[OrientationScore] = []
    for flip_x, flip_y in config.FLIP_CANDIDATES:
        matrix = _flip_matrix(scan_box, product_box, flip_x, flip_y)
        warped_solid = cv2.warpAffine(scan_solid, matrix, (width, height))
        warped_holes = cv2.warpAffine(scan_holes, matrix, (width, height))
        warped_band = cv2.bitwise_or(
            boundary_band(warped_solid), boundary_band(warped_holes)
        )
        scores.append(
            OrientationScore(
                flip_x=flip_x,
                flip_y=flip_y,
                outline_iou=intersection_over_union(warped_solid, product_solid),
                hole_iou=intersection_over_union(warped_holes, product_holes),
                band_iou=intersection_over_union(warped_band, product_band),
            )
        )
    return scores


def estimate_alignment(
    scan_mask: np.ndarray,
    product_mask: np.ndarray,
    *,
    flip_x: bool | None = None,
    flip_y: bool | None = None,
) -> Alignment:
    """Return the best scan-to-product transform, or the requested orientation.

    Passing both flip_x and flip_y skips the automatic decision entirely; that
    is how a confirmed alignment is replayed for later scans of the same part
    number. Pinning only one axis constrains that axis and still decides the
    other automatically, so the ambiguity check keeps applying to it.
    """
    scores = score_orientations(scan_mask, product_mask)
    ranked = sorted(scores, key=lambda item: item.score, reverse=True)
    allowed = [
        item
        for item in ranked
        if (flip_x is None or item.flip_x == flip_x)
        and (flip_y is None or item.flip_y == flip_y)
    ]

    chosen = allowed[0]
    overridden = flip_x is not None and flip_y is not None
    # 비교는 아직 자동으로 정해야 하는 후보들 사이에서만 뜻이 있다. 두 축을 모두
    # 지정했으면 남은 후보가 하나뿐이라 비교할 대상이 없다.
    margin = allowed[0].score - allowed[1].score if len(allowed) > 1 else 0.0

    scan_solid = fill_silhouette(scan_mask)
    product_solid = fill_silhouette(product_mask)
    scan_holes = hole_mask(scan_mask, scan_solid)
    product_holes = hole_mask(product_mask, product_solid)
    scan_box = bounding_box(scan_solid)
    product_box = bounding_box(product_solid)

    matrix = _refine(
        _flip_matrix(scan_box, product_box, chosen.flip_x, chosen.flip_y),
        product_box,
        scan_solid,
        scan_holes,
        product_solid,
        product_holes,
    )

    # 보고하는 일치도는 방향 판정에 쓴 bbox 변환이 아니라 실제로 좌표를 옮길
    # 보정된 변환 기준이어야 한다.
    height, width = product_mask.shape[:2]
    warped_solid = cv2.warpAffine(scan_solid, matrix, (width, height))
    warped_holes = cv2.warpAffine(scan_holes, matrix, (width, height))
    outline_iou = intersection_over_union(warped_solid, product_solid)
    hole_iou = intersection_over_union(warped_holes, product_holes)
    band_iou = intersection_over_union(
        cv2.bitwise_or(boundary_band(warped_solid), boundary_band(warped_holes)),
        cv2.bitwise_or(boundary_band(product_solid), boundary_band(product_holes)),
    )

    warnings: list[str] = []
    if outline_iou < config.MIN_OUTLINE_IOU:
        warnings.append(
            f"스캔과 제품데이터의 외형 일치도가 낮습니다 (IoU {outline_iou:.3f}). "
            "같은 품번의 제품데이터가 맞는지 확인하세요."
        )
    if not overridden and margin < config.MIN_DECISION_MARGIN:
        warnings.append(
            f"1위와 2위 방향의 점수 차가 {margin:.3f}로 작아 방향을 자동으로 "
            "확정할 수 없습니다. 좌우·상하 반전을 직접 확인하세요."
        )

    scan_height, scan_width = scan_mask.shape[:2]
    product_height, product_width = product_mask.shape[:2]
    return Alignment(
        matrix=tuple(matrix.reshape(-1).tolist()),  # type: ignore[arg-type]
        flip_x=chosen.flip_x,
        flip_y=chosen.flip_y,
        outline_iou=outline_iou,
        hole_iou=hole_iou,
        band_iou=band_iou,
        margin=margin,
        scan_size=(scan_width, scan_height),
        product_size=(product_width, product_height),
        overridden=overridden,
        candidates=ranked,
        warnings=warnings,
    )


def map_point(alignment: Alignment, x: float, y: float) -> tuple[float, float]:
    """Map one scan pixel coordinate into the product-data image."""
    a, b, tx, c, d, ty = alignment.matrix
    return a * x + b * y + tx, c * x + d * y + ty


def map_points(
    alignment: Alignment, points: Iterable[Sequence[float]]
) -> list[tuple[float, float]]:
    """Map scan pixel coordinates into the product-data image."""
    return [map_point(alignment, point[0], point[1]) for point in points]


def is_inside(alignment: Alignment, x: float, y: float) -> bool:
    """Return True when a mapped coordinate falls inside the product image."""
    width, height = alignment.product_size
    return 0 <= x < width and 0 <= y < height


def warp_scan_mask(alignment: Alignment, scan_mask: np.ndarray) -> np.ndarray:
    """Warp a scan mask into the product frame, for a confirmation overlay."""
    width, height = alignment.product_size
    return cv2.warpAffine(scan_mask, alignment.as_array(), (width, height))
