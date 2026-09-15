"""CATIA 관리면을 스캔 PNG 좌표계로 옮기고 목표 보정량을 계산한다.

관리면 트리의 업무 규칙은 다음과 같다.

* 상위 Geometrical Set 이름의 수치가 관리 목표값이다.
* ``SPEC SURFACE`` 는 원래 곡면이다.
* ``방향 적용면`` 은 실제 오프셋 결과면이다.
* ``BOUNDARY`` 는 관리치가 적용되는 외곽선이다.

현업 CATPart에는 적용면 이름의 부호와 실제 형상 방향이 다른 사례가 있다.
따라서 이 모듈은 적용면 이름의 수치를 절대로 목표값으로 사용하지 않는다.

정합을 새로 구현하지도 않는다. CAD 뷰어와 같은
``cad_import.overlay.ViewFit`` 및 ``to_pixels`` 를 받아 사용한다. 이 덕분에
화면에서 자동/수동으로 맞춘 자세와 관리면 오버레이가 서로 어긋나지 않는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from . import overlay as cad_overlay


_GROUP_VALUE = re.compile(
    r"[\"']?\s*([+-]?\s*\d+(?:\.\d+)?)\s*[\"']?\s*"
    r"(?:관리\s*기준면|기준\s*관리면)",
    re.IGNORECASE,
)
_SPEC = re.compile(r"^\s*SPEC\s*SURFACE(?:\s*[-_]\s*(.+?))?\s*$", re.IGNORECASE)
_APPLIED = re.compile(r"^\s*([^)]*?)\s*\)\s*.*방향\s*적용면\s*$", re.IGNORECASE)
_BOUNDARY = re.compile(r"^\s*BOUNDARY(?:\s*[-_].*)?\s*$", re.IGNORECASE)


@dataclass
class ManagementSurface:
    """한 원면과 그에 대응하는 Offset 적용면."""

    surface_id: str
    spec_name: str
    applied_name: str
    spec_object: Any = field(default=None, repr=False)
    applied_object: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surfaceId": self.surface_id,
            "specSurface": self.spec_name,
            "appliedSurface": self.applied_name,
        }


@dataclass
class ManagementGroup:
    """상위 Geometrical Set 하나에서 읽은 관리치 정의."""

    name: str
    target_mm: float
    surfaces: list[ManagementSurface]
    boundary_names: list[str]
    group_object: Any = field(default=None, repr=False)
    boundary_objects: list[Any] = field(default_factory=list, repr=False)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "targetMm": self.target_mm,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "boundaries": list(self.boundary_names),
            "warnings": list(self.warnings),
        }


@dataclass
class ManagementResult:
    """스캔 PNG 한 장에서 계산한 관리영역 결과."""

    group_name: str
    target_mm: float
    measured_mm: float | None
    delta_mm: float | None
    correction_mm: float | None
    visible_ratio: float
    status: str
    pixel_count: int
    polygon_px: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "groupName": payload["group_name"],
            "targetMm": payload["target_mm"],
            "measuredMm": payload["measured_mm"],
            "deltaMm": payload["delta_mm"],
            "correctionMm": payload["correction_mm"],
            "visibleRatio": payload["visible_ratio"],
            "status": payload["status"],
            "pixelCount": payload["pixel_count"],
            "polygonPx": payload["polygon_px"],
        }


def parse_management_value(name: str) -> float | None:
    """상위 그룹명에서만 관리 목표값(mm)을 읽는다."""
    match = _GROUP_VALUE.search(str(name))
    if not match:
        return None
    return float(match.group(1).replace(" ", ""))


def _normalise_id(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", "", value).upper()


def _iter_com(collection: Any) -> Iterable[Any]:
    """CATIA의 1-based collection과 일반 Python iterable을 함께 순회한다."""
    if collection is None:
        return
    count = getattr(collection, "Count", None)
    item = getattr(collection, "Item", None)
    if count is not None and callable(item):
        for index in range(1, int(count) + 1):
            yield item(index)
        return
    yield from collection


def _children(group: Any) -> list[Any]:
    """그룹 바로 아래의 HybridShape만 읽는다.

    관리 그룹 내부의 중첩 Geometrical Set은 별도 그룹으로 재귀 탐색한다.
    같은 객체를 두 번 읽지 않도록 여기서는 HybridShapes만 반환한다.
    """
    return list(_iter_com(getattr(group, "HybridShapes", None)))


def _pair_surfaces(
    children: Sequence[Any], boundary_groups: Sequence[Any] = (),
) -> tuple[list[ManagementSurface], list[Any], list[str]]:
    specs: list[tuple[str, str, Any]] = []
    applied: list[tuple[str, str, Any]] = []
    boundaries: list[Any] = [
        group for group in boundary_groups
        if _BOUNDARY.match(str(getattr(group, "Name", "")))
    ]
    warnings: list[str] = []

    for child in children:
        name = str(getattr(child, "Name", ""))
        if _BOUNDARY.match(name):
            boundaries.append(child)
            continue
        match = _SPEC.match(name)
        if match:
            specs.append((_normalise_id(match.group(1)), name, child))
            continue
        match = _APPLIED.match(name)
        if match:
            # 숫자와 부호는 의도적으로 버린다. ')' 앞의 A/B/01만 식별자다.
            applied.append((_normalise_id(match.group(1)), name, child))

    pairs: list[ManagementSurface] = []
    used: set[int] = set()
    for spec_index, (spec_id, spec_name, spec_object) in enumerate(specs):
        found = None
        if spec_id:
            found = next(
                (index for index, (candidate, _name, _obj) in enumerate(applied)
                 if index not in used and candidate == spec_id),
                None,
            )
        if found is None and len(specs) == len(applied):
            # 01처럼 표기 방식이 다르거나 ID가 없는 경우에는 CATIA 트리 순서를
            # 사용한다. 이 선택은 경고로 남겨 현업 파일 규칙 변경을 드러낸다.
            found = next((index for index in range(len(applied)) if index not in used), None)
            if found is not None and spec_id != applied[found][0]:
                warnings.append(f"{spec_name}: 적용면을 트리 순서로 연결했습니다.")
        if found is None:
            warnings.append(f"{spec_name}: 대응하는 방향 적용면이 없습니다.")
            continue
        used.add(found)
        applied_id, applied_name, applied_object = applied[found]
        pairs.append(ManagementSurface(
            surface_id=spec_id or applied_id or str(spec_index + 1),
            spec_name=spec_name,
            applied_name=applied_name,
            spec_object=spec_object,
            applied_object=applied_object,
        ))

    for index, (_surface_id, name, _obj) in enumerate(applied):
        if index not in used:
            warnings.append(f"{name}: 대응하는 SPEC SURFACE가 없습니다.")
    if not boundaries:
        warnings.append("BOUNDARY가 없습니다.")
    return pairs, boundaries, warnings


def discover_management_groups(part: Any) -> list[ManagementGroup]:
    """CATIA ``Part``의 Geometrical Set 트리에서 관리 그룹을 찾는다."""
    found: list[ManagementGroup] = []

    def visit(groups: Any) -> None:
        for group in _iter_com(groups):
            name = str(getattr(group, "Name", ""))
            target = parse_management_value(name)
            if target is not None:
                nested = list(_iter_com(getattr(group, "HybridBodies", None)))
                pairs, boundaries, warnings = _pair_surfaces(_children(group), nested)
                if not pairs:
                    warnings.append("SPEC SURFACE와 방향 적용면 쌍이 없습니다.")
                found.append(ManagementGroup(
                    name=name,
                    target_mm=target,
                    surfaces=pairs,
                    boundary_names=[str(getattr(item, "Name", "")) for item in boundaries],
                    group_object=group,
                    boundary_objects=boundaries,
                    warnings=warnings,
                ))
            visit(getattr(group, "HybridBodies", None))

    visit(getattr(part, "HybridBodies", None))
    return found


def fit_with_cad_viewer(
    vertices: np.ndarray,
    faces: np.ndarray,
    scan_part_mask: np.ndarray,
    *,
    mesh: Any = None,
    polish: bool = True,
) -> cad_overlay.ViewFit:
    """CAD 뷰어와 동일한 정합 엔진으로 자세를 구하고 명중률로 다듬는다."""
    fit = cad_overlay.fit_view(vertices, faces, scan_part_mask)
    if not polish:
        return fit
    if mesh is None:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return cad_overlay.polish_by_hit_rate(
        fit, vertices, faces, scan_part_mask, mesh,
    )


def project_boundary(
    boundary_points_cad: Sequence[Sequence[float]],
    fit: cad_overlay.ViewFit,
    image_shape: Sequence[int],
    *,
    cad_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, list[list[float]], float]:
    """CATIA BOUNDARY의 3D 점을 스캔 PNG의 채워진 마스크로 투영한다.

    ``cad_offset``은 CAD 뷰어가 원점 이동 때 사용한 값이다. 원본 CATIA
    좌표를 뷰어 정합 좌표로 바꾸기 위해 투영 전에 뺀다.

    반환하는 visible_ratio는 이미지 안에 들어온 경계점 비율이다. 이것은
    화면 밖 여부만 판정하며, 다른 CAD 면에 가려지는지까지 보장하지 않는다.
    """
    points = np.asarray(boundary_points_cad, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("BOUNDARY에는 3개 이상의 3D 점이 필요합니다.")
    points = points - np.asarray(cad_offset, dtype=float).reshape(1, 3)
    xs, ys = cad_overlay.to_pixels(points, fit)
    projected = np.stack([xs, ys], axis=1)
    height, width = int(image_shape[0]), int(image_shape[1])
    finite = np.isfinite(projected).all(axis=1)
    inside = (finite & (projected[:, 0] >= 0) & (projected[:, 0] < width)
              & (projected[:, 1] >= 0) & (projected[:, 1] < height))
    visible_ratio = float(inside.mean()) if len(inside) else 0.0

    mask = np.zeros((height, width), np.uint8)
    polygon = projected[finite]
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    return mask, projected.tolist(), visible_ratio


def calculate_management_result(
    group: ManagementGroup,
    deviation_values: np.ndarray,
    boundary_mask: np.ndarray,
    *,
    polygon_px: Sequence[Sequence[float]] = (),
    visible_ratio: float = 1.0,
    part_mask: np.ndarray | None = None,
    machining_sign: float = 1.0,
    minimum_visible_ratio: float = 0.70,
    minimum_pixels: int = 20,
) -> ManagementResult:
    """관리영역의 대표 편차에서 관리치를 뺀 수정 편차를 계산한다.

    대표값은 라벨/OCR 잔상과 경계 오차에 강한 중앙값을 사용한다.
    ``delta_mm``은 ``측정 편차 - 관리치``다. ``machining_sign``은 회사의
    절삭/용접 부호 규칙이며 기본값은 수정 편차를 그대로 반환한다.
    """
    values = np.asarray(deviation_values, dtype=float)
    mask = np.asarray(boundary_mask) > 0
    if values.shape != mask.shape:
        raise ValueError("편차 배열과 BOUNDARY 마스크 크기가 다릅니다.")
    if part_mask is not None:
        part = np.asarray(part_mask) > 0
        if part.shape != mask.shape:
            raise ValueError("부품 마스크와 BOUNDARY 마스크 크기가 다릅니다.")
        mask &= part
    mask &= np.isfinite(values)
    selected = values[mask]

    status = "VISIBLE"
    if visible_ratio < minimum_visible_ratio:
        status = "HIDDEN" if visible_ratio <= 0.05 else "PARTIAL"
    if len(selected) < minimum_pixels:
        status = "HIDDEN" if not len(selected) else "PARTIAL"

    measured = float(np.median(selected)) if len(selected) else None
    delta = measured - group.target_mm if measured is not None else None
    correction = machining_sign * delta if delta is not None else None
    return ManagementResult(
        group_name=group.name,
        target_mm=group.target_mm,
        measured_mm=round(measured, 4) if measured is not None else None,
        delta_mm=round(delta, 4) if delta is not None else None,
        correction_mm=round(correction, 4) if correction is not None else None,
        visible_ratio=round(float(visible_ratio), 4),
        status=status,
        pixel_count=int(len(selected)),
        polygon_px=[[round(float(x), 2), round(float(y), 2)] for x, y in polygon_px],
    )


def apply_management_value(
    deviation_values: np.ndarray,
    group: ManagementGroup,
    boundary_mask: np.ndarray,
    *,
    part_mask: np.ndarray | None = None,
) -> np.ndarray:
    """BOUNDARY 내부 편차를 ``편차 - 상위 관리치``로 수정한다.

    입력 배열은 변경하지 않고 복사본을 반환한다. NaN 같은 미측정 픽셀과
    부품 밖 픽셀은 그대로 둔다.
    """
    corrected = np.asarray(deviation_values, dtype=float).copy()
    mask = np.asarray(boundary_mask) > 0
    if corrected.shape != mask.shape:
        raise ValueError("편차 배열과 BOUNDARY 마스크 크기가 다릅니다.")
    if part_mask is not None:
        part = np.asarray(part_mask) > 0
        if part.shape != mask.shape:
            raise ValueError("부품 마스크와 BOUNDARY 마스크 크기가 다릅니다.")
        mask &= part
    mask &= np.isfinite(corrected)
    corrected[mask] -= group.target_mm
    return corrected


__all__ = [
    "ManagementGroup", "ManagementResult", "ManagementSurface",
    "apply_management_value", "calculate_management_result", "discover_management_groups",
    "fit_with_cad_viewer", "parse_management_value", "project_boundary",
]
