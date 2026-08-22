"""파트 간 주고받는 데이터 구조 정의.

각 파트는 내부 구현을 자유롭게 바꿔도 되지만,
아래 구조로 결과를 내보내는 규약만은 지킨다.
UI(파트 1)는 이 구조만 읽는다.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# ── [2] 0-Line 검출 결과 ─────────────────────────────────────────
@dataclass
class ZeroLineRegion:
    """0-Line 으로 판정된 영역 1개."""

    region_id: int
    area_px: int                 # 픽셀 면적
    centroid_x: float            # 중심 좌표 (원본 이미지 기준)
    centroid_y: float
    bbox_x: int                  # 외접 사각형
    bbox_y: int
    bbox_w: int
    bbox_h: int
    perimeter_px: float
    mean_value: float            # 영역 평균 편차 (정규화 또는 mm)
    unit: Literal["normalized", "mm"] = "normalized"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ZeroLineResult:
    """0-Line 검출 모듈의 전체 결과."""

    source_image: str
    image_width: int
    image_height: int
    regions: list[ZeroLineRegion] = field(default_factory=list)
    total_zero_px: int = 0
    part_px: int = 0                     # 히트맵(부품) 전체 픽셀 수
    zero_ratio: float = 0.0              # total_zero_px / part_px
    tolerance: float = 0.0               # 판정에 사용한 허용 범위
    tolerance_unit: Literal["normalized", "mm"] = "normalized"
    colorbar: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["regions"] = [r.to_dict() for r in self.regions]
        return d


# ── 컬러바 보정 정보 ─────────────────────────────────────────────
@dataclass
class ColorbarInfo:
    """이미지에서 검출한 컬러바(범례) 정보."""

    side: Literal["left", "right"]
    x0: int
    x1: int
    y0: int
    y1: int
    n_samples: int
    vmin_at: Literal["top", "bottom"]    # 최솟값이 위인지 아래인지
    vmin: float | None = None            # 지정된 경우에만
    vmax: float | None = None
    symmetric: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── UI(파트 1)가 읽는 통합 결과 ──────────────────────────────────
@dataclass
class DemoResult:
    """data/output/result.json 의 최상위 구조."""

    part_no: str = ""
    source_image: str = ""
    generated_at: str = ""
    images: dict[str, str] = field(default_factory=dict)   # 표시용 이미지 경로
    tables: dict[str, str] = field(default_factory=dict)   # CSV 경로
    zero_line: dict[str, Any] = field(default_factory=dict)
    deviation: dict[str, Any] = field(default_factory=dict)
    depth: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ZeroLineRegion", "ZeroLineResult", "ColorbarInfo", "DemoResult",
]
