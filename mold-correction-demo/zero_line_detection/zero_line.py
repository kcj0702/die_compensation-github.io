"""0-Line 영역 검출 — 파트 2 핵심 로직.

[정의]
0-Line = 3D 스캔 편차가 0 근처인 영역. 금형을 깎지도 붙이지도 않고
그대로 두는 구간이며, 보정시트에서 노란 '0' 라벨과 빨간 점선 영역으로 표기된다.

[처리 흐름]
    1. 컬러바 검출        이미지 안의 범례에서 색->편차값 대응표를 만든다
    2. 주석 마스킹        라벨 박스·지시선을 제외한다
    3. 값 변환            모든 픽셀을 편차값으로 바꾼다
    4. 부품 영역 확정     컬러바 색과 일치하는 픽셀만 남긴다
    5. 0 밴드 추출        |편차| <= 허용오차
    6. 형태 정리          잡음 제거, 끊긴 부분 연결, 작은 조각 제거
    7. 영역·윤곽 산출     영역별 통계와 폴리라인

[허용오차]
기본값은 컬러바 반경의 10%다. 즉 범위가 ±3.0 이면 ±0.3mm,
±2.0 이면 ±0.2mm 가 된다. 보정시트에서 '0' 으로 표기된 지점과
'-0.5' 로 표기된 지점이 구분되는 수준이며, 현장 기준이 확인되면
--tolerance 로 고정값을 주면 된다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.schemas import ZeroLineRegion, ZeroLineResult  # noqa: E402
from zero_line_detection.annotations import build_annotation_mask  # noqa: E402
from zero_line_detection.colorbar import (  # noqa: E402
    Colorbar, canonical_colorbar, detect_colorbar,
)


@dataclass
class ZeroLineConfig:
    """검출 파라미터. 전부 CLI 로 조정 가능하다."""

    tolerance: float | None = None      # 허용오차 절대값 (mm 또는 정규화 단위)
    tolerance_ratio: float = 0.10       # 미지정 시 컬러바 반경 대비 비율
    color_max_dist: float = 14.0        # 컬러바 색과의 Lab 거리 허용치
    smooth_ksize: int = 5               # 편차값 중앙값 필터 크기 (0 이면 미적용)
    morph_open: int = 2                 # 점 잡음 제거
    morph_close: int = 4                # 끊긴 영역 연결
    min_region_area: int = 80           # 이보다 작은 조각은 버린다
    min_part_area: int = 500            # 부품 영역 최소 크기 (절대)
    part_area_ratio: float = 0.05       # 가장 큰 덩어리 대비 이 비율 미만이면 버린다
    use_annotation_mask: bool = True
    vmin: float | None = None
    vmax: float | None = None
    emit_centerline: bool = True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ZeroLineOutput:
    """검출 결과 묶음."""

    result: ZeroLineResult
    mask: np.ndarray                    # (H,W) uint8  0/255  (허용오차 기반 0 '영역')
    zero_crossing: np.ndarray           # (H,W) uint8  0/255  (부호 경계 = 임계값 없는 0 '선')
    centerline: np.ndarray | None       # (H,W) uint8  0/255
    values: np.ndarray                  # (H,W) float32 편차값
    part_mask: np.ndarray               # (H,W) bool   히트맵 영역
    contours: list = field(default_factory=list)
    colorbar: Colorbar | None = None
    warnings: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
def _clean_binary(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k * 2 + 1,) * 2)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k * 2 + 1,) * 2)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out.astype(bool)


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(mask.shape, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = True
    return keep


def _keep_main_parts(mask: np.ndarray, min_area: int, ratio: float) -> np.ndarray:
    """부품 본체만 남기고 주석 덩어리를 버린다.

    제목 텍스트("26.01.29", "LH")나 부품번호를 가린 파란 사각형은
    색상만 보면 히트맵 데이터와 구분되지 않는다. 다만 부품 본체에 비하면
    훨씬 작고 따로 떨어져 있으므로, 가장 큰 덩어리 대비 면적비로 걸러낸다.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = max(min_area, int(areas.max() * ratio))
    keep = np.zeros(mask.shape, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= threshold:
            keep[labels == i] = True
    return keep


def zero_crossing_line(
    values: np.ndarray, part: np.ndarray, min_len: int = 25
) -> np.ndarray:
    """부호가 바뀌는 경계를 찾는다 — 허용오차가 필요 없는 진짜 0-Line.

    [왜 이게 필요한가]
    |편차| <= tol 로 뽑는 0 '영역' 은 tol 을 얼마로 잡느냐에 따라 넓어지고 좁아진다.
    그 값을 우리가 정하면 결과도 우리가 정한 것이 되어 근거를 대기 어렵다.

    반면 편차가 +에서 - 로 바뀌는 지점은 **정의상 편차가 정확히 0인 곳**이다.
    임계값을 하나도 쓰지 않으므로 "임의로 그었다" 는 지적을 받지 않는다.

    구현은 단순하다. 양수 영역과 음수 영역을 각각 한 픽셀씩 부풀려
    겹치는 곳이 곧 두 영역이 만나는 경계다. 부품 안쪽으로 한정하므로
    부품 외곽선은 잡히지 않는다.
    """
    pos = ((values > 0) & part).astype(np.uint8)
    neg = ((values < 0) & part).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    crossing = (cv2.dilate(pos, k) > 0) & (cv2.dilate(neg, k) > 0) & part

    # 잡음으로 생긴 짧은 조각은 버린다
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        crossing.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(crossing.shape, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_len or max(w, h) >= min_len:
            keep[labels == i] = True
    return (keep.astype(np.uint8)) * 255


def tolerance_sweep(
    values: np.ndarray, part: np.ndarray, tolerances: list
) -> list:
    """허용오차를 바꿔가며 0 영역 면적이 어떻게 변하는지 표로 만든다.

    "왜 하필 그 값이냐" 는 질문에 숫자로 답하기 위한 것이다.
    민감도를 보여주면 임계값 선택이 결과를 얼마나 좌우하는지 드러난다.
    """
    part_px = int(part.sum())
    rows = []
    for t in tolerances:
        area = int((part & (np.abs(values) <= t)).sum())
        rows.append({
            "tolerance": float(t),
            "area_px": area,
            "ratio_of_part": (area / part_px) if part_px else 0.0,
        })
    return rows


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 세선화. 0 밴드의 중심선을 뽑는다.

    0-Line 은 이름 그대로 '선' 이므로, 폭을 가진 밴드보다
    중심선 쪽이 보정시트에 옮겨 적기 쉽다.
    """
    img = (mask > 0).astype(np.uint8)
    for _ in range(200):                      # 안전 상한
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2 = p[:-2, 1:-1]
            P3 = p[:-2, 2:]
            P4 = p[1:-1, 2:]
            P5 = p[2:, 2:]
            P6 = p[2:, 1:-1]
            P7 = p[2:, :-2]
            P8 = p[1:-1, :-2]
            P9 = p[:-2, :-2]

            B = (P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9).astype(np.int16)
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros(img.shape, dtype=np.int16)
            for i in range(8):
                A += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16)

            if step == 0:
                c1 = P2 * P4 * P6
                c2 = P4 * P6 * P8
            else:
                c1 = P2 * P4 * P8
                c2 = P2 * P6 * P8

            remove = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
def detect_zero_line(
    rgb: np.ndarray,
    config: ZeroLineConfig | None = None,
    source_name: str = "",
) -> ZeroLineOutput:
    """3D 스캔 편차 이미지에서 0-Line 영역을 검출한다."""
    cfg = config or ZeroLineConfig()
    warnings: list = []
    h, w = rgb.shape[:2]

    # 1) 컬러바 -------------------------------------------------------
    # 범례가 잘려 나갔거나 캡처에 안 담긴 이미지가 들어온다. 예전에는
    # 여기서 예외가 나면 제로라인 단계 전체가 "실행 실패" 로 끝났다.
    # 컬러바 범위를 아는 품번이면(vmin/vmax 를 받았으면) 표준 램프로
    # 이어서 진행한다 — 정확도는 떨어지지만 아무것도 못 내는 것보다 낫다.
    try:
        cb = detect_colorbar(rgb, vmin=cfg.vmin, vmax=cfg.vmax)
    except RuntimeError:
        if cfg.vmin is None or cfg.vmax is None:
            raise
        cb = canonical_colorbar(cfg.vmin, cfg.vmax)
        warnings.append(
            f"이미지에서 컬러바를 찾지 못해 표준 무지개 램프({cfg.vmin:+.1f} ~ "
            f"{cfg.vmax:+.1f}mm)를 기준으로 색을 값으로 옮겼습니다. 실제 범례가 "
            "표준 램프의 일부만 쓰고 있으면 값이 어긋날 수 있으니, 범례가 "
            "보이는 원본으로 다시 확인하세요."
        )

    if cb.is_clipped and (cfg.vmin is None or cfg.vmax is None):
        lo, hi = cb.endpoint_gaps
        warnings.append(
            f"컬러바가 이미지 경계에서 잘렸습니다 (끝점 색 오차 {lo:.0f}/{hi:.0f}). "
            "보이는 구간이 전체 범위가 아니므로 '중앙 = 편차 0' 가정이 성립하지 "
            "않습니다. 컬러바에 적힌 최소·최대값을 --vmin / --vmax 로 지정하세요. "
            # 현업이 알려준 실제 컬러바 범위(simple_zero_line.PRODUCT_COLORBAR_MM).
            # 전에 -1.5 로 적어 뒀는데 JD_64XX 의 하단은 -1.6 이다.
            "(JD_64XX 는 --vmin -1.6 --vmax 2.0, JD_67XX 는 -3.0/3.0, "
            "JD_71XX 는 -2.0/2.0)"
        )

    # 2) 주석 마스크 --------------------------------------------------
    ann = np.zeros((h, w), dtype=bool)
    if cfg.use_annotation_mask:
        ann = build_annotation_mask(
            rgb, colorbar_bbox=(cb.info.x0, cb.info.y0, cb.info.x1, cb.info.y1)
        )

    # 3) 색 -> 편차값 -------------------------------------------------
    values, color_valid = cb.map_image(rgb, max_dist=cfg.color_max_dist)

    if cfg.smooth_ksize > 0:
        values = cv2.medianBlur(values, cfg.smooth_ksize | 1)

    # 4) 부품(히트맵) 영역 --------------------------------------------
    part = color_valid & ~ann
    part = _clean_binary(part, open_k=1, close_k=2)
    part = _keep_main_parts(part, cfg.min_part_area, cfg.part_area_ratio)
    part_px = int(part.sum())
    if part_px == 0:
        warnings.append("히트맵 영역을 찾지 못했습니다. --color-max-dist 를 키워 보세요.")

    # 5) 0 밴드 -------------------------------------------------------
    tol = cfg.tolerance if cfg.tolerance is not None \
        else cfg.tolerance_ratio * cb.half_span
    zero = part & (np.abs(values) <= tol)

    # 6) 형태 정리 ----------------------------------------------------
    zero = _clean_binary(zero, cfg.morph_open, cfg.morph_close)
    zero = zero & part
    zero = _drop_small(zero, cfg.min_region_area)
    mask_u8 = zero.astype(np.uint8) * 255

    # 7) 영역 통계와 윤곽 ---------------------------------------------
    regions: list = []
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        zero.astype(np.uint8), connectivity=8
    )
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = float(sum(cv2.arcLength(c, True) for c in cnts))
        regions.append(ZeroLineRegion(
            region_id=int(i),
            area_px=int(area),
            centroid_x=float(centroids[i][0]),
            centroid_y=float(centroids[i][1]),
            bbox_x=int(x), bbox_y=int(y), bbox_w=int(bw), bbox_h=int(bh),
            perimeter_px=perim,
            mean_value=float(values[labels == i].mean()) if area else 0.0,
            unit=cb.unit,
        ))
    regions.sort(key=lambda r: r.area_px, reverse=True)

    contours, _ = cv2.findContours(
        zero.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    crossing = zero_crossing_line(values, part)
    centerline = skeletonize(zero) if (cfg.emit_centerline and zero.any()) else None

    total = int(zero.sum())
    result = ZeroLineResult(
        source_image=source_name,
        image_width=w,
        image_height=h,
        regions=regions,
        total_zero_px=total,
        part_px=part_px,
        zero_ratio=(total / part_px) if part_px else 0.0,
        tolerance=float(tol),
        tolerance_unit=cb.unit,
        colorbar=cb.to_dict(),
        params=cfg.to_dict(),
    )
    result.params["tolerance_sweep"] = tolerance_sweep(
        values, part,
        [round(cb.half_span * r, 4) for r in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)],
    )
    result.params["zero_crossing_px"] = int((crossing > 0).sum())

    return ZeroLineOutput(
        result=result, mask=mask_u8, zero_crossing=crossing, centerline=centerline,
        values=values, part_mask=part, contours=list(contours),
        colorbar=cb, warnings=warnings,
    )


__all__ = [
    "ZeroLineConfig", "ZeroLineOutput", "detect_zero_line",
    "skeletonize", "zero_crossing_line", "tolerance_sweep",
]
