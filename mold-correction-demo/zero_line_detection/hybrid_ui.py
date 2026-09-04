"""UI adapter for the agreed hybrid zero-line engine.

Case 1 keeps the in-house area decision.  Case 2 runs the preserved original
route selector as one in-memory pipeline.  Keeping this adapter
here lets the UI use one engine without copying either implementation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from zero_line_detection.visualize import make_overlay
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "zero_line_area_edge_preview"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))


@dataclass
class HybridZeroLineOutput:
    mask: np.ndarray
    overlay_rgb: np.ndarray
    case: int
    regions: int
    ratio: float
    lines: list[dict]
    warnings: list[str]


def _scale_for(filename: str) -> float:
    name = filename.upper()
    if "67XX6" in name:
        return 3.0
    return 2.0


def _matching_review_spec(filename: str):
    """Return the bundled review specification for a known standard scan.

    The correction-only review inputs include the inpainted scan, its colour
    map and the gray-area sign assignment.  They are the authoritative input
    for the supplied standard scans; re-estimating those from an uploaded PNG
    changes the connected components and can select the wrong case.
    """
    import generate_adaptive_zero_line_preview as kdt

    normalized = filename.upper()
    return next(
        (spec for spec in kdt.SPECS if spec.key.upper() in normalized), None
    )


def _mask_contours_as_lines(mask: np.ndarray) -> list[dict]:
    """Expose each final zero region boundary in the response schema.

    `cv2.findContours` gives an open point list (last point does not repeat
    the first) even though the boundary is a closed loop. Consumers that draw
    this as an open polyline(<polyline> in the UI) would then show every
    region missing its last edge. Repeating the first point closes it.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    lines: list[dict] = []
    for index, contour in enumerate(contours, start=1):
        points = contour.reshape(-1, 2)
        if len(points) >= 2:
            closed = np.vstack([points, points[:1]])
            lines.append({"id": index, "points": closed.tolist()})
    return lines


def _detect_from_review_inputs(
    image_bgr: np.ndarray, filename: str
) -> HybridZeroLineOutput | None:
    """Run the exact review pipeline for a bundled standard scan.

    ``None`` means arbitrary input, which continues through the generic
    in-memory path.  A size mismatch also must not apply a mask at wrong
    pixels.
    """
    import generate_adaptive_zero_line_preview as kdt
    import generate_final_hybrid_zero_line as hybrid

    spec = _matching_review_spec(filename)
    if spec is None:
        return None
    common = hybrid.load_common_inputs(spec, hybrid.DEFAULT_CORRECTION_DIR)
    if tuple(common["part"].shape) != tuple(image_bgr.shape[:2]):
        return None

    selected_case = hybrid.select_case(common["zero_ratio"], common["zero_count"])
    if selected_case == 1:
        final_mask, details = hybrid.run_case1(common)
        lines = _mask_contours_as_lines(final_mask)
        # Keep the review result untouched: it is drawn on the inpainted
        # source, with the blue zero-area fill and its 4 px boundary.  The
        # uploaded original has labels/noise that were explicitly removed
        # before correction and zero-line detection.
        _board, overlay = kdt.build_board(
            common["image"],
            common["positive"],
            common["negative"],
            common["zero"],
            "case1_contour_polygon",
            details,
            None,
            common["zero_ratio"],
            common["zero_count"],
        )
    else:
        final_mask, details = hybrid.run_case2(common)
        lines = []
        for index, selection in enumerate(details["selections"], start=1):
            points = np.asarray(
                selection["closure_validation"]["route"]["path_points"],
                dtype=np.int32,
            )
            if len(points) >= 2:
                lines.append({"id": index, "points": points.tolist()})
        _construction, overlay = hybrid.draw_team_route_view(
            common["image"].copy(), details, final_mask
        )

    return HybridZeroLineOutput(
        mask=final_mask.astype(bool),
        overlay_rgb=overlay,
        case=selected_case,
        regions=len(lines),
        ratio=float(final_mask.sum()) / max(1, int(common["part_px"])),
        lines=lines,
        warnings=[
            f"검토 입력 재사용: {spec.key} / {selected_case}번 방식 "
            f"(제로 가능영역 {common['zero_ratio']:.2%}, "
            f"{common['zero_count']}개)"
        ],
    )


def detect_hybrid_zero_line(image_bgr: np.ndarray, filename: str) -> HybridZeroLineOutput:
    """Detect a UI-ready zero result, with a safe case-1 fallback.

    The distribution rule is shared with the review engine: separated zero
    components whose total area is below 40% choose case 1; all other inputs
    choose the original case-2 routing implementation.
    """
    try:
        review_result = _detect_from_review_inputs(image_bgr, filename)
    except Exception:
        # A missing review asset must not make the standard-scan shortcut a
        # single point of failure for an otherwise valid upload.
        review_result = None
    if review_result is not None:
        return review_result

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base = detect_zero_line(rgb, ZeroLineConfig(), source_name=filename)
    fallback_part_px = max(1, int(base.part_mask.sum()))
    fallback_ratio = float(base.mask.astype(bool).sum()) / fallback_part_px
    try:
        import case2_route_adapter as case2  # loaded from EXPERIMENT_DIR
        import generate_final_hybrid_zero_line as hybrid
        import generate_adaptive_zero_line_preview as kdt

        # UI에서도 검토 엔진과 같은 ±0.6 mm 보정영역 기준을 사용한다.
        # 기존 zero_line의 자동 tolerance(컬러바 범위의 10%)는 여기서 쓰지 않는다.
        part = base.part_mask.astype(bool)
        part_px = max(1, int(part.sum()))
        positive = part & (base.values > 0.6)
        negative = part & (base.values < -0.6)
        raw_zero = part & ~(positive | negative)
        zero, _labels, rows, _raw_rows = kdt.filter_components_by_ratio(
            raw_zero, part_px, kdt.ZERO_COMPONENT_MIN_RATIO
        )
        ratio = float(zero.sum()) / part_px
        is_case1 = ratio < 0.40 and len(rows) > 1

        if is_case1:
            final_mask, _details = hybrid.run_case1({
                "zero": zero, "positive": positive, "negative": negative,
                "part": part, "part_px": part_px,
            })
            overlay = rgb.copy()
            tint = np.zeros_like(overlay)
            tint[final_mask] = (0, 235, 255)
            overlay = cv2.addWeighted(overlay, 1.0, tint, 0.58, 0.0)
            # [버그였던 부분] Case 1(영역/다각형) 은 "선" 이 아니라 "면" 이라는
            # 이유로 lines=[] 를 그냥 박아 뒀다. 하지만 다각형도 윤곽선을 따면
            # 얼마든지 폴리라인으로 낼 수 있다 — 검토용(리뷰 자산) 경로의
            # `_mask_contours_as_lines` 가 이미 이 방식을 쓴다. 여기서도 같은
            # 함수를 재사용하면, 시트 화면(제품데이터 위 SVG 오버레이)에도
            # Case 1 결과가 실제로 그려진다. 지금까지는 review 자산이 없는
            # 이 PC 에서 Case 1 로 떨어진 스캔(67XX6 등)은 항상 "라인 없음"
            # 으로 보였다.
            return HybridZeroLineOutput(
                mask=final_mask.astype(bool), overlay_rgb=overlay, case=1,
                regions=int(cv2.connectedComponents(final_mask.astype(np.uint8))[0] - 1),
                ratio=float(final_mask.sum()) / part_px,
                lines=_mask_contours_as_lines(final_mask.astype(np.uint8)),
                warnings=list(base.warnings) + ["하이브리드 Case 1: ±0.6 mm 보정영역 기반 오프셋 다각형 결과입니다."],
            )

        routed = case2.run_original_case2_pipeline(
            original_bgr=image_bgr, scale_max_mm=_scale_for(filename)
        )
        line_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        lines: list[dict] = []
        for index, selection in enumerate(routed["selections"], start=1):
            points = np.asarray(selection["closure_validation"]["route"]["path_points"], dtype=np.int32)
            if len(points) < 2:
                continue
            cv2.polylines(line_mask, [points.reshape(-1, 1, 2)], False, 255, 5, cv2.LINE_AA)
            lines.append({"id": index, "points": points.tolist()})
        mask = line_mask.astype(bool)
        overlay_base = cv2.cvtColor(routed["cleaned_bgr"], cv2.COLOR_BGR2RGB)
        _construction, overlay = hybrid.draw_team_route_view(overlay_base, routed, mask)
        route_ratio = float(mask.sum()) / part_px
        return HybridZeroLineOutput(
            mask=mask, overlay_rgb=overlay, case=2, regions=len(lines), ratio=route_ratio,
            lines=lines, warnings=list(base.warnings),
        )
    except Exception as exc:
        overlay = make_overlay(rgb, base.mask, base.centerline, zero_crossing=base.zero_crossing)
        # 위 Case 1 분기와 같은 이유로, 여기서도 base.mask(영역) 를 폴리라인
        # 윤곽선으로 뽑아 lines 를 채운다 — 컬러바 인식 실패 등으로 case2 가
        # 죽어 이 최종 폴백까지 떨어져도 최소한 "면적 윤곽" 은 벡터로 보인다.
        try:
            fallback_lines = _mask_contours_as_lines(base.mask.astype(np.uint8))
        except Exception:
            fallback_lines = []
        return HybridZeroLineOutput(
            mask=base.mask.astype(bool), overlay_rgb=overlay, case=1,
            regions=len(base.result.regions), ratio=fallback_ratio, lines=fallback_lines,
            warnings=list(base.warnings) + [f"Case 2 경로 계산 실패로 Case 1 후보를 표시했습니다: {exc}"],
        )
