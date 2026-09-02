"""UI adapter for the agreed hybrid zero-line engine.

Case 1 keeps the in-house area decision.  Case 2 runs the preserved original
Park Junhyeok route selector as one in-memory pipeline.  Keeping this adapter
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


def detect_hybrid_zero_line(image_bgr: np.ndarray, filename: str) -> HybridZeroLineOutput:
    """Detect a UI-ready zero result, with a safe case-1 fallback.

    The distribution rule is shared with the review engine: separated zero
    components whose total area is below 40% choose case 1; all other inputs
    choose Park Junhyeok's original case-2 routing implementation.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base = detect_zero_line(rgb, ZeroLineConfig(), source_name=filename)
    part_px = max(1, int(base.part_mask.sum()))
    count, _labels = cv2.connectedComponents(base.mask.astype(np.uint8))
    component_count = max(0, count - 1)
    ratio = float(base.mask.astype(bool).sum()) / part_px
    is_case1 = ratio < 0.40 and component_count > 1

    if is_case1:
        overlay = make_overlay(rgb, base.mask, base.centerline, zero_crossing=base.zero_crossing)
        return HybridZeroLineOutput(
            mask=base.mask.astype(bool), overlay_rgb=overlay, case=1,
            regions=len(base.result.regions), ratio=ratio, lines=[],
            warnings=list(base.warnings) + ["하이브리드 Case 1: 영역 기반 후보를 표시했습니다."],
        )

    try:
        import park_junhyeok_adapter as park  # loaded from EXPERIMENT_DIR
        import generate_final_hybrid_zero_line as hybrid

        routed = park.run_original_case2_pipeline(
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
        return HybridZeroLineOutput(
            mask=base.mask.astype(bool), overlay_rgb=overlay, case=1,
            regions=len(base.result.regions), ratio=ratio, lines=[],
            warnings=list(base.warnings) + [f"Case 2 경로 계산 실패로 Case 1 후보를 표시했습니다: {exc}"],
        )
