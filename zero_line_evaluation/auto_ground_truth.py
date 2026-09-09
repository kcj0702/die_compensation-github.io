"""Create reviewable zero-point candidates from historical correction sheets.

This module deliberately creates *candidates*, not trusted ground truth.  The
source sheets contain correction leaders and zero annotations in the same
colours, so every generated annotation remains ``reviewed: false`` until the
operator accepts it in ``review_sample.py``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_zero_line import DEMO_ROOT, rasterize_lines, read_image, write_image


@dataclass
class Candidate:
    sheet_point: tuple[float, float]
    scan_point: tuple[float, float]
    confidence: float
    source: str
    label_box: tuple[int, int, int, int] | None = None


def _component_rows(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _count, labels, stats, _centres = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return labels, stats


def yellow_zero_boxes(sheet: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find narrow yellow boxes; in these forms they contain exactly ``0``."""
    hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (18, 120, 120), (42, 255, 255)) > 0
    _labels, stats = _component_rows(yellow)
    plausible: list[tuple[int, int, int, int, int]] = []
    for x, y, width, height, area in stats[1:]:
        if 12 <= height <= 40 and 12 <= width <= 70 and area >= 120:
            plausible.append((int(x), int(y), int(width), int(height), int(area)))
    regular_widths = [width for _x, _y, width, _height, _area in plausible if width >= 32]
    if not regular_widths:
        return []
    narrow_limit = 0.67 * float(np.median(regular_widths))
    return [
        (x, y, width, height)
        for x, y, width, height, _area in plausible
        if width <= narrow_limit
    ]


def sheet_product_components(sheet: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Remove thin leaders/labels and retain the large coloured product views."""
    hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    labels, stats = _component_rows(mask > 0)
    minimum = sheet.shape[0] * sheet.shape[1] * 0.025
    rows: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    for index, (x, y, width, height, area) in enumerate(stats[1:], start=1):
        if area < minimum or width < sheet.shape[1] * 0.2 or height < sheet.shape[0] * 0.18:
            continue
        rows.append(((labels == index).astype(np.uint8) * 255,
                     (int(x), int(y), int(width), int(height)), int(area)))
    rows.sort(key=lambda item: item[2], reverse=True)
    return [(mask_item, box) for mask_item, box, _area in rows]


def scan_part_mask(scan: np.ndarray, filename: str) -> np.ndarray:
    if str(DEMO_ROOT) not in sys.path:
        sys.path.insert(0, str(DEMO_ROOT))
    from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line

    output = detect_zero_line(
        cv2.cvtColor(scan, cv2.COLOR_BGR2RGB), ZeroLineConfig(), source_name=filename
    )
    return output.part_mask.astype(np.uint8) * 255


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("빈 제품 마스크입니다.")
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _rect_distance(point: tuple[float, float], rect: tuple[int, int, int, int]) -> float:
    x, y = point
    rx, ry, width, height = rect
    dx = max(rx - x, 0.0, x - (rx + width))
    dy = max(ry - y, 0.0, y - (ry + height))
    return float(np.hypot(dx, dy))


def _inside(point: tuple[float, float], rect: tuple[int, int, int, int], margin: int = 16) -> bool:
    x, y = point
    rx, ry, width, height = rect
    return rx - margin <= x <= rx + width + margin and ry - margin <= y <= ry + height + margin


def leader_target(
    sheet: np.ndarray,
    label_box: tuple[int, int, int, int],
    product_boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[float, float], float] | None:
    """Follow the strongest straight leader leaving one yellow zero label."""
    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 170)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=18,
        minLineLength=12, maxLineGap=7,
    )
    if lines is None:
        return None
    hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)
    candidates: list[tuple[float, tuple[float, float], float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        p1, p2 = (float(x1), float(y1)), (float(x2), float(y2))
        d1, d2 = _rect_distance(p1, label_box), _rect_distance(p2, label_box)
        if d1 <= 11 < d2:
            target, near = p2, p1
        elif d2 <= 11 < d1:
            target, near = p1, p2
        else:
            continue
        length = float(np.hypot(target[0] - near[0], target[1] - near[1]))
        if not 12 <= length <= 320:
            continue
        in_product = any(_inside(target, box) for box in product_boxes)
        tx, ty = int(round(target[0])), int(round(target[1]))
        y0, y1c = max(0, ty - 5), min(sheet.shape[0], ty + 6)
        x0, x1c = max(0, tx - 5), min(sheet.shape[1], tx + 6)
        local = hsv[y0:y1c, x0:x1c]
        coloured_dot = float(np.mean(local[:, :, 1] > 130)) if local.size else 0.0
        score = length + (85.0 if in_product else -80.0) + 45.0 * coloured_dot
        confidence = min(0.92, 0.42 + (0.25 if in_product else 0.0) + 0.2 * coloured_dot)
        candidates.append((score, target, confidence))
    if not candidates:
        return None
    _score, target, confidence = max(candidates, key=lambda item: item[0])
    return target, confidence


def affine_from_boxes(
    source: tuple[int, int, int, int], target: tuple[int, int, int, int]
) -> np.ndarray:
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    return np.asarray([[tw / sw, 0.0, tx - sx * tw / sw],
                       [0.0, th / sh, ty - sy * th / sh]], dtype=np.float64)


def apply_affine(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    value = matrix @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    return float(value[0]), float(value[1])


def sift_homography(sheet: np.ndarray, scan: np.ndarray) -> tuple[np.ndarray | None, float]:
    sift = cv2.SIFT_create(nfeatures=5000)
    sheet_gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    scan_gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    sheet_keys, sheet_desc = sift.detectAndCompute(sheet_gray, None)
    scan_keys, scan_desc = sift.detectAndCompute(scan_gray, None)
    if sheet_desc is None or scan_desc is None:
        return None, 0.0
    matches = cv2.BFMatcher().knnMatch(sheet_desc, scan_desc, k=2)
    good = [first for first, second in matches if first.distance < 0.72 * second.distance]
    if len(good) < 6:
        return None, 0.0
    source = np.float32([sheet_keys[item.queryIdx].pt for item in good])
    target = np.float32([scan_keys[item.trainIdx].pt for item in good])
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    if matrix is None or inliers is None:
        return None, 0.0
    confidence = min(0.75, float(inliers.sum()) / max(12.0, len(good)))
    return matrix, confidence


def apply_homography(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    source = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
    target = cv2.perspectiveTransform(source, matrix)[0, 0]
    return float(target[0]), float(target[1])


def _deduplicate(candidates: list[Candidate], minimum_distance: float = 12.0) -> list[Candidate]:
    selected: list[Candidate] = []
    for item in sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True):
        if all(np.hypot(item.scan_point[0] - other.scan_point[0],
                        item.scan_point[1] - other.scan_point[1]) >= minimum_distance
               for other in selected):
            selected.append(item)
    return sorted(selected, key=lambda candidate: (candidate.scan_point[1], candidate.scan_point[0]))


def generate_candidates(scan: np.ndarray, sheet: np.ndarray, sample: str, scan_name: str) -> tuple[list[Candidate], dict[str, Any]]:
    components = sheet_product_components(sheet)
    product_boxes = [box for _mask, box in components]
    scan_mask = scan_part_mask(scan, scan_name)
    scan_box = bbox(scan_mask)
    candidates: list[Candidate] = []
    details: dict[str, Any] = {"sample": sample, "scan_box": scan_box, "sheet_product_boxes": product_boxes}

    if "64XX2" in sample.upper():
        if not components:
            return [], {**details, "reason": "sheet product mask not found"}
        sheet_mask, sheet_box = components[0]
        try:
            if str(DEMO_ROOT) not in sys.path:
                sys.path.insert(0, str(DEMO_ROOT))
            from product_alignment.alignment import estimate_alignment
            alignment = estimate_alignment(sheet_mask, scan_mask)
            matrix = alignment.as_array()
            # Template-relative endpoints of the two visible '0 LINE' leaders.
            sx, sy, sw, sh = sheet_box
            anchors = [(sx + 0.192 * sw, sy + 0.008 * sh),
                       (sx + 0.820 * sw, sy - 0.030 * sh)]
            for point in anchors:
                candidates.append(Candidate(point, apply_affine(matrix, point),
                                            0.78 if alignment.confident else 0.48,
                                            "64XX2 ‘0 LINE’ template"))
            details["alignment"] = alignment.to_dict()
        except Exception as exc:
            details["reason"] = str(exc)
    else:
        boxes = yellow_zero_boxes(sheet)
        details["yellow_zero_boxes"] = boxes
        targets: list[tuple[tuple[int, int, int, int], tuple[float, float], float]] = []
        for label_box in boxes:
            found = leader_target(sheet, label_box, product_boxes)
            if found is not None:
                target, confidence = found
                targets.append((label_box, target, confidence))
        details["leader_target_count"] = len(targets)
        if "67XX6" in sample.upper() and components:
            matrix = affine_from_boxes(components[0][1], scan_box)
            for label_box, point, confidence in targets:
                mapped = apply_affine(matrix, point)
                candidates.append(Candidate(point, mapped, confidence * 0.78,
                                            "yellow 0 + leader + bbox alignment", label_box))
            details["alignment"] = {"method": "product_bbox", "matrix": matrix.reshape(-1).tolist(), "confidence": 0.78}
        else:
            matrix, alignment_confidence = sift_homography(sheet, scan)
            if matrix is not None:
                for label_box, point, confidence in targets:
                    mapped = apply_homography(matrix, point)
                    if _inside(mapped, scan_box, margin=40):
                        candidates.append(Candidate(point, mapped, confidence * alignment_confidence,
                                                    "yellow 0 + leader + SIFT alignment", label_box))
                details["alignment"] = {"method": "SIFT homography", "matrix": matrix.reshape(-1).tolist(),
                                        "confidence": alignment_confidence}
            else:
                details["reason"] = "SIFT alignment failed"
    return _deduplicate(candidates), details


def preview_images(
    scan: np.ndarray, sheet: np.ndarray, candidates: list[Candidate]
) -> tuple[np.ndarray, np.ndarray]:
    scan_view, sheet_view = scan.copy(), sheet.copy()
    for index, item in enumerate(candidates, start=1):
        sx, sy = map(lambda value: int(round(value)), item.sheet_point)
        px, py = map(lambda value: int(round(value)), item.scan_point)
        if item.label_box:
            x, y, width, height = item.label_box
            cv2.rectangle(sheet_view, (x, y), (x + width, y + height), (0, 180, 0), 2)
        cv2.circle(sheet_view, (sx, sy), 8, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(sheet_view, f"A{index}", (sx + 7, sy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        cv2.circle(scan_view, (px, py), 9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(scan_view, f"A{index}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
    return scan_view, sheet_view


def write_candidate_annotation(
    annotation_path: Path,
    scan_path: Path,
    sheet_path: Path,
    sample: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if annotation_path.exists() and not force:
        existing = json.loads(annotation_path.read_text(encoding="utf-8"))
        if existing.get("reviewed") or existing.get("lines"):
            return existing
    scan, sheet = read_image(scan_path), read_image(sheet_path)
    candidates, details = generate_candidates(scan, sheet, sample, scan_path.name)
    payload = {
        "schema": 1,
        "scan_image": str(scan_path.resolve()),
        "sheet_image": str(sheet_path.resolve()),
        "kind": "points",
        "reviewed": False,
        "review_status": "auto_candidates_unreviewed",
        "auto_candidate_count": len(candidates),
        "auto_confidence": round(float(np.mean([item.confidence for item in candidates])), 3) if candidates else 0.0,
        "auto_details": details,
        "lines": [{"id": index, "points": [list(item.scan_point)],
                   "confidence": round(item.confidence, 3), "source": item.source}
                  for index, item in enumerate(candidates, start=1)],
    }
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    scan_preview, sheet_preview = preview_images(scan, sheet, candidates)
    write_image(annotation_path.parent / "auto_candidates_scan.png", scan_preview)
    write_image(annotation_path.parent / "auto_candidates_sheet.png", sheet_preview)
    return payload


__all__ = [
    "Candidate", "generate_candidates", "leader_target", "sheet_product_components",
    "write_candidate_annotation", "yellow_zero_boxes",
]
