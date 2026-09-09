"""Evaluate the current zero-line detector against a reviewed correction sheet.

The ground-truth JSON is created by ``annotate_ground_truth.py``.  Coordinates
are stored in the original scan-image coordinate system, so the sheet's crop,
rotation and scale cannot silently corrupt the metric.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEMO_ROOT = REPO_ROOT / "mold-correction-demo"


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read paths containing Korean characters on Windows."""
    payload = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(payload, flags)
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"이미지를 인코딩하지 못했습니다: {path}")
    path.write_bytes(encoded.tobytes())


def resolve_annotation_path(annotation_file: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [annotation_file.parent / path, REPO_ROOT / path]
    return next((item.resolve() for item in candidates if item.exists()), candidates[0].resolve())


def rasterize_lines(shape: tuple[int, int], lines: list[Any], thickness: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for entry in lines:
        points = entry.get("points", []) if isinstance(entry, dict) else entry
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2) if points else np.empty((0, 2))
        if len(pts) == 0:
            continue
        pts[:, 0] = np.clip(np.rint(pts[:, 0]), 0, shape[1] - 1)
        pts[:, 1] = np.clip(np.rint(pts[:, 1]), 0, shape[0] - 1)
        points_i = pts.astype(np.int32)
        if len(points_i) == 1:
            cv2.circle(mask, tuple(points_i[0]), max(0, thickness // 2), 255, -1)
        else:
            cv2.polylines(mask, [points_i.reshape(-1, 1, 2)], False, 255, thickness, cv2.LINE_8)
    return mask > 0


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton fallback when a predictor exposes only an area mask."""
    source = (mask.astype(np.uint8) * 255).copy()
    skeleton = np.zeros_like(source)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(source):
        opened = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(source, opened))
        source = cv2.erode(source, kernel)
    return skeleton > 0


@dataclass
class Prediction:
    line_mask: np.ndarray
    area_mask: np.ndarray
    overlay_bgr: np.ndarray
    line_count: int
    case: int | None
    warnings: list[str]


def current_prediction(scan_bgr: np.ndarray, filename: str) -> Prediction:
    if str(DEMO_ROOT) not in sys.path:
        sys.path.insert(0, str(DEMO_ROOT))
    from zero_line_detection.hybrid_ui import detect_hybrid_zero_line

    output = detect_hybrid_zero_line(scan_bgr, filename)
    lines = list(output.lines or [])
    line_mask = rasterize_lines(scan_bgr.shape[:2], lines, thickness=1)
    if not line_mask.any():
        line_mask = skeletonize(output.mask.astype(bool))
    overlay_bgr = cv2.cvtColor(output.overlay_rgb, cv2.COLOR_RGB2BGR)
    return Prediction(
        line_mask=line_mask,
        area_mask=output.mask.astype(bool),
        overlay_bgr=overlay_bgr,
        line_count=len(lines),
        case=int(output.case),
        warnings=list(output.warnings),
    )


def mask_prediction(mask_path: Path, scan_bgr: np.ndarray) -> Prediction:
    raw = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
    if raw.shape != scan_bgr.shape[:2]:
        raise ValueError(f"검출 마스크 크기 {raw.shape}와 스캔 크기 {scan_bgr.shape[:2]}가 다릅니다.")
    area = raw > 0
    line = skeletonize(area)
    overlay = scan_bgr.copy()
    overlay[area] = (0, 0, 255)
    return Prediction(line, area, overlay, 0, None, [])


def distance_to(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    inverse = (~mask).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def safe_percent(value: float | None) -> str:
    return "해당 없음" if value is None else f"{value * 100:.1f}%"


def evaluate_line(pred: np.ndarray, gt: np.ndarray, tolerance_px: float) -> dict[str, Any]:
    pred_to_gt = distance_to(gt)[pred]
    gt_to_pred = distance_to(pred)[gt]
    precision = float(np.mean(pred_to_gt <= tolerance_px)) if len(pred_to_gt) else 0.0
    recall = float(np.mean(gt_to_pred <= tolerance_px)) if len(gt_to_pred) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    distances = np.concatenate([pred_to_gt, gt_to_pred]) if len(pred_to_gt) + len(gt_to_pred) else np.array([np.inf])
    radius = max(1, int(round(tolerance_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    pred_band = cv2.dilate(pred.astype(np.uint8), kernel) > 0
    gt_band = cv2.dilate(gt.astype(np.uint8), kernel) > 0
    union = np.logical_or(pred_band, gt_band).sum()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_symmetric_distance_px": float(np.mean(distances)),
        "p95_symmetric_distance_px": float(np.percentile(distances, 95)),
        "band_iou": float(np.logical_and(pred_band, gt_band).sum() / union) if union else 0.0,
        "prediction_line_pixels": int(pred.sum()),
        "ground_truth_line_pixels": int(gt.sum()),
    }


def annotation_points(lines: list[Any]) -> np.ndarray:
    collected: list[list[float]] = []
    for entry in lines:
        points = entry.get("points", []) if isinstance(entry, dict) else entry
        collected.extend(points)
    return np.asarray(collected, dtype=np.float32).reshape(-1, 2) if collected else np.empty((0, 2))


def evaluate_points(pred: np.ndarray, points: np.ndarray, tolerance_px: float) -> dict[str, Any]:
    if len(points) == 0:
        raise ValueError("points 정답에 좌표가 없습니다.")
    h, w = pred.shape
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1)
    distances = distance_to(pred)[ys, xs]
    return {
        "point_hit_rate": float(np.mean(distances <= tolerance_px)),
        "mean_point_distance_px": float(np.mean(distances)),
        "p95_point_distance_px": float(np.percentile(distances, 95)),
        "ground_truth_point_count": int(len(points)),
        "matched_point_count": int(np.sum(distances <= tolerance_px)),
    }


def add_units(metrics: dict[str, Any], mm_per_pixel: float | None) -> None:
    if mm_per_pixel is None:
        return
    for key, value in list(metrics.items()):
        if key.endswith("_distance_px"):
            metrics[key.removesuffix("_px") + "_mm"] = float(value) * mm_per_pixel


def comparison_image(scan: np.ndarray, pred: np.ndarray, gt: np.ndarray, tolerance_px: float) -> np.ndarray:
    canvas = cv2.addWeighted(scan, 0.58, np.full_like(scan, 245), 0.42, 0)
    radius = max(1, int(round(tolerance_px)))
    near_gt = distance_to(gt) <= tolerance_px
    near_pred = distance_to(pred) <= tolerance_px
    pred_only = pred & ~near_gt
    gt_only = gt & ~near_pred
    matched = (pred & near_gt) | (gt & near_pred)
    show_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_show = cv2.dilate(pred_only.astype(np.uint8), show_kernel) > 0
    gt_show = cv2.dilate(gt_only.astype(np.uint8), show_kernel) > 0
    match_show = cv2.dilate(matched.astype(np.uint8), show_kernel) > 0
    canvas[pred_show] = (40, 40, 235)       # red: false positive / displaced prediction
    canvas[gt_show] = (40, 200, 40)         # green: missed ground truth
    canvas[match_show] = (0, 220, 255)      # yellow: match
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 760), 43), (255, 255, 255), -1)
    cv2.putText(canvas, "YELLOW match   RED prediction-only   GREEN ground-truth-only",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"tolerance={tolerance_px:g}px (display radius={radius}px)",
                (max(12, canvas.shape[1] - 360), 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def image_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_html(report_path: Path, result: dict[str, Any], image_paths: dict[str, Path]) -> None:
    mode = result["ground_truth_kind"]
    metrics = result["metrics"]
    if mode == "line":
        cards = [
            ("Line F1", safe_percent(metrics["f1"])),
            ("검출 정확도", safe_percent(metrics["precision"])),
            ("실제선 재현율", safe_percent(metrics["recall"])),
            ("Band IoU", safe_percent(metrics["band_iou"])),
            ("평균 거리", f"{metrics['mean_symmetric_distance_px']:.2f} px"),
            ("95% 거리", f"{metrics['p95_symmetric_distance_px']:.2f} px"),
        ]
    else:
        cards = [
            ("제로 포인트 적중률", safe_percent(metrics["point_hit_rate"])),
            ("일치 포인트", f"{metrics['matched_point_count']} / {metrics['ground_truth_point_count']}"),
            ("평균 거리", f"{metrics['mean_point_distance_px']:.2f} px"),
            ("95% 거리", f"{metrics['p95_point_distance_px']:.2f} px"),
        ]
    if result.get("mm_per_pixel"):
        distance_keys = [key for key in metrics if key.endswith("_distance_mm")]
        cards.extend((key.replace("_", " "), f"{metrics[key]:.2f} mm") for key in distance_keys)
    card_html = "".join(
        f"<div class='card'><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>"
        for label, value in cards
    )
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in result.get("warnings", []))
    images = "".join(
        f"<section><h2>{html.escape(label)}</h2><img src='{image_data_uri(path)}' alt='{html.escape(label)}'></section>"
        for label, path in image_paths.items()
    )
    report_path.write_text(f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><title>제로라인 평가 결과</title>
<style>
body{{font-family:Arial,'Malgun Gothic',sans-serif;margin:24px;background:#f4f6f8;color:#18212b}}
h1{{margin-bottom:6px}} .meta{{color:#52606d;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.card{{background:white;border:1px solid #d8dee4;border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:8px}}
.card b{{font-size:24px;color:#0969da}} section{{background:white;margin-top:18px;padding:14px;border-radius:10px}}
img{{max-width:100%;height:auto;border:1px solid #d8dee4}} code{{background:#eaeef2;padding:2px 5px}}
</style></head><body>
<h1>제로라인 평가 결과</h1>
<div class='meta'>{html.escape(result['scan'])} · 정답 방식 <code>{mode}</code> · 허용거리 {result['tolerance_px']:g}px</div>
<div class='cards'>{card_html}</div>
<section><h2>주의사항</h2><ul>{warning_html or '<li>없음</li>'}</ul></section>
{images}
</body></html>""", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실제 보정시트 기준 제로라인 평가")
    parser.add_argument("--annotation", type=Path, required=True, help="주석 도구가 저장한 JSON")
    parser.add_argument("--output", type=Path, required=True, help="결과 디렉터리")
    parser.add_argument("--prediction-mask", type=Path, help="현재 검출기 대신 사용할 이진 마스크")
    parser.add_argument("--tolerance-px", type=float, default=8.0)
    parser.add_argument("--tolerance-mm", type=float)
    parser.add_argument("--mm-per-pixel", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotation_file = args.annotation.resolve()
    annotation = json.loads(annotation_file.read_text(encoding="utf-8"))
    if not annotation.get("reviewed", False):
        raise ValueError("정답 JSON의 reviewed가 false입니다. 주석 창에서 S로 검토·저장하세요.")
    scan_path = resolve_annotation_path(annotation_file, annotation["scan_image"])
    sheet_path = resolve_annotation_path(annotation_file, annotation["sheet_image"])
    scan = read_image(scan_path)
    lines = list(annotation.get("lines", []))
    kind = str(annotation.get("kind", "line")).lower()
    if kind not in {"line", "points"}:
        raise ValueError("정답 kind는 line 또는 points여야 합니다.")
    mm_per_pixel = args.mm_per_pixel
    if args.tolerance_mm is not None:
        if not mm_per_pixel or mm_per_pixel <= 0:
            raise ValueError("--tolerance-mm 사용 시 양수 --mm-per-pixel이 필요합니다.")
        tolerance_px = args.tolerance_mm / mm_per_pixel
    else:
        tolerance_px = args.tolerance_px
    if tolerance_px <= 0:
        raise ValueError("허용거리는 0보다 커야 합니다.")
    prediction = mask_prediction(args.prediction_mask.resolve(), scan) if args.prediction_mask else current_prediction(scan, scan_path.name)
    gt_mask = rasterize_lines(scan.shape[:2], lines, thickness=1)
    if not gt_mask.any():
        raise ValueError("정답 선/점이 비어 있습니다.")
    metrics = evaluate_line(prediction.line_mask, gt_mask, tolerance_px) if kind == "line" else evaluate_points(
        prediction.line_mask, annotation_points(lines), tolerance_px
    )
    add_units(metrics, mm_per_pixel)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison_path = output / "comparison.png"
    prediction_path = output / "prediction_overlay.png"
    sheet_copy_path = output / "reference_sheet.png"
    write_image(comparison_path, comparison_image(scan, prediction.line_mask, gt_mask, tolerance_px))
    write_image(prediction_path, prediction.overlay_bgr)
    write_image(sheet_copy_path, read_image(sheet_path))
    warnings = list(prediction.warnings)
    review_status = str(annotation.get("review_status", "operator_reviewed"))
    if review_status != "operator_reviewed":
        warnings.append(
            "정답선은 작업자 확정본이 아니라 초기 시각 판독본입니다. 최종 정확도로 사용하기 전에 주석 도구에서 확인하세요."
        )
    warnings.extend(str(item) for item in annotation.get("notes", []))
    if mm_per_pixel is None:
        warnings.append("실제 축척이 없어 거리는 px 단위입니다. 서로 다른 해상도끼리 수치를 직접 비교하지 마세요.")
    if kind == "points":
        warnings.append("실제 시트가 점 위치만 제공하므로 Line F1 대신 제로 포인트 적중률을 사용했습니다.")
    result = {
        "scan": str(scan_path),
        "sheet": str(sheet_path),
        "annotation": str(annotation_file),
        "ground_truth_kind": kind,
        "reviewed": True,
        "review_status": review_status,
        "detector_case": prediction.case,
        "detected_line_count": prediction.line_count,
        "tolerance_px": float(tolerance_px),
        "mm_per_pixel": mm_per_pixel,
        "metrics": metrics,
        "warnings": warnings,
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    build_html(output / "report.html", result, {
        "정답과 검출 비교": comparison_path,
        "현재 검출 결과": prediction_path,
        "실제 보정시트": sheet_copy_path,
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"보고서: {output / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
