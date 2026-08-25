"""Shared HSV engine for product-specific key zero-point selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent
CONTOUR_GRAPH_OUTPUT = LAB_ROOT / "contour_graph" / "output"
ZERO_POINT_OUTPUT = HERE / "output"
MIN_COLOR_SATURATION = 45
MIN_COLOR_VALUE = 35


@dataclass(frozen=True)
class ProductConfig:
    engine_name: str
    product_prefix: str
    colorbar_top_mm: float
    colorbar_bottom_mm: float
    key_threshold_mm: float = 0.50


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"PNG 인코딩 실패: {path}")
    encoded.tofile(path)


def find_product_directory(root: Path, product_prefix: str) -> Path:
    matches = sorted(
        path for path in root.iterdir() if path.name.startswith(product_prefix)
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{root}에서 {product_prefix} 제품 폴더를 하나만 찾을 수 있어야 합니다."
        )
    return matches[0]


def build_hue_to_deviation_lut(
    original: np.ndarray, config: ProductConfig
) -> tuple[np.ndarray, dict]:
    """Read the vertical color bar and map HSV hue to deviation mm."""
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    first_x = int(round(width * 0.96))
    candidate_x = range(first_x, width)
    saturated_counts = [
        int(np.count_nonzero((hsv[:, x, 1] > 220) & (hsv[:, x, 2] > 220)))
        for x in candidate_x
    ]
    bar_x = first_x + int(np.argmax(saturated_counts))
    valid_y = np.flatnonzero(
        (hsv[:, bar_x, 1] > 220) & (hsv[:, bar_x, 2] > 220)
    )
    if len(valid_y) < height * 0.50:
        raise RuntimeError(f"{config.engine_name} 우측 HSV 컬러바를 검출하지 못했습니다.")

    top_y = int(valid_y.min())
    bottom_y = int(valid_y.max())
    reference_hues = hsv[valid_y, bar_x, 0].astype(np.float64)
    reference_values = config.colorbar_top_mm + (
        (valid_y.astype(np.float64) - top_y) / max(bottom_y - top_y, 1)
    ) * (config.colorbar_bottom_mm - config.colorbar_top_mm)

    hue_lut = np.empty(180, dtype=np.float64)
    for hue in range(180):
        delta = np.abs(reference_hues - hue)
        circular_delta = np.minimum(delta, 180.0 - delta)
        hue_lut[hue] = reference_values[int(np.argmin(circular_delta))]

    metadata = {
        "x": bar_x,
        "top_y": top_y,
        "bottom_y": bottom_y,
        "top_value_mm": config.colorbar_top_mm,
        "bottom_value_mm": config.colorbar_bottom_mm,
    }
    return hue_lut, metadata


def build_product_mask(
    shape: tuple[int, int], contour_loops: list[dict]
) -> np.ndarray:
    """Build a geometry mask when an explicit product outer loop exists.

    Nested-shell products such as JD_67XX do not have a single outer loop in
    the contour result.  For those products the HSV color-pixel condition in
    ``analyze_candidate`` excludes the white background directly.
    """
    mask = np.full(shape, 255, dtype=np.uint8)
    outer = next(
        (loop for loop in contour_loops if loop["name"] == "product_outer"), None
    )
    if outer is None:
        return mask

    mask.fill(0)
    outer_points = np.asarray(
        [sample["contour_point"] for sample in outer["samples"]], dtype=np.int32
    )
    cv2.fillPoly(mask, [outer_points], 255)

    for loop in contour_loops:
        if loop["name"] == "product_outer":
            continue
        opening = np.asarray(
            [sample["contour_point"] for sample in loop["samples"]], dtype=np.int32
        )
        cv2.fillPoly(mask, [opening], 0)
    return mask


def mean_scan_point_spacing(samples: list[dict]) -> float:
    points = np.asarray([sample["contour_point"] for sample in samples], np.float64)
    following = np.roll(points, -1, axis=0)
    return float(np.linalg.norm(following - points, axis=1).mean())


def analyze_candidate(
    zero_point: dict,
    hsv: np.ndarray,
    product_mask: np.ndarray,
    hue_lut: np.ndarray,
    radius_px: float,
    threshold_mm: float,
) -> dict:
    center = np.asarray(zero_point["point"], dtype=np.float64)
    center_pixel = tuple(np.rint(center).astype(np.int32).tolist())
    circle_mask = np.zeros(product_mask.shape, dtype=np.uint8)
    cv2.circle(circle_mask, center_pixel, int(round(radius_px)), 255, -1)

    color_pixels = (
        (circle_mask > 0)
        & (product_mask > 0)
        & (hsv[:, :, 1] >= MIN_COLOR_SATURATION)
        & (hsv[:, :, 2] >= MIN_COLOR_VALUE)
    )
    hue_values = hsv[:, :, 0][color_pixels]
    if len(hue_values) == 0:
        raise RuntimeError(f"{center_pixel} 주변에서 유효한 제품 색상 픽셀이 없습니다.")

    signed_deviation = hue_lut[hue_values]
    absolute_deviation = np.abs(signed_deviation)
    mean_absolute = float(absolute_deviation.mean())
    return {
        "point": [float(center[0]), float(center[1])],
        "source_type": zero_point["type"],
        "radius_px": round(radius_px, 5),
        "product_color_pixel_count": int(len(hue_values)),
        "mean_abs_deviation_mm": round(mean_absolute, 5),
        "median_abs_deviation_mm": round(float(np.median(absolute_deviation)), 5),
        "threshold_mm": threshold_mm,
        "selected": mean_absolute < threshold_mm,
    }


def draw_result(
    base: np.ndarray,
    candidates: list[dict],
    threshold_mm: float,
) -> np.ndarray:
    canvas = base.copy()
    overlay = canvas.copy()

    for candidate in candidates:
        center = tuple(np.rint(candidate["point"]).astype(np.int32).tolist())
        radius = int(round(candidate["radius_px"]))
        color = (40, 210, 40) if candidate["selected"] else (130, 130, 130)
        cv2.circle(overlay, center, radius, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)

    key_index = 1
    for candidate in candidates:
        center = tuple(np.rint(candidate["point"]).astype(np.int32).tolist())
        if candidate["selected"]:
            cv2.circle(canvas, center, 10, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.circle(canvas, center, 7, (40, 230, 40), -1, cv2.LINE_AA)
            label = f"K{key_index} {candidate['mean_abs_deviation_mm']:.2f}"
            key_index += 1
        else:
            cv2.drawMarker(
                canvas,
                center,
                (120, 120, 120),
                cv2.MARKER_TILTED_CROSS,
                12,
                2,
                cv2.LINE_AA,
            )
            label = f"{candidate['mean_abs_deviation_mm']:.2f}"
        cv2.putText(
            canvas,
            label,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (330, 18), (687, 91), (255, 255, 255), -1)
    cv2.rectangle(canvas, (330, 18), (687, 91), (40, 40, 40), 1)
    cv2.circle(canvas, (348, 41), 7, (40, 230, 40), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "key zero point",
        (363, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.drawMarker(canvas, (497, 41), (120, 120, 120), cv2.MARKER_TILTED_CROSS, 12, 2)
    cv2.putText(
        canvas,
        "rejected",
        (512, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    selected_count = sum(candidate["selected"] for candidate in candidates)
    cv2.putText(
        canvas,
        f"mean |deviation| < {threshold_mm:.2f} mm   key points: {selected_count}",
        (344, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_key_points_only(base: np.ndarray, candidates: list[dict]) -> np.ndarray:
    """Draw only the selected key points for downstream zero-line work."""
    canvas = base.copy()
    selected = [candidate for candidate in candidates if candidate["selected"]]

    for index, candidate in enumerate(selected, start=1):
        center = tuple(np.rint(candidate["point"]).astype(np.int32).tolist())
        cv2.circle(canvas, center, 10, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(canvas, center, 7, (40, 230, 40), -1, cv2.LINE_AA)
        label = f"K{index}"
        cv2.putText(
            canvas,
            label,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (18, 122), (235, 164), (255, 255, 255), -1)
    cv2.rectangle(canvas, (18, 122), (235, 164), (40, 40, 40), 1)
    cv2.circle(canvas, (36, 143), 7, (40, 230, 40), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"key zero points: {len(selected)}",
        (51, 149),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas


def run_product(config: ProductConfig) -> tuple[Path, Path, Path]:
    contour_dir = find_product_directory(CONTOUR_GRAPH_OUTPUT, config.product_prefix)
    output_dir = find_product_directory(ZERO_POINT_OUTPUT, config.product_prefix)
    contour_payload = json.loads(
        (contour_dir / "deviation_graph.json").read_text(encoding="utf-8")
    )
    zero_payload = json.loads(
        (output_dir / "zero_points.json").read_text(encoding="utf-8")
    )

    original = read_image(Path(contour_payload["source_original_image"]))
    clean = read_image(Path(contour_payload["source_clean_image"]))
    graph_image = read_image(contour_dir / "01_deviation_graph.png")
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    hue_lut, colorbar = build_hue_to_deviation_lut(original, config)
    product_mask = build_product_mask(hsv.shape[:2], contour_payload["loops"])
    contour_by_name = {loop["name"]: loop for loop in contour_payload["loops"]}

    output_loops = []
    all_candidates: list[dict] = []
    for zero_loop in zero_payload["loops"]:
        contour_loop = contour_by_name[zero_loop["name"]]
        radius = mean_scan_point_spacing(contour_loop["samples"])
        candidates = [
            analyze_candidate(
                point,
                hsv,
                product_mask,
                hue_lut,
                radius,
                config.key_threshold_mm,
            )
            for point in zero_loop["zero_points"]
        ]
        all_candidates.extend(candidates)
        output_loops.append(
            {
                "name": zero_loop["name"],
                "scan_point_mean_spacing_px": round(radius, 5),
                "candidate_count": len(candidates),
                "key_zero_point_count": sum(item["selected"] for item in candidates),
                "candidates": candidates,
                "key_zero_points": [item for item in candidates if item["selected"]],
            }
        )

    result = {
        "product_engine": config.engine_name,
        "source_zero_points": str((output_dir / "zero_points.json").resolve()),
        "coordinate_system": "source image absolute pixel coordinates (x, y)",
        "radius_rule": "mean adjacent scan-point spacing of each closed contour",
        "pixel_rule": "circle AND product mask AND valid HSV color pixels",
        "score_rule": "mean(abs(deviation_mm)) after HSV color-bar conversion",
        "key_threshold_mm": config.key_threshold_mm,
        "colorbar": colorbar,
        "candidate_count": len(all_candidates),
        "key_zero_point_count": sum(item["selected"] for item in all_candidates),
        "loops": output_loops,
    }

    image_output = output_dir / "02_key_zero_points.png"
    only_image_output = output_dir / "03_key_zero_points_only.png"
    json_output = output_dir / "key_zero_points.json"
    write_png(
        image_output,
        draw_result(graph_image, all_candidates, config.key_threshold_mm),
    )
    write_png(
        only_image_output,
        draw_key_points_only(graph_image, all_candidates),
    )
    json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return image_output, only_image_output, json_output
