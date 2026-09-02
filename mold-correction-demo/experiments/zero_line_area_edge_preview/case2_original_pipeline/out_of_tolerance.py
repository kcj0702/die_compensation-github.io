"""Overlay +/-0.7 mm correction regions on the step-03 zero-point result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SCAN_SCALES = {
    "JD_64XX2-DR000": 2.0,
    "JD_67XX6-DR000": 3.0,
    "JD_71XX2-DR000": 2.0,
}
TOLERANCE_MM = 0.7
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode PNG: {path}")
    encoded.tofile(path)


def find_single_image(directory: Path, name_contains: str = "") -> Path:
    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and name_contains.lower() in path.stem.lower()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one matching image in {directory}, found {len(matches)}"
        )
    return matches[0]


def product_prefix(product_root: Path) -> str:
    for prefix in SCAN_SCALES:
        if product_root.name.startswith(prefix):
            return prefix
    raise ValueError(f"No color-map scale is registered for {product_root.name}")


def locate_colorbar(image: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the tall saturated color bar near the right image edge."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 140) & (hsv[:, :, 2] >= 100)
    search_x0 = int(round(width * 0.94))
    column_counts = saturated[:, search_x0:].sum(axis=0)
    best_local_x = int(np.argmax(column_counts))
    if int(column_counts[best_local_x]) < int(height * 0.60):
        raise RuntimeError("Could not locate the vertical source color bar")

    strong_columns = np.where(
        column_counts >= column_counts[best_local_x] * 0.75
    )[0]
    nearby = strong_columns[np.abs(strong_columns - best_local_x) <= 32]
    x0 = search_x0 + int(nearby.min())
    x1 = search_x0 + int(nearby.max()) + 1
    row_has_color = saturated[:, x0:x1].sum(axis=1) >= max(1, (x1 - x0) // 3)
    ys = np.where(row_has_color)[0]
    if len(ys) < int(height * 0.60):
        raise RuntimeError("Detected source color bar is too short")
    return x0, int(ys.min()), x1, int(ys.max()) + 1


def sample_bar_hue(
    image: np.ndarray,
    bar: tuple[int, int, int, int],
    deviation_mm: float,
    scale_max_mm: float,
) -> float:
    x0, y0, x1, y1 = bar
    fraction = (scale_max_mm - deviation_mm) / (2.0 * scale_max_mm)
    center_y = y0 + fraction * (y1 - y0 - 1)
    sample_y0 = max(y0, int(round(center_y)) - 4)
    sample_y1 = min(y1, int(round(center_y)) + 5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    roi = hsv[sample_y0:sample_y1, x0:x1]
    valid = (roi[:, :, 1] >= 100) & (roi[:, :, 2] >= 80)
    hues = roi[:, :, 0][valid]
    if len(hues) == 0:
        raise RuntimeError(f"Could not sample the color bar at {deviation_mm} mm")
    return float(np.median(hues))


def remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_area:
            result[labels == component] = 255
    return result


def build_correction_masks(
    cleaned: np.ndarray,
    positive_hue_limit: float,
    negative_hue_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    valid_color = (saturation >= 38) & (value >= 35)

    # The tolerance boundary itself is included in the correction region.
    positive = np.where(valid_color & (hue <= positive_hue_limit), 255, 0).astype(np.uint8)
    negative = np.where(valid_color & (hue >= negative_hue_limit), 255, 0).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    positive = cv2.morphologyEx(positive, cv2.MORPH_CLOSE, close_kernel)
    negative = cv2.morphologyEx(negative, cv2.MORPH_CLOSE, close_kernel)

    # Broad low-saturation scan surfaces are values clipped beyond the color
    # scale.  Thin gray CAD outlines are removed morphologically.
    gray_candidate = (
        (saturation < 38) & (value >= 55) & (value < 245)
    ).astype(np.uint8) * 255
    minimum_dimension = min(cleaned.shape[:2])
    gray_kernel_size = max(3, int(round(minimum_dimension * 0.006)))
    if gray_kernel_size % 2 == 0:
        gray_kernel_size += 1
    gray_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gray_kernel_size, gray_kernel_size)
    )
    gray = cv2.morphologyEx(gray_candidate, cv2.MORPH_OPEN, gray_kernel)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
    minimum_area = max(20, int(round(minimum_dimension**2 * 0.00005)))
    gray = remove_small_components(gray, minimum_area)
    gray = cv2.bitwise_and(cv2.dilate(gray, close_kernel), gray_candidate)

    return (
        remove_small_components(positive, 12),
        remove_small_components(negative, 12),
        gray,
    )


def render_overlay(
    step03: np.ndarray,
    cleaned: np.ndarray,
    correction_mask: np.ndarray,
) -> np.ndarray:
    result = step03.copy()

    # Pixels introduced in steps 02/03 are the contour graph, zero points,
    # labels, and legends.  Protect them so the red surface overlay stays
    # visually behind the existing analysis result.
    difference = cv2.absdiff(step03, cleaned).max(axis=2)
    protected = np.where(difference >= 24, 255, 0).astype(np.uint8)
    protected = cv2.dilate(
        protected,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    paint_mask = cv2.bitwise_and(correction_mask, cv2.bitwise_not(protected))

    red = result.copy()
    red[paint_mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(red, 0.62, result, 0.38, 0.0)
    result[paint_mask > 0] = blended[paint_mask > 0]

    contours, _ = cv2.findContours(
        correction_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (0, 0, 185), 1, cv2.LINE_AA)

    box_width = min(500, result.shape[1] - 18)
    x0 = max(18, result.shape[1] - box_width - 18)
    x1 = result.shape[1] - 18
    overlay = result.copy()
    cv2.rectangle(overlay, (x0, 18), (x1, 75), (255, 255, 255), -1)
    cv2.rectangle(overlay, (x0, 18), (x1, 75), (40, 40, 40), 1)
    cv2.addWeighted(overlay, 0.92, result, 0.08, 0.0, result)
    cv2.rectangle(result, (x0 + 15, 34), (x0 + 43, 50), (0, 0, 230), -1)
    cv2.putText(
        result,
        f"correction: <= -{TOLERANCE_MM:.1f} mm, >= +{TOLERANCE_MM:.1f} mm, or gray clipped",
        (x0 + 54, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "red overlay is behind contour graphs and zero points",
        (x0 + 15, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return result


def process(
    cleaned_path: Path,
    original_path: Path,
    step03_path: Path,
    output_dir: Path,
    scale_max_mm: float,
) -> tuple[Path, Path, Path]:
    cleaned = read_image(cleaned_path)
    original = read_image(original_path)
    step03 = read_image(step03_path)
    if cleaned.shape[:2] != original.shape[:2] or cleaned.shape[:2] != step03.shape[:2]:
        raise ValueError("Clean, source, and step-03 images must have identical dimensions")

    colorbar = locate_colorbar(original)
    positive_hue = sample_bar_hue(original, colorbar, TOLERANCE_MM, scale_max_mm)
    negative_hue = sample_bar_hue(original, colorbar, -TOLERANCE_MM, scale_max_mm)
    if positive_hue >= negative_hue:
        raise RuntimeError(
            f"Unexpected hue order: +{TOLERANCE_MM:.1f}={positive_hue}, "
            f"-{TOLERANCE_MM:.1f}={negative_hue}"
        )
    positive, negative, gray = build_correction_masks(
        cleaned, positive_hue, negative_hue
    )
    correction = cv2.bitwise_or(cv2.bitwise_or(positive, negative), gray)
    rendered = render_overlay(step03, cleaned, correction)

    output_dir.mkdir(parents=True, exist_ok=True)
    tolerance_label = f"{TOLERANCE_MM:.1f}".replace(".", "p")
    image_path = output_dir / f"04_outside_pm{tolerance_label}.png"
    mask_path = output_dir / f"outside_pm{tolerance_label}_mask.png"
    json_path = output_dir / "out_of_tolerance.json"
    write_png(image_path, rendered)
    write_png(mask_path, correction)
    payload = {
        "source_clean_image": str(cleaned_path.resolve()),
        "source_original_image": str(original_path.resolve()),
        "source_step03_image": str(step03_path.resolve()),
        "tolerance_mm": TOLERANCE_MM,
        "inclusive_rule": (
            f"deviation <= -{TOLERANCE_MM:.1f} mm or "
            f"deviation >= +{TOLERANCE_MM:.1f} mm"
        ),
        "gray_clipped_included": True,
        "scale_max_mm": scale_max_mm,
        "positive_hue_limit": positive_hue,
        "negative_hue_limit": negative_hue,
        "positive_pixel_count": int(cv2.countNonZero(positive)),
        "negative_pixel_count": int(cv2.countNonZero(negative)),
        "gray_clipped_pixel_count": int(cv2.countNonZero(gray)),
        "correction_pixel_count": int(cv2.countNonZero(correction)),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, mask_path, json_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    product_root = script_dir.parent.parent
    prefix = product_prefix(product_root)
    parser = argparse.ArgumentParser(
        description=f"Overlay +/-{TOLERANCE_MM:.1f} mm correction regions"
    )
    parser.add_argument("--clean-image", type=Path)
    parser.add_argument("--original-image", type=Path)
    parser.add_argument("--step03-image", type=Path, default=product_root / "output" / "03_zero_point_selection" / "03_zero_points.png")
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    args = parser.parse_args()
    if args.clean_image is None:
        args.clean_image = find_single_image(product_root / "output" / "01_label_removal", "4_labels_points_inpainted")
    if args.original_image is None:
        args.original_image = find_single_image(product_root / "input")
    args.scale_max_mm = SCAN_SCALES[prefix]
    return args


def main() -> None:
    args = parse_args()
    paths = process(
        args.clean_image.resolve(),
        args.original_image.resolve(),
        args.step03_image.resolve(),
        args.output_dir.resolve(),
        args.scale_max_mm,
    )
    for path in paths:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
