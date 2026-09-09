"""Merge nearby +/-0.7 correction areas and remove small isolated regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


GAP_RATIO = 0.014
MIN_LARGEST_RATIO = 0.06
MIN_PRODUCT_RATIO = 0.001


def read_image(path: Path, mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, mode)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode PNG: {path}")
    encoded.tofile(path)


def find_clean_image(directory: Path) -> Path:
    matches = sorted(directory.glob("*4_labels_points_inpainted.png"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one final label-removal image in {directory}, found {len(matches)}"
        )
    return matches[0]


def build_product_mask(image: np.ndarray) -> np.ndarray:
    distance_from_white = 255 - image.min(axis=2)
    foreground = np.where(distance_from_white >= 18, 255, 0).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    if count <= 1:
        raise RuntimeError("No product foreground was detected")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == component, 255, 0).astype(np.uint8)


def odd_kernel_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    if size % 2 == 0:
        size += 1
    return size


def component_areas(mask: np.ndarray) -> list[int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    return sorted(
        [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)],
        reverse=True,
    )


def merge_and_filter(
    correction_mask: np.ndarray,
    product_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    minimum_dimension = min(correction_mask.shape)
    gap_kernel_size = odd_kernel_size(minimum_dimension * GAP_RATIO)
    gap_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gap_kernel_size, gap_kernel_size)
    )

    source = cv2.bitwise_and(
        np.where(correction_mask > 0, 255, 0).astype(np.uint8),
        product_mask,
    )
    merged = cv2.morphologyEx(source, cv2.MORPH_CLOSE, gap_kernel)
    merged = cv2.bitwise_and(merged, product_mask)
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (merged > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(source), {
            "gap_kernel_size_px": gap_kernel_size,
            "minimum_area_px": 0.0,
            "components": [],
        }

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    product_area = int(cv2.countNonZero(product_mask))
    minimum_area = max(
        largest_area * MIN_LARGEST_RATIO,
        product_area * MIN_PRODUCT_RATIO,
    )
    ranked_ids = sorted(
        range(1, count),
        key=lambda component_id: int(stats[component_id, cv2.CC_STAT_AREA]),
        reverse=True,
    )

    filtered = np.zeros_like(source)
    components: list[dict] = []
    for rank, component_id in enumerate(ranked_ids, start=1):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        kept = area >= minimum_area
        if kept:
            filtered[labels == component_id] = 255
        components.append(
            {
                "rank_by_area": rank,
                "area_px": area,
                "kept": kept,
                "centroid": [
                    round(float(centroids[component_id, 0]), 3),
                    round(float(centroids[component_id, 1]), 3),
                ],
                "bounding_box": [
                    int(stats[component_id, cv2.CC_STAT_LEFT]),
                    int(stats[component_id, cv2.CC_STAT_TOP]),
                    int(stats[component_id, cv2.CC_STAT_WIDTH]),
                    int(stats[component_id, cv2.CC_STAT_HEIGHT]),
                ],
            }
        )

    filtered = cv2.morphologyEx(
        filtered,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    filtered = cv2.bitwise_and(filtered, product_mask)
    return filtered, {
        "gap_kernel_size_px": gap_kernel_size,
        "largest_merged_area_px": largest_area,
        "product_area_px": product_area,
        "minimum_area_px": minimum_area,
        "components": components,
    }


def render_overlay(
    step03: np.ndarray,
    cleaned: np.ndarray,
    merged_mask: np.ndarray,
    gap_kernel_size: int,
    kept_count: int,
    merged_count: int,
) -> np.ndarray:
    result = step03.copy()
    difference = cv2.absdiff(step03, cleaned).max(axis=2)
    protected = np.where(difference >= 24, 255, 0).astype(np.uint8)
    protected = cv2.dilate(
        protected,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    paint_mask = cv2.bitwise_and(merged_mask, cv2.bitwise_not(protected))

    red = result.copy()
    red[paint_mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(red, 0.62, result, 0.38, 0.0)
    result[paint_mask > 0] = blended[paint_mask > 0]
    contours, _ = cv2.findContours(
        merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (0, 0, 185), 2, cv2.LINE_AA)

    box_width = min(500, result.shape[1] - 18)
    x0 = max(18, result.shape[1] - box_width - 18)
    x1 = result.shape[1] - 18
    overlay = result.copy()
    cv2.rectangle(overlay, (x0, 18), (x1, 87), (255, 255, 255), -1)
    cv2.rectangle(overlay, (x0, 18), (x1, 87), (40, 40, 40), 1)
    cv2.addWeighted(overlay, 0.92, result, 0.08, 0.0, result)
    cv2.rectangle(result, (x0 + 15, 33), (x0 + 43, 49), (0, 0, 230), -1)
    cv2.putText(
        result,
        "merged large correction regions (+/-0.7 mm)",
        (x0 + 54, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"gap kernel={gap_kernel_size}px, retained={kept_count}/{merged_count}",
        (x0 + 15, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    return result


def process(
    correction_mask_path: Path,
    cleaned_path: Path,
    step03_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    correction_mask = read_image(correction_mask_path, cv2.IMREAD_GRAYSCALE)
    cleaned = read_image(cleaned_path)
    step03 = read_image(step03_path)
    if correction_mask.shape != cleaned.shape[:2] or cleaned.shape[:2] != step03.shape[:2]:
        raise ValueError("Mask, clean image, and step-03 image must have identical dimensions")

    product_mask = build_product_mask(cleaned)
    source_areas = component_areas(correction_mask)
    merged_mask, details = merge_and_filter(correction_mask, product_mask)
    kept_count = sum(1 for item in details["components"] if item["kept"])
    rendered = render_overlay(
        step03,
        cleaned,
        merged_mask,
        int(details["gap_kernel_size_px"]),
        kept_count,
        len(details["components"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "05_merged_large_regions.png"
    mask_path = output_dir / "merged_large_regions_mask.png"
    json_path = output_dir / "merge_regions.json"
    write_png(image_path, rendered)
    write_png(mask_path, merged_mask)
    payload = {
        "source_correction_mask": str(correction_mask_path.resolve()),
        "source_clean_image": str(cleaned_path.resolve()),
        "source_step03_image": str(step03_path.resolve()),
        "merge_rule": {
            "operation": "elliptical morphological closing",
            "gap_ratio_of_short_image_side": GAP_RATIO,
            "gap_kernel_size_px": details["gap_kernel_size_px"],
            "constrained_to_product_mask": True,
        },
        "small_region_rule": {
            "minimum_largest_component_ratio": MIN_LARGEST_RATIO,
            "minimum_product_area_ratio": MIN_PRODUCT_RATIO,
            "minimum_area_px": details["minimum_area_px"],
        },
        "source_component_count": len(source_areas),
        "source_component_areas_descending": source_areas,
        "merged_component_count_before_filter": len(details["components"]),
        "retained_component_count": kept_count,
        "removed_component_count": len(details["components"]) - kept_count,
        "retained_pixel_count": int(cv2.countNonZero(merged_mask)),
        "components_after_gap_merge_sorted_by_area": details["components"],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, mask_path, json_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    product_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description="Merge nearby +/-0.7 correction regions and remove small regions"
    )
    parser.add_argument(
        "--correction-mask",
        type=Path,
        default=product_root / "output" / "04_out_of_tolerance" / "outside_pm0p7_mask.png",
    )
    parser.add_argument("--clean-image", type=Path)
    parser.add_argument(
        "--step03-image",
        type=Path,
        default=product_root / "output" / "03_zero_point_selection" / "03_zero_points.png",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    args = parser.parse_args()
    if args.clean_image is None:
        args.clean_image = find_clean_image(product_root / "output" / "01_label_removal")
    return args


def main() -> None:
    args = parse_args()
    paths = process(
        args.correction_mask.resolve(),
        args.clean_image.resolve(),
        args.step03_image.resolve(),
        args.output_dir.resolve(),
    )
    for path in paths:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
