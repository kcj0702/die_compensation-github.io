"""Draw color-map deviation graphs on the true boundaries of a scanned part.

The previous laboratory version connected printed measurement points.  This
version extracts the real raster boundary of the part, including every
meaningful inner opening, and samples the scan color immediately inside each
boundary.  The source image's color bar is detected automatically and used as
a normalized +1..-1 deviation scale, so product-specific limits are not
hard-coded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


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
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and name_contains.lower() in path.stem.lower()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one matching image in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def build_product_mask(image: np.ndarray) -> np.ndarray:
    """Separate the largest real product from its white background."""
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


def contour_depth(hierarchy: np.ndarray, index: int) -> int:
    depth = 0
    parent = int(hierarchy[index][3])
    while parent >= 0:
        depth += 1
        parent = int(hierarchy[parent][3])
    return depth


def resample_closed_curve(points: np.ndarray, spacing: float) -> np.ndarray:
    points = points.astype(np.float64)
    if len(points) < 3:
        return points
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    keep = lengths > 1e-6
    points = points[keep]
    if len(points) < 3:
        return points
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    sample_count = max(8, int(np.ceil(total / spacing)))
    distances = np.linspace(0.0, total, sample_count, endpoint=False)
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(points) - 1)
    ratios = (distances - cumulative[indices]) / np.maximum(lengths[indices], 1e-6)
    return points[indices] + (following[indices] - points[indices]) * ratios[:, None]


def extract_true_contours(mask: np.ndarray) -> list[dict]:
    """Return the outer boundary and all meaningful nested hole boundaries."""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    height, width = mask.shape
    minimum_area = max(18.0, height * width * 0.000012)
    minimum_perimeter = max(18.0, min(height, width) * 0.018)
    spacing = max(3.0, min(height, width) * 0.0045)
    extracted: list[dict] = []

    for index, contour in enumerate(contours):
        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area < minimum_area or perimeter < minimum_perimeter:
            continue
        points = contour[:, 0, :].astype(np.float64)
        points = resample_closed_curve(points, spacing)
        if len(points) < 8:
            continue
        depth = contour_depth(hierarchy, index)
        extracted.append(
            {
                "source_index": index,
                "depth": depth,
                "kind": "unclassified",
                "area_px": area,
                "perimeter_px": perimeter,
                "points": points,
            }
        )

    outer_areas = [item["area_px"] for item in extracted if item["depth"] == 0]
    largest_outer_area = max(outer_areas, default=1.0)
    inner_area_threshold = max(500.0, largest_outer_area * 0.018)
    for item in extracted:
        depth = item["depth"]
        if depth == 0:
            item["kind"] = "outer"
        elif depth % 2 == 0:
            item["kind"] = "inner_island"
        elif item["area_px"] >= inner_area_threshold:
            item["kind"] = "inner"
        else:
            item["kind"] = "hole"

    extracted.sort(key=lambda item: (item["depth"], -item["area_px"]))
    for sequence, item in enumerate(extracted, start=1):
        item["name"] = f"C{sequence}_{item['kind']}"
    return extracted


def mask_value(mask: np.ndarray, point: np.ndarray) -> int:
    height, width = mask.shape
    x = int(np.clip(round(float(point[0])), 0, width - 1))
    y = int(np.clip(round(float(point[1])), 0, height - 1))
    return int(mask[y, x] > 0)


def background_normals(points: np.ndarray, product_mask: np.ndarray) -> np.ndarray:
    """Calculate normals that point from material toward background or a hole."""
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    tangent = following - previous
    normals = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths < 1e-6] = 1.0
    normals /= lengths[:, None]

    for index, (point, normal) in enumerate(zip(points, normals)):
        plus_material = sum(
            mask_value(product_mask, point + normal * distance)
            for distance in (2.0, 4.0, 6.0)
        )
        minus_material = sum(
            mask_value(product_mask, point - normal * distance)
            for distance in (2.0, 4.0, 6.0)
        )
        if plus_material > minus_material:
            normals[index] *= -1.0
    return normals


def detect_colorbar(original: np.ndarray) -> dict:
    """Find the vertical rainbow bar at the right edge of the source image."""
    height, width = original.shape[:2]
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 80)
    search_x0 = int(round(width * 0.94))
    counts = saturated[:, search_x0:].sum(axis=0)
    best_x = search_x0 + int(np.argmax(counts))
    active_y = np.where(saturated[:, best_x])[0]
    if len(active_y) < height * 0.35:
        raise RuntimeError("The source color bar could not be detected")

    # Select the longest nearly continuous vertical run.  A one-pixel gap is
    # allowed for export artifacts without joining unrelated red annotations.
    runs: list[tuple[int, int]] = []
    start = int(active_y[0])
    previous = int(active_y[0])
    for value in active_y[1:]:
        value = int(value)
        if value - previous > 2:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    y0, y1 = max(runs, key=lambda pair: pair[1] - pair[0])
    if y1 - y0 < height * 0.30:
        raise RuntimeError("Detected color-bar run is too short")

    x0 = max(0, best_x - 2)
    x1 = min(width, best_x + 3)
    palette_bgr = np.median(original[y0 : y1 + 1, x0:x1], axis=1).astype(np.uint8)
    palette_hsv = cv2.cvtColor(palette_bgr[:, None, :], cv2.COLOR_BGR2HSV)[:, 0, :]
    normalized = np.linspace(1.0, -1.0, len(palette_bgr), dtype=np.float64)
    return {
        "x": best_x,
        "y0": y0,
        "y1": y1,
        "bgr": palette_bgr,
        "hsv": palette_hsv,
        "normalized": normalized,
    }


def sample_interior_colors(
    image: np.ndarray,
    product_mask: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample robust surface colors just inside each detected boundary."""
    height, width = product_mask.shape
    distance = cv2.distanceTransform(product_mask, cv2.DIST_L2, 5)
    colors: list[np.ndarray] = []
    sample_points: list[list[int]] = []
    ray_length = max(7, int(round(min(height, width) * 0.012)))

    for point, outward in zip(points, normals):
        candidates: list[tuple[float, int, int]] = []
        for offset in np.linspace(2.0, float(ray_length), ray_length * 2):
            candidate = point - outward * offset
            x = int(np.clip(round(float(candidate[0])), 0, width - 1))
            y = int(np.clip(round(float(candidate[1])), 0, height - 1))
            if product_mask[y, x] > 0:
                candidates.append((float(distance[y, x]), x, y))
        if candidates:
            _, x, y = max(candidates, key=lambda item: item[0])
        else:
            x = int(np.clip(round(float(point[0])), 0, width - 1))
            y = int(np.clip(round(float(point[1])), 0, height - 1))

        # Use the median of material pixels in a compact patch so a single
        # dark CAD edge does not determine the contour's map color.
        px0, px1 = max(0, x - 1), min(width, x + 2)
        py0, py1 = max(0, y - 1), min(height, y + 2)
        patch = image[py0:py1, px0:px1]
        valid = product_mask[py0:py1, px0:px1] > 0
        color = np.median(patch[valid], axis=0).astype(np.uint8)
        colors.append(color)
        sample_points.append([x, y])

    return np.asarray(colors, dtype=np.uint8), np.asarray(sample_points, dtype=np.int32)


def map_colors_to_values(colors: np.ndarray, colorbar: dict) -> tuple[np.ndarray, np.ndarray]:
    sample_hsv = cv2.cvtColor(colors[:, None, :], cv2.COLOR_BGR2HSV)[:, 0, :]
    palette_hsv = colorbar["hsv"].astype(np.float64)
    values = np.full(len(colors), np.nan, dtype=np.float64)
    valid = (sample_hsv[:, 1] >= 38) & (sample_hsv[:, 2] >= 35)

    for index in np.where(valid)[0]:
        hue = float(sample_hsv[index, 0])
        hue_delta = np.abs(palette_hsv[:, 0] - hue)
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
        saturation_delta = np.abs(
            palette_hsv[:, 1] - float(sample_hsv[index, 1])
        )
        value_delta = np.abs(
            palette_hsv[:, 2] - float(sample_hsv[index, 2])
        )
        score = hue_delta + saturation_delta * 0.035 + value_delta * 0.018
        palette_index = int(np.argmin(score))
        values[index] = float(colorbar["normalized"][palette_index])

    return values, valid


def fill_out_of_range_values(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Give gray out-of-range runs the sign of their nearest colored neighbor."""
    filled = values.copy()
    valid_indices = np.where(valid & np.isfinite(values))[0]
    if len(valid_indices) == 0:
        return np.ones_like(values) * 1.05
    count = len(values)
    for index in np.where(~valid | ~np.isfinite(values))[0]:
        circular_distance = np.minimum(
            np.abs(valid_indices - index), count - np.abs(valid_indices - index)
        )
        nearest = int(valid_indices[int(np.argmin(circular_distance))])
        sign = 1.0 if values[nearest] >= 0.0 else -1.0
        filled[index] = sign * 1.05
    return filled


def circular_smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 9:
        return values.copy()
    weights = np.asarray(
        [1.0, 4.0, 10.0, 16.0, 19.0, 16.0, 10.0, 4.0, 1.0],
        dtype=np.float64,
    ) / 81.0
    padded = np.concatenate((values[-4:], values, values[:4]))
    return np.convolve(padded, weights, mode="valid")


def draw_legend(image: np.ndarray) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (18, 18), (390, 108), (255, 255, 255), -1)
    cv2.rectangle(overlay, (18, 18), (390, 108), (40, 40, 40), 1)
    cv2.addWeighted(overlay, 0.91, image, 0.09, 0.0, image)
    cv2.putText(image, "True-boundary color-map graph", (32, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(image, (34, 67), (70, 67), (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(image, "actual contour", (80, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(image, (210, 67), (246, 67), (0, 220, 255), 3, cv2.LINE_AA)
    cv2.putText(image, "color graph", (256, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(image, "gray scan color = outside displayed range", (32, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (70, 70, 70), 1, cv2.LINE_AA)


def render_graph(
    image: np.ndarray,
    product_mask: np.ndarray,
    contours: list[dict],
    colorbar: dict,
) -> tuple[np.ndarray, list[dict]]:
    canvas = image.copy()
    height, width = image.shape[:2]
    maximum_offset = max(16.0, min(height, width) * 0.035)
    output_contours: list[dict] = []

    for contour in contours:
        points = contour["points"]
        normals = background_normals(points, product_mask)
        colors, sample_points = sample_interior_colors(
            image, product_mask, points, normals
        )
        raw_values, valid = map_colors_to_values(colors, colorbar)
        filled_values = fill_out_of_range_values(raw_values, valid)
        values = circular_smooth(filled_values)
        if contour["kind"] == "outer":
            contour_offset = maximum_offset
        else:
            equivalent_radius = float(np.sqrt(contour["area_px"] / np.pi))
            contour_offset = min(maximum_offset, max(2.5, equivalent_radius * 0.48))
        graph_points = points + normals * values[:, None] * contour_offset
        contour_pixels = np.rint(points).astype(np.int32)
        graph_pixels = np.rint(graph_points).astype(np.int32)

        cv2.polylines(canvas, [contour_pixels.reshape(-1, 1, 2)], True, (15, 15, 15), 4, cv2.LINE_AA)
        cv2.polylines(canvas, [contour_pixels.reshape(-1, 1, 2)], True, (245, 245, 245), 1, cv2.LINE_AA)

        spoke_step = max(1, len(points) // 80)
        for index in range(0, len(points), spoke_step):
            cv2.line(canvas, tuple(contour_pixels[index]), tuple(graph_pixels[index]), (90, 90, 90), 1, cv2.LINE_AA)
        for index in range(len(points)):
            following = (index + 1) % len(points)
            color = tuple(int(channel) for channel in colors[index])
            cv2.line(canvas, tuple(graph_pixels[index]), tuple(graph_pixels[following]), color, 3, cv2.LINE_AA)

        output_contours.append(
            {
                "name": contour["name"],
                "kind": contour["kind"],
                "depth": int(contour["depth"]),
                "closed": True,
                "area_px": round(float(contour["area_px"]), 3),
                "perimeter_px": round(float(contour["perimeter_px"]), 3),
                "samples": [
                    {
                        "contour_point": [round(float(point[0]), 4), round(float(point[1]), 4)],
                        "sample_point": [int(sample[0]), int(sample[1])],
                        "color_bgr": [int(channel) for channel in color],
                        "raw_normalized_value": None if not np.isfinite(raw) else round(float(raw), 6),
                        "normalized_value": round(float(value), 6),
                        "zero_eligible": bool(is_valid),
                        "graph_point": [round(float(graph[0]), 4), round(float(graph[1]), 4)],
                    }
                    for point, sample, color, raw, value, is_valid, graph in zip(
                        points, sample_points, colors, raw_values, values, valid, graph_points
                    )
                ],
            }
        )

    draw_legend(canvas)
    return canvas, output_contours


def process(clean_path: Path, original_path: Path, output_dir: Path) -> tuple[Path, Path]:
    clean = read_image(clean_path)
    original = read_image(original_path)
    if clean.shape[:2] != original.shape[:2]:
        raise ValueError("Clean and original images must have the same dimensions")
    product_mask = build_product_mask(clean)
    contours = extract_true_contours(product_mask)
    if not contours:
        raise RuntimeError("No true product contour was extracted")
    colorbar = detect_colorbar(original)
    rendered, contour_payload = render_graph(clean, product_mask, contours, colorbar)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "02_contour_graph.png"
    json_path = output_dir / "contour_graph.json"
    write_png(image_path, rendered)
    payload = {
        "source_clean_image": str(clean_path.resolve()),
        "source_original_image": str(original_path.resolve()),
        "contour_source": "largest true product raster mask",
        "value_source": "automatically detected source color bar",
        "value_unit": "normalized color-map range",
        "value_range": {"top": 1.0, "zero": 0.0, "bottom": -1.0},
        "gray_rule": "outside displayed range; sign inferred only for graph continuity; excluded from zero selection",
        "colorbar": {"x": int(colorbar["x"]), "y0": int(colorbar["y0"]), "y1": int(colorbar["y1"])},
        "contour_count": len(contour_payload),
        "contours": contour_payload,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, json_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    product_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Draw true-boundary color-map graphs")
    parser.add_argument("--clean-image", type=Path)
    parser.add_argument("--original-image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    args = parser.parse_args()
    if args.clean_image is None:
        args.clean_image = find_single_image(product_root / "output" / "01_label_removal", "4_labels_points_inpainted")
    if args.original_image is None:
        args.original_image = find_single_image(product_root / "input")
    return args


def main() -> None:
    args = parse_args()
    image_path, json_path = process(args.clean_image.resolve(), args.original_image.resolve(), args.output_dir.resolve())
    print(f"Created: {image_path}")
    print(f"Created: {json_path}")


if __name__ == "__main__":
    main()
