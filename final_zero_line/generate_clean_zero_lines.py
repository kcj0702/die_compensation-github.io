"""Generate portable, clutter-free zero-line deliverables for JD64 and JD71."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


PRODUCTS = (
    "JD_64XX2-DR000 3D 스캔",
    "JD_71XX2-DR000 3D 스캔",
)
OUTLINE_COLOR_BGR = (255, 255, 255)
ZERO_LINE_COLOR_BGR = (255, 255, 0)
OUTLINE_WIDTH_PX = 7
ZERO_LINE_WIDTH_PX = 4


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def only_png(directory: Path) -> Path:
    paths = sorted(directory.glob("*.png"))
    if len(paths) != 1:
        raise ValueError(f"Expected one PNG in {directory}, found {len(paths)}")
    return paths[0]


def extract_routes(payload: dict) -> list[dict]:
    routes: list[dict] = []
    for region in payload.get("regions", []):
        closure = region.get("closure_validation") or {}
        route = closure.get("route") or {}
        points = route.get("path_points") or []
        if len(points) < 2:
            continue
        clean_points = [[int(round(point[0])), int(round(point[1]))] for point in points]
        selected = region.get("selected_zero_points") or []
        routes.append(
            {
                "region_label": region.get("region_label"),
                "zero_points": [
                    {
                        "label": item.get("label"),
                        "point": [
                            float(item["point"][0]),
                            float(item["point"][1]),
                        ],
                    }
                    for item in selected
                    if item.get("point") and len(item["point"]) >= 2
                ],
                "polyline": clean_points,
                "path_length_pixels": float(route.get("path_length_pixels", 0.0)),
                "path_bend_count": int(route.get("path_bend_count", max(0, len(clean_points) - 2))),
            }
        )
    if not routes:
        raise ValueError("No validated zero-line route found")
    return routes


def draw_routes(image: np.ndarray, routes: list[dict]) -> np.ndarray:
    result = image.copy()
    for route in routes:
        points = np.asarray(route["polyline"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            result,
            [points],
            False,
            OUTLINE_COLOR_BGR,
            OUTLINE_WIDTH_PX,
            cv2.LINE_AA,
        )
        cv2.polylines(
            result,
            [points],
            False,
            ZERO_LINE_COLOR_BGR,
            ZERO_LINE_WIDTH_PX,
            cv2.LINE_AA,
        )
    return result


def relative_to_product(path: Path, product_dir: Path) -> str:
    return path.relative_to(product_dir).as_posix()


def generate_product(package_dir: Path, product_name: str) -> None:
    product_dir = package_dir / product_name
    output_root = product_dir / "output"
    clean_dir = output_root / "07_clean_zero_line"
    clean_dir.mkdir(parents=True, exist_ok=True)

    original_path = only_png(product_dir / "input")
    label_removed_candidates = sorted(
        (output_root / "01_label_removal").glob("*_4_labels_points_inpainted.png")
    )
    if len(label_removed_candidates) != 1:
        raise ValueError(
            f"Expected one label-removed PNG for {product_name}, "
            f"found {len(label_removed_candidates)}"
        )
    label_removed_path = label_removed_candidates[0]
    source_json_path = output_root / "06_nearest_zero_points" / "nearest_zero_points.json"
    source_payload = json.loads(source_json_path.read_text(encoding="utf-8"))
    routes = extract_routes(source_payload)

    original = read_image(original_path)
    label_removed = read_image(label_removed_path)
    if original.shape != label_removed.shape:
        raise ValueError(
            f"Image size mismatch for {product_name}: "
            f"original={original.shape}, label_removed={label_removed.shape}"
        )

    label_removed_output = clean_dir / "01_zero_line_on_label_removed.png"
    original_output = clean_dir / "02_zero_line_on_original.png"
    json_output = clean_dir / "clean_zero_lines.json"
    write_png(label_removed_output, draw_routes(label_removed, routes))
    write_png(original_output, draw_routes(original, routes))

    height, width = original.shape[:2]
    deliverable = {
        "format_version": 1,
        "product": product_name,
        "coordinate_system": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "unit": "pixel",
            "image_width": int(width),
            "image_height": int(height),
        },
        "source": {
            "nearest_zero_points_json": relative_to_product(source_json_path, product_dir),
            "label_removed_image": relative_to_product(label_removed_path, product_dir),
            "original_image": relative_to_product(original_path, product_dir),
        },
        "outputs": {
            "zero_line_on_label_removed": label_removed_output.name,
            "zero_line_on_original": original_output.name,
        },
        "rendering": {
            "outline_color_bgr": list(OUTLINE_COLOR_BGR),
            "outline_width_px": OUTLINE_WIDTH_PX,
            "zero_line_color_bgr": list(ZERO_LINE_COLOR_BGR),
            "zero_line_width_px": ZERO_LINE_WIDTH_PX,
            "anti_alias": True,
        },
        "route_count": len(routes),
        "routes": routes,
    }
    json_output.write_text(
        json.dumps(deliverable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Created: {label_removed_output}")
    print(f"Created: {original_output}")
    print(f"Created: {json_output}")


def main() -> None:
    package_dir = Path(__file__).resolve().parent
    for product_name in PRODUCTS:
        generate_product(package_dir, product_name)


if __name__ == "__main__":
    main()
