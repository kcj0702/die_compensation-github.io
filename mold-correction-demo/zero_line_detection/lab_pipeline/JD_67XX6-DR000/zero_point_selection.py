"""Select JD_67XX zero points with the my_lab printed-label mechanism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ZERO_EPSILON = 1e-9


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


def interpolate_zero(first: dict, second: dict) -> dict:
    first_value = float(first["value_mm"])
    second_value = float(second["value_mm"])
    ratio = -first_value / (second_value - first_value)
    first_point = np.asarray(first["contour_point"], dtype=np.float64)
    second_point = np.asarray(second["contour_point"], dtype=np.float64)
    point = first_point + (second_point - first_point) * ratio
    return {
        "type": "sign_change_interpolation",
        "point": [round(float(point[0]), 5), round(float(point[1]), 5)],
        "between_values_mm": [first_value, second_value],
        "interpolation_ratio": round(float(ratio), 8),
    }


def select_contour_zero_points(contour: dict) -> list[dict]:
    samples = contour["samples"]
    if not samples:
        return []

    # This intentionally matches my_lab/zero_point_selection: a printed zero
    # is selected directly, while opposite-sign adjacent labels are linearly
    # interpolated on the closed scan-point loop.
    zero_points: list[dict] = []
    for index, sample in enumerate(samples):
        value = float(sample["value_mm"])
        if abs(value) <= ZERO_EPSILON:
            zero_points.append(
                {
                    "type": "exact_label_zero",
                    "point": [float(value) for value in sample["contour_point"]],
                    "sample_index": index,
                    "value_mm": value,
                }
            )

    count = len(samples)
    for first_index in range(count):
        second_index = (first_index + 1) % count
        first = samples[first_index]
        second = samples[second_index]
        first_value = float(first["value_mm"])
        second_value = float(second["value_mm"])
        if first_value * second_value >= 0.0:
            continue
        item = interpolate_zero(first, second)
        item["between_sample_indices"] = [first_index, second_index]
        zero_points.append(item)

    return zero_points


def draw_zero_points(image: np.ndarray, contours: list[dict]) -> np.ndarray:
    canvas = image.copy()
    sequence = 1
    for contour in contours:
        for zero_point in contour["zero_points"]:
            point = tuple(np.rint(zero_point["point"]).astype(np.int32).tolist())
            exact = zero_point["type"] == "exact_label_zero"
            color = (40, 220, 40) if exact else (0, 230, 255)
            cv2.circle(canvas, point, 9, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.circle(canvas, point, 6, color, -1, cv2.LINE_AA)
            label = f"Z{sequence}"
            cv2.putText(canvas, label, (point[0] + 9, point[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (point[0] + 9, point[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            sequence += 1

    overlay = canvas.copy()
    cv2.rectangle(overlay, (405, 18), (746, 89), (255, 255, 255), -1)
    cv2.rectangle(overlay, (405, 18), (746, 89), (40, 40, 40), 1)
    cv2.addWeighted(overlay, 0.91, canvas, 0.09, 0.0, canvas)
    cv2.circle(canvas, (425, 42), 6, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "sign-change zero", (439, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.circle(canvas, (606, 42), 6, (40, 220, 40), -1, cv2.LINE_AA)
    cv2.putText(canvas, "printed 0", (620, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"zero points: {sequence - 1}", (422, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def process(graph_json: Path, graph_image: Path, output_dir: Path) -> tuple[Path, Path]:
    payload = json.loads(graph_json.read_text(encoding="utf-8"))
    image = read_image(graph_image)
    output_contours: list[dict] = []
    for contour in payload["loops"]:
        zero_points = select_contour_zero_points(contour)
        output_contours.append(
            {
                "name": contour["name"],
                "kind": "mylab_scan_loop",
                "closed": bool(contour.get("closed", True)),
                "zero_point_count": len(zero_points),
                "zero_points": zero_points,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "03_zero_points.png"
    json_path = output_dir / "zero_points.json"
    write_png(image_path, draw_zero_points(image, output_contours))
    result = {
        "source_deviation_graph": str(graph_json.resolve()),
        "source_graph_image": str(graph_image.resolve()),
        "value_source": payload.get("value_source", "printed_label_ocr"),
        "value_unit": payload.get("value_unit", "mm"),
        "selection_rule": {
            "mechanism": "same as my_lab/zero_point_selection/select_zero_points.py",
            "exact_zero": "printed label value equals 0",
            "interpolated_zero": "adjacent printed label values have opposite signs",
            "unrelated_geometric_crossings_excluded": True,
            "closed_last_to_first_interval_included": True,
        },
        "zero_point_count": sum(item["zero_point_count"] for item in output_contours),
        "contours": output_contours,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, json_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Select JD_67XX zero points with the my_lab printed-label mechanism")
    parser.add_argument("--graph-json", type=Path, default=script_dir / "mylab_deviation_graph.json")
    parser.add_argument("--graph-image", type=Path, default=script_dir / "mylab_deviation_graph.png")
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path, json_path = process(args.graph_json.resolve(), args.graph_image.resolve(), args.output_dir.resolve())
    print(f"Created: {image_path}")
    print(f"Created: {json_path}")


if __name__ == "__main__":
    main()
