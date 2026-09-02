"""Mark where a true-boundary color graph intersects or touches its baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ZERO_TOUCH_EPSILON = 0.012


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
    first_value = float(first["normalized_value"])
    second_value = float(second["normalized_value"])
    ratio = -first_value / (second_value - first_value)
    first_point = np.asarray(first["contour_point"], dtype=np.float64)
    second_point = np.asarray(second["contour_point"], dtype=np.float64)
    point = first_point + (second_point - first_point) * ratio
    return {
        "type": "graph_contour_intersection",
        "point": [round(float(point[0]), 5), round(float(point[1]), 5)],
        "between_values": [first_value, second_value],
        "interpolation_ratio": round(float(ratio), 8),
    }


def select_contour_zero_points(contour: dict) -> list[dict]:
    samples = contour["samples"]
    if not samples:
        return []
    count = len(samples)
    candidates: list[dict] = []

    # A near-zero run means that the color graph touches/runs on the contour.
    near_zero = np.asarray(
        [
            bool(sample.get("zero_eligible", False))
            and abs(float(sample["normalized_value"])) <= ZERO_TOUCH_EPSILON
            for sample in samples
        ],
        dtype=bool,
    )
    visited = np.zeros(count, dtype=bool)
    for start in range(count):
        if not near_zero[start] or visited[start]:
            continue
        run: list[int] = []
        index = start
        while near_zero[index] and not visited[index]:
            visited[index] = True
            run.append(index)
            index = (index + 1) % count
        best = min(run, key=lambda item: abs(float(samples[item]["normalized_value"])))
        candidates.append(
            {
                "type": "graph_contour_touch",
                "point": [float(value) for value in samples[best]["contour_point"]],
                "sample_index": best,
                "normalized_value": float(samples[best]["normalized_value"]),
            }
        )

    for first_index in range(count):
        second_index = (first_index + 1) % count
        first = samples[first_index]
        second = samples[second_index]
        if not first.get("zero_eligible", False) or not second.get("zero_eligible", False):
            continue
        first_value = float(first["normalized_value"])
        second_value = float(second["normalized_value"])
        if abs(first_value) <= ZERO_TOUCH_EPSILON or abs(second_value) <= ZERO_TOUCH_EPSILON:
            continue
        if first_value * second_value < 0.0:
            item = interpolate_zero(first, second)
            item["between_sample_indices"] = [first_index, second_index]
            candidates.append(item)

    # Close samples and noisy hue changes can describe the same crossing more
    # than once.  Keep only spatially distinct points on this contour.
    selected: list[dict] = []
    for candidate in candidates:
        point = np.asarray(candidate["point"], dtype=np.float64)
        if any(np.linalg.norm(point - np.asarray(item["point"])) < 7.0 for item in selected):
            continue
        selected.append(candidate)
    return selected


def draw_zero_points(image: np.ndarray, contours: list[dict]) -> np.ndarray:
    canvas = image.copy()
    sequence = 1
    for contour in contours:
        for zero_point in contour["zero_points"]:
            point = tuple(np.rint(zero_point["point"]).astype(np.int32).tolist())
            touch = zero_point["type"] == "graph_contour_touch"
            color = (40, 220, 40) if touch else (0, 220, 255)
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
    cv2.putText(canvas, "graph crosses contour", (439, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.circle(canvas, (606, 42), 6, (40, 220, 40), -1, cv2.LINE_AA)
    cv2.putText(canvas, "graph touches 0", (620, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"zero points: {sequence - 1}", (422, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def process(graph_json: Path, graph_image: Path, output_dir: Path) -> tuple[Path, Path]:
    payload = json.loads(graph_json.read_text(encoding="utf-8"))
    image = read_image(graph_image)
    output_contours: list[dict] = []
    for contour in payload["contours"]:
        # Zero-line construction uses only the product's true outer boundary.
        # Inner openings and holes keep their graphs in step 02, but they do
        # not contribute zero-point candidates in this step.
        if contour.get("kind") != "outer":
            continue
        zero_points = select_contour_zero_points(contour)
        output_contours.append(
            {
                "name": contour["name"],
                "kind": contour["kind"],
                "closed": True,
                "zero_point_count": len(zero_points),
                "zero_points": zero_points,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "03_zero_points.png"
    json_path = output_dir / "zero_points.json"
    write_png(image_path, draw_zero_points(image, output_contours))
    result = {
        "source_contour_graph": str(graph_json.resolve()),
        "source_graph_image": str(graph_image.resolve()),
        "selection_rule": {
            "intersection": "adjacent valid color-map values have opposite signs",
            "touch": f"absolute normalized color-map value <= {ZERO_TOUCH_EPSILON}",
            "gray_out_of_range_excluded": True,
            "outer_contour_only": True,
            "inner_and_hole_contours_excluded": True,
        },
        "zero_point_count": sum(item["zero_point_count"] for item in output_contours),
        "contours": output_contours,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, json_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    graph_dir = script_dir.parent / "02_contour_graph"
    parser = argparse.ArgumentParser(description="Select zero points on true-boundary color graphs")
    parser.add_argument("--graph-json", type=Path, default=graph_dir / "contour_graph.json")
    parser.add_argument("--graph-image", type=Path, default=graph_dir / "02_contour_graph.png")
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path, json_path = process(args.graph_json.resolve(), args.graph_image.resolve(), args.output_dir.resolve())
    print(f"Created: {image_path}")
    print(f"Created: {json_path}")


if __name__ == "__main__":
    main()
