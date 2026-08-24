"""Select zero-deviation points from contour deviation graphs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent
CONTOUR_GRAPH_OUTPUT = LAB_ROOT / "contour_graph" / "output"
OUTPUT_ROOT = HERE / "output"
ZERO_EPSILON = 1e-9


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"PNG 인코딩 실패: {path}")
    encoded.tofile(path)


def interpolate_zero(
    first_point: np.ndarray,
    first_value: float,
    second_point: np.ndarray,
    second_value: float,
) -> tuple[np.ndarray, float]:
    """Linearly locate value=0 between two opposite-sign samples."""
    ratio = -first_value / (second_value - first_value)
    point = first_point + (second_point - first_point) * ratio
    return point, float(ratio)


def select_loop_zero_points(loop: dict) -> list[dict]:
    samples = loop["samples"]
    if not samples:
        return []

    zero_points: list[dict] = []

    # A label whose printed value is exactly 0 is itself a zero point.  It is
    # kept even when the graph only touches the contour and turns back.
    for index, sample in enumerate(samples):
        value = float(sample["value_mm"])
        if abs(value) <= ZERO_EPSILON:
            point = [float(coordinate) for coordinate in sample["contour_point"]]
            zero_points.append(
                {
                    "type": "exact_label_zero",
                    "point": point,
                    "sample_index": index,
                    "value_mm": value,
                }
            )

    # Each loop is closed, so the last-to-first interval is included.  Strict
    # sign changes only are considered here; intervals touching an exact zero
    # have already been represented by the exact sample above.
    count = len(samples)
    for first_index in range(count):
        second_index = (first_index + 1) % count
        first = samples[first_index]
        second = samples[second_index]
        first_value = float(first["value_mm"])
        second_value = float(second["value_mm"])
        if first_value * second_value >= 0.0:
            continue

        first_point = np.asarray(first["contour_point"], dtype=np.float64)
        second_point = np.asarray(second["contour_point"], dtype=np.float64)
        point, ratio = interpolate_zero(
            first_point, first_value, second_point, second_value
        )
        zero_points.append(
            {
                "type": "sign_change_interpolation",
                "point": [round(float(point[0]), 5), round(float(point[1]), 5)],
                "between_sample_indices": [first_index, second_index],
                "between_values_mm": [first_value, second_value],
                "interpolation_ratio": round(ratio, 8),
            }
        )

    return zero_points


def draw_zero_points(image: np.ndarray, loops: list[dict]) -> np.ndarray:
    canvas = image.copy()
    sequence = 1

    for loop in loops:
        for zero_point in loop["zero_points"]:
            point = tuple(
                np.rint(np.asarray(zero_point["point"], dtype=np.float64))
                .astype(np.int32)
                .tolist()
            )
            exact = zero_point["type"] == "exact_label_zero"
            color = (40, 220, 40) if exact else (0, 230, 255)
            cv2.circle(canvas, point, 9, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.circle(canvas, point, 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"Z{sequence}",
                (point[0] + 9, point[1] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Z{sequence}",
                (point[0] + 9, point[1] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            sequence += 1

    overlay = canvas.copy()
    cv2.rectangle(overlay, (18, 122), (330, 190), (255, 255, 255), -1)
    cv2.rectangle(overlay, (18, 122), (330, 190), (40, 40, 40), 1)
    cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0.0, canvas)
    cv2.circle(canvas, (36, 145), 6, (0, 230, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "sign-change zero",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, (190, 145), 6, (40, 220, 40), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "printed 0",
        (204, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    total = sequence - 1
    cv2.putText(
        canvas,
        f"zero points: {total}",
        (32, 177),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas


def process_graph(graph_json_path: Path) -> Path:
    payload = json.loads(graph_json_path.read_text(encoding="utf-8"))
    graph_image_path = graph_json_path.parent / "01_deviation_graph.png"
    image = read_image(graph_image_path)

    output_loops: list[dict] = []
    for loop in payload["loops"]:
        zero_points = select_loop_zero_points(loop)
        output_loops.append(
            {
                "name": loop["name"],
                "closed": bool(loop.get("closed", True)),
                "zero_point_count": len(zero_points),
                "zero_points": zero_points,
            }
        )

    output_payload = {
        "source_deviation_graph": str(graph_json_path.resolve()),
        "source_graph_image": str(graph_image_path.resolve()),
        "value_source": payload.get("value_source", "printed_label_ocr"),
        "value_unit": payload.get("value_unit", "mm"),
        "selection_rule": {
            "exact_zero": "printed label value equals 0",
            "interpolated_zero": "adjacent values have opposite signs",
            "unrelated_geometric_crossings_excluded": True,
            "closed_last_to_first_interval_included": True,
        },
        "zero_point_count": sum(loop["zero_point_count"] for loop in output_loops),
        "loops": output_loops,
    }

    destination = OUTPUT_ROOT / graph_json_path.parent.name
    destination.mkdir(parents=True, exist_ok=True)
    image_output = destination / "01_zero_points.png"
    json_output = destination / "zero_points.json"
    write_png(image_output, draw_zero_points(image, output_loops))
    json_output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return image_output


def run_all() -> list[Path]:
    graph_files = sorted(CONTOUR_GRAPH_OUTPUT.glob("*/deviation_graph.json"))
    if not graph_files:
        raise FileNotFoundError(
            f"contour_graph 결과를 찾을 수 없습니다: {CONTOUR_GRAPH_OUTPUT}"
        )
    return [process_graph(path) for path in graph_files]


if __name__ == "__main__":
    for output in run_all():
        print(f"[완료] {output}")
