"""Draw normalized deviation graphs along scan-point contour loops."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from label_value_reader import LabelReading, match_readings_to_points, read_labels


HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent
CONTOUR_OUTPUT = LAB_ROOT / "scan_point_contour" / "output"
CLEAN_IMAGE_DIR = (
    LAB_ROOT / "label_removal" / "output" / "2_labels_inpainted"
)
OUTPUT_ROOT = HERE / "output"

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


def outward_normals(points: np.ndarray) -> np.ndarray:
    """Calculate outward unit normals using the closed polygon itself.

    The two perpendicular directions at each contour point are probed with
    ``pointPolygonTest``.  The direction whose probe has the smaller signed
    distance (outside is negative, inside is positive) is the outward one.
    This avoids the wrong flips that a centroid-based guess can produce on
    long, asymmetric, or concave parts.
    """
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    tangent = following - previous
    normals = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths < 1e-6] = 1.0
    normals /= lengths[:, None]

    polygon = points.astype(np.float32).reshape(-1, 1, 2)
    # A probe should clear the antialiased contour line but remain local to
    # the scan point.  Scale it gently for images of different resolutions.
    span = np.ptp(points, axis=0)
    probe_distance = max(3.0, min(float(span[0]), float(span[1])) * 0.008)

    for index, (point, normal) in enumerate(zip(points, normals)):
        plus = point + normal * probe_distance
        minus = point - normal * probe_distance
        plus_distance = cv2.pointPolygonTest(
            polygon, (float(plus[0]), float(plus[1])), True
        )
        minus_distance = cv2.pointPolygonTest(
            polygon, (float(minus[0]), float(minus[1])), True
        )

        # pointPolygonTest: positive=inside, negative=outside.  Therefore the
        # candidate with the smaller signed distance is the outward direction.
        if plus_distance > minus_distance:
            normals[index] *= -1.0
    return normals


def sign_color(value: float) -> tuple[int, int, int]:
    if value > 0.035:
        return (0, 45, 245)  # positive: red
    if value < -0.035:
        return (245, 90, 20)  # negative: blue
    return (0, 200, 70)  # near zero: green


def draw_legend(image: np.ndarray) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (18, 18), (318, 112), (255, 255, 255), -1)
    cv2.rectangle(overlay, (18, 18), (318, 112), (40, 40, 40), 1)
    cv2.addWeighted(overlay, 0.88, image, 0.12, 0.0, image)
    cv2.putText(
        image,
        "Printed-label deviation graph (mm)",
        (32, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    entries = (
        ("+ positive", (0, 45, 245)),
        ("0", (0, 200, 70)),
        ("- negative", (245, 90, 20)),
    )
    for index, (label, color) in enumerate(entries):
        x = 32 + index * 92
        cv2.line(image, (x, 69), (x + 22, 69), color, 3, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (x, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )


def render_graph(
    clean: np.ndarray,
    loops: list[dict],
    readings_by_point: dict[tuple[int, int], LabelReading],
    pixels_per_mm: float,
) -> tuple[np.ndarray, list[dict]]:
    canvas = clean.copy()
    output_loops: list[dict] = []

    for loop in loops:
        points = np.asarray(loop["connection_path"], dtype=np.float32)
        if len(points) < 3:
            continue
        normals = outward_normals(points)
        readings = [
            readings_by_point.get(tuple(map(int, point))) for point in points
        ]
        values = np.asarray(
            [reading.value if reading is not None else 0.0 for reading in readings],
            dtype=np.float32,
        )
        graph_points = points + normals * values[:, None] * pixels_per_mm
        graph_pixels = np.rint(graph_points).astype(np.int32)
        base_pixels = np.rint(points).astype(np.int32)

        # The measured contour is the zero baseline.
        cv2.polylines(
            canvas,
            [base_pixels.reshape(-1, 1, 2)],
            True,
            (25, 25, 25),
            5,
            cv2.LINE_AA,
        )
        cv2.polylines(
            canvas,
            [base_pixels.reshape(-1, 1, 2)],
            True,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        count = len(points)
        for index in range(count):
            following = (index + 1) % count
            spoke_color = sign_color(float(values[index]))
            cv2.line(
                canvas,
                tuple(base_pixels[index]),
                tuple(graph_pixels[index]),
                spoke_color,
                1,
                cv2.LINE_AA,
            )
            segment_value = float((values[index] + values[following]) / 2.0)
            cv2.line(
                canvas,
                tuple(graph_pixels[index]),
                tuple(graph_pixels[following]),
                sign_color(segment_value),
                3,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                tuple(graph_pixels[index]),
                3,
                spoke_color,
                -1,
                cv2.LINE_AA,
            )

        output_loops.append(
            {
                "name": loop["name"],
                "closed": True,
                "samples": [
                    {
                        "contour_point": [int(point[0]), int(point[1])],
                        "value_mm": round(float(value), 5),
                        "ocr_confidence": round(
                            float(reading.ocr_confidence), 5
                        ) if reading is not None else 0.0,
                        "label_box": list(reading.box) if reading is not None else None,
                        "leader_traced": bool(
                            reading is not None and reading.traced_point is not None
                        ),
                        "graph_point": [int(graph[0]), int(graph[1])],
                    }
                    for point, value, reading, graph in zip(
                        base_pixels, values, readings, graph_pixels
                    )
                ],
            }
        )

    draw_legend(canvas)
    return canvas, output_loops


def find_clean_image(stem: str) -> Path:
    candidates = sorted(
        path
        for path in CLEAN_IMAGE_DIR.iterdir()
        if path.is_file() and stem.lower() in path.stem.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"라벨 제거 이미지를 찾을 수 없습니다: {stem}")
    return candidates[0]


def process_contour(contour_json: Path) -> Path:
    payload = json.loads(contour_json.read_text(encoding="utf-8"))
    stem = contour_json.parent.name
    clean_path = find_clean_image(stem)
    clean = read_image(clean_path)
    original_path = Path(payload["source"])
    original = read_image(original_path)
    contour_points: list[tuple[int, int]] = []
    for loop in payload["loops"]:
        for point in loop["connection_path"]:
            converted = tuple(map(int, point))
            if converted not in contour_points:
                contour_points.append(converted)
    label_readings = read_labels(original)
    readings_by_point = match_readings_to_points(
        label_readings, contour_points, original.shape
    )
    if len(readings_by_point) != len(contour_points):
        missing = len(contour_points) - len(readings_by_point)
        raise RuntimeError(f"라벨 숫자와 연결되지 않은 윤곽점이 {missing}개 있습니다: {stem}")

    maximum_abs_value = max(
        max(abs(reading.value) for reading in readings_by_point.values()), 0.1
    )
    maximum_offset = max(18.0, min(clean.shape[:2]) * 0.035)
    pixels_per_mm = maximum_offset / maximum_abs_value
    rendered, loops = render_graph(
        clean, payload["loops"], readings_by_point, pixels_per_mm
    )

    output_dir = OUTPUT_ROOT / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "01_deviation_graph.png"
    write_png(graph_path, rendered)
    result = {
        "source_contour": str(contour_json),
        "source_clean_image": str(clean_path),
        "source_original_image": str(original_path),
        "value_source": "printed_label_ocr",
        "value_unit": "mm",
        "pixels_per_mm": pixels_per_mm,
        "matched_label_count": len(readings_by_point),
        "zero_points_calculated": False,
        "loops": loops,
    }
    (output_dir / "deviation_graph.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return graph_path


def run_all() -> list[Path]:
    contour_files = sorted(CONTOUR_OUTPUT.glob("*/scan_point_loops.json"))
    if not contour_files:
        raise FileNotFoundError(f"윤곽선 JSON이 없습니다: {CONTOUR_OUTPUT}")
    return [process_contour(path) for path in contour_files]


if __name__ == "__main__":
    for result_path in run_all():
        print(f"[완료] {result_path}")
