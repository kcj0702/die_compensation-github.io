"""Overlay the JD_67XX branch-defined zero region on the label-only-clean scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ZERO_REGION_COLOR = (255, 80, 200)
ZERO_REGION_ALPHA = 0.46


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(path)


def render(base: np.ndarray, zero_region_mask: np.ndarray) -> np.ndarray:
    selected = zero_region_mask > 127
    canvas = base.copy()
    color_layer = canvas.copy()
    color_layer[selected] = ZERO_REGION_COLOR
    blended = cv2.addWeighted(
        color_layer,
        ZERO_REGION_ALPHA,
        canvas,
        1.0 - ZERO_REGION_ALPHA,
        0.0,
    )
    canvas[selected] = blended[selected]

    contours, _ = cv2.findContours(
        np.where(selected, 255, 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.drawContours(canvas, contours, -1, (185, 30, 145), 2, cv2.LINE_AA)

    width = canvas.shape[1]
    x0 = max(1030, width - 480)
    x1 = width - 18
    cv2.rectangle(canvas, (x0, 18), (x1, 76), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0, 18), (x1, 76), (40, 40, 40), 1)
    cv2.rectangle(canvas, (x0 + 16, 34), (x0 + 46, 53), ZERO_REGION_COLOR, -1)
    cv2.putText(
        canvas,
        "branch-defined zero region",
        (x0 + 59, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "base: labels removed only",
        (x0 + 16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    return canvas


def process(
    base_path: Path,
    zero_region_mask_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    base = read_image(base_path)
    zero_region_mask = read_image(zero_region_mask_path, cv2.IMREAD_GRAYSCALE)
    if base.shape[:2] != zero_region_mask.shape:
        raise ValueError("Base image and zero-region mask sizes differ")

    result = render(base, zero_region_mask)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "08_zero_region_on_label_removed_scan.png"
    json_path = output_dir / "zero_region_overlay.json"
    write_png(image_path, result)
    payload = {
        "product": "JD_67XX6-DR000",
        "source_label_only_removed_scan": str(base_path.resolve()),
        "source_branch_defined_zero_region": str(zero_region_mask_path.resolve()),
        "zero_region_pixel_count": int(cv2.countNonZero(zero_region_mask)),
        "overlay": {
            "color_bgr": list(ZERO_REGION_COLOR),
            "alpha": ZERO_REGION_ALPHA,
            "branch_centerlines_drawn": False,
            "zero_point_markers_drawn": False,
            "correction_regions_drawn": False,
            "contour_graphs_drawn": False,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, json_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=(
            output_root
            / "01_label_removal"
            / "JD_67XX6-DR000 3D 스캔_2_labels_inpainted.png"
        ),
    )
    parser.add_argument(
        "--zero-region-mask",
        type=Path,
        default=(
            output_root
            / "07_zero_line_branch_expansion"
            / "branch_expanded_area_mask.png"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in process(
        args.base.resolve(),
        args.zero_region_mask.resolve(),
        args.output_dir.resolve(),
    ):
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
