"""Side-by-side reviewer for correction-sheet zero-line ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_zero_line import read_image


WINDOW = "Zero-line ground truth | sheet LEFT / scan RIGHT"


def fit(image: np.ndarray, max_width: int = 860, max_height: int = 780) -> tuple[np.ndarray, float]:
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실제 시트를 참고해 스캔 좌표에 제로라인 정답 표시")
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["line", "points"], default="line")
    return parser.parse_args()


def load_existing(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if not path.exists():
        return "line", []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("kind", "line")), list(payload.get("lines", []))


def main() -> int:
    args = parse_args()
    scan_path, sheet_path, output_path = args.scan.resolve(), args.sheet.resolve(), args.output.resolve()
    scan, sheet = read_image(scan_path), read_image(sheet_path)
    sheet_view, _sheet_scale = fit(sheet)
    scan_view, scan_scale = fit(scan)
    divider = 8
    left_width = sheet_view.shape[1]
    height = max(sheet_view.shape[0], scan_view.shape[0])
    kind, saved_lines = load_existing(output_path)
    if not output_path.exists():
        kind = args.kind
    lines: list[list[list[float]]] = [list(item.get("points", [])) for item in saved_lines]
    current: list[list[float]] = []

    def compose() -> np.ndarray:
        canvas = np.full((height + 72, left_width + divider + scan_view.shape[1], 3), 242, np.uint8)
        canvas[:sheet_view.shape[0], :left_width] = sheet_view
        x0 = left_width + divider
        canvas[:scan_view.shape[0], x0:x0 + scan_view.shape[1]] = scan_view
        cv2.rectangle(canvas, (left_width, 0), (left_width + divider - 1, height), (40, 40, 40), -1)
        for line in lines + ([current] if current else []):
            points = np.asarray([[x0 + x * scan_scale, y * scan_scale] for x, y in line], np.int32)
            if len(points) == 1:
                cv2.circle(canvas, tuple(points[0]), 6, (0, 0, 255), -1, cv2.LINE_AA)
            elif len(points) > 1:
                cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, (0, 0, 255), 3, cv2.LINE_AA)
                for point in points:
                    cv2.circle(canvas, tuple(point), 4, (0, 255, 255), -1, cv2.LINE_AA)
        message = f"mode={kind} | left=add right=delete-nearest N=finish U=undo C=clear M=mode S=save Q=quit"
        cv2.putText(canvas, message, (12, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"saved lines={len(lines)}  current points={len(current)}", (12, height + 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 1, cv2.LINE_AA)
        return canvas

    def finish_current() -> None:
        nonlocal current
        if current:
            lines.append(current)
            current = []

    def mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        x0 = left_width + divider
        if event == cv2.EVENT_RBUTTONDOWN:
            if kind == "points" and x0 <= x < x0 + scan_view.shape[1] and 0 <= y < scan_view.shape[0]:
                click = np.asarray([(x - x0) / scan_scale, y / scan_scale], dtype=float)
                distances = [
                    float(np.linalg.norm(np.asarray(line[0], dtype=float) - click))
                    if len(line) == 1 else float("inf")
                    for line in lines
                ]
                if distances and min(distances) <= 24.0 / scan_scale:
                    lines.pop(int(np.argmin(distances)))
            else:
                finish_current()
        elif event == cv2.EVENT_LBUTTONDOWN and x0 <= x < x0 + scan_view.shape[1] and 0 <= y < scan_view.shape[0]:
            point = [(x - x0) / scan_scale, y / scan_scale]
            if kind == "points":
                lines.append([point])
            else:
                current.append(point)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse)
    while True:
        cv2.imshow(WINDOW, compose())
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("n"):
            finish_current()
        elif key == ord("u"):
            if current:
                current.pop()
            elif lines:
                current = lines.pop()
        elif key == ord("d"):
            if current:
                current = []
            elif lines:
                lines.pop()
        elif key == ord("c"):
            current = []
            lines.clear()
        elif key == ord("m"):
            finish_current()
            kind = "points" if kind == "line" else "line"
        elif key == ord("s"):
            finish_current()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": 1,
                "scan_image": str(scan_path),
                "sheet_image": str(sheet_path),
                "kind": kind,
                "reviewed": True,
                "lines": [{"id": index, "points": points} for index, points in enumerate(lines, 1)],
            }
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"저장했습니다: {output_path}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
