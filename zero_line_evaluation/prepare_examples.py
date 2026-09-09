"""Prepare side-by-side review boards for the supplied historical examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_zero_line import REPO_ROOT, current_prediction, read_image, write_image
from auto_ground_truth import write_candidate_annotation


DEFAULT_EXAMPLES = REPO_ROOT / "경북대KDT(14기) 자료" / "3D 스캔 이미지 및 보정치 시트_예시"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "work"


def prefix(path: Path) -> str:
    name = path.stem
    for marker in (" 3D 스캔", " 보정시트", " 보정 시트"):
        if marker in name:
            return name.split(marker, 1)[0]
    return name


def fit(image: np.ndarray, max_width: int = 1000, max_height: int = 720) -> np.ndarray:
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def board(sheet: np.ndarray, prediction: np.ndarray, title: str) -> np.ndarray:
    left, right = fit(sheet), fit(prediction)
    top = 54
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1] + 12
    canvas = np.full((height + top, width, 3), 247, np.uint8)
    canvas[top:top + left.shape[0], :left.shape[1]] = left
    x = left.shape[1] + 12
    canvas[top:top + right.shape[0], x:x + right.shape[1]] = right
    cv2.rectangle(canvas, (left.shape[1], top), (x - 1, height + top), (50, 50, 50), -1)
    cv2.putText(canvas, "ACTUAL CORRECTION SHEET", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"CURRENT DETECTION - {title}", (x + 12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실제 시트/현재 검출 검토판 준비")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force-auto",
        action="store_true",
        help="기존 파일을 덮어쓰고 자동 후보를 다시 생성",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = args.examples.resolve()
    output = args.output.resolve()
    scans = {prefix(path): path for path in examples.glob("*3D 스캔.png")}
    sheets = {prefix(path): path for path in examples.glob("*보정*시트.png")}
    common = sorted(scans.keys() & sheets.keys())
    if not common:
        raise FileNotFoundError(f"스캔/보정시트 쌍을 찾지 못했습니다: {examples}")
    for key in common:
        scan_path, sheet_path = scans[key], sheets[key]
        scan, sheet = read_image(scan_path), read_image(sheet_path)
        prediction = current_prediction(scan, scan_path.name)
        item_dir = output / key
        item_dir.mkdir(parents=True, exist_ok=True)
        write_image(item_dir / "prediction_overlay.png", prediction.overlay_bgr)
        write_image(item_dir / "review_board.png", board(sheet, prediction.overlay_bgr, key))
        annotation_path = item_dir / "annotation.json"
        suggested_kind = "points"
        annotation = write_candidate_annotation(
            annotation_path, scan_path, sheet_path, key, force=args.force_auto
        )
        summary = {
            "sample": key,
            "detector_case": prediction.case,
            "detected_line_count": prediction.line_count,
            "detected_area_pixels": int(prediction.area_mask.sum()),
            "annotation_ready": annotation_path.exists(),
            "auto_candidate_count": int(annotation.get("auto_candidate_count", 0)),
            "auto_confidence": float(annotation.get("auto_confidence", 0.0)),
        }
        (item_dir / "prediction_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (item_dir / "NEXT_STEP.txt").write_text(
            "자동 후보가 먼저 표시됩니다. 실제 보정시트와 비교해 확인/수정하세요.\n\n"
            f'.venv\\Scripts\\python.exe zero_line_evaluation\\review_sample.py {key}\n\n'
            "조작: 왼쪽 클릭=추가, 후보에 오른쪽 클릭=삭제, S=확정 저장, Q=종료\n",
            encoding="utf-8",
        )
        print(f"{key}: {item_dir / 'review_board.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
