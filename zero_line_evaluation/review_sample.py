"""Open one prepared sample for review and evaluate it after the window closes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="제로라인 예시 검토 후 바로 평가")
    parser.add_argument("sample", help="64XX2, 67XX6 또는 71XX2")
    parser.add_argument("--tolerance-px", type=float, default=8.0)
    args = parser.parse_args()
    candidates = [path for path in (HERE / "work").iterdir() if args.sample.upper() in path.name.upper()]
    if len(candidates) != 1:
        raise ValueError(f"샘플을 하나로 찾지 못했습니다: {args.sample} ({len(candidates)}개)")
    item_dir = candidates[0]
    annotation_path = item_dir / "annotation.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotate = [
        sys.executable, str(HERE / "annotate_ground_truth.py"),
        "--scan", payload["scan_image"], "--sheet", payload["sheet_image"],
        "--output", str(annotation_path), "--kind", payload.get("kind", "line"),
    ]
    subprocess.run(annotate, check=True)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not payload.get("reviewed") or not payload.get("lines"):
        print("정답을 저장하지 않아 평가는 실행하지 않았습니다. S로 저장한 뒤 다시 실행하세요.")
        return 1
    result_dir = item_dir / "result"
    evaluate = [
        sys.executable, str(HERE / "evaluate_zero_line.py"),
        "--annotation", str(annotation_path), "--output", str(result_dir),
        "--tolerance-px", str(args.tolerance_px),
    ]
    subprocess.run(evaluate, check=True)
    print(f"완료: {result_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
