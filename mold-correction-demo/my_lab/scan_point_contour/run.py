"""CLI for connecting detected scan points into closed shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

from connect_scan_points import IMAGE_SUFFIXES, find_clean_image, process_image


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=here.parent / "label_removal" / "input")
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=here.parent / "label_removal" / "output" / "2_labels_inpainted",
    )
    parser.add_argument("--output", type=Path, default=here / "output")
    args = parser.parse_args()

    originals = sorted(
        path
        for path in args.input.resolve().iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not originals:
        raise SystemExit("입력 이미지가 없습니다.")

    failures = 0
    for original in originals:
        try:
            clean = find_clean_image(original, args.clean_dir.resolve())
            output_dir = process_image(original, clean, args.output.resolve())
            print(f"[완료] {original.name} -> {output_dir}")
        except Exception as exc:
            failures += 1
            print(f"[실패] {original.name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

