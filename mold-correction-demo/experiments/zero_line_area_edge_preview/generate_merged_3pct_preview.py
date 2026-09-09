"""Run the merged-correction preview with a 3% final zero-area cutoff.

All other rules are identical to ``generate_merged_4pct_preview.py``:
same-sign correction gap 24 px, no distance merging of zero areas, gray
unmeasured faces assigned +3.01 mm, and conditional correction-hole filling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_merged_4pct_preview import process_one, write_summary
from generate_preview import DEFAULT_INPUT, SPECS


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_merged_correction_3pct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-mm", type=float, default=0.5)
    parser.add_argument("--gray-sentinel-mm", type=float, default=3.01)
    parser.add_argument("--min-zero-ratio", type=float, default=0.03)
    parser.add_argument("--correction-merge-gap", type=int, default=24)
    parser.add_argument("--zero-group-gap", type=int, default=0)
    parser.add_argument("--sign-adjacency", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gray_sentinel_mm <= 3.0:
        raise ValueError("--gray-sentinel-mm must be greater than 3.0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        process_one(
            spec,
            args.input_dir,
            args.output_dir,
            args.threshold_mm,
            args.gray_sentinel_mm,
            args.min_zero_ratio,
            args.correction_merge_gap,
            args.zero_group_gap,
            args.sign_adjacency,
        )
        for spec in SPECS
    ]
    write_summary(args.output_dir, summaries)
    for row in summaries:
        part_px = row["part_px"]
        filled = row["positive_holes_filled_px"] + row["negative_holes_filled_px"]
        print(
            f"{Path(row['source_image']).name}: "
            f"final={row['final_zero_region_count']} "
            f"({row['final_zero_line_ratio_of_part'] * 100:.2f}%), "
            f"holes-filled={filled / part_px * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
