"""보정시트의 제로라인을 품번별 라이브러리에 등록한다.

사용법:
    python -m zero_line_detection.register_sheet \
        --scan  "data/intermediate/JD_64XX2-DR000 3D 스캔.png" \
        --sheet "data/intermediate/JD_64XX2-DR000 보정시트.png" \
        --vmin -1.5 --vmax 2.0

등록해두면 같은 품번의 스캔이 들어왔을 때 서버가 이 선을 그대로 쓴다.
결과는 zero_line_detection/zero_line_library.json 에 쌓인다.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from zero_line_detection.sheet_reference import (
    extract_sheet_zero_line, save_to_library,
)
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line


LIBRARY_PATH = Path(__file__).resolve().parent / "zero_line_library.json"


def part_no_from_name(name: str) -> str:
    """파일명에서 품번을 뽑는다 (프론트엔드와 같은 규칙)."""
    match = re.search(r"[0-9]{2}[A-Z0-9]{2,4}", name.upper())
    return match.group(0) if match else Path(name).stem


def read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="보정시트 제로라인을 품번별로 등록")
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--part-no", type=str)
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--library", type=Path, default=LIBRARY_PATH)
    args = parser.parse_args()

    scan = read_bgr(args.scan)
    config = ZeroLineConfig(vmin=args.vmin, vmax=args.vmax)
    output = detect_zero_line(cv2.cvtColor(scan, cv2.COLOR_BGR2RGB), config)

    part_no = args.part_no or part_no_from_name(args.scan.name)
    reference = extract_sheet_zero_line(
        args.sheet, output.part_mask, part_no, values=output.values
    )
    save_to_library(args.library, reference)

    print(f"[등록] 품번 {part_no}")
    print(f"  꼭짓점 {len(reference.points)}개, 좌우반전 {reference.mirrored}")
    print(f"  시트 {args.sheet.name}")
    print(f"  저장 {args.library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
