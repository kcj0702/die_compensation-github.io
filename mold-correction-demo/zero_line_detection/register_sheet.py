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
    MIN_ASPECT_AGREEMENT, extract_sheet_zero_areas, extract_sheet_zero_line,
    save_to_library,
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
    parser.add_argument(
        "--confirm-areas", action="store_true",
        help="살몬색 영역이 공정 구역이 아니라 정말 제로존임을 확인했을 때 붙인다",
    )
    args = parser.parse_args()

    scan = read_bgr(args.scan)
    config = ZeroLineConfig(vmin=args.vmin, vmax=args.vmax)
    output = detect_zero_line(cv2.cvtColor(scan, cv2.COLOR_BGR2RGB), config)

    part_no = args.part_no or part_no_from_name(args.scan.name)
    # 시트마다 제로 표기가 다르다 — 이어진 빨간 선이면 선으로, 살몬색
    # 구획이면 여러 존(면)으로 읽는다. 선을 먼저 보고 없으면 면으로 간다.
    try:
        reference = extract_sheet_zero_line(
            args.sheet, output.part_mask, part_no, values=output.values
        )
        kind = "line"
        detail = f"꼭짓점 {len(reference.points)}개"
    except ValueError:
        reference = extract_sheet_zero_areas(
            args.sheet, output.part_mask, part_no, values=output.values
        )
        kind = "areas"
        detail = f"존 {len(reference.contours)}개"

    # 시트 패널을 잘못 골랐으면(확대도를 부품 전체로 착각) 좌표가 통째로
    # 어긋난다. 조용히 저장하면 그 뒤 평가·학습이 전부 오염되므로 막는다.
    # 실측(JD_71XX2): 다중 뷰 시트의 확대 패널을 골라 정답 존 5개 중 3개가
    # 부품 바깥으로 나갔는데도 아무 경고 없이 등록됐었다.
    if not reference.reliable:
        print(f"[거부] 품번 {part_no}: 시트 패널과 스캔 부품의 종횡비가 "
              f"맞지 않습니다 (일치도 {reference.aspect_agreement:.2f} < "
              f"{MIN_ASPECT_AGREEMENT})")
        print("  확대도(부분 뷰)를 부품 전체로 착각했을 가능성이 큽니다 —")
        print("  다중 뷰 시트는 아직 지원하지 않습니다. 등록하지 않았습니다.")
        return 1

    if kind == "areas" and not args.confirm_areas:
        # 살몬/분홍 영역이 항상 제로존인 건 아니다. JD_67XX6 은 범례에
        # `"0" 라인 = 빨간 점선 + 살몬 채움` 이라고 명시해서 맞지만,
        # JD_71XX2 는 같은 색으로 **공정 구역**을 칠하고 "①: 하형 용접",
        # "②: 상형 심고음" 이라고 적어놨다 — 그걸 제로존으로 등록하면
        # 완전히 틀린 정답이 된다(실측으로 확인).
        print(f"[확인 필요] 품번 {part_no}: 살몬색 영역 {len(reference.contours)}개를 "
              f"제로존으로 읽었습니다.")
        print("  이 색이 시트에서 정말 '0' 표기인지 확인하세요 —")
        print("  공정 구역(예: \"①: 하형 용접\")을 같은 색으로 칠한 시트도 있습니다.")
        print("  맞으면 --confirm-areas 를 붙여 다시 실행하세요.")
        return 2

    save_to_library(args.library, reference, kind=kind)

    print(f"[등록] 품번 {part_no} ({kind})")
    print(f"  {detail}, 좌우반전 {reference.mirrored}")
    print(f"  종횡비 일치도 {reference.aspect_agreement:.2f}")
    print(f"  시트 {args.sheet.name}")
    print(f"  저장 {args.library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
