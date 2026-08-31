"""제품데이터 정렬과 포인트 전사의 명령행 진입점."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
# deviation_extraction은 `import config` 형태의 지역 임포트를 쓰므로 자기 폴더가
# 경로에 있어야 한다. product_alignment는 `from . import config`라 겹치지 않는다.
DEVIATION_DIR = PROJECT_DIR / "deviation_extraction"
if str(DEVIATION_DIR) not in sys.path:
    sys.path.insert(0, str(DEVIATION_DIR))

from product_alignment import config  # noqa: E402
from product_alignment.alignment import estimate_alignment, is_inside, map_point  # noqa: E402
from product_alignment.compose import (  # noqa: E402
    SheetPoint, render_alignment_overlay, render_points,
)
from product_alignment.masks import build_product_mask, build_scan_mask  # noqa: E402
from product_alignment.registry import (  # noqa: E402
    AlignmentStore, ProductLibrary, part_number_from_name, read_image, write_png,
)


def _resolve_product(args: argparse.Namespace, part_number: str | None) -> Path:
    """Return the product-data image path from the flag or the library."""
    if args.product is not None:
        if not args.product.is_file():
            raise SystemExit(f"오류: 제품데이터 이미지를 찾을 수 없음: {args.product}")
        return args.product

    if part_number is None:
        raise SystemExit(
            "오류: 스캔 파일명에서 품번을 찾지 못했습니다. --product로 직접 지정하세요."
        )
    match = ProductLibrary(args.product_dir).find(part_number)
    if match is None:
        raise SystemExit(
            f"오류: 품번 {part_number}의 제품데이터가 등록되어 있지 않습니다. "
            "--product로 지정하면 --register로 등록할 수 있습니다."
        )
    if not match.exact:
        print(f"주의: {part_number} 대신 등록된 {match.part_number} 제품데이터를 사용합니다.")
    return match.path


def main() -> None:
    """CLI 인자를 해석해 정렬을 추정하고 포인트를 옮겨 그린다."""
    parser = argparse.ArgumentParser(
        description="3D 스캔의 측정점을 제품데이터 이미지 위로 옮긴다."
    )
    parser.add_argument("--scan", type=Path, required=True, help="3D 스캔 편차 맵 경로")
    parser.add_argument(
        "--product", type=Path, default=None, help="제품데이터 이미지 경로 (생략 시 품번으로 조회)"
    )
    parser.add_argument(
        "--product-dir", type=Path, default=config.PRODUCT_DIR, help="제품데이터 라이브러리 폴더"
    )
    parser.add_argument(
        "--alignment-dir", type=Path, default=config.ALIGNMENT_DIR, help="확정 정렬 저장 폴더"
    )
    parser.add_argument("--out", type=Path, default=None, help="합성 이미지 저장 경로")
    parser.add_argument("--overlay-out", type=Path, default=None, help="정렬 확인 오버레이 저장 경로")
    parser.add_argument(
        "--flip-x", dest="flip_x", action="store_true", default=None, help="좌우 반전 강제"
    )
    parser.add_argument(
        "--no-flip-x", dest="flip_x", action="store_false", help="좌우 반전 없음 강제"
    )
    parser.add_argument(
        "--flip-y", dest="flip_y", action="store_true", default=None, help="상하 반전 강제"
    )
    parser.add_argument(
        "--no-flip-y", dest="flip_y", action="store_false", help="상하 반전 없음 강제"
    )
    parser.add_argument(
        "--use-saved", action="store_true", help="품번에 확정 저장된 방향을 사용"
    )
    parser.add_argument(
        "--confirm", action="store_true", help="이번 정렬을 품번에 확정 저장"
    )
    parser.add_argument(
        "--register", action="store_true", help="--product 이미지를 품번에 등록"
    )
    args = parser.parse_args()

    if not args.scan.is_file():
        parser.error(f"스캔 이미지를 찾을 수 없음: {args.scan}")

    part_number = part_number_from_name(args.scan.name)
    product_path = _resolve_product(args, part_number)

    scan_image = read_image(args.scan)
    product_image = read_image(product_path)
    scan_mask = build_scan_mask(scan_image)
    product_mask = build_product_mask(product_image)

    flip_x, flip_y = args.flip_x, args.flip_y
    store = AlignmentStore(args.alignment_dir)
    if args.use_saved and part_number and flip_x is None and flip_y is None:
        saved = store.load(part_number)
        if saved is not None:
            flip_x, flip_y = saved.flip_x, saved.flip_y
            print(f"저장된 정렬 사용: {part_number} (좌우 {flip_x}, 상하 {flip_y})")

    alignment = estimate_alignment(
        scan_mask, product_mask, flip_x=flip_x, flip_y=flip_y
    )

    print(f"품번: {part_number or '미확인'} | 제품데이터: {product_path.name}")
    for candidate in alignment.candidates:
        mark = (
            "  <-"
            if candidate.flip_x == alignment.flip_x
            and candidate.flip_y == alignment.flip_y
            else ""
        )
        print(
            f"  좌우={int(candidate.flip_x)} 상하={int(candidate.flip_y)}  "
            f"외곽={candidate.outline_iou:.3f} 구멍={candidate.hole_iou:.3f} "
            f"밴드={candidate.band_iou:.3f}  점수={candidate.score:.3f}{mark}"
        )
    print(
        f"선택: 좌우 반전={alignment.flip_x}, 상하 반전={alignment.flip_y} | "
        f"2위와 격차 {alignment.margin:.3f} | "
        f"{'자동 판정 신뢰 가능' if alignment.confident else '사람 확인 필요'}"
    )
    for warning in alignment.warnings:
        print(f"주의: {warning}")

    from label_detector import detect_labels  # noqa: E402

    candidates = detect_labels(scan_image)
    traced = [item for item in candidates if item.traced and item.point_xy is not None]
    points: list[SheetPoint] = []
    dropped = 0
    for index, candidate in enumerate(traced, start=1):
        x, y = map_point(alignment, *candidate.point_xy)
        if not is_inside(alignment, x, y):
            dropped += 1
            continue
        points.append(
            SheetPoint(
                point_id=f"P-{index:02d}",
                x=x,
                y=y,
                label_color=candidate.label_color,
            )
        )

    print(
        f"라벨 후보 {len(candidates)}개 | 지시선 추적 {len(traced)}개 | "
        f"제품데이터로 전사 {len(points)}개"
    )
    if dropped:
        print(f"주의: 제품데이터 범위를 벗어난 포인트 {dropped}개는 제외했습니다.")

    if args.register and args.product is not None and part_number:
        registered = ProductLibrary(args.product_dir).register(part_number, product_image)
        print(f"제품데이터 등록 -> {registered}")
    if args.confirm and part_number:
        alignment.overridden = True
        saved_path = store.save(part_number, alignment)
        print(f"정렬 확정 저장 -> {saved_path}")

    if args.out is not None:
        write_png(args.out, render_points(product_image, points, show_values=False))
        print(f"합성 이미지 저장 -> {args.out}")
    if args.overlay_out is not None:
        from product_alignment.alignment import warp_scan_mask

        overlay = render_alignment_overlay(
            product_image, warp_scan_mask(alignment, scan_mask)
        )
        write_png(args.overlay_out, overlay)
        print(f"정렬 오버레이 저장 -> {args.overlay_out}")

    if not points:
        print("오류: 제품데이터로 옮길 포인트가 없습니다.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
