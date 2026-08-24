"""편차 포인트 추출 파이프라인의 명령행 진입점."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:  # `python -m deviation_extraction.run`과 직접 실행을 모두 지원한다.
    from . import config
    from .point_extractor import (
        ValueReader,
        extract_points,
        save_csv,
        save_debug_image,
    )
else:  # pragma: no cover - 직접 스크립트 실행 경로
    import config
    from point_extractor import ValueReader, extract_points, save_csv, save_debug_image


def _build_reader(model_id: str, device: str | None, offline: bool) -> ValueReader:
    """무거운 VLM 의존성은 실제 라벨이 있을 때만 불러온다."""
    if __package__:
        from .vlm_reader import LabelValueReader
    else:  # pragma: no cover - 직접 스크립트 실행 경로
        from vlm_reader import LabelValueReader

    return LabelValueReader(
        model_id=model_id,
        device=device,
        local_files_only=offline,
    )


def main() -> None:
    """CLI 인자를 해석해 추출, CSV 저장, 선택적 시각화를 실행한다."""
    parser = argparse.ArgumentParser(description="편차 맵의 라벨 좌표와 값을 CSV로 추출")
    parser.add_argument(
        "--image", type=Path, default=config.DEVIATION_MAP_PATH, help="편차 맵 경로"
    )
    parser.add_argument(
        "--zero-line-mask",
        type=Path,
        default=config.ZERO_LINE_MASK_PATH,
        help="선택적인 제로 라인 마스크 경로",
    )
    parser.add_argument("--out", type=Path, default=config.OUTPUT_CSV_PATH, help="CSV 저장 경로")
    parser.add_argument("--model", type=str, default=config.VLM_MODEL_ID, help="VLM 모델 ID")
    parser.add_argument("--device", type=str, default=None, help="추론 장치 예: cuda, cuda:0, cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.VLM_BATCH_SIZE,
        help="한 번에 판독할 라벨 crop 수",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="로컬 캐시만 사용하고 모델 다운로드 차단",
    )
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="좌표 색상의 컬러맵 값과 판독값 비교",
    )
    parser.add_argument("--debug", action="store_true", help="검출 오버레이 이미지 저장")
    parser.add_argument(
        "--debug-out",
        type=Path,
        default=config.DEBUG_IMAGE_PATH,
        help="디버그 이미지 저장 경로",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"편차 이미지를 찾을 수 없음: {args.image}")
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")

    points = extract_points(
        image_path=args.image,
        zero_line_mask_path=args.zero_line_mask,
        reader_factory=lambda: _build_reader(args.model, args.device, args.offline),
        cross_check=args.cross_check,
        batch_size=args.batch_size,
    )

    save_csv(points, out_path=args.out)
    coordinate_count = sum(point.x_px is not None for point in points)
    value_count = sum(point.value_mm is not None for point in points)
    print(
        f"라벨 {len(points)}개 | 좌표 {coordinate_count}개 | "
        f"값 {value_count}개 -> {args.out}"
    )

    low_conf = [p for p in points if p.confidence != "ok"]
    if low_conf:
        print(f"주의: {len(low_conf)}개 포인트는 검토 필요(confidence != ok)")

    if args.debug:
        save_debug_image(args.image, points, out_path=args.debug_out)
        print(f"디버그 이미지 저장 -> {args.debug_out}")

    if not points:
        print("오류: 라벨을 찾지 못했습니다. 입력 이미지와 검출 임계값을 확인하세요.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
