"""zero_line_advance 명령행 실행기."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from advance import (
    AdvanceConfig,
    detect_advanced_zero_line,
    read_image,
    render_debug_images,
    write_image,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def infer_value_range(name: str) -> tuple[float, float] | None:
    upper = name.upper()
    if "JD_64XX2" in upper:
        return -1.5, 2.0
    if "JD_71XX2" in upper:
        return -2.0, 2.0
    return None


def find_clean_image(source: Path, clean_dir: Path | None) -> Path | None:
    if clean_dir is None or not clean_dir.is_dir():
        return None
    exact = clean_dir / f"{source.stem}_2_labels_inpainted{source.suffix}"
    if exact.is_file():
        return exact
    candidates = [
        path
        for path in clean_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and source.stem.lower() in path.stem.lower()
        and "2_labels_inpainted" in path.stem.lower()
    ]
    return sorted(candidates)[0] if candidates else None


def load_anchors(path: Path | None, source: Path) -> list[tuple[int, int]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        selected = payload
    elif isinstance(payload, dict):
        selected = []
        for key, value in payload.items():
            if key.lower() in source.stem.lower():
                selected = value
                break
    else:
        raise ValueError("anchors JSON은 좌표 목록 또는 파일명별 좌표 객체여야 합니다.")
    return [(int(point[0]), int(point[1])) for point in selected]


def list_inputs(path: Path, include_jd67: bool) -> list[Path]:
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(
            item
            for item in path.iterdir()
            if item.is_file()
            and item.suffix.lower() in IMAGE_SUFFIXES
            and "2_labels_inpainted" not in item.stem.lower()
        )
    else:
        raise FileNotFoundError(f"입력 경로가 없습니다: {path}")
    if not include_jd67:
        paths = [item for item in paths if "JD_67" not in item.name.upper()]
    return paths


def save_result(source: Path, clean_path: Path | None, output_root: Path, result, clean_bgr):
    output_dir = output_root / source.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in render_debug_images(clean_bgr, result).items():
        write_image(output_dir / name, image)
    report = {
        "source": str(source),
        "clean_image": str(clean_path) if clean_path else None,
        "config": result.config.to_dict(),
        "colorbar": result.colorbar,
        "detected_points": [anchor.to_dict() for anchor in result.detected_points],
        "candidate_anchors": [anchor.to_dict() for anchor in result.candidate_anchors],
        "zero_candidates": [anchor.to_dict() for anchor in result.zero_candidates],
        "line_zero_waypoints": [anchor.to_dict() for anchor in result.zero_waypoints],
        "selected_zero_points": [anchor.to_dict() for anchor in result.zero_points],
        "numeric_zero_points": [anchor.to_dict() for anchor in result.zero_points],
        "line_anchors": [anchor.to_dict() for anchor in result.line_anchors],
        "snapped_anchors": result.snapped_anchors,
        "raw_path_points": len(result.raw_path),
        "smooth_path": [[round(x, 2), round(y, 2)] for x, y in result.smooth_path],
        "point_only_path": [
            [round(x, 2), round(y, 2)] for x, y in result.point_only_path
        ],
        "warnings": result.warnings,
        "note": "작업자 판단을 보조하는 실험용 대표선이며 수학적 편차 0 등고선 정답이 아닙니다.",
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="-0.5~+0.5 후보 영역과 라벨 포인트를 이용해 단일 대표선을 생성합니다."
    )
    parser.add_argument("--input", type=Path, default=here / "input")
    parser.add_argument("--clean", type=Path, help="단일 입력용 2_labels_inpainted 이미지")
    parser.add_argument("--clean-dir", type=Path, default=here / "input")
    parser.add_argument("--output", type=Path, default=here / "output")
    parser.add_argument("--anchors", type=Path, help="수동 기준점 JSON. 지정 순서대로 연결")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--band-low", type=float, default=-0.5)
    parser.add_argument("--band-high", type=float, default=0.5)
    parser.add_argument("--color-max-dist", type=float, default=14.0)
    parser.add_argument("--anchor-snap-radius", type=float, default=80.0)
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=10,
        help="최종 작업선의 목표 최대 꼭짓점 수 (기본값: 10)",
    )
    parser.add_argument(
        "--simplify-epsilon",
        type=float,
        default=8.0,
        help="작은 요철을 무시하는 최소 픽셀 거리 (기본값: 8)",
    )
    parser.add_argument("--include-jd67", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sources = list_inputs(args.input.resolve(), args.include_jd67)
    if not sources:
        print("처리할 이미지가 없습니다. JD_67XX는 기본적으로 제외됩니다.")
        return 1

    failures = 0
    for source in sources:
        try:
            profile = infer_value_range(source.name)
            if args.vmin is not None or args.vmax is not None:
                if args.vmin is None or args.vmax is None:
                    raise ValueError("--vmin과 --vmax는 함께 지정해야 합니다.")
                vmin, vmax = args.vmin, args.vmax
            elif profile is not None:
                vmin, vmax = profile
            else:
                raise ValueError(
                    "이 파일의 컬러바 범위를 알 수 없습니다. --vmin과 --vmax를 지정하세요."
                )

            clean_path = args.clean.resolve() if args.clean else find_clean_image(
                source, args.clean_dir.resolve() if args.clean_dir else None
            )
            original = read_image(source)
            clean = read_image(clean_path) if clean_path else original.copy()
            if clean_path is None:
                print(f"[경고] {source.name}: 라벨 복원 이미지를 찾지 못해 원본을 사용합니다.")
            config = AdvanceConfig(
                band_low=args.band_low,
                band_high=args.band_high,
                color_max_dist=args.color_max_dist,
                anchor_snap_radius=args.anchor_snap_radius,
                simplify_epsilon=args.simplify_epsilon,
                max_vertices=args.max_vertices,
            )
            manual = load_anchors(args.anchors.resolve() if args.anchors else None, source)
            result = detect_advanced_zero_line(
                original,
                clean,
                vmin=vmin,
                vmax=vmax,
                config=config,
                manual_anchors=manual,
            )
            output_dir = save_result(source, clean_path, args.output.resolve(), result, clean)
            print(
                f"[완료] {source.name} -> {output_dir} "
                f"(후보 포인트 {len(result.candidate_anchors)}개, "
                f"연결 포인트 {len(result.line_anchors)}개, "
                f"경로 {len(result.raw_path)}px)"
            )
            for warning in result.warnings:
                print(f"  [주의] {warning}")
        except Exception as exc:
            failures += 1
            print(f"[실패] {source.name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
