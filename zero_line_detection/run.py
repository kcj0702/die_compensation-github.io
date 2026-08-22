"""0-Line 검출 실행 진입점 (파트 2).

프로젝트 공통 입출력 규격에 맞춰 실행한다.

    python -m zero_line_detection.run
    python -m zero_line_detection.run --input "data/sample/sample_deviation_map.png"
    python -m zero_line_detection.run --vmin -1.5 --vmax 2.0      # 컬러바가 잘린 경우
    python -m zero_line_detection.run --tolerance 0.3             # 허용오차 고정(mm)
    python -m zero_line_detection.run --all-samples               # 샘플 일괄 처리

입력 우선순위
    1. --input 으로 직접 지정한 파일
    2. data/intermediate/clean_deviation_map.png   (파트 4 산출물)
    3. data/intermediate/deviation_map.png         (원본)

출력 (data/intermediate/)
    zero_line_mask.png        [필수] 0 영역 이진 마스크 — 파트 간 공통 규격
    zero_line_crossing.png    부호 경계선 (허용오차 없이 결정되는 0-Line)
    zero_line_tolerance_sweep.csv  허용오차별 면적 변화 (민감도)
    zero_line_overlay.png     원본 위에 얹은 검증용 이미지
    zero_line_centerline.png  0 밴드 중심선
    zero_line_regions.csv     영역별 면적·중심·평균편차
    zero_line_contours.json   영역 윤곽 폴리라인
    zero_line_report.json     처리 파라미터·통계·경고
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import constants as C  # noqa: E402
from shared.utils import get_logger, imwrite, read_rgb, save_json, write_rgb  # noqa: E402
from zero_line_detection.visualize import make_overlay  # noqa: E402
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line  # noqa: E402

log = get_logger("zero_line")


def resolve_input(explicit: str | None) -> Path:
    """규격에 따라 입력 이미지를 고른다."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = C.ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"입력 이미지가 없습니다: {p}")
        return p

    for name in (C.CLEAN_DEVIATION_MAP, C.DEVIATION_MAP):
        p = C.INTERMEDIATE / name
        if p.exists():
            if name == C.CLEAN_DEVIATION_MAP:
                log.info("파트 4의 라벨 제거 이미지를 사용합니다.")
            return p

    sample = C.SAMPLE / "sample_deviation_map.png"
    if sample.exists():
        log.info("중간 산출물이 없어 샘플 이미지로 실행합니다.")
        return sample

    raise FileNotFoundError(
        f"입력 이미지를 찾지 못했습니다. {C.INTERMEDIATE / C.DEVIATION_MAP} 에 "
        "편차 이미지를 두거나 --input 으로 지정하세요."
    )


def build_config(args: argparse.Namespace) -> ZeroLineConfig:
    return ZeroLineConfig(
        tolerance=args.tolerance,
        tolerance_ratio=args.tolerance_ratio,
        color_max_dist=args.color_max_dist,
        smooth_ksize=args.smooth,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        min_region_area=args.min_region_area,
        use_annotation_mask=not args.no_annotation_mask,
        vmin=args.vmin,
        vmax=args.vmax,
        emit_centerline=not args.no_centerline,
    )


def process(src: Path, cfg: ZeroLineConfig, outdir: Path, prefix: str = "") -> dict:
    """이미지 1장을 처리하고 산출물을 저장한다."""
    rgb = read_rgb(src)
    out = detect_zero_line(rgb, cfg, source_name=src.name)
    r = out.result

    outdir.mkdir(parents=True, exist_ok=True)
    p = lambda name: outdir / f"{prefix}{name}"  # noqa: E731

    imwrite(p(C.ZERO_LINE_MASK), out.mask)
    imwrite(p(C.ZERO_LINE_CROSSING), out.zero_crossing)
    write_rgb(p(C.ZERO_LINE_OVERLAY),
              make_overlay(rgb, out.mask, out.centerline,
                           zero_crossing=out.zero_crossing))
    if out.centerline is not None:
        imwrite(p("zero_line_centerline.png"), out.centerline)

    # 허용오차 민감도 — "왜 하필 그 값이냐" 에 숫자로 답하기 위한 표
    pd.DataFrame(r.params.get("tolerance_sweep", [])).to_csv(
        p(C.ZERO_LINE_SWEEP), index=False, encoding="utf-8-sig"
    )

    pd.DataFrame([x.to_dict() for x in r.regions]).to_csv(
        p(C.ZERO_LINE_REGIONS), index=False, encoding="utf-8-sig"
    )

    save_json(p(C.ZERO_LINE_CONTOURS), {
        "source_image": r.source_image,
        "image_width": r.image_width,
        "image_height": r.image_height,
        "contours": [c.reshape(-1, 2).tolist() for c in out.contours],
    })

    report = r.to_dict()
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    report["warnings"] = out.warnings
    save_json(p(C.ZERO_LINE_REPORT), report)

    # ── 콘솔 요약 ────────────────────────────────────────────────
    log.info("─" * 62)
    log.info("입력      : %s (%d x %d)", src.name, r.image_width, r.image_height)
    cbi = r.colorbar
    log.info("컬러바    : %s측 x=%d~%d, 최솟값=%s쪽",
             "좌" if cbi["side"] == "left" else "우",
             cbi["x0"], cbi["x1"], "위" if cbi["vmin_at"] == "top" else "아래")
    log.info("허용오차  : ±%.3f %s", r.tolerance, r.tolerance_unit)
    log.info("부품 영역 : %d px", r.part_px)
    log.info("0 영역    : %d px (부품의 %.1f%%), 영역 %d 개  ← 허용오차에 좌우됨",
             r.total_zero_px, r.zero_ratio * 100, len(r.regions))
    log.info("0-Line 선 : %d px  ← 부호 경계, 허용오차 무관",
             r.params.get("zero_crossing_px", 0))
    for w in out.warnings:
        log.warning("%s", w)
    if r.regions:
        log.info("상위 영역 : " + ", ".join(
            f"#{i}({x.area_px}px)" for i, x in enumerate(r.regions[:5], 1)))
    return report


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="3D 스캔 편차 이미지에서 0-Line 영역을 검출합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", help="입력 이미지 경로 (미지정 시 규격 경로 자동 탐색)")
    ap.add_argument("--outdir", default=str(C.INTERMEDIATE), help="출력 폴더")
    ap.add_argument("--all-samples", action="store_true",
                    help="data/sample 의 모든 이미지를 일괄 처리")

    g = ap.add_argument_group("판정 기준")
    g.add_argument("--tolerance", type=float, default=None,
                   help="0 판정 허용오차 절대값. 미지정 시 --tolerance-ratio 사용")
    g.add_argument("--tolerance-ratio", type=float, default=0.10,
                   help="컬러바 반경 대비 허용오차 비율")
    g.add_argument("--vmin", type=float, default=None,
                   help="컬러바 최솟값 (mm). 컬러바가 잘렸을 때 필수")
    g.add_argument("--vmax", type=float, default=None, help="컬러바 최댓값 (mm)")

    g2 = ap.add_argument_group("영상 처리")
    g2.add_argument("--color-max-dist", type=float, default=14.0)
    g2.add_argument("--smooth", type=int, default=5, help="중앙값 필터 크기 (0=미적용)")
    g2.add_argument("--morph-open", type=int, default=2)
    g2.add_argument("--morph-close", type=int, default=4)
    g2.add_argument("--min-region-area", type=int, default=80)
    g2.add_argument("--no-annotation-mask", action="store_true",
                    help="라벨·지시선 제거를 끈다 (파트 4 결과를 쓸 때)")
    g2.add_argument("--no-centerline", action="store_true", help="중심선 생성 생략")

    args = ap.parse_args(argv)
    cfg = build_config(args)
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = C.ROOT / outdir

    try:
        if args.all_samples:
            images = sorted(
                q for q in C.SAMPLE.glob("*.png") if "mask" not in q.name
            )
            if not images:
                log.error("data/sample 에 이미지가 없습니다.")
                return 1
            for q in images:
                process(q, cfg, outdir, prefix=f"{q.stem}__")
            log.info("─" * 62)
            log.info("샘플 %d 장 처리 완료 → %s", len(images), outdir)
        else:
            src = resolve_input(args.input)
            process(src, cfg, outdir)
            log.info("─" * 62)
            log.info("완료 → %s", outdir)
    except Exception as e:                       # noqa: BLE001
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
