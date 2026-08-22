"""전체 데모 실행 — 각 파트 산출물을 모아 result.json 을 만든다.

각 파트는 독립적으로 개발되므로, 아직 없는 산출물은 건너뛰고
있는 것만 모은다. 파트가 하나씩 완성될 때마다 result.json 이 채워진다.

    python pipeline/run_demo.py
    python pipeline/run_demo.py --input "data/sample/sample_deviation_map.png"
    python pipeline/run_demo.py --skip-zero-line      # 이미 돌렸을 때
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import constants as C  # noqa: E402
from shared.schemas import DemoResult  # noqa: E402
from shared.utils import get_logger, load_json, save_json  # noqa: E402

log = get_logger("run_demo")


def rel(p: Path) -> str:
    """result.json 에는 프로젝트 루트 기준 상대경로로 적는다."""
    try:
        return str(p.relative_to(C.ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def collect() -> DemoResult:
    """data/intermediate 를 훑어 존재하는 산출물만 모은다."""
    res = DemoResult(generated_at=datetime.now().isoformat(timespec="seconds"))

    # ── 이미지 ────────────────────────────────────────────────────
    image_specs = [
        ("deviation_map", C.DEVIATION_MAP, "원본 편차 이미지"),
        ("clean_deviation_map", C.CLEAN_DEVIATION_MAP, "[4] 라벨 제거 이미지"),
        ("zero_line_mask", C.ZERO_LINE_MASK, "[2] 0-Line 마스크"),
        ("zero_line_overlay", C.ZERO_LINE_OVERLAY, "[2] 0-Line 오버레이"),
        ("zero_line_centerline", "zero_line_centerline.png", "[2] 0-Line 중심선"),
    ]
    for key, name, desc in image_specs:
        p = C.INTERMEDIATE / name
        if p.exists():
            res.images[key] = rel(p)
        else:
            res.warnings.append(f"없음: {name} ({desc})")

    # ── 표 ────────────────────────────────────────────────────────
    table_specs = [
        ("deviation_points", C.DEVIATION_POINTS, "[3] 편차값·좌표"),
        ("depth_measurements", C.DEPTH_MEASUREMENTS, "[5] 깊이 측정"),
        ("zero_line_regions", C.ZERO_LINE_REGIONS, "[2] 0-Line 영역"),
    ]
    for key, name, desc in table_specs:
        p = C.INTERMEDIATE / name
        if p.exists():
            res.tables[key] = rel(p)
        else:
            res.warnings.append(f"없음: {name} ({desc})")

    # ── 0-Line 요약 ───────────────────────────────────────────────
    report = C.INTERMEDIATE / C.ZERO_LINE_REPORT
    if report.exists():
        r = load_json(report)
        res.source_image = r.get("source_image", "")
        res.zero_line = {
            "total_zero_px": r.get("total_zero_px"),
            "part_px": r.get("part_px"),
            "zero_ratio": r.get("zero_ratio"),
            "tolerance": r.get("tolerance"),
            "tolerance_unit": r.get("tolerance_unit"),
            "region_count": len(r.get("regions", [])),
            "colorbar": r.get("colorbar", {}),
            "warnings": r.get("warnings", []),
        }
        res.warnings.extend(r.get("warnings", []))
    return res


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="데모 전체 실행")
    ap.add_argument("--input", help="편차 이미지 경로")
    ap.add_argument("--skip-zero-line", action="store_true")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=None)
    args = ap.parse_args(argv)

    # ── [2] 0-Line 검출 ───────────────────────────────────────────
    if not args.skip_zero_line:
        from zero_line_detection.run import main as zero_main

        argv2: list = []
        if args.input:
            argv2 += ["--input", args.input]
        for flag, val in (("--vmin", args.vmin), ("--vmax", args.vmax),
                          ("--tolerance", args.tolerance)):
            if val is not None:
                argv2 += [flag, str(val)]
        if zero_main(argv2) != 0:
            log.error("0-Line 검출 실패")
            return 1

    # ── 나머지 파트 ───────────────────────────────────────────────
    # 완성되면 여기에 호출을 추가한다. 지금은 산출물이 있으면 모으기만 한다.
    #   from label_removal.run import main as label_main
    #   from deviation_extraction.run import main as dev_main
    #   from depth_measurement.run import main as depth_main

    res = collect()
    out = C.OUTPUT / C.RESULT_JSON
    save_json(out, res.to_dict())

    log.info("=" * 62)
    log.info("이미지 %d 개, 표 %d 개 수집", len(res.images), len(res.tables))
    if res.zero_line:
        z = res.zero_line
        log.info("0-Line : %s px (부품의 %.1f%%), 영역 %s 개",
                 z["total_zero_px"], (z["zero_ratio"] or 0) * 100, z["region_count"])
    for w in res.warnings:
        log.warning("%s", w)
    log.info("결과 → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
