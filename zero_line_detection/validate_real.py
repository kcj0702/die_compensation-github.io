"""실제 스캔 이미지에 대한 색->편차값 변환 정확도 검증.

[왜 필요한가]
지금까지의 IoU 0.914 는 **내가 만든 합성 데이터**에 대한 점수다.
합성 생성기와 검출기가 같은 무지개 램프를 전제하므로 자기 채점에 가깝다.
실제 스캔에서 얼마나 맞는지는 다른 문제다.

[어떻게 검증하는가]
스캔 이미지에는 사람이 적어 둔 정답이 이미 들어 있다.

    [-1.7] ──────────● 이 지점의 편차는 -1.7 이다

라벨 박스와 지시선 끝점(앵커)을 자동으로 찾고, 그 지점의 색을 편차값으로
되읽어 라벨 값과 비교한다. 오차를 mm 단위로 낼 수 있다.

[사용법]
숫자를 읽는 OCR 이 아직 없으므로(파트 3 담당) 값은 사람이 넣는다.

    # 1) 번호가 매겨진 이미지를 만든다
    python -m zero_line_detection.validate_real --image "..." --emit-template

    # 2) 만들어진 CSV 의 value 칸에 이미지를 보고 값을 적는다
    #    (틀린 라벨은 value 를 비워 두면 건너뛴다)

    # 3) 검증한다
    python -m zero_line_detection.validate_real --image "..." --truth truth.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import constants as C  # noqa: E402
from shared.utils import get_logger, read_rgb, write_rgb  # noqa: E402
from zero_line_detection.colorbar import detect_colorbar  # noqa: E402
from zero_line_detection.label_anchors import draw_anchors, find_anchors  # noqa: E402
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line  # noqa: E402

log = get_logger("validate_real")


def sample_value(
    values: np.ndarray, part: np.ndarray, x: int, y: int, radius: int = 4
) -> float | None:
    """앵커 주변의 편차값 중앙값.

    앵커는 지시선 끝점이라 정확히 그 픽셀이 지시선 자체이거나 부품 경계일 수 있다.
    주변을 조금 넓게 보고 부품으로 인정된 픽셀만 중앙값을 낸다.

    반경을 무한정 넓히지는 않는다. 편차는 위치에 따라 변하므로
    넓게 평균낼수록 그 자체가 오차가 된다. 12px 안에서 부품 픽셀을 못 찾으면
    앵커가 부품 위에 있지 않다는 뜻이므로 그 지점은 검증에서 뺀다.
    """
    h, w = values.shape
    for r in (radius, radius * 2, radius * 3):
        y0, y1 = max(y - r, 0), min(y + r + 1, h)
        x0, x1 = max(x - r, 0), min(x + r + 1, w)
        vals = values[y0:y1, x0:x1][part[y0:y1, x0:x1]]
        if vals.size >= 5:
            return float(np.median(vals))
    return None


def run(
    image: Path,
    truth_csv: Path | None,
    vmin: float | None,
    vmax: float | None,
    radius: int,
    outdir: Path,
) -> int:
    rgb = read_rgb(image)
    anchors = find_anchors(rgb)
    log.info("라벨 박스 %d개 검출 (지시선 추적 성공 %d개)",
             len(anchors), sum(1 for a in anchors if a.leader_len > 5))

    outdir.mkdir(parents=True, exist_ok=True)
    stem = image.stem
    numbered = outdir / f"{stem}__labels.png"
    write_rgb(numbered, draw_anchors(rgb, anchors))
    log.info("번호 이미지 → %s", numbered)

    if truth_csv is None:
        tmpl = outdir / f"{stem}__truth_template.csv"
        pd.DataFrame([
            {"label_id": a.label_id, "kind": a.kind,
             "box_cx": a.box_cx, "box_cy": a.box_cy,
             "anchor_x": a.anchor_x, "anchor_y": a.anchor_y,
             "leader_len": a.leader_len, "value": ""}
            for a in anchors
        ]).to_csv(tmpl, index=False, encoding="utf-8-sig")
        log.info("정답 입력용 CSV → %s", tmpl)
        log.info("이미지를 보고 value 칸을 채운 뒤 --truth 로 넘기세요.")
        return 0

    truth = pd.read_csv(truth_csv)
    truth = truth[truth["value"].notna()]
    tmap = {int(r.label_id): float(r.value) for r in truth.itertuples()}
    if not tmap:
        log.error("정답 CSV 의 value 칸이 모두 비어 있습니다.")
        return 1

    cfg = ZeroLineConfig(vmin=vmin, vmax=vmax)
    out = detect_zero_line(rgb, cfg, source_name=image.name)
    cb = out.colorbar
    unit = cb.unit
    # 정규화 모드면 라벨 값(mm)과 견줄 수 있게 스케일을 맞춘다
    scale = 1.0
    if unit == "normalized":
        span = max(abs(v) for v in tmap.values())
        scale = span / max(abs(out.values[out.part_mask]).max(), 1e-9) if span else 1.0

    rows = []
    for a in anchors:
        if a.label_id not in tmap:
            continue
        got = sample_value(out.values, out.part_mask, a.anchor_x, a.anchor_y, radius)
        if got is None:
            log.warning("라벨 %d: 앵커 주변에 부품 픽셀이 없어 건너뜁니다.", a.label_id)
            continue
        rows.append({
            "label_id": a.label_id, "kind": a.kind,
            "anchor_x": a.anchor_x, "anchor_y": a.anchor_y,
            "label_value": tmap[a.label_id],
            "read_value": round(got * scale, 3),
            "error": round(got * scale - tmap[a.label_id], 3),
        })

    if not rows:
        log.error("비교할 지점이 없습니다.")
        return 1

    df = pd.DataFrame(rows)
    df["abs_error"] = df["error"].abs()
    csv = outdir / f"{stem}__validation.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    mae = df["abs_error"].mean()
    rmse = float(np.sqrt((df["error"] ** 2).mean()))
    corr = df["label_value"].corr(df["read_value"])

    log.info("=" * 66)
    log.info("실제 스캔 검증 — %s  (지점 %d개, 단위 %s)", image.name, len(df), unit)
    log.info("=" * 66)
    log.info("  MAE          : %.3f mm", mae)
    log.info("  RMSE         : %.3f mm", rmse)
    log.info("  최대 오차    : %.3f mm  (라벨 %d)",
             df["abs_error"].max(), int(df.loc[df["abs_error"].idxmax(), "label_id"]))
    log.info("  상관계수     : %.4f", corr)
    log.info("  0.2mm 이내   : %d / %d (%.0f%%)",
             (df["abs_error"] <= 0.2).sum(), len(df),
             (df["abs_error"] <= 0.2).mean() * 100)
    log.info("  0.5mm 이내   : %d / %d (%.0f%%)",
             (df["abs_error"] <= 0.5).sum(), len(df),
             (df["abs_error"] <= 0.5).mean() * 100)

    worst = df.nlargest(min(5, len(df)), "abs_error")
    log.info("  오차 큰 순:")
    for r in worst.itertuples():
        log.info("    라벨 %2d (%s)  적힌값 %+.1f  판독 %+.2f  오차 %+.2f",
                 r.label_id, r.kind, r.label_value, r.read_value, r.error)
    log.info("  상세 → %s", csv)
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="실제 스캔에 대한 편차 판독 정확도 검증")
    ap.add_argument("--image", required=True, help="검증할 스캔 이미지")
    ap.add_argument("--truth", help="라벨 값이 채워진 CSV")
    ap.add_argument("--emit-template", action="store_true",
                    help="정답 입력용 CSV 와 번호 이미지만 만든다")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--radius", type=int, default=4, help="앵커 주변 샘플 반경 (px)")
    ap.add_argument("--outdir", default=str(C.OUTPUT))
    args = ap.parse_args(argv)

    image = Path(args.image)
    if not image.is_absolute():
        image = C.ROOT / image
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = C.ROOT / outdir

    truth = None if args.emit_template else (Path(args.truth) if args.truth else None)
    if truth is not None and not truth.is_absolute():
        truth = C.ROOT / truth

    try:
        return run(image, truth, args.vmin, args.vmax, args.radius, outdir)
    except Exception as e:                       # noqa: BLE001
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
