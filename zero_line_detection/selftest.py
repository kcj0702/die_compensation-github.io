"""자체 점검 — 검출기가 현실적인 변형에서도 버티는지 확인한다.

    python -m zero_line_detection.selftest

[왜 필요한가]
합성 샘플 한 장에 대한 IoU 하나만 보고 "잘 된다" 고 하면 안 된다.
실제 스캔 이미지는 크기가 제각각이고(8/21 회의 기록), 화면 캡처나
문서 삽입을 거치면서 압축·리사이즈된다. 그 과정에서 컬러바 가장자리가
배경과 섞이는데, 이것 때문에 실제로 컬러바 검출이 통째로 실패한 적이 있다
(0.75배 축소 시 재현). 그 회귀를 다시 놓치지 않으려고 남긴다.

pytest 없이 그냥 실행되며, 하나라도 실패하면 종료코드 1 을 낸다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.utils import get_logger  # noqa: E402
from zero_line_detection.make_sample import render  # noqa: E402
from zero_line_detection.zero_line import (  # noqa: E402
    ZeroLineConfig, detect_zero_line,
)

log = get_logger("selftest")

TOL = 0.2
VMIN, VMAX = -2.0, 2.0
MIN_IOU = 0.80          # 이 아래로 떨어지면 실패로 본다


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return inter / union if union else 0.0


def _run(img: np.ndarray, gt: np.ndarray, **kw) -> tuple[float, object]:
    cfg = ZeroLineConfig(tolerance=TOL, vmin=VMIN, vmax=VMAX, **kw)
    out = detect_zero_line(img, cfg)
    return _iou(out.mask > 0, gt), out


def main() -> int:
    rgb, truth, _ = render(760, 1200, VMIN, VMAX, seed=7)
    gt = np.isfinite(truth) & (np.abs(np.nan_to_num(truth, nan=99)) <= TOL)

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        log.info("%s %-34s %s", "PASS" if ok else "FAIL", name, detail)

    # 1) 시드를 바꿔가며 — 특정 그림에만 맞춘 게 아닌지
    ious = []
    for seed in (7, 11, 23, 42, 99):
        r, t, _ = render(760, 1200, VMIN, VMAX, seed)
        g = np.isfinite(t) & (np.abs(np.nan_to_num(t, nan=99)) <= TOL)
        ious.append(_run(r, g)[0])
    check("여러 시드 (5장)", min(ious) >= MIN_IOU,
          f"평균 {np.mean(ious):.3f} 최저 {min(ious):.3f}")

    # 2) 크기 변경 — 실제 스캔은 크기가 제각각이다
    for s in (0.5, 0.75, 1.25, 1.5, 2.0):
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        r2 = cv2.resize(rgb, None, fx=s, fy=s, interpolation=interp)
        g2 = cv2.resize(gt.astype(np.uint8), (r2.shape[1], r2.shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
        try:
            v, _ = _run(r2, g2)
            check(f"크기 {s:.2f}배", v >= MIN_IOU, f"IoU {v:.3f}")
        except Exception as e:                       # noqa: BLE001
            check(f"크기 {s:.2f}배", False, str(e)[:60])

    # 3) JPEG 압축 — 문서·메신저를 거치면 반드시 열화된다
    for q in (95, 80, 60, 40):
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
        try:
            v, _ = _run(dec, gt)
            check(f"JPEG 품질 {q}", v >= MIN_IOU, f"IoU {v:.3f}")
        except Exception as e:                       # noqa: BLE001
            check(f"JPEG 품질 {q}", False, str(e)[:60])

    # 4) 180도 회전 — JD_67XX6 / JD_64XX2 가 실제로 이 상태다
    rot = cv2.rotate(rgb, cv2.ROTATE_180)
    gtr = cv2.rotate(gt.astype(np.uint8), cv2.ROTATE_180) > 0
    v, out = _run(rot, gtr)
    side_ok = out.colorbar.info.side == "left" and out.colorbar.info.vmin_at == "top"
    check("180도 회전", v >= MIN_IOU and side_ok,
          f"IoU {v:.3f}, 컬러바 {out.colorbar.info.side}/{out.colorbar.info.vmin_at}")

    # 5) 허용오차를 바꿔도 부호 경계선은 그대로여야 한다
    a = detect_zero_line(rgb, ZeroLineConfig(tolerance=0.1, vmin=VMIN, vmax=VMAX))
    b = detect_zero_line(rgb, ZeroLineConfig(tolerance=0.4, vmin=VMIN, vmax=VMAX))
    same = int((a.zero_crossing > 0).sum()) == int((b.zero_crossing > 0).sum())
    area_differs = a.result.total_zero_px != b.result.total_zero_px
    check("부호 경계선 = 허용오차 무관", same and area_differs,
          f"선 {int((a.zero_crossing>0).sum())}px 동일, "
          f"면 {a.result.total_zero_px}→{b.result.total_zero_px}")

    # 6) 잘린 컬러바를 감지하는가
    clipped = rgb[130:, :].copy()
    out = detect_zero_line(clipped, ZeroLineConfig(tolerance=TOL))
    check("잘린 컬러바 감지", out.colorbar.is_clipped and len(out.warnings) > 0,
          f"잘림={out.colorbar.is_clipped}, 경고 {len(out.warnings)}건")

    # ── 요약 ──────────────────────────────────────────────────────
    failed = [n for n, ok, _ in results if not ok]
    log.info("─" * 62)
    if failed:
        log.error("실패 %d / %d 건: %s", len(failed), len(results), ", ".join(failed))
        return 1
    log.info("전체 %d 건 통과", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
