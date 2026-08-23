"""합성 편차 이미지 생성기 — 회사 데이터 없이 데모·테스트를 돌리기 위한 도구.

[왜 필요한가]
실제 3D 스캔 이미지는 아진산업 생산 데이터다. 공개 저장소에 올릴 수 없고,
팀원 각자의 PC에도 배포하기 곤란하다. 그렇다고 샘플이 없으면
저장소를 받은 사람이 코드를 실행해 볼 수 없다.

그래서 실제 스캔과 같은 구조(무지개 히트맵 + 컬러바 + 라벨 박스 + 지시선)를
가진 가짜 이미지를 만든다. 검출기 입장에서는 실제 이미지와 구분되지 않는다.

[덤으로 얻는 것]
합성 이미지는 편차 분포를 우리가 정했으므로 **정답을 안다.**
따라서 검출 결과를 정답과 비교해 IoU 로 정확도를 측정할 수 있다.
실제 스캔 이미지로는 불가능한 검증이다.

    python -m zero_line_detection.make_sample
    python -m zero_line_detection.make_sample --evaluate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import constants as C  # noqa: E402
from shared.utils import get_logger, imwrite, read_rgb, write_rgb  # noqa: E402

log = get_logger("make_sample")

# 무지개 램프 제어점 (최솟값 -> 최댓값)
RAMP = [
    (255, 0, 255),    # 마젠타
    (0, 0, 255),      # 파랑
    (0, 255, 255),    # 시안
    (0, 255, 0),      # 초록   <- 편차 0 부근
    (255, 255, 0),    # 노랑
    (255, 0, 0),      # 빨강
]


def build_ramp(n: int = 256) -> np.ndarray:
    """제어점을 선형 보간해 (n,3) uint8 램프를 만든다."""
    ctrl = np.array(RAMP, dtype=np.float32)
    xs = np.linspace(0, len(ctrl) - 1, n)
    lo = np.floor(xs).astype(int)
    hi = np.clip(lo + 1, 0, len(ctrl) - 1)
    t = (xs - lo)[:, None]
    return (ctrl[lo] * (1 - t) + ctrl[hi] * t).astype(np.uint8)


def part_mask(h: int, w: int) -> np.ndarray:
    """선루프 보강재를 닮은 사각 링 형상."""
    m = np.zeros((h, w), dtype=np.uint8)
    outer = (int(w * 0.09), int(h * 0.13), int(w * 0.82), int(h * 0.74))
    inner = (int(w * 0.24), int(h * 0.30), int(w * 0.52), int(h * 0.40))
    cv2.rectangle(m, outer[:2], (outer[0] + outer[2], outer[1] + outer[3]), 255, -1)
    cv2.rectangle(m, inner[:2], (inner[0] + inner[2], inner[1] + inner[3]), 0, -1)
    # 모서리를 둥글게
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    )
    return m > 0


def deviation_field(h: int, w: int, vmax: float, seed: int = 7) -> np.ndarray:
    """부드럽게 변하는 가짜 편차 분포. 0 등고선이 여러 갈래로 지나가게 만든다."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = xx / w, yy / h

    f = (
        1.35 * np.sin(2.3 * np.pi * nx + 0.6)
        + 1.05 * np.cos(1.9 * np.pi * ny - 0.4)
        + 0.55 * np.sin(3.7 * np.pi * (nx + ny))
        - 0.45 * np.cos(4.1 * np.pi * (nx - ny))
    )
    noise = cv2.GaussianBlur(
        rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=18
    )
    f = f + 2.2 * noise
    f = f / (np.abs(f).max() + 1e-9) * vmax
    return f.astype(np.float32)


def render(
    h: int, w: int, vmin: float, vmax: float, seed: int = 7,
    shading: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """합성 스캔 이미지를 만든다.

    Returns:
        rgb    : 완성된 이미지
        field  : 편차 정답값 (부품 밖은 NaN)
        pmask  : 부품 영역
    """
    ramp = build_ramp(512)
    pmask = part_mask(h, w)
    field = deviation_field(h, w, max(abs(vmin), abs(vmax)), seed)

    t = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    idx = (t * (len(ramp) - 1)).astype(int)
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    rgb[pmask] = ramp[idx[pmask]]

    if shading:
        # 3D 렌더링의 조명 음영을 흉내낸다. 검출기가 밝기에 흔들리지 않는지 시험한다.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        light = 0.72 + 0.38 * np.cos(3.0 * np.pi * yy / h) * np.sin(2.0 * np.pi * xx / w)
        light = np.clip(cv2.GaussianBlur(light, (0, 0), 9), 0.55, 1.15)[..., None]
        shaded = np.clip(rgb.astype(np.float32) * light, 0, 255).astype(np.uint8)
        rgb[pmask] = shaded[pmask]

    _draw_colorbar(rgb, ramp, vmin, vmax)
    _draw_annotations(rgb, field, pmask, seed)
    return rgb, np.where(pmask, field, np.nan).astype(np.float32), pmask


def _draw_colorbar(rgb: np.ndarray, ramp: np.ndarray, vmin: float, vmax: float) -> None:
    """우측에 컬러바와 눈금을 그린다 (JD_71XX2 와 같은 배치)."""
    h, w = rgb.shape[:2]
    x0, x1 = int(w * 0.955), int(w * 0.975)
    y0, y1 = int(h * 0.02), int(h * 0.98)

    for y in range(y0, y1):
        t = 1.0 - (y - y0) / max(y1 - y0 - 1, 1)      # 위가 최댓값
        rgb[y, x0:x1] = ramp[int(t * (len(ramp) - 1))]

    for i in range(9):
        t = i / 8
        y = int(y0 + t * (y1 - y0 - 1))
        val = vmax + t * (vmin - vmax)
        cv2.line(rgb, (x1, y), (x1 + 5, y), (0, 0, 0), 1)
        cv2.putText(rgb, f"{val:.2f}", (x1 + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_annotations(
    rgb: np.ndarray, field: np.ndarray, pmask: np.ndarray, seed: int
) -> None:
    """라벨 박스와 지시선을 얹는다. 검출기가 이것을 걸러내는지 시험한다."""
    rng = np.random.default_rng(seed + 1)
    h, w = rgb.shape[:2]
    ys, xs = np.where(pmask)
    if len(xs) == 0:
        return

    for _ in range(22):
        k = rng.integers(0, len(xs))
        px, py = int(xs[k]), int(ys[k])
        val = float(field[py, px])

        # 라벨은 부품 바깥 여백에 놓고 지시선으로 연결한다
        lx = int(np.clip(px + rng.integers(-260, 260), 40, w - 120))
        ly = int(np.clip(py + rng.integers(-190, 190), 24, h - 24))
        if pmask[ly, lx]:
            ly = 20 if py < h / 2 else h - 26

        cv2.line(rgb, (px, py), (lx, ly), (40, 60, 200), 1, cv2.LINE_AA)
        cv2.circle(rgb, (px, py), 2, (40, 60, 200), -1)

        text = f"{val:+.1f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        pad = 4
        tl = (lx - pad, ly - th - pad)
        br = (lx + tw + pad, ly + pad)
        if abs(val) >= 1.0:                       # 큰 편차는 빨간 박스 + 흰 글씨
            cv2.rectangle(rgb, tl, br, (220, 20, 20), -1)
            cv2.rectangle(rgb, tl, br, (120, 0, 0), 1)
            color = (255, 255, 255)
        else:                                     # 작은 편차는 흰 박스 + 검은 글씨
            cv2.rectangle(rgb, tl, br, (250, 250, 250), -1)
            cv2.rectangle(rgb, tl, br, (90, 90, 90), 1)
            color = (20, 20, 20)
        cv2.putText(rgb, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
def evaluate(rgb: np.ndarray, truth: np.ndarray, tol: float) -> dict:
    """검출 결과를 정답과 비교한다 (IoU)."""
    from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line

    out = detect_zero_line(rgb, ZeroLineConfig(tolerance=tol, vmin=-2.0, vmax=2.0))
    pred = out.mask > 0
    gt = np.isfinite(truth) & (np.abs(np.nan_to_num(truth, nan=99)) <= tol)

    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return {
        "iou": inter / union if union else 0.0,
        "precision": inter / pred.sum() if pred.sum() else 0.0,
        "recall": inter / gt.sum() if gt.sum() else 0.0,
        "pred_px": int(pred.sum()),
        "gt_px": int(gt.sum()),
    }


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="합성 편차 이미지를 생성합니다.")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=760)
    ap.add_argument("--vmin", type=float, default=-2.0)
    ap.add_argument("--vmax", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--count", type=int, default=1, help="생성할 장수 (시드를 바꿔가며)")
    ap.add_argument("--outdir", default=str(C.SAMPLE))
    ap.add_argument("--evaluate", action="store_true",
                    help="생성 직후 검출기를 돌려 IoU 를 측정한다")
    ap.add_argument("--tolerance", type=float, default=0.2)
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = C.ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        seed = args.seed + i
        rgb, truth, _ = render(args.height, args.width, args.vmin, args.vmax, seed)
        stem = "sample_deviation_map" if args.count == 1 else f"sample_deviation_map_{i+1}"
        write_rgb(outdir / f"{stem}.png", rgb)
        np.save(outdir / f"{stem}_truth.npy", truth)

        gt = np.isfinite(truth) & (np.abs(np.nan_to_num(truth, nan=99)) <= args.tolerance)
        imwrite(outdir / f"{stem}_truth_mask.png", (gt * 255).astype(np.uint8))
        log.info("생성 : %s (seed=%d, 정답 0영역 %d px)", f"{stem}.png", seed, int(gt.sum()))

        if args.evaluate:
            m = evaluate(rgb, truth, args.tolerance)
            log.info("  검증 : IoU=%.3f  정밀도=%.3f  재현율=%.3f  (검출 %d px / 정답 %d px)",
                     m["iou"], m["precision"], m["recall"], m["pred_px"], m["gt_px"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
