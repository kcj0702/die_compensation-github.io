"""현장 시연용 실행기.

    python -m zero_line_detection.demo                       # 샘플로 시연
    python -m zero_line_detection.demo --image "새파일.png"   # 즉석에서 받은 파일
    python -m zero_line_detection.demo --image "..." --sweep  # 허용오차 민감도까지

[설계 의도]
현장에서는 다음 세 가지가 중요하다.

  1. **한 줄로 끝나야 한다.** 옵션을 기억할 여유가 없다.
  2. **모르는 이미지를 받아도 돌아가야 한다.** 실무자가 그 자리에서
     다른 스캔을 줄 수 있고, 그때 죽으면 시연이 아니라 사고다.
  3. **실패해도 설명이 되어야 한다.** 파이썬 트레이스백 대신
     무엇이 왜 안 됐는지 사람 말로 나와야 한다.

그래서 이 파일은 예외를 전부 잡아 한국어로 안내하고,
결과 이미지를 한 장으로 합쳐 바로 띄운다.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import constants as C  # noqa: E402
from shared.utils import read_rgb, write_rgb  # noqa: E402
from zero_line_detection.visualize import make_overlay, make_value_map  # noqa: E402
from zero_line_detection.zero_line import (  # noqa: E402
    ZeroLineConfig, detect_zero_line,
)

BAR = "═" * 68


def _say(msg: str = "") -> None:
    print(msg, flush=True)


_FONT_CACHE: dict = {}

_FONT_PATHS = [
    "C:/Windows/Fonts/malgun.ttf",          # 맑은 고딕
    "C:/Windows/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


def _korean_font(size: int):
    """한글 폰트를 찾는다. 없으면 None (그러면 OpenCV 로 그린다)."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    try:
        from PIL import ImageFont

        for path in _FONT_PATHS:
            if Path(path).exists():
                font = ImageFont.truetype(path, size)
                break
    except Exception:                              # noqa: BLE001
        font = None
    _FONT_CACHE[size] = font
    return font


def _put_text(img, text: str, xy: tuple, size: int = 24):
    """이미지에 한글 텍스트를 얹는다.

    OpenCV 5 부터는 putText 가 한글을 그리지만 4.x 는 물음표로 바꿔 버린다.
    팀원 PC 의 버전이 제각각일 수 있으므로 PIL 을 우선 쓴다.
    """
    font = _korean_font(size)
    if font is None:
        cv2.putText(img, text, (xy[0], xy[1] + size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 32.0,
                    (40, 40, 40), 2, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=(40, 40, 40))
    img[:] = np.asarray(pil)
    return img


def _panel(images: list, titles: list, target_w: int = 1500) -> np.ndarray:
    """여러 장을 한 장으로 합친다. 가로로 긴 부품이면 세로로 쌓는다."""
    h0, w0 = images[0].shape[:2]
    stack_vertical = (w0 / max(h0, 1)) > 1.35

    if stack_vertical:
        each_w = target_w
        resized = [cv2.resize(im, (each_w, int(im.shape[0] * each_w / im.shape[1])))
                   for im in images]
        gap = 16
        total_h = sum(r.shape[0] for r in resized) + gap * (len(resized) - 1) + 34 * len(resized)
        canvas = np.full((total_h, each_w, 3), 250, np.uint8)
        y = 0
        for r, t in zip(resized, titles):
            _put_text(canvas, t, (10, y + 4))
            y += 34
            canvas[y:y + r.shape[0], :] = r
            y += r.shape[0] + gap
        return canvas

    each_h = 620
    resized = [cv2.resize(im, (int(im.shape[1] * each_h / im.shape[0]), each_h))
               for im in images]
    gap = 16
    total_w = sum(r.shape[1] for r in resized) + gap * (len(resized) - 1)
    canvas = np.full((each_h + 34, total_w, 3), 250, np.uint8)
    x = 0
    for r, t in zip(resized, titles):
        _put_text(canvas, t, (x + 10, 4))
        canvas[34:34 + each_h, x:x + r.shape[1]] = r
        x += r.shape[1] + gap
    return canvas


def _open(path: Path) -> None:
    """OS 기본 뷰어로 띄운다. 실패해도 시연이 멈추지는 않는다."""
    try:
        if os.name == "nt":
            os.startfile(str(path))            # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:                          # noqa: BLE001
        _say(f"  (뷰어를 열지 못했습니다. 직접 열어 주세요: {path})")


def _pick_default() -> Path | None:
    """인자 없이 실행했을 때 쓸 이미지를 찾는다."""
    for cand in (C.INTERMEDIATE / C.CLEAN_DEVIATION_MAP,
                 C.INTERMEDIATE / C.DEVIATION_MAP,
                 C.SAMPLE / "sample_deviation_map.png"):
        if cand.exists():
            return cand
    imgs = sorted(C.INTERMEDIATE.glob("*.png")) + sorted(C.RAW.glob("*.png"))
    imgs = [p for p in imgs if "mask" not in p.name and "overlay" not in p.name]
    return imgs[0] if imgs else None


def run(args: argparse.Namespace) -> int:
    # ── 입력 결정 ────────────────────────────────────────────────
    if args.image:
        src = Path(args.image)
        if not src.is_absolute():
            src = C.ROOT / src
        if not src.exists():
            _say(f"[안내] 파일을 찾지 못했습니다: {src}")
            return 1
    else:
        src = _pick_default()
        if src is None:
            _say("[안내] 보여줄 이미지가 없습니다.")
            _say("       python -m zero_line_detection.make_sample  로 샘플을 먼저 만드세요.")
            return 1

    _say(BAR)
    _say(" 3D 스캔 편차 이미지 → 0-Line 자동 검출")
    _say(BAR)
    _say(f" 입력 : {src.name}")

    try:
        rgb = read_rgb(src)
    except Exception as e:                     # noqa: BLE001
        _say(f"[안내] 이미지를 읽지 못했습니다: {e}")
        return 1
    _say(f" 크기 : {rgb.shape[1]} x {rgb.shape[0]} px")
    _say("")

    # ── 검출 ─────────────────────────────────────────────────────
    cfg = ZeroLineConfig(vmin=args.vmin, vmax=args.vmax, tolerance=args.tolerance)
    t0 = time.perf_counter()
    try:
        out = detect_zero_line(rgb, cfg, source_name=src.name)
    except RuntimeError as e:
        _say("[안내] 컬러바(범례)를 찾지 못했습니다.")
        _say(f"       {e}")
        _say("")
        _say("  이 시스템은 이미지 안의 컬러바를 읽어 색을 편차값으로 바꿉니다.")
        _say("  컬러바가 잘렸거나 지워진 이미지는 처리할 수 없습니다.")
        _say("  → 컬러바가 보이는 원본을 주시면 바로 됩니다.")
        return 1
    except Exception as e:                     # noqa: BLE001
        _say(f"[안내] 처리 중 문제가 생겼습니다: {e}")
        return 1
    elapsed = time.perf_counter() - t0

    r = out.result
    cb = out.colorbar
    i = cb.info

    # ── 콘솔 요약 ────────────────────────────────────────────────
    _say(" [1] 컬러바 자동 인식")
    _say(f"     위치      : 이미지 {'왼쪽' if i.side == 'left' else '오른쪽'}"
         f"  (x = {i.x0} ~ {i.x1})")
    _say(f"     방향      : 최솟값이 {'위' if i.vmin_at == 'top' else '아래'}쪽"
         f"   → 이미지가 {'뒤집혀 있습니다' if i.vmin_at == 'top' else '정방향입니다'}")
    if cb.is_clipped:
        pct = cb.zero_index / max(len(cb.colors_rgb) - 1, 1) * 100
        _say(f"     잘림      : 감지됨. 편차 0 을 컬러바의 {pct:.1f}% 지점으로 자동 보정")
    else:
        _say("     잘림      : 없음. 편차 0 은 컬러바 중앙")
    _say("")

    _say(" [2] 편차 판독 및 0-Line 추출")
    _say(f"     부품 영역 : {r.part_px:,} px")
    _say(f"     0-Line 선 : {r.params.get('zero_crossing_px', 0):,} px"
         "   ← 부호가 바뀌는 경계, 허용오차와 무관")
    _say(f"     0 영역    : {r.total_zero_px:,} px (부품의 {r.zero_ratio * 100:.1f}%)"
         f", 영역 {len(r.regions)}개")
    if r.tolerance_unit == "mm":
        _say(f"     허용오차  : ±{r.tolerance:.2f} mm")
    else:
        _say(f"     허용오차  : ±{r.tolerance:.3f} (정규화 단위)")
        _say("                 컬러바에 적힌 최소·최대값을 알려 주시면 mm 로 바꿔 보여드립니다.")
        _say("                 예)  --vmin -2.0 --vmax 2.0")
    for w in out.warnings:
        _say(f"     [참고] {w}")
    _say("")

    # ── 산출물 ───────────────────────────────────────────────────
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = C.ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    overlay = make_overlay(rgb, out.mask, out.centerline,
                           zero_crossing=out.zero_crossing)
    vmap = make_value_map(out.values, out.part_mask, cb.vmin, cb.vmax)

    if args.rotate:
        rot = cv2.ROTATE_180
        rgb_s, overlay, vmap = (cv2.rotate(x, rot) for x in (rgb, overlay, vmap))
    else:
        rgb_s = rgb

    if args.verify:
        panel = _panel(
            [rgb_s, vmap, overlay],
            ["1. 원본 스캔",
             "2. 색을 편차값으로 되읽은 결과 — 원본과 같으면 색 보정이 맞는 것",
             "3. 0-Line 검출 (검은선 = 부호 경계, 분홍면 = 0 영역)"],
        )
    else:
        panel = _panel(
            [rgb_s, overlay],
            ["1. 원본 스캔",
             "2. 0-Line 검출 (검은선 = 부호 경계, 분홍면 = 0 영역)"],
        )
    panel_path = outdir / f"시연_{src.stem}.png"
    write_rgb(panel_path, panel)

    _say(" [3] 결과")
    _say(f"     처리 시간 : {elapsed:.2f}초")
    _say(f"     결과 이미지 : {panel_path.name}")
    _say("")

    # ── 허용오차 민감도 ──────────────────────────────────────────
    if args.sweep:
        _say(" [4] 허용오차를 바꾸면 0 영역이 얼마나 달라지는가")
        _say("     (0-Line '선' 은 아래 어느 값에서도 똑같습니다)")
        _say("")
        unit = "mm" if r.tolerance_unit == "mm" else ""
        _say(f"       허용오차{unit:>4}     0 영역        부품 대비")
        _say("     " + "─" * 44)
        for row in r.params.get("tolerance_sweep", []):
            bar = "█" * max(int(row["ratio_of_part"] * 40), 1)
            _say(f"       ±{row['tolerance']:.2f}{unit:>4}  {row['area_px']:>8,} px"
                 f"     {row['ratio_of_part'] * 100:5.1f}%  {bar}")
        _say("")
        _say("     → 이 값은 현장 기준으로 정해 주셔야 합니다.")
        _say("")

    _say(BAR)
    if not args.no_open:
        _open(panel_path)
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="현장 시연용 0-Line 검출")
    ap.add_argument("--image", help="스캔 이미지 (생략 시 자동 선택)")
    ap.add_argument("--vmin", type=float, default=None, help="컬러바 최솟값 (mm)")
    ap.add_argument("--vmax", type=float, default=None, help="컬러바 최댓값 (mm)")
    ap.add_argument("--tolerance", type=float, default=None, help="0 판정 허용오차")
    ap.add_argument("--sweep", action="store_true", help="허용오차 민감도 표를 함께 표시")
    ap.add_argument("--rotate", action="store_true", help="결과를 180도 돌려서 표시")
    ap.add_argument("--verify", action="store_true",
                    help="색→값 재변환 패널을 함께 표시 (보정이 맞는지 보여줄 때)")
    ap.add_argument("--outdir", default=str(C.OUTPUT))
    ap.add_argument("--no-open", action="store_true", help="이미지 뷰어를 열지 않는다")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        _say("\n중단했습니다.")
        return 130
    except Exception as e:                     # noqa: BLE001
        _say(f"[안내] 예기치 못한 문제: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
