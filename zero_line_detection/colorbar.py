"""컬러바(범례) 자동 검출 및 색상→편차값 보정.

[왜 필요한가]
3D 스캔 히트맵의 색은 이미지마다 의미가 다르다.

    JD_67XX6 / JD_64XX2 : 컬러바 좌측, 이미지 180° 회전, 범위 +-3.0
    JD_71XX2            : 컬러바 우측, 정상 방향,        범위 +-2.0

따라서 "초록색 = 편차 0" 같은 고정 HSV 범위로는 세 장을 모두 처리할 수 없다.
이 모듈은 이미지 안에 들어 있는 컬러바 자체를 읽어
색 -> 편차값 대응표를 매번 새로 만든다. 사람이 범례를 보고 판독하는 것과 같은 방식이다.

[색상 순서]
무지개 계열 히트맵은 값이 커질수록 다음 순서로 변한다.

    마젠타 -> 파랑 -> 시안 -> 초록 -> 노랑 -> 주황 -> 빨강
    (Hue  150  ->  120 ->  90  ->  60  ->  30  ->  15 ->  0)

즉 Hue 는 최솟값 쪽에서 최댓값 쪽으로 단조 감소한다.
이 성질로 컬러바의 위아래 방향(어느 쪽이 음수인지)을 OCR 없이 판정한다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.schemas import ColorbarInfo  # noqa: E402


# 무지개 램프의 정식 양 끝 색. 이 색에서 멀면 컬러바가 잘린 것으로 본다.
CANONICAL_VMIN_RGB = (255, 0, 255)   # 마젠타 = 최솟값
CANONICAL_VMAX_RGB = (255, 0, 0)     # 빨강   = 최댓값
ENDPOINT_TOL = 70.0                  # RGB 유클리드 거리 허용치


@dataclass
class Colorbar:
    """검출된 컬러바와 색상->값 변환기."""

    info: ColorbarInfo
    colors_rgb: np.ndarray      # (N,3) uint8, 최솟값 -> 최댓값 순서
    lab: np.ndarray             # (N,3) float32, CIELAB (색 거리 계산용)

    # ── 값 축 ────────────────────────────────────────────────────
    @property
    def vmin(self) -> float:
        return self.info.vmin if self.info.vmin is not None else -1.0

    @property
    def vmax(self) -> float:
        return self.info.vmax if self.info.vmax is not None else 1.0

    def index_to_value(self, idx: np.ndarray) -> np.ndarray:
        """컬러바 인덱스 -> 편차값.

        vmin/vmax 가 주어지면 mm 단위, 아니면 -1~+1 정규화 값이다.
        어느 쪽이든 편차 0 은 zero_index 위치에 놓인다.
        """
        n = max(len(self.colors_rgb) - 1, 1)
        t = idx.astype(np.float32) / n
        return (self.vmin + t * (self.vmax - self.vmin)).astype(np.float32)

    # ── 잘림 판정 및 0 위치 ──────────────────────────────────────
    @property
    def endpoint_gaps(self) -> tuple[float, float]:
        """양 끝 색이 정식 램프 끝(마젠타/빨강)에서 얼마나 벗어났는지."""
        lo = float(np.linalg.norm(
            self.colors_rgb[0].astype(float) - np.array(CANONICAL_VMIN_RGB, float)))
        hi = float(np.linalg.norm(
            self.colors_rgb[-1].astype(float) - np.array(CANONICAL_VMAX_RGB, float)))
        return lo, hi

    @property
    def is_clipped(self) -> bool:
        """컬러바가 이미지 경계에서 잘렸는지.

        잘린 컬러바는 눈에 보이는 구간이 전체 범위가 아니므로
        '중앙 = 편차 0' 가정이 성립하지 않는다.
        실제로 JD_64XX2 는 위가 잘려 -1.50 ~ +2.00 의 비대칭 범위이며,
        이 경우 0 은 중앙이 아니라 아래에서 42.9% 지점에 있다.
        """
        lo, hi = self.endpoint_gaps
        return lo > ENDPOINT_TOL or hi > ENDPOINT_TOL

    @property
    def zero_index(self) -> float:
        """편차 0 에 해당하는 컬러바 인덱스."""
        n = len(self.colors_rgb) - 1
        if self.info.vmin is not None and self.info.vmax is not None:
            span = self.info.vmax - self.info.vmin
            if abs(span) < 1e-9:
                raise ValueError("vmin 과 vmax 가 같습니다.")
            return n * (0.0 - self.info.vmin) / span
        return n / 2.0                      # 대칭 범위 가정

    @property
    def half_span(self) -> float:
        """0 에서 양 끝까지 중 짧은 쪽 거리. 허용오차를 비율로 줄 때 기준."""
        if self.info.vmin is not None and self.info.vmax is not None:
            return min(abs(self.info.vmin), abs(self.info.vmax))
        return 1.0

    @property
    def unit(self) -> str:
        return "mm" if (self.info.vmin is not None and self.info.vmax is not None)             else "normalized"

    # ── 색 -> 값 ─────────────────────────────────────────────────
    def map_image(
        self,
        rgb: np.ndarray,
        max_dist: float = 14.0,
        method: str = "hue",
        s_min: int = 55,
        v_min: int = 40,
    ) -> tuple[np.ndarray, np.ndarray]:
        """이미지 전체를 편차값 배열로 변환한다.

        method="hue" (기본)
            색상(Hue) 만으로 매칭한다. 3D 스캔 화면은 조명 음영이 들어가
            같은 편차라도 면 기울기에 따라 밝기가 크게 달라진다.
            밝기를 빼고 색상만 보면 이 음영에 영향을 받지 않는다.
            무지개 램프는 Hue 가 단조 변하므로 np.interp 로 바로 역변환된다.

        method="lab"
            CIELAB 최근접. 음영이 없는 평면 히트맵(예: 파트 4가 만든
            clean_deviation_map)에서 더 정확하다.

        Returns:
            values : (H,W) float32  편차값
            valid  : (H,W) bool     히트맵 색으로 인정된 픽셀
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # 채도·명도가 낮으면 흰 배경, 회색 미측정면, 검은 글씨다
        valid = (sat >= s_min) & (val >= v_min)

        if method == "lab":
            values, close = self._map_lab(rgb, max_dist)
            return values, valid & close

        # ── Hue 기반 ─────────────────────────────────────────────
        bar_hue = _unwrapped_hue(self.colors_rgb)          # (N,) vmin -> vmax
        idx = np.arange(len(bar_hue), dtype=np.float32)

        # np.interp 는 xp 가 증가해야 한다. Hue 는 vmin->vmax 로 감소하므로 뒤집는다.
        xp = bar_hue[::-1].astype(np.float32)
        fp = idx[::-1]
        xp, fp = _make_increasing(xp, fp)

        px_hue = _unwrapped_hue(rgb.reshape(-1, 3)).reshape(rgb.shape[:2])
        matched = np.interp(px_hue, xp, fp).astype(np.float32)
        values = self.index_to_value(matched)

        # 램프 밖 색상(예: 순수 파랑보다 더 파란 값)은 신뢰하지 않는다
        in_range = (px_hue >= xp.min() - 3) & (px_hue <= xp.max() + 3)
        return values, valid & in_range

    def _map_lab(self, rgb: np.ndarray, max_dist: float):
        """CIELAB 최근접 매칭. 고유색만 계산해 되돌려 붙인다."""
        h, w, _ = rgb.shape
        flat = rgb.reshape(-1, 3)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        uniq_lab = _rgb_to_lab(uniq)

        best_idx = np.empty(len(uniq), dtype=np.int32)
        best_dist = np.empty(len(uniq), dtype=np.float32)
        step = 4096
        for s in range(0, len(uniq), step):
            block = uniq_lab[s:s + step]
            d = np.linalg.norm(block[:, None, :] - self.lab[None, :, :], axis=2)
            best_idx[s:s + step] = np.argmin(d, axis=1)
            best_dist[s:s + step] = np.min(d, axis=1)

        values = self.index_to_value(best_idx.astype(np.float32))[inv].reshape(h, w)
        close = (best_dist[inv] <= max_dist).reshape(h, w)
        return values.astype(np.float32), close

    def to_dict(self) -> dict:
        d = self.info.to_dict()
        d["colors_preview"] = {
            "vmin_end": self.colors_rgb[0].tolist(),
            "mid": self.colors_rgb[len(self.colors_rgb) // 2].tolist(),
            "vmax_end": self.colors_rgb[-1].tolist(),
        }
        return d


# ─────────────────────────────────────────────────────────────────
def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N,3) uint8 RGB -> (N,3) float32 CIELAB."""
    arr = rgb.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3)


def _unwrapped_hue(rgb: np.ndarray) -> np.ndarray:
    """Hue 를 0~179 대신 단조 비교가 가능한 형태로 편다.

    빨강은 0 또는 179 양쪽으로 나올 수 있어 그대로 쓰면 순서가 뒤집힌다.
    170 이상은 음수로 돌려 빨강이 항상 축의 한쪽 끝에 오도록 한다.
    """
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv.reshape(-1, 3)[:, 0].astype(np.float32)
    hue[hue > 170] -= 180.0
    return hue


def _make_increasing(xp: np.ndarray, fp: np.ndarray):
    """np.interp 용으로 xp 를 순증가 수열로 다듬는다.

    컬러바를 픽셀에서 읽어 오므로 Hue 가 미세하게 흔들린다.
    누적 최댓값으로 단조성을 강제한 뒤 중복 지점을 합친다.
    """
    xp = np.maximum.accumulate(xp.astype(np.float64))
    keep = np.concatenate(([True], np.diff(xp) > 1e-6))
    return xp[keep].astype(np.float32), fp[keep].astype(np.float32)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위 상관계수. 단조성 판정용 (선형성까지는 요구하지 않는다)."""
    if len(a) < 4:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def detect_colorbar(
    rgb: np.ndarray,
    margin_frac: float = 0.18,
    min_height_frac: float = 0.35,
    max_width_frac: float = 0.06,
    n_samples: int = 256,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Colorbar:
    """이미지 좌우 여백에서 컬러바를 찾는다.

    판정 기준 네 가지를 모두 만족하는 세로 띠를 컬러바로 본다.
      1) 채도가 높은 픽셀이 세로로 길게 이어진다
      2) 가로 폭이 좁다 (이미지 폭의 max_width_frac 이하)
      3) 각 행의 색이 폭 방향으로 균일하다 (컬러바는 행마다 단색)
      4) Hue 가 세로 방향으로 단조 변한다

    3)과 4)가 부품 본체와 컬러바를 가르는 핵심 조건이다.
    부품 가장자리도 채도가 높고 세로로 길지만, 행 방향 색이 균일하지 않고
    Hue 가 단조롭게 변하지도 않는다.
    """
    h, w, _ = rgb.shape
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    colorful = (sat > 90) & (val > 60)

    m = max(int(w * margin_frac), 8)
    candidates = list(range(0, m)) + list(range(w - m, w))
    col_ok = {x: colorful[:, x].mean() > 0.45 for x in candidates}

    # 연속 구간으로 묶기
    groups: list[list[int]] = []
    run: list[int] = []
    for x in candidates:
        if col_ok.get(x):
            if run and x != run[-1] + 1:
                groups.append(run)
                run = []
            run.append(x)
        elif run:
            groups.append(run)
            run = []
    if run:
        groups.append(run)

    best: tuple[float, Colorbar] | None = None
    for g in groups:
        x0, x1 = g[0], g[-1]
        if (x1 - x0 + 1) > w * max_width_frac:
            continue

        rows_ok = colorful[:, x0:x1 + 1].mean(axis=1) > 0.6
        ys = np.where(rows_ok)[0]
        if len(ys) < h * min_height_frac:
            continue
        y0, y1 = int(ys.min()), int(ys.max())

        # 가장자리 열은 흰 배경과 섞여(안티앨리어싱) 색이 흐려진다.
        # 이미지를 축소·확대하거나 화면 캡처로 만든 이미지에는 반드시 생기며,
        # 그대로 두면 아래 균일도 검사에 걸려 컬러바를 놓친다.
        # 따라서 양옆을 잘라내고 안쪽만 보고 판단한다.
        width = x1 - x0 + 1
        pad = min(max(1, int(width * 0.2)), max((width - 1) // 2, 0))
        xa, xb = x0 + pad, x1 - pad
        band = rgb[y0:y1 + 1, xa:xb + 1, :]

        # 3) 행 내부 색 균일도 — 컬러바는 행마다 단색이라 표준편차가 작다
        row_std = band.reshape(band.shape[0], -1, 3).std(axis=1).mean()
        if row_std > 18:
            continue

        row_color = np.median(band, axis=1).astype(np.uint8)   # (L,3)
        hue = _unwrapped_hue(row_color)
        rho = _spearman(np.arange(len(hue), dtype=np.float64), hue.astype(np.float64))
        if abs(rho) < 0.85:                                    # 4) 단조성
            continue

        # rho > 0 : 아래로 갈수록 Hue 증가 = 아래쪽이 최솟값
        vmin_at = "bottom" if rho > 0 else "top"
        ordered = row_color[::-1] if vmin_at == "bottom" else row_color

        # 균일 간격으로 재샘플링
        idx = np.linspace(0, len(ordered) - 1, n_samples).astype(int)
        colors = ordered[idx]

        info = ColorbarInfo(
            side="left" if x0 < w / 2 else "right",
            x0=x0, x1=x1, y0=y0, y1=y1,
            n_samples=len(colors),
            vmin_at=vmin_at,
            vmin=vmin, vmax=vmax,
            symmetric=(vmin is None or vmax is None or abs(vmin + vmax) < 1e-6),
        )
        score = (y1 - y0) * abs(rho)
        cb = Colorbar(info=info, colors_rgb=colors, lab=_rgb_to_lab(colors))
        if best is None or score > best[0]:
            best = (score, cb)

    if best is None:
        raise RuntimeError(
            "컬러바를 찾지 못했습니다. 이미지 좌우 여백에 범례가 보이는지 확인하거나, "
            "--no-colorbar 옵션으로 고정 색상 기준 모드를 사용하세요."
        )
    return best[1]


__all__ = ["Colorbar", "detect_colorbar"]
