"""제로라인 판정 기준 — 스프링백 기준면/기준선을 스캔만으로 찾는다.

[문제]
지금까지는 "편차가 0에 가까운 색" 을 모두 0 영역으로 잡았다. 그런데 그렇게
잡으면 실제로 기준이 될 수 없는 곳까지 섞여 들어온다. 그리고 새 데이터는
보정 시트 없이 3D 스캔 사진만 오므로, 스캔 하나만 보고 판정할 기준이 필요하다.

[스프링백 관점에서 제로라인이란]
성형 후 하중을 빼면 소재가 탄성 복원한다. 이때 부품 전체가 한 덩어리로
움직이는 게 아니라, **거의 움직이지 않는 부분을 축으로 나머지가 돌아간다.**
그 움직이지 않는 부분이 제로라인이고, 보정량을 재는 기준이 된다.
넓으면 기준'면', 좁고 길면 기준'선' 이 된다.

[그래서 무엇을 보아야 하는가]
편차가 0인 것만으로는 부족하다. 두 가지를 함께 봐야 한다.

    1. 편차가 0에 가까울 것          |v| <= 허용오차
    2. 그 주변이 평탄할 것           |∇v| 가 작을 것

2번이 핵심이다. 편차가 0을 스치듯 지나가는 곳(기울기가 큰 곳)은 기준이 될 수
없다. 측정 위치가 몇 mm만 틀어져도 값이 확 달라지기 때문이다. 반대로 평탄한
곳은 위치가 조금 흔들려도 값이 그대로라서 기준으로 쓸 수 있다.

계측에서 데이텀을 평평한 면에 잡는 것과 같은 이유다.

[면인가 선인가]
평탄한 0 영역을 찾은 뒤 형태로 나눈다.

    면 (plateau)   넓고 뭉툭하다. 폼 좌면처럼 통째로 기준이 되는 면
    선 (ridge)     좁고 길다. 능선처럼 이어지는 기준선

[기울기 문턱값을 백분위로 잡는 이유]
절대값으로 정하면 이미지마다 편차 범위(+-2mm / +-3mm)와 해상도가 달라
그대로 쓸 수 없다. 0 밴드 안에서의 기울기 분포를 기준으로 하위 몇 %를
평탄하다고 볼지 정하면 이미지가 바뀌어도 같은 의미를 유지한다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class ZeroCandidate:
    """제로라인 후보 영역 하나."""

    candidate_id: int
    kind: str                 # "plateau" (면) 또는 "ridge" (선)
    area_px: int
    centroid_x: float
    centroid_y: float
    bbox: tuple               # (x, y, w, h)
    mean_deviation: float     # 영역 평균 편차
    max_abs_deviation: float  # 영역 내 최대 |편차|
    mean_gradient: float      # 영역 평균 기울기 (작을수록 평탄)
    elongation: float         # 길쭉한 정도. 1에 가까우면 뭉툭
    nearness: float           # 0에 얼마나 가까운가 (0~1)
    flatness: float           # 얼마나 평탄한가 (0~1)
    score: float              # 종합 점수 (0~1). 기준으로 삼기 좋은 정도

    def to_dict(self) -> dict:
        return asdict(self)


def deviation_gradient(values: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """편차장의 기울기 크기.

    먼저 흐리게 만든다. 스캔 이미지는 색 양자화 때문에 픽셀 단위로 값이
    계단처럼 튀는데, 그대로 미분하면 계단마다 큰 기울기가 잡혀 평탄한
    면까지 울퉁불퉁하게 나온다.
    """
    smooth = cv2.GaussianBlur(values.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=5)
    return np.sqrt(gx * gx + gy * gy)


def find_zero_candidates(
    values: np.ndarray,
    part_mask: np.ndarray,
    tolerance: float,
    flat_percentile: float = 40.0,
    min_area: int = 300,
    close_ksize: int = 7,
    ridge_elongation: float = 3.0,
) -> tuple:
    """제로라인 후보를 찾고 면/선으로 나눈다.

    Args:
        values:           편차값 배열
        part_mask:        부품 영역
        tolerance:        0 으로 볼 허용오차
        flat_percentile:  0 밴드 안 기울기 분포의 하위 몇 % 를 평탄으로 볼지
        min_area:         이보다 작은 후보는 버린다
        close_ksize:      잔구멍 메우기 강도
        ridge_elongation: 이 이상 길쭉하면 '선', 아니면 '면'

    Returns:
        (후보 목록, 평탄 마스크, 기울기 배열)
    """
    grad = deviation_gradient(values)
    zero_band = (np.abs(values) <= tolerance) & part_mask

    if not zero_band.any():
        return [], np.zeros_like(part_mask, dtype=bool), grad

    # 기울기 문턱값은 0 밴드 안에서의 분포로 정한다 (이미지마다 자동 적응)
    grad_threshold = float(np.percentile(grad[zero_band], flat_percentile))
    flat = zero_band & (grad <= grad_threshold)

    if close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize,) * 2)
        flat = cv2.morphologyEx(flat.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        flat &= part_mask

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        flat.astype(np.uint8), connectivity=8
    )

    candidates: list = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        region = labels == i
        vals = values[region]
        grads = grad[region]

        # 길쭉한 정도 — 최소 외접 회전 사각형의 장변/단변
        contours, _ = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        elong = 1.0
        if contours:
            (_, (w, h), _) = cv2.minAreaRect(max(contours, key=cv2.contourArea))
            short, long_ = sorted((w, h))
            elong = float(long_ / short) if short > 1e-6 else 999.0

        mean_abs = float(np.abs(vals).mean())
        nearness = max(0.0, 1.0 - mean_abs / max(tolerance, 1e-9))
        flatness = max(0.0, 1.0 - float(grads.mean()) / max(grad_threshold, 1e-9))
        # 면적은 로그로 반영한다. 넓을수록 좋지만 선형으로 주면 큰 덩어리 하나가
        # 나머지를 전부 눌러 버린다.
        size_w = float(np.log10(area) / 5.0)
        score = float(np.clip(nearness * 0.4 + flatness * 0.35 + size_w * 0.25, 0, 1))

        candidates.append(ZeroCandidate(
            candidate_id=0,
            kind="ridge" if elong >= ridge_elongation else "plateau",
            area_px=area,
            centroid_x=round(float(centroids[i][0]), 1),
            centroid_y=round(float(centroids[i][1]), 1),
            bbox=(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                  int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
            mean_deviation=round(float(vals.mean()), 4),
            max_abs_deviation=round(float(np.abs(vals).max()), 4),
            mean_gradient=round(float(grads.mean()), 4),
            elongation=round(elong, 2),
            nearness=round(nearness, 3),
            flatness=round(flatness, 3),
            score=round(score, 3),
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    for i, c in enumerate(candidates, start=1):
        c.candidate_id = i
    return candidates, flat, grad


def candidates_to_mask(candidates: list, flat: np.ndarray, top_n: int | None = None) -> np.ndarray:
    """상위 후보만 남긴 마스크. top_n 이 None 이면 전부."""
    picked = candidates if top_n is None else candidates[:top_n]
    if not picked:
        return np.zeros(flat.shape, dtype=np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        flat.astype(np.uint8), connectivity=8
    )
    out = np.zeros(flat.shape, dtype=np.uint8)
    wanted = {(c.bbox, c.area_px) for c in picked}
    for i in range(1, n):
        key = ((int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
               int(stats[i, cv2.CC_STAT_AREA]))
        if key in wanted:
            out[labels == i] = 255
    return out


__all__ = [
    "ZeroCandidate", "deviation_gradient",
    "find_zero_candidates", "candidates_to_mask",
]
