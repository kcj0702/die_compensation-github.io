"""VLM이 라벨에서 읽은 실측값으로 컬러바 기반 편차값장을 보정한다.

[왜 필요한가]
detect_zero_line() 의 값(values)은 컬러바 색상 보간만으로 만든 것이라,
컬러바가 이미지 경계에서 잘렸거나(클리핑) JPEG 압축으로 색이 살짝 틀어지면
전체 값장이 같은 방향으로 밀린다. 지금까지는 부품마다 --vmin/--vmax 를
수동으로 넣어야 했다(JD_64XX2 는 -1.5/2.0 — zero_line_detection/README.md).

반면 VLM(Qwen)이 라벨 숫자를 직접 읽은 값(points, confidence=="ok")은
사람이 시트에 적어둔 실측값 그 자체다. 이 점들에서 "컬러바 추정값"과
"VLM 실측값" 을 비교해 선형 보정(스케일+오프셋)을 구하면, 부품마다
수동으로 vmin/vmax 를 맞출 필요 없이 같은 효과를 자동으로 낼 수 있다.

[왜 그냥 최소제곱이 아니라 RANSAC 인가]
실측 검증(2026-08-24, JD_64XX2/DASH UPR): VLM 판독 64개 중 약 20%(13개)가
"8.000", "9.000mm" 처럼 물리적으로 말이 안 되는 값이었다 — 라벨 숫자를
잘못 읽은 것으로 보인다(부호나 소수점을 놓친 듯). 이런 점 몇 개만 있어도
일반 최소제곱은 심하게 틀어진다(실측 테스트: 잔차 평균 2.4mm).

컬러바 자체가 잘린 경우(클리핑), 잘린 구간 근처 픽셀은 색상만으로는
채도가 포화돼 실제 값을 복원할 수 없다 — 이런 점도 선형관계에서 벗어난
이상치로 나타난다.

RANSAC(무작위로 점 2개씩 뽑아 직선을 만들고, 그 직선에 잘 맞는 점이
가장 많은 조합을 채택 후 그 점들로만 재적합)을 쓰면 이 두 종류의 이상치를
전부 자동으로 걸러낸다. 같은 데이터로 검증: 인라이어 33/64, 잔차 평균
0.10mm, 복원된 vmin/vmax(-1.64/2.00)가 수동으로 넣었던 값(-1.5/2.0)과
거의 일치했다.
"""

from __future__ import annotations

import numpy as np


def _ransac_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    inlier_thresh: float,
    n_iters: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray] | None:
    """무작위 점 2개로 직선을 그어보며, 잔차가 작은 점(인라이어)이 가장
    많은 직선을 찾는다. scipy/sklearn 없이 순수 numpy로 충분히 빠르다
    (점이 수십~수백 개 수준이라 n_iters 회 반복해도 수 ms).
    """
    n = len(x)
    if n < 2:
        return None
    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(n_iters):
        i, j = rng.choice(n, size=2, replace=False)
        if x[i] == x[j]:
            continue
        scale = (y[j] - y[i]) / (x[j] - x[i])
        offset = y[i] - scale * x[i]
        residual = np.abs(y - (scale * x + offset))
        inliers = residual < inlier_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None:
        return None
    xi, yi = x[best_inliers], y[best_inliers]
    a = np.vstack([xi, np.ones_like(xi)]).T
    (scale, offset), *_ = np.linalg.lstsq(a, yi, rcond=None)
    return float(scale), float(offset), best_inliers


def calibrate_with_points(
    values: np.ndarray,
    points: list,
    min_points: int = 6,
    inlier_thresh: float = 0.4,
    n_iters: int = 500,
    min_inliers: int = 4,
) -> tuple[np.ndarray, dict | None]:
    """values 를 points 의 실측값에 맞춰 선형 보정한다(스케일+오프셋).

    VLM 오독점·컬러바 포화점을 RANSAC 으로 걸러낸 뒤, 남은 점들로만
    최종 직선을 적합한다.

    Returns:
        (보정된 values, 보정 통계) — 유효한 점이 min_points 개 미만이거나
        RANSAC 이 충분한 인라이어를 못 찾으면 원본 values 를 그대로
        돌려주고 통계는 None.
    """
    ok = [p for p in points if p.get("confidence") == "ok"]
    if len(ok) < min_points:
        return values, None

    h, w = values.shape
    xs = np.clip(np.array([p["xPx"] for p in ok], dtype=int), 0, w - 1)
    ys = np.clip(np.array([p["yPx"] for p in ok], dtype=int), 0, h - 1)
    vlm_vals = np.array([p["value"] for p in ok], dtype=float)
    colorbar_vals = values[ys, xs]

    finite = np.isfinite(colorbar_vals)
    if finite.sum() < min_points:
        return values, None

    cb_vals = colorbar_vals[finite]
    vlm = vlm_vals[finite]

    rng = np.random.default_rng(0)  # 재현 가능하도록 고정 시드
    fit = _ransac_linear_fit(cb_vals, vlm, inlier_thresh, n_iters, rng)
    if fit is None or int(fit[2].sum()) < min_inliers:
        return values, None
    scale, offset, inliers = fit

    corrected = values * scale + offset
    residual = vlm[inliers] - (cb_vals[inliers] * scale + offset)
    stats = {
        "nPoints": int(finite.sum()),
        "nInliers": int(inliers.sum()),
        "scale": round(scale, 4),
        "offset": round(offset, 4),
        "residualMeanAbs": round(float(np.abs(residual).mean()), 4),
        "residualMax": round(float(np.abs(residual).max()), 4),
    }
    return corrected.astype(values.dtype), stats


__all__ = ["calibrate_with_points"]
