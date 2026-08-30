"""보정량만큼 형상을 밀어 "보정 후" 메시를 만든다.

[할 수 있는 것과 없는 것]
원본 CAD 는 자유곡면(NURBS)이다. 그 제어점을 보정량만큼 움직여 면
연속성(G1/G2)을 지키며 변형하는 것은 전용 소프트웨어의 영역이다.
어설프게 하면 인접 면과 단차가 생겨 **가공할 수 없는 형상**이 나온다.
그래서 여기서는 B-Rep 을 건드리지 않는다.

대신 삼각망을 민다. 정점마다 보정량을 보간해 법선 방향으로 옮긴다.
결과는 메시라 그대로 가공에 쓸 수는 없지만,

    - 보정 후 형상이 어떻게 되는지 눈으로 본다
    - 원본과 겹쳐 어디가 얼마나 달라지는지 잰다
    - STL 로 내보내 다른 도구에 넘긴다

는 할 수 있다. 비교가 목적이면 이걸로 충분하다.

[보간을 어떻게 하나]
보정 포인트는 수십 개고 정점은 수십만 개다. 그 사이를 메워야 한다.
역거리 가중(inverse distance weighting)을 쓴다 — 가까운 포인트일수록
크게 반영하고, 영향 반경 밖은 0 으로 떨어뜨린다. 반경을 두지 않으면
부품 반대편 보정량까지 옅게 섞여 형상이 전체적으로 부푼다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# 보정 포인트 하나가 영향을 미치는 거리. 부품 대각선 대비 비율이다.
#
# [0.18 로 시작했다가 0.04 로 내렸다 — 실측 71XX1, 대각선 2083mm]
# 반경이 넓으면 이웃한 반대 부호 보정이 서로 상쇄돼 **지정한 보정량이
# 그대로 안 들어간다.** 포인트 자리에서 값이 얼마나 재현되는지 재보면 —
#
#     비율   반경mm   밀린 정점   최대 이동   포인트에서 값 재현
#     0.02      42       9.3%      2.000            100%
#     0.04      83      33.2%      2.000            100%
#     0.06     125      57.1%      2.000            100%
#     0.10     208      86.4%      2.000             85%
#     0.18     375     100.0%      1.917             53%
#
# 0.18 은 보정량을 절반만 반영한다 — 비교용으로도 못 쓴다. 0.04 면
# 값을 그대로 재현하면서 부품의 3분의 1만 건드린다.
DEFAULT_REACH_RATIO = 0.04
# 가중치가 급격히 떨어지는 정도. 클수록 포인트 주변만 뾰족하게 밀린다.
FALLOFF_POWER = 2.0


@dataclass
class MorphResult:
    """보정 후 형상과 원본의 차이."""

    moved: int              # 실제로 밀린 정점 수
    max_shift: float        # 가장 많이 밀린 양(mm)
    mean_shift: float
    reach_mm: float

    def to_dict(self) -> dict:
        return asdict(self)


def displacement_field(
    vertices: np.ndarray,
    normals: np.ndarray,
    spots: np.ndarray,
    values: np.ndarray,
    reach: float,
    power: float = FALLOFF_POWER,
) -> np.ndarray:
    """정점마다 밀어낼 양(mm)을 구한다.

    Args:
        vertices: (N, 3) 정점.
        normals: (N, 3) 정점 법선.
        spots: (M, 3) 보정 포인트 위치.
        values: (M,) 보정량(mm). 양수면 살을 붙이는 쪽이다.
        reach: 영향 반경(mm).
    """
    if not len(spots):
        return np.zeros(len(vertices), dtype=float)

    shift = np.zeros(len(vertices), dtype=float)
    weight_sum = np.zeros(len(vertices), dtype=float)
    for spot, value in zip(spots, values):
        distance = np.linalg.norm(vertices - spot, axis=1)
        near = distance < reach
        if not near.any():
            continue
        # 반경 끝에서 0 이 되도록 정규화한 거리로 가중치를 만든다
        ratio = np.clip(distance[near] / reach, 0.0, 1.0)
        weight = (1.0 - ratio) ** power
        shift[near] += weight * value
        weight_sum[near] += weight

    busy = weight_sum > 1e-9
    shift[busy] /= weight_sum[busy]
    shift[~busy] = 0.0
    return shift


def morph(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    spots,
    values,
    reach_ratio: float = DEFAULT_REACH_RATIO,
) -> tuple:
    """보정량만큼 민 정점과 그 통계를 준다.

    Returns:
        (밀린 정점 (N,3), 정점별 이동량 (N,), MorphResult)
    """
    vertices = np.asarray(vertices, dtype=float)
    normals = np.asarray(normals, dtype=float)
    spots = np.asarray(spots, dtype=float).reshape(-1, 3)
    values = np.asarray(values, dtype=float).reshape(-1)

    span = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    reach = max(span * reach_ratio, 1e-6)

    shift = displacement_field(vertices, normals, spots, values, reach)
    moved = vertices + normals * shift[:, None]

    touched = np.abs(shift) > 1e-6
    result = MorphResult(
        moved=int(touched.sum()),
        max_shift=round(float(np.abs(shift).max()) if len(shift) else 0.0, 4),
        mean_shift=round(
            float(np.abs(shift[touched]).mean()) if touched.any() else 0.0, 4),
        reach_mm=round(reach, 2),
    )
    return moved, shift, result


__all__ = [
    "DEFAULT_REACH_RATIO", "FALLOFF_POWER",
    "MorphResult", "displacement_field", "morph",
]
