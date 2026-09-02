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

다만 정규화만 하면 안 된다. 포인트가 하나일 때 반경 안이 통째로 같은
값이 되어 원판을 찍어 놓은 꼴이 되고, 반경 끝에서 절벽이 생긴다.
값(정규화한 보간)과 덮개(포인트에 얼마나 가까운가)를 나눠 구해 곱한다.

[거리는 직선이 아니라 표면을 따라 잰다]
판금은 접혀 있다. 플랜지를 세워 두면 직선거리로는 붙어 있어도 **판을
따라가면 멀다.** 직선거리로 재면 보정 포인트 하나가 틈 건너편 살까지
같이 밀어 버려서, 손대지 않아야 할 자리가 움직인다.

실측 71XX1-DR000(정점 183,821 · 삼각형 225,358 · 대각선 2083mm,
반경 83mm)에서 포인트 하나가 미는 정점을 세어 보면 —

    직선거리   6,720 개
    표면거리   1,037 개   (직선으로 잡히던 것의 85% 가 건너편이었다)

그래서 메시의 모서리를 따라 다익스트라로 거리를 잰다. 반경 안에서만
퍼지므로 18만 정점 · 포인트 8개에 1.0초다. scipy 가 없으면 직선거리로
물러난다 — 없는 것보다는 낫다.
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
    faces: np.ndarray | None = None,
) -> np.ndarray:
    """정점마다 밀어낼 양(mm)을 구한다.

    Args:
        vertices: (N, 3) 정점.
        normals: (N, 3) 정점 법선.
        spots: (M, 3) 보정 포인트 위치.
        values: (M,) 보정량(mm). 양수면 살을 붙이는 쪽이다.
        reach: 영향 반경(mm).
        faces: 주면 표면을 따라 거리를 잰다. 없으면 직선거리다.
    """
    if not len(spots):
        return np.zeros(len(vertices), dtype=float)

    # 표면을 따라 잰 거리. 못 재면 None 이고, 그때는 직선거리를 쓴다.
    along = surface_distances(vertices, faces, spots, reach)         if faces is not None else None

    shift = np.zeros(len(vertices), dtype=float)
    weight_sum = np.zeros(len(vertices), dtype=float)
    # 가장자리에서 부드럽게 0 으로 내리는 덮개.
    #
    # 역거리 가중을 **정규화만** 하면 포인트가 하나일 때 반경 안이
    # 통째로 같은 값이 된다 — 나누는 순간 가중치가 지워지기 때문이다.
    # 그러면 보정 후 형상이 원판을 찍어 놓은 것처럼 되고 반경 끝에서
    # 절벽이 생긴다. 금형 형상으로 쓸 수 없다.
    #
    # 그래서 값과 덮개를 나눠서 구한다 — 값은 정규화한 보간값이고,
    # 덮개는 "이 자리가 어느 포인트에든 얼마나 가까운가" 다.
    # 1 - Π(1 - w) 는 포인트 자리에서 1, 모든 반경 밖에서 0 이고,
    # 가중치가 양 끝에서 기울기 0 이라 이어 붙인 자리도 매끄럽다.
    cover = np.ones(len(vertices), dtype=float)
    for index, (spot, value) in enumerate(zip(spots, values)):
        distance = (along[index] if along is not None
                    else np.linalg.norm(vertices - spot, axis=1))
        near = distance < reach
        if not near.any():
            continue
        # 반경 끝에서 0 이 되도록 정규화한 거리로 가중치를 만든다
        ratio = np.clip(distance[near] / reach, 0.0, 1.0)
        # (1-r^2)^2 은 r=0 과 r=1 **양쪽에서** 기울기가 0 이다.
        # 예전 (1-r)^power 는 포인트 자리에서 기울기가 살아 있어
        # 뾰족한 꼭짓점이 남았다 — 밀어낸 자리가 원뿔처럼 보였다.
        weight = (1.0 - ratio ** 2) ** power
        shift[near] += weight * value
        weight_sum[near] += weight
        cover[near] *= (1.0 - weight)

    busy = weight_sum > 1e-9
    shift[busy] /= weight_sum[busy]
    shift[~busy] = 0.0
    return shift * (1.0 - cover)


def surface_distances(vertices: np.ndarray, faces: np.ndarray,
                      spots: np.ndarray, reach: float):
    """보정 포인트마다 "판을 따라간 거리" 를 잰다.

    Returns:
        (M, N) 거리 배열. 반경 밖은 무한대. 잴 수 없으면 None.
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
    except Exception:
        return None

    faces = np.asarray(faces).reshape(-1, 3)
    if not len(faces):
        return None

    # 먼저 겹친 정점을 합친다.
    #
    # STEP 을 잘게 나눌 때 면마다 따로 삼각형을 만들어서, 맞닿은 면의
    # 경계 정점이 **같은 자리에 두 개씩** 있다. 그대로 그래프를 만들면
    # 면과 면 사이가 끊겨 거리가 그 면 안에서 멈춘다 — 실측 71XX1 에서
    # 반경 83mm 안에 정점이 45 개밖에 안 잡혔다(작은 면 하나 크기다).
    span = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    grid = max(span * 1e-6, 1e-9)
    _keys, welded = np.unique(np.round(vertices / grid).astype(np.int64),
                              axis=0, return_inverse=True)
    welded = np.asarray(welded).ravel()
    count = int(welded.max()) + 1

    # 삼각형 세 변을 그래프의 간선으로 쓴다. 가중치는 변의 길이다.
    raw = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    length = np.linalg.norm(vertices[raw[:, 0]] - vertices[raw[:, 1]], axis=1)
    edges = welded[raw]

    graph = coo_matrix(
        (np.concatenate([length, length]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(count, count)).tocsr()

    # 각 포인트에서 가장 가까운 정점을 출발점으로 삼는다
    seeds = [int(welded[int(np.argmin(np.linalg.norm(vertices - spot, axis=1)))])
             for spot in spots]
    # limit 를 주면 그 밖은 계산하지 않는다 — 30만 정점이어도 빠르다
    found = dijkstra(graph, directed=False, indices=seeds, limit=reach)
    # 합친 좌표계의 거리를 원래 정점 번호로 되돌린다
    return np.asarray(found)[:, welded]


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

    shift = displacement_field(vertices, normals, spots, values, reach,
                               faces=np.asarray(faces))
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
    "MorphResult", "WorkVolume", "WORK_FLOOR_MM",
    "displacement_field", "morph", "split_by_process",
    "surface_distances", "work_volumes",
]


# ── 공정별 물량 ──────────────────────────────────────────────────
# 보정량의 **부호가 곧 공정**이다.
#   + 살을 붙인다  -> 용접(덧살)
#   - 살을 깎는다  -> CNC 가공
# 두 일은 작업자도 견적도 다르므로 물량을 따로 내야 한다.

# 이보다 얇게 밀린 자리는 물량에서 뺀다(mm). 측정 잡음과 보간 꼬리가
# 넓은 면적에 얇게 깔려 부피를 부풀린다.
WORK_FLOOR_MM = 0.05


@dataclass
class WorkVolume:
    """한 공정의 물량."""

    kind: str               # "weld"(용접) | "cut"(가공)
    area_mm2: float         # 손대는 면 넓이
    volume_mm3: float       # 붙이거나 깎는 부피
    max_mm: float           # 가장 두꺼운 자리
    mean_mm: float
    faces: int              # 해당하는 삼각형 수

    def to_dict(self) -> dict:
        return asdict(self)


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def work_volumes(vertices, faces, shift, floor_mm: float = WORK_FLOOR_MM):
    """민 양을 공정별(용접/가공) 물량으로 나눈다.

    부피는 삼각형마다 `넓이 x 그 삼각형 세 꼭짓점의 평균 이동량` 을 더해
    구한다. 얇은 껍질을 법선 방향으로 미는 것이므로 이 근사가 맞다.

    Args:
        shift: 정점별 이동량(mm). 양수면 살을 붙이는 쪽.
        floor_mm: 이보다 얇으면 물량에서 뺀다.

    Returns:
        [WorkVolume, ...] — 해당하는 것만. 없으면 빈 목록.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces).reshape(-1, 3)
    shift = np.asarray(shift, dtype=float).reshape(-1)
    if not len(faces) or not len(shift):
        return []

    areas = _triangle_areas(vertices, faces)
    per_face = shift[faces].mean(axis=1)

    out: list = []
    for kind, picked in (("weld", per_face > floor_mm),
                         ("cut", per_face < -floor_mm)):
        if not picked.any():
            continue
        thick = np.abs(per_face[picked])
        face_area = areas[picked]
        out.append(WorkVolume(
            kind=kind,
            area_mm2=round(float(face_area.sum()), 1),
            volume_mm3=round(float((face_area * thick).sum()), 1),
            max_mm=round(float(thick.max()), 3),
            mean_mm=round(float(np.average(thick, weights=face_area)), 3),
            faces=int(picked.sum()),
        ))
    return out


def split_by_process(faces, shift, floor_mm: float = WORK_FLOOR_MM):
    """삼각형을 공정별로 갈라 준다 — 각각 따로 내보내려고.

    Returns:
        {"weld": (M,3) 면, "cut": (K,3) 면}
    """
    faces = np.asarray(faces).reshape(-1, 3)
    shift = np.asarray(shift, dtype=float).reshape(-1)
    if not len(faces) or not len(shift):
        return {"weld": faces[:0], "cut": faces[:0]}
    per_face = shift[faces].mean(axis=1)
    return {"weld": faces[per_face > floor_mm],
            "cut": faces[per_face < -floor_mm]}
