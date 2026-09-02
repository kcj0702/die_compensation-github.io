"""스캔을 CAD 데이텀에 맞춰 세운다 — 작업자마다 달라지지 않게.

[무엇을 푸는가]
현행 문제는 "작업자마다 제로라인이 다르다" 인데, 그 뿌리는 **정렬**이다.
검사 소프트웨어에서 스캔을 CAD 에 얹을 때 사람이 손으로 맞추므로 기준
좌표계가 매번 조금씩 틀어진다. 기준이 흔들리면 같은 제품을 재도 수치와
보정 방향이 달라진다.

여기서는 그 과정을 계산으로 고정한다. 사람이 개입하는 곳은 **어느 홀이
데이텀인가** 하나뿐이고, 그 뒤는 전부 결정론이다. 같은 입력이면 항상
같은 행렬이 나온다.

[두 단계로 맞춘다]
1. 데이텀 정합 (Kabsch/Umeyama)
   대응하는 점 3개 이상이면 회전+평행이동이 닫힌 형태로 **한 번에**
   나온다. 반복도 초기값도 필요 없다. RPS 가 정확히 이 방식이다.
2. ICP 다듬기 (선택)
   데이텀 좌표 자체에 측정 오차가 있으면 1번 결과가 조금 남는다.
   표면 전체를 써서 그 나머지를 줄인다. 다만 **1번이 틀리면 2번은
   엉뚱한 곳으로 수렴한다** — ICP 는 가까운 점을 찾는 것이라 초기값이
   나쁘면 국소 최적에 빠진다. 그래서 1번을 먼저 하고 2번은 다듬기로만
   쓴다.

[소수점 4자리는 약속할 수 없다]
"완벽히 일치" 는 두 데이텀 삼각형이 합동일 때만 성립한다. 실제 측정
데이텀에는 오차가 있어서 잔차가 남는다. 그래서 이 모듈은 정확도를
주장하지 않고 **재서 돌려준다**(datum_rmse · surface_rmse). 쓰는 쪽이
공차를 정해 판정한다. 숫자를 숨기고 "맞췄다" 고 하는 게 제일 위험하다.

[의존성]
trimesh 만 쓴다. Open3D 도 같은 일을 하지만 이 프로젝트가 이미 trimesh
로 메시를 다루고 있어 새로 들일 이유가 없다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

# 데이텀은 최소 3점이어야 자세가 정해진다. 2점이면 그 축 둘레로 자유롭다.
MIN_DATUMS = 3
# 데이텀 세 점이 일직선에 가까우면 회전이 불안정하다. 삼각형 넓이를
# 변 길이로 정규화해 이 값보다 작으면 거부한다.
MIN_SPREAD = 0.02
DEFAULT_ICP_ITERS = 60
# ICP 에 넣을 점 수 상한.
#
# 스캔 정점을 전부 넣으면 못 쓴다 — 실측 64XX1 은 정점이 302,340 개라
# 반복마다 그만큼 "메시 위 가장 가까운 점" 을 찾아야 하고, 4분이 넘도록
# 끝나지 않았다. ICP 는 자세를 다듬는 것이라 점이 그렇게 많이 필요하지
# 않다. 고르게 뽑아 쓴다.
ICP_SAMPLE = 4000


@dataclass
class AlignResult:
    """정렬 결과와 그 품질."""

    matrix: list                 # 4x4. 스캔 좌표 -> CAD 좌표
    datum_rmse: float            # 데이텀 점들이 얼마나 남았나 (mm)
    datum_max: float             # 그중 가장 나쁜 점 (mm)
    surface_rmse: float | None   # 표면 전체 (ICP 를 돌렸을 때만)
    used_icp: bool
    datum_count: int
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def apply(self, points) -> np.ndarray:
        """스캔 좌표를 CAD 좌표로 옮긴다."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        matrix = np.asarray(self.matrix, dtype=float)
        return pts @ matrix[:3, :3].T + matrix[:3, 3]


def _spread(points: np.ndarray) -> float:
    """세 점이 삼각형을 이루는 정도. 일직선이면 0."""
    if len(points) < 3:
        return 0.0
    edges = points[1:] - points[0]
    area = 0.0
    for i in range(len(edges) - 1):
        area = max(area, float(np.linalg.norm(np.cross(edges[i], edges[i + 1]))))
    longest = float(np.max([np.linalg.norm(e) for e in edges])) or 1.0
    return area / (longest * longest)


def rigid_from_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """대응하는 점 무리를 겹치는 4x4 강체 변환 (Kabsch).

    반복이 없다 — 닫힌 형태로 한 번에 나온다. 그래서 초기값도, 수렴
    걱정도 없고 같은 입력이면 항상 같은 답이다.

    Args:
        source: (N,3) 스캔 쪽 점.
        target: (N,3) CAD 쪽 점. 순서가 source 와 짝이어야 한다.
    """
    src = np.asarray(source, dtype=float).reshape(-1, 3)
    dst = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(src) != len(dst):
        raise ValueError(f"점 수가 다릅니다: 스캔 {len(src)} · CAD {len(dst)}")
    if len(src) < MIN_DATUMS:
        raise ValueError(f"데이텀이 {MIN_DATUMS}점 이상이어야 자세가 정해집니다.")

    src_mid = src.mean(axis=0)
    dst_mid = dst.mean(axis=0)
    covariance = (src - src_mid).T @ (dst - dst_mid)
    u, _s, vt = np.linalg.svd(covariance)

    # 거울상(det = -1)이 나올 수 있다. 부품을 뒤집어 맞추면 안 되므로
    # 마지막 축을 뒤집어 회전으로 되돌린다.
    flip = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        flip[2, 2] = -1.0
    rotation = vt.T @ flip @ u.T

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = dst_mid - rotation @ src_mid
    return matrix


def _rmse(a: np.ndarray, b: np.ndarray) -> tuple:
    gap = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)
    return float(np.sqrt((gap ** 2).mean())), float(gap.max())


def align(scan_points, cad_datums, scan_datums,
          scan_mesh=None, cad_mesh=None,
          use_icp: bool = True,
          icp_iterations: int = DEFAULT_ICP_ITERS) -> AlignResult:
    """스캔을 CAD 좌표계로 세운다.

    Args:
        scan_points: 옮길 스캔 점들 (N,3). 메시 정점이어도 되고 점군이어도 된다.
        cad_datums: CAD 쪽 데이텀 좌표 (M,3). step_reader 가 뽑은 홀 중심을
            그대로 쓸 수 있다.
        scan_datums: 스캔 쪽 같은 데이텀 (M,3). 순서가 cad_datums 와 짝이어야
            한다 — 여기서 짝을 잘못 지으면 나머지가 전부 틀어진다.
        scan_mesh, cad_mesh: 주면 ICP 로 다듬는다 (trimesh.Trimesh).
        use_icp: 다듬기를 할지.

    Returns:
        AlignResult. 정확도를 주장하지 않고 **재서** 돌려준다.
    """
    cad_pts = np.asarray(cad_datums, dtype=float).reshape(-1, 3)
    scan_pts = np.asarray(scan_datums, dtype=float).reshape(-1, 3)
    notes: list = []

    spread = min(_spread(cad_pts), _spread(scan_pts))
    if spread < MIN_SPREAD:
        notes.append(
            f"데이텀 점들이 일직선에 가깝습니다(퍼짐 {spread:.4f}). "
            "그 축 둘레로 자세가 흔들립니다 — 떨어진 세 점을 고르세요.")

    matrix = rigid_from_points(scan_pts, cad_pts)
    moved = scan_pts @ matrix[:3, :3].T + matrix[:3, 3]
    datum_rmse, datum_max = _rmse(moved, cad_pts)

    surface_rmse = None
    used_icp = False
    if use_icp and scan_mesh is not None and cad_mesh is not None:
        try:
            from trimesh.registration import icp

            cloud = np.asarray(scan_mesh.vertices, dtype=float)
            if len(cloud) > ICP_SAMPLE:
                # 고르게 솎는다. 무작위로 뽑으면 같은 입력에 다른 답이
                # 나와서, 이 모듈이 지키려는 "항상 같은 결과" 가 깨진다.
                step = int(np.ceil(len(cloud) / ICP_SAMPLE))
                cloud = cloud[::step]
            refined, _moved, cost = icp(
                cloud, cad_mesh, initial=matrix,
                max_iterations=int(icp_iterations),
                scale=False)          # 배율은 고정 — 부품이 늘어나면 안 된다
            candidate = np.asarray(refined, dtype=float)
            check = scan_pts @ candidate[:3, :3].T + candidate[:3, 3]
            check_rmse, check_max = _rmse(check, cad_pts)

            # ICP 가 데이텀을 되레 밀어내면 쓰지 않는다. 표면을 잘 맞추려고
            # 기준을 버리면 본말이 뒤집힌다 — 우리가 지키려는 게 기준이다.
            if check_rmse <= datum_rmse * 1.5:
                matrix = candidate
                datum_rmse, datum_max = check_rmse, check_max
                surface_rmse = float(cost)
                used_icp = True
            else:
                notes.append(
                    f"ICP 가 데이텀을 밀어내 쓰지 않았습니다 "
                    f"({datum_rmse:.4f} -> {check_rmse:.4f} mm).")
        except Exception as exc:      # ICP 는 어디까지나 다듬기다
            notes.append(f"ICP 를 건너뛰었습니다: {exc}")

    return AlignResult(
        matrix=[[round(float(v), 8) for v in row] for row in matrix],
        datum_rmse=round(datum_rmse, 5),
        datum_max=round(datum_max, 5),
        surface_rmse=None if surface_rmse is None else round(surface_rmse, 5),
        used_icp=used_icp,
        datum_count=len(cad_pts),
        notes=notes,
    )


def datum_candidates(holes: list, want: int = 3) -> list:
    """조립 홀 목록에서 데이텀으로 쓸 만한 것을 고른다.

    **어느 홀이 진짜 데이텀인지는 도면이 정한다.** 여기서 하는 일은
    "자세를 안정적으로 잡아 주는 조합" 을 추천하는 것뿐이다 — 서로 멀수록
    회전이 덜 흔들린다. 사람이 확인하고 바꿔야 한다.

    Args:
        holes: step_reader 가 뽑은 홀 목록 ({center, diameter, ...}).
    """
    centres = [np.asarray(h["center"], dtype=float) for h in holes
               if h.get("center") is not None]
    if len(centres) < want:
        return list(range(len(centres)))

    points = np.stack(centres)
    # 가장 멀리 떨어진 두 점에서 시작해, 그 선에서 가장 먼 점을 더한다
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(gaps)), gaps.shape)
    picked = [int(first), int(second)]
    while len(picked) < want:
        rest = [i for i in range(len(points)) if i not in picked]
        if not rest:
            break
        best = max(rest, key=lambda i: _spread(points[picked + [i]]))
        picked.append(int(best))
    return picked


__all__ = ["MIN_DATUMS", "MIN_SPREAD", "AlignResult",
           "align", "datum_candidates", "rigid_from_points"]
