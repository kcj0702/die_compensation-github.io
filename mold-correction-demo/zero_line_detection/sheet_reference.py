"""보정시트에 그려진 제로라인을 그대로 읽어와 스캔 좌표로 옮긴다.

[왜 이게 필요한가]
스캔만 보고 제로라인을 추론하는 방법을 여러 개 시도했지만 전부 실패했다
(비용 최소경로, 부호 분리도, 등고선, 라벨 근접도 등 8가지 — 정답이 항상
3~4위에 머물렀다). 이유는 두 가지다:

1. 제로라인은 "편차가 0인 등고선"이 아니라 **이번 공정에서 어디까지
   손댈지의 가공 범위 경계**다. 실측으로 확인: 정답선 위 |편차| 평균이
   0.273 이고(0 이 아니다), 최단경로가 아니라 개구부 위로 일부러 우회한다.
   JD_71XX2 시트는 아예 영역을 칠하고 "①: 하형 용접", "②: 상형 심고음"
   이라고 공정을 적어놨다.

2. 가공 범위는 스캔에 찍히는 물리량이 아니라 공정 판단이다. 실제로
   JD_64XX2 시트의 한쪽 "0" 표기는 대응하는 측정값이 스캔에 없다.

그래서 추론하지 않는다. **정답지에 이미 그려져 있으면 그걸 읽는다.**
한 번 등록해두면 같은 품번이 다시 들어왔을 때 시트와 100% 동일한 선이
나온다. 회의록의 "수정 결과 저장 -> 데이터 축적" 방향과 같다.

[주의 — 스캔과 시트가 짝이 맞는지 확인할 것]
2026-08-24 확인: JD_64XX2 스캔 라벨을 부호 반전하면 보정치가 양수(용접)
34개 / 음수(가공) 11개인데, 실제 시트는 14개가 전부 음수(가공)였다.
부호가 반대로 나오므로 이 스캔과 이 시트는 같은 회차가 아닐 수 있다.
check_pairing() 으로 먼저 짝을 확인하고 쓰는 것을 권한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SheetZeroLine:
    """보정시트에서 읽어 스캔 좌표로 옮긴 제로라인."""

    part_no: str
    points: list              # [[x, y], ...] 스캔 이미지 좌표
    source_sheet: str
    sheet_bbox: list          # 시트에서 부품이 차지한 영역
    scan_bbox: list           # 스캔에서 부품이 차지한 영역
    n_raw_pixels: int
    mirrored: bool = False    # 시트 그림이 스캔과 좌우반전이었는지

    def to_dict(self) -> dict:
        return asdict(self)


def _imread(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def _sheet_part_bbox(sheet_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """시트에서 파랗게 렌더된 부품 본체가 차지한 영역을 찾는다.

    표제부(상단 표)와 하단 설명 글자는 부품이 아니므로 세로로 잘라낸다.
    """
    b, g, r = (sheet_bgr[..., i].astype(int) for i in range(3))
    height = sheet_bgr.shape[0]
    body = (b > r + 15) & (b > 80) & ~((r > 240) & (g > 240) & (b > 240))
    body[: int(height * 0.25), :] = False
    body[int(height * 0.92):, :] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        body.astype(np.uint8), connectivity=8
    )
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 1000]
    if not keep:
        raise ValueError("시트에서 부품 형상을 찾지 못했습니다.")
    mask = np.isin(labels, keep)
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _red_curve_mask(sheet_bgr: np.ndarray) -> np.ndarray:
    """시트의 빨간 선 중 가장 큰 덩어리(=제로라인 본선)만 남긴다.

    빨간색은 제로라인 말고도 라벨 박스·글자에 쓰인다. 그것들은 작고
    떨어져 있으므로 가장 큰 연결 성분 하나만 취하면 선만 남는다.
    """
    b, g, r = (sheet_bgr[..., i].astype(int) for i in range(3))
    red = ((r > 180) & (g < 80) & (b < 80)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    if count <= 1:
        raise ValueError("시트에서 빨간 제로라인을 찾지 못했습니다.")
    largest = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return (labels == largest)


def _bfs_farthest(mask: np.ndarray, start: tuple[int, int]):
    """마스크 위에서 start 로부터 BFS. (거리, 부모, 가장 먼 점) 반환."""
    height, width = mask.shape
    flat_start = start[0] * width + start[1]
    distance = np.full(height * width, -1, dtype=np.int32)
    parent = np.full(height * width, -1, dtype=np.int32)
    distance[flat_start] = 0
    queue = [flat_start]
    farthest = flat_start
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        nxt = []
        for index in queue:
            y, x = divmod(index, width)
            if distance[index] > distance[farthest]:
                farthest = index
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width) or not mask[ny, nx]:
                    continue
                neighbour = ny * width + nx
                if distance[neighbour] >= 0:
                    continue
                distance[neighbour] = distance[index] + 1
                parent[neighbour] = index
                nxt.append(neighbour)
        queue = nxt
    return distance, parent, farthest


def _order_curve(mask: np.ndarray) -> np.ndarray:
    """연결된 곡선 마스크를 한 줄로 편다 (양 끝을 잇는 최장 경로).

    빨간 덩어리에는 "0 LINE" 라벨로 가는 지시선 가지가 붙어 있다.
    최장 경로를 취하면 한쪽 라벨 -> 본선 -> 반대쪽 라벨로 이어져
    시트에 그려진 전체 구간이 그대로 나온다.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return np.stack([xs, ys], axis=1)
    width = mask.shape[1]
    _, _, far_a = _bfs_farthest(mask, (int(ys[0]), int(xs[0])))
    ay, ax = divmod(int(far_a), width)
    _, parent, far_b = _bfs_farthest(mask, (ay, ax))

    path = []
    index = int(far_b)
    while index >= 0:
        y, x = divmod(index, width)
        path.append((x, y))
        index = int(parent[index])
    return np.array(path[::-1])


def extract_sheet_zero_line(
    sheet_path: str | Path,
    scan_part_mask: np.ndarray,
    part_no: str,
    values: np.ndarray | None = None,
    simplify_eps: float = 3.0,
) -> SheetZeroLine:
    """보정시트의 제로라인을 읽어 스캔 좌표계로 옮긴다.

    [좌우반전 자동 판별]
    2026-08-24 현업 확인: 3D스캔 그림과 보정시트 그림이 좌우반전인 경우가
    있다. 실측(JD_64XX2)에서 반전을 적용하니 선이 부품 안에 들어가는 비율이
    65%->83%, 선 위 |편차| 평균이 0.247->0.172 로 뚜렷하게 좋아졌다.
    그래서 values 가 주어지면 정방향/반전 둘 다 계산해 선 위 |편차| 가
    낮은 쪽을 채택한다. values 가 없으면 부품 마스크 안에 더 많이 들어가는
    쪽을 고른다.

    시트(CAD 렌더)와 스캔은 투영이 달라 완전 정합은 안 된다. 부품 사각영역
    (bbox)을 맞추는 선형 변환이라 부품 크기 대비 5% 안팎의 오차가 남는다.
    """
    sheet = _imread(sheet_path)
    sx0, sy0, sx1, sy1 = _sheet_part_bbox(sheet)
    mask = _red_curve_mask(sheet)
    ordered = _order_curve(mask)

    ys, xs = np.nonzero(scan_part_mask)
    kx0, ky0, kx1, ky1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    height, width = scan_part_mask.shape
    smoothed = None
    if values is not None:
        smoothed = cv2.GaussianBlur(values, (0, 0), 15)

    def project(mirror: bool) -> np.ndarray:
        nx = (ordered[:, 0] - sx0) / max(sx1 - sx0, 1)
        if mirror:
            nx = 1.0 - nx
        ny = (ordered[:, 1] - sy0) / max(sy1 - sy0, 1)
        px = np.clip((nx * (kx1 - kx0) + kx0).astype(int), 0, width - 1)
        py = np.clip((ny * (ky1 - ky0) + ky0).astype(int), 0, height - 1)
        return np.stack([px, py], axis=1)

    def quality(points: np.ndarray) -> float:
        inside = scan_part_mask[points[:, 1], points[:, 0]]
        if not inside.any():
            return float("inf")
        if smoothed is None:
            return -float(inside.mean())
        return float(np.abs(smoothed[points[:, 1], points[:, 0]][inside]).mean())

    straight = project(False)
    flipped = project(True)
    mirrored = quality(flipped) < quality(straight)
    mapped = flipped if mirrored else straight

    simplified = cv2.approxPolyDP(
        mapped.astype(np.int32).reshape(-1, 1, 2), simplify_eps, False
    ).reshape(-1, 2)

    return SheetZeroLine(
        part_no=part_no,
        points=simplified.tolist(),
        source_sheet=str(sheet_path),
        sheet_bbox=[sx0, sy0, sx1, sy1],
        scan_bbox=[kx0, ky0, kx1, ky1],
        n_raw_pixels=int(mask.sum()),
        mirrored=bool(mirrored),
    )


def check_pairing(sheet_values: list, scan_values: list) -> dict:
    """스캔과 시트가 같은 회차인지 부호 분포로 가늠한다.

    보정치는 스캔 편차의 부호를 뒤집은 값이므로, 짝이 맞으면 두 분포의
    부호가 반대여야 한다. 같은 방향이면 다른 회차이거나 기준면이 다르다.
    """
    s = np.array([v for v in sheet_values if v is not None], dtype=float)
    k = np.array([v for v in scan_values if v is not None], dtype=float)
    if not len(s) or not len(k):
        return {"ok": False, "reason": "값이 없습니다."}
    flipped = -k
    sheet_pos = float((s > 0).mean())
    flipped_pos = float((flipped > 0).mean())
    consistent = abs(sheet_pos - flipped_pos) < 0.35
    return {
        "ok": bool(consistent),
        "sheetPositiveRatio": round(sheet_pos, 3),
        "scanFlippedPositiveRatio": round(flipped_pos, 3),
        "reason": (
            "부호 분포가 비슷합니다 (짝이 맞을 가능성)." if consistent
            else "부호 분포가 반대입니다 — 다른 회차이거나 기준면이 다를 수 있습니다."
        ),
    }


def load_library(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_to_library(path: str | Path, entry: SheetZeroLine) -> dict:
    p = Path(path)
    library = load_library(p)
    library[entry.part_no] = entry.to_dict()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    return library


__all__ = [
    "SheetZeroLine", "extract_sheet_zero_line", "check_pairing",
    "load_library", "save_to_library",
]
