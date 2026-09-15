"""축에서 벗어난 시점(등각 등)의 실루엣에 CAD 메시를 완전 3D 자세로 맞춘다.

`overlay.fit_view` 는 X/Y/Z 축 정면 정사영 6방향만 후보로 둔다. 실측 스캔은 대개 축에서
2~6° 안쪽이라 그걸로 충분하지만, 작업자가 등각으로 캡처한 이미지는 어느 축에도 걸리지 않는다.
이 모듈은 학습 모델 없이 실루엣 겹침만으로 회전 3자유도 + 배율 + 이동을 찾는다.

    1. 구면 위 시선 방향(피보나치 격자) × 화면 회전(주축 정렬, 180°) 후보를 128px 로 채점
    2. 상위 16개를 128px → 4개를 256px → 1개를 원본 해상도에서 좌표 하강으로 다듬기
    3. 결과 `Pose` 를 `pose_to_view()` 로 기존 `ViewFit` + **회전된 정점**으로 바꿔 넘긴다.

[엔진과 잇는 방식]
`ViewFit` 은 축 정면만 표현할 수 있다. 그래서 자세의 회전을 정점에 미리 적용해(`vertices @ M.T`)
Z축 정면 `ViewFit` 으로 만든다. 그 뒤 `to_pixels`/`unproject`/`polish_by_hit_rate` 는 그대로 쓰고,
`unproject` 가 돌려준 3D 점은 `points @ M` 으로 원래 부품 좌표로 되돌린다. M 은 직교 행렬이다.

[한계]
실루엣만으로는 반대편 시점 + 좌우 반전이 구분되지 않아 반전은 기본 제외한다.
실측(합성 정답): 회전 오차 약 3°, 배율 3%. 파트당 90초 안팎 — 결과는 디스크에 캐시한다.
원근은 없다고 가정한다(CATIA 평행 투영).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .overlay import ViewFit

FALLBACK_OUTLINE_IOU = 0.85      # 축 정면 정합의 바깥 윤곽 겹침이 이보다 낮으면 3D 탐색으로 넘어간다
SEARCH_DIRECTIONS = 800
SEARCH_MAX_FACES = 30_000


# ---------------------------------------------------------------- 실루엣 도구

def outer_fill(binary: np.ndarray) -> np.ndarray:
    """가장 큰 덩어리의 바깥 윤곽을 채운다(구멍 제거)."""
    contours, _ = cv2.findContours((np.asarray(binary) > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(np.asarray(binary).shape, np.uint8)
    if contours:
        cv2.fillPoly(filled, [max(contours, key=cv2.contourArea)], 255)
    return filled


def render_silhouette_mask(image: np.ndarray) -> np.ndarray:
    """CAD 렌더(CATIA 캡처)에서 부품 실루엣을 잡는다.

    실측 64XX2-DC000: 배경은 채도 75~116·명도 108~135 의 회보라 그라데이션이고, 부품은 채도 255 의
    파랑 아니면 채도 0 의 검은 윤곽선이다. 배경색 거리만 쓰면 아래쪽 진한 파랑이 배경으로 빠진다.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    margin = max(4, width // 40)
    border = np.concatenate([lab[:, :margin], lab[:, width - margin:]], axis=1)
    row_background = np.median(border, axis=1)
    distance = np.linalg.norm(lab - row_background[:, None, :], axis=2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated_or_dark = (hsv[:, :, 1] > 150) | (hsv[:, :, 2] < 70)
    mask = ((distance > 18) | saturated_or_dark).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return outer_fill(mask)


def rasterize(points_px: np.ndarray, faces: np.ndarray, shape: tuple) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    triangles = np.rint(points_px[faces]).astype(np.int32)
    cv2.fillPoly(canvas, list(triangles), 255)
    return outer_fill(cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def mask_box(mask: np.ndarray) -> tuple:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def downscale(mask: np.ndarray, size: int) -> tuple:
    height, width = mask.shape
    factor = size / max(height, width)
    small = cv2.resize(mask, (max(1, round(width * factor)), max(1, round(height * factor))),
                       interpolation=cv2.INTER_AREA)
    return outer_fill(small > 127), factor


# ---------------------------------------------------------------- 회전 도구

def rotation_from_view(direction: np.ndarray, roll: float) -> np.ndarray:
    """시선 방향(부품 좌표계 단위벡터)에서 부품을 바라보는 회전. 반환 R 은 부품→카메라 좌표.

    카메라 z 축이 시선 방향(화면 안쪽), x/y 가 화면 축(오른쪽/아래). roll 은 화면 안 회전(라디안).
    """
    z = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    base = np.stack([x, y, z])
    cos, sin = np.cos(roll), np.sin(roll)
    spin = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return spin @ base


def small_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """카메라 좌표계에서의 작은 회전(라디안). rz 가 화면 안 회전."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def fibonacci_directions(count: int) -> np.ndarray:
    index = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * index / count)
    theta = np.pi * (1 + 5 ** 0.5) * index
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def principal_angle(xs: np.ndarray, ys: np.ndarray) -> float:
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy
    return 0.5 * np.arctan2(2 * (dx * dy).mean(), (dx * dx).mean() - (dy * dy).mean())


def euler_degrees(rotation: np.ndarray) -> list:
    """사람이 읽는 오일러각(도). 부품 X/Y/Z 축을 차례로 돌린 값."""
    sy = -rotation[2, 0]
    cy = np.sqrt(max(0.0, 1 - sy * sy))
    if cy > 1e-6:
        rx = np.arctan2(rotation[2, 1], rotation[2, 2])
        ry = np.arcsin(np.clip(sy, -1, 1))
        rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = np.arctan2(-rotation[1, 2], rotation[1, 1])
        ry = np.arcsin(np.clip(sy, -1, 1))
        rz = 0.0
    return [round(float(np.degrees(value)), 2) for value in (rx, ry, rz)]


# ---------------------------------------------------------------- 자세

class Pose:
    """부품 좌표 → 화면 픽셀: p_px = (R p)[:2] * mirror / mm_per_px + offset. 정점은 중심을 뺀 좌표다."""

    def __init__(self, rotation: np.ndarray, mirror: bool, mm_per_px: float, offset: np.ndarray):
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.mirror = bool(mirror)
        self.mm_per_px = float(mm_per_px)
        self.offset = np.asarray(offset, dtype=np.float64)

    def project(self, vertices: np.ndarray) -> np.ndarray:
        cam = vertices @ self.rotation.T
        xy = cam[:, :2].copy()
        if self.mirror:
            xy[:, 0] = -xy[:, 0]
        return xy / self.mm_per_px + self.offset

    def to_dict(self) -> dict:
        return {"rotation": self.rotation.round(6).tolist(), "mirror": self.mirror,
                "mm_per_px": round(self.mm_per_px, 6), "offset_px": self.offset.round(3).tolist(),
                "euler_deg": euler_degrees(self.rotation)}

    @classmethod
    def from_dict(cls, payload: dict) -> "Pose":
        return cls(np.asarray(payload["rotation"]), payload["mirror"], payload["mm_per_px"], np.asarray(payload["offset_px"]))


def fit_scale_offset(xy_mm: np.ndarray, box: tuple, scale: float | None = None) -> tuple:
    """투영(mm)을 마스크 경계 상자에 맞추는 배율(mm/px)과 원점. 배율을 주면 원점만 맞춘다."""
    x0, x1, y0, y1 = box
    span_x, span_y = np.ptp(xy_mm[:, 0]), np.ptp(xy_mm[:, 1])
    if scale is None:
        scale = max(span_x / max(1, x1 - x0), span_y / max(1, y1 - y0))
    centre_px = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    centre_mm = np.array([(xy_mm[:, 0].min() + xy_mm[:, 0].max()) / 2,
                          (xy_mm[:, 1].min() + xy_mm[:, 1].max()) / 2])
    return scale, centre_px - centre_mm / scale


# ---------------------------------------------------------------- 탐색

def _prepare_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int = SEARCH_MAX_FACES) -> tuple:
    """면을 솎고 쓰이는 정점만 남긴다. 중심을 빼서 회전이 부품 가운데를 축으로 돌게 한다."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) > max_faces:
        faces = faces[::int(np.ceil(len(faces) / max_faces))]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    centroid = vertices[used].mean(axis=0)
    return vertices[used] - centroid, remap[faces], centroid


def coarse_search(vertices, faces, mask, directions, allow_mirror, size=128, keep=16, log=print):
    small, factor = downscale(mask, size)
    box = mask_box(small)
    ys, xs = np.nonzero(small)
    mask_angle = principal_angle(xs.astype(float), ys.astype(float))
    scored = []
    started = time.time()
    for direction in fibonacci_directions(directions):
        base = rotation_from_view(direction, 0.0)
        cam = vertices @ base.T
        shape_angle = principal_angle(cam[:, 0], cam[:, 1])
        for mirror in ((False, True) if allow_mirror else (False,)):
            for extra in (0.0, np.pi):
                roll = (mask_angle - (-shape_angle if mirror else shape_angle)) + extra
                rotation = rotation_from_view(direction, roll if not mirror else -roll)
                xy = Pose(rotation, mirror, 1.0, np.zeros(2)).project(vertices)
                scale, offset = fit_scale_offset(xy, box)
                pose = Pose(rotation, mirror, scale, offset)
                scored.append((iou(rasterize(pose.project(vertices), faces, small.shape), small), pose))
    scored.sort(key=lambda item: -item[0])
    log(f"pose coarse: {len(scored)} candidates {time.time() - started:.1f}s best={scored[0][0]:.4f}")
    return [(score, Pose(pose.rotation, pose.mirror, pose.mm_per_px * factor, pose.offset / factor))
            for score, pose in scored[:keep]]


REFINE_STEPS = (
    ("rx", (-12, -6, -3, -1.5, -0.5, 0.5, 1.5, 3, 6, 12)),
    ("ry", (-12, -6, -3, -1.5, -0.5, 0.5, 1.5, 3, 6, 12)),
    ("rz", (-20, -10, -4, -2, -1, -0.25, 0.25, 1, 2, 4, 10, 20)),
    ("scale", (0.94, 0.97, 0.99, 0.995, 1.005, 1.01, 1.03, 1.06)),
    ("dx", (-0.03, -0.01, -0.003, 0.003, 0.01, 0.03)),
    ("dy", (-0.03, -0.01, -0.003, 0.003, 0.01, 0.03)),
)


def refine(pose: Pose, vertices, faces, mask, size: int, rounds: int = 4) -> tuple:
    small, factor = downscale(mask, size)
    height, width = small.shape
    centre = np.array([width / 2, height / 2])
    current = Pose(pose.rotation, pose.mirror, pose.mm_per_px / factor, pose.offset * factor)
    best_score = iou(rasterize(current.project(vertices), faces, small.shape), small)
    for _ in range(rounds):
        moved = False
        for key, options in REFINE_STEPS:
            for option in options:
                if key in ("rx", "ry", "rz"):
                    angles = {axis: (np.radians(option) if key == axis else 0.0) for axis in ("rx", "ry", "rz")}
                    if current.mirror and key == "rz":
                        angles["rz"] = -angles["rz"]
                    rotation = small_rotation(angles["rx"], angles["ry"], angles["rz"]) @ current.rotation
                    probe = Pose(rotation, current.mirror, current.mm_per_px, current.offset)
                    # 화면 중심을 축으로 돌도록 원점을 보정한다.
                    shift = current.project(vertices).mean(axis=0) - probe.project(vertices).mean(axis=0)
                    candidate = Pose(rotation, current.mirror, current.mm_per_px, current.offset + shift)
                elif key == "scale":
                    candidate = Pose(current.rotation, current.mirror, current.mm_per_px / option,
                                     centre + (current.offset - centre) * option)
                else:
                    shift = np.array([option * width, 0.0]) if key == "dx" else np.array([0.0, option * height])
                    candidate = Pose(current.rotation, current.mirror, current.mm_per_px, current.offset + shift)
                score = iou(rasterize(candidate.project(vertices), faces, small.shape), small)
                if score > best_score + 1e-4:
                    current, best_score, moved = candidate, score, True
        if not moved:
            break
    return best_score, Pose(current.rotation, current.mirror, current.mm_per_px * factor, current.offset / factor)


def search_pose(vertices, faces, mask, *, directions: int = SEARCH_DIRECTIONS, allow_mirror: bool = False,
                log: Callable[[str], None] = print) -> tuple:
    """(겹침, Pose). Pose 는 **원본 부품 좌표**(중심을 빼지 않은) 기준으로 돌려준다.

    거친 격자(≈7° 간격)는 정답 근처 후보도 0.83 정도로 낮게 채점해 다른 골짜기와 구분이 안 된다.
    그래서 상위 후보를 넉넉히(16개) 128px 에서 먼저 다듬어 골짜기를 가른 뒤 좁혀 간다.
    """
    mask = outer_fill(mask)
    centred, faces_used, centroid = _prepare_mesh(vertices, faces)
    candidates = coarse_search(centred, faces_used, mask, directions, allow_mirror, log=log)
    started = time.time()
    low = sorted((refine(pose, centred, faces_used, mask, 128, rounds=3) for _s, pose in candidates), key=lambda i: -i[0])
    log(f"pose refine@128: top={[round(i[0], 3) for i in low[:4]]} {time.time() - started:.1f}s")
    mid = sorted((refine(pose, centred, faces_used, mask, 256, rounds=3) for _s, pose in low[:4]), key=lambda i: -i[0])
    score, pose = refine(mid[0][1], centred, faces_used, mask, max(mask.shape), rounds=4)
    log(f"pose refine@full: iou={score:.4f} euler={euler_degrees(pose.rotation)} mirror={pose.mirror}")
    # 중심을 뺀 좌표의 원점을 원본 좌표 기준으로 옮긴다: (R(v-c))/mm + off = (Rv)/mm + (off - (Rc)/mm)
    centroid_px = Pose(pose.rotation, pose.mirror, pose.mm_per_px, np.zeros(2)).project(centroid[None, :])[0]
    return score, Pose(pose.rotation, pose.mirror, pose.mm_per_px, pose.offset - centroid_px)


# ---------------------------------------------------------------- 엔진 연결

def pose_matrix(pose: Pose) -> np.ndarray:
    """부품 좌표 → Z축 정면 ViewFit 이 기대하는 좌표. 열벡터 기준 M: v' = M v.

    카메라 z 는 화면 안쪽(멀어지는 방향)인데 `unproject` 는 큰 축값을 가까운 면으로 보므로 z 를 뒤집는다.
    좌우 반전이면 x 도 뒤집는다. M 은 직교 행렬이라 되돌릴 때는 v = Mᵀ v' 즉 `points @ M`.
    """
    flip = np.diag([-1.0 if pose.mirror else 1.0, 1.0, -1.0])
    return flip @ pose.rotation


def pose_to_view(pose: Pose, vertices: np.ndarray, score: float) -> tuple:
    """(ViewFit, 회전된 정점, M). 회전된 정점을 to_pixels/unproject/polish 에 그대로 쓴다."""
    matrix = pose_matrix(pose)
    rotated = np.asarray(vertices, dtype=np.float64) @ matrix.T
    fit = ViewFit(axis=2, sign=1, flip_u=False, flip_v=False, mm_per_px=pose.mm_per_px,
                  origin_u=float(-pose.offset[0] * pose.mm_per_px), origin_v=float(-pose.offset[1] * pose.mm_per_px),
                  iou=round(float(score), 4), detail_iou=0.0, hit_rate=0.0, swap=False, angle=0.0, reliable=False)
    return fit, rotated, matrix


def _cache_key(vertices: np.ndarray, mask: np.ndarray) -> str:
    small, _ = downscale(outer_fill(mask), 256)
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(small.tobytes())
    digest.update(np.asarray(mask.shape).tobytes())
    digest.update(np.asarray([len(vertices)]).tobytes())
    digest.update(np.asarray(vertices, dtype=np.float64).min(axis=0).round(2).tobytes())
    digest.update(np.asarray(vertices, dtype=np.float64).max(axis=0).round(2).tobytes())
    return digest.hexdigest()[:16]


def cached_search_pose(vertices, faces, mask, cache_dir: str | Path | None, *, log: Callable[[str], None] = print,
                       **kwargs) -> tuple:
    """같은 실루엣·같은 메시면 90초짜리 탐색을 다시 하지 않는다."""
    path = None
    if cache_dir:
        path = Path(cache_dir) / f"pose_{_cache_key(vertices, mask)}.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                log(f"pose cache-hit {path.name} iou={payload['iou']}")
                return float(payload["iou"]), Pose.from_dict(payload["pose"])
            except (OSError, ValueError, KeyError):
                pass
    score, pose = search_pose(vertices, faces, mask, log=log, **kwargs)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"iou": round(score, 4), "pose": pose.to_dict()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return score, pose


__all__ = ["FALLBACK_OUTLINE_IOU", "Pose", "cached_search_pose", "euler_degrees", "outer_fill", "pose_matrix",
           "pose_to_view", "rasterize", "render_silhouette_mask", "search_pose"]
