"""3D 메시(STL/PLY/OBJ/glTF)를 읽어 웹 뷰어가 쓸 형태로 바꾼다.

[왜 필요한가]
현업 자료(2026-08-25)가 정리한 제로라인 판정 4가지 방법 중 3가지
(RPS 정렬, 수축 중심선, 단면 분석)는 전부 3D 데이터가 있어야 한다.
지금까지 우리가 가진 건 3D 스캔의 *결과 이미지*(2D 히트맵 PNG)뿐이라
그 3가지가 통째로 막혀 있었다.

[CATPart 를 직접 못 읽는 이유]
`999 REINF SIDE OTR.CATPart`(CATIA V5 R34, 53.7MB)를 받아 분석했으나
형상이 CGM 독자 포맷으로 인코딩돼 있다 — zlib 스트림 0개, 엔트로피
7.6/8.0. 명세 없이 파싱 불가하고 이 PC엔 CATIA 도 없다. 그래서
**공개 포맷(STEP/STL)으로 내보낸 파일**을 받는 걸 전제로 한다.

- STL/PLY/OBJ : 삼각망만. 화면 표시·편차 계산은 되지만 홀·평면 정보는
  이미 삼각형으로 뭉개져 사라진다.
- STEP        : B-Rep 유지. 홀 중심·기준평면을 뽑을 수 있어 **RPS 정렬**
  이 가능하다 (step_reader.py 참고).

스캔 데이터는 보통 STL/PLY 로, CAD 는 STEP 으로 온다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import sys

import trimesh

# 웹으로 내보낼 때의 기본 삼각형 상한. 자동차 패널 스캔은 수백만
# 삼각형이 예사라 그대로 JSON 으로 말면 브라우저가 죽는다. 표시용은
# 줄이고, 계산용 원본은 서버에 그대로 둔다.
# 표시용 삼각형 상한.
#
# 15만으로 두었더니 형상이 눈에 띄게 망가졌다 — 얇은 리브와 플랜지가
# 뭉개져 "깨져 보인다" 는 말이 나왔다. 실측 67XX6(원본 363,431 면)에서
# 원본 정점이 간략화 표면에서 얼마나 떨어지는지 재보면 —
#
#     15만 면   중앙값 0.009 · 95% 1.807 · 최대 18.94 mm
#     25만 면   중앙값 0.000 · 95% 0.482 · 최대  7.51 mm
#     32만 면   중앙값 0.000 · 95% 0.000 · 최대  2.57 mm
#
# 보정량이 +-3mm 인 부품에서 19mm 오차는 쓸 수 없다. 40만으로 올리면
# 우리가 다루는 부품(22~37만 면)은 간략화를 아예 안 거치고, 그보다 큰
# 것만 줄어든다. RTX 급 GPU 에서 40만 면은 부담이 안 된다.
DEFAULT_MAX_FACES = 400_000

MESH_SUFFIXES = {".stl", ".ply", ".obj", ".off", ".glb", ".gltf", ".3mf"}
STEP_SUFFIXES = {".step", ".stp"}
# CATIA COM 을 통해 로컬에서 변환하는 확장자. 실제 로드는 STEP 캐시를 통해.
CATIA_SUFFIXES = {".catpart", ".catproduct"}
# 이 모듈이 직접 열 수 있는 모든 확장자.
SUPPORTED_SUFFIXES = MESH_SUFFIXES | STEP_SUFFIXES | CATIA_SUFFIXES


@dataclass
class MeshBounds:
    """부품 크기. 단위는 파일에 안 적혀 있으면 mm 로 본다(CAD 관례)."""

    min: list
    max: list
    size: list
    center: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeshSummary:
    name: str
    source_format: str
    bounds: MeshBounds
    n_vertices: int
    n_faces: int
    n_faces_display: int
    watertight: bool
    units: str = "mm"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["bounds"] = self.bounds.to_dict()
        return out


def is_mesh_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MESH_SUFFIXES


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """메시 파일을 읽어 삼각망 하나로 만든다.

    Scene(여러 바디)으로 들어오면 하나로 합친다 — 부품 하나를 보는 게
    목적이라 바디별 구분은 지금 단계에선 필요 없다.
    """
    path = Path(path)
    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.size == 0:
        raise ValueError(f"삼각망을 읽지 못했습니다: {path.name}")
    return loaded


def load_any(path: str | Path, *, cache_dir: str | Path | None = None) -> trimesh.Trimesh:
    """확장자에 맞는 리더로 삼각망을 돌려준다.

    STEP 은 OCCT 로 열어 tessellate 한 뒤 trimesh 로 감싸고, .CATPart 는
    CATIA COM 으로 STEP 을 뽑아 캐시한 뒤 그 STEP 경로로 재귀 호출한다.
    나머지는 기존 load_mesh 그대로. 필요한 옵셔널 의존성(OCCT, pywin32)
    이 없으면 각 브랜치가 명확한 오류를 던져 상위가 원인을 안내한다.

    Args:
        cache_dir: CATIA→STEP 변환 캐시가 쓰일 폴더. 지정하지 않으면 원본
            옆 `.cache/` 를 쓴다. 원본이 외부(회사 공용) 폴더라 쓰기 권한이
            없거나 저장소 밖으로 캐시를 새어 나가게 하기 싫을 때 지정한다.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in MESH_SUFFIXES:
        return load_mesh(path)
    if suffix in STEP_SUFFIXES:
        # OCCT 테셀레이션은 크기에 따라 15~30초 걸린다. 같은 STEP 을 매번
        # 다시 잘게 나누지 않도록, 결과를 STL 로 캐시해 두고 다음번에는
        # 곧바로 STL 을 읽는다. STEP 이 새로 저장되면 mtime 비교로 재실행.
        cache_root = Path(cache_dir) if cache_dir is not None else path.parent / ".cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        cached_stl = cache_root / f"{path.stem}__from_step.stl"
        if cached_stl.is_file() and cached_stl.stat().st_mtime >= path.stat().st_mtime:
            return load_mesh(cached_stl)
        from .step_reader import load_step, tessellate
        vertices, faces = tessellate(load_step(path))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        try:
            mesh.export(str(cached_stl))
        except Exception:
            # 캐시 저장이 실패해도 이번 분석은 계속 진행 — 다음번에도 그냥
            # STEP 을 다시 잘게 나눌 뿐, 결과는 같다.
            pass
        return mesh
    if suffix in CATIA_SUFFIXES:
        from .catia_convert import convert_to_mesh
        target_cache = Path(cache_dir) if cache_dir is not None else path.parent / ".cache"
        # STL 우선, STEP/IGES fallback — 결과 확장자가 그에 따라 달라지므로
        # 실제 반환 경로를 다시 load_any 로 넘겨 각 브랜치가 처리하게 한다.
        converted = convert_to_mesh(path, target_cache)
        return load_any(converted, cache_dir=cache_dir)
    raise ValueError(f"지원하지 않는 형식입니다: {path.suffix}")


def mesh_bounds(mesh: trimesh.Trimesh) -> MeshBounds:
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    return MeshBounds(
        min=[round(float(v), 3) for v in lo],
        max=[round(float(v), 3) for v in hi],
        size=[round(float(v), 3) for v in (hi - lo)],
        center=[round(float(v), 3) for v in (lo + hi) / 2.0],
    )


def split_symmetric_pair(
    vertices: np.ndarray, faces: np.ndarray,
    gap_ratio_threshold: float = 0.15, min_side_fraction: float = 0.15,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """LH/RH 대칭쌍처럼 한 CATPart 안에 부품 두 개가 같이 들어 있으면 둘로 쪼갠다.

    [왜 필요한가 — 실측 71XX2]
    현업 CATPart 는 부품 하나가 아니라 좌우 대칭쌍을 한 파일에 담아 두는
    경우가 있다. 스캔은 그중 한쪽만 찍은 것이라, 통짜 mesh 로
    `overlay.fit_view` 를 돌리면 "둘을 합친 실루엣" 과 "부품 하나짜리
    스캔" 을 맞추려다 엉뚱한 축이나 방향으로 수렴한다.

    실측 71XX2 CATPart 에서 축별 정점 좌표를 정렬해 가장 큰 빈틈을 재면
    Y 축에서 전체 범위의 80.8% 가 빈 공간이었고, 그 틈을 기준으로 정점이
    44181 / 44194 개로 거의 정확히 반으로 갈렸다 — 흔한 STEP 테셀레이션
    잡음(면 경계 이음매)과 달리, 이건 "몸통 두 개가 멀리 떨어져 있다" 는
    분명한 신호다. `trimesh.split()` 은 이 경우 못 쓴다 — STEP 이 면마다
    따로 테셀레이션돼서 연결성 기준으로 쪼개면 수만 개의 가짜 조각이
    나온다(실측: 76,000+). 그래서 정점 좌표의 축별 최대 간격만 본다.

    Returns:
        틈이 안 보이면(=부품 하나) 원본 그대로 담긴 리스트 길이 1.
        틈이 보이면(=둘) 각 절반의 (vertices, faces) 리스트 길이 2 —
        정점 인덱스는 각 절반 안에서 0 부터 다시 매긴 것이라 원본 faces
        인덱스와 안 맞는다.
    """
    v = np.asarray(vertices, dtype=np.float64)
    best_axis = -1
    best_gap_ratio = 0.0
    best_split_value = 0.0
    for axis in range(3):
        vals = np.sort(v[:, axis])
        total_extent = vals[-1] - vals[0]
        if total_extent <= 0:
            continue
        gaps = np.diff(vals)
        idx = int(np.argmax(gaps))
        gap_ratio = float(gaps[idx]) / total_extent
        left_fraction = (idx + 1) / len(vals)
        right_fraction = 1.0 - left_fraction
        if (gap_ratio > best_gap_ratio and gap_ratio >= gap_ratio_threshold
                and left_fraction >= min_side_fraction and right_fraction >= min_side_fraction):
            best_axis = axis
            best_gap_ratio = gap_ratio
            best_split_value = float((vals[idx] + vals[idx + 1]) / 2.0)

    if best_axis < 0:
        return [(vertices, faces)]

    faces_arr = np.asarray(faces, dtype=np.int64)
    vertex_side = v[:, best_axis] > best_split_value  # True = 오른쪽(위) 절반

    parts: list[tuple[np.ndarray, np.ndarray]] = []
    for side in (False, True):
        keep_vertex = vertex_side == side
        # 삼각형 세 꼭짓점이 전부 같은 절반에 있어야 그 절반의 면으로 인정한다.
        # 갭이 이만큼 크면(15%+) 갭을 가로지르는 면은 사실상 없다.
        keep_face = keep_vertex[faces_arr].all(axis=1)
        if not keep_face.any():
            continue
        old_indices = np.nonzero(keep_vertex)[0]
        remap = np.full(len(v), -1, dtype=np.int64)
        remap[old_indices] = np.arange(len(old_indices))
        part_vertices = v[old_indices]
        part_faces = remap[faces_arr[keep_face]]
        parts.append((part_vertices, part_faces))

    return parts if len(parts) == 2 else [(vertices, faces)]


def simplify_for_display(
    mesh: trimesh.Trimesh, max_faces: int = DEFAULT_MAX_FACES
) -> trimesh.Trimesh:
    """표시용으로만 삼각형 수를 줄인다. 실패하면 원본을 그대로 쓴다.

    간략화는 없으면 없는 대로 동작해야 한다 — 뷰어가 조금 무거워질 뿐
    결과가 틀리지는 않는다. 그래서 실패해도 원본을 돌려준다.

    [조용히 실패하고 있었다]
    trimesh 5.0 은 간략화를 fast_simplification 패키지에 맡기는데 그게
    안 깔려 있었다. 예외를 그냥 삼키고 있어서 상한이 안 걸리는 걸
    아무도 몰랐다 — 실제로 22~37만 면짜리 메시가 그대로 브라우저로
    가고 있었다. 이제 왜 실패했는지 로그로 남긴다.
    """
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception as exc:
        print(f"[cad_import] 표시용 간략화를 건너뜁니다 "
              f"({len(mesh.faces):,}면 그대로): {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return mesh


def to_web_mesh(
    mesh: trimesh.Trimesh,
    name: str = "part",
    source_format: str = "",
    max_faces: int = DEFAULT_MAX_FACES,
    recenter: bool = True,
) -> dict:
    """three.js 가 바로 먹을 수 있는 형태로 만든다.

    recenter=True 면 원점 기준으로 옮긴다. 자동차 부품은 차량 좌표계
    기준이라 원점에서 수천 mm 떨어져 있는 경우가 많은데, 그대로 두면
    카메라가 부품을 못 잡는다. 원래 위치는 bounds 에 남겨둔다.
    """
    display = simplify_for_display(mesh, max_faces)
    bounds = mesh_bounds(mesh)

    vertices = np.asarray(display.vertices, dtype=np.float64)
    if recenter:
        vertices = vertices - np.asarray(bounds.center, dtype=np.float64)

    return {
        "summary": MeshSummary(
            name=name,
            source_format=source_format or "mesh",
            bounds=bounds,
            n_vertices=int(len(mesh.vertices)),
            n_faces=int(len(mesh.faces)),
            n_faces_display=int(len(display.faces)),
            watertight=bool(mesh.is_watertight),
        ).to_dict(),
        "positions": [round(float(v), 4) for v in vertices.ravel()],
        "indices": [int(i) for i in np.asarray(display.faces, dtype=np.int64).ravel()],
        "recentered": bool(recenter),
    }


__all__ = [
    "DEFAULT_MAX_FACES", "MESH_SUFFIXES", "STEP_SUFFIXES", "SUPPORTED_SUFFIXES",
    "MeshBounds", "MeshSummary",
    "is_mesh_file", "load_mesh", "load_any", "mesh_bounds",
    "simplify_for_display", "split_symmetric_pair", "to_web_mesh",
]
