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
import trimesh

# 웹으로 내보낼 때의 기본 삼각형 상한. 자동차 패널 스캔은 수백만
# 삼각형이 예사라 그대로 JSON 으로 말면 브라우저가 죽는다. 표시용은
# 줄이고, 계산용 원본은 서버에 그대로 둔다.
DEFAULT_MAX_FACES = 150_000

MESH_SUFFIXES = {".stl", ".ply", ".obj", ".off", ".glb", ".gltf", ".3mf"}


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


def mesh_bounds(mesh: trimesh.Trimesh) -> MeshBounds:
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    return MeshBounds(
        min=[round(float(v), 3) for v in lo],
        max=[round(float(v), 3) for v in hi],
        size=[round(float(v), 3) for v in (hi - lo)],
        center=[round(float(v), 3) for v in (lo + hi) / 2.0],
    )


def simplify_for_display(
    mesh: trimesh.Trimesh, max_faces: int = DEFAULT_MAX_FACES
) -> trimesh.Trimesh:
    """표시용으로만 삼각형 수를 줄인다. 실패하면 원본을 그대로 쓴다.

    간략화는 없으면 없는 대로 동작해야 한다 — 뷰어가 조금 무거워질 뿐
    결과가 틀리지는 않는다. 그래서 예외를 삼키고 원본을 돌려준다.
    """
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception:
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
    "DEFAULT_MAX_FACES", "MESH_SUFFIXES",
    "MeshBounds", "MeshSummary",
    "is_mesh_file", "load_mesh", "mesh_bounds",
    "simplify_for_display", "to_web_mesh",
]
