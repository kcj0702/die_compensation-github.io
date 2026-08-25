"""3D CAD/스캔 데이터 가져오기.

제로라인 판정의 나머지 절반을 열기 위한 모듈이다. 2D 히트맵만으로는
컬러맵 제로존(현업 자료 3번 방법)밖에 못 하고, RPS 정렬·수축 중심선·
단면 분석(1·2·4번)은 3D 데이터가 있어야 한다.

    mesh_io      STL/PLY/OBJ/glTF -> 웹 뷰어용 삼각망
    step_reader  STEP -> 삼각망 + 홀/기준평면(RPS 후보)

CATIA 네이티브(.CATPart)는 지원하지 않는다 — CGM 독자 포맷이라 명세
없이 파싱이 불가하다. 현업에 STEP(AP214) 또는 STL 내보내기를 요청한다.
자세한 근거는 README.md 참고.
"""

from cad_import.mesh_io import (
    MESH_SUFFIXES, MeshBounds, MeshSummary,
    is_mesh_file, load_mesh, mesh_bounds, to_web_mesh,
)
from cad_import.step_reader import (
    STEP_SUFFIXES, Cylinder, PlaneFace,
    find_cylinders, find_planes, is_step_file, load_step,
    read_step_full, tessellate,
)

__all__ = [
    "MESH_SUFFIXES", "MeshBounds", "MeshSummary",
    "is_mesh_file", "load_mesh", "mesh_bounds", "to_web_mesh",
    "STEP_SUFFIXES", "Cylinder", "PlaneFace",
    "find_cylinders", "find_planes", "is_step_file", "load_step",
    "read_step_full", "tessellate",
]
