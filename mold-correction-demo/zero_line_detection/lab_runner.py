"""받은 제로라인 파이프라인(lab_pipeline)을 우리 쪽에서 돌린다.

[왜 따로 두나]
lab_pipeline 안의 스크립트는 받은 그대로다 — 손대지 않는다. 그런데 그
스크립트들은 `<품번> 3D 스캔/output/<단계>/` 구조를 전제하고, 자기가 놓인
폴더 위치(`script_dir.parent.parent`)로 품번을 알아낸다. 그래서 그 구조를
임시 폴더에 만들어 주고 순서대로 부르는 껍데기가 필요하다.

[왜 subprocess 인가]
import 해서 함수만 부르면 깔끔하겠지만, 각 스크립트가 `__file__` 위치로
경로를 정하고 argparse 기본값을 그 자리에서 만든다. 그대로 두려면 파일을
그 자리에 놓고 그 안에서 실행하는 게 맞다. 코드를 고치지 않겠다는 원칙이
먼저다.

[캐시]
5단계를 다 도는 데 시간이 걸린다. 같은 그림이면 다시 돌리지 않도록 결과를
그림 내용 해시로 들고 있는다 — 라벨 판독 캐시와 같은 이유다.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "lab_pipeline"

# out_of_tolerance.py 의 SCAN_SCALES 에 등록된 품번들. 여기 없으면 못 돈다.
KNOWN_PREFIXES = {
    "64XX2": "JD_64XX2-DR000",
    "67XX6": "JD_67XX6-DR000",
    "71XX2": "JD_71XX2-DR000",
}

STAGES = [
    # 라벨 제거도 **그쪽 것**을 쓴다. 우리 label_removal 은 컬러바 범례의
    # 눈금 숫자까지 라벨로 오인해 범례를 통째로 지워, 2단계가
    # "color bar could not be detected" 로 멈춘다(실측 64XX2·67XX6).
    ("01_label_removal", "remove_labels.py"),
    ("02_contour_graph", "contour_graph.py"),
    ("03_zero_point_selection", "zero_point_selection.py"),
    ("04_out_of_tolerance", "out_of_tolerance.py"),
    ("05_merged_correction_regions", "merge_correction_regions.py"),
    ("06_nearest_zero_points", "select_nearest_zero_points.py"),
]

_cache: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 8


def prefix_for(part_no: str | None) -> str | None:
    """품번 -> 파이프라인이 아는 이름. 모르면 None."""
    folded = str(part_no or "").upper().replace("-", "").replace("_", "")
    for key, prefix in KNOWN_PREFIXES.items():
        if key in folded:
            return prefix
    return None


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 로 만들지 못했습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def run(scan_bgr: np.ndarray, part_no: str | None,
        keep_dir: Path | None = None) -> dict:
    """스캔 한 장으로 제로라인을 만든다.

    Args:
        scan_bgr: 원본 스캔 (BGR). 라벨 제거부터 이 파이프라인이 한다.
        part_no: 품번. SCAN_SCALES 에 등록된 것이어야 한다.
        keep_dir: 주면 중간 산출물을 그 폴더에 남긴다(디버깅용).

    Returns:
        {"prefix", "regions": [...], "lines": [[[x,y],...]], "raw": {...}}
        돌릴 수 없으면 {"error": "..."}.
    """
    prefix = prefix_for(part_no)
    if prefix is None:
        return {"error": f"이 파이프라인에 등록되지 않은 품번입니다: {part_no}"}

    key = hashlib.blake2b(
        np.ascontiguousarray(scan_bgr).tobytes() + prefix.encode(),
        digest_size=16).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / f"{prefix} 3D 스캔"
        _write(root / "input" / f"{prefix} 3D 스캔.png", scan_bgr)

        for folder, script in STAGES:
            stage_dir = root / "output" / folder
            stage_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PIPELINE / script, stage_dir / script)
            # 1단계만 자기 폴더 아래 input/output 을 본다. 나머지는 품번
            # 폴더 기준이라 인자 없이 돈다.
            argv = [sys.executable, script]
            if folder == "01_label_removal":
                argv += ["--input-dir", str(root / "input"),
                         "--output-dir", str(stage_dir)]
            done = subprocess.run(
                argv, cwd=str(stage_dir),
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if done.returncode != 0:
                tail = (done.stderr or done.stdout or "").strip().splitlines()
                return {"error": f"{folder} 단계에서 멈췄습니다: "
                                 f"{tail[-1] if tail else '알 수 없음'}"}

            if folder == "01_label_removal":
                # 1단계는 결과를 네 갈래(1_labels_white · 2_labels_inpainted ·
                # 3_labels_points_white · 4_labels_points_inpainted)로 나눠
                # 하위 폴더에 넣는다. 2단계는 그중 4번 한 장을 자기 위
                # 폴더에서 찾으므로 올려 준다.
                made = list((stage_dir / "4_labels_points_inpainted").glob("*.png"))
                if not made:
                    return {"error": "라벨 제거 결과(4_labels_points_inpainted)를 "
                                     "찾지 못했습니다."}
                shutil.copy2(made[0], stage_dir / made[0].name)

        picked = json.loads(
            (root / "output" / "06_nearest_zero_points"
             / "nearest_zero_points.json").read_text(encoding="utf-8"))
        if keep_dir is not None:
            shutil.copytree(root, keep_dir, dirs_exist_ok=True)

    result = {"prefix": prefix, "raw": picked,
              "regions": _regions_of(picked), "lines": _lines_of(picked)}
    _cache[key] = result
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return result


def _regions_of(picked: dict) -> list:
    """보정 영역 요약 — 화면에 무엇을 왜 골랐는지 보여주려고."""
    out = []
    for region in picked.get("regions", []):
        chosen = region.get("selected_zero_points") or []
        out.append({
            "label": region.get("region_label"),
            "area": region.get("area_px"),
            "status": region.get("selection_status"),
            "zeroPoints": [z.get("label") for z in chosen],
            "attempts": region.get("pair_attempt_count"),
            "coverage": (region.get("closure_validation") or {})
                        .get("target_coverage_ratio"),
        })
    return out


def _lines_of(picked: dict) -> list:
    """구역마다 제로라인 하나 — [[x, y], ...] 목록.

    좌표는 closure_validation.route.path_points 에 있다. 그 옆의
    contour_arc_points 는 영역을 닫을 때 쓴 **윤곽 쪽 호**라 제로라인이
    아니다 — 부품 테두리를 그대로 따라간다.
    """
    lines = []
    for region in picked.get("regions", []):
        route = (region.get("closure_validation") or {}).get("route") or {}
        points = route.get("path_points") or []
        if len(points) >= 2:
            lines.append([[float(x), float(y)] for x, y in points])
    return lines


__all__ = ["KNOWN_PREFIXES", "PIPELINE", "prefix_for", "run"]
