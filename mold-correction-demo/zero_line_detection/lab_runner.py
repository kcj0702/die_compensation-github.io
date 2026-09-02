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
단계를 다 도는 데 시간이 걸린다 — 실측 64XX2 가 **117초**로, 분석 한
번에서 가장 큰 덩어리다(Qwen 판독 57초보다 크다). 그래서 같은 그림이면
다시 돌리지 않는다.

메모리에만 들고 있으면 서버를 껐다 켤 때마다 다시 돈다. 파이썬 코드를
고치면 엔진을 다시 띄워야 하는 프로젝트라 그 일이 잦다. 그래서 디스크에도
남긴다 — 열쇠에 **스크립트 내용 해시**를 넣으므로, 받은 코드가 바뀌면
저절로 무효가 된다.
"""
from __future__ import annotations

import hashlib
import json
import os
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
# 환경변수로 옮길 수 있다(ADC_LAB_CACHE) — 시험이 실제 캐시를 건드리지
# 않게 하려고 넣었고, 운영에서는 공유 폴더로 돌릴 수 있다.
CACHE_DIR = Path(os.environ.get("ADC_LAB_CACHE") or (HERE / ".lab_cache"))

# out_of_tolerance.py 의 SCAN_SCALES 에 등록된 품번들. 여기 없으면 못 돈다.
KNOWN_PREFIXES = {
    "64XX2": "JD_64XX2-DR000",
    "67XX6": "JD_67XX6-DR000",
    "71XX2": "JD_71XX2-DR000",
}

# 라벨 제거도 **그쪽 것**을 쓴다. 우리 label_removal 은 컬러바 범례의
# 눈금 숫자까지 라벨로 오인해 범례를 통째로 지워, 2단계가
# "color bar could not be detected" 로 멈춘다(실측 64XX2·67XX6).
COMMON_STAGES = [
    ("01_label_removal", "remove_labels.py"),
    ("02_contour_graph", "contour_graph.py"),
    ("03_zero_point_selection", "zero_point_selection.py"),
    ("04_out_of_tolerance", "out_of_tolerance.py"),
    ("05_merged_correction_regions", "merge_correction_regions.py"),
    ("06_nearest_zero_points", "select_nearest_zero_points.py"),
]

# 67XX6 은 두 단계가 더 있다.
#
# 이 부품은 선루프라 가운데가 통째로 비어 있는데, 6단계까지의 규칙은
# "inner openings and holes are traversable" 라 그 빈 데를 대각선으로
# 가로질렀다. 7단계는 **바깥 윤곽·안쪽 윤곽·보정 영역을 뺀 통로 안에서만**
# 제로라인을 뻗어 그 문제를 없앤다. 결과가 선이 아니라 가지 마스크라,
# 이 부품의 제로라인은 **영역**으로 다룬다 — 시트 표기와도 맞는다.
EXTRA_STAGES = {
    "JD_67XX6-DR000": [
        ("07_zero_line_branch_expansion", "branch_expand_zero_lines.py"),
        ("08_zero_region_on_label_removed_scan", "overlay_zero_region.py"),
    ],
}


def stages_for(prefix: str) -> list:
    return COMMON_STAGES + EXTRA_STAGES.get(prefix, [])

_cache: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 8


def _script_stamp(prefix: str) -> str:
    """이 부품이 쓰는 스크립트 내용의 해시.

    받은 코드가 바뀌면 캐시가 저절로 무효가 되게 하려고 넣는다. 사람이
    판 번호를 올리는 것을 잊어도 안전하다.
    """
    digest = hashlib.blake2b(digest_size=8)
    folder = PIPELINE / prefix
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _disk_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


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
        np.ascontiguousarray(scan_bgr).tobytes() + prefix.encode()
        + _script_stamp(prefix).encode(),
        digest_size=16).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    stored = _disk_path(key)
    if stored.exists():
        try:
            found = json.loads(stored.read_text(encoding="utf-8"))
            _cache[key] = found
            return found
        except Exception:
            stored.unlink(missing_ok=True)   # 깨졌으면 그냥 다시 돈다

    area_contours: list = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / f"{prefix} 3D 스캔"
        _write(root / "input" / f"{prefix} 3D 스캔.png", scan_bgr)

        for folder, script in stages_for(prefix):
            stage_dir = root / "output" / folder
            stage_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PIPELINE / prefix / script, stage_dir / script)
            # 어떤 단계는 미리 만들어진 자료를 자기 폴더에서 입력으로
            # 찾는다(67XX6 3단계의 mylab_deviation_graph.json). 2단계
            # 산출물이 아니라 my_lab 이 따로 만든 것이라 같이 넣어 준다.
            assets = PIPELINE / prefix / "assets" / folder
            if assets.is_dir():
                for item in assets.iterdir():
                    shutil.copy2(item, stage_dir / item.name)
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
                # 2단계는 4번 갈래를, 67XX6 의 8단계는 2번 갈래를 각각
                # 자기 위 폴더에서 찾는다. 둘 다 올려 준다.
                lifted = 0
                for branch in ("4_labels_points_inpainted", "2_labels_inpainted"):
                    for made in (stage_dir / branch).glob("*.png"):
                        shutil.copy2(made, stage_dir / made.name)
                        lifted += 1
                if not lifted:
                    return {"error": "라벨 제거 결과를 찾지 못했습니다."}

        picked = json.loads(
            (root / "output" / "06_nearest_zero_points"
             / "nearest_zero_points.json").read_text(encoding="utf-8"))

        # 7단계가 있으면 그 가지 마스크를 제로 **영역**으로 쓴다.
        # 6단계 선보다 뒤에 나온 결과이고, 빈 공간을 지나지 않는다.
        branch = root / "output" / "07_zero_line_branch_expansion"             / "branch_expanded_area_mask.png"
        if branch.exists():
            data = np.fromfile(str(branch), dtype=np.uint8)
            mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                found, _h = cv2.findContours(
                    (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                area_contours = [c.reshape(-1, 2).tolist() for c in found
                                 if cv2.contourArea(c) >= 200]

        if keep_dir is not None:
            shutil.copytree(root, keep_dir, dirs_exist_ok=True)

    result = {"prefix": prefix, "raw": picked,
               "regions": _regions_of(picked), "lines": _lines_of(picked),
               "areas": area_contours}
    if area_contours:
        # 영역이 있으면 그것이 이 부품의 제로라인이다 — 선은 6단계
        # 중간 결과라 화면에는 쓰지 않는다.
        result["lines"] = []
    _cache[key] = result
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        _disk_path(key).write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass          # 캐시를 못 써도 결과는 이미 나왔다
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
