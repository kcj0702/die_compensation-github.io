"""받은 '최종 제로라인 검토 코드'(zero_line (2))를 우리 쪽에서 돌린다.

[무엇이 달라지나 — 67XX6]
앞서 쓰던 7·8단계(가지 뻗기)는 링을 따라 구불구불한 띠 하나를 냈다.
그것을 네모로 다듬어 봤지만, 굽은 띠를 네모로 싸는 이상 헛덮음이 40%
아래로 안 내려갔다.

이 묶음은 아예 다르게 푼다. 비보정 영역이 부품의 40% 미만이고 서로
떨어져 있으면(Case 1) — 67XX6 이 그렇다 — 부품 윤곽을 안쪽으로 여섯 겹
등간격으로 뜬 뒤 보정 경계와의 **교점을 직선으로 이어 다각형**을 만든다.
경계가 직선이라 처음부터 깔끔하다.

    실측 67XX6: Z1~Z8 여덟 구역, 전부 직선 다각형

64XX2·71XX2 는 Case 2(보로노이 분리선)로 간다.

[왜 껍데기가 필요한가]
그쪽 스크립트는 `<루트>/experiments/zero_line_area_edge_preview/` 에
자기가 있고 `<루트>/zero_line_detection/zero_boundary.py` 를 import 한다고
전제한다(`DEMO_ROOT = HERE.parents[1]`). 그 구조를 임시 폴더에 그대로
만들어 주고 그 안에서 부른다 — lab_runner 와 같은 원칙이다. **받은 코드는
한 줄도 고치지 않는다.**

그리고 그쪽 main() 은 SPECS 세 부품을 모두 돌며, 하나라도 입력이 없으면
멈춘다. 앱에서는 한 부품만 있는 게 보통이라, 우리 껍데기에서 SPECS 를
그 부품 하나로 좁혀 놓고 main() 을 부른다. 이것도 그쪽 코드를 고치는
것이 아니라 **부르는 쪽에서 범위를 정하는** 것이다.

[입력]
  <입력폴더>/<키> 3D 스캔_2_labels_inpainted.png   라벨 제거본(1단계 2번 갈래)
  <입력폴더>/colormap/<키> 3D 스캔.png             컬러바 범례를 잘라 낸 그림
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
BUNDLE = HERE / "adaptive_bundle"
CACHE_DIR = Path(os.environ.get("ADC_ADAPTIVE_CACHE") or (HERE / ".adaptive_cache"))

# 그쪽 SPECS 의 값 범위. 부품마다 컬러바 눈금이 다르다.
SPAN = {
    "JD_64XX2-DR000": (-1.5, 2.0),
    "JD_67XX6-DR000": (-3.0, 3.0),
    "JD_71XX2-DR000": (-2.0, 2.0),
}
# 컬러바를 자를 때 둘레로 남기는 여백(px). 그쪽 extract_color_ramp 는
# 램프가 그림 높이의 65% 를 넘어야 받아들이므로 넉넉히 두면 안 된다.
LEGEND_PAD = 6
# 이보다 작은 조각은 버린다(px).
MIN_AREA_PX = 200
# 윤곽을 얼마나 단순화할지(둘레 대비). 마스크에서 딴 윤곽은 픽셀 계단이
# 그대로 남아 꼭짓점이 88~253 개다. 이 묶음이 만드는 경계는 원래
# **직선**이므로, 계단만 걷어내면 꼭짓점 7~14 개짜리 다각형이 된다.
SIMPLIFY = 0.01

_cache: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 8


def key_for(part_no: str | None) -> str | None:
    """품번 -> 이 묶음이 아는 이름. 모르면 None."""
    folded = str(part_no or "").upper().replace("-", "").replace("_", "")
    for key in SPAN:
        if key.split("_")[1].split("-")[0] in folded:
            return key
    return None


def _legend_crop(scan_bgr: np.ndarray, key: str) -> np.ndarray | None:
    """원본에서 컬러바 범례만 잘라 낸다."""
    from zero_line_detection.colorbar import detect_colorbar

    vmin, vmax = SPAN[key]
    try:
        bar = detect_colorbar(
            cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2RGB), vmin=vmin, vmax=vmax)
    except Exception:
        return None
    box = bar.info
    return scan_bgr[max(0, box.y0 - LEGEND_PAD):box.y1 + LEGEND_PAD,
                    max(0, box.x0 - LEGEND_PAD):box.x1 + LEGEND_PAD]


def _shim(key: str, script: str, argv: list) -> str:
    """SPECS 를 한 부품으로 좁히고 그쪽 main() 을 부르는 껍데기."""
    return (
        "import sys\n"
        f"sys.argv = {['x'] + argv!r}\n"
        f"import {script} as step\n"
        "step.SPECS = tuple(s for s in step.SPECS "
        f"if s.key == {key!r})\n"
        "step.main()\n"
    )


def run(scan_bgr: np.ndarray, cleaned_bgr: np.ndarray,
        part_no: str | None, keep_dir: Path | None = None) -> dict:
    """스캔과 라벨 제거본으로 제로 영역을 만든다.

    Args:
        scan_bgr: 원본 스캔(BGR). 컬러바 범례를 여기서 잘라 낸다.
        cleaned_bgr: 라벨 제거본(1단계 2번 갈래).
        part_no: 품번.

    Returns:
        {"key", "areas": [[[x,y], ...]], "summary": {...}}
        돌릴 수 없으면 {"error": "..."}.
    """
    key = key_for(part_no)
    if key is None:
        return {"error": f"이 묶음에 등록되지 않은 품번입니다: {part_no}"}

    # 열쇠에 **이 파일 자신도** 넣는다.
    #
    # 받은 코드만 해시했다가, 우리 쪽 단순화 값(SIMPLIFY)을 바꿨는데도
    # 캐시가 그대로 걸려 꼭짓점이 88~248 개인 옛 결과가 나왔다. 결과를
    # 바꾸는 것은 받은 코드만이 아니다.
    stamp = hashlib.blake2b(digest_size=8)
    for path in sorted(BUNDLE.rglob("*.py")) + [Path(__file__).resolve()]:
        stamp.update(path.read_bytes())
    cache_key = hashlib.blake2b(
        np.ascontiguousarray(cleaned_bgr).tobytes()
        + np.ascontiguousarray(scan_bgr).tobytes()
        + key.encode() + stamp.hexdigest().encode(),
        digest_size=16).hexdigest()
    if cache_key in _cache:
        _cache.move_to_end(cache_key)
        return _cache[cache_key]
    stored = CACHE_DIR / f"{cache_key}.json"
    if stored.exists():
        try:
            found = json.loads(stored.read_text(encoding="utf-8"))
            _cache[cache_key] = found
            return found
        except Exception:
            stored.unlink(missing_ok=True)

    legend = _legend_crop(scan_bgr, key)
    if legend is None or not legend.size:
        return {"error": "컬러바 범례를 찾지 못해 이 방식을 쓸 수 없습니다."}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work = root / "experiments" / "zero_line_area_edge_preview"
        work.mkdir(parents=True)
        (root / "zero_line_detection").mkdir()
        for path in BUNDLE.glob("*.py"):
            if path.name == "zero_boundary.py":
                shutil.copy2(path, root / "zero_line_detection" / path.name)
            else:
                shutil.copy2(path, work / path.name)
        (root / "zero_line_detection" / "__init__.py").write_text("", "utf-8")

        feed = root / "input"
        (feed / "colormap").mkdir(parents=True)
        _write(feed / f"{key} 3D 스캔_2_labels_inpainted.png", cleaned_bgr)
        _write(feed / "colormap" / f"{key} 3D 스캔.png", legend)

        for script, argv in (
                ("generate_correction_only_3pct_preview",
                 ["--input-dir", str(feed)]),
                ("generate_adaptive_zero_line_preview", [])):
            shim = work / f"_run_{script}.py"
            shim.write_text(_shim(key, script, argv), encoding="utf-8")
            done = subprocess.run(
                [sys.executable, shim.name], cwd=str(work),
                capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            if done.returncode != 0:
                tail = (done.stderr or done.stdout or "").strip().splitlines()
                return {"error": f"{script} 에서 멈췄습니다: "
                                 f"{tail[-1] if tail else '알 수 없음'}"}

        made = work / "results_adaptive_zero_line_2pct" / key
        mask_path = made / "final_zero_line_mask.png"
        if not mask_path.exists():
            return {"error": "제로 영역 결과를 찾지 못했습니다."}
        data = np.fromfile(str(mask_path), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        found, _h = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for contour in found:
            if cv2.contourArea(contour) < MIN_AREA_PX:
                continue
            slack = SIMPLIFY * cv2.arcLength(contour, True)
            simple = cv2.approxPolyDP(contour, slack, True).reshape(-1, 2)
            areas.append((simple if len(simple) >= 3
                          else contour.reshape(-1, 2)).tolist())

        summary = {}
        try:
            summary = json.loads(
                (made / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        if keep_dir is not None:
            shutil.copytree(made, keep_dir, dirs_exist_ok=True)

    result = {"key": key, "areas": areas,
              "method": summary.get("selected_method"),
              "zeroRatio": summary.get("zero_area_ratio_of_part")}
    _cache[cache_key] = result
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        stored.write_text(json.dumps(result, ensure_ascii=False), "utf-8")
    except Exception:
        pass
    return result


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 로 만들지 못했습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


__all__ = ["BUNDLE", "SPAN", "key_for", "run"]
