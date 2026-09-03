"""3D 시각화 백엔드 — CAD 읽기 · 스캔 얹기 · 보정 후 형상.

[왜 따로 두나]
main 의 server.py 는 분석·보정시트·이력 쪽으로 커졌고, 3D 는 파일 하나
읽는 수준만 남아 있었다. 3D 배선을 그 파일에 다시 끼워 넣으면 700 줄이
섞여 서로 건드리기 어려워진다. 3D 는 입력(스캔 분석 결과 + CAD 파일)과
출력(얹은 좌표)이 분명해서 떼어 놓기 좋다.

server.py 는 이 모듈에서 두 가지만 가져다 쓴다 —
  · remember_analysis(...)  분석 결과를 3D 가 쓸 수 있게 맡긴다
  · ROUTES                  3D 라우트 목록

[무엇이 들어 있나]
  /api/cad             STEP·STL 을 읽어 형상 + 조립 홀 + 기준면
  /api/cad-overlay     스캔의 제로라인·보정량을 CAD 표면 위로 옮긴다
  /api/cad-sections    시트 단면 표기(H·T)로 CAD 를 잘라 제로라인을 만든다
  /api/cad-morph       보정량만큼 민 "보정 후" 형상
  /api/cad-morph-stl   그 형상을 STL 로 (전체 · 덧살만 · 깎기만)
  /api/cad-morph-open  그 형상을 새 CAD 로 등록해 원본처럼 다룬다
"""
from __future__ import annotations

import json
import tempfile
import urllib.parse
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

# 업로드 상한. 실측 STEP 이 215MB 까지 온다.
MAX_UPLOAD_BYTES = 300 * 1024 * 1024

# ── 읽어 둔 CAD ──────────────────────────────────────────────────
# 215MB STEP 이 42~100초 걸린다. 오버레이를 그릴 때마다 다시 읽을 수 없다.
# 여러 개를 열어 놓고 골라 보므로 3개로는 모자란다(4개째를 열면 첫 파일이
# 밀려나 "CAD 가 만료됐습니다" 가 뜬다). 실측 64XX1 이 파싱 후 약 40MB다.
_cad_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_CAD_CACHE_MAX = 6


def _cache_cad(entry: dict[str, Any]) -> str:
    cad_id = uuid.uuid4().hex
    _cad_cache[cad_id] = entry
    while len(_cad_cache) > _CAD_CACHE_MAX:
        _cad_cache.popitem(last=False)
    return cad_id


# ── 스캔 분석 결과 ───────────────────────────────────────────────
# 3D 는 분석을 다시 하지 않는다. 화면이 분석해 둔 것을 아이디로 가리킨다.
_analysis_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_ANALYSIS_CACHE_MAX = 8


def remember_analysis(entry: dict[str, Any]) -> str:
    """분석 결과를 3D 가 쓸 수 있게 맡긴다. 아이디를 돌려준다.

    Args:
        entry: 적어도 part_mask(부품 마스크)와 deviation_points 가 있어야
            한다. values(픽셀별 편차) · zero_lines · part_no 는 있으면 쓴다.
    """
    analysis_id = uuid.uuid4().hex
    _analysis_cache[analysis_id] = entry
    while len(_analysis_cache) > _ANALYSIS_CACHE_MAX:
        _analysis_cache.popitem(last=False)
    return analysis_id


def reset_caches() -> None:
    """시험에서 서로 영향을 주지 않게 비운다."""
    _cad_cache.clear()
    _analysis_cache.clear()
    _overlay_cache.clear()


def _decode_image(payload: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다.")
    return image


def _shift_centers(features: list[dict], offset) -> list[dict]:
    """center 를 offset 만큼 평행이동한다. axis/normal 은 방향이라 두 손 뗀다."""
    if not offset:
        return features
    ox, oy, oz = (float(v) for v in offset)
    moved = []
    for feature in features:
        item = dict(feature)
        centre = item.get("center")
        if centre and len(centre) == 3:
            item["center"] = [round(centre[0] - ox, 4),
                              round(centre[1] - oy, 4),
                              round(centre[2] - oz, 4)]
        moved.append(item)
    return moved


def load_cad_payload(payload: bytes, filename: str) -> dict[str, Any]:
    """업로드된 3D 파일을 읽어 뷰어용 메시 + RPS 후보를 만든다.

    STEP 이면 홀·기준평면까지 뽑고, STL/PLY/OBJ 면 메시만 준다 —
    삼각망으로 내보낸 시점에 원통면 정보가 이미 사라지기 때문이다.
    """
    from cad_import import mesh_io, step_reader

    suffix = Path(filename).suffix.lower()
    # OCCT/trimesh 는 경로 기반이라 임시 파일로 떨군다
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (Path(filename).name or "upload")
        path.write_bytes(payload)

        if step_reader.is_step_file(path):
            parsed = step_reader.read_step_full(path)
            web = mesh_io.to_web_mesh(
                parsed["mesh"], name=path.stem, source_format="step")
            # to_web_mesh 는 정점을 원점으로 옮긴다(자동차 부품은 차량
            # 좌표계라 원점에서 수천 mm 떨어져 있다). 홀·평면 좌표도 같은
            # 만큼 옮겨야 형상 위에 얹힌다 — 안 옮기면 실측 001 REINF SIDE
            # OTR 기준 중심이 (1584, -653, 491) 이라 홀이 1.8m 밖에 뜬다.
            # 축과 법선은 방향이라 그대로 둔다.
            offset = web["summary"]["bounds"]["center"] if web.get("recentered") else None
            web["holes"] = _shift_centers(parsed["holes"], offset)
            web["planes"] = _shift_centers(parsed["planes"][:50], offset)
            web["counts"] = parsed["counts"]

            # 오버레이(제로라인·보정량)를 그리려면 원본 삼각망이 필요하다.
            # 화면용 메시는 간략화돼 있어 광선 교차에 쓰면 어긋난다.
            mesh_full = parsed["mesh"]
            web["cadId"] = _cache_cad({
                "mesh": mesh_full,
                "offset": np.asarray(offset, dtype=float) if offset else np.zeros(3),
                "name": path.stem,
                # 히트맵은 화면에 그리는 정점에 입혀야 한다. 원본 삼각망은
                # 간략화 전이라 개수가 달라서 그대로 쓰면 어긋난다.
                "display_vertices": np.asarray(
                    web["positions"], dtype=float).reshape(-1, 3),
                "display_faces": np.asarray(web["indices"]).reshape(-1, 3),
            })
            return web

        if mesh_io.is_mesh_file(path):
            mesh = mesh_io.load_mesh(path)
            web = mesh_io.to_web_mesh(
                mesh, name=path.stem, source_format=suffix.lstrip("."))
            # 삼각망엔 B-Rep 특징이 없다. 빈 배열로 명시해 프론트가
            # "아직 안 읽음"과 "원래 없음"을 구분할 수 있게 한다.
            web["holes"] = []
            web["planes"] = []
            web["counts"] = {"cylinders": 0, "holes": 0, "planes": 0}
            web["note"] = (
                "삼각망 파일이라 홀·기준평면 정보가 없습니다. "
                "RPS 정렬이 필요하면 STEP(AP214)으로 받아야 합니다."
            )
            return web

    if suffix == ".catpart":
        raise ValueError(
            "CATIA 네이티브(.CATPart)는 독자 포맷이라 읽을 수 없습니다. "
            "CATIA에서 STEP(AP214) 또는 STL로 내보내 주세요."
        )
    raise ValueError(
        f"지원하지 않는 형식입니다: {suffix or '확장자 없음'} "
        f"(지원: STEP/STP, STL, PLY, OBJ, GLB/GLTF, 3MF)"
    )


# CAD 파일명과 스캔 품번이 다르다. 현업 제품데이터 폴더가 짝을 보여준다 —
#   64XX1-DR000_HDCT1860.CATPart  <->  64XX2-DR000 제품데이터.png (LH/RH)
#   67XX6-DR050_HDCT1750.CATPart  <->  67XX6-DR050 제품데이터.png
#   71XX1-DR000_HDCT0458.CATPart  <->  71XX2-DR000 제품데이터.png
# 끝자리 1/2 는 좌우 대칭품이라 CAD 는 한쪽만 온다.
CAD_TO_SCAN_PART = {
    "64XX1": "64XX2",
    "71XX1": "71XX2",
    "67XX6": "67XX6",
}


def scan_part_for_cad(cad_name: str) -> str | None:
    """CAD 파일명에서 짝이 되는 스캔 품번을 찾는다."""
    folded = str(cad_name or "").upper().replace("-", "").replace("_", "")
    for cad_key, scan_key in CAD_TO_SCAN_PART.items():
        if cad_key in folded:
            return scan_key
    return None


def apply_zero_edits(raw_lines: list, zero_edits: list | None) -> list:
    """보정시트에서 손본 제로라인을 그대로 따른다.

    3D 의 제로라인은 시트가 정하는 것이라, 시트에서 옮기거나 숨긴 것이
    3D 에 안 비치면 둘이 어긋난다. 옮김은 그림 좌표(px)로 받아 **광선을
    쏘기 전에** 더한다 — 3D 에서 따로 밀면 표면에서 떠 버린다.
    """
    if not zero_edits:
        return raw_lines
    moved: list = []
    for index, line in enumerate(raw_lines):
        edit = next((e for e in zero_edits
                     if int(e.get("index", -1)) == index), None)
        if edit is None:
            moved.append(line)
            continue
        if edit.get("hidden"):
            continue
        dx = float(edit.get("dx") or 0.0)
        dy = float(edit.get("dy") or 0.0)
        moved.append({**line, "points": [[p[0] + dx, p[1] + dy]
                                         for p in line["points"]]})
    return moved


# 적응형 제로라인(zero_line (2) 묶음)을 쓸 부품.
# 이 묶음의 Case 1(윤곽 교점 다각형)로 가는 부품만 넣는다.
ADAPTIVE_PARTS = {"JD_67XX6-DR000"}

# 자세 후보를 몇 개까지 광선으로 검증할지. 하나에 1초 안쪽이다.

OVERLAY_TRIES = 6

# 스캔을 CAD 에 얹은 결과. 같은 짝을 다시 보는 일이 잦다(탭을 옮기거나
# 제로라인을 손봤다 되돌리거나). 실측 20초짜리를 다시 돌릴 이유가 없다.
_overlay_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_OVERLAY_CACHE_MAX = 12


def reset_overlay_cache() -> None:
    """시험에서 캐시가 새어 나가지 않게 비운다."""
    _overlay_cache.clear()


def _overlay_key(cad_id: str, analysis_id: str, zero_edits,
                 fit_adjust=None) -> str:
    return "|".join([cad_id, analysis_id,
                     json.dumps(zero_edits or [], sort_keys=True),
                     json.dumps(fit_adjust or {}, sort_keys=True)])


def cad_overlay_for(cad_id: str, analysis_id: str,
                    zero_edits: list | None = None,
                    fit_adjust: dict | None = None) -> dict[str, Any]:
    """스캔에서 뽑은 제로라인·포인트를 CAD 표면 위의 3D 좌표로 옮긴다.

    보정량은 여기서 계산하지 않는다. 최종 보정시트는 작업자가 값을
    고치고 포인트를 빼기도 하는데, 그 상태는 화면이 들고 있다. 백엔드가
    따로 계산하면 3D 와 시트가 어긋난다 — 위치만 주고 값은 화면이 정한다.

    실루엣을 맞춰 어느 방향에서 본 그림인지 찾고, 그 방향으로 광선을
    쏴서 표면에 얹는다. 데이텀 정합이 아니라 겉모양 정합이라
    fit.iou 를 같이 내보낸다 — 낮으면 화면에서 경고해야 한다.
    """
    from cad_import import overlay as ov

    cad_entry = _cad_cache.get(cad_id)
    if cad_entry is None:
        raise ValueError("CAD 가 만료됐습니다. 3D 파일을 다시 여세요.")
    analysis = _analysis_cache.get(analysis_id)
    if analysis is None:
        raise ValueError("분석 결과가 만료됐습니다. 이미지를 다시 분석하세요.")

    cache_key = _overlay_key(cad_id, analysis_id, zero_edits, fit_adjust)
    if cache_key in _overlay_cache:
        _overlay_cache.move_to_end(cache_key)
        return _overlay_cache[cache_key]

    mesh = cad_entry["mesh"]
    offset = cad_entry["offset"]
    vertices = np.asarray(mesh.vertices, dtype=float) - offset
    faces = np.asarray(mesh.faces)

    import trimesh

    # LH·RH 가 한 파일에 들어 있으면 갈라서 **각각** 맞춰 보고 잘 맞는
    # 쪽을 쓴다. 스캔은 한 짝뿐이라 두 짝을 합친 실루엣과는 맞지 않는다
    # (실측 71XX1: 통째로 30% -> 한 짝만 쓰면 아래 hit_rate 참고).
    best = None
    for half in ov.split_sides(vertices, faces):
        half_vertices, half_faces, cut_axis, cut_mid, cut_side = half
        piece = trimesh.Trimesh(vertices=half_vertices, faces=half_faces,
                                process=False)
        # 겹침 넓이가 아니라 **점이 형상에 얹히는 비율**로 판정한다.
        # 껍질 겹침은 후하고(64XX2 96.9%) 실루엣은 박하다(42.2%) — 둘 다
        # 오버레이가 쓸 만한지를 못 가린다. overlay.MIN_HIT_RATE 참고.
        #
        # 그래서 자세 후보를 몇 개 받아 **광선을 쏴서** 고른다. 겹침으로
        # 하나만 받으면 얹힘이 더 좋은 자세를 놓칠 수 있다.
        seen: set = set()
        for guess in ov.fit_view(half_vertices, half_faces,
                                 analysis["part_mask"], top_k=OVERLAY_TRIES):
            key = (guess.axis, guess.sign, guess.flip_u, guess.flip_v,
                   round(guess.angle, 3), round(guess.mm_per_px, 4),
                   round(guess.origin_u, 1), round(guess.origin_v, 1))
            if key in seen:
                continue          # 같은 자세가 여러 번 올라온다
            seen.add(key)
            guess.hit_rate = round(ov.measure_hit_rate(
                guess, half_vertices, half_faces,
                analysis["part_mask"], piece), 4)
            if best is None or guess.hit_rate > best[0].hit_rate:
                best = (guess, half_vertices, half_faces, piece,
                        cut_axis, cut_mid, cut_side)

    fit, vertices, faces, shifted, cut_axis, cut_mid, cut_side = best

    # 마지막으로 **얹힘 비율을 직접 올린다.** 자세 찾기는 실루엣 겹침을
    # 보는데, 그건 대리 지표라 부품에 따라 크게 어긋난다.
    #   64XX2 99.7 -> 99.7 · 67XX6 88.0 -> 99.0 · 71XX2 67.7 -> 90.0
    fit = ov.polish_by_hit_rate(
        fit, vertices, faces, analysis["part_mask"], shifted)

    # 작업자가 손으로 맞춘 값이 있으면 그대로 따른다. 자동 정합은
    # 실루엣만 보므로 몇 퍼센트가 모자랄 수 있는데, 그때 사람이 조금
    # 돌리고 옮겨서 맞출 수 있어야 쓸 수 있는 도구가 된다.
    if fit_adjust:
        fit = ov.nudge_fit(
            fit, analysis["part_mask"].shape,
            angle_deg=float(fit_adjust.get("angle") or 0.0),
            dx=float(fit_adjust.get("dx") or 0.0),
            dy=float(fit_adjust.get("dy") or 0.0),
            scale=float(fit_adjust.get("scale") or 1.0))
        # 손으로 옮겼으면 얹힘도 다시 잰다 — 나아졌는지 보여야 한다
        fit.hit_rate = round(ov.measure_hit_rate(
            fit, vertices, faces, analysis["part_mask"], shifted), 4)

    fit.reliable = fit.hit_rate >= ov.MIN_HIT_RATE

    # 표면에 얹지 못한 점(광선이 빗나간 자리)은 뺀다. 예전에는 아무
    # 정점으로나 채워서 제로라인이 부품 밖으로 길게 뻗었다.
    def _densify(points: list, step_px: float = 4.0) -> list:
        """선 위를 촘촘히 채운다.

        꼭짓점만 표면에 얹으면 그 사이는 공중을 가로지른다. 촘촘히
        쏴야 곡면을 그대로 따라간다 — 3D 에서 "선을 얹은 느낌" 을
        없애는 진짜 방법이다(표면을 칠하면 리브와 구멍에서 조각난다).
        """
        dense: list = []
        for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
            span = float(np.hypot(bx - ax, by - ay))
            count = max(int(span / step_px), 1)
            for k in range(count):
                t = k / count
                dense.append([ax + (bx - ax) * t, ay + (by - ay) * t])
        dense.append(list(points[-1]))
        return dense

    # 우선순위: 현업 파이프라인 > 시트 등록 정답 > 우리 검출
    raw_lines = [{"line_id": i + 1, "points": pts}
                 for i, pts in enumerate(analysis.get("lab_zero_lines") or [])]
    if not raw_lines:
        raw_lines = [{"line_id": l.get("line_id"), "points": l["points"]}
                     for l in analysis.get("zero_lines", [])]

    raw_lines = apply_zero_edits(raw_lines, zero_edits)

    lines = []
    dropped_line_points = 0
    for line in raw_lines:
        pts = line["points"]
        if len(pts) < 2:
            continue
        placed = ov.unproject(_densify(pts), vertices, faces, fit, shifted)
        kept = [spot for spot in placed if spot is not None]
        dropped_line_points += len(placed) - len(kept)
        if len(kept) < 2:
            continue

        # 표면에 얹힌 구간과 **빈 공간을 지나는 구간**을 나눠 준다.
        #
        # 광선이 빗나갔다는 것은 그 자리에 부품이 없다는 뜻이다(구멍·
        # 개구부). 이어서 실선으로 그리면 없는 자리에 선이 있는 것처럼
        # 보인다 — 받은 파이프라인은 "inner openings and holes are
        # traversable" 라 링 부품에서 실제로 빈 데를 지난다. 그 구간은
        # 화면에서 점선으로 그려 사실대로 보이게 한다.
        runs: list = []
        current: list = []
        for spot in placed:
            if spot is None:
                if len(current) >= 2:
                    runs.append(current)
                current = []
            else:
                current.append(spot)
        if len(current) >= 2:
            runs.append(current)

        gaps = [[runs[i][-1], runs[i + 1][0]] for i in range(len(runs) - 1)]
        lines.append({
            "line_id": line.get("line_id"),
            "points": kept,          # 예전 형식 — 통째로 이은 것
            "runs": runs,            # 표면에 얹힌 구간 (실선)
            "gaps": gaps,            # 빈 공간을 지나는 구간 (점선)
        })

    # 컬러바 범위 밖의 값은 판독 오류다. 실측(JD_67XX6, 컬러바 +3.0~-3.0)
    # 에서 +9.00 이 5건 나왔다. 3D 에 얹으면 화살표 길이 기준을 잡아먹어
    # 진짜 보정량(0.1~3mm)이 전부 점만 해진다.
    # 컬러바 범위는 **분석이 알려 준 것**만 쓴다. 품번별 표를 여기에
    # 또 두면 분석 쪽과 갈라진다 — 없으면 거르지 않는다.
    span = analysis.get("colorbar_span")
    limit = max(abs(span[0]), abs(span[1])) * 1.05 if span else None

    wanted, rejected = [], []
    for point in analysis.get("deviation_points", []):
        value = float(point.get("value", 0.0))
        if limit is not None and abs(value) > limit:
            rejected.append({"id": point.get("id"), "value": round(value, 3)})
            continue
        wanted.append(point)

    # 광선은 한 번에 쏘는 게 훨씬 빠르다
    placed = ov.unproject([[p["xPx"], p["yPx"]] for p in wanted],
                          vertices, faces, fit, shifted)
    points = [
        {"id": point.get("id"), "position": spot,
         "value": round(float(point.get("value", 0.0)), 3)}
        for point, spot in zip(wanted, placed) if spot is not None
    ]
    dropped_points = sum(1 for spot in placed if spot is None)

    # ── 제로라인을 표면에 칠할 도장 ──────────────────────────
    # 3D 공간에 관(tube)으로 띄우면 곡면 위에서 형상과 떠서 "선을 얹은
    # 느낌" 이 난다. 표면 자체를 칠하면 굴곡을 그대로 따라간다.
    #   1 = 선(띠)   2 = 영역
    display = cad_entry.get("display_vertices")
    stencil = np.zeros(analysis["part_mask"].shape, np.uint8)
    reference = analysis.get("zero_reference") or {}
    # zero_shapes 는 굽은 띠를 읽을 수 있는 네모 몇 개로 바꾼다.
    from zero_line_detection import zero_shapes
    # 영역은 **최소 외접 사각형**으로 그린다.
    #
    # 윤곽 그대로 칠했더니 가장자리를 따라 실오라기처럼 갈라져 "물감
    # 칠한 느낌" 이 났다. 시트도 제로 영역을 빗금 친 **네모**로 표기한다 —
    # 경계가 반듯해야 어디까지가 그 영역인지 읽힌다.
    area_contours = list(analysis.get("lab_zero_areas") or [])
    if not area_contours:
        # 시트에 등록해 둔 정답 영역도 같은 규칙으로 도형으로 만든다
        area_contours = zero_shapes.clean(reference.get("contours") or [])

    # 제로 영역의 **테두리**를 따로 만들어 준다.
    #
    # 지금까지는 네모를 표면에 칠하기만 했다. 칠하기는 정점 단위라
    # 경계가 삼각망을 따라 들쭉날쭉해진다 — 네모로 만들어 놓고도
    # 화면에서는 네모로 안 보였다. 네모의 네 변을 제로라인과 똑같이
    # 촘촘히 쏴서 표면에 얹으면 곡면을 타면서도 경계가 반듯하다.
    area_outlines: list = []
    for contour in area_contours:
        pts = np.rint(np.asarray(contour, dtype=float)).astype(np.int32)
        if len(pts) < 3:
            continue

        # 분석 때 이미 네모로 바꿔 두었다(zero_shapes). 여기서 또
        # 손대면 시트에 그린 것과 3D 에 그린 것이 달라진다.
        shape = pts
        cv2.fillPoly(stencil, [shape], 2)
        ring = [[float(x), float(y)] for x, y in shape]
        ring.append(ring[0])          # 닫는다
        area_outlines.append(ring)
    # 우선순위: 현업 파이프라인 > 시트에 등록된 정답 > 우리 검출.
    # 앞의 것일수록 근거가 분명하다.
    # 도장은 **영역**에만 쓴다(67XX6 처럼 시트가 면으로 표기한 경우).
    #
    # 선까지 칠했더니 리브와 구멍이 많은 면에서 조각조각 갈라져
    # "물감 칠한 느낌" 이 났다. 선은 선으로 그리는 게 맞고, 곡면을
    # 따라가게 하려면 촘촘히 쏴서 표면에 얹으면 된다(_densify).

    # 화면용 메시에서 **고른 쪽**만 남기는 가리개.
    #
    # LH·RH 가 한 파일이면 한 짝에 맞춘 자세로 색을 칠하게 되는데,
    # 그대로 두면 반대쪽 살에도 색이 묻는다. 자세를 잡을 때 쓴 것과
    # 같은 평면으로 가른다.
    def _own_side(points: np.ndarray) -> np.ndarray:
        if cut_axis < 0:
            return np.ones(len(points), dtype=bool)
        left = points[:, cut_axis] < cut_mid
        return left if cut_side < 0 else ~left

    # 제로 영역을 표면에 **칠하지 않는다.**
    #
    # 정점 단위로 칠하면 세 꼭짓점이 모두 영역에 든 삼각형만 남아,
    # 리브와 구멍이 많은 면에서 조각조각 갈라진다 — 네모로 만들어
    # 놓고도 화면에서는 물감 칠한 것처럼 보였다. 이제 네모의 테두리를
    # 표면에 얹어 그린다(zero_areas). 도장 계산은 그만큼 뺀다 —
    # 화면용 정점이 40만 개라 공짜가 아니다.
    zero_surface: list = []

    # 영역 테두리를 표면 위로 옮긴다 — 제로라인과 같은 방식이다.
    zero_areas: list = []
    for ring in area_outlines:
        placed = ov.unproject(_densify(ring), vertices, faces, fit, shifted)
        runs, current = [], []
        for spot in placed:
            if spot is None:
                if len(current) >= 2:
                    runs.append(current)
                current = []
            else:
                current.append(spot)
        if len(current) >= 2:
            runs.append(current)
        if runs:
            zero_areas.append({
                "runs": runs,
                "gaps": [[runs[i][-1], runs[i + 1][0]]
                         for i in range(len(runs) - 1)],
            })

    # 표면에 입힐 편차 — 화면용 정점 하나하나에 스캔 값을 찍는다.
    surface = []
    if display is not None:
        spots = np.asarray(display, dtype=float)
        surface = ov.sample_deviation(
            spots, fit, analysis["values"], analysis["part_mask"])
        mine = _own_side(spots)
        if not mine.all():
            surface = [v if mine[i] else None
                       for i, v in enumerate(surface)]

    # 보정 포인트를 **CAD 원래 좌표**로도 준다.
    #
    # CAD(STEP)를 그대로 고쳐 내보내는 것은 못 한다 — 자유곡면 제어점을
    # 연속성 지키며 옮기는 건 전용 소프트웨어 영역이고, 면으로 쪼갠
    # STEP 은 실측 삼각형당 1.6ms · 2.4KB 라 이 부품(363,431 삼각형)이면
    # 10분 · 852MB 다. 대신 CATIA 작업자가 그 자리에 그 값을 넣을 수
    # 있도록 **부품 좌표와 보정량**을 준다. 화면 좌표는 원점을 옮겨
    # 놓았으므로 offset 을 되돌린다.
    back = np.asarray(offset, dtype=float)
    for point in points:
        spot = np.asarray(point["position"], dtype=float) + back
        point["cad"] = [round(float(v), 3) for v in spot]

    answer = {
            "fit": fit.to_dict(), "zeroLines": lines, "points": points,
            "rejected": rejected,
            "zeroSurface": zero_surface,
            "zeroAreas": zero_areas,
            "zeroKind": "areas" if area_contours else (reference.get("kind") or "line"),
            "droppedPoints": dropped_points,
            "droppedLinePoints": dropped_line_points,
            "colorbarLimit": round(limit, 2) if limit else None,
            "scanPart": scan_part_for_cad(cad_entry.get("name", "")),
            "surfaceDeviation": surface,
            "deviationRange": (
                [round(float(np.nanmin(analysis["values"][analysis["part_mask"] > 0])), 2),
                 round(float(np.nanmax(analysis["values"][analysis["part_mask"] > 0])), 2)]
                if analysis.get("part_mask") is not None else None)}

    _overlay_cache[cache_key] = answer
    while len(_overlay_cache) > _OVERLAY_CACHE_MAX:
        _overlay_cache.popitem(last=False)
    return answer


def sheet_excel_for(analysis_id: str, corrections: dict,
                    meta: dict, images: list | None = None) -> bytes:
    """최종 보정시트를 현업 엑셀 양식으로 만든다.

    보정량은 화면이 준다 — 작업자가 고친 값과 계수가 반영된 최종값이다.
    여기서 다시 계산하면 시트와 엑셀이 어긋난다.

    [그림]
    현업 시트("보정 적용 내용")는 스캔 히트맵이 아니라 **3D 형상 그림**을
    쓴다. 그래서 화면에서 찍은 3D 뷰를 받으면 그걸 쓰고, 없으면 스캔에
    콜아웃을 그려 넣은 그림으로 대신한다.
    여러 장을 주면 페이지를 나눠 넣는다 — 실제 시트도 전체도와 확대도를
    따로 싣는다.
    """
    from zero_line_detection.sheet_excel import (
        SheetPoint, build_workbook, draw_sheet_image,
    )

    entry = _analysis_cache.get(analysis_id)
    if entry is None:
        raise ValueError("분석 결과가 만료됐습니다. 이미지를 다시 분석하세요.")

    base_rgb = entry.get("overlay_base")
    if base_rgb is None:
        raise ValueError("시트에 쓸 그림이 없습니다.")
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

    points = []
    for point in entry.get("deviation_points", []):
        point_id = point.get("id")
        if point_id not in corrections:      # 작업자가 숨긴 포인트
            continue
        points.append(SheetPoint(
            point_id=point_id,
            x_px=int(point.get("xPx", 0)),
            y_px=int(point.get("yPx", 0)),
            deviation=float(point.get("value", 0.0)),
            correction=float(corrections[point_id]),
        ))
    points.sort(key=lambda p: p.point_id)

    pages: list = []
    for raw in (images or []):
        text = str(raw or "")
        if "," in text:
            text = text.split(",", 1)[1]
        try:
            buffer = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
            shot = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception:
            shot = None
        if shot is not None:
            pages.append(shot)
    # 1쪽은 **항상** 스캔에 보정치를 그린 전체도다. 예전에는 3D 화면을
    # 담아 두면 그걸로 1쪽을 통째로 대체해서, 정합이 어긋난 CAD 캡처
    # 한 장이 시트가 됐다 — "71XX2 로 만들었는데 다른 제품이 나온다" 는
    # 말이 그것이었다. 현업 시트도 전체도가 먼저고 상세도가 뒤따른다.
    pages = [draw_sheet_image(base_bgr, points)] + pages

    return build_workbook(
        pages, points,
        part_no=str(meta.get("partNo") or entry.get("part_no") or ""),
        part_name=str(meta.get("partName") or ""),
        process=str(meta.get("process") or ""),
        material=str(meta.get("material") or ""),
        control_no=str(meta.get("controlNo") or ""),
        applied_at=str(meta.get("appliedAt") or "") or None,
        coefficient=float(meta.get("coefficient") or 1.0),
        processes=[str(x) for x in (meta.get("processes") or [])],
    )


async def sheet_excel(request: Request) -> Response:
    try:
        body = await request.json()
        corrections = body.get("corrections") or {}
        if not isinstance(corrections, dict) or not corrections:
            return JSONResponse({"error": "보정량이 비어 있습니다."}, status_code=400)
        images = body.get("images")
        if isinstance(images, str):
            images = [images]
        payload = await run_in_threadpool(
            sheet_excel_for, str(body.get("analysisId") or ""),
            {str(k): float(v) for k, v in corrections.items()},
            body.get("meta") or {},
            images if isinstance(images, list) else None,
        )
        name = str(body.get("filename") or "보정시트") + ".xlsx"
        quoted = urllib.parse.quote(name)
        return Response(
            payload,
            media_type=("application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"),
            headers={"Content-Disposition":
                     f"attachment; filename*=UTF-8''{quoted}"},
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


def cad_morph_for(cad_id: str, corrections: dict, positions: dict,
                  reach_ratio: float) -> dict[str, Any]:
    """보정량만큼 민 "보정 후" 형상을 만든다.

    B-Rep 은 건드리지 않는다 — 자유곡면을 연속성 지키며 변형하는 건
    전용 소프트웨어 영역이고, 어설프게 하면 가공 못 할 형상이 나온다.
    삼각망만 밀어서 **비교용**으로 쓴다(cad_import/morph.py 참고).
    """
    import trimesh
    from cad_import import morph as mp

    entry = _cad_cache.get(cad_id)
    if entry is None:
        raise ValueError("CAD 가 만료됐습니다. 3D 파일을 다시 여세요.")

    display = entry.get("display_vertices")
    if display is None:
        raise ValueError("표시용 형상이 없습니다.")
    vertices = np.asarray(display, dtype=float)
    faces = np.asarray(entry["display_faces"])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=float)

    spots, values = [], []
    for point_id, value in corrections.items():
        where = positions.get(point_id)
        if where and len(where) == 3:
            spots.append([float(v) for v in where])
            values.append(float(value))
    if not spots:
        raise ValueError("3D 에 올라간 보정 포인트가 없습니다.")

    moved, shift, stats = mp.morph(
        vertices, faces, normals, spots, values, reach_ratio)

    # 보정량의 **부호가 곧 공정**이다 — + 용접(덧살), - CNC 가공.
    # 두 일은 작업자도 견적도 다르므로 물량을 따로 낸다.
    work = mp.work_volumes(vertices, faces, shift)

    return {
        "positions": [round(float(v), 4) for v in moved.ravel()],
        "shift": [round(float(v), 4) for v in shift],
        "stats": stats.to_dict(),
        "points": len(spots),
        "work": [w.to_dict() for w in work],
    }


async def cad_morph(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        corrections = body.get("corrections") or {}
        positions = body.get("positions") or {}
        result = await run_in_threadpool(
            cad_morph_for, str(body.get("cadId") or ""),
            {str(k): float(v) for k, v in corrections.items()},
            {str(k): v for k, v in positions.items()},
            float(body.get("reachRatio") or 0.04),
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


def open_morph_as_cad(cad_id: str, corrections: dict, positions: dict,
                      reach_ratio: float) -> dict[str, Any]:
    """보정 후 형상을 **새 CAD 로 등록**해 원본처럼 다루게 한다.

    [왜 이렇게 하나]
    화면에서만 밀어 보여 주면, 내보낸 파일이 정말 그 형상인지 알 수 없다.
    고친 형상을 실제로 만들어 다시 올리면 —
      · 원본과 같은 도구(단면·측정·주석·공정 구역)를 그대로 쓴다
      · 탭을 오가며 원본과 견준다
      · 화면에 보이는 것이 곧 내보내는 STL 이다(둘이 갈라지지 않는다)

    B-Rep 은 아니다. 원본 STEP 의 자유곡면을 고쳐 내보내는 것은 전용
    소프트웨어 영역이고, 면으로 쪼갠 STEP 은 실측 삼각형당 1.6ms ·
    2.4KB 라 이 부품(363,431 삼각형)이면 10분 · 852MB 다. 여기서 만드는
    것은 **삼각망**이고, 그것이 STL 로 나가는 것과 같은 형상이다.
    """
    import trimesh
    from cad_import import mesh_io

    entry = _cad_cache.get(cad_id)
    if entry is None:
        raise ValueError("CAD 가 만료됐습니다. 3D 파일을 다시 여세요.")

    result = cad_morph_for(cad_id, corrections, positions, reach_ratio)
    moved = np.asarray(result["positions"], dtype=float).reshape(-1, 3)
    faces = np.asarray(entry["display_faces"])
    # 화면용 정점은 원점을 옮겨 놓았다. 새 CAD 도 같은 자리에 서야
    # 원본과 겹쳐 볼 수 있으므로 되돌리지 않는다.
    mesh = trimesh.Trimesh(vertices=moved, faces=faces, process=False)

    name = f"{entry.get('name', 'part')}_보정후"
    # **다시 원점으로 옮기지 않는다.**
    #
    # 화면용 정점은 이미 원본을 옮겨 놓은 좌표계에 있다. to_web_mesh 가
    # 기본으로 자기 바운딩 상자 중심으로 또 옮기면 새 형상이 원본과
    # 어긋나 겹쳐 볼 수가 없다 — 실측에서 최대 이동이 2.886mm 로 나왔다
    # (보정 최대는 2.000mm 인데 상수 오프셋이 얹힌 것이다).
    web = mesh_io.to_web_mesh(mesh, name=name, source_format="morph",
                              recenter=False)
    web["holes"] = []
    web["planes"] = []
    web["counts"] = {"cylinders": 0, "holes": 0, "planes": 0}
    web["cadId"] = _cache_cad({
        "mesh": mesh,
        "offset": np.asarray(entry.get("offset"), dtype=float),
        "name": name,
        "display_vertices": np.asarray(
            web["positions"], dtype=float).reshape(-1, 3),
        "display_faces": np.asarray(web["indices"]).reshape(-1, 3),
    })
    web["work"] = result.get("work") or []
    # 이 탭이 무엇인지 적어 둔다. 안 적으면 "홀을 찾지 못했습니다" 가
    # 떠서 읽기 실패처럼 보인다 — 파생 형상이라 홀 정보가 없는 게 맞다.
    volumes = " · ".join(
        f"{'용접' if w['kind'] == 'weld' else '가공'} "
        f"{w['volume_mm3'] / 1000:.1f}cc"
        for w in (result.get("work") or []))
    web["note"] = (
        f"보정 후 형상입니다 — {entry.get('name', '원본')} 을(를) "
        f"보정 포인트 {result.get('points', 0)}개로 민 삼각망입니다. "
        f"조립 홀·기준면은 원본 탭에서 봅니다"
        + (f" · {volumes}" if volumes else "") + ".")
    return web


async def cad_morph_open(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await run_in_threadpool(
            open_morph_as_cad, str(body.get("cadId") or ""),
            {str(k): float(v) for k, v in (body.get("corrections") or {}).items()},
            {str(k): v for k, v in (body.get("positions") or {}).items()},
            float(body.get("reachRatio") or 0.04),
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def cad_morph_stl(request: Request) -> Response:
    """보정 후 형상을 STL 로 내보낸다.

    `part` 로 무엇을 담을지 고른다 —
      "after"(기본)  보정 후 형상 전체
      "weld"         살을 붙일 자리만 (용접 지시용)
      "cut"          살을 깎을 자리만 (CNC 가공 지시용)

    현장에서는 전체 형상보다 **자기 공정 자리만** 있는 편이 낫다.
    용접공에게 깎을 자리를 같이 주면 헷갈린다.
    """
    try:
        import trimesh
        from cad_import import morph as mp
        body = await request.json()
        want = str(body.get("part") or "after")
        result = await run_in_threadpool(
            cad_morph_for, str(body.get("cadId") or ""),
            {str(k): float(v) for k, v in (body.get("corrections") or {}).items()},
            {str(k): v for k, v in (body.get("positions") or {}).items()},
            float(body.get("reachRatio") or 0.04),
        )
        entry = _cad_cache.get(str(body.get("cadId") or "")) or {}
        moved = np.asarray(result["positions"], dtype=float).reshape(-1, 3)
        faces = np.asarray(entry["display_faces"])
        tail = "보정후"
        if want in ("weld", "cut"):
            split = mp.split_by_process(faces, result["shift"])
            faces = split[want]
            if not len(faces):
                return JSONResponse(
                    {"error": "그 공정에 해당하는 자리가 없습니다."},
                    status_code=404)
            tail = "덧살(용접)" if want == "weld" else "깎기(가공)"
        mesh = trimesh.Trimesh(vertices=moved, faces=faces, process=False)
        payload = mesh.export(file_type="stl")
        name = f"{entry.get('name', 'part')}_{tail}.stl"
        quoted = urllib.parse.quote(name)
        return Response(payload, media_type="model/stl", headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quoted}"})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)



def cad_sections_for(cad_id: str, notes: str, side: str) -> dict[str, Any]:
    """보정시트가 적어 둔 단면 위치(H:300 · T:1700)로 제로라인을 계산한다.

    다른 제로라인은 전부 추정이다 — 색을 읽거나 실루엣을 맞춘다. 이건
    아니다. 시트가 준 숫자로 CAD 를 자르기만 하므로 오차가 없다.
    (section_zero.py 에 축을 어떻게 확정했는지 적어 뒀다.)

    좌표는 화면용 메시와 같은 자리로 옮겨서 준다 — to_web_mesh 가 정점을
    원점으로 당겨 놨기 때문에 그만큼 빼지 않으면 형상에서 멀리 뜬다.
    """
    from zero_line_detection.section_zero import (
        parse_notes, zero_lines_from_notes,
    )

    entry = _cad_cache.get(cad_id)
    if entry is None:
        raise ValueError("CAD 가 만료됐습니다. 3D 파일을 다시 여세요.")

    parsed = parse_notes(notes)
    if not parsed:
        raise ValueError(
            "단면 표기를 찾지 못했습니다. 시트에 적힌 대로 "
            "'H : 300' 이나 'T : 1700' 처럼 넣어 주세요.")

    lines = zero_lines_from_notes(entry["mesh"], parsed, side=side)
    # offset 은 numpy 배열일 수 있다 — `or` 로 기본값을 주면
    # "truth value of an array is ambiguous" 로 터진다.
    raw = entry.get("offset")
    offset = [0.0, 0.0, 0.0] if raw is None else [float(v) for v in raw]
    shifted = []
    for line in lines:
        moved = [[[p[0] - offset[0], p[1] - offset[1], p[2] - offset[2]]
                  for p in poly] for poly in line.polylines]
        item = line.to_dict()
        item["polylines"] = moved
        shifted.append(item)

    return {"notes": [f"{k}:{v:g}" for k, v in parsed],
            "sections": shifted,
            "unmatched": [f"{k}:{v:g}" for k, v in parsed
                          if not any(s["label"] == f"{k}:{v:g}" for s in shifted)]}


async def cad_sections(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await run_in_threadpool(
            cad_sections_for,
            str(body.get("cadId") or ""),
            str(body.get("notes") or ""),
            str(body.get("side") or "both"),
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def cad_overlay(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await run_in_threadpool(
            cad_overlay_for,
            str(body.get("cadId") or ""),
            str(body.get("analysisId") or ""),
            body.get("zeroEdits") or None,
            body.get("fitAdjust") or None,
        )
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


async def cad(request: Request) -> JSONResponse:
    """3D 파일을 읽어 형상 + 조립 홀 + 기준면을 준다."""
    try:
        form = await request.form(max_files=1, max_fields=4,
                                  max_part_size=MAX_UPLOAD_BYTES)
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "3D 파일이 필요합니다."},
                                status_code=400)
        payload = await upload.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"error": f"파일이 너무 큽니다 ({len(payload) / 1024 / 1024:.0f}MB). "
                          f"최대 {MAX_UPLOAD_BYTES // 1024 // 1024}MB"},
                status_code=413)
        result = await run_in_threadpool(
            load_cad_payload, payload,
            getattr(upload, "filename", "part.step"))
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


# server.py 가 자기 라우트 목록에 이어 붙인다.
ROUTES = [
    Route("/api/cad", cad, methods=["POST"]),
    Route("/api/cad-overlay", cad_overlay, methods=["POST"]),
    Route("/api/cad-sections", cad_sections, methods=["POST"]),
    Route("/api/cad-morph", cad_morph, methods=["POST"]),
    Route("/api/cad-morph-open", cad_morph_open, methods=["POST"]),
    Route("/api/cad-morph-stl", cad_morph_stl, methods=["POST"]),
]

__all__ = ["ROUTES", "MAX_UPLOAD_BYTES", "load_cad_payload",
           "cad_overlay_for", "cad_morph_for", "open_morph_as_cad",
           "cad_sections_for", "remember_analysis", "reset_caches"]
