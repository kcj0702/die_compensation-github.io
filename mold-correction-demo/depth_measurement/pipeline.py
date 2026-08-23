"""
금형 보정 데모 - 파이프라인
3D 스캔 편차 히트맵 PNG 한 장에서 형상 추출·정면도·형상 특징까지.
OpenCV 고전 기법만. 네트워크/모델 다운로드 없음.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------- 공통 유틸 (한글 경로 대응) ----------

def imread_u(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_u(path: str | Path, img: np.ndarray) -> bool:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def apply_flip(img: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    out = img
    if flip_h:
        out = cv2.flip(out, 1)
    if flip_v:
        out = cv2.flip(out, 0)
    return out


# ---------- 형상 추출 ----------

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """마스크 내부의 구멍을 flood fill 로 채운다 (홀이 흰색이라 배경으로 잘못
    빠진 픽셀을 되찾음)."""
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 255)
    outside = padded[1:-1, 1:-1]
    inside = cv2.bitwise_not(outside)
    return cv2.bitwise_or(mask, inside)


def get_body_mask(img_bgr: np.ndarray, white_thresh: int = 235) -> np.ndarray:
    """배경 흰색 제거 → 얇은 지시선 끊고 최대 연결요소만 살려 본체 마스크 생성.
    본체 안쪽 흰 영역(=홀)은 flood fill 로 다시 채워 넣음."""
    b, g, r = cv2.split(img_bgr)
    white = (b > white_thresh) & (g > white_thresh) & (r > white_thresh)
    non_white = (~white).astype(np.uint8) * 255

    # 얇은 지시선 끊기
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded = cv2.erode(non_white, k7, iterations=1)

    # 최대 연결요소만
    num, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if num <= 1:
        return np.zeros_like(non_white)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    biggest = (labels == largest).astype(np.uint8) * 255

    # 복원
    dilated = cv2.dilate(biggest, k7, iterations=1)
    body = cv2.bitwise_and(dilated, non_white)

    # 정리
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k5, iterations=2)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, k5, iterations=1)

    # 홀 채우기 (본체 안쪽 흰 영역 = 실제 홀 → 마스크에 포함)
    body = _fill_holes(body)
    return body


def _filter_boxes(mask: np.ndarray, color_name: str) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict[str, Any]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300 or area > 7000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (22 <= w <= 130 and 18 <= h <= 70):
            continue
        bbox_area = w * h
        fill = area / bbox_area if bbox_area else 0.0
        if fill < 0.65:
            continue
        out.append(
            dict(x=int(x), y=int(y), w=int(w), h=int(h),
                 cx=int(x + w / 2), cy=int(y + h / 2), color=color_name)
        )
    return out


def detect_label_boxes(img_bgr: np.ndarray, dilate: int = 5) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """빨강/민트/회색 라벨박스 검출. 사각형 필터를 통과한 것만 통합 마스크에 사각형으로 그림."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img_bgr)

    red_mask = ((r > 150) & (g < 95) & (b < 95)).astype(np.uint8) * 255
    mint_mask = ((h >= 40) & (h <= 95) & (s >= 15) & (s <= 115) & (v > 165)).astype(np.uint8) * 255
    gray_mask = ((s < 45) & (v > 150) & (v < 245)).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k3)
    mint_mask = cv2.morphologyEx(mint_mask, cv2.MORPH_CLOSE, k3)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, k3)

    boxes: list[dict[str, Any]] = []
    boxes += _filter_boxes(red_mask, "red")
    boxes += _filter_boxes(mint_mask, "mint")
    boxes += _filter_boxes(gray_mask, "gray")

    combined = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for bx in boxes:
        cv2.rectangle(combined, (bx["x"], bx["y"]),
                      (bx["x"] + bx["w"], bx["y"] + bx["h"]), 255, thickness=-1)

    if dilate > 0 and boxes:
        kd = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        combined = cv2.dilate(combined, kd, iterations=1)

    return combined, boxes


def clean_and_extract(img_bgr: np.ndarray, body: np.ndarray, boxes_mask: np.ndarray) -> dict[str, np.ndarray]:
    """본체 안쪽 라벨박스만 inpaint 하고, 부품만 흰 배경 위에 남긴 이미지도 반환."""
    boxes_in = cv2.bitwise_and(boxes_mask, body)
    if cv2.countNonZero(boxes_in) > 0:
        clean = cv2.inpaint(img_bgr, boxes_in, 6, cv2.INPAINT_TELEA)
    else:
        clean = img_bgr.copy()

    white_bg = np.full_like(clean, 255)
    body_3 = cv2.merge([body, body, body])
    part = np.where(body_3 > 0, clean, white_bg)
    return {"clean": clean, "part": part}


# ---------- 형상 특징 ----------

def to_grayscale_frontview(clean_bgr: np.ndarray) -> np.ndarray:
    """색 벗기고 음영만 남긴 정면도(3채널 BGR). 밝은 회색으로 나오도록 처리."""
    gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
    # 히트맵의 원색은 낮은 밝기라 그대로 회색 변환하면 어두움.
    # 히스토그램의 하위 2%~상위 98%를 [80..245] 범위로 스트레치해 밝은 회색으로.
    lo = float(np.percentile(gray, 2))
    hi = float(np.percentile(gray, 98))
    if hi <= lo:
        stretched = gray
    else:
        stretched = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0, 1)
        stretched = (stretched * (245 - 80) + 80).astype(np.uint8)
    stretched = cv2.GaussianBlur(stretched, (3, 3), 0)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def extract_relief(clean_bgr: np.ndarray, sigma: float = 21.0) -> np.ndarray:
    """하이패스로 비드·필릿·형상선 강조. 반환: uint8 3채널 BGR."""
    gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), sigma)
    hp = gray - low
    hp_norm = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 살짝 CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    hp_eq = clahe.apply(hp_norm)
    return cv2.cvtColor(hp_eq, cv2.COLOR_GRAY2BGR)


def _flat_regions(relief_gray: np.ndarray, body_mask: np.ndarray,
                  block: int = 24, std_thresh: float = 6.0) -> np.ndarray:
    """릴리프의 블록별 표준편차가 낮은 영역 = 평면.
    relief 기반이라 inpaint 자국이 평면으로 오검출되지 않는다."""
    h, w = relief_gray.shape
    flat = np.zeros((h, w), dtype=np.uint8)
    r32 = relief_gray.astype(np.float32)
    for y in range(0, h, block):
        for x in range(0, w, block):
            y2, x2 = min(y + block, h), min(x + block, w)
            patch = r32[y:y2, x:x2]
            mpatch = body_mask[y:y2, x:x2]
            if cv2.countNonZero(mpatch) < patch.size * 0.6:
                continue
            if float(patch.std()) < std_thresh:
                flat[y:y2, x:x2] = 255
    flat = cv2.bitwise_and(flat, body_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, k)
    return flat


def _classify_hole(contour: np.ndarray) -> tuple[str, dict[str, int]]:
    """컨투어를 원/사각/기타로 분류.
    - 원형도 = 4πA/P²  (1에 가까우면 원)
    - 사각도 = A / (bbox 면적)  + approxPolyDP 4각 여부
    반환: (kind, geom)  kind ∈ {"circle","rect","other"}
    """
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri <= 0 or area <= 0:
        return "other", {}
    circularity = 4.0 * np.pi * area / (peri * peri)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    x, y, w, h = cv2.boundingRect(contour)
    bbox_fill = area / float(w * h) if w * h else 0.0

    aspect = max(w, h) / max(1, min(w, h))
    # 원과 bbox 면적 비: 원이면 π·r²/(2r)² = π/4 ≈ 0.785
    # 정사각형이면 1.0. 이 차이로 원/사각 구분.
    circle_bbox_fill = area / (np.pi * radius * radius) if radius > 0 else 0
    # circle_bbox_fill: 실제면적 / minEnclosingCircle 면적
    #   원=1, 정사각=2/π≈0.637, 직사각=더 낮음

    # 원: 세 조건 모두 만족
    #  - 원형도 ≥ 0.85 (정사각형 0.785 초과)
    #  - bbox 종횡비 1:1 근접 (aspect < 1.30)
    #  - minEnclosingCircle 채움도 ≥ 0.75
    #    (원=0.85~0.95, 정사각형=2/π≈0.637 — 확실히 분리됨)
    if (circularity >= 0.85 and aspect < 1.30 and radius > 4
            and circle_bbox_fill >= 0.75):
        return "circle", {"cx": int(cx), "cy": int(cy), "r": int(radius)}
    # 사각: bbox 잘 채움 + 종횡비 5:1 이내
    if bbox_fill > 0.72 and aspect < 5.0:
        return "rect", {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    return "other", {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def detect_holes(img_bgr: np.ndarray, body_mask: np.ndarray,
                 white_thresh: int = 235, min_hole_r: int = 8) -> dict[str, list[dict[str, int]]]:
    """원본에서 '본체 안쪽 흰 영역' = 실제 구멍/창.
    컨투어별로 원/사각 분류.
    """
    b, g, r = cv2.split(img_bgr)
    # 임계값을 살짝 낮춰 라벨박스 자리의 옅은 회색 얼룩도 흡수
    soft = max(180, white_thresh - 55)
    white = (b > soft) & (g > soft) & (r > soft)
    white_u8 = (white.astype(np.uint8) * 255)
    # 본체 안쪽 흰 영역만
    holes = cv2.bitwise_and(white_u8, body_mask)
    # 얇은 노이즈 제거 후, close 로 창 내부 회색 얼룩(라벨 지운 자국) 흡수
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, k3)
    # close 크기를 min_hole_r 이하로 → 작은 홀도 살아남게
    close_sz = max(3, min(min_hole_r - 1, 7)) | 1  # 홀수화
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))
    holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, kc)

    min_area = np.pi * (min_hole_r ** 2)
    max_area = img_bgr.shape[0] * img_bgr.shape[1] * 0.20

    circles: list[dict[str, int]] = []
    rects: list[dict[str, int]] = []
    others: list[dict[str, int]] = []

    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        kind, geom = _classify_hole(c)
        if kind == "circle":
            if geom["r"] >= min_hole_r:
                circles.append(geom)
        elif kind == "rect":
            if geom["w"] >= min_hole_r * 2 and geom["h"] >= min_hole_r * 2:
                rects.append(geom)
        else:
            if geom.get("w", 0) >= min_hole_r * 2:
                others.append(geom)
    return {"circles": circles, "rects": rects, "others": others}


def detect_features(
    img_bgr: np.ndarray,
    clean_bgr: np.ndarray,
    relief_bgr: np.ndarray,
    body_mask: np.ndarray,
    min_hole_r: int = 8,
    white_thresh: int = 235,
) -> dict[str, Any]:
    """원/사각/기타 홀 + 라인 + 평면 검출.
    홀은 원본의 '본체 안쪽 흰 영역' 컨투어를 원/사각으로 분류.
    라인·평면은 릴리프 기반.
    """
    relief_gray = cv2.cvtColor(relief_bgr, cv2.COLOR_BGR2GRAY)
    h, w = relief_gray.shape

    # (a) 홀 (원/사각/기타)
    holes = detect_holes(img_bgr, body_mask,
                         white_thresh=white_thresh, min_hole_r=min_hole_r)
    circles_out = holes["circles"]
    rects_out = holes["rects"]
    others_out = holes["others"]
    hole_bboxes = (
        [(c["cx"] - c["r"], c["cy"] - c["r"], 2 * c["r"], 2 * c["r"]) for c in circles_out]
        + [(r_["x"], r_["y"], r_["w"], r_["h"]) for r_ in rects_out]
        + [(o["x"], o["y"], o["w"], o["h"]) for o in others_out]
    )

    # (b) 긴 엣지/비드 라인 — 홀 경계에서 파생된 라인은 제외
    edges = cv2.Canny(relief_gray, 60, 160)
    edges = cv2.bitwise_and(edges, body_mask)

    lines_raw: list[tuple[int, int, int, int, int, float]] = []
    # 최소 길이 상향 → 짧은 자잘한 직선 제거
    min_len = int(max(80, min(h, w) * 0.18))
    hl = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=130,
        minLineLength=min_len, maxLineGap=6,
    )
    if hl is not None:
        for x1, y1, x2, y2 in hl.reshape(-1, 4):
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            if not (0 <= mx < w and 0 <= my < h and body_mask[my, mx] > 0):
                continue
            # 홀 근처(테두리) 라인은 스킵
            near_hole = False
            for bx, by, bw, bh in hole_bboxes:
                if (bx - 6) <= mx <= (bx + bw + 6) and (by - 6) <= my <= (by + bh + 6):
                    near_hole = True
                    break
            if near_hole:
                continue
            length = int(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.arctan2(y2 - y1, x2 - x1))
            lines_raw.append((int(x1), int(y1), int(x2), int(y2), length, angle))

    lines_raw.sort(key=lambda t: t[4], reverse=True)
    lines_out: list[dict[str, int]] = []
    kept_meta: list[tuple[float, float, float]] = []
    for x1, y1, x2, y2, _length, angle in lines_raw:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dup = False
        for kcx, kcy, kang in kept_meta:
            da = abs(angle - kang)
            da = min(da, np.pi - da)
            # 각도 20도 이내이고 중심 60px 이내면 중복으로 간주 → 대표 라인만 유지
            if da < np.deg2rad(20) and np.hypot(cx - kcx, cy - kcy) < 60:
                dup = True
                break
        if dup:
            continue
        lines_out.append(dict(x1=x1, y1=y1, x2=x2, y2=y2))
        kept_meta.append((cx, cy, angle))
        # 상한 크게 낮춰 화면 가독성 확보
        if len(lines_out) >= 12:
            break

    # (c) 평면 영역 (릴리프 기반)
    flat_mask = _flat_regions(relief_gray, body_mask)

    return {
        "circles": circles_out,
        "rects": rects_out,
        "others": others_out,
        "lines": lines_out,
        "flat_mask": flat_mask,
    }


def render_feature_overlay(
    base_bgr: np.ndarray,
    features: dict[str, Any],
    show: dict[str, bool] | None = None,
) -> np.ndarray:
    """검출 결과를 색 오버레이로 그림. 색: 원=파랑, 사각=주황, 라인=녹색, 평면=회색 반투명."""
    if show is None:
        show = dict(circles=True, rects=True, lines=True, flat=True)
    out = base_bgr.copy()

    if show.get("flat", True):
        flat = features.get("flat_mask")
        if flat is not None and cv2.countNonZero(flat) > 0:
            overlay = out.copy()
            overlay[flat > 0] = (170, 170, 170)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    if show.get("lines", True):
        for ln in features.get("lines", []):
            cv2.line(out, (ln["x1"], ln["y1"]), (ln["x2"], ln["y2"]),
                     (60, 200, 60), 2, cv2.LINE_AA)

    if show.get("rects", True):
        for rc in features.get("rects", []):
            cv2.rectangle(out, (rc["x"], rc["y"]),
                          (rc["x"] + rc["w"], rc["y"] + rc["h"]),
                          (0, 140, 255), 2, cv2.LINE_AA)

    if show.get("others", True):
        for ot in features.get("others", []):
            cv2.rectangle(out, (ot["x"], ot["y"]),
                          (ot["x"] + ot["w"], ot["y"] + ot["h"]),
                          (200, 60, 200), 2, cv2.LINE_AA)  # 자홍 = 기타 홀

    if show.get("circles", True):
        for ci in features.get("circles", []):
            cv2.circle(out, (ci["cx"], ci["cy"]), ci["r"],
                       (230, 120, 40), 2, cv2.LINE_AA)
            cv2.circle(out, (ci["cx"], ci["cy"]), 2,
                       (230, 120, 40), -1, cv2.LINE_AA)

    return out


# ---------- 편차 (색 → mm) ----------

def detect_colorbar_bbox(img_bgr: np.ndarray, body_mask: np.ndarray,
                         side_margin_ratio: float = 0.20) -> tuple[int, int, int, int] | None:
    """이미지 좌/우 20% 안에서 세로로 긴 컬러바 영역 자동 검출.
    반환: (x1, y1, x2, y2) 또는 None.
    조건: 채도 높음(S>60) + 세로/가로 종횡비 > 3 + body 밖.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # 컬러바 후보: 채도 높음 + 밝음 + body 바깥
    cand = ((sat > 60) & (val > 60) & (body_mask == 0)).astype(np.uint8) * 255
    # 세로 연결
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        return None
    best = None
    best_score = 0.0
    xm = int(w * side_margin_ratio)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if hh < h * 0.30:
            continue
        if hh / max(1, ww) < 3.0:
            continue
        # 좌우 20% 안에 있어야 함
        if not (x < xm or (x + ww) > (w - xm)):
            continue
        score = float(hh) * (hh / max(1, ww))
        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(x + ww), int(y + hh))
    return best


def build_color_mm_lut(img_bgr: np.ndarray, bbox: tuple[int, int, int, int],
                       mm_min: float, mm_max: float,
                       n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """컬러바 세로 중앙선을 훑어 색→mm 대응표 생성.
    반환: (colors_bgr: (N,3) uint8, values_mm: (N,) float32).
    보통 컬러바는 위=max, 아래=min 이지만 반대일 수도 있어 mm_min/max 방향에 맞춰 매핑.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    ys = np.linspace(y1 + 2, y2 - 2, n_samples).astype(int)
    # 컬러바 폭의 중앙 5픽셀 평균으로 노이즈 제거
    xs = [max(x1, cx - 2), cx, min(x2 - 1, cx + 2)]
    cols = img_bgr[ys[:, None], xs].mean(axis=1)  # (N, 3)
    cols = np.clip(cols, 0, 255).astype(np.uint8)
    # 위 픽셀 = mm_max (뷰어 관례), 아래 = mm_min
    values = np.linspace(mm_max, mm_min, n_samples).astype(np.float32)
    return cols, values


def image_to_deviation_mm(img_bgr: np.ndarray, body_mask: np.ndarray,
                          colors_lut: np.ndarray, values_lut: np.ndarray) -> np.ndarray:
    """모든 픽셀을 컬러바 LUT 로 매칭해 mm 값 배열 반환.
    body 밖은 NaN.
    벡터화: L*a*b 공간 거리로 최근접 매칭.
    """
    h, w = img_bgr.shape[:2]
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lut_bgr = colors_lut.reshape(-1, 1, 3)
    lut_lab = cv2.cvtColor(lut_bgr, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)

    # 픽셀 x LUT 거리 (메모리 절약: 청크 처리)
    flat = lab_img.reshape(-1, 3)
    mm = np.full(flat.shape[0], np.nan, dtype=np.float32)
    body_flat = body_mask.reshape(-1) > 0
    idx_all = np.where(body_flat)[0]

    chunk = 20000
    for start in range(0, len(idx_all), chunk):
        idx = idx_all[start:start + chunk]
        # (chunk, 1, 3) - (1, N, 3) -> (chunk, N)
        d2 = ((flat[idx][:, None, :] - lut_lab[None, :, :]) ** 2).sum(-1)
        nearest = np.argmin(d2, axis=1)
        mm[idx] = values_lut[nearest]
    return mm.reshape(h, w)


# ---------- 편차 × 형상특징 결합 ----------

FEATURE_KINDS = ("hole_circle", "hole_rect", "hole_other",
                 "near_edge", "bead", "flat", "surface")

# 형상별 보정 계수 (참고값 — 실제 값은 실무 데이터로 보정 필요)
# 보정치 = -편차 × 계수.  이번 데모 범위에서는 산출하지 않고 참고로만 표기.
FEATURE_COEFF_HINT = {
    "flat": 0.50,        # 평면: 스프링백 큼
    "bead": 0.30,        # 비드: 강성 높아 스프링백 작음
    "near_edge": 0.35,   # 형상선 근처
    "surface": 0.45,     # 일반 곡면
    "hole_circle": None,  # 홀 내부는 보정 대상 아님
    "hole_rect": None,
    "hole_other": None,
}


def _point_in_bbox(x: int, y: int, bx: int, by: int, bw: int, bh: int, pad: int = 0) -> bool:
    return (bx - pad) <= x <= (bx + bw + pad) and (by - pad) <= y <= (by + bh + pad)


def classify_point_feature(x: int, y: int, features: dict[str, Any],
                           relief_gray: np.ndarray, body_mask: np.ndarray,
                           near_px: int = 14) -> str:
    """한 지점의 형상 특징 분류.

    우선순위: 홀 내부 > 형상선(엣지) 근처 > 비드(릴리프 강함) > 평면 > 일반 곡면
    """
    h, w = relief_gray.shape
    if not (0 <= x < w and 0 <= y < h) or body_mask[y, x] == 0:
        return "unknown"

    # 1) 홀 내부
    for c in features.get("circles", []):
        if np.hypot(x - c["cx"], y - c["cy"]) <= c["r"]:
            return "hole_circle"
    for r_ in features.get("rects", []):
        if _point_in_bbox(x, y, r_["x"], r_["y"], r_["w"], r_["h"]):
            return "hole_rect"
    for o in features.get("others", []):
        if _point_in_bbox(x, y, o["x"], o["y"], o["w"], o["h"]):
            return "hole_other"

    # 2) 검출된 형상선(엣지) 근처
    for ln in features.get("lines", []):
        x1, y1, x2, y2 = ln["x1"], ln["y1"], ln["x2"], ln["y2"]
        vx, vy = x2 - x1, y2 - y1
        seg2 = vx * vx + vy * vy
        if seg2 == 0:
            continue
        t = max(0.0, min(1.0, ((x - x1) * vx + (y - y1) * vy) / seg2))
        px, py = x1 + t * vx, y1 + t * vy
        if np.hypot(x - px, y - py) <= near_px:
            return "near_edge"

    # 3) 릴리프 국소 강도로 비드/평면/곡면 구분
    r = 6
    y0, y1_ = max(0, y - r), min(h, y + r + 1)
    x0, x1_ = max(0, x - r), min(w, x + r + 1)
    patch = relief_gray[y0:y1_, x0:x1_].astype(np.float32)
    if patch.size == 0:
        return "surface"
    local_std = float(patch.std())
    flat_mask = features.get("flat_mask")
    if flat_mask is not None and flat_mask[y, x] > 0:
        return "flat"
    if local_std > 26.0:
        return "bead"
    if local_std < 8.0:
        return "flat"
    return "surface"


def build_point_table(mm_arr: np.ndarray, features: dict[str, Any],
                      relief_bgr: np.ndarray, body_mask: np.ndarray,
                      step: int = 24, tolerance_mm: float = 0.5) -> list[dict[str, Any]]:
    """격자 샘플링으로 (좌표, 편차, 형상특징) 통합 레코드 생성.

    보정치 산출 직전 단계의 산출물. 계수는 참고값만 붙이고 곱하지 않는다.
    """
    relief_gray = cv2.cvtColor(relief_bgr, cv2.COLOR_BGR2GRAY)
    h, w = body_mask.shape
    rows: list[dict[str, Any]] = []
    pid = 0
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if body_mask[y, x] == 0:
                continue
            mm = mm_arr[y, x] if mm_arr is not None else np.nan
            if mm_arr is not None and np.isnan(mm):
                continue
            kind = classify_point_feature(x, y, features, relief_gray, body_mask)
            if kind in ("hole_circle", "hole_rect", "hole_other", "unknown"):
                continue  # 홀 내부는 보정 대상 아님
            pid += 1
            val = float(mm) if mm_arr is not None else None
            rows.append({
                "point_id": f"P{pid:04d}",
                "x_px": int(x),
                "y_px": int(y),
                "x_norm": round(x / w, 4),
                "y_norm": round(y / h, 4),
                "deviation_mm": None if val is None else round(val, 3),
                "feature": kind,
                "in_tolerance": None if val is None else bool(abs(val) <= tolerance_mm),
                "coeff_hint": FEATURE_COEFF_HINT.get(kind),
            })
    return rows


def summarize_features(features: dict[str, Any], body_mask: np.ndarray) -> dict[str, Any]:
    """검출 결과 요약 통계."""
    flat = features.get("flat_mask")
    body_px = int(cv2.countNonZero(body_mask))
    flat_px = int(cv2.countNonZero(flat)) if flat is not None else 0
    return {
        "n_circles": len(features.get("circles", [])),
        "n_rects": len(features.get("rects", [])),
        "n_others": len(features.get("others", [])),
        "n_lines": len(features.get("lines", [])),
        "body_px": body_px,
        "flat_ratio": round(flat_px / max(1, body_px) * 100, 1),
    }


# ---------- 파이프라인 오케스트레이션 ----------

def run_pipeline(
    img_bgr: np.ndarray,
    white_thresh: int = 235,
    sigma: float = 21.0,
    min_hole_r: int = 8,
    show: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """단계별 결과 이미지 + 처리 시간(ms) 반환."""
    timings: dict[str, float] = {}

    t = time.perf_counter()
    body = get_body_mask(img_bgr, white_thresh=white_thresh)
    timings["body_mask"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    boxes_mask, boxes = detect_label_boxes(img_bgr)
    timings["label_boxes"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    ext = clean_and_extract(img_bgr, body, boxes_mask)
    clean, part = ext["clean"], ext["part"]
    timings["inpaint_extract"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    frontview = to_grayscale_frontview(clean)
    timings["frontview"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    relief = extract_relief(clean, sigma=sigma)
    timings["relief"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    features = detect_features(img_bgr, clean, relief, body,
                               min_hole_r=min_hole_r, white_thresh=white_thresh)
    timings["detect_features"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    # 배경을 part(흰바탕)로 그려야 바깥 라벨박스가 안 보임
    overlay = render_feature_overlay(part, features, show=show)
    timings["render_overlay"] = (time.perf_counter() - t) * 1000

    return {
        "body_mask": body,
        "boxes": boxes,
        "part": part,
        "clean": clean,
        "frontview": frontview,
        "relief": relief,
        "features": features,
        "overlay": overlay,
        "timings_ms": timings,
    }


# ---------- 독립 실행 ----------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python pipeline.py <input.png> [out_dir]")
        sys.exit(1)
    src = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = imread_u(src)
    if img is None:
        print(f"cannot read: {src}")
        sys.exit(2)

    res = run_pipeline(img)
    imwrite_u(str(out_dir / "01_original.png"), img)
    imwrite_u(str(out_dir / "02_part.png"), res["part"])
    imwrite_u(str(out_dir / "03_frontview.png"), res["frontview"])
    imwrite_u(str(out_dir / "04_relief.png"), res["relief"])
    imwrite_u(str(out_dir / "05_overlay.png"), res["overlay"])

    print("timings (ms):")
    for k, v in res["timings_ms"].items():
        print(f"  {k:20s} {v:8.1f}")
    print(f"circles={len(res['features']['circles'])}, "
          f"rects={len(res['features']['rects'])}, "
          f"others={len(res['features']['others'])}, "
          f"lines={len(res['features']['lines'])}")
    print(f"saved to: {out_dir.resolve()}")
