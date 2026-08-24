"""0-Line 영역을 다각형으로 정리한다.

[왜 필요한가]
픽셀 마스크의 경계는 톱니처럼 너덜너덜하다. JD_71XX2 의 가장 큰 영역은
면적 47,574px 에 둘레가 5,128px 였다. 같은 면적의 매끈한 도형이면 800px 남짓이면
충분하니, 둘레의 대부분이 1픽셀짜리 요철인 셈이다. 그대로 그리면 화면이
지저분해서 어디가 0 영역인지 눈에 안 들어온다.

[어떻게 하는가]
    1. 닫기 연산으로 잔구멍과 톱니를 메운다
    2. 열기 연산으로 삐져나온 실오라기를 떼어낸다
    3. 외곽선과 구멍을 함께 찾는다 (RETR_CCOMP)
    4. Douglas-Peucker 로 꼭짓점을 줄인다

[구멍을 살리는 이유]
0 영역은 가늘고 긴 띠 모양인 경우가 많다. 외곽선만 따서 채우면 띠가 감싸는
안쪽 빈 공간까지 0 영역으로 칠해져 실제와 크게 달라진다. 구멍을 함께
살리면 원본 마스크 대비 IoU 가 0.80 에서 0.86 으로 올라간다.

[모양 보존 vs 단순함]
둘은 맞바꾸는 관계다. 꼭짓점을 990개에서 460개로 줄이면 IoU 가 0.86 에서
0.69 로 떨어진다. 용도에 따라 preset 으로 고른다.

    "accurate"  모양을 최대한 지킨다. 수치를 뽑아 쓸 때
    "balanced"  기본값. 화면 표시용
    "clean"     가장 단순하게. 발표 자료·외주 전달용
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


PRESETS = {
    "accurate": dict(close_ksize=3, open_ksize=3, min_area=150, epsilon_frac=0.001),
    "balanced": dict(close_ksize=5, open_ksize=3, min_area=250, epsilon_frac=0.002),
    "clean":    dict(close_ksize=9, open_ksize=5, min_area=400, epsilon_frac=0.006),
}


@dataclass
class ZeroPolygon:
    """0-Line 영역 하나. 바깥 테두리와 내부 구멍으로 이뤄진다."""

    polygon_id: int
    exterior: list                  # [[x, y], ...] 바깥 테두리
    holes: list                     # [[[x, y], ...], ...] 내부 구멍들
    area_px: float                  # 구멍을 뺀 실제 면적
    perimeter_px: float
    centroid_x: float
    centroid_y: float
    n_vertices: int                 # 구멍 포함 전체 꼭짓점 수

    def to_dict(self) -> dict:
        return asdict(self)


def polygonize(
    mask: np.ndarray,
    preset: str = "balanced",
    close_ksize: int | None = None,
    open_ksize: int | None = None,
    min_area: int | None = None,
    epsilon_frac: float | None = None,
) -> list:
    """0-Line 마스크 → 다각형 목록.

    preset 으로 기본값을 고르고, 개별 인자를 주면 그것이 우선한다.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset 은 {list(PRESETS)} 중 하나여야 합니다: {preset!r}")
    cfg = dict(PRESETS[preset])
    for key, val in (("close_ksize", close_ksize), ("open_ksize", open_ksize),
                     ("min_area", min_area), ("epsilon_frac", epsilon_frac)):
        if val is not None:
            cfg[key] = val

    binary = (mask > 0).astype(np.uint8)
    if cfg["close_ksize"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["close_ksize"],) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    if cfg["open_ksize"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["open_ksize"],) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    # RETR_CCOMP 는 바깥 테두리와 구멍을 2단계로 나눠 준다.
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []

    def simplify(contour):
        eps = cfg["epsilon_frac"] * cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, eps, True)
        return poly if len(poly) >= 3 else None

    # 바깥 테두리(부모 없음)를 먼저 모으고, 각 구멍을 부모에 붙인다.
    outers: dict = {}
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:                  # 구멍은 나중에
            continue
        if cv2.contourArea(contour) < cfg["min_area"]:
            continue
        poly = simplify(contour)
        if poly is not None:
            outers[i] = {"exterior": poly, "holes": []}

    for i, contour in enumerate(contours):
        parent = hierarchy[0][i][3]
        if parent == -1 or parent not in outers:
            continue
        if cv2.contourArea(contour) < cfg["min_area"] * 0.5:
            continue
        poly = simplify(contour)
        if poly is not None:
            outers[parent]["holes"].append(poly)

    polygons: list = []
    for item in outers.values():
        ext = item["exterior"]
        holes = item["holes"]
        area = float(cv2.contourArea(ext)) - sum(
            float(cv2.contourArea(h)) for h in holes
        )
        if area < cfg["min_area"]:
            continue
        moments = cv2.moments(ext)
        if moments["m00"] == 0:
            continue
        polygons.append(ZeroPolygon(
            polygon_id=0,
            exterior=ext.reshape(-1, 2).tolist(),
            holes=[h.reshape(-1, 2).tolist() for h in holes],
            area_px=round(area, 1),
            perimeter_px=round(float(cv2.arcLength(ext, True)), 1),
            centroid_x=round(moments["m10"] / moments["m00"], 1),
            centroid_y=round(moments["m01"] / moments["m00"], 1),
            n_vertices=len(ext) + sum(len(h) for h in holes),
        ))

    polygons.sort(key=lambda p: p.area_px, reverse=True)
    for i, p in enumerate(polygons, start=1):
        p.polygon_id = i
    return polygons


def polygons_to_mask(polygons: list, shape: tuple) -> np.ndarray:
    """다각형 목록 → 채워진 마스크 (구멍은 비운다)."""
    out = np.zeros(shape[:2], dtype=np.uint8)
    for p in polygons:
        cv2.fillPoly(out, [np.asarray(p.exterior, dtype=np.int32)], 255)
        for hole in p.holes:
            cv2.fillPoly(out, [np.asarray(hole, dtype=np.int32)], 0)
    return out


def draw_polygons(
    rgb: np.ndarray,
    polygons: list,
    fill_alpha: float = 0.32,
    fill_color: tuple = (255, 0, 200),
    edge_color: tuple = (15, 15, 15),
    edge_thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """원본 위에 다각형을 얹는다.

    면은 옅게 칠하고 테두리를 또렷하게 그려 경계가 눈에 들어오게 한다.
    """
    out = rgb.copy()

    if fill_alpha > 0 and polygons:
        filled = polygons_to_mask(polygons, rgb.shape) > 0
        tint = np.empty_like(out)
        tint[:] = fill_color
        out[filled] = (
            out[filled] * (1 - fill_alpha) + tint[filled] * fill_alpha
        ).astype(np.uint8)

    for p in polygons:
        cv2.polylines(out, [np.asarray(p.exterior, dtype=np.int32)], True,
                      edge_color, edge_thickness, cv2.LINE_AA)
        for hole in p.holes:
            cv2.polylines(out, [np.asarray(hole, dtype=np.int32)], True,
                          edge_color, max(edge_thickness - 1, 1), cv2.LINE_AA)

    if show_labels:
        for p in polygons:
            cx, cy = int(p.centroid_x), int(p.centroid_y)
            text = str(p.polygon_id)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (cx - tw // 2 - 4, cy - th - 4),
                          (cx + tw // 2 + 4, cy + 5), (255, 255, 255), -1)
            cv2.rectangle(out, (cx - tw // 2 - 4, cy - th - 4),
                          (cx + tw // 2 + 4, cy + 5), edge_color, 1)
            cv2.putText(out, text, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, edge_color, 2, cv2.LINE_AA)
    return out


__all__ = ["ZeroPolygon", "PRESETS", "polygonize", "polygons_to_mask", "draw_polygons"]
