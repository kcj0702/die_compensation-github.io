"""제로라인 후보 지점에 학습용 라벨을 붙인다.

[목적]
my_lab 파이프라인이 뽑은 0포인트 후보들 중 어느 것이 진짜 제로라인/존인지
학습하려면 (특징, 정답) 쌍이 필요하다. 정답은 보정시트에서 자동으로
읽고(sheet_reference.py), 특징은 오직 스캔 실측값에서만 뽑는다 — 시트
좌표를 특징으로 쓰면 학습이 아니라 베끼기가 된다. 시트 좌표는 여기서
"정답(label)"으로만 쓰이고, 모델 입력(feature)에는 절대 들어가지 않는다.

[왜 픽셀 전체가 아니라 후보 단위인가]
실제로 걸러야 하는 문제는 "이 후보가 진짜 제로라인인가 노이즈인가"다
(현업 확인: JD_64는 0포인트가 너무 많이 찍힘). 그래서 my_lab이 이미 뽑아둔
후보(부호 전환 지점) 각각에 라벨을 붙인다.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from zero_line_detection.sheet_reference import (
    extract_sheet_zero_areas, extract_sheet_zero_line,
)
from zero_line_detection.zero_line import ZeroLineConfig, detect_zero_line
from zero_line_detection.zero_points import load_zero_points

HERE = Path(__file__).resolve().parent

# 원본 스캔·보정시트 이미지는 이 저장소가 아니라 로컬 작업 폴더에 있다
# (용량 문제로 커밋 안 함). 필요하면 --data-dir 로 다른 위치를 지정한다.
DEFAULT_DATA_DIR = Path(
    r"C:\Users\KDT033\Desktop\0line\die_compensation-github.io\data\intermediate"
)
ZERO_POINTS_DIR = HERE / "zero_points_data"
DATASET_DIR = HERE / "ml_data"

# 시트 좌표는 절대 여기 들어가지 않는다 — 전부 스캔 실측값에서만 계산된다.
FEATURE_NAMES = [
    "abs_dev_s5", "abs_dev_s15", "abs_dev_s40",
    "grad_s5", "grad_s15", "grad_s40",
    "dist_sign_boundary", "dist_edge", "local_std",
    "rule_strength",
]


@dataclass
class CaseConfig:
    part_no: str
    scan_name: str
    sheet_name: str
    config: ZeroLineConfig


CASES = [
    CaseConfig("64XX2", "JD_64XX2-DR000 3D 스캔.png", "JD_64XX2-DR000 보정시트.png",
               ZeroLineConfig(vmin=-1.5, vmax=2.0)),
    CaseConfig("67XX6", "JD_67XX6-DR000 3D 스캔.png", "JD_67XX6-DR000 보정시트.png",
               ZeroLineConfig()),
    CaseConfig("71XX2", "JD_71XX2-DR000 3D 스캔.png", "JD_71XX2-DR000 보정 시트.png",
               ZeroLineConfig()),
]


def _imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def build_feature_maps(values: np.ndarray, part_mask: np.ndarray) -> dict:
    """스캔 실측값만으로 계산 가능한 특징 맵. 시트 정보는 일절 쓰지 않는다."""
    feats: dict = {}
    for sigma in (5, 15, 40):
        smooth = cv2.GaussianBlur(values, (0, 0), sigma)
        feats[f"abs_dev_s{sigma}"] = np.abs(smooth)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=5)
        feats[f"grad_s{sigma}"] = np.hypot(gx, gy)

    smooth15 = cv2.GaussianBlur(values, (0, 0), 15)
    pos = ((smooth15 > 0) & part_mask).astype(np.uint8)
    neg = ((smooth15 < 0) & part_mask).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    crossing = (cv2.dilate(pos, k) > 0) & (cv2.dilate(neg, k) > 0) & part_mask
    feats["dist_sign_boundary"] = cv2.distanceTransform(
        (~crossing).astype(np.uint8), cv2.DIST_L2, 3)
    feats["dist_edge"] = cv2.distanceTransform(
        part_mask.astype(np.uint8), cv2.DIST_L2, 3)

    mean = cv2.blur(values, (31, 31))
    sq = cv2.blur(values * values, (31, 31))
    feats["local_std"] = np.sqrt(np.maximum(sq - mean * mean, 0))
    return feats


def _sample(maps: dict, x: float, y: float, shape: tuple) -> dict:
    h, w = shape
    px = int(np.clip(round(x), 0, w - 1))
    py = int(np.clip(round(y), 0, h - 1))
    return {name: float(field[py, px]) for name, field in maps.items()}


def ground_truth_mask(
    sheet_path: Path, part_mask: np.ndarray, part_no: str,
    values: np.ndarray, tolerance_px: int = 45,
):
    """보정시트에서 정답 위치를 읽어 허용오차만큼 부풀린 마스크로 만든다.

    이 마스크는 오직 라벨(정답)을 매기는 데만 쓰인다 — 특징으로는 안 쓴다.

    tolerance_px=45 는 실측으로 고른 값이다. 후보-정답 거리를 재보면
    뚜렷이 두 무리로 갈린다 — 진짜 근처(대개 5~35px)와 완전히 딴 곳
    (80px~수백px). 3부품 모두에서 그 경계가 40~60px 사이였다.
    """
    zero_mask = np.zeros(part_mask.shape, np.uint8)
    try:
        ref = extract_sheet_zero_line(sheet_path, part_mask, part_no, values=values)
        pts = np.array(ref.points, np.int32).reshape(-1, 1, 2)
        cv2.polylines(zero_mask, [pts], False, 1, 3)
        kind = "line"
    except ValueError:
        ref = extract_sheet_zero_areas(sheet_path, part_mask, part_no, values=values)
        for contour in ref.contours:
            cv2.fillPoly(zero_mask, [np.array(contour, np.int32)], 1)
        kind = "areas"
    k = np.ones((tolerance_px * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(zero_mask, k) > 0
    return dilated, kind


def build_dataset(
    cases: list = CASES, data_dir: Path = DEFAULT_DATA_DIR, tolerance_px: int = 45,
) -> list:
    """부품별로 my_lab 후보에 (특징, 정답) 라벨을 붙여 한 데이터셋으로 합친다."""
    rows: list = []
    for case in cases:
        scan_path = data_dir / case.scan_name
        sheet_path = data_dir / case.sheet_name
        points_path = ZERO_POINTS_DIR / f"{case.part_no}.json"
        if not (scan_path.exists() and sheet_path.exists() and points_path.exists()):
            print(f"[스킵] {case.part_no}: 파일이 없습니다 "
                  f"(scan={scan_path.exists()}, sheet={sheet_path.exists()}, "
                  f"points={points_path.exists()})")
            continue

        scan = _imread(scan_path)
        output = detect_zero_line(cv2.cvtColor(scan, cv2.COLOR_BGR2RGB), case.config)
        values, part_mask = output.values, output.part_mask

        truth_mask, kind = ground_truth_mask(
            sheet_path, part_mask, case.part_no, values, tolerance_px)
        feature_maps = build_feature_maps(values, part_mask)
        candidates = load_zero_points(points_path)

        h, w = values.shape
        n_pos = 0
        for point in candidates:
            features = _sample(feature_maps, point.x, point.y, values.shape)
            features["rule_strength"] = float(point.strength)
            px = int(np.clip(round(point.x), 0, w - 1))
            py = int(np.clip(round(point.y), 0, h - 1))
            label = int(truth_mask[py, px])
            n_pos += label
            rows.append({
                "part_no": case.part_no, "loop": point.loop,
                "x": point.x, "y": point.y, "point_kind": point.kind,
                "label": label, **features,
            })
        print(f"[{case.part_no}] 정답 종류={kind} 후보 {len(candidates)}개 "
              f"중 양성 {n_pos}개")
    return rows


def save_dataset(rows: list, out_dir: Path = DATASET_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "candidate_dataset.csv"
    fieldnames = ["part_no", "loop", "x", "y", "point_kind", "label", *FEATURE_NAMES]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="제로라인 후보 학습 데이터셋 생성")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tolerance-px", type=int, default=45)
    args = parser.parse_args()

    rows = build_dataset(CASES, args.data_dir, args.tolerance_px)
    if not rows:
        print("생성된 행이 없습니다.")
        return 1
    path = save_dataset(rows)
    n_pos = sum(r["label"] for r in rows)
    print(f"\n총 {len(rows)}개 후보 (양성 {n_pos}, 음성 {len(rows) - n_pos}) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
