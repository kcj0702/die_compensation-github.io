"""보정시트 콜아웃 값과 스캔 실측 편차값의 차이로 학습 데이터를 만든다.

[핵심 전제]
`sheet_reference.py` 에 이미 경고가 있다 — **보정치는 스캔 편차의 부호를
뒤집은 값**이다(가공 공정이면 편차가 +일 때 -로 깎는 식). 그래서 그냥
`correction - scan_value` 를 신호로 쓰면 안 된다. 부호반전이 기본이고,
그 기본에서 얼마나 벗어났는지가 진짜 학습 신호다:

    raw_diff = correction - scan_value          (참고용, 해석 주의)
    residual = correction - (-scan_value)
             = correction + scan_value           (부호반전만 했으면 0 근처)

residual 이 0에서 크게 벗어난 지점이 "측정값 그대로가 아니라 추가 판단이
들어간 자리"다 — 안전마진, RPS 대비 위치, 형상 강성 같은 것들.

[짝 확인이 먼저다]
같은 파일에 이미 있는 경고: JD_64XX2 스캔·시트가 다른 회차일 수 있다는
확인(2026-08-24)이 있다. `check_pairing()` 결과를 데이터셋과 함께
저장한다 — 짝이 안 맞는 부품의 diff 숫자를 신호로 오인하면 안 된다.

[매칭 방법]
1. 시트에서 콜아웃(값+점)을 읽는다 (`sheet_values.py`, VLM 판독).
2. 등록된 zero_line_library.json 의 sheet_bbox/scan_bbox/mirrored 로
   콜아웃 점을 스캔 좌표로 투영한다 — `sheet_reference.py` 가 제로라인을
   등록할 때 쓴 것과 같은 변환이라 일관적이다.
3. 투영된 점에 가장 가까운 스캔 측정점(라벨 판독값)을 찾는다
   (`/api/analyze` 가 돌려주는 P-01..P-NN, 최대 매칭거리 이내).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # Windows 콘솔 기본 cp949 로는 이 스크립트가 쓰는 이모지 아닌 특수문자
    # (em dash 등)조차 깨진다. 표준출력을 UTF-8로 강제한다.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
API_BASE = "http://127.0.0.1:8000"
LIBRARY_PATH = HERE / "zero_line_library.json"
DATASET_DIR = HERE / "ml_data"
MAX_MATCH_PX = 60.0
REQUEST_TIMEOUT = 900


def project_sheet_point(
    point: tuple, sheet_bbox: list, scan_bbox: list, mirrored: bool,
) -> tuple:
    """시트 픽셀 좌표를 스캔 픽셀 좌표로 옮긴다.

    `sheet_reference.extract_sheet_zero_line` 이 제로라인을 등록할 때 쓴
    것과 같은 bbox 선형 변환이다 — 같은 부품이면 같은 변환을 써야
    좌표계가 어긋나지 않는다.
    """
    sx0, sy0, sx1, sy1 = sheet_bbox
    kx0, ky0, kx1, ky1 = scan_bbox
    x, y = point
    nx = (x - sx0) / max(sx1 - sx0, 1)
    if mirrored:
        nx = 1.0 - nx
    ny = (y - sy0) / max(sy1 - sy0, 1)
    px = nx * (kx1 - kx0) + kx0
    py = ny * (ky1 - ky0) + ky0
    return px, py


def fetch_scan_points(scan_path: Path) -> list:
    """`/api/analyze` 로 스캔의 실측 편차 라벨(P-01..)을 읽는다."""
    with open(scan_path, "rb") as fh:
        resp = requests.post(
            f"{API_BASE}/api/analyze",
            files={"file": (scan_path.name, fh, "image/png")},
            timeout=REQUEST_TIMEOUT,
        )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"scan 분석 실패: {data['error']}")
    return data.get("points", [])


def fetch_sheet_callouts(sheet_path: Path) -> list:
    """`/api/sheet-values` 로 보정시트의 보정치 콜아웃을 읽는다."""
    with open(sheet_path, "rb") as fh:
        resp = requests.post(
            f"{API_BASE}/api/sheet-values",
            files={"file": (sheet_path.name, fh, "image/png")},
            timeout=REQUEST_TIMEOUT,
        )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"sheet 판독 실패: {data['error']}")
    return data.get("callouts", [])


def check_pairing(sheet_values: list, scan_values: list) -> dict:
    """스캔과 시트가 같은 회차인지 부호 분포로 가늠한다 (sheet_reference.py 와 동일 로직)."""
    s = np.array([v for v in sheet_values if v is not None], dtype=float)
    k = np.array([v for v in scan_values if v is not None], dtype=float)
    if not len(s) or not len(k):
        return {"ok": False, "reason": "값이 없습니다."}
    flipped = -k
    sheet_pos = float((s > 0).mean())
    flipped_pos = float((flipped > 0).mean())
    consistent = abs(sheet_pos - flipped_pos) < 0.35
    return {
        "ok": bool(consistent),
        "sheetPositiveRatio": round(sheet_pos, 3),
        "scanFlippedPositiveRatio": round(flipped_pos, 3),
        "reason": (
            "부호 분포가 비슷합니다 (짝이 맞을 가능성)." if consistent
            else "부호 분포가 반대입니다 — 다른 회차이거나 기준면이 다를 수 있습니다."
        ),
    }


def match_and_diff(
    part_no: str,
    scan_points: list,
    callouts: list,
    sheet_bbox: list,
    scan_bbox: list,
    mirrored: bool,
    max_match_px: float = MAX_MATCH_PX,
) -> list:
    """콜아웃마다 가장 가까운 스캔 측정점을 찾아 diff 행을 만든다."""
    if not scan_points:
        return []
    scan_xy = np.array([[p["xPx"], p["yPx"]] for p in scan_points], dtype=float)

    rows = []
    for callout in callouts:
        projected = project_sheet_point(
            callout["point"], sheet_bbox, scan_bbox, mirrored)
        dists = np.hypot(*(scan_xy - np.array(projected)).T)
        idx = int(np.argmin(dists))
        distance = float(dists[idx])
        if distance > max_match_px:
            continue
        scan_point = scan_points[idx]
        correction = float(callout["value"])
        scan_value = float(scan_point["value"])
        rows.append({
            "part_no": part_no,
            "scan_point_id": scan_point["id"],
            "scan_x": scan_point["xPx"], "scan_y": scan_point["yPx"],
            "scan_value": scan_value,
            "sheet_x": callout["point"][0], "sheet_y": callout["point"][1],
            "projected_x": round(projected[0], 1), "projected_y": round(projected[1], 1),
            "correction_value": correction,
            "match_distance_px": round(distance, 1),
            "raw_diff": round(correction - scan_value, 3),
            "residual": round(correction + scan_value, 3),
        })
    return rows


def build_dataset(cases: list) -> tuple:
    """부품 목록을 돌며 diff 행과 부품별 짝 확인 결과를 모은다."""
    library = json.loads(LIBRARY_PATH.read_text(encoding="utf-8")) if LIBRARY_PATH.is_file() else {}

    all_rows: list = []
    pairing_report: dict[str, Any] = {}

    for case in cases:
        part_no = case["part_no"]
        entry = library.get(part_no)
        if entry is None:
            print(f"[스킵] {part_no}: zero_line_library.json 에 등록 안 됨 "
                  f"(register_sheet 먼저 실행)")
            continue

        print(f"\n[{part_no}] 스캔 분석 중...")
        scan_points = fetch_scan_points(case["scan_path"])
        print(f"  스캔 측정점 {len(scan_points)}개")

        print(f"[{part_no}] 시트 콜아웃 판독 중...")
        callouts = fetch_sheet_callouts(case["sheet_path"])
        print(f"  콜아웃 {len(callouts)}개")

        pairing = check_pairing(
            [c["value"] for c in callouts],
            [p["value"] for p in scan_points],
        )
        pairing_report[part_no] = pairing
        mark = "OK" if pairing["ok"] else "경고"
        print(f"  짝 확인: [{mark}] {pairing['reason']}")

        rows = match_and_diff(
            part_no, scan_points, callouts,
            entry["sheet_bbox"], entry["scan_bbox"], entry.get("mirrored", False),
        )
        print(f"  매칭된 diff 행 {len(rows)}/{len(callouts)}")
        all_rows.extend(rows)

    return all_rows, pairing_report


def save_dataset(rows: list, pairing_report: dict, out_dir: Path = DATASET_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "correction_diff_dataset.csv"
    fieldnames = [
        "part_no", "scan_point_id", "scan_x", "scan_y", "scan_value",
        "sheet_x", "sheet_y", "projected_x", "projected_y",
        "correction_value", "match_distance_px", "raw_diff", "residual",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report_path = out_dir / "correction_diff_pairing.json"
    report_path.write_text(
        json.dumps(pairing_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="보정치-실측값 diff 데이터셋 생성")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(r"C:\Users\KDT033\Desktop\0line\die_compensation-github.io\data\intermediate"),
    )
    args = parser.parse_args()

    cases = [
        {"part_no": "64XX2", "scan_path": args.data_dir / "JD_64XX2-DR000 3D 스캔.png",
         "sheet_path": args.data_dir / "JD_64XX2-DR000 보정시트.png"},
        {"part_no": "67XX6", "scan_path": args.data_dir / "JD_67XX6-DR000 3D 스캔.png",
         "sheet_path": args.data_dir / "JD_67XX6-DR000 보정시트.png"},
        {"part_no": "71XX2", "scan_path": args.data_dir / "JD_71XX2-DR000 3D 스캔.png",
         "sheet_path": args.data_dir / "JD_71XX2-DR000 보정 시트.png"},
    ]

    rows, pairing_report = build_dataset(cases)
    if not rows:
        print("\n생성된 행이 없습니다.")
        return 1

    path = save_dataset(rows, pairing_report)
    print(f"\n총 {len(rows)}개 diff 행 -> {path}")
    print("\n=== 짝 확인 요약 ===")
    for part_no, report in pairing_report.items():
        mark = "OK" if report["ok"] else "경고"
        print(f"  [{mark}] {part_no}: {report.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
