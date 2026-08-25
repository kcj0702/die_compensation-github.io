"""프로덕션 제로라인 파이프라인을 보정시트 정답과 대조해 채점한다.

[왜 필요한가]
지금까지는 "느낌상 괜찮아 보인다"로 판단했다. 규칙을 고치든(지금 할 수
있는 일), 나중에 사례가 늘어 학습 모델을 붙이든, "이전보다 나아졌는가"를
숫자로 봐야 한다. 이 스크립트가 그 기준선(baseline)이다. 데이터가 더
안 들어와도 지금 3부품으로 계속 돌릴 수 있고, 부품이 늘면 그대로 표본만
늘어난다.

[뭘 채점하는가]
server.py 가 실제로 쓰는 경로 그대로 재현한다 —
load_zero_points -> cluster_zero_points -> connect_strongest_pair.
결과 선의 두 끝점이 진짜 제로라인 끝점에 가까운지가 핵심이다. 지금 규칙
("가장 강한 부호전환 2개")이 JD_64XX2에서 진짜 끝점을 강도 순위 6위·3위로
매겨 놓친 게 이미 알려진 문제라, 이 스크립트로 그걸 숫자로 재확인하고
규칙을 고친 뒤 다시 돌려 실제 나아졌는지 본다.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from zero_line_detection.ml_dataset import CASES, DEFAULT_DATA_DIR, ZERO_POINTS_DIR, _imread
from zero_line_detection.sheet_reference import (
    extract_sheet_zero_areas, extract_sheet_zero_line,
)
from zero_line_detection.zero_line import detect_zero_line
from zero_line_detection.zero_points import (
    cluster_zero_points, connect_strongest_pair, filter_to_key_points,
    load_loop_paths, load_zero_points,
)

LOOP_PATHS_DIR = ZERO_POINTS_DIR.parent.parent / "my_lab" / "scan_point_contour" / "output"
KEY_POINTS_DIR = ZERO_POINTS_DIR.parent.parent / "my_lab" / "zero_point_selection" / "output"


def find_key_points_json(part_key: str):
    """품번으로 key_zero_points.json 을 찾는다(있으면). 없으면 None."""
    for folder in KEY_POINTS_DIR.glob("*"):
        if part_key in folder.name.upper():
            candidate = folder / "key_zero_points.json"
            if candidate.is_file():
                return candidate
    return None


def truth_points(sheet_path, part_mask, part_no, values):
    try:
        ref = extract_sheet_zero_line(sheet_path, part_mask, part_no, values=values)
        return np.array(ref.points, dtype=float), "line"
    except ValueError:
        ref = extract_sheet_zero_areas(sheet_path, part_mask, part_no, values=values)
        return np.vstack([np.array(c, dtype=float) for c in ref.contours]), "areas"


def find_loop_paths(part_key: str) -> dict:
    """server.py 의 매칭 로직과 동일하게 품번으로 loop 폴더를 찾는다."""
    for folder in LOOP_PATHS_DIR.glob("*"):
        if part_key in folder.name.upper():
            candidate = folder / "scan_point_loops.json"
            if candidate.is_file():
                return load_loop_paths(candidate)
    return {}


def evaluate_case(case, use_key_filter: bool = False) -> dict:
    """use_key_filter=True 면 key_zero_point_engine 이 '핵심'으로 남긴
    후보만 클러스터링에 넣는다(있는 품번만 — 없으면 원래대로 전체 후보).
    """
    scan_path = DEFAULT_DATA_DIR / case.scan_name
    sheet_path = DEFAULT_DATA_DIR / case.sheet_name
    points_path = ZERO_POINTS_DIR / f"{case.part_no}.json"
    result: dict = {"part_no": case.part_no}
    if not (scan_path.exists() and sheet_path.exists() and points_path.exists()):
        result["status"] = "파일 없음"
        return result

    scan = _imread(scan_path)
    output = detect_zero_line(cv2.cvtColor(scan, cv2.COLOR_BGR2RGB), case.config)
    values, part_mask = output.values, output.part_mask
    diag = float(np.hypot(*values.shape[::-1]))

    truth, truth_kind = truth_points(sheet_path, part_mask, case.part_no, values)
    result["truth_kind"] = truth_kind

    candidates = load_zero_points(points_path)
    result["n_candidates_before_key_filter"] = len(candidates)
    if use_key_filter:
        key_json = find_key_points_json(case.part_no)
        if key_json is not None:
            candidates = filter_to_key_points(candidates, key_json)
    loop_paths = find_loop_paths(case.part_no)
    clusters = cluster_zero_points(candidates, loop_paths=loop_paths)
    result["n_candidates"] = len(candidates)
    result["n_clusters"] = len(clusters)

    line = connect_strongest_pair(
        clusters, values, part_mask, float(output.result.tolerance))
    if line is None:
        result["status"] = "선 생성 실패 (끝점 후보 부족)"
        return result

    pred_pts = np.array(line.points, dtype=float)
    d_fwd = np.array([np.hypot(*(truth - p).T).min() for p in pred_pts])
    d_bwd = np.array([np.hypot(*(pred_pts - t).T).min() for t in truth])

    # 실제로 골라진 두 끝점이 정답에 얼마나 가까운가 — 지금 규칙의 핵심
    # 약점(가장 강한 부호전환 2개 = 진짜 끝점이라는 보장이 없음)을 직접 잰다.
    endpoints = pred_pts[[0, -1]]
    d_endpoints = np.array([np.hypot(*(truth - p).T).min() for p in endpoints])

    result.update({
        "status": "ok",
        "n_pred_points": len(pred_pts),
        "pred_to_truth_median_pct": round(float(np.median(d_fwd) / diag * 100), 2),
        "truth_to_pred_median_pct": round(float(np.median(d_bwd) / diag * 100), 2),
        "endpoint_dist_px": [round(float(d), 1) for d in d_endpoints],
        "endpoint_dist_pct": [round(float(d) / diag * 100, 2) for d in d_endpoints],
    })
    return result


def _print_table(rows: list) -> None:
    print(f"{'품번':8s} {'정답':6s} {'후보':>4s} {'군집':>4s} {'상태':10s} "
          f"{'선->정답%':>9s} {'정답->선%':>9s} {'끝점오차%':>16s}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['part_no']:8s} {'':6s} {'':>4s} {'':>4s} "
                  f"{r.get('status', '?'):10s}")
            continue
        ep = "/".join(f"{v:.1f}" for v in r["endpoint_dist_pct"])
        print(f"{r['part_no']:8s} {r['truth_kind']:6s} {r['n_candidates']:4d} "
              f"{r['n_clusters']:4d} {'ok':10s} "
              f"{r['pred_to_truth_median_pct']:9.2f} "
              f"{r['truth_to_pred_median_pct']:9.2f} {ep:>16s}")


def main() -> int:
    print("=== 기본 (전체 0포인트 후보) ===")
    baseline = [evaluate_case(case, use_key_filter=False) for case in CASES]
    _print_table(baseline)

    print("\n=== key_zero_point_engine 필터 적용 (있는 품번만) ===")
    filtered = [evaluate_case(case, use_key_filter=True) for case in CASES]
    _print_table(filtered)

    print(
        "\n끝점오차%는 실제로 골라진 두 끝점 각각이 정답에서 얼마나 떨어져"
        "\n있는지(대각선 대비 %)다. 이 값이 크면 '가장 강한 부호전환 2개'"
        "\n규칙이 진짜 끝점을 놓쳤다는 뜻. key 필터는 컬러바 HSV로 후보"
        "\n주변 실제 편차를 다시 확인해 노이즈 후보를 줄인다(현업 제공,"
        "\n2026-08-25) — 위 두 표를 비교해 실제로 나아졌는지 본다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
