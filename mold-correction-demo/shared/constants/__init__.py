"""프로젝트 공통 상수 — 경로와 파일명 규격.

모든 파트는 이 상수를 통해 입출력 경로를 참조한다.
경로를 각자 하드코딩하면 통합 시점에 반드시 깨지므로,
파일명 변경이 필요하면 이 파일 하나만 고친다.
"""

from pathlib import Path

# ── 프로젝트 루트 ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"                  # 원본 스캔 (Git 제외)
SAMPLE = DATA / "sample"            # 데모용 샘플
INTERMEDIATE = DATA / "intermediate"  # 파트 간 연결 지점
OUTPUT = DATA / "output"            # 최종 결과

# ── 파트 간 공통 파일명 (input-output-contract.md 와 일치) ────────
DEVIATION_MAP = "deviation_map.png"              # [입력] 원본 편차 이미지
CLEAN_DEVIATION_MAP = "clean_deviation_map.png"  # [4] 라벨 제거 편차 이미지
ZERO_LINE_MASK = "zero_line_mask.png"            # [2] 0-Line 마스크
DEVIATION_POINTS = "deviation_points.csv"        # [3] 편차값·좌표
DEPTH_MEASUREMENTS = "depth_measurements.csv"    # [5] 깊이 측정
RESULT_JSON = "result.json"                      # [1] UI 표시용 통합 결과

# ── [2] 0-Line 파트의 부가 산출물 ────────────────────────────────
ZERO_LINE_OVERLAY = "zero_line_overlay.png"      # 육안 검증용 오버레이
ZERO_LINE_CROSSING = "zero_line_crossing.png"    # 부호 경계선 (임계값 무관)
ZERO_LINE_SWEEP = "zero_line_tolerance_sweep.csv"  # 허용오차 민감도
ZERO_LINE_REGIONS = "zero_line_regions.csv"      # 영역별 면적·중심
ZERO_LINE_CONTOURS = "zero_line_contours.json"   # 0-Line 폴리라인 좌표
ZERO_LINE_REPORT = "zero_line_report.json"       # 처리 파라미터·통계

__all__ = [
    "ROOT", "DATA", "RAW", "SAMPLE", "INTERMEDIATE", "OUTPUT",
    "DEVIATION_MAP", "CLEAN_DEVIATION_MAP", "ZERO_LINE_MASK",
    "DEVIATION_POINTS", "DEPTH_MEASUREMENTS", "RESULT_JSON",
    "ZERO_LINE_OVERLAY", "ZERO_LINE_CROSSING", "ZERO_LINE_SWEEP",
    "ZERO_LINE_REGIONS",
    "ZERO_LINE_CONTOURS", "ZERO_LINE_REPORT",
]
