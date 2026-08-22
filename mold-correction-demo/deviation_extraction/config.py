"""편차 추출 단계에서 공유하는 경로와 경험적 검출 임계값."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT_DIR / "data" / "sample"
INTERMEDIATE_DIR = ROOT_DIR / "data" / "intermediate"

DEVIATION_MAP_PATH = INTERMEDIATE_DIR / "deviation_map.png"
ZERO_LINE_MASK_PATH = INTERMEDIATE_DIR / "zero_line_mask.png"
OUTPUT_CSV_PATH = INTERMEDIATE_DIR / "deviation_points.csv"
DEBUG_IMAGE_PATH = INTERMEDIATE_DIR / "deviation_points_debug.png"

VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# 면적과 거리는 원본 해상도 기준이므로 입력 배율이 바뀌면 함께 조정한다.
LABEL_BORDER_MAX_GRAY = 90
LABEL_BORDER_MAX_SATURATION = 80
LABEL_RED_HUE_MAX = 10
LABEL_RED_HUE_MIN = 170
LABEL_RED_MIN_SATURATION = 80
LABEL_RED_MIN_VALUE = 80
MIN_LABEL_AREA = 150
MAX_LABEL_AREA = 3000
MIN_LABEL_EXTENT = 0.65  # contour area / 축 정렬 bounding-box area
LABEL_MIN_ASPECT = 1.1
LABEL_MAX_ASPECT = 4.0

# OpenCV HSV 범위는 H 0~179, S/V 0~255를 사용한다.
LEADER_LINE_HSV_LOWER = (100, 60, 60)
LEADER_LINE_HSV_UPPER = (130, 255, 255)
LEADER_ANCHOR_RADIUS = 10
MAX_LEADER_LINE_LEN = 260
HOUGH_THRESHOLD = 12
HOUGH_MIN_LINE_LENGTH = 8
HOUGH_MAX_LINE_GAP = 4

DOT_HSV_LOWER = (100, 60, 60)
DOT_HSV_UPPER = (130, 255, 255)
MIN_DOT_AREA = 2
MAX_DOT_AREA = 40
DOT_SNAP_RADIUS = 12

# ROI가 없으면 실제 범례 대신 지정한 Matplotlib 컬러맵으로 교차검증한다.
COLORBAR_ROI: tuple[int, int, int, int] | None = None
COLORBAR_MIN_MM = -3.0
COLORBAR_MAX_MM = 3.0
FALLBACK_COLORMAP = "jet"
CROSS_CHECK_MISMATCH_MM = 1.0
