"""보정시트 엑셀을 만들 때 쓰는 기하값과 서식값."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
# 사내 양식. 없으면 빈 통합문서로 대체한다.
DEFAULT_TEMPLATE = ROOT_DIR / "data" / "template" / "보정시트_양식.xlsx"

# 엑셀 좌표 단위. 1인치 = 914400 EMU = 96px 이므로 1px = 9525 EMU.
EMU_PER_PIXEL = 9525

# 양식의 시트 치수에서 계산한 도면 영역(px).
# 행 높이 13.5pt = 18px, 표제란이 1~6행이라 도면은 7행(108px)부터다.
# 열 폭은 1~22열 3.78(31px), 23열 1.33(14px), 24~30열 5.11(41px) -> A~AD 983px.
SHEET_WIDTH = 983
DRAWING_TOP = 108
DRAWING_BOTTOM = 738

# 라벨은 부품 바깥 여백에 두고 지시선으로 잇는다. 원본 시트와 같은 배치다.
LABEL_WIDTH = 44
LABEL_HEIGHT = 18
LABEL_GUTTER = 16          # 이미지 가장자리와 라벨 사이 간격
LABEL_MIN_GAP = 4          # 같은 변에 놓인 라벨끼리의 최소 간격
LABEL_FONT_SIZE = 900      # 1/100 pt 단위. 900 = 9pt
LABEL_FILL = "FFFFFF"
LABEL_LINE = "404040"
LEADER_LINE = "C05000"
LEADER_WIDTH = 9525

# 뷰 배치: 정면도는 위쪽, Detail View 는 아래쪽에 가로로 늘어놓는다.
VIEW_MARGIN = 12
FRONT_HEIGHT_RATIO = 0.62  # 정면도가 가져갈 도면 영역 높이 비율
DETAIL_GAP = 14
DETAIL_TITLE_HEIGHT = 14

# 표제란 셀 위치. 양식의 병합 셀 기준이다.
TITLE_CELLS = {
    "management_no": "M1",
    "part_name": "Y1",
    "process": "M3",
    "part_no": "Y3",
    "material": "M5",
    "applied_date": "Y5",
}
