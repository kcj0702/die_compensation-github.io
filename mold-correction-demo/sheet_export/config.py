"""보정시트 엑셀을 만들 때 쓰는 기하값과 서식값."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
# 사내 양식. 없으면 빈 통합문서로 대체한다.
DEFAULT_TEMPLATE = ROOT_DIR / "data" / "template" / "보정시트_양식.xlsx"

# 엑셀 좌표 단위. 1인치 = 914400 EMU = 96px 이므로 1px = 9525 EMU.
EMU_PER_PIXEL = 9525

# 양식의 시트 치수에서 계산한 도면 영역(px).
# 행 높이 13.5pt = 18px, 표제란이 1~6행이라 도면은 7행(108px)부터다.
# 인쇄 영역은 40행(=720px)에서 끝나므로 도면 바닥도 여기에 맞춘다. 예전에는
# 738px(=41행 위쪽)로 계산해서 마지막 행 폭만큼 도면 바닥의 주석이 인쇄
# 영역 바깥으로 밀려나 잘려 보였다.
# 열 폭은 1~22열 3.78(31px), 23열 1.33(14px), 24~30열 5.11(41px) -> A~AD 983px.
SHEET_WIDTH = 983
DRAWING_TOP = 108
DRAWING_BOTTOM = 720

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

# 측정 포인트에 찍는 작은 원. 지금까지는 지시선 끝의 tailEnd 만 있어서 눈에 잘
# 안 띄고, 라벨/지시선과 함께 잡아 옮길 대상도 없었다. 라벨과 같은 그룹에
# 넣어 셋이 한 덩어리로 움직이도록 만든다.
POINT_DOT_RADIUS = 4       # px
POINT_DOT_COLOR = "9B1C1C"

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

TITLE_LABEL_CELLS = {
    "management_label": "J1",
    "part_name_label": "U1",
    "process_label": "J3",
    "part_no_label": "U3",
    "material_label": "J5",
    "applied_date_label": "U5",
}

TITLE_BLOCK_MERGES = (
    "A1:I6",
    "J1:L2", "M1:T2", "U1:X2", "Y1:AD2",
    "J3:L4", "M3:T4", "U3:X4", "Y3:AD4",
    "J5:L6", "M5:T6", "U5:X6", "Y5:AD6",
)

TITLE_DEFAULT_FONTS = {
    "heading": "휴먼옛체",
    "management_label": "돋움",
    "management_no": "맑은 고딕",
    "part_name_label": "돋움",
    "part_name": "맑은 고딕",
    "process_label": "돋움",
    "process": "맑은 고딕",
    "part_no_label": "돋움",
    "part_no": "맑은 고딕",
    "material_label": "돋움",
    "material": "맑은 고딕",
    "applied_date_label": "돋움",
    "applied_date": "맑은 고딕",
}

TITLE_DEFAULT_SIZES = {
    "heading": 18,
    "management_label": 10, "management_no": 10,
    "part_name_label": 10, "part_name": 10,
    "process_label": 10, "process": 10,
    "part_no_label": 10, "part_no": 10,
    "material_label": 10, "material": 10,
    "applied_date_label": 10, "applied_date": 10,
}

PRINT_AREA = "A1:AD40"
PRINT_PAGE_ROWS = 40
PAGE_BREAK_PREVIEW_ZOOM = 85

# 표제란 왼쪽 여백에 들어가는 시트 제목. 양식에는 비어 있는 A1:I6 영역이라
# 새로 병합해서 채운다.
SHEET_HEADING_TEXT = "보정 적용 내용"
SHEET_HEADING_RANGE = "A1:I6"
SHEET_HEADING_ANCHOR = "A1"
SHEET_HEADING_FONT_SIZE = 18
