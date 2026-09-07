"""보정시트를 현업 엑셀 양식으로 낸다.

[양식을 뜯어보고 알게 된 것]
현업이 준 `보정시트_양식.xlsx` 와 실제 작성 사례 두 건을 열어 봤다.

    보정시트_양식.xlsx          40행 x 30열, 인쇄영역 A1:AD40, A4 가로
    CD8 71XX2_22 보정내용.xlsx  이미지 10개, 값 있는 칸 48
    JM 67312-DZ000_보정적용.xlsx 이미지 7개,  값 있는 칸 40

**표가 아니라 그림이다.** 머리말 여섯 칸(관리 NO / PART NAME / 공정 /
PART NO / 원소재 / 적용일자)만 글자고, 나머지는 보정치를 그려 넣은
스캔 그림을 통째로 붙인다. 그리고 그 40행 묶음이 공정마다 반복된다
(1행, 41행, 81행 …).

    J1:L2 관리 NO   M1:T2 (값)   U1:X2 PART NAME  Y1:AD2 (값)
    J3:L4 공   정   M3:T4 (값)   U3:X4 PART NO    Y3:AD4 (값)
    J5:L6 원소재    M5:T6 (값)   U5:X6 적용일자   Y5:AD6 (값)

[그래서 이렇게 만든다]
1쪽 — 양식 그대로. 머리말을 채우고 보정치를 그린 그림을 붙인다.
2쪽 — 포인트 표. 원래 양식에는 없지만 "포인트 별 편차 엑셀 자동 작성"
      이 향후 계획 항목이라 따로 시트를 붙인다. 사람이 그림을 보고,
      기계가 표를 읽는다.

[그림은 여기서 그린다]
화면(DOM)을 이미지로 굽는 방법도 있지만 html2canvas 같은 걸 새로
받아야 한다. 사내망에서 도는 게 이 프로젝트의 전제라 서버에서
OpenCV 로 직접 그린다.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

TEMPLATE = Path(__file__).resolve().parent / "templates" / "보정시트_양식.xlsx"

# 양식의 머리말 자리 — 왼쪽 라벨은 이미 적혀 있고 값 칸만 채운다.
HEADER_CELLS = {
    "control_no": "M1",
    "part_name": "Y1",
    "process": "M3",
    "part_no": "Y3",
    "material": "M5",
    "applied_at": "Y5",
}
# 그림이 들어갈 자리. 머리말이 6행까지라 8행부터 쓴다.
IMAGE_ANCHOR = "A8"
IMAGE_ROWS = 32           # 8행부터 40행까지
ROW_POINTS = 14.0         # 양식의 행 높이(pt)
PAGE_ROWS = 40            # 한 페이지가 40행. 실제 시트도 이만큼씩 반복된다
PROCESS_ROW = 36          # 공정 표기가 들어갈 자리(그림 아래)
PROCESS_COL = 2           # B 열
# 양식의 A~AD 열을 실측한 폭(px). 가로로 긴 3D 화면이 인쇄영역 밖으로
# 나가지 않게 여기에 맞춘다.
IMAGE_WIDTH_PX = 1670
IMAGE_HEIGHT_PX = 591     # 8~40행을 실측한 높이(px)
CAPTION_ROW = 7           # 그림 바로 위 줄 — 어느 시점 그림인지 적는다
CAPTION_COL = 2           # B 열

PLUS = (63, 72, 224)      # BGR — 살이 많다(깎는다)
MINUS = (224, 127, 47)    # BGR — 살이 부족하다(붙인다)


@dataclass
class SheetPoint:
    """보정시트에 찍힐 포인트 하나."""

    point_id: str
    x_px: int
    y_px: int
    deviation: float      # 스캔에서 읽은 편차
    correction: float     # 최종 보정량 (작업자 수정 반영)


def draw_sheet_image(base_bgr: np.ndarray, points: list) -> np.ndarray:
    """스캔 그림 위에 보정치를 콜아웃으로 그린다."""
    canvas = base_bgr.copy()
    height, width = canvas.shape[:2]
    scale = max(min(width, height) / 900.0, 0.55)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for point in points:
        colour = PLUS if point.correction > 0 else MINUS
        centre = (int(point.x_px), int(point.y_px))
        cv2.circle(canvas, centre, max(int(5 * scale), 3), colour, -1)
        cv2.circle(canvas, centre, max(int(5 * scale), 3), (255, 255, 255),
                   max(int(1.4 * scale), 1))

        text = f"{'+' if point.correction > 0 else ''}{point.correction:.1f}"
        size, _ = cv2.getTextSize(text, font, 0.5 * scale, max(int(1.4 * scale), 1))
        # 콜아웃은 점 오른쪽 위. 화면 밖으로 나가면 반대편에 붙인다.
        box_x = centre[0] + int(11 * scale)
        box_y = centre[1] - int(11 * scale)
        if box_x + size[0] + 10 > width:
            box_x = centre[0] - int(11 * scale) - size[0] - 10
        if box_y - size[1] - 8 < 0:
            box_y = centre[1] + int(11 * scale) + size[1]

        pad = int(4 * scale)
        cv2.rectangle(canvas,
                      (box_x - pad, box_y - size[1] - pad),
                      (box_x + size[0] + pad, box_y + pad),
                      colour, -1)
        cv2.line(canvas, centre, (box_x, box_y - size[1] // 2), colour,
                 max(int(1.2 * scale), 1))
        cv2.putText(canvas, text, (box_x, box_y), font, 0.5 * scale,
                    (255, 255, 255), max(int(1.3 * scale), 1), cv2.LINE_AA)
    return canvas


def trim_border(image, slack: int = 12):
    """그림 둘레의 빈 바탕을 잘라낸다.

    [왜 필요한가]
    3D 화면은 가로로 넓은데(실측 캔버스가 2.1:1) 부품은 가운데만 차지한다.
    그대로 시트에 실으면 부품이 작게 떠 있고 둘레가 허옇게 남아, 현업
    시트처럼 "부품이 자리를 채운" 그림이 안 나온다.

    네 귀퉁이 색을 바탕으로 보고, 그와 다른 화소가 처음 나오는 자리까지
    자른다. 흰 바탕이든 어두운 바탕이든 같은 방법으로 된다. 여백을 조금
    남겨야 콜아웃 글자가 잘리지 않는다.
    """
    if image is None or image.size == 0:
        return image
    high, wide = image.shape[:2]
    corners = np.array([image[0, 0], image[0, wide - 1],
                        image[high - 1, 0], image[high - 1, wide - 1]],
                       dtype=np.int16)
    paper = np.median(corners, axis=0)
    # 바탕과 얼마나 다른가. 8 이면 눈에 안 보이는 그라데이션은 넘긴다.
    unlike = np.abs(image.astype(np.int16) - paper).max(axis=2) > 8
    rows = np.flatnonzero(unlike.any(axis=1))
    cols = np.flatnonzero(unlike.any(axis=0))
    if not len(rows) or not len(cols):
        return image
    top = max(0, int(rows[0]) - slack)
    bottom = min(high, int(rows[-1]) + 1 + slack)
    left = max(0, int(cols[0]) - slack)
    right = min(wide, int(cols[-1]) + 1 + slack)
    if bottom - top < 8 or right - left < 8:
        return image
    return image[top:bottom, left:right]


def fit_on_page(image, width: int = IMAGE_WIDTH_PX,
                height: int = IMAGE_HEIGHT_PX, pad: int = 16):
    """그림을 양식의 그림 자리에 꼭 맞는 한 장으로 앉힌다.

    [왜 필요한가 — 지금 엑셀이 못 쓸 물건이었다]
    쪽마다 그림 크기가 제각각이라 어떤 쪽은 가로로 늘어지고 어떤 쪽은
    구석에 작게 박혔다. 비율을 지키느라 폭이나 높이 하나만 맞췄기 때문이다.
    시트는 40행 묶음이 되풀이되는 물건이라 **쪽마다 그림 자리가 같아야**
    넘길 때 눈이 안 흔들린다.

    그래서 자리 크기(A~AD x 8~40행, 실측 1670x591)의 흰 종이를 먼저 깔고,
    부품을 비율 그대로 최대한 키워 가운데 놓는다. 엑셀에는 늘 같은 크기로
    붙으므로 쪽이 몇 장이든 줄이 맞는다. 테두리를 얇게 둘러 화면의
    "정면도 · FRONT VIEW" 틀과 같은 인상을 준다.

    글자는 여기서 넣지 않는다 — OpenCV 는 한글을 못 그린다. 쪽 이름은
    엑셀 칸에 적는다(CAPTION_ROW).
    """
    if image is None or image.size == 0:
        return np.full((height, width, 3), 255, np.uint8)

    paper = np.full((height, width, 3), 255, np.uint8)
    room_w, room_h = width - pad * 2, height - pad * 2
    high, wide = image.shape[:2]
    scale = min(room_w / wide, room_h / high)
    # 작은 그림을 억지로 키우면 뭉갠다. 3배까지만 키운다.
    scale = min(scale, 3.0)
    new_w, new_h = max(1, int(wide * scale)), max(1, int(high * scale))
    shrunk = cv2.resize(image, (new_w, new_h),
                        interpolation=(cv2.INTER_AREA if scale < 1
                                       else cv2.INTER_CUBIC))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    paper[top:top + new_h, left:left + new_w] = shrunk
    # 얇은 테두리 — 인쇄했을 때 그림 자리가 어디까지인지 보인다.
    cv2.rectangle(paper, (0, 0), (width - 1, height - 1), (214, 218, 222), 1)
    return paper


def build_workbook(
    pages: list,
    points: list,
    part_no: str = "",
    part_name: str = "",
    process: str = "",
    material: str = "",
    control_no: str = "",
    applied_at: str | None = None,
    coefficient: float = 1.0,
    processes: list | None = None,
    captions: list | None = None,
) -> bytes:
    """현업 양식으로 채운 엑셀 파일을 바이트로 준다."""
    import openpyxl
    from openpyxl.drawing.image import Image as XlImage
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    if not TEMPLATE.exists():
        raise FileNotFoundError(f"보정시트 양식이 없습니다: {TEMPLATE}")

    book = openpyxl.load_workbook(TEMPLATE)
    page = book["00"]

    if isinstance(pages, np.ndarray):      # 한 장만 준 경우도 받는다
        pages = [pages]

    values = {
        "control_no": control_no or f"{part_no}-01",
        "part_name": part_name,
        "process": process or "OP10",
        "part_no": part_no,
        "material": material,
        "applied_at": applied_at or date.today().isoformat(),
    }
    # ── 그림 붙이기 ──────────────────────────────────────────
    # 실제 시트는 40행 묶음이 공정마다 반복된다(1행, 41행, 81행 …).
    # 여러 장을 받으면 같은 방식으로 페이지를 늘린다.
    box_height = int(IMAGE_ROWS * ROW_POINTS * 4 / 3)
    for index, image in enumerate(pages):
        offset = index * PAGE_ROWS
        for key, cell in HEADER_CELLS.items():
            column = "".join(ch for ch in cell if ch.isalpha())
            row = int("".join(ch for ch in cell if ch.isdigit())) + offset
            page[f"{column}{row}"] = values[key]

        # 빈 바탕을 잘라내고, 양식의 그림 자리에 꼭 맞는 한 장으로 앉힌다.
        # 쪽마다 같은 크기라야 넘길 때 줄이 맞는다.
        image = fit_on_page(trim_border(image))
        box_width, box_height_here = IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("시트 그림을 PNG 로 만들지 못했습니다.")
        picture = XlImage(io.BytesIO(buffer.tobytes()))
        picture.width, picture.height = box_width, box_height_here
        anchor_row = int("".join(ch for ch in IMAGE_ANCHOR if ch.isdigit())) + offset
        anchor_col = "".join(ch for ch in IMAGE_ANCHOR if ch.isalpha())
        page.add_image(picture, f"{anchor_col}{anchor_row}")

        # 그림 바로 위에 그 쪽이 무슨 그림인지 적는다 — "3D 형상 · 우측"
        # 처럼. 3D 는 돌려 놓고 찍으므로 방향을 안 적으면 나중에 못 가린다.
        if captions and index < len(captions) and captions[index]:
            head = page.cell(CAPTION_ROW + offset, CAPTION_COL)
            head.value = str(captions[index])
            head.font = Font(size=9, bold=True)
            head.alignment = Alignment(horizontal="left", vertical="center")

    if len(pages) > 1:
        page.print_area = f"A1:AD{PAGE_ROWS * len(pages)}"

    # 공정 표기 — 실제 시트도 그림 아래에 "① : 하형 용접" 처럼 적는다.
    if processes:
        from openpyxl.styles import Font as _Font
        for order, line in enumerate(processes):
            cell = page.cell(PROCESS_ROW + order, PROCESS_COL)
            cell.value = line
            cell.font = _Font(bold=True, color="B31563", size=11)

    # ── 포인트 표 ────────────────────────────────────────────
    table = book.create_sheet("포인트")
    headers = ["포인트", "X(px)", "Y(px)", "편차(mm)", "보정량(mm)", "방향"]
    table.append(headers)
    for index, name in enumerate(headers, start=1):
        cell = table.cell(1, index)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
    for point in points:
        table.append([
            point.point_id, point.x_px, point.y_px,
            round(point.deviation, 2), round(point.correction, 2),
            "가공(살빼기)" if point.correction < 0 else "용접(살붙이기)",
        ])
    for index, size in enumerate((12, 10, 10, 12, 13, 16), start=1):
        table.column_dimensions[get_column_letter(index)].width = size
    table.freeze_panes = "A2"

    note = table.cell(len(points) + 3, 1)
    note.value = (f"보정 계수 {coefficient:.2f}x · 보정량은 작업자 수정을 "
                  f"반영한 최종값입니다.")
    note.font = Font(italic=True, size=9)

    for order, line in enumerate(processes or []):
        row = table.cell(len(points) + 5 + order, 1)
        row.value = line
        row.font = Font(bold=True, size=9)

    stream = io.BytesIO()
    book.save(stream)
    return stream.getvalue()


__all__ = ["SheetPoint", "TEMPLATE", "draw_sheet_image", "build_workbook"]
