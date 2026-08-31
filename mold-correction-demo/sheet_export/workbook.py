"""Assemble the correction sheet workbook.

openpyxl handles the cells and the pictures -- including the media parts and
relationships -- and the value labels are appended to the drawing part
afterwards, because openpyxl cannot create text boxes or connectors.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import openpyxl
from openpyxl.drawing.image import Image as XLImage
from openpyxl.drawing.spreadsheet_drawing import AbsoluteAnchor
from openpyxl.drawing.xdr import XDRPoint2D, XDRPositiveSize2D

from . import config, drawing
from .layout import SheetView, default_layout, place_labels


@dataclass
class TitleBlock:
    """The values that go into the sheet's header cells."""

    management_no: str = ""
    part_name: str = ""
    process: str = ""
    part_no: str = ""
    material: str = ""
    applied_date: str = ""

    def as_cells(self) -> dict[str, str]:
        return {
            cell: getattr(self, field_name)
            for field_name, cell in config.TITLE_CELLS.items()
            if getattr(self, field_name)
        }


@dataclass
class BuildReport:
    """What actually landed on the sheet."""

    path: Path
    pictures: int = 0
    labels: int = 0
    leaders: int = 0
    warnings: list[str] = field(default_factory=list)


def _encode_png(image: np.ndarray) -> io.BytesIO:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("뷰 이미지를 PNG로 변환하지 못했습니다.")
    return io.BytesIO(encoded.tobytes())


def _open_template(template: Path | None) -> openpyxl.Workbook:
    """Use the company form when it is available, otherwise a blank book.

    openpyxl drops the form's own drawing and printer settings on save, so the
    blank fallback keeps the tool usable rather than matching the form exactly.
    """
    candidate = Path(template) if template else config.DEFAULT_TEMPLATE
    if candidate.is_file():
        return openpyxl.load_workbook(candidate)
    return openpyxl.Workbook()


def build_sheet(
    views: Sequence[SheetView],
    title: TitleBlock | None = None,
    *,
    output: Path,
    template: Path | None = None,
) -> BuildReport:
    """Write the correction sheet and report what it contains."""
    if not views:
        raise ValueError("시트에 올릴 뷰가 없습니다.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    resolved = Path(template) if template else config.DEFAULT_TEMPLATE
    if not resolved.is_file():
        warnings.append(
            f"양식 파일이 없어 빈 통합문서로 만들었습니다: {resolved}"
        )

    book = _open_template(template)
    sheet = book.active
    for cell, value in (title or TitleBlock()).as_cells().items():
        sheet[cell] = value

    default_layout(list(views))

    anchors: list[str] = []
    shape_id = 1000
    labels = leaders = 0
    for view in views:
        box_x, box_y, box_width, box_height = view.box
        picture = XLImage(_encode_png(view.image))
        picture.anchor = AbsoluteAnchor(
            pos=XDRPoint2D(drawing.emu(box_x), drawing.emu(box_y)),
            ext=XDRPositiveSize2D(drawing.emu(box_width), drawing.emu(box_height)),
        )
        sheet.add_image(picture)

        if view.title:
            shape_id += 1
            anchors.append(
                drawing.caption(
                    shape_id,
                    view.title,
                    box_x,
                    box_y - config.DETAIL_TITLE_HEIGHT - 2,
                    box_width,
                )
            )

        for placed in place_labels(view):
            shape_id += 1
            anchors.append(
                drawing.leader(
                    shape_id,
                    placed.label_x + config.LABEL_WIDTH / 2,
                    placed.label_y + config.LABEL_HEIGHT / 2,
                    placed.point_x,
                    placed.point_y,
                )
            )
            leaders += 1
            shape_id += 1
            anchors.append(
                drawing.text_box(shape_id, placed.text, placed.label_x, placed.label_y)
            )
            labels += 1

    book.save(output)
    _append_anchors(output, anchors)

    return BuildReport(
        path=output,
        pictures=len(views),
        labels=labels,
        leaders=leaders,
        warnings=warnings,
    )


def _append_anchors(path: Path, anchors: list[str]) -> None:
    """Rewrite the workbook zip with the extra shapes in the drawing part.

    The scratch file sits next to the output so the final move stays on one
    filesystem and no temporary handle is left open on Windows.
    """
    if not anchors:
        return
    temporary = path.with_name(path.name + ".building")
    temporary.unlink(missing_ok=True)
    try:
        injected = False
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for info in source.infolist():
                payload = source.read(info.filename)
                if drawing.DRAWING_PART.match(info.filename):
                    payload = drawing.inject(
                        payload.decode("utf-8"), anchors
                    ).encode("utf-8")
                    injected = True
                target.writestr(info, payload)
        if not injected:
            raise ValueError("이미지가 없어 라벨을 넣을 drawing 파트를 찾지 못했습니다.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
