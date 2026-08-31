"""Build the drawing XML that openpyxl cannot: text boxes and leader lines.

openpyxl writes pictures and nothing else, so the value labels and their leaders
are appended to the drawing part directly. openpyxl emits the spreadsheetDrawing
namespace as the default one -- `<wsDr>`, `<pic>`, no `xdr:` prefix -- so these
elements must be written the same way or Excel silently drops them.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape

from . import config


def emu(pixels: float) -> int:
    """Convert pixels to EMU, the unit Excel drawings use."""
    return int(round(pixels * config.EMU_PER_PIXEL))


def text_box(shape_id: int, text: str, x: float, y: float,
             width: float | None = None, height: float | None = None) -> str:
    """A bordered white box holding one value, positioned absolutely."""
    width = config.LABEL_WIDTH if width is None else width
    height = config.LABEL_HEIGHT if height is None else height
    return (
        f'<absoluteAnchor><pos x="{emu(x)}" y="{emu(y)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'<sp macro="" textlink=""><nvSpPr>'
        f'<cNvPr id="{shape_id}" name="TextBox {shape_id}"/>'
        f'<cNvSpPr txBox="1"/></nvSpPr>'
        f'<spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{config.LABEL_FILL}"/></a:solidFill>'
        f'<a:ln w="9525"><a:solidFill>'
        f'<a:srgbClr val="{config.LABEL_LINE}"/></a:solidFill></a:ln>'
        f'</spPr><txBody>'
        f'<a:bodyPr vertOverflow="clip" horzOverflow="clip" wrap="none"'
        f' lIns="9000" tIns="4500" rIns="9000" bIns="4500" anchor="ctr"/>'
        f'<a:lstStyle/><a:p><a:pPr algn="ctr"/>'
        f'<a:r><a:rPr lang="ko-KR" sz="{config.LABEL_FONT_SIZE}" b="1"/>'
        f'<a:t>{escape(text)}</a:t></a:r></a:p>'
        f'</txBody></sp><clientData/></absoluteAnchor>'
    )


def leader(shape_id: int, x1: float, y1: float, x2: float, y2: float) -> str:
    """A straight line from a label to its measurement point.

    A line's extent is always positive; the direction is carried by flipH/flipV.
    """
    left, top = min(x1, x2), min(y1, y2)
    width = max(abs(x2 - x1), 1.0)
    height = max(abs(y2 - y1), 1.0)
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    return (
        f'<absoluteAnchor><pos x="{emu(left)}" y="{emu(top)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'<cxnSp macro=""><nvCxnSpPr>'
        f'<cNvPr id="{shape_id}" name="Leader {shape_id}"/>'
        f'<cNvCxnSpPr/></nvCxnSpPr>'
        f'<spPr><a:xfrm{flip_h}{flip_v}>'
        f'<a:off x="{emu(left)}" y="{emu(top)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
        f'<a:ln w="{config.LEADER_WIDTH}">'
        f'<a:solidFill><a:srgbClr val="{config.LEADER_LINE}"/></a:solidFill>'
        f'<a:tailEnd type="oval" w="sm" len="sm"/></a:ln>'
        f'</spPr></cxnSp><clientData/></absoluteAnchor>'
    )


def caption(shape_id: int, text: str, x: float, y: float, width: float) -> str:
    """A borderless left-aligned caption, used for view titles."""
    height = config.DETAIL_TITLE_HEIGHT
    return (
        f'<absoluteAnchor><pos x="{emu(x)}" y="{emu(y)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'<sp macro="" textlink=""><nvSpPr>'
        f'<cNvPr id="{shape_id}" name="Caption {shape_id}"/>'
        f'<cNvSpPr txBox="1"/></nvSpPr>'
        f'<spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:noFill/><a:ln><a:noFill/></a:ln></spPr><txBody>'
        f'<a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr"/>'
        f'<a:lstStyle/><a:p>'
        f'<a:r><a:rPr lang="ko-KR" sz="800" b="1"/>'
        f'<a:t>{escape(text)}</a:t></a:r></a:p>'
        f'</txBody></sp><clientData/></absoluteAnchor>'
    )


def inject(drawing_xml: str, anchors: list[str]) -> str:
    """Append anchors just before the drawing part closes."""
    if not anchors:
        return drawing_xml
    if "</wsDr>" not in drawing_xml:
        raise ValueError("drawing XML의 닫는 태그를 찾지 못했습니다.")
    return drawing_xml.replace("</wsDr>", "".join(anchors) + "</wsDr>")


DRAWING_PART = re.compile(r"xl/drawings/drawing\d+\.xml$")
