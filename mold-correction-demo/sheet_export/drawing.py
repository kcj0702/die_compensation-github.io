"""Build the drawing XML that openpyxl cannot: text boxes, leaders, and dots.

openpyxl writes pictures and nothing else, so the value labels and their
leaders are appended to the drawing part directly. openpyxl emits the
spreadsheetDrawing namespace as the default one -- ``<wsDr>``, ``<pic>``, no
``xdr:`` prefix -- so these elements must be written the same way or Excel
silently drops them.

A callout is three separate anchors: a dot at the measurement point, the
value label, and a connector line between them. The connector carries
``<a:stCxn>`` / ``<a:endCxn>`` referencing the label's and dot's shape ids,
which tells Excel to keep them attached -- drag the label and Excel
re-routes the line to still touch both ends. An earlier version bundled
them in ``<grpSp>`` so they moved as one rigid unit, but that dragged the
dot away from its measurement location.

Only two guarantees are load-bearing here: the label is a real text box
(``txBox="1"``) so Excel opens it for editing on double-click, and the
connector carries ``<a:stCxn>`` / ``<a:endCxn>`` referencing the label's
and dot's shape ids so Excel stretches the line when the label is dragged.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape

from . import config


def emu(pixels: float) -> int:
    """Convert pixels to EMU, the unit Excel drawings use."""
    return int(round(pixels * config.EMU_PER_PIXEL))


# --- shape bodies (no anchor wrapper; safe to nest inside a group) --------

def _text_box_body(
    shape_id: int, text: str, x: float, y: float, width: float, height: float,
    font_family: str | None = None,
) -> str:
    """Bordered white text box body. `txBox="1"` is what makes Excel treat it
    as editable text on double-click.

    ``<a:spLocks>`` explicitly set to all zeros is what keeps the label
    freely selectable and text-editable when sheet-level object protection
    is on. Excel treats a shape without `<a:spLocks>` as locked by default
    once protection is enabled, so writing every attribute as ``0`` is what
    turns the label into an exception.
    """
    typeface = escape(font_family) if font_family else ""
    font_xml = (
        f'<a:latin typeface="{typeface}"/><a:ea typeface="{typeface}"/>'
        if typeface else ""
    )
    return (
        f'<sp macro="" textlink="">'
        f'<nvSpPr>'
        f'<cNvPr id="{shape_id}" name="TextBox {shape_id}"/>'
        f'<cNvSpPr txBox="1">'
        f'<a:spLocks noGrp="0" noSelect="0" noRot="0" noChangeAspect="0"'
        f' noMove="0" noResize="0" noEditPoints="0" noAdjustHandles="0"'
        f' noChangeArrowheads="0" noChangeShapeType="0" noTextEdit="0"/>'
        f'</cNvSpPr></nvSpPr>'
        f'<spPr>'
        f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{config.LABEL_FILL}"/></a:solidFill>'
        f'<a:ln w="9525"><a:solidFill>'
        f'<a:srgbClr val="{config.LABEL_LINE}"/></a:solidFill></a:ln>'
        f'</spPr><txBody>'
        f'<a:bodyPr vertOverflow="clip" horzOverflow="clip" wrap="none"'
        f' lIns="9000" tIns="4500" rIns="9000" bIns="4500" anchor="ctr"/>'
        f'<a:lstStyle/><a:p><a:pPr algn="ctr"/>'
        f'<a:r><a:rPr lang="ko-KR" sz="{config.LABEL_FONT_SIZE}" b="1">{font_xml}</a:rPr>'
        f'<a:t>{escape(text)}</a:t></a:r></a:p>'
        f'</txBody></sp>'
    )


def _leader_body(
    shape_id: int, x1: float, y1: float, x2: float, y2: float,
    *,
    start_shape_id: int | None = None, start_idx: int = 0,
    end_shape_id: int | None = None, end_idx: int = 0,
) -> str:
    """Straight line from label to point.

    When ``start_shape_id`` and ``end_shape_id`` are given, the connector is
    *attached* to those shapes via ``<a:stCxn>`` / ``<a:endCxn>``. Excel
    then re-routes the line whenever either shape moves, which is what
    makes the label draggable without stranding its leader.

    A line's ``<a:ext>`` is always positive; direction is carried by
    ``flipH``/``flipV`` on the transform. Those values are the *initial*
    geometry -- Excel overwrites them once it recomputes the attached
    routing.
    """
    left, top = min(x1, x2), min(y1, y2)
    width = max(abs(x2 - x1), 1.0)
    height = max(abs(y2 - y1), 1.0)
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    connection_parts = []
    if start_shape_id is not None:
        connection_parts.append(
            f'<a:stCxn id="{start_shape_id}" idx="{start_idx}"/>'
        )
    if end_shape_id is not None:
        connection_parts.append(
            f'<a:endCxn id="{end_shape_id}" idx="{end_idx}"/>'
        )
    connection_body = "".join(connection_parts)
    cxn_pr = (
        f'<cNvCxnSpPr>{connection_body}</cNvCxnSpPr>'
        if connection_body
        else '<cNvCxnSpPr/>'
    )
    return (
        f'<cxnSp macro="">'
        f'<nvCxnSpPr>'
        f'<cNvPr id="{shape_id}" name="Leader {shape_id}"/>'
        f'{cxn_pr}'
        f'</nvCxnSpPr>'
        f'<spPr>'
        f'<a:xfrm{flip_h}{flip_v}>'
        f'<a:off x="{emu(left)}" y="{emu(top)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
        f'<a:ln w="{config.LEADER_WIDTH}">'
        f'<a:solidFill><a:srgbClr val="{config.LEADER_LINE}"/></a:solidFill>'
        f'<a:tailEnd type="none"/></a:ln>'
        f'</spPr></cxnSp>'
    )


def _dot_body(shape_id: int, cx: float, cy: float, radius: float) -> str:
    """Filled circle at the measurement point."""
    x = cx - radius
    y = cy - radius
    diameter = radius * 2
    return (
        f'<sp macro="" textlink="">'
        f'<nvSpPr>'
        f'<cNvPr id="{shape_id}" name="Point {shape_id}"/>'
        f'<cNvSpPr/></nvSpPr>'
        f'<spPr>'
        f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(diameter)}" cy="{emu(diameter)}"/></a:xfrm>'
        f'<a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{config.POINT_DOT_COLOR}"/></a:solidFill>'
        f'<a:ln><a:noFill/></a:ln>'
        f'</spPr></sp>'
    )


# --- anchors -------------------------------------------------------------

def text_box(
    shape_id: int, text: str, x: float, y: float,
    width: float | None = None, height: float | None = None,
    font_family: str | None = None,
) -> str:
    """Standalone text box anchor for a label.

    ``fLocksWithSheet="0"`` exempts the label from the sheet's object-level
    protection so users can still select, move, and text-edit it while
    the anchor dots and leader lines around it stay locked.
    """
    width = config.LABEL_WIDTH if width is None else width
    height = config.LABEL_HEIGHT if height is None else height
    body = _text_box_body(shape_id, text, x, y, width, height, font_family)
    return (
        f'<absoluteAnchor>'
        f'<pos x="{emu(x)}" y="{emu(y)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'{body}<clientData fLocksWithSheet="0"/></absoluteAnchor>'
    )


def leader(
    shape_id: int, x1: float, y1: float, x2: float, y2: float,
    *,
    start_shape_id: int | None = None, start_idx: int = 0,
    end_shape_id: int | None = None, end_idx: int = 0,
) -> str:
    """Standalone leader anchor. Pass ``start_shape_id`` / ``end_shape_id``
    to attach the connector so Excel re-routes it when either end moves."""
    left, top = min(x1, x2), min(y1, y2)
    width = max(abs(x2 - x1), 1.0)
    height = max(abs(y2 - y1), 1.0)
    body = _leader_body(
        shape_id, x1, y1, x2, y2,
        start_shape_id=start_shape_id, start_idx=start_idx,
        end_shape_id=end_shape_id, end_idx=end_idx,
    )
    return (
        f'<absoluteAnchor>'
        f'<pos x="{emu(left)}" y="{emu(top)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'{body}<clientData/></absoluteAnchor>'
    )


def dot(shape_id: int, cx: float, cy: float, radius: float) -> str:
    """Standalone anchor for a filled circle at a measurement point."""
    x = cx - radius
    y = cy - radius
    diameter = radius * 2
    body = _dot_body(shape_id, cx, cy, radius)
    return (
        f'<absoluteAnchor>'
        f'<pos x="{emu(x)}" y="{emu(y)}"/>'
        f'<ext cx="{emu(diameter)}" cy="{emu(diameter)}"/>'
        f'{body}<clientData/></absoluteAnchor>'
    )


def _hex_to_srgb(color: str, fallback: str = "E8802F") -> str:
    """Normalize a #rrggbb or #rgb string into DrawingML's 6-hex form."""
    value = (color or "").lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) == 6:
        try:
            int(value, 16)
        except ValueError:
            return fallback
        return value.upper()
    return fallback


def annotation_rect(
    shape_id: int, x: float, y: float, width: float, height: float,
    color: str, filled: bool = False, line_width_emu: int = 19050,
) -> str:
    """Bordered rectangle — the UI's rect / ellipse annotation shapes.

    ``filled=False`` mirrors the on-screen rectangle: outline only, so the
    part of the picture underneath stays visible.
    """
    rgb = _hex_to_srgb(color)
    fill = (
        f'<a:solidFill><a:srgbClr val="{rgb}"><a:alpha val="20000"/>'
        f'</a:srgbClr></a:solidFill>'
        if filled else '<a:noFill/>'
    )
    return (
        f'<absoluteAnchor><pos x="{emu(x)}" y="{emu(y)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'<sp macro="" textlink=""><nvSpPr>'
        f'<cNvPr id="{shape_id}" name="Annotation {shape_id}"/>'
        f'<cNvSpPr/></nvSpPr>'
        f'<spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'{fill}'
        f'<a:ln w="{line_width_emu}"><a:solidFill>'
        f'<a:srgbClr val="{rgb}"/></a:solidFill></a:ln>'
        f'</spPr></sp><clientData/></absoluteAnchor>'
    )


def annotation_ellipse(
    shape_id: int, x: float, y: float, width: float, height: float,
    color: str, line_width_emu: int = 19050,
) -> str:
    """Bordered ellipse annotation."""
    rgb = _hex_to_srgb(color)
    return (
        f'<absoluteAnchor><pos x="{emu(x)}" y="{emu(y)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'<sp macro="" textlink=""><nvSpPr>'
        f'<cNvPr id="{shape_id}" name="Annotation {shape_id}"/>'
        f'<cNvSpPr/></nvSpPr>'
        f'<spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>'
        f'<a:noFill/>'
        f'<a:ln w="{line_width_emu}"><a:solidFill>'
        f'<a:srgbClr val="{rgb}"/></a:solidFill></a:ln>'
        f'</spPr></sp><clientData/></absoluteAnchor>'
    )


def annotation_text(
    shape_id: int, text: str, x: float, y: float,
    width: float, height: float, color: str,
    font_size_pt: float,
) -> str:
    """Borderless text annotation. Colored text on transparent background."""
    rgb = _hex_to_srgb(color)
    size_hundredths = max(500, int(round(font_size_pt * 100)))
    return (
        f'<absoluteAnchor><pos x="{emu(x)}" y="{emu(y)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'<sp macro="" textlink=""><nvSpPr>'
        f'<cNvPr id="{shape_id}" name="Annotation {shape_id}"/>'
        f'<cNvSpPr txBox="1"/></nvSpPr>'
        f'<spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:noFill/><a:ln><a:noFill/></a:ln></spPr>'
        f'<txBody>'
        f'<a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"'
        f' anchor="t"/>'
        f'<a:lstStyle/><a:p>'
        f'<a:r><a:rPr lang="ko-KR" sz="{size_hundredths}" b="1">'
        f'<a:solidFill><a:srgbClr val="{rgb}"/></a:solidFill>'
        f'</a:rPr><a:t>{escape(text)}</a:t></a:r></a:p>'
        f'</txBody></sp><clientData fLocksWithSheet="0"/></absoluteAnchor>'
    )


def annotation_arrow(
    shape_id: int, x1: float, y1: float, x2: float, y2: float,
    color: str, line_width_emu: int = 19050,
) -> str:
    """Straight arrow with a triangular head at the (x2, y2) end."""
    rgb = _hex_to_srgb(color)
    left, top = min(x1, x2), min(y1, y2)
    width = max(abs(x2 - x1), 1.0)
    height = max(abs(y2 - y1), 1.0)
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    return (
        f'<absoluteAnchor><pos x="{emu(left)}" y="{emu(top)}"/>'
        f'<ext cx="{emu(width)}" cy="{emu(height)}"/>'
        f'<cxnSp macro=""><nvCxnSpPr>'
        f'<cNvPr id="{shape_id}" name="Annotation {shape_id}"/>'
        f'<cNvCxnSpPr/></nvCxnSpPr>'
        f'<spPr><a:xfrm{flip_h}{flip_v}>'
        f'<a:off x="{emu(left)}" y="{emu(top)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        f'<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
        f'<a:ln w="{line_width_emu}"><a:solidFill>'
        f'<a:srgbClr val="{rgb}"/></a:solidFill>'
        f'<a:tailEnd type="triangle" w="med" len="med"/>'
        f'</a:ln></spPr></cxnSp><clientData/></absoluteAnchor>'
    )


def caption(shape_id: int, text: str, x: float, y: float, width: float) -> str:
    """Borderless left-aligned caption, used for view titles."""
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


_DRAWINGML_MAIN_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def inject(drawing_xml: str, anchors: list[str]) -> str:
    """Append anchors just before the drawing part closes.

    openpyxl leaves ``xmlns:a`` off the root ``<wsDr>`` element and instead
    redeclares it on each ``<a:blip>``/``<a:prstGeom>`` inside its own
    picture anchors. Our injected text boxes, dots, and connectors use
    ``<a:xfrm>``, ``<a:off>``, ``<a:solidFill>``… without redeclaring the
    prefix. Excel tolerates the omission on a freshly built single-block
    file but treats the merged (multi-block) file as corrupt on open, so
    the prefix is declared on the root here where it covers every case.

    openpyxl also omits ``<pic><spPr><a:xfrm>``, which recent Excel builds
    treat as "picture has no placement" and render as a blank
    "cannot display picture" placeholder even though the outer anchor's
    ``<ext>`` is set. ``_add_picture_xfrm`` copies the anchor's position
    and extent into the picture body so the shape gets its own geometry
    and displays normally.
    """
    xml = _ensure_root_ns(drawing_xml)
    xml = _add_picture_xfrm(xml)
    if not anchors:
        return xml
    if "</wsDr>" not in xml:
        raise ValueError("drawing XML의 닫는 태그를 찾지 못했습니다.")
    return xml.replace("</wsDr>", "".join(anchors) + "</wsDr>")


def _add_picture_xfrm(drawing_xml: str) -> str:
    """Give every ``<pic>`` its own ``<a:xfrm>`` inside ``<spPr>``.

    The transform mirrors the enclosing ``<absoluteAnchor>``'s ``<pos>``
    and ``<ext>``; from Excel's point of view the two must agree, which
    is trivially true here because both sides come from the same source.
    """
    pattern = re.compile(
        r'(<absoluteAnchor>\s*<pos x="(\d+)" y="(\d+)"/>\s*<ext cx="(\d+)" cy="(\d+)"/>'
        r'\s*<pic\b.*?<spPr>)(\s*)(<a:prstGeom\b.*?</pic>\s*<clientData/>\s*</absoluteAnchor>)',
        re.DOTALL,
    )

    def _replace(match: re.Match[str]) -> str:
        head, x, y, cx, cy, gap, tail = match.groups()
        if '<a:xfrm' in head:
            return match.group(0)
        xfrm = (
            f'<a:xfrm><a:off x="{x}" y="{y}"/>'
            f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        )
        return f'{head}{xfrm}{gap}{tail}'

    return pattern.sub(_replace, drawing_xml)


def _ensure_root_ns(drawing_xml: str) -> str:
    """Declare ``xmlns:a`` on ``<wsDr>`` if openpyxl left it off."""
    match = re.match(r"(<wsDr\b)([^>]*)(>)", drawing_xml)
    if match is None:
        return drawing_xml
    open_tag_attrs = match.group(2)
    if 'xmlns:a=' in open_tag_attrs:
        return drawing_xml
    new_attrs = f'{open_tag_attrs} xmlns:a="{_DRAWINGML_MAIN_NS}"'
    return f"{match.group(1)}{new_attrs}{match.group(3)}{drawing_xml[match.end():]}"


DRAWING_PART = re.compile(r"xl/drawings/drawing\d+\.xml$")
