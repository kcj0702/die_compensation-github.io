"""Label removal that keeps the product outline, with or without the CAD shape.

Removing labels and leader lines from an inspection export should leave the part exactly as it was.
Two things used to go wrong:

* An annotation that crosses or overlaps the part edge is cut out, and the edge behind it is lost
  (the part gets a white notch or a straight cut where the label was).
* Clipping the part with a projected CAD mask cuts real edge pixels wherever the registration is
  off by a pixel or two (scan 2 of zeroline_datas lost 18 % of its edge band that way).
* Leader lines of upscaled images are blurred, so the exact-blue tracer misses them and they stay.

What this module does
---------------------
1. Annotations: label boxes (remove_labels.detect_label_boxes), exact-blue leaders
   (detect_exact_hsv_leader_lines), blurred / thick blue leaders touching a label box, and the
   measurement markers at leader ends (build_measurement_point_mask). On the part, blurred blue is
   a leader only on the straight continuation of a leader seen off the part (leader_corridors), so
   a blue deviation rim along the edge is kept.
2. Product mask, image only: remove_labels.build_scan_mask with an opening sized to the measured
   leader width (scan_mask_and_core), so leaders of upscaled exports detach as well. Leaders and
   markers are part where they lie on the dense core; a label box is part where the part edge,
   continued through the box from where it enters and leaves, encloses it (_bridge_annotations).
3. Product mask with CAD: the CAD silhouette (the STEP file meshed and projected into the image with
   the scan registration) decides the shape. The image-only mask and image pixels within
   ``tolerance_px`` of the silhouette are kept - real edge pixels are never cut for a one-pixel
   registration error - annotation pixels inside the silhouette are restored, and everything
   farther outside is background.
4. Annotation pixels inside the product are inpainted from nearby scan colours; the rest is white.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:  # package import (tests, pipeline) or run from inside label_removal/
    from label_removal import remove_labels as base
except ImportError:  # pragma: no cover
    import remove_labels as base


@dataclass
class CleaningResult:
    image: np.ndarray                 # cleaned BGR image
    product_mask: np.ndarray          # bool, part pixels in the cleaned image
    annotation_mask: np.ndarray       # bool, everything recognised as annotation
    restored_mask: np.ndarray         # bool, annotation pixels given back to the part and inpainted
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------- CAD silhouette
def triangles_from_step(path: str | Path, deflection_mm: float = 0.3) -> np.ndarray:
    """Triangles (N, 3, 3) in mm meshed from a STEP (.stp/.step) file with OpenCASCADE (OCP).

    The silhouette must come from the same CAD the scan was registered with, so the STEP is read
    directly rather than a separately exported mesh, which may hold another part version or a
    mirrored left/right pair.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    path = Path(path)
    if path.suffix.lower() not in (".stp", ".step") or not path.is_file():
        raise FileNotFoundError(f"STEP file not found: {path}")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"STEP reader failed: {path}")
    reader.SetSystemLengthUnit(1.0)
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"STEP holds no shape: {path}")
    BRepMesh_IncrementalMesh(shape, float(deflection_mm), False, 0.35, True)
    pieces: list[np.ndarray] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        explorer.Next()
        if triangulation is None or triangulation.NbTriangles() == 0:
            continue
        transform = location.Transformation()
        nodes = np.empty((triangulation.NbNodes(), 3))
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(transform)
            nodes[index - 1] = (point.X(), point.Y(), point.Z())
        faces = np.empty((triangulation.NbTriangles(), 3), dtype=np.int64)
        for index in range(1, triangulation.NbTriangles() + 1):
            faces[index - 1] = triangulation.Triangle(index).Get()
        pieces.append(nodes[faces - 1])
    if not pieces:
        raise ValueError(f"STEP could not be meshed: {path}")
    return np.concatenate(pieces, axis=0)


def mesh_silhouette(
    triangles_xyz: np.ndarray,
    cad_xy_to_image_uv_matrix: np.ndarray,
    plane_axes: tuple[int, int] | list[int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Pixels covered by the mesh seen along the registration's view: the exact part shadow."""
    from label_removal.cad_outline import silhouette_from_pixels

    matrix = np.asarray(cad_xy_to_image_uv_matrix, dtype=np.float64)
    plane = list(plane_axes)
    flat = np.asarray(triangles_xyz, dtype=np.float64)[:, :, plane].reshape(-1, 2)
    uv = flat @ matrix[:, :2].T + matrix[:, 2]
    faces = np.arange(len(uv), dtype=np.int64).reshape(-1, 3)
    return silhouette_from_pixels(uv, faces, image_shape[:2])


def mesh_matches_points(triangles_xyz: np.ndarray, reference_points_xyz: np.ndarray, tolerance_mm: float = 5.0) -> dict:
    """Does the meshed STEP cover the same part as the points the registration was made with?

    Bounding boxes are compared on every axis. A STEP holding another part version, or a mirrored
    left/right pair, fails, and its silhouette must not be used.
    """
    vertices = np.asarray(triangles_xyz, dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(reference_points_xyz, dtype=np.float64)
    low_gap = np.abs(vertices.min(axis=0) - reference.min(axis=0))
    high_gap = np.abs(vertices.max(axis=0) - reference.max(axis=0))
    worst = float(max(low_gap.max(), high_gap.max()))
    return {"matches": worst <= tolerance_mm, "worst_bbox_gap_mm": worst,
            "min_gap_mm": low_gap.round(2).tolist(), "max_gap_mm": high_gap.round(2).tolist()}


# ---------------------------------------------------------------- annotations
def blurred_leader_lines(
    image: np.ndarray, label_boxes: list[tuple[int, int, int, int]], max_half_width_px: float = 4.0
) -> np.ndarray:
    """Blue-dominant thin strokes that touch a label box, including blurred / resampled ones.

    The exact-blue tracer needs pure (255, 0, 0) pixels, which an upscaled export no longer has.
    Wide blue areas belong to the deviation map, so only the thin part of a blue component is kept.
    """
    height, width = image.shape[:2]
    blue, green, red = (image[:, :, channel].astype(np.int16) for channel in range(3))
    bluish = ((blue >= 120) & (blue >= red + 50) & (blue >= green + 50)).astype(np.uint8)
    if not bluish.any() or not label_boxes:
        return np.zeros((height, width), dtype=bool)
    wide_size = int(2 * np.ceil(max_half_width_px) + 3)
    wide = cv2.morphologyEx(bluish, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wide_size, wide_size)))
    thin = bluish & (cv2.dilate(wide, np.ones((3, 3), np.uint8)) == 0)
    reach = max(8, int(round(min(height, width) * 0.012)))
    near_box = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in label_boxes:
        near_box[max(0, y0 - reach):min(height, y1 + reach), max(0, x0 - reach):min(width, x1 + reach)] = True
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), connectivity=8)
    touching = np.unique(labels[near_box & (labels > 0)])
    lines = np.isin(labels, touching[touching > 0])
    lines &= np.isin(labels, np.flatnonzero(np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= 6))
    # Anti-aliased fringe beside the stroke.
    fringe = ((blue >= green + 15) & (blue >= red + 15)) | bluish.astype(bool)
    return (cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & fringe) | lines


def estimate_leader_width(image: np.ndarray) -> float:
    """Typical stroke width (px) of the blue leader lines; about 1-2 px in a native export."""
    blue, green, red = (image[:, :, channel].astype(np.int16) for channel in range(3))
    bluish = ((blue >= 120) & (blue >= red + 50) & (blue >= green + 50)).astype(np.uint8)
    if bluish.sum() < 50:
        return 2.0
    distance = cv2.distanceTransform(bluish, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ridge = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))
    values = distance[ridge]
    values = values[values <= 6.0]                    # wide blue areas are deviation colour, not strokes
    # The centre pixel of a w px stroke lies (w + 1) / 2 from the nearest pixel outside it.
    return float(2.0 * np.median(values) - 1.0) if values.size else 2.0


def scan_mask_and_core(image: np.ndarray, stroke_width_px: float, white_level: int = 235) -> tuple[np.ndarray, np.ndarray]:
    """(scan mask, dense core): remove_labels.build_scan_mask with an opening that fits the leaders.

    The fixed opening (0.6 % of the short side) is 5 px on an upscaled 578 x 338 export, where the
    leaders are ~5 px wide, so every leader stayed attached to the part. The core is the opened
    part: it holds the real outline, but no thin stroke lying on the white background.
    """
    height, width = image.shape[:2]
    size = max(5, int(round(min(height, width) * 0.006)), int(np.ceil(stroke_width_px)) * 2 + 1)
    size += 1 - size % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    foreground = image.min(axis=2) < white_level
    core = base.largest_component(cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_OPEN, kernel)) > 0
    restored = cv2.dilate(core.astype(np.uint8), kernel).astype(bool) & foreground
    scan = (cv2.dilate(restored.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & foreground) | restored
    return scan, core


def scan_mask_for_width(image: np.ndarray, stroke_width_px: float, white_level: int = 235) -> np.ndarray:
    return scan_mask_and_core(image, stroke_width_px, white_level)[0]


def leader_corridors(strokes: np.ndarray, off_part: np.ndarray, stroke_width_px: float, reach_px: float) -> np.ndarray:
    """Straight continuations of the leader strokes seen off the part, reach_px beyond both ends.

    A leader is a straight line from its label to the measured point, so on the part only the
    pixels on that line are leader. A blue rim along the part edge is thin and blue as well, but
    does not lie on any leader's line and so is not taken for one.
    """
    height, width = strokes.shape
    corridor = np.zeros((height, width), dtype=np.uint8)
    thickness = int(np.ceil(stroke_width_px)) + 3
    count, labels, stats, _ = cv2.connectedComponentsWithStats((strokes & off_part).astype(np.uint8), connectivity=8)
    for label in range(1, count):
        if max(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]) < 10:
            continue
        ys, xs = np.nonzero(labels == label)
        points = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        along = (xs - x0) * vx + (ys - y0) * vy
        across = np.abs((xs - x0) * vy - (ys - y0) * vx)
        if np.percentile(across, 90) > stroke_width_px + 2:     # not a straight stroke
            continue
        start, end = along.min() - reach_px, along.max() + reach_px
        cv2.line(corridor, (int(round(x0 + start * vx)), int(round(y0 + start * vy))),
                 (int(round(x0 + end * vx)), int(round(y0 + end * vy))), 1, thickness)
    return corridor.astype(bool)


def detect_annotations(
    image: np.ndarray,
    off_part: np.ndarray | None = None,
    stroke_width_px: float | None = None,
    *,
    context: dict | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """(annotation mask, report, layers): label boxes, leader lines and measurement markers.

    off_part: pixels surely outside the part. Blurred blue strokes there are leaders; on the part
    only their straight continuations are (leader_corridors) - a blue deviation rim is thin and blue
    too. layers holds "boxes" (wide), "thin" (leaders and markers) and "points" (markers only)
    separately - remove_labels keeps the markers in versions 1 and 2 and drops them in 3 and 4.
    context: filled with "label_boxes" and "deviation_candidates" like remove_labels.create_versions.
    """
    height, width = image.shape[:2]
    scan_mask = base.build_scan_mask(image)
    boxes = base.detect_label_boxes(image)
    exact_lines, _ = base.detect_exact_hsv_leader_lines(image, boxes, scan_mask)
    soft_lines = blurred_leader_lines(image, boxes)
    if off_part is not None:
        stroke = estimate_leader_width(image) if stroke_width_px is None else stroke_width_px
        off_part = off_part.astype(bool)
        soft_lines &= off_part | leader_corridors(soft_lines, off_part, stroke, reach_px=3 * stroke + 6)
    candidates = None
    try:
        try:
            from label_detector import detect_labels
        except ImportError:
            from deviation_extraction.label_detector import detect_labels
        candidates = detect_labels(image)
        points = base.build_measurement_point_mask(image, scan_mask, boxes, candidates) > 0
    except Exception:  # the optional label detector is unavailable or failed on this export
        points = np.zeros((height, width), dtype=bool)
    if context is not None:
        context["label_boxes"] = boxes
        if candidates is not None:
            context["deviation_candidates"] = candidates
    box_mask = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        box_mask[y0:y1, x0:x1] = True
    thin = ((exact_lines > 0) | soft_lines | points) & ~box_mask
    annotation = box_mask | thin
    report = {"label_boxes": len(boxes), "exact_leader_px": int((exact_lines > 0).sum()),
              "blurred_leader_px": int(soft_lines.sum()), "marker_px": int(points.sum())}
    return annotation, report, {"boxes": box_mask, "thin": thin, "points": points & ~box_mask}


# ---------------------------------------------------------------- product mask
def _largest(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        raise ValueError("no product region found")
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _edge_direction(edge: np.ndarray, point: np.ndarray, inward: np.ndarray, radius: int) -> np.ndarray | None:
    """Unit direction of the part edge near point, pointing into the covered component."""
    ys, xs = np.nonzero(edge)
    near = (xs - point[0]) ** 2 + (ys - point[1]) ** 2 <= radius * radius
    if near.sum() < 4:
        return None
    vx, vy, _, _ = cv2.fitLine(np.column_stack([xs[near], ys[near]]).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    direction = np.array([vx, vy], dtype=np.float64)
    return direction if direction @ inward > 0 else -direction


def _continue_edge(start, end, d_start, d_end, box) -> list:
    """Path from start to end inside the box: the two edges extended to where they meet, else a chord."""
    if d_start is None or d_end is None:
        return []
    matrix = np.column_stack([d_start, -d_end])
    if abs(np.linalg.det(matrix)) < 0.25:                      # nearly parallel: the edge runs straight on
        return []
    t_start, t_end = np.linalg.solve(matrix, np.asarray(end, float) - np.asarray(start, float))
    corner = np.asarray(start, float) + t_start * d_start
    x0, y0, x1, y1 = box
    if t_start <= 0 or t_end <= 0 or not (x0 - 2 <= corner[0] <= x1 + 2 and y0 - 2 <= corner[1] <= y1 + 2):
        return []
    return [corner]


def _bridge_annotations(
    body: np.ndarray, annotation: np.ndarray, close_px: int, hull_margin_px: int, background: np.ndarray | None = None
) -> np.ndarray:
    """Part pixels hidden under annotations: thin crossings by closing, wide ones by edge continuation.

    Walking once around just outside a wide annotation (a label box), every stretch of that ring
    lying on the part is kept, and between the stretch where the part leaves the ring and the one
    where it comes back the part edge is continued into the box: the two edge directions are
    extended to the corner where they meet, or joined straight when they run on in one line. A
    straight edge stays straight, a part corner or hole corner under a label is completed, and a
    flange passing under a label keeps both of its edges.
    background: pixels surely not part; ring pixels that are neither part nor background (another
    label, a leader) take the class of the nearest classified ring pixel.
    """
    closed = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))).astype(bool)
    restored = closed & annotation
    height, width = body.shape
    if background is None:
        background = ~body & ~annotation
    count, labels, stats, _ = cv2.connectedComponentsWithStats(annotation.astype(np.uint8), connectivity=8)
    touching_body = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for label in range(1, count):
        x, y, w, h = stats[label, :4]
        if w < close_px and h < close_px:
            continue
        x0, y0 = max(0, x - hull_margin_px), max(0, y - hull_margin_px)
        x1, y1 = min(width, x + w + hull_margin_px), min(height, y + h + hull_margin_px)
        component = labels[y0:y1, x0:x1] == label
        if not (component & touching_body[y0:y1, x0:x1]).any():
            continue
        local_body, local_background = body[y0:y1, x0:x1], background[y0:y1, x0:x1]
        grown = cv2.dilate(component.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ring = max(contours, key=len)[:, 0, :]
        kind = np.where(local_body[ring[:, 1], ring[:, 0]], 1, np.where(local_background[ring[:, 1], ring[:, 0]], 0, -1))
        if not (kind == 1).any():
            continue
        if not (kind == 0).any():
            restored[y0:y1, x0:x1] |= component
            continue
        known = np.flatnonzero(kind >= 0)
        if (kind < 0).any():
            n = len(kind)
            unknown = np.flatnonzero(kind < 0)
            gaps = np.abs(unknown[:, None] - known[None, :])
            kind[unknown] = kind[known[np.argmin(np.minimum(gaps, n - gaps), axis=1)]]
        # Stretches on the part, in walking order, starting just after a background pixel.
        shift = int(np.flatnonzero(kind == 0)[0])
        order = np.roll(np.arange(len(kind)), -shift)
        runs, current = [], []
        for index in order:
            if kind[index] == 1:
                current.append(index)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        runs = [run for run in runs if len(run) >= 3]
        if not runs:
            continue
        edge = local_body & ~cv2.erode(local_body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        edge &= ~cv2.dilate(grown, np.ones((3, 3), np.uint8)).astype(bool)
        centre = np.array([x - x0 + w / 2.0, y - y0 + h / 2.0])
        box = (x - x0, y - y0, x - x0 + w - 1, y - y0 + h - 1)
        polygon = []
        for number, run in enumerate(runs):
            polygon.extend(ring[run].astype(np.float64))
            leave, back = ring[run[-1]].astype(float), ring[runs[(number + 1) % len(runs)][0]].astype(float)
            polygon.extend(_continue_edge(leave, back,
                                          _edge_direction(edge, leave, centre - leave, hull_margin_px),
                                          _edge_direction(edge, back, centre - back, hull_margin_px), box))
        fill = np.zeros(component.shape, dtype=np.uint8)
        cv2.fillPoly(fill, [np.round(np.asarray(polygon)).astype(np.int32)], 1)
        restored[y0:y1, x0:x1] |= component & fill.astype(bool)
    return restored


def product_mask_from_image(
    image: np.ndarray,
    layers: dict,
    scan_mask: np.ndarray,
    core: np.ndarray,
    *,
    close_px: int = 7,
    hull_margin_px: int = 12,
    min_piece_px: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """(product mask, restored annotation pixels) from the image alone.

    scan_mask (scan_mask_and_core) already cut unattached leaders off, as remove_labels always did,
    and keeps the whole outline. Within it:
    * leaders and markers are part where they lie on the dense core (a leader end or a marker on
      the part, also right at the edge) and background where they stick out of it (leader stubs);
    * a label box is part only where the part surrounds it (one local hull), so a box overlapping
      the edge neither notches the edge nor stays as a bump.
    Every sizeable piece is kept: a leader crossing a flange must not make the smaller side disappear.
    """
    scan = scan_mask.astype(bool)
    near_core = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # The whole box, its white text included: over the part that text is part too.
    boxes = layers["boxes"]
    box_guard = cv2.dilate(boxes.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (scan | boxes)
    body = scan & ~box_guard & ~(layers["thin"] & ~near_core)
    white = ~scan & ~cv2.dilate((box_guard | layers["thin"]).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    restored = _bridge_annotations(body, box_guard, close_px, hull_margin_px, background=white)
    product = body | restored
    # Box fringe pixels that continue the part rather than the box.
    fringe = box_guard & ~boxes
    product |= fringe & cv2.dilate(product.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & near_core
    count, labels, stats, _ = cv2.connectedComponentsWithStats(product.astype(np.uint8), connectivity=8)
    if count > 1:
        big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_piece_px)
        product &= np.isin(labels, big[big > 0])
    return product, (restored | (layers["thin"] & near_core)) & product


def product_mask_with_cad(
    image: np.ndarray,
    annotation: np.ndarray,
    cad_silhouette: np.ndarray,
    *,
    image_product: np.ndarray | None = None,
    tolerance_px: int = 3,
    white_level: int = 235,
) -> tuple[np.ndarray, np.ndarray]:
    """(product mask, restored annotation pixels) with the CAD silhouette deciding the shape.

    image_product (product_mask_from_image), when given, is the starting point: the CAD then only
    adds the part pixels under annotations and removes whatever lies beyond tolerance_px of the
    silhouette, so the result never loses more outline than the image alone.
    """
    silhouette = cad_silhouette.astype(bool)
    size = 2 * int(tolerance_px) + 1
    near = cv2.dilate(silhouette.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))).astype(bool)
    foreground = image.min(axis=2) < white_level
    guard = cv2.dilate(annotation.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    image_body = foreground & ~guard & near
    image_body = cv2.morphologyEx(image_body.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    if image_product is not None:
        image_body |= image_product.astype(bool) & near
    # Annotation over the part: the CAD says where the part is, so give those pixels back.
    restored = guard & silhouette
    product = image_body | restored
    count, labels, stats, _ = cv2.connectedComponentsWithStats(product.astype(np.uint8), connectivity=8)
    if count > 1:
        # drop specks that survive outside the part body (e.g. a label tail inside the tolerance band)
        big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(50, 0.001 * silhouette.sum()))
        product &= np.isin(labels, big[big > 0])
    return product, restored


# ---------------------------------------------------------------- whole image
def analyse_scan(
    image: np.ndarray,
    *,
    cad_silhouette: np.ndarray | None = None,
    tolerance_px: int = 3,
    context: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """(product, annotation, restored, layers, report) for this scan."""
    stroke = estimate_leader_width(image)
    scan, core = scan_mask_and_core(image, stroke)
    if cad_silhouette is not None:
        off_part = ~cv2.dilate(cad_silhouette.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        annotation, report, layers = detect_annotations(image, off_part, stroke, context=context)
        image_product, _ = product_mask_from_image(image, layers, scan, core)
        product, restored = product_mask_with_cad(image, annotation, cad_silhouette,
                                                  image_product=image_product, tolerance_px=tolerance_px)
        report["mode"] = "cad_silhouette"
    else:
        annotation, report, layers = detect_annotations(image, ~scan, stroke, context=context)
        product, restored = product_mask_from_image(image, layers, scan, core)
        report["mode"] = "image_only"
    report["leader_width_px"] = round(stroke, 2)
    report["product_px"] = int(product.sum())
    report["restored_px"] = int(restored.sum())
    return product, annotation, restored, layers, report


def _rim(mask: np.ndarray) -> np.ndarray:
    """The anti-aliased fringe around an annotation, which must not be an inpainting colour source."""
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)


def remove_labels_preserving_outline(
    image: np.ndarray,
    *,
    cad_silhouette: np.ndarray | None = None,
    tolerance_px: int = 3,
) -> CleaningResult:
    """Cleaned image (labels, leaders and markers removed, outline kept)."""
    product, annotation, restored, _, report = analyse_scan(
        image, cad_silhouette=cad_silhouette, tolerance_px=tolerance_px)
    inpaint = (_rim(annotation) | restored) & product
    cleaned = base.render_inpaint_version(image, product.astype(np.uint8) * 255, inpaint.astype(np.uint8) * 255)
    report["inpainted_px"] = int(inpaint.sum())
    return CleaningResult(cleaned, product, annotation, restored, report)


def create_versions_preserving_outline(
    image: np.ndarray,
    *,
    cad_silhouette: np.ndarray | None = None,
    tolerance_px: int = 3,
    context: dict | None = None,
    report: dict | None = None,
) -> dict[str, np.ndarray]:
    """remove_labels.create_versions with the product outline kept: same four keys, same meaning.

    Versions 1 and 2 remove the labels and leaders but keep the measurement markers, versions 3 and
    4 remove the markers too - the pixel difference between 2 and 4 is where the markers were, which
    the caller reads deviation-point centres from. Outside the product every version is pure white,
    which is what the zero-line part detection looks for.
    """
    product, annotation, restored, layers, made = analyse_scan(
        image, cad_silhouette=cad_silhouette, tolerance_px=tolerance_px, context=context)
    if report is not None:
        report.update(made)
    scan_mask = product.astype(np.uint8) * 255
    points = layers["points"] & product
    labels_only = annotation & ~layers["points"]
    labels_white_mask = ((_rim(labels_only) & ~points) | restored) & product
    labels_and_points_mask = (_rim(annotation) | restored) & product

    labels_white = base.render_white_version(image, scan_mask, labels_white_mask.astype(np.uint8) * 255)
    labels_inpainted = base.render_inpaint_version(image, scan_mask, labels_white_mask.astype(np.uint8) * 255)
    labels_points_white = labels_white.copy()
    labels_points_white[points] = 255
    labels_points_inpainted = base.render_inpaint_version(image, scan_mask, labels_and_points_mask.astype(np.uint8) * 255)
    return {
        "1_labels_white": labels_white,
        "2_labels_inpainted": labels_inpainted,
        "3_labels_points_white": labels_points_white,
        "4_labels_points_inpainted": labels_points_inpainted,
    }


__all__ = [
    "CleaningResult",
    "analyse_scan",
    "blurred_leader_lines",
    "create_versions_preserving_outline",
    "detect_annotations",
    "estimate_leader_width",
    "leader_corridors",
    "mesh_matches_points",
    "mesh_silhouette",
    "product_mask_from_image",
    "product_mask_with_cad",
    "remove_labels_preserving_outline",
    "scan_mask_and_core",
    "scan_mask_for_width",
    "triangles_from_step",
]
