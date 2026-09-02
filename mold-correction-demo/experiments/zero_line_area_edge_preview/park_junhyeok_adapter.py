"""Adapter between the unified detector and Park Junhyeok's route selector.

The selector module is kept as the original team source.  This module owns
integration-only policy such as Boolean-mask conversion and the 100 px
minimum final route length.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import cv2

import park_junhyeok_route_selector as selector
from park_junhyeok_original import contour_graph
from park_junhyeok_original import merge_correction_regions
from park_junhyeok_original import out_of_tolerance
from park_junhyeok_original import remove_labels
from park_junhyeok_original import zero_point_selection


DEFAULT_MINIMUM_ROUTE_LENGTH_PX = 100.0


def _flatten_zero_points(contours: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    sequence = 1
    for contour in contours:
        if contour.get("kind") != "outer":
            continue
        for item in zero_point_selection.select_contour_zero_points(contour):
            if "sample_index" in item:
                sample_index = int(item["sample_index"])
            else:
                before, after = item["between_sample_indices"]
                ratio = float(item.get("interpolation_ratio", 0.5))
                sample_index = int(before if ratio < 0.5 else after)
            x, y = item["point"]
            points.append(
                {
                    "label": f"Z{sequence}",
                    "point": (float(x), float(y)),
                    "type": item.get("type", "unknown"),
                    "contour": contour.get("name", "unknown"),
                    "sample_index": sample_index,
                }
            )
            sequence += 1
    if len(points) < 2:
        raise ValueError("Park Junhyeok pipeline requires at least two zero points")
    return points


def _outer_geometry(
    contours: list[dict[str, Any]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    product_mask = np.zeros(shape, dtype=np.uint8)
    outer_silhouette = np.zeros(shape, dtype=np.uint8)
    outer_points: np.ndarray | None = None
    for contour in contours:
        points = np.rint(
            np.asarray(
                [sample["contour_point"] for sample in contour.get("samples", [])],
                dtype=np.float64,
            )
        ).astype(np.int32)
        if len(points) < 3:
            continue
        if contour.get("kind") == "outer":
            cv2.fillPoly(product_mask, [points], 255, cv2.LINE_8)
            cv2.fillPoly(outer_silhouette, [points], 255, cv2.LINE_8)
            if outer_points is None:
                outer_points = np.asarray(
                    [
                        sample["contour_point"]
                        for sample in contour.get("samples", [])
                    ],
                    dtype=np.float64,
                )
        else:
            cv2.fillPoly(product_mask, [points], 0, cv2.LINE_8)
    if outer_points is None or len(outer_points) < 3:
        raise ValueError("No Park Junhyeok outer contour was generated")
    return outer_points, product_mask, outer_silhouette


def _route_length(selection: dict[str, Any]) -> float:
    route = selection.get("closure_validation", {}).get("route", {})
    return float(route.get("path_length_pixels", 0.0))


def run_route_selector(
    *,
    correction_mask: np.ndarray,
    zero_points: list[dict[str, Any]],
    contour_points: np.ndarray,
    outer_silhouette: np.ndarray,
    minimum_route_length_px: float = DEFAULT_MINIMUM_ROUTE_LENGTH_PX,
) -> dict[str, Any]:
    """Convert shared inputs, call the original selector, and apply engine policy."""
    correction_u8 = np.asarray(correction_mask, dtype=bool).astype(np.uint8) * 255
    silhouette_u8 = np.asarray(outer_silhouette, dtype=bool).astype(np.uint8) * 255
    regions = selector.extract_regions(correction_u8)
    if not regions:
        raise ValueError("No correction regions are available for case 2")

    selected, contact_radius, route_clearance = selector.select_along_outer_contour(
        regions,
        zero_points,
        np.asarray(contour_points, dtype=np.float64),
        correction_u8.shape,
        correction_u8,
        silhouette_u8,
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for selection in selected:
        length = _route_length(selection)
        if np.isfinite(length) and length >= minimum_route_length_px:
            accepted.append(selection)
            continue
        region = selection.get("region", {})
        rejected.append(
            {
                "region_label": region.get("label"),
                "path_length_pixels": length,
                "reason": "route_shorter_than_minimum",
            }
        )

    return {
        "selections": accepted,
        "rejected_routes": rejected,
        "contact_radius_px": contact_radius,
        "route_clearance_px": route_clearance,
        "source_correction_region_count": len(regions),
        "minimum_route_length_px": float(minimum_route_length_px),
    }


def run_original_case2_pipeline(
    *,
    original_bgr: np.ndarray,
    scale_max_mm: float,
    minimum_route_length_px: float = DEFAULT_MINIMUM_ROUTE_LENGTH_PX,
) -> dict[str, Any]:
    """Run Park Junhyeok's stages 01-06 in memory as one engine branch."""
    versions = remove_labels.create_versions(original_bgr)
    cleaned = versions["4_labels_points_inpainted"]

    product_mask = contour_graph.build_product_mask(cleaned)
    contours = contour_graph.extract_true_contours(product_mask)
    if not contours:
        raise ValueError("Park Junhyeok pipeline found no product contour")
    colorbar = contour_graph.detect_colorbar(original_bgr)
    _, contour_payload = contour_graph.render_graph(
        cleaned,
        product_mask,
        contours,
        colorbar,
    )
    zero_points = _flatten_zero_points(contour_payload)
    contour_points, route_product_mask, outer_silhouette = _outer_geometry(
        contour_payload,
        cleaned.shape[:2],
    )

    source_bar = out_of_tolerance.locate_colorbar(original_bgr)
    positive_hue = out_of_tolerance.sample_bar_hue(
        original_bgr,
        source_bar,
        out_of_tolerance.TOLERANCE_MM,
        scale_max_mm,
    )
    negative_hue = out_of_tolerance.sample_bar_hue(
        original_bgr,
        source_bar,
        -out_of_tolerance.TOLERANCE_MM,
        scale_max_mm,
    )
    positive, negative, gray = out_of_tolerance.build_correction_masks(
        cleaned,
        positive_hue,
        negative_hue,
    )
    source_correction = cv2.bitwise_or(cv2.bitwise_or(positive, negative), gray)
    merge_product_mask = merge_correction_regions.build_product_mask(cleaned)
    merged_correction, merge_details = merge_correction_regions.merge_and_filter(
        source_correction,
        merge_product_mask,
    )

    routed = run_route_selector(
        correction_mask=merged_correction,
        zero_points=zero_points,
        contour_points=contour_points,
        outer_silhouette=outer_silhouette,
        minimum_route_length_px=minimum_route_length_px,
    )
    return {
        **routed,
        "cleaned_bgr": cleaned,
        "contour_points": contour_points,
        "product_mask": route_product_mask,
        "outer_silhouette": outer_silhouette,
        "zero_points": zero_points,
        "source_correction_mask": source_correction,
        "merged_correction_mask": merged_correction,
        "merge_details": merge_details,
        "tolerance_mm": float(out_of_tolerance.TOLERANCE_MM),
        "positive_hue_limit": float(positive_hue),
        "negative_hue_limit": float(negative_hue),
    }
