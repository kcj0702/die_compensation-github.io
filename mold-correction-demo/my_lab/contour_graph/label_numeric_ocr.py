"""Small offline OCR specialized for decimal deviation labels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = tuple(
    path
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    )
    if path.exists()
)
NORMALIZED_HEIGHT = 34
CANVAS_SHAPE = (46, 104)


def extract_text_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract white-on-red or dark-on-light numeric text from a label box."""
    blue, green, red = cv2.split(crop_bgr)
    red_fill = (red >= 210) & (green <= 90) & (blue <= 90)
    is_red_box = np.count_nonzero(red_fill) >= crop_bgr.shape[0] * crop_bgr.shape[1] * 0.20
    if is_red_box:
        # Keep anti-aliased white punctuation as well as the solid strokes.
        mask = (
            (blue >= 110) & (green >= 110) & (red >= 180)
        ).astype(np.uint8) * 255
    else:
        mask = (crop_bgr.max(axis=2) <= 105).astype(np.uint8) * 255

    border = max(2, int(round(min(crop_bgr.shape[:2]) * 0.10)))
    mask[:border] = 0
    mask[-border:] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if area < 2 or width > crop_bgr.shape[1] * 0.55:
            continue
        cleaned[labels == component] = 255
    return cleaned


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.where(mask > 0)
    if not len(xx):
        return np.zeros((1, 1), dtype=np.uint8)
    return mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    tight = _tight_crop(mask)
    scale = NORMALIZED_HEIGHT / max(tight.shape[0], 1)
    width = max(1, int(round(tight.shape[1] * scale)))
    width = min(width, CANVAS_SHAPE[1] - 4)
    resized = cv2.resize(
        tight, (width, NORMALIZED_HEIGHT), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros(CANVAS_SHAPE, dtype=np.uint8)
    y = (CANVAS_SHAPE[0] - NORMALIZED_HEIGHT) // 2
    x = (CANVAS_SHAPE[1] - width) // 2
    canvas[y : y + NORMALIZED_HEIGHT, x : x + width] = resized
    return canvas


def _render_template(text: str, font_path: Path, font_size: int = 30) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), font_size)
    image = Image.new("L", (180, 70), 0)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((8 - bbox[0], 6 - bbox[1]), text, font=font, fill=255)
    return normalize_mask(np.asarray(image, dtype=np.uint8))


@lru_cache(maxsize=1)
def _templates() -> tuple[tuple[float, str, np.ndarray], ...]:
    templates: list[tuple[float, str, np.ndarray]] = []
    fonts = FONT_PATHS or (Path(r"C:\Windows\Fonts\arial.ttf"),)
    for tenth in range(0, 51):
        value = tenth / 10.0
        text = f"{value:.1f}"
        for font_path in fonts:
            templates.append((value, font_path.name, _render_template(text, font_path)))
    return tuple(templates)


@lru_cache(maxsize=1)
def _digit_templates() -> tuple[tuple[int, str, np.ndarray], ...]:
    templates: list[tuple[int, str, np.ndarray]] = []
    fonts = FONT_PATHS or (Path(r"C:\Windows\Fonts\arial.ttf"),)
    for digit in range(10):
        for font_path in fonts:
            templates.append(
                (digit, font_path.name, _render_template(str(digit), font_path))
            )
    return tuple(templates)


def _chamfer_score(observed: np.ndarray, template: np.ndarray) -> float:
    observed_binary = observed > 0
    template_binary = template > 0
    if not np.any(observed_binary) or not np.any(template_binary):
        return float("inf")
    distance_to_template = cv2.distanceTransform(
        (~template_binary).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_observed = cv2.distanceTransform(
        (~observed_binary).astype(np.uint8), cv2.DIST_L2, 3
    )
    forward = float(distance_to_template[observed_binary].mean())
    backward = float(distance_to_observed[template_binary].mean())
    area_ratio = abs(
        np.count_nonzero(observed_binary) - np.count_nonzero(template_binary)
    ) / max(np.count_nonzero(observed_binary), np.count_nonzero(template_binary))
    return forward + backward + area_ratio * 1.5


def _read_digit(mask: np.ndarray) -> tuple[int, float]:
    observed = normalize_mask(mask)
    scored = sorted(
        (_chamfer_score(observed, template), digit)
        for digit, _font, template in _digit_templates()
    )
    best_score, best_digit = scored[0]
    second_score = next(
        (score for score, digit in scored[1:] if digit != best_digit), best_score
    )
    confidence = float(
        np.clip((second_score - best_score) / max(second_score, 1e-6), 0.0, 1.0)
    )
    return int(best_digit), confidence


def read_numeric_label(crop_bgr: np.ndarray) -> tuple[float | None, float]:
    raw_mask = extract_text_mask(crop_bgr)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, 8)
    components = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if area >= 2:
            components.append((component, x, y, width, height, area, centroids[component]))
    if not components:
        return None, 0.0

    tallest = max(component[4] for component in components)
    tall_components = [component for component in components if component[4] >= tallest * 0.60]
    if not tall_components:
        return None, 0.0
    left_digit_x = min(component[1] for component in tall_components)
    digit_top = min(component[2] for component in tall_components)
    digit_bottom = max(component[2] + component[4] for component in tall_components)
    digit_mid_y = (digit_top + digit_bottom) / 2.0

    minus_component: int | None = None
    decimal_found = False
    for component, x, y, width, height, _area, centroid in components:
        center_x, center_y = map(float, centroid)
        short = height <= max(4, tallest * 0.34)
        if (
            short
            and width >= 3
            and width / max(height, 1) >= 1.7
            and center_x < left_digit_x
            and abs(center_y - digit_mid_y) <= tallest * 0.30
        ):
            minus_component = component
        if (
            short
            and width <= max(4, tallest * 0.35)
            and center_x > left_digit_x
            and center_y >= digit_mid_y
        ):
            decimal_found = True

    # Every deviation label in this export uses one decimal place. This also
    # rejects title letters and dates accidentally detected as rounded boxes.
    if not decimal_found:
        return None, 0.0
    digit_components = sorted(tall_components, key=lambda component: component[1])
    if len(digit_components) == 2:
        digits: list[int] = []
        digit_confidences: list[float] = []
        for component, _x, _y, _width, _height, _area, _centroid in digit_components:
            digit_mask = np.where(labels == component, 255, 0).astype(np.uint8)
            digit, digit_confidence = _read_digit(digit_mask)
            digits.append(digit)
            digit_confidences.append(digit_confidence)
        magnitude = digits[0] + digits[1] / 10.0
        signed_value = -magnitude if minus_component is not None else magnitude
        return float(signed_value), float(min(digit_confidences))

    magnitude_mask = raw_mask.copy()
    if minus_component is not None:
        magnitude_mask[labels == minus_component] = 0
    observed = normalize_mask(magnitude_mask)
    if np.count_nonzero(observed) < 8:
        return None, 0.0
    scored = sorted(
        (_chamfer_score(observed, template), value)
        for value, _font, template in _templates()
    )
    best_score, best_value = scored[0]
    second_score = next(
        (score for score, value in scored[1:] if value != best_value), best_score
    )
    margin = max(second_score - best_score, 0.0)
    confidence = float(np.clip(margin / max(second_score, 1e-6), 0.0, 1.0))
    signed_value = -best_value if minus_component is not None else best_value
    return float(signed_value), confidence


__all__ = ["extract_text_mask", "read_numeric_label"]
