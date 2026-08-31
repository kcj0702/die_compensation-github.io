"""회사 원본 없이 제품데이터 정렬과 포인트 전사 계약을 검증한다."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from product_alignment import config  # noqa: E402
from product_alignment.alignment import (  # noqa: E402
    Alignment, _flip_matrix, _overlap, estimate_alignment, is_inside, map_point,
    score_orientations,
)
from product_alignment.compose import (  # noqa: E402
    SheetPoint, compose_scale, render_alignment_overlay, render_points,
)
from product_alignment.masks import (  # noqa: E402
    bounding_box, build_product_mask, build_scan_mask, fill_silhouette, hole_mask,
)
from product_alignment.registry import (  # noqa: E402
    AlignmentStore, ProductLibrary, base_number, part_number_from_name,
)


GREEN = (60, 200, 60)
WHITE = (255, 255, 255)


def _panel(width: int, height: int) -> np.ndarray:
    """비대칭 노치와 크기가 다른 구멍 두 개를 가진 합성 부품."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (width - 11, height - 11), GREEN, -1)
    # 좌상단만 잘라내 상하·좌우 대칭을 깬다.
    cv2.rectangle(image, (10, 10), (10 + width // 5, 10 + height // 4), WHITE, -1)
    cv2.circle(image, (width // 3, height // 2), max(4, height // 10), WHITE, -1)
    cv2.circle(image, (2 * width // 3, height // 2), max(3, height // 16), WHITE, -1)
    return image


def _ring(width: int, height: int) -> np.ndarray:
    """상하좌우가 모두 대칭이라 방향을 가릴 수 없는 합성 부품."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (width - 11, height - 11), GREEN, -1)
    cv2.rectangle(
        image,
        (10 + width // 6, 10 + height // 6),
        (width - 11 - width // 6, height - 11 - height // 6),
        WHITE,
        -1,
    )
    return image


class MaskTest(unittest.TestCase):
    def test_product_mask_keeps_the_panel_and_drops_the_background(self) -> None:
        mask = build_product_mask(_panel(200, 120))

        self.assertEqual(mask.shape, (120, 200))
        self.assertTrue(np.any(mask > 0))
        self.assertEqual(mask[2, 2], 0)

    def test_holes_are_separated_from_the_filled_silhouette(self) -> None:
        mask = build_product_mask(_panel(200, 120))

        filled = fill_silhouette(mask)
        holes = hole_mask(mask, filled)

        self.assertGreater(np.count_nonzero(filled), np.count_nonzero(mask))
        self.assertTrue(np.any(holes > 0))
        self.assertFalse(np.any((holes > 0) & (mask > 0)))

    def test_scan_mask_uses_the_existing_label_removal_rule(self) -> None:
        mask = build_scan_mask(_panel(200, 120))

        self.assertEqual(mask.shape, (120, 200))
        self.assertTrue(np.any(mask > 0))


class AlignmentTest(unittest.TestCase):
    def _pair(self, flip_code: int) -> tuple[np.ndarray, np.ndarray]:
        """제품데이터와, 그것을 뒤집어 2배로 키운 가짜 스캔을 만든다."""
        product = _panel(200, 120)
        scan = cv2.resize(
            cv2.flip(product, flip_code), (400, 240), interpolation=cv2.INTER_NEAREST
        )
        return build_product_mask(scan), build_product_mask(product)

    def test_every_flip_is_recovered(self) -> None:
        for flip_code, expected in ((1, (True, False)), (0, (False, True)), (-1, (True, True))):
            with self.subTest(flip_code=flip_code):
                scan_mask, product_mask = self._pair(flip_code)

                alignment = estimate_alignment(scan_mask, product_mask)

                self.assertEqual((alignment.flip_x, alignment.flip_y), expected)
                self.assertGreaterEqual(alignment.outline_iou, config.MIN_OUTLINE_IOU)
                self.assertTrue(alignment.confident)

    def test_mapped_point_lands_on_the_matching_product_pixel(self) -> None:
        scan_mask, product_mask = self._pair(-1)
        alignment = estimate_alignment(scan_mask, product_mask)

        # 180도 뒤집힌 2배 스캔이므로 (x, y) -> ((399-x)/2, (239-y)/2) 이 정답이다.
        for scan_x, scan_y in ((100, 60), (300, 180), (240, 96)):
            mapped_x, mapped_y = map_point(alignment, scan_x, scan_y)
            self.assertAlmostEqual(mapped_x, (399 - scan_x) / 2, delta=3.0)
            self.assertAlmostEqual(mapped_y, (239 - scan_y) / 2, delta=3.0)

    def _overlap_of(self, matrix, scan_mask, product_mask) -> float:
        scan_solid = fill_silhouette(scan_mask)
        product_solid = fill_silhouette(product_mask)
        return _overlap(
            matrix,
            scan_solid,
            hole_mask(scan_mask, scan_solid),
            product_solid,
            hole_mask(product_mask, product_solid),
        )

    def test_refinement_never_lowers_the_overlap(self) -> None:
        """보정 단계는 겹침이 좋아질 때만 변환을 바꾼다.

        bbox 매칭은 두 마스크가 가장자리에 무엇을 포함하는지를 그대로 물려받아
        부품을 조금 축소시킨다. 실측에서는 구멍 중심이 반경 방향으로 3.35px
        안쪽으로 당겨졌고 보정 후 0.02px로 줄었다. 여기서는 어떤 입력에서도
        보정이 겹침을 떨어뜨리지 않는다는 점만 고정한다.
        """
        product = _panel(200, 120)
        product_mask = build_product_mask(product)
        base = build_product_mask(
            cv2.resize(
                cv2.flip(product, -1), (400, 240), interpolation=cv2.INTER_NEAREST
            )
        )
        # 가장자리 두께가 같은 경우와 스캔만 두꺼운 경우를 모두 확인한다.
        for size in (0, 3, 5):
            with self.subTest(dilate=size):
                scan_mask = base if size == 0 else cv2.dilate(
                    base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                )
                alignment = estimate_alignment(scan_mask, product_mask)
                raw = _flip_matrix(
                    bounding_box(fill_silhouette(scan_mask)),
                    bounding_box(fill_silhouette(product_mask)),
                    alignment.flip_x,
                    alignment.flip_y,
                )

                self.assertGreaterEqual(
                    self._overlap_of(alignment.as_array(), scan_mask, product_mask),
                    self._overlap_of(raw, scan_mask, product_mask),
                )

    def test_four_orientations_are_always_scored(self) -> None:
        scan_mask, product_mask = self._pair(-1)

        scores = score_orientations(scan_mask, product_mask)

        self.assertEqual(len(scores), 4)
        self.assertEqual(
            {(item.flip_x, item.flip_y) for item in scores},
            set(config.FLIP_CANDIDATES),
        )

    def test_symmetric_part_is_reported_instead_of_guessed(self) -> None:
        ring = _ring(200, 120)
        scan_mask = build_product_mask(
            cv2.resize(cv2.flip(ring, -1), (400, 240), interpolation=cv2.INTER_NEAREST)
        )

        alignment = estimate_alignment(scan_mask, build_product_mask(ring))

        self.assertLess(alignment.margin, config.MIN_DECISION_MARGIN)
        self.assertFalse(alignment.confident)
        self.assertTrue(any("방향" in warning for warning in alignment.warnings))

    def test_mismatched_outline_is_warned_about(self) -> None:
        product = _panel(200, 120)
        ellipse = np.full((120, 200, 3), 255, dtype=np.uint8)
        cv2.ellipse(ellipse, (100, 60), (90, 50), 0, 0, 360, GREEN, -1)

        alignment = estimate_alignment(
            build_product_mask(ellipse), build_product_mask(product)
        )

        self.assertLess(alignment.outline_iou, config.MIN_OUTLINE_IOU)
        self.assertFalse(alignment.confident)
        self.assertTrue(any("외형" in warning for warning in alignment.warnings))

    def test_confirmed_orientation_overrides_the_estimate(self) -> None:
        ring = _ring(200, 120)
        scan_mask = build_product_mask(
            cv2.resize(ring, (400, 240), interpolation=cv2.INTER_NEAREST)
        )

        alignment = estimate_alignment(
            scan_mask, build_product_mask(ring), flip_x=True, flip_y=False
        )

        self.assertEqual((alignment.flip_x, alignment.flip_y), (True, False))
        self.assertTrue(alignment.overridden)
        self.assertTrue(alignment.confident)
        self.assertFalse(any("방향" in warning for warning in alignment.warnings))

    def test_pinning_one_axis_still_decides_the_other(self) -> None:
        # 정답은 (True, True)다. 맞는 축을 고정하면 나머지 축은 자동으로 맞아야 한다.
        scan_mask, product_mask = self._pair(-1)

        alignment = estimate_alignment(scan_mask, product_mask, flip_x=True)

        self.assertTrue(alignment.flip_x)
        self.assertTrue(alignment.flip_y)
        # 한 축만 지정한 것은 사람이 방향을 확정한 것이 아니므로 모호성 검사가 남는다.
        self.assertFalse(alignment.overridden)

    def test_pinned_axis_is_never_silently_ignored(self) -> None:
        # 틀린 축을 고정해도 그 고정은 지켜지고, 남은 축 중 최고점이 뽑혀야 한다.
        scan_mask, product_mask = self._pair(-1)
        best_with_pin = max(
            (item for item in score_orientations(scan_mask, product_mask) if not item.flip_x),
            key=lambda item: item.score,
        )

        alignment = estimate_alignment(scan_mask, product_mask, flip_x=False)

        self.assertFalse(alignment.flip_x)
        self.assertEqual(alignment.flip_y, best_with_pin.flip_y)
        self.assertFalse(alignment.overridden)

    def test_confirmed_orientation_cannot_vouch_for_the_wrong_panel(self) -> None:
        # 확정 저장된 방향을 엉뚱한 제품데이터에 적용하면 방향은 사람이 정했어도
        # 부품이 다르다는 사실은 그대로 드러나야 한다.
        ellipse = np.full((120, 200, 3), 255, dtype=np.uint8)
        cv2.ellipse(ellipse, (100, 60), (90, 50), 0, 0, 360, GREEN, -1)

        alignment = estimate_alignment(
            build_product_mask(ellipse),
            build_product_mask(_panel(200, 120)),
            flip_x=True,
            flip_y=True,
        )

        self.assertTrue(alignment.overridden)
        self.assertLess(alignment.outline_iou, config.MIN_OUTLINE_IOU)
        self.assertFalse(alignment.confident)
        self.assertTrue(any("외형" in warning for warning in alignment.warnings))

    def test_is_inside_uses_the_product_image_bounds(self) -> None:
        scan_mask, product_mask = self._pair(-1)
        alignment = estimate_alignment(scan_mask, product_mask)

        self.assertTrue(is_inside(alignment, 0, 0))
        self.assertTrue(is_inside(alignment, 199, 119))
        self.assertFalse(is_inside(alignment, 200, 60))
        self.assertFalse(is_inside(alignment, -1, 60))

    def test_alignment_survives_a_json_round_trip(self) -> None:
        scan_mask, product_mask = self._pair(-1)
        alignment = estimate_alignment(scan_mask, product_mask)

        restored = Alignment.from_dict(alignment.to_dict())

        self.assertEqual(restored.flip_x, alignment.flip_x)
        self.assertEqual(restored.flip_y, alignment.flip_y)
        self.assertEqual(restored.product_size, alignment.product_size)
        for original, copied in zip(alignment.matrix, restored.matrix):
            self.assertAlmostEqual(original, copied, places=5)


class RegistryTest(unittest.TestCase):
    def test_part_number_is_read_from_both_file_naming_styles(self) -> None:
        self.assertEqual(
            part_number_from_name("JD_64XX2-DR000 3D 스캔.png"), "64XX2-DR000"
        )
        self.assertEqual(
            part_number_from_name("64XX2-DR000 제품데이터.png"), "64XX2-DR000"
        )
        self.assertEqual(
            part_number_from_name("JM_67312-DZ000_DASH LWR_OP10_260825.zip"),
            "67312-DZ000",
        )
        self.assertIsNone(part_number_from_name("scan.png"))

    def test_base_number_drops_the_variant_code(self) -> None:
        self.assertEqual(base_number("67XX6-DR050"), "67XX6")
        self.assertEqual(base_number("67XX6"), "67XX6")

    def test_registered_product_is_found_by_exact_part_number(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            library = ProductLibrary(Path(temp_dir))
            library.register("64XX2-DR000", _panel(200, 120))

            match = library.find("64XX2-DR000")

            self.assertIsNotNone(match)
            self.assertTrue(match.exact)
            self.assertEqual(match.part_number, "64XX2-DR000")
            self.assertEqual(library.registered(), ["64XX2-DR000"])

    def test_variant_suffix_still_finds_the_same_panel(self) -> None:
        # 실제 자료에서 67XX6-DR000 스캔의 제품데이터는 67XX6-DR050으로 저장돼 있다.
        with tempfile.TemporaryDirectory() as temp_dir:
            library = ProductLibrary(Path(temp_dir))
            library.register("67XX6-DR050", _panel(200, 120))

            match = library.find("67XX6-DR000")

            self.assertIsNotNone(match)
            self.assertFalse(match.exact)
            self.assertEqual(match.part_number, "67XX6-DR050")

    def test_unrelated_part_number_is_not_matched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            library = ProductLibrary(Path(temp_dir))
            library.register("64XX2-DR000", _panel(200, 120))

            self.assertIsNone(library.find("71XX2-DR000"))

    def test_confirmed_alignment_is_stored_per_part_number(self) -> None:
        product = _panel(200, 120)
        scan = cv2.resize(
            cv2.flip(product, -1), (400, 240), interpolation=cv2.INTER_NEAREST
        )
        alignment = estimate_alignment(
            build_product_mask(scan), build_product_mask(product)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = AlignmentStore(Path(temp_dir))
            self.assertIsNone(store.load("64XX2-DR000"))

            store.save("64XX2-DR000", alignment)
            restored = store.load("64XX2-DR000")

            self.assertIsNotNone(restored)
            self.assertEqual(restored.flip_x, alignment.flip_x)
            self.assertEqual(restored.flip_y, alignment.flip_y)
            self.assertTrue(store.forget("64XX2-DR000"))
            self.assertFalse(store.forget("64XX2-DR000"))


class ComposeTest(unittest.TestCase):
    def test_small_product_image_is_upscaled_within_the_cap(self) -> None:
        self.assertEqual(compose_scale(653), 3)
        self.assertEqual(compose_scale(2000), 1)
        self.assertEqual(compose_scale(10), config.COMPOSE_MAX_SCALE)

    def test_points_are_drawn_on_the_product_image(self) -> None:
        product = _panel(200, 120)
        points = [
            SheetPoint("P-01", 60.0, 40.0, value=-0.7, label_color="red"),
            SheetPoint("P-02", 140.0, 80.0, value=0.4, label_color="white"),
        ]

        canvas = render_points(product, points)

        scale = compose_scale(product.shape[1])
        self.assertEqual(canvas.shape[1], product.shape[1] * scale)
        marker = canvas[int(40 * scale), int(60 * scale)]
        self.assertLess(int(marker[1]), 120)

    def test_point_outside_the_image_is_skipped_without_error(self) -> None:
        product = _panel(200, 120)

        canvas = render_points(product, [SheetPoint("P-01", 999.0, 999.0)])

        self.assertEqual(canvas.shape[1], product.shape[1] * compose_scale(200))

    def test_values_can_be_omitted(self) -> None:
        product = _panel(200, 120)
        points = [SheetPoint("P-01", 60.0, 40.0, value=-0.7)]

        with_values = render_points(product, points)
        without_values = render_points(product, points, show_values=False)

        self.assertFalse(np.array_equal(with_values, without_values))

    def test_overlay_traces_the_scan_outline_and_its_holes(self) -> None:
        product = _panel(200, 120)
        product_mask = build_product_mask(product)

        aligned = render_alignment_overlay(product, product_mask)
        shifted = render_alignment_overlay(product, np.roll(product_mask, 12, axis=1))

        # 윤곽선만 칠하므로 원본과 다른 픽셀은 전체의 일부에 그친다.
        changed = np.any(aligned != product, axis=2)
        self.assertTrue(changed.any())
        self.assertLess(changed.mean(), 0.25)
        # 구멍 경계까지 그리므로 부품 안쪽에도 선이 남는다.
        self.assertTrue(np.any(changed[50:70, 60:80]))
        self.assertFalse(np.array_equal(aligned, shifted))


if __name__ == "__main__":
    unittest.main()
