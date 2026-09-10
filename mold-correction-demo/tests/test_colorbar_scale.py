from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import cv2

from zero_line_detection.colorbar_scale import _endpoint_crop, read_colorbar_range_mm
from zero_line_detection.case2_original_pipeline.out_of_tolerance import sample_bar_hue
from zero_line_detection.case2_route_adapter import _reviewed_case2_range


class _Reader:
    def __init__(self, values):
        self.values = values
        self.received = None

    def read_values(self, crops, batch_size):
        self.received = (crops, batch_size)
        return self.values


class ColorbarScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((300, 400, 3), dtype=np.uint8)
        self.info = SimpleNamespace(
            x0=350, x1=365, y0=40, y1=260, vmin_at="bottom"
        )

    def test_reads_asymmetric_range_from_endpoint_labels(self) -> None:
        reader = _Reader([-1.5, 2.0])
        self.assertEqual(
            read_colorbar_range_mm(self.image, self.info, reader),
            (-1.5, 2.0),
        )
        self.assertEqual(reader.received[1], 2)

    def test_endpoint_crop_excludes_the_adjacent_tick_row(self) -> None:
        crop = _endpoint_crop(self.image, self.info, "min")
        self.assertLessEqual(crop.height, 21)

    def test_does_not_substitute_a_product_default_when_ocr_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "품번 기본값"):
            read_colorbar_range_mm(self.image, self.info, _Reader([None, 2.0]))

    def test_case2_samples_an_asymmetric_physical_range(self) -> None:
        hsv = np.zeros((100, 8, 3), dtype=np.uint8)
        hsv[:, :, 0] = np.arange(100, dtype=np.uint8)[:, None]
        hsv[:, :, 1:] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        sampled = sample_bar_hue(bgr, (0, 0, 8, 100), 0.6, -1.5, 2.0)

        # (2.0 - 0.6) / (2.0 - -1.5) = 0.4, hence row/hue 40.
        self.assertAlmostEqual(sampled, 40.0, delta=1.0)

    def test_reviewed_case2_range_is_derived_without_product_defaults(self) -> None:
        self.assertEqual(_reviewed_case2_range((-1.5, 2.0)), (-2.0, 2.0))
        self.assertEqual(_reviewed_case2_range((-4.25, 3.0)), (-4.25, 4.25))


if __name__ == "__main__":
    unittest.main()
