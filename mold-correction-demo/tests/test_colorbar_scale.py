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


class _FocusedReader(_Reader):
    def __init__(self, values, focused_values):
        super().__init__(values)
        self.focused_values = iter(focused_values)
        self.focused_calls = 0

    def read_value_focused(self, crop):
        self.focused_calls += 1
        return next(self.focused_values)


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
        self.assertLessEqual(crop.height, 49)

    def test_does_not_substitute_a_product_default_when_ocr_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "품번 기본값"):
            read_colorbar_range_mm(self.image, self.info, _Reader([None, 2.0]))

    def test_retries_zero_minimum_with_focused_ocr(self) -> None:
        reader = _FocusedReader([0.0, 2.0], [-1.5])
        self.assertEqual(
            read_colorbar_range_mm(self.image, self.info, reader),
            (-1.5, 2.0),
        )
        self.assertEqual(reader.focused_calls, 1)

    def test_rejects_zero_as_a_colorbar_endpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "음수 최솟값"):
            read_colorbar_range_mm(self.image, self.info, _Reader([0.0, 2.0]))

    def test_reads_an_upside_down_export_by_turning_the_crop(self) -> None:
        """A scan saved upside down prints its endpoint numbers upside down too."""

        class _TurnedReader:
            def __init__(self) -> None:
                self.turned = [-2.0, 2.0]
                self.calls = 0

            def read_values(self, crops, batch_size):
                self.calls += 1
                if len(crops) > 1:
                    return [None, None]          # the upright crops show the digits upside down
                return [self.turned[0] if self.calls <= 4 else self.turned[1]]

        reader = _TurnedReader()
        self.assertEqual(read_colorbar_range_mm(self.image, self.info, reader), (-2.0, 2.0))

    def test_a_number_that_reads_differently_every_time_is_not_accepted(self) -> None:
        """A clipped endpoint label reads as something different each look - refuse, never guess."""

        class _UnstableReader:
            def __init__(self) -> None:
                self.answers = iter([None, 2.0, 1.0, 5.0, 0.5, 3.0, 4.0, 1.5])

            def read_values(self, crops, batch_size):
                if len(crops) > 1:
                    return [-2.0, None]
                return [next(self.answers, None)]

        with self.assertRaisesRegex(RuntimeError, "음수 최솟값|품번 기본값"):
            read_colorbar_range_mm(self.image, self.info, _UnstableReader())

    def test_a_small_export_is_retried_enlarged(self) -> None:
        """A 578 x 338 export prints the endpoint number 6 px tall; the reader needs real pixels."""

        class _EnlargedOnlyReader:
            """Reads nothing from the small crop and the right number from the enlarged one."""

            def __init__(self) -> None:
                self.sizes = []
                self.enlarged_reads = 0

            def read_values(self, crops, batch_size):
                self.sizes.extend((crop.width, crop.height) for crop in crops)
                if len(crops) > 1:
                    return [None, None]
                if crops[0].height <= 100:
                    return [None]
                self.enlarged_reads += 1            # two enlarged variants per endpoint
                return [-2.0 if self.enlarged_reads <= 2 else 2.0]

        reader = _EnlargedOnlyReader()
        self.assertEqual(read_colorbar_range_mm(self.image, self.info, reader), (-2.0, 2.0))
        self.assertTrue(any(height > 100 for _width, height in reader.sizes), "no enlarged crop was tried")
        # The first attempt still uses the crop as it was measured.
        self.assertLessEqual(_endpoint_crop(self.image, self.info, "min").height, 49)

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
