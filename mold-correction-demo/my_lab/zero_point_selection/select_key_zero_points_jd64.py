"""JD_64XX product-specific key zero-point engine."""

from key_zero_point_engine import ProductConfig, run_product


CONFIG = ProductConfig(
    engine_name="JD_64XX2",
    product_prefix="JD_64XX2",
    colorbar_top_mm=2.0,
    colorbar_bottom_mm=-1.5,
    key_threshold_mm=0.40,
)


def run_jd64():
    return run_product(CONFIG)


if __name__ == "__main__":
    for output in run_jd64():
        print(f"[완료] {output}")
