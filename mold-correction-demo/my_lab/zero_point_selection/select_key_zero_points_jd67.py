"""JD_67XX product-specific key zero-point engine."""

from key_zero_point_engine import ProductConfig, run_product


CONFIG = ProductConfig(
    engine_name="JD_67XX6",
    product_prefix="JD_67XX6",
    colorbar_top_mm=3.0,
    colorbar_bottom_mm=-3.0,
    key_threshold_mm=0.80,
)


def run_jd67():
    return run_product(CONFIG)


if __name__ == "__main__":
    for output in run_jd67():
        print(f"[완료] {output}")
