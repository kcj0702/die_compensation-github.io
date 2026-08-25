"""JD_71XX product-specific key zero-point engine."""

from key_zero_point_engine import ProductConfig, run_product


CONFIG = ProductConfig(
    engine_name="JD_71XX2",
    product_prefix="JD_71XX2",
    colorbar_top_mm=2.0,
    colorbar_bottom_mm=-2.0,
)


def run_jd71():
    return run_product(CONFIG)


if __name__ == "__main__":
    for output in run_jd71():
        print(f"[완료] {output}")
