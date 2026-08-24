"""Run zero-point selection for every contour deviation graph."""

from select_zero_points import run_all


if __name__ == "__main__":
    for path in run_all():
        print(f"[완료] {path}")
