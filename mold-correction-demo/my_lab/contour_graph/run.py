"""Run deviation-graph rendering for every scan-point contour result."""

from plot_deviation_graph import run_all


if __name__ == "__main__":
    for path in run_all():
        print(f"[완료] {path}")
