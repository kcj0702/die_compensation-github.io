"""Run raw and key zero-point selection for JD_71XX only."""

from select_key_zero_points_jd71 import run_jd71
from select_zero_points import CONTOUR_GRAPH_OUTPUT, process_graph


if __name__ == "__main__":
    graphs = sorted(CONTOUR_GRAPH_OUTPUT.glob("JD_71XX2*/deviation_graph.json"))
    if len(graphs) != 1:
        raise FileNotFoundError("JD_71XX2 contour_graph 결과를 하나만 찾을 수 있어야 합니다.")
    process_graph(graphs[0])
    for path in run_jd71():
        print(f"[완료] {path}")
