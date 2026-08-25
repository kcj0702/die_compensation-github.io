"""Run raw and key zero-point selection for JD_67XX only."""

from select_key_zero_points_jd67 import run_jd67
from select_zero_points import CONTOUR_GRAPH_OUTPUT, process_graph


if __name__ == "__main__":
    graphs = sorted(CONTOUR_GRAPH_OUTPUT.glob("JD_67XX6*/deviation_graph.json"))
    if len(graphs) != 1:
        raise FileNotFoundError("JD_67XX6 contour_graph 결과를 하나만 찾을 수 있어야 합니다.")
    process_graph(graphs[0])
    for path in run_jd67():
        print(f"[완료] {path}")
