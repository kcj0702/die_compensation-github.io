"""CATIA 관리면 BOUNDARY를 기존 CAD 뷰어 정합으로 스캔 PNG에 표시한다."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import mesh_io
from cad_import.catia_management_geometry import extract_management_boundaries
from cad_import.management_surface import fit_with_cad_viewer
from cad_import.overlay import to_pixels
from product_alignment.masks import build_scan_mask


PALETTE = {
    0.5: (20, 190, 255),    # orange
    -0.5: (255, 120, 30),   # blue
    -0.7: (220, 50, 200),   # magenta
    -1.0: (50, 70, 240),    # red
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"스캔 PNG를 읽지 못했습니다: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"결과 이미지를 인코딩하지 못했습니다: {path}")
    encoded.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catpart", type=Path, required=True,
                        help="관리면 트리가 들어 있는 CATPart")
    parser.add_argument("--scan", type=Path, required=True,
                        help="정합할 3D 스캔 편차 PNG")
    parser.add_argument("--alignment-cad", type=Path,
                        help="CAD 뷰어 정합용 CAD. 생략하면 --catpart 사용")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    scan = read_image(args.scan)
    scan_mask = build_scan_mask(scan)
    print("stage=scan-mask", flush=True)
    cache = args.cache_dir or args.output.parent / ".cache"
    # CATIA COM은 OpenCV/OCP의 병렬 작업을 시작하기 전에 끝낸다. 일부 V5
    # 환경에서는 계산 스레드가 만들어진 뒤 DispatchEx를 부르면 대기한다.
    management = extract_management_boundaries(
        args.catpart, cache_dir=cache / "management",
        progress=lambda message: print(f"stage={message}", flush=True))
    print(f"stage=management groups={len(management)}", flush=True)
    alignment_cad = args.alignment_cad or args.catpart
    mesh = mesh_io.load_any(alignment_cad, cache_dir=cache)
    print(f"stage=cad-mesh vertices={len(mesh.vertices)} faces={len(mesh.faces)}", flush=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    # 확인용 정지 이미지는 CAD 뷰어의 실루엣 정합까지만 사용한다. 광선
    # 명중률 polish는 고밀도 메시에서 오래 걸리며 UI 최종 투영 때 수행한다.
    fit = fit_with_cad_viewer(vertices, faces, scan_mask, mesh=mesh, polish=False)
    print(f"stage=fit iou={fit.iou:.4f}", flush=True)

    fill = np.zeros_like(scan)
    outlines: list[tuple[np.ndarray, tuple[int, int, int], float]] = []
    height, width = scan.shape[:2]
    for group, polylines in management:
        color = PALETTE.get(group.target_mm, (40, 210, 80))
        for polyline in polylines:
            xs, ys = to_pixels(polyline, fit)
            points = np.stack([xs, ys], axis=1)
            finite = np.isfinite(points).all(axis=1)
            points = np.rint(points[finite]).astype(np.int32)
            if len(points) < 3:
                continue
            # 완전히 화면 밖인 옆면은 이 PNG에 표시하지 않는다.
            if (points[:, 0].max() < 0 or points[:, 0].min() >= width
                    or points[:, 1].max() < 0 or points[:, 1].min() >= height):
                continue
            closed = float(np.linalg.norm(polyline[0] - polyline[-1])) <= 1.0
            if closed:
                cv2.fillPoly(fill, [points], color)
            outlines.append((points, color, group.target_mm))

    result = cv2.addWeighted(scan, 1.0, fill, 0.30, 0.0)
    thickness = max(2, round(max(height, width) / 600))
    for points, color, _target in outlines:
        cv2.polylines(result, [points], True, color, thickness, cv2.LINE_AA)

    # 곡선마다 라벨을 붙이면 실제 9개 영역에서 글자가 겹친다. 흰 여백에
    # 관리치별 범례를 한 번만 표시해 형상 경계를 가리지 않는다.
    targets = sorted({target for _points, _color, target in outlines}, reverse=True)
    if targets:
        scale = max(0.55, max(height, width) / 1800)
        line_gap = max(24, round(34 * scale))
        x0, y0 = 24, 30
        panel_width = max(260, round(360 * scale))
        panel_height = line_gap * (len(targets) + 1) + 12
        cv2.rectangle(result, (x0 - 12, y0 - 24),
                      (x0 + panel_width, y0 - 24 + panel_height),
                      (255, 255, 255), -1)
        cv2.rectangle(result, (x0 - 12, y0 - 24),
                      (x0 + panel_width, y0 - 24 + panel_height),
                      (70, 70, 70), 1)
        cv2.putText(result, f"MANAGEMENT SURFACE  IoU {fit.iou * 100:.1f}%",
                    (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (30, 30, 30), max(1, thickness - 1), cv2.LINE_AA)
        for index, target in enumerate(targets, start=1):
            color = PALETTE.get(target, (40, 210, 80))
            y = y0 + index * line_gap
            cv2.line(result, (x0, y - 5), (x0 + 48, y - 5), color,
                     thickness + 2, cv2.LINE_AA)
            cv2.putText(result, f"SPEC {target:+.1f} mm",
                        (x0 + 62, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (30, 30, 30), max(1, thickness - 1), cv2.LINE_AA)

    write_image(args.output, result)
    print(f"output={args.output.resolve()}")
    print(f"fit_iou={fit.iou:.4f} hit_rate={fit.hit_rate:.4f} reliable={fit.reliable}")
    print(f"management_groups={len(management)} rendered_boundaries={len(outlines)}")


if __name__ == "__main__":
    main()
