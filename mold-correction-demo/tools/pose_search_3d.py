"""검토 도구: 축에서 벗어난(등각 등) 시점의 이미지에 CAD 메시를 완전 3D 자세로 정합하고 결과를 그린다.

엔진 구현은 `cad_import.pose_search` 에 있다. 여기서는 입력을 읽고 결과 PNG/JSON 을 `outputs/` 에 남긴다.

    ..\\.venv\\Scripts\\python.exe tools\\pose_search_3d.py --mesh data\\product_mesh\\64XX2-DR000.catpart ^
        --image data\\sample\\64XX2-DC000.png --output outputs\\pose_64XX2-DC000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cad_import import mesh_io  # noqa: E402
from cad_import.mesh_io import split_symmetric_pair  # noqa: E402
from cad_import.overlay import to_pixels, unproject  # noqa: E402
from cad_import.pose_search import (  # noqa: E402
    cached_search_pose, outer_fill, pose_to_view, rasterize, render_silhouette_mask,
)
from product_alignment.masks import build_scan_mask  # noqa: E402


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")
    return image


def render_review(image, mask, fit, rotated, faces, score, path: Path) -> None:
    xs, ys = to_pixels(rotated, fit)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    raster = rasterize(points, faces, mask.shape)
    out = image.copy()
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, (0, 0, 255), 2, cv2.LINE_AA)
    target, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if target:
        cv2.drawContours(out, [max(target, key=cv2.contourArea)], -1, (0, 200, 0), 1, cv2.LINE_AA)
    sample = points[::max(1, len(points) // 6000)].astype(int)
    height, width = mask.shape
    inside = (sample[:, 0] >= 0) & (sample[:, 0] < width) & (sample[:, 1] >= 0) & (sample[:, 1] < height)
    out[sample[inside, 1], sample[inside, 0]] = (255, 255, 255)
    text = f"3D pose search  IoU {score * 100:.1f}%"
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", out)
    if ok:
        encoded.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True, help="CATPart/STEP/STL")
    parser.add_argument("--image", type=Path, required=True, help="정합할 이미지")
    parser.add_argument("--kind", choices=("render", "scan"), default="render", help="render: CAD 렌더 / scan: 편차맵")
    parser.add_argument("--output", type=Path, required=True, help="결과 파일 접두(…png, …json)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.management_overlay_cache"))
    parser.add_argument("--directions", type=int, default=800)
    parser.add_argument("--mirror", action="store_true", help="좌우 반전 후보 포함(반대편 시점과 구분 안 됨)")
    parser.add_argument("--part", type=int, default=0, help="대칭쌍일 때 사용할 절반(0/1)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    started = time.time()
    image = read_image(args.image)
    mask = outer_fill(build_scan_mask(image)) if args.kind == "scan" else render_silhouette_mask(image)
    print(f"mask: {mask.shape[1]}x{mask.shape[0]} nonzero={int((mask > 0).sum())}", flush=True)
    mesh = mesh_io.load_any(args.mesh, cache_dir=args.cache_dir)
    parts = split_symmetric_pair(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64))
    vertices, faces = parts[min(args.part, len(parts) - 1)]
    print(f"mesh: vertices={len(vertices)} faces={len(faces)} parts={len(parts)}", flush=True)
    score, pose = cached_search_pose(vertices, faces, mask, None if args.no_cache else args.cache_dir,
                                     directions=args.directions, allow_mirror=args.mirror,
                                     log=lambda message: print(message, flush=True))
    fit, rotated, matrix = pose_to_view(pose, vertices, score)
    # 되돌리기 검증: 화면 위 몇 점을 표면에 얹고 원래 좌표로 돌려 메시 정점과의 거리를 잰다.
    ys, xs = np.nonzero(mask)
    pick = np.linspace(0, len(xs) - 1, 25).astype(int)
    hits = unproject([[int(xs[i]), int(ys[i])] for i in pick], rotated, faces, fit, None)
    restored = [np.asarray(hit) @ matrix for hit in hits if hit is not None]
    if restored:
        nearest = [float(np.linalg.norm(vertices - point, axis=1).min()) for point in restored]
        print(f"unproject check: {len(restored)}/{len(pick)} hits, nearest-vertex distance median={np.median(nearest):.2f}mm max={max(nearest):.2f}mm")
    elapsed = time.time() - started
    result = {"image": str(args.image), "mesh": str(args.mesh), "iou": round(score, 4), "elapsed_s": round(elapsed, 1),
              **pose.to_dict(), "view_fit": fit.to_dict(), "matrix": matrix.round(6).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output) + ".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    render_review(image, mask, fit, rotated, faces, score, Path(str(args.output) + ".png"))
    print(f"result: IoU={score:.4f} euler={result['euler_deg']} mirror={pose.mirror} mm/px={pose.mm_per_px:.4f} ({elapsed:.0f}s)")
    print(f"review: {args.output}.png")


if __name__ == "__main__":
    main()
