"""제로 영역을 네모로 바꾸는 규칙."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_line_detection import zero_shapes  # noqa: E402


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _fill(shapes, size=(400, 400)):
    canvas = np.zeros(size, np.uint8)
    for shape in shapes:
        cv2.fillPoly(canvas, [np.asarray(shape, np.int32)], 1)
    return canvas


def test_네모는_그대로_네모_하나다():
    boxes = zero_shapes.boxes_of(_rect(50, 50, 300, 150))
    assert len(boxes) == 1
    assert len(boxes[0]) == 4


def test_ㄱ자는_여러_네모로_갈린다():
    """꺾인 띠를 네모 하나로 감싸면 빈 데까지 덮는다."""
    corner = np.zeros((400, 400), np.uint8)
    cv2.fillPoly(corner, [np.asarray(_rect(40, 40, 360, 110), np.int32)], 1)
    cv2.fillPoly(corner, [np.asarray(_rect(290, 40, 360, 360), np.int32)], 1)
    found, _h = cv2.findContours(corner, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    contour = found[0].reshape(-1, 2)

    boxes = zero_shapes.boxes_of(contour)
    assert len(boxes) >= 2, "ㄱ 자를 네모 하나로 덮으면 안 된다"

    # 통짜 네모보다 헛덮는 넓이가 적어야 한다
    whole = np.rint(cv2.boxPoints(cv2.minAreaRect(contour))).astype(np.int32)
    real = int(corner.sum())
    over_whole = int(_fill([whole]).sum()) - real
    over_boxes = int((_fill(boxes) & (corner == 0)).sum())
    assert over_boxes < over_whole * 0.6, (
        f"헛덮음이 안 줄었다: 통짜 {over_whole} · 나눔 {over_boxes}")


def test_진짜_영역을_거의_다_덮는다():
    corner = np.zeros((400, 400), np.uint8)
    cv2.fillPoly(corner, [np.asarray(_rect(40, 40, 360, 110), np.int32)], 1)
    cv2.fillPoly(corner, [np.asarray(_rect(290, 40, 360, 360), np.int32)], 1)
    found, _h = cv2.findContours(corner, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    boxes = zero_shapes.boxes_of(found[0].reshape(-1, 2))
    covered = int((_fill(boxes) & corner).sum())
    assert covered / int(corner.sum()) > 0.9


def test_너무_작은_것은_버린다():
    assert zero_shapes.boxes_of(_rect(0, 0, 4, 4)) == []


def test_조각_수_상한을_지킨다():
    corner = np.zeros((400, 400), np.uint8)
    cv2.circle(corner, (200, 200), 150, 1, 12)      # 얇은 고리 — 제일 성기다
    found, _h = cv2.findContours(corner, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    boxes = zero_shapes.boxes_of(found[0].reshape(-1, 2), budget=4)
    assert len(boxes) <= 4


def test_빈_입력에도_안_터진다():
    assert zero_shapes.clean([]) == []
    assert zero_shapes.clean(None) == []
    assert zero_shapes.boxes_of([[1, 1], [2, 2]]) == []


def test_받은_파이프라인_결과를_디스크에_남긴다(tmp_path):
    """실측 64XX2 가 117초다 — 엔진을 다시 띄울 때마다 다시 돌 수 없다."""
    import json

    from zero_line_detection import lab_runner

    key = "시험용열쇠"
    old_dir = lab_runner.CACHE_DIR
    lab_runner.CACHE_DIR = tmp_path
    try:
        answer = {"prefix": "JD_64XX2-DR000", "lines": [[[1, 2], [3, 4]]],
                  "areas": [], "regions": []}
        lab_runner._disk_path(key).write_text(
            json.dumps(answer, ensure_ascii=False), encoding="utf-8")
        found = json.loads(
            lab_runner._disk_path(key).read_text(encoding="utf-8"))
        assert found == answer
    finally:
        lab_runner.CACHE_DIR = old_dir


def test_스크립트가_바뀌면_열쇠가_바뀐다():
    """받은 코드를 갈아 끼우면 캐시가 저절로 무효가 돼야 한다."""
    from zero_line_detection import lab_runner

    stamp = lab_runner._script_stamp("JD_64XX2-DR000")
    assert stamp and stamp == lab_runner._script_stamp("JD_64XX2-DR000")
    assert stamp != lab_runner._script_stamp("JD_67XX6-DR000"), (
        "부품이 다르면 스크립트도 다르다")
