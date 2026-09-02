"""적응형 제로라인 묶음(zero_line (2))을 부르는 껍데기.

받은 코드는 한 줄도 고치지 않는다. 대신 그쪽이 전제하는 폴더 구조를
임시로 만들어 주고, SPECS 를 한 부품으로 좁혀 main() 을 부른다.
여기서 지키는 것은 그 **약속**이다 — 깨지면 67XX6 제로라인이 조용히
예전 방식으로 돌아간다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_line_detection import adaptive_runner as ar  # noqa: E402


def test_품번을_묶음_이름으로_바꾼다():
    assert ar.key_for("67XX6") == "JD_67XX6-DR000"
    assert ar.key_for("JD_67XX6-DR000") == "JD_67XX6-DR000"
    assert ar.key_for("64XX2") == "JD_64XX2-DR000"
    assert ar.key_for("없는품번") is None
    assert ar.key_for(None) is None


def test_받은_코드가_그대로_있다():
    """묶음 파일이 빠지면 67XX6 이 조용히 예전 방식으로 돌아간다."""
    for name in ("generate_correction_only_3pct_preview.py",
                 "generate_adaptive_zero_line_preview.py",
                 "generate_preview.py", "zero_boundary.py"):
        assert (ar.BUNDLE / name).is_file(), f"{name} 이 없다"


def test_껍데기가_한_부품만_남긴다():
    """그쪽 main() 은 SPECS 세 부품을 다 돈다. 앱에서는 보통 한 장뿐이라
    그대로 부르면 나머지 둘에서 멈춘다."""
    code = ar._shim("JD_67XX6-DR000", "generate_preview", ["--input-dir", "x"])
    assert "step.SPECS = tuple(" in code
    assert "JD_67XX6-DR000" in code
    assert "step.main()" in code
    # 인자도 그쪽이 읽을 수 있게 넘겨야 한다
    assert "'--input-dir'" in code and "'x'" in code


def test_알_수_없는_품번은_돌리지_않는다():
    blank = np.zeros((8, 8, 3), np.uint8)
    got = ar.run(blank, blank, "없는품번")
    assert "error" in got and "등록되지" in got["error"]


def test_컬러바가_없으면_솔직히_알린다():
    """범례를 못 찾으면 이 방식을 쓸 수 없다 — 조용히 넘어가면 안 된다."""
    blank = np.zeros((64, 64, 3), np.uint8)
    got = ar.run(blank, blank, "67XX6")
    assert "error" in got and "컬러바" in got["error"]


def test_캐시_열쇠에_우리_코드도_들어간다():
    """받은 코드만 해시했다가, 우리 쪽 단순화 값을 바꿔도 옛 결과가
    그대로 나왔다(꼭짓점 88~248개)."""
    source = (Path(ar.__file__)).read_text(encoding="utf-8")
    assert "Path(__file__).resolve()" in source
    assert "sorted(BUNDLE.rglob" in source
