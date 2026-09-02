"""시험이 실제 캐시 파일을 건드리지 않게 막는다.

[왜 필요한가]
판독 캐시와 현업 파이프라인 캐시를 디스크에 남기게 하면서 문제가
생겼다. 시험이 분석 경로를 지나가면 `_remember_labels` 가 그때의
(거의 비어 있는) 캐시를 **진짜 파일에 덮어쓴다.** 실제로 데워 둔
77 항목이 1 항목으로 줄었고, 다음 분석이 Qwen 을 71초 다시 돌았다.

[왜 픽스처가 아니라 여기서 하나]
시험 모듈이 server.py 를 importlib 로 **자기 사본**을 만들어 쓴다.
픽스처는 수집(import)이 끝난 뒤에 도니 그 사본에는 못 미친다.
conftest 의 모듈 수준 코드는 수집보다 먼저 돌므로, 여기서 환경변수를
잡아 두면 어떤 사본이 만들어져도 임시 폴더를 본다.
"""
from __future__ import annotations

import atexit
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_ROOM = tempfile.TemporaryDirectory(prefix="adc-test-cache-")
os.environ["ADC_LABEL_CACHE"] = str(Path(_ROOM.name) / "labels.json")
os.environ["ADC_LAB_CACHE"] = str(Path(_ROOM.name) / "lab")
os.environ["ADC_ADAPTIVE_CACHE"] = str(Path(_ROOM.name) / "adaptive")
atexit.register(_ROOM.cleanup)
