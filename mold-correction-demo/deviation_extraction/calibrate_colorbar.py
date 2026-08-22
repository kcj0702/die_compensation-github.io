"""컬러바 ROI와 양 끝값을 확인해 설정값을 산출하는 보조 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

import config
from vlm_reader import LabelValueReader


def _end_crop(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    side: Literal["min", "max"],
    margin: int,
) -> np.ndarray:
    """범례 방향 규칙에 따라 최솟값 또는 최댓값 끝부분을 자른다."""
    x, y, w, h = roi
    if h >= w:
        if side == "max":
            y0, y1 = max(y - margin, 0), y + margin
        else:
            y0, y1 = y + h - margin, min(y + h + margin, bgr.shape[0])
        return bgr[y0:y1, x:x + w]
    else:
        if side == "min":
            x0, x1 = max(x - margin, 0), x + margin
        else:
            x0, x1 = x + w - margin, min(x + w + margin, bgr.shape[1])
        return bgr[y:y + h, x0:x1]


def main() -> None:
    """컬러바 양 끝 숫자를 판독해 config.py 형식으로 출력한다."""
    parser = argparse.ArgumentParser(description="컬러바 ROI와 값 범위 확인")
    parser.add_argument("--image", type=Path, required=True, help="편차 맵 경로")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="원본 이미지 기준 컬러바 영역",
    )
    parser.add_argument("--margin", type=int, default=30, help="긴 축 끝의 숫자 crop 여백")
    parser.add_argument("--model", type=str, default=config.VLM_MODEL_ID, help="VLM 모델 ID")
    parser.add_argument("--device", type=str, default=None, help="추론 장치")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="로컬 캐시만 사용하고 모델 다운로드 차단",
    )
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(args.image)

    roi = tuple(args.roi)
    reader = LabelValueReader(
        model_id=args.model,
        device=args.device,
        local_files_only=args.offline,
    )

    min_crop = cv2.cvtColor(_end_crop(bgr, roi, "min", args.margin), cv2.COLOR_BGR2RGB)
    max_crop = cv2.cvtColor(_end_crop(bgr, roi, "max", args.margin), cv2.COLOR_BGR2RGB)

    min_mm = reader.read_value(Image.fromarray(min_crop))
    max_mm = reader.read_value(Image.fromarray(max_crop))

    print(f"COLORBAR_ROI = {roi}")
    print(f"COLORBAR_MIN_MM = {min_mm}")
    print(f"COLORBAR_MAX_MM = {max_mm}")


if __name__ == "__main__":
    main()
