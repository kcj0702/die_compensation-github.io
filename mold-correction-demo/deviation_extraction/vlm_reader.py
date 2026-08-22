"""로컬 VLM으로 라벨 crop을 판독하고 첫 번째 숫자를 추출한다."""

from __future__ import annotations

import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils.versions import require_version

import config

_PROMPT = (
    "이 이미지는 도면 위에 표시된 숫자 라벨이다. "
    "부호(+/-)와 소수점을 포함한 숫자만 정확히 출력하라. "
    "숫자 외의 텍스트는 출력하지 마라."
)


class LabelValueReader:
    """VLM을 한 번 로드하고 여러 라벨 crop의 숫자를 순차 판독한다."""

    def __init__(
        self,
        model_id: str = config.VLM_MODEL_ID,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        require_version(
            "transformers>=4.49.0",
            "Qwen2.5-VL 판독기를 사용하려면 requirements.txt의 버전을 설치하세요.",
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu를 사용하세요.")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
            .to(self.device)
            .eval()
        )

    def read_value(self, crop: Image.Image) -> float | None:
        """생성문에서 처음 발견한 부호 있는 정수 또는 소수를 반환한다."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt_text],
            images=[crop],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=16)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return self._parse_number(decoded)

    @staticmethod
    def _parse_number(text: str) -> float | None:
        """파싱할 수 있는 숫자가 없으면 None을 반환한다."""
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(" ", ""))
        return float(match.group()) if match else None
