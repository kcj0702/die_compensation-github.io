"""로컬 VLM으로 라벨 crop을 판독하고 첫 번째 숫자를 추출한다."""

from __future__ import annotations

import re
from threading import Lock

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
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
        use_8bit: bool | None = None,
    ) -> None:
        require_version(
            "transformers>=4.49.0",
            "Qwen2.5-VL 판독기를 사용하려면 requirements.txt의 버전을 설치하세요.",
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu를 사용하세요.")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        if use_8bit is None:
            # Qwen2.5-VL-3B in FP16 leaves almost no activation headroom on an
            # 8 GB RTX 4070. LLM.int8 keeps the model near 4 GB and preserves
            # the outlier path in FP16, which is a good fit for label OCR.
            use_8bit = self.device.startswith("cuda") and (
                torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3
            )
        self.use_8bit = use_8bit
        self._inference_lock = Lock()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            use_fast=False,
        )
        load_kwargs = {
            "local_files_only": local_files_only,
            "dtype": dtype,
        }
        if self.use_8bit:
            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **load_kwargs
            ).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **load_kwargs
            ).to(self.device).eval()

    def read_value(self, crop: Image.Image) -> float | None:
        """생성문에서 처음 발견한 부호 있는 정수 또는 소수를 반환한다."""
        return self.read_values([crop], batch_size=1)[0]

    def read_values(
        self, crops: list[Image.Image], batch_size: int = 8
    ) -> list[float | None]:
        """여러 라벨을 GPU 배치로 판독해 모델 호출 횟수를 줄인다."""
        if not crops:
            return []
        results: list[float | None] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            results.extend(self._read_batch(batch))
        return results

    def _read_batch(self, crops: list[Image.Image]) -> list[float | None]:
        """동일한 프롬프트를 적용한 라벨 crop 한 묶음을 생성한다."""
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop},
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ]
            for crop in crops
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]
        inputs = self.processor(
            text=prompt_texts,
            images=crops,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # GPU generation is serialized because the local API can receive more
        # than one image while a previous batch is still being decoded.
        with self._inference_lock, torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                use_cache=True,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [self._parse_number(text) for text in decoded]

    @staticmethod
    def _parse_number(text: str) -> float | None:
        """파싱할 수 있는 숫자가 없으면 None을 반환한다."""
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(" ", ""))
        return float(match.group()) if match else None
