"""로컬 VLM으로 라벨 crop을 판독하고 첫 번째 숫자를 추출한다."""

from __future__ import annotations

import re
from threading import Lock
from unicodedata import normalize

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.utils.versions import require_version

if __package__:  # 패키지 import와 직접 스크립트 실행을 모두 지원한다.
    from . import config
else:  # pragma: no cover - 직접 스크립트 실행 경로
    import config

_PROMPT = (
    "이 이미지는 3D 스캔 도면의 빨간색 또는 흰색 편차 라벨 하나다. "
    "라벨 안에 실제로 보이는 부호(+/-), 소수점, 숫자를 그대로 읽어라. "
    "숫자 하나만 출력하고 확실히 읽을 수 없으면 NONE만 출력하라."
)


class LabelValueReader:
    """VLM을 한 번 로드하고 여러 라벨 crop의 숫자를 배치 판독한다."""

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
            if self.device.startswith("cuda"):
                selected_device = torch.device(self.device)
                device_index = selected_device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                use_8bit = (
                    torch.cuda.get_device_properties(device_index).total_memory
                    < 10 * 1024**3
                )
            else:
                use_8bit = False
        self.use_8bit = use_8bit
        self._inference_lock = Lock()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            use_fast=False,
        )
        load_kwargs = {
            "local_files_only": local_files_only,
            "torch_dtype": dtype,
        }
        if self.use_8bit:
            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map={"": self.device},
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
        """Qwen 1차 판독 후 미판독 라벨만 고대비 이미지로 재시도한다."""
        if not crops:
            return []
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        prepared = [self._prepare_crop(crop) for crop in crops]
        results = self._read_prepared_values(prepared, batch_size)

        for invert in (False, True):
            unread = [index for index, value in enumerate(results) if value is None]
            if not unread:
                break
            retry_crops = [
                self._prepare_retry_crop(crops[index], invert=invert)
                for index in unread
            ]
            retry_values = self._read_prepared_values(retry_crops, batch_size)
            for index, value in zip(unread, retry_values):
                if value is not None:
                    results[index] = value
        return results

    def _read_prepared_values(
        self, crops: list[Image.Image], batch_size: int
    ) -> list[float | None]:
        """준비된 crop을 순서 보존 배치로 판독하고 개수 불일치를 거부한다."""
        results: list[float | None] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            batch_results = list(self._read_batch(batch))
            if len(batch_results) != len(batch):
                raise ValueError("Qwen 판독 결과 수가 라벨 crop 수와 다릅니다.")
            results.extend(batch_results)
        return results

    @staticmethod
    def _prepare_crop(crop: Image.Image) -> Image.Image:
        """작은 라벨은 종횡비를 유지해 확대하고 RGB 모드로 정규화한다."""
        prepared = crop.convert("RGB")
        return LabelValueReader._resize_crop(
            prepared,
            minimum_height=config.VLM_MIN_CROP_HEIGHT,
            maximum_scale=config.VLM_MAX_CROP_SCALE,
        )

    @staticmethod
    def _prepare_retry_crop(crop: Image.Image, *, invert: bool) -> Image.Image:
        """색상 영향을 줄인 고대비 라벨을 만들고 두 글자 극성을 모두 시도한다."""
        gray = ImageOps.autocontrast(crop.convert("L"), cutoff=1)
        if invert:
            gray = ImageOps.invert(gray)
        prepared = ImageOps.expand(gray.convert("RGB"), border=4, fill="white")
        return LabelValueReader._resize_crop(
            prepared,
            minimum_height=config.VLM_RETRY_MIN_CROP_HEIGHT,
            maximum_scale=config.VLM_RETRY_MAX_CROP_SCALE,
        )

    @staticmethod
    def _resize_crop(
        prepared: Image.Image,
        *,
        minimum_height: int,
        maximum_scale: float,
    ) -> Image.Image:
        """지정한 최소 높이와 최대 배율 사이에서 종횡비를 유지해 확대한다."""
        if prepared.height <= 0 or prepared.width <= 0:
            raise ValueError("비어 있는 라벨 crop은 판독할 수 없습니다.")
        scale = min(
            maximum_scale,
            max(1.0, minimum_height / float(prepared.height)),
        )
        if scale == 1.0:
            return prepared
        size = (
            max(1, int(round(prepared.width * scale))),
            max(1, int(round(prepared.height * scale))),
        )
        return prepared.resize(size, Image.Resampling.LANCZOS)

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
        normalized_text = normalize("NFKC", text).translate(
            str.maketrans({"−": "-", "–": "-", "—": "-", "，": ","})
        )
        compact_text = "".join(normalized_text.split())
        match = re.search(r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)", compact_text)
        if not match:
            return None
        number = match.group().replace(",", ".")
        if number.startswith("."):
            number = f"0{number}"
        if number.startswith(("+.", "-.")):
            number = f"{number[0]}0{number[1:]}"
        return float(number)
