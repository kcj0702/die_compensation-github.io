"""로컬 VLM으로 라벨 crop을 판독하고 첫 번째 숫자를 추출한다."""

from __future__ import annotations

import math
import re
from collections import Counter
from threading import Lock
from unicodedata import normalize

import torch
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.utils.versions import require_version

if __package__:  # 패키지 import와 직접 스크립트 실행을 모두 지원한다.
    from . import config
else:  # pragma: no cover - 직접 스크립트 실행 경로
    import config

_PROMPT = (
    "이미지 중앙의 편차 라벨을 확대해서 그 안에 실제로 인쇄된 값만 읽어라. "
    "라벨 바깥의 파란 지시선과 테두리는 무시하고 부호, 소수점, 숫자를 "
    "보이는 그대로 보존해야 한다. 명확한 값이 있으면 숫자 하나를 반드시 "
    "출력하고, 라벨이 없거나 숫자가 실제로 보이지 않을 때만 NONE을 출력하라."
)

_FOCUSED_PROMPT = (
    "OCR the exact numeric text visibly printed inside the centered label. "
    "Zero and 0.0 are valid measurements. Preserve the sign and decimal "
    "point. Reply with exactly one number, or NONE only if no digits are "
    "actually visible."
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
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            # Decoder-only generation must pad variable-width visual prompts
            # on the left. Right padding makes larger OCR batches decode from
            # padding positions and was the main source of batch-size-related
            # unread labels.
            tokenizer.padding_side = "left"
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

    def read_value_focused(self, crop: Image.Image) -> float | None:
        """Read one previously-unread label with at most two Qwen generations.

        The regular batch path remains the fast path. This method is intended
        only for crops that stayed ``None`` after that path. A high-contrast
        binary singleton gives the model the full visual-token budget; only
        when it stays unread does an inverted singleton get one final try. It
        never derives or substitutes a value from image colour; only a number
        actually returned by the same Qwen model is used.
        """
        prepared_views = (
            self._prepare_retry_crop(crop, variant="binary"),
            self._prepare_retry_crop(crop, variant="binary_inverted"),
        )
        for prepared in prepared_views:
            try:
                values = list(
                    self._read_batch([prepared], prompt=_FOCUSED_PROMPT)
                )
                if len(values) != 1:
                    raise ValueError(
                        "Focused Qwen result count must match its single crop."
                    )
                value = values[0]
            except (RuntimeError, ValueError, IndexError):
                # Keep prior batch results intact when a focused attempt fails.
                continue
            if value is not None and math.isfinite(value):
                return float(value)
        return None

    def read_values(
        self, crops: list[Image.Image], batch_size: int = config.VLM_BATCH_SIZE
    ) -> list[float | None]:
        """Qwen 1차 판독 후 미판독 라벨만 여러 실제 crop 보기로 재시도한다.

        재시도 결과는 원래 crop 인덱스별로 모은다. 서로 충돌하는 단발성
        판독은 버리고, 한 값만 읽혔거나 한 값이 명확한 다수일 때만 채운다.
        따라서 전처리 순서가 달라져도 다른 라벨의 값이 이동하지 않는다.
        """
        if not crops:
            return []
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        prepared = [self._prepare_crop(crop) for crop in crops]
        results = self._read_prepared_values(prepared, batch_size)

        unread = [index for index, value in enumerate(results) if value is None]
        if not unread:
            return results

        retry_votes: dict[int, list[float]] = {index: [] for index in unread}
        pending = unread
        variants = tuple(config.VLM_RETRY_VARIANTS)
        stage_size = config.VLM_RETRY_STAGE_SIZE
        retry_batch_size = min(batch_size, config.VLM_RETRY_BATCH_SIZE)
        if stage_size < 1:
            raise ValueError("VLM_RETRY_STAGE_SIZE must be at least 1.")
        if retry_batch_size < 1:
            raise ValueError("VLM_RETRY_BATCH_SIZE must be at least 1.")

        for stage_start in range(0, len(variants), stage_size):
            stage_variants = variants[stage_start:stage_start + stage_size]
            retry_crops: list[Image.Image] = []
            retry_indices: list[int] = []
            # Variant-major order keeps similarly-sized views next to each
            # other while retry_indices preserves the original crop mapping.
            for variant in stage_variants:
                for index in pending:
                    retry_crops.append(
                        self._prepare_retry_crop(crops[index], variant=variant)
                    )
                    retry_indices.append(index)

            retry_values = self._read_prepared_values(
                retry_crops,
                retry_batch_size,
            )
            for index, value in zip(retry_indices, retry_values):
                if value is not None:
                    retry_votes[index].append(value)

            is_last_stage = stage_start + stage_size >= len(variants)
            next_pending: list[int] = []
            for index in pending:
                if is_last_stage:
                    results[index] = self._resolve_retry_votes(
                        retry_votes[index]
                    )
                    continue
                consensus = self._resolve_retry_consensus(retry_votes[index])
                if consensus is None:
                    next_pending.append(index)
                else:
                    results[index] = consensus
            pending = next_pending
            if not pending:
                break
        return results

    @staticmethod
    def _resolve_retry_consensus(values: list[float]) -> float | None:
        """Return an early result only when independent Qwen views agree."""
        finite_values = [float(value) for value in values if math.isfinite(value)]
        if not finite_values:
            return None
        counts = Counter(finite_values)
        ranked = counts.most_common()
        best_value, best_count = ranked[0]
        next_count = ranked[1][1] if len(ranked) > 1 else 0
        if (
            best_count >= config.VLM_RETRY_EARLY_CONSENSUS
            and best_count > next_count
        ):
            return best_value
        return None

    @staticmethod
    def _resolve_retry_votes(values: list[float]) -> float | None:
        """같은 crop의 Qwen 재판독만 합의하고 충돌하는 단발성 값은 거부한다."""
        finite_values = [float(value) for value in values if math.isfinite(value)]
        if not finite_values:
            return None
        counts = Counter(finite_values)
        if len(counts) == 1:
            return finite_values[0]

        ranked = counts.most_common()
        best_value, best_count = ranked[0]
        next_count = ranked[1][1]
        if best_count >= 2 and best_count > next_count:
            return best_value
        return None

    def _read_prepared_values(
        self, crops: list[Image.Image], batch_size: int
    ) -> list[float | None]:
        """준비된 crop을 순서 보존 배치로 판독하고 개수 불일치를 거부한다."""
        results: list[float | None] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            batch_results = self._read_batch_with_oom_split(batch)
            if len(batch_results) != len(batch):
                raise ValueError("Qwen 판독 결과 수가 라벨 crop 수와 다릅니다.")
            results.extend(batch_results)
        return results

    def _read_batch_with_oom_split(
        self, crops: list[Image.Image]
    ) -> list[float | None]:
        """Retry a CUDA-OOM batch as ordered halves without losing indices."""
        try:
            values = list(self._read_batch(crops))
            if len(values) != len(crops):
                raise ValueError("Qwen 판독 결과 수가 라벨 crop 수와 다릅니다.")
            return values
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc) or len(crops) <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            middle = len(crops) // 2
            return (
                self._read_batch_with_oom_split(crops[:middle])
                + self._read_batch_with_oom_split(crops[middle:])
            )

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        if oom_type and isinstance(exc, oom_type):
            return True
        message = str(exc).casefold()
        return "out of memory" in message and (
            "cuda" in message or "cudnn" in message
        )

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
    def _prepare_retry_crop(crop: Image.Image, *, variant: str) -> Image.Image:
        """작은 글자를 보존하며 색상/명암/이진 극성이 다른 실제 crop을 만든다."""
        content = LabelValueReader._trim_retry_crop(crop).convert("RGB")
        resample = Image.Resampling.LANCZOS

        if variant == "color_sharp":
            prepared = ImageEnhance.Contrast(content).enhance(1.35)
        else:
            gray = ImageOps.autocontrast(content.convert("L"))
            if variant == "grayscale":
                prepared = gray.convert("RGB")
            elif variant in {"binary", "binary_inverted"}:
                threshold = LabelValueReader._otsu_threshold(gray)
                binary = gray.point(
                    lambda pixel: 255 if pixel > threshold else 0,
                    mode="1",
                ).convert("L")
                if variant == "binary_inverted":
                    binary = ImageOps.invert(binary)
                prepared = binary.convert("RGB")
                resample = Image.Resampling.NEAREST
            else:
                raise ValueError(f"지원하지 않는 Qwen 재시도 변형입니다: {variant}")

        prepared = LabelValueReader._resize_crop(
            prepared,
            minimum_height=config.VLM_RETRY_MIN_CROP_HEIGHT,
            maximum_scale=config.VLM_RETRY_MAX_CROP_SCALE,
            resample=resample,
        )
        if variant == "color_sharp":
            prepared = ImageEnhance.Sharpness(prepared).enhance(2.0)
        return ImageOps.expand(
            prepared,
            border=config.VLM_RETRY_BORDER,
            fill="white",
        )

    @staticmethod
    def _trim_retry_crop(crop: Image.Image) -> Image.Image:
        """바깥 리더선/이웃 테두리만 소량 제거하고 라벨 내부는 보존한다."""
        if crop.height <= 0 or crop.width <= 0:
            raise ValueError("비어 있는 라벨 crop은 판독할 수 없습니다.")
        trim = int(round(min(crop.size) * config.VLM_RETRY_TRIM_RATIO))
        trim = min(
            trim,
            max(0, (crop.width - 2) // 2),
            max(0, (crop.height - 2) // 2),
        )
        if trim < 1:
            return crop
        return crop.crop((trim, trim, crop.width - trim, crop.height - trim))

    @staticmethod
    def _otsu_threshold(gray: Image.Image) -> int:
        """빨간 바탕의 흰 글자와 흰 바탕의 검은 글자에 공통 임계값을 구한다."""
        histogram = gray.histogram()
        total = sum(histogram)
        if total <= 0:
            return 127

        weighted_total = sum(level * count for level, count in enumerate(histogram))
        background_count = 0
        background_sum = 0
        best_threshold = 127
        best_variance = -1.0
        for threshold, count in enumerate(histogram):
            background_count += count
            if background_count == 0:
                continue
            foreground_count = total - background_count
            if foreground_count == 0:
                break
            background_sum += threshold * count
            background_mean = background_sum / background_count
            foreground_mean = (
                weighted_total - background_sum
            ) / foreground_count
            between_variance = (
                background_count
                * foreground_count
                * (background_mean - foreground_mean) ** 2
            )
            if between_variance > best_variance:
                best_variance = between_variance
                best_threshold = threshold
        return best_threshold

    @staticmethod
    def _resize_crop(
        prepared: Image.Image,
        *,
        minimum_height: int,
        maximum_scale: float,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
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
        return prepared.resize(size, resample)

    def _read_batch(
        self,
        crops: list[Image.Image],
        *,
        prompt: str = _PROMPT,
    ) -> list[float | None]:
        """동일한 프롬프트를 적용한 라벨 crop 한 묶음을 생성한다."""
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop},
                        {"type": "text", "text": prompt},
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
                max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [self._parse_number(text) for text in decoded]

    @staticmethod
    def _parse_number(text: str) -> float | None:
        """Qwen이 한 값만 명시적으로 판독한 응답만 숫자로 반환한다."""
        normalized_text = normalize("NFKC", text).translate(
            str.maketrans({"−": "-", "–": "-", "—": "-", "，": ","})
        )
        if "none" in normalized_text.casefold():
            return None
        compact_text = "".join(normalized_text.split())
        matches = re.findall(r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)", compact_text)
        if len(matches) != 1:
            return None
        number = matches[0].replace(",", ".")
        if number.startswith("."):
            number = f"0{number}"
        if number.startswith(("+.", "-.")):
            number = f"{number[0]}0{number[1:]}"
        value = float(number)
        return value if math.isfinite(value) else None
