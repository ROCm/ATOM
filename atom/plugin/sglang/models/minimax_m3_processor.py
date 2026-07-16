"""Text-only processor registration for MiniMax-M3 in SGLang plugin mode."""

from __future__ import annotations

try:
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
except Exception:
    BaseMultimodalProcessor = object


class MiniMaxM3SparseForCausalLM:
    pass


class MiniMaxM3SparseForConditionalGeneration:
    pass


class MiniMaxM3TextOnlyProcessor(BaseMultimodalProcessor):
    """SGLang processor placeholder for text-only MiniMax-M3 serving."""

    models = [MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration]

    @staticmethod
    def _has_multimodal_payload(value) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, bytes)):
            return bool(value)
        if isinstance(value, dict):
            return any(
                MiniMaxM3TextOnlyProcessor._has_multimodal_payload(v)
                for v in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                MiniMaxM3TextOnlyProcessor._has_multimodal_payload(item)
                for item in value
            )
        return bool(value)

    @classmethod
    def _reject_multimodal_inputs(
        cls, image_data=None, audio_data=None, video_data=None
    ):
        if (
            cls._has_multimodal_payload(image_data)
            or cls._has_multimodal_payload(audio_data)
            or cls._has_multimodal_payload(video_data)
        ):
            raise ValueError(
                "ATOM SGLang MiniMax-M3 plugin currently supports text-only "
                "serving; multimodal inputs are not supported."
            )

    def process_mm_data(
        self,
        input_text,
        images=None,
        videos=None,
        audios=None,
        **kwargs,
    ) -> dict:
        del input_text, kwargs
        self._reject_multimodal_inputs(images, audios, videos)
        return {}

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        del input_text, request_obj, kwargs
        self._reject_multimodal_inputs(image_data, audio_data)
        return {}


def register_minimax_m3_text_only_processor() -> None:
    """Let SGLang tokenizer init accept MiniMax-M3 text-only serving.

    MiniMax-M3 checkpoints advertise a conditional-generation architecture and
    include multimodal sub-configs, so SGLang asks for a multimodal processor
    before model workers start.  The ATOM SGLang path currently supports only
    the language model, so plain text requests need a processor placeholder
    that rejects actual multimodal inputs.
    """

    try:
        from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING
    except Exception:
        return

    PROCESSOR_MAPPING.setdefault(MiniMaxM3SparseForCausalLM, MiniMaxM3TextOnlyProcessor)
    PROCESSOR_MAPPING.setdefault(
        MiniMaxM3SparseForConditionalGeneration,
        MiniMaxM3TextOnlyProcessor,
    )
