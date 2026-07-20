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

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        del image_data, audio_data, input_text, request_obj, kwargs
        return None


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
