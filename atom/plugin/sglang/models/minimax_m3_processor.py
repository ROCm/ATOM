"""Text-only processor registration for MiniMax-M3 in SGLang plugin mode."""

from __future__ import annotations

try:
    from sglang.srt.managers.schedule_batch import Modality
    from sglang.srt.multimodal.processors.transformers_auto import (
        TransformersAutoMultimodalProcessor,
    )
except Exception:
    Modality = None
    TransformersAutoMultimodalProcessor = object


class MiniMaxM3SparseForCausalLM:
    pass


class MiniMaxM3SparseForConditionalGeneration:
    pass


class MiniMaxM3TextOnlyProcessor(TransformersAutoMultimodalProcessor):
    """Use SGLang's generic HF processor path for MiniMax-M3 inputs."""

    models = [MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration]
    supports_transformers_backend = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for processor_name in ("image_processor", "video_processor"):
            processor = getattr(self._processor, processor_name, None)
            default_resample = getattr(processor, "resample", None)
            preprocess = getattr(processor, "_preprocess", None)
            if (
                processor is None
                or default_resample is None
                or preprocess is None
                or getattr(processor, "_atom_resample_patched", False)
            ):
                continue

            def _preprocess_with_resample(
                *args,
                _preprocess=preprocess,
                _default_resample=default_resample,
                **kwargs,
            ):
                kwargs.setdefault("resample", _default_resample)
                return _preprocess(*args, **kwargs)

            processor._preprocess = _preprocess_with_resample
            processor._atom_resample_patched = True

    def _build_mm_items(self, processor_output: dict, input_ids):
        items = self.collect_mm_items_from_processor_output(processor_output)

        modality_to_token_id = {
            Modality.IMAGE: self.mm_tokens.image_token_id,
            Modality.VIDEO: self.mm_tokens.video_token_id,
            Modality.AUDIO: self.mm_tokens.audio_token_id,
        }
        if hasattr(Modality, "MULTI_IMAGES"):
            modality_to_token_id[Modality.MULTI_IMAGES] = self.mm_tokens.image_token_id

        for item in items:
            token_id = modality_to_token_id.get(item.modality)
            if token_id is not None:
                item.offsets = self.get_mm_items_offset(input_ids, token_id)

        return items


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
