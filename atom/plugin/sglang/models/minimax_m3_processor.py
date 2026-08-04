"""Processor registration for MiniMax-M3 in SGLang plugin mode."""

from __future__ import annotations

from typing import ClassVar

try:
    from sglang.srt.managers.schedule_batch import Modality
    from sglang.srt.multimodal.processors.transformers_auto import (
        TransformersAutoMultimodalProcessor,
    )
except Exception:  # noqa: BLE001 - SGLang multimodal symbols are optional
    Modality = None
    TransformersAutoMultimodalProcessor = object
else:
    if not hasattr(Modality, "MULTI_IMAGES"):
        Modality.MULTI_IMAGES = Modality.IMAGE


class MiniMaxM3SparseForCausalLM:
    pass


class MiniMaxM3SparseForConditionalGeneration:
    pass


class MiniMaxM3TextOnlyProcessor(TransformersAutoMultimodalProcessor):
    """Use SGLang's generic HF processor path for MiniMax-M3 inputs."""

    models: ClassVar[list[type]] = [
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    ]
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


def register_minimax_m3_text_only_processor() -> None:
    """Register MiniMax-M3 on SGLang's generic HF processor path."""

    try:
        from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING
    except Exception:  # noqa: BLE001 - processor mapping is optional outside SGLang
        return

    PROCESSOR_MAPPING.setdefault(MiniMaxM3SparseForCausalLM, MiniMaxM3TextOnlyProcessor)
    PROCESSOR_MAPPING.setdefault(
        MiniMaxM3SparseForConditionalGeneration,
        MiniMaxM3TextOnlyProcessor,
    )
