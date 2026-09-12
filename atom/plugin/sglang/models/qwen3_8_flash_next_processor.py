"""Processor registration for Qwen3.8-Flash-Next in SGLang plugin mode."""

from __future__ import annotations

from typing import ClassVar

try:
    from sglang.srt.multimodal.processors.transformers_auto import (
        TransformersAutoMultimodalProcessor,
    )
except Exception:  # noqa: BLE001 - SGLang multimodal symbols are optional
    TransformersAutoMultimodalProcessor = object


class Qwen4ExpForConditionalGeneration:
    pass


class Qwen4ExpTextOnlyProcessor(TransformersAutoMultimodalProcessor):
    """Use SGLang's generic HF processor path for Flash-Next text inputs."""

    models: ClassVar[list[type]] = [Qwen4ExpForConditionalGeneration]
    supports_transformers_backend = True


def register_qwen4_exp_text_only_processor() -> None:
    """Register Qwen4Exp on SGLang's generic HF processor path."""

    try:
        from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING
    except Exception:  # noqa: BLE001 - processor mapping is optional outside SGLang
        return

    PROCESSOR_MAPPING.setdefault(
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpTextOnlyProcessor,
    )
