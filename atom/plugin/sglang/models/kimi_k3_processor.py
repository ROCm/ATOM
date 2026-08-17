"""Processor registration for Kimi-K3 in SGLang plugin mode."""

from __future__ import annotations

from typing import ClassVar

try:
    from sglang.srt.multimodal.processors.transformers_auto import (
        TransformersAutoMultimodalProcessor,
    )
except Exception:  # noqa: BLE001 - SGLang multimodal symbols are optional
    TransformersAutoMultimodalProcessor = object


class KimiK3ForConditionalGeneration:
    pass


class KimiK3TextOnlyProcessor(TransformersAutoMultimodalProcessor):
    """Use SGLang's generic HF processor path for Kimi-K3 text inputs."""

    models: ClassVar[list[type]] = [KimiK3ForConditionalGeneration]
    supports_transformers_backend = True


def register_kimi_k3_text_only_processor() -> None:
    """Register Kimi-K3 on SGLang's generic HF processor path."""

    try:
        from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING
    except Exception:  # noqa: BLE001 - processor mapping is optional outside SGLang
        return

    PROCESSOR_MAPPING.setdefault(
        KimiK3ForConditionalGeneration,
        KimiK3TextOnlyProcessor,
    )
