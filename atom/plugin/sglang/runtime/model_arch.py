"""SGLang plugin model adapter registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class SGLangModelAdapterSpec:
    """Adapter hooks for one SGLang plugin model architecture.

    The first version keeps the existing runtime flags while adding function
    hooks for config preparation and install-time model adaptation. This avoids
    growing a long list of booleans in the generic wrapper as new models arrive.
    """

    wrapper_binds_gdn_context: bool = False
    prepare_config: Optional[Callable[[Any, str], None]] = None
    install_adapters: Optional[Callable[[Any], None]] = None


def _prepare_qwen35_config(atom_config: Any, model_arch: str) -> None:
    from atom.plugin.sglang.models.qwen3_5 import apply_prepare_model_adaptations

    apply_prepare_model_adaptations(atom_config, model_arch)


def _install_deepseek_mla_adapters(model: Any) -> None:
    from atom.plugin.sglang.models.deepseek_mla import setup_deepseek_for_sglang

    setup_deepseek_for_sglang(model)


MODEL_ADAPTER_SPECS = {
    "DeepseekV3ForCausalLM": SGLangModelAdapterSpec(
        install_adapters=_install_deepseek_mla_adapters,
    ),
    "Qwen3MoeForCausalLM": SGLangModelAdapterSpec(),
    "Qwen3NextForCausalLM": SGLangModelAdapterSpec(
        wrapper_binds_gdn_context=True,
    ),
    "Qwen3_5ForConditionalGeneration": SGLangModelAdapterSpec(
        prepare_config=_prepare_qwen35_config,
    ),
    "Qwen3_5MoeForConditionalGeneration": SGLangModelAdapterSpec(
        prepare_config=_prepare_qwen35_config,
    ),
}

# Backwards-compatible alias for callers that only need generated EntryClass names.
MODEL_ARCH_SPECS = MODEL_ADAPTER_SPECS


def get_model_arch_spec(model_arch: str) -> SGLangModelAdapterSpec:
    return MODEL_ADAPTER_SPECS.get(model_arch, SGLangModelAdapterSpec())
