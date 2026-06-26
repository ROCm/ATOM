"""ATOM wrappers for rtp-llm external model loading.

Loaded via:
    RTP_LLM_EXTERNAL_MODEL_PACKAGES=atom.plugin.rtpllm.models

This module intentionally keeps runtime behavior compatible with rtp-llm's
native qwen3.5-moe implementation while providing a plugin entrypoint that can
be extended with ATOM-specific logic later.
"""

from rtp_llm.model_factory_register import (
    _hf_architecture_2_ft,
    _model_factory,
    register_model,
)

from atom.plugin.rtpllm.models.glm5 import ATOMGlm5Moe
from atom.plugin.rtpllm.models.qwen3_5 import ATOMQwen35Moe


def _register_atom_qwen35_moe() -> None:
    """Register ATOM's rtp-llm model hook for qwen3_5moe."""
    # Extra model type for explicit selection.
    register_model("atom_qwen35_moe", ATOMQwen35Moe, [])

    # Override built-in mapping so standard qwen3.5-moe checkpoints start via
    # ATOM runtime.
    _model_factory["qwen35_moe"] = ATOMQwen35Moe
    _hf_architecture_2_ft["Qwen3_5MoeForConditionalGeneration"] = "qwen35_moe"


def _register_atom_glm5_moe() -> None:
    """Register ATOM's rtp-llm model hook for GLM5."""
    register_model("atom_glm5_moe", ATOMGlm5Moe, [])
    _model_factory["glm_5"] = ATOMGlm5Moe
    _hf_architecture_2_ft["GlmMoeDsaForCausalLM"] = "glm_5"


_register_atom_qwen35_moe()
_register_atom_glm5_moe()
