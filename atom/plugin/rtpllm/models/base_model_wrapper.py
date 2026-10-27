"""ATOM wrappers for rtp-llm external model loading.

Loaded via:
    --external_model_packages atom.plugin.rtpllm.models

This module intentionally keeps runtime behavior compatible with rtp-llm's
native qwen3.5-moe implementation while providing a plugin entrypoint that can
be extended with ATOM-specific logic later.
"""

from rtp_llm.model_factory_register import (
    _hf_architecture_2_ft,
    _model_factory,
    register_model,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.qwen35_renderer import Qwen35Renderer

from atom.plugin.rtpllm.models import qwen4_exp_vision  # noqa: F401
from atom.plugin.rtpllm.models.glm5 import ATOMGlm5Moe
from atom.plugin.rtpllm.models.qwen3_5 import ATOMQwen35Moe
from atom.plugin.rtpllm.models.qwen4_exp import ATOMQwen4Exp, ATOMQwen4ExpMTP


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

# Qwen3.8-Flash-Next uses the Qwen4Exp architecture, not Qwen3.5-MoE.
register_model("atom_qwen4_exp", ATOMQwen4Exp, ["Qwen4ExpForConditionalGeneration"])
register_model("atom_qwen4_exp_mtp", ATOMQwen4ExpMTP, [])
register_renderer("atom_qwen4_exp", Qwen35Renderer)
