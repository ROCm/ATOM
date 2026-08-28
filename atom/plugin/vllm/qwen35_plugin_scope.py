"""Scope helpers for Qwen3.5 vLLM plugin patches.

The block-FP8 FULL-cudagraph MHA/MoE fixes are validated on Qwen3.5-35B-A3B only.
Larger Qwen3.5 variants (e.g. 397B) keep the default vLLM plugin paths.
"""

from __future__ import annotations

from atom.config import get_current_atom_config

# Qwen3.5-35B-A3B text config fingerprint (distinct from 397B-A17B).
_QWEN35_35B_A3B_HIDDEN = 2048
_QWEN35_35B_A3B_LAYERS = 40
_QWEN35_35B_A3B_EXPERTS = 256


def _qwen35_text_config():
    atom_config = get_current_atom_config()
    if atom_config is None:
        return None
    hf_config = atom_config.hf_config
    text_config = getattr(hf_config, "text_config", None)
    return text_config if text_config is not None else hf_config


def is_qwen35_vllm_plugin_model() -> bool:
    try:
        atom_config = get_current_atom_config()
        if atom_config is None or not getattr(
            atom_config.plugin_config, "is_vllm", False
        ):
            return False
        archs = getattr(atom_config.hf_config, "architectures", None) or []
        qwen35_archs = {
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
        }
        if any(a in qwen35_archs for a in archs):
            return True
        hf_config = atom_config.hf_config
        model_type = getattr(hf_config, "model_type", "") or ""
        text_config = getattr(hf_config, "text_config", None)
        text_model_type = getattr(text_config, "model_type", "") if text_config else ""
        return model_type.startswith("qwen3_5") or text_model_type.startswith("qwen3_5")
    except Exception:  # noqa: BLE001
        return False


def is_qwen35_35b_a3b_vllm_plugin_model() -> bool:
    """True for Qwen3.5-35B-A3B hybrid models under the vLLM plugin."""
    if not is_qwen35_vllm_plugin_model():
        return False
    try:
        text_config = _qwen35_text_config()
        if text_config is None:
            return False
        return (
            getattr(text_config, "hidden_size", None) == _QWEN35_35B_A3B_HIDDEN
            and getattr(text_config, "num_hidden_layers", None)
            == _QWEN35_35B_A3B_LAYERS
            and getattr(text_config, "num_experts", None) == _QWEN35_35B_A3B_EXPERTS
        )
    except Exception:  # noqa: BLE001
        return False
