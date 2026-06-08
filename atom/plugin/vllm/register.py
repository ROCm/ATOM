from typing import Optional
import logging

import torch
from atom.plugin.prepare import _set_framework_backbone
from atom.utils import envs
from atom.plugin.vllm.spec_decode_patch import apply_vllm_spec_decode_patch

logger = logging.getLogger("atom")

# this flag is used to enable the vllm plugin mode
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN

# those 2 models are covering most of dense and moe models
ATOM_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMForCausalLM"
ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMMoEForCausalLM"

# when register new model to vllm, add here
# Keys is from hf config arch name
_VLLM_MODEL_REGISTRY_OVERRIDES: dict[str, str] = {
    "LlamaForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3ForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GptOssForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV3ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV32ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Glm4MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GlmMoeDsaForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepSeekMTPModel": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Glm4MoeMTPModel": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3NextForCausalLM": "atom.plugin.vllm.models.qwen3_next:Qwen3NextForCausalLMVllm",
    "Qwen3NextMTP": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3_5ForConditionalGeneration": "atom.plugin.vllm.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "atom.plugin.vllm.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    "KimiK25ForConditionalGeneration": "atom.plugin.vllm.models.kimi_k25:KimiK25ForConditionalGeneration",
    "MiniMaxM2ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "MiniMaxM3SparseForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "MiniMaxM3SparseForConditionalGeneration": "atom.plugin.vllm.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
}


def _set_plugin_mode() -> None:
    _set_framework_backbone("vllm")


def _register_minimax_m3_hf_config() -> None:
    from transformers import AutoConfig

    from atom.plugin.vllm.models.minimax_m3_config import MiniMaxM3Config

    try:
        AutoConfig.register(MiniMaxM3Config.model_type, MiniMaxM3Config)
    except ValueError as exc:
        if "already used by a Transformers config" not in str(exc):
            raise


def register_platform() -> Optional[str]:

    if disable_vllm_plugin:
        # return None instead of error because the flag can be used to
        # run pure vllm mode without ATOM plugin
        logger.info("Disable ATOM OOT plugin platforms")
        return None

    # Do not call _set_plugin_mode() here. SGLang (and other stacks) discover
    # vllm.platform_plugins and would set atom's backbone to "vllm" before
    # importing SGLang plugin modules — then atom.models.qwen3_5's ``if is_vllm():``
    # branch runs and requires vllm.model_executor.models.qwen3_5, which may be
    # absent. Backbone is set in register_model() for real vLLM runs.

    # return the ATOM platform to vllm
    return "atom.plugin.vllm.platform.ATOMPlatform"


def _patch_vllm_attention_process_weights_after_loading(attention) -> None:
    orig = attention.process_weights_after_loading

    if getattr(orig, "_atom_default_act_dtype_patched", False):
        return

    try:
        import inspect

        sig = inspect.signature(orig)
        act_dtype_param = sig.parameters.get("act_dtype")
        if (
            act_dtype_param is not None
            and act_dtype_param.default is not inspect._empty
        ):
            return
    except Exception:
        pass

    import functools

    @functools.wraps(orig)
    def wrapped(self, act_dtype: "torch.dtype" = torch.bfloat16):
        return orig(self, act_dtype)

    setattr(wrapped, "_atom_default_act_dtype_patched", True)
    attention.process_weights_after_loading = wrapped


def _patch_vllm_mxfp8_quant_config_for_atom_minimax_m3() -> None:
    """Let ATOM-owned MiniMax-M3 handle checkpoint MXFP8 metadata itself.

    vLLM's online MXFP8 shorthand is valid for CLI usage, but vLLM currently
    refuses to build that quant config from a checkpoint-side
    ``quantization_config``.  MiniMax-M3 is registered to ATOM's model wrapper,
    whose loader parses the original HF quantization metadata independently, so
    only vLLM's early quant_config construction needs to skip the checkpoint
    dict.
    """
    import functools

    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    orig = weight_utils.get_quant_config
    if getattr(orig, "_atom_minimax_m3_mxfp8_patched", False):
        return

    def is_atom_minimax_m3(model_config) -> bool:
        architectures = getattr(model_config, "architectures", None)
        if not architectures:
            hf_config = getattr(model_config, "hf_config", None)
            architectures = getattr(hf_config, "architectures", None)
        return bool(
            architectures
            and architectures[0]
            in {
                "MiniMaxM3SparseForCausalLM",
                "MiniMaxM3SparseForConditionalGeneration",
            }
        )

    @functools.wraps(orig)
    def wrapped(model_config, load_config):
        if (
            is_atom_minimax_m3(model_config)
            and getattr(model_config, "quantization", None) == "mxfp8"
            and getattr(model_config, "quantization_config", None) is not None
        ):
            return OnlineQuantizationConfig(
                args=getattr(model_config, "quantization_config")
            )
        return orig(model_config, load_config)

    setattr(wrapped, "_atom_minimax_m3_mxfp8_patched", True)
    weight_utils.get_quant_config = wrapped


def register_model() -> None:
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register")
        return

    _set_plugin_mode()
    _register_minimax_m3_hf_config()

    import vllm.model_executor.models.registry as vllm_model_registry

    any_updated = False
    for arch, qual in _VLLM_MODEL_REGISTRY_OVERRIDES.items():
        module_name, class_name = qual.split(":", 1)
        existing = vllm_model_registry.ModelRegistry.models.get(arch)
        if existing is not None:
            # If already overridden to the same target, skip re-registering.
            if (
                getattr(existing, "module_name", None) == module_name
                and getattr(existing, "class_name", None) == class_name
            ):
                continue

        logger.info(f"Register model {arch} to vLLM with {qual}")
        vllm_model_registry.ModelRegistry.register_model(arch, qual)
        any_updated = True

    # clear lru cache
    if any_updated:
        vllm_model_registry._try_load_model_cls.cache_clear()
        vllm_model_registry._try_inspect_model_cls.cache_clear()

    # patch attention process weights after loading
    # to avoid the specific handle in ATOM loader
    try:
        from vllm.attention.layer import Attention, MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import Attention, MLAAttention

    _patch_vllm_attention_process_weights_after_loading(Attention)
    _patch_vllm_attention_process_weights_after_loading(MLAAttention)
    _patch_vllm_mxfp8_quant_config_for_atom_minimax_m3()
    # vLLM's speculative decoder keeps an allow-list of attention metadata
    # classes. ATOM-vLLM uses its own metadata classes after attention
    # isolation, so extend that allow-list before MTP/Eagle proposal runs.
    apply_vllm_spec_decode_patch()

    # Patch vLLM graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in fused_allreduce_rmsnorm when model uses aiter collectives
    from atom.plugin.vllm.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()
