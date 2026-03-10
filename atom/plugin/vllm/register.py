from typing import Optional
import logging

import torch
from atom.plugin.prepare import _set_framework_backbone
from atom.utils import envs

logger = logging.getLogger("atom")

# this flag is used to enable the vllm plugin mode
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN
disable_vllm_plugin_attention = envs.ATOM_DISABLE_VLLM_PLUGIN_ATTENTION

# those 2 models are covering most of dense and moe models
ATOM_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMForCausalLM"
ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMMoEForCausalLM"

# when register new model to vllm, add here
# Keys is from hf config arch name
_VLLM_MODEL_REGISTRY_OVERRIDES: dict[str, str] = {
    "Qwen3ForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GptOssForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV3ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Glm4MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
}


def _set_plugin_mode() -> None:
    _set_framework_backbone("vllm")


def register_platform() -> Optional[str]:

    if disable_vllm_plugin:
        # return None instead of error because the flag can be used to
        # run pure vllm mode without ATOM plugin
        logger.info("Disable ATOM OOT plugin platforms")
        return None

    _set_plugin_mode()

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


def set_default_quant_scales(
    layer: torch.nn.Module, register_buffer: bool = False
) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._prob_scale_float = 1.0


def _replace_vllm_mla_attention_process_weights_after_loading() -> None:
    try:
        from vllm.attention.layer import MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import MLAAttention

    def _process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(self.impl, "process_weights_after_loading"):
            if disable_vllm_plugin_attention:
                self.impl.process_weights_after_loading(act_dtype)
            else:
                self.impl.process_weights_after_loading()

        set_default_quant_scales(self, register_buffer=False)

    MLAAttention.process_weights_after_loading = _process_weights_after_loading


def register_model() -> None:
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register")
        return

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
    _replace_vllm_mla_attention_process_weights_after_loading()
    _patch_vllm_attention_process_weights_after_loading(MLAAttention)
