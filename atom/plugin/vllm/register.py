from typing import Optional
import logging

import torch
from atom.plugin.prepare import _set_framework_backbone
from atom.utils import envs
from atom.plugin.vllm.mla_patch import patch_vllm_mla_attention

logger = logging.getLogger("atom")

# this flag is used to enable the vllm plugin mode
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN
disable_vllm_plugin_attention = envs.ATOM_DISABLE_VLLM_PLUGIN_ATTENTION
disable_vllm_mori = envs.ATOM_DISABLE_VLLM_MORI
_warned_vllm_mori_import_opt_out = False

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
    "Glm4MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GlmMoeDsaForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3NextForCausalLM": "atom.models.qwen3_next:Qwen3NextForCausalLMVllm",
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    "KimiK25ForConditionalGeneration": "atom.plugin.vllm.models.kimi_k25:KimiK25ForConditionalGeneration",
}


def _set_plugin_mode() -> None:
    _set_framework_backbone("vllm")


def _maybe_disable_vllm_mori_import() -> None:
    """Control whether vLLM can see its own optional ``mori`` package.

    In ATOM-vLLM plugin mode, ATOM provides the model implementation while vLLM
    mainly acts as the runtime shell. For the MoE path, ATOM uses ATOM's own
    mori integration instead of vLLM's. However, newer vLLM versions may import
    their optional mori path very early during quantization/config validation,
    before the model is actually built. If the environment contains a mori build
    that matches ATOM's path but not vLLM's ABI expectations, that eager import
    can fail the whole process even though the run never intends to use vLLM's
    mori implementation.

    To keep plugin mode stable, we hide vLLM's own ``has_mori()`` detection by
    default. Users can opt out with ``ATOM_DISABLE_VLLM_MORI=0`` if they
    intentionally want vLLM to use its mori path, but then they are responsible
    for installing a mori build that is compatible with their vLLM environment.
    """
    global _warned_vllm_mori_import_opt_out

    if not disable_vllm_mori:
        if not _warned_vllm_mori_import_opt_out:
            logger.warning(
                "ATOM plugin: keeping vLLM mori detection enabled because "
                "ATOM_DISABLE_VLLM_MORI=0. In ATOM-vLLM plugin mode, "
                "ATOM normally uses ATOM's mori path instead of vLLM's. "
                "If vLLM imports its own mori path, ABI/environment conflicts "
                "may occur. Please make sure you installed a mori version "
                "compatible with this vLLM environment."
            )
            _warned_vllm_mori_import_opt_out = True
        return

    try:
        import vllm.utils.import_utils as vllm_import_utils
    except ImportError:
        logger.debug(
            "ATOM plugin: skip disabling vLLM mori import; import utils unavailable"
        )
        return

    has_mori = getattr(vllm_import_utils, "has_mori", None)
    if has_mori is None:
        logger.debug("ATOM plugin: skip disabling vLLM mori import; has_mori missing")
        return

    if getattr(has_mori, "_atom_vllm_mori_disabled", False):
        return

    def _disabled_has_mori() -> bool:
        # ATOM plugin mode intentionally hides vLLM's optional mori path.
        return False

    setattr(_disabled_has_mori, "_atom_vllm_mori_disabled", True)
    vllm_import_utils.has_mori = _disabled_has_mori
    logger.info(
        "ATOM plugin: disabled vLLM mori detection in plugin mode so ATOM "
        "uses its own mori path without pulling in vLLM's optional mori import"
    )


def register_platform() -> Optional[str]:

    if disable_vllm_plugin:
        # return None instead of error because the flag can be used to
        # run pure vllm mode without ATOM plugin
        logger.info("Disable ATOM OOT plugin platforms")
        return None

    _maybe_disable_vllm_mori_import()
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


def register_model() -> None:
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register")
        return

    _maybe_disable_vllm_mori_import()
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

    patch_vllm_mla_attention()
    # patch attention process weights after loading
    # to avoid the specific handle in ATOM loader
    try:
        from vllm.attention.layer import Attention, MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import Attention, MLAAttention

    _patch_vllm_attention_process_weights_after_loading(Attention)
    _patch_vllm_attention_process_weights_after_loading(MLAAttention)

    # Patch vLLM graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in fused_allreduce_rmsnorm when model uses aiter collectives
    from atom.plugin.vllm.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()
