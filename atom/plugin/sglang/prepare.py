from __future__ import annotations

import inspect
import logging
from contextlib import nullcontext
from typing import Any

from atom.plugin.prepare import _set_framework_backbone

logger = logging.getLogger("atom")


def _remap_quant_config_for_sglang_plugin(atom_config: Any, model_cls: type) -> None:
    quant_config = getattr(atom_config, "quant_config", None)
    if quant_config is None:
        return

    quant_config.remap_layer_name(
        atom_config.hf_config,
        packed_modules_mapping=getattr(model_cls, "packed_modules_mapping", {}),
        weights_mapper=getattr(model_cls, "hf_to_atom_mapper", {}),
        quant_exclude_name_mapping=getattr(model_cls, "quant_exclude_name_mapping", {}),
    )

    default_excludes = getattr(model_cls, "quant_default_exclude_layers", [])
    if default_excludes:
        quant_config.apply_default_exclude_layers(default_excludes)


def prepare_model(config: Any):
    """Prepare an ATOM model for SGLang plugin mode."""
    logger.info("Prepare model for plugin mode, the upper engine is sglang")
    _set_framework_backbone("sglang")

    from atom.plugin.sglang.runtime import resolve_model_arch_spec

    source_model_arch = (getattr(config, "architectures", None) or [""])[0]
    model_arch, model_adapter = resolve_model_arch_spec(config)
    if model_arch == "DeepseekV4ForCausalLM":
        from atom.plugin.sglang.deepseek_v4_bridge import (
            install_deepseek_v4_proxy_pool_patch,
        )

        install_deepseek_v4_proxy_pool_patch()
    elif model_arch == "KimiK3ForConditionalGeneration":
        from atom.plugin.sglang.kimi_k3_bridge import install_kimi_k3_pool_patch

        # Model construction precedes ModelRunner.init_memory_pool(), so install
        # the K3-only pool hooks here after the architecture is known.
        install_kimi_k3_pool_patch()

    # Import here to avoid partial initialization while SGLang discovers models.
    from atom.plugin.register import (
        _ATOM_SUPPORTED_MODELS,
        init_aiter_dist,
        register_ops_to_sglang,
        set_attn_cls,
    )

    if model_arch not in _ATOM_SUPPORTED_MODELS:
        supported_archs = list(_ATOM_SUPPORTED_MODELS.keys())
        raise ValueError(
            "ATOM does not support the required model architecture: "
            f"{source_model_arch or '<unknown>'} (resolved as "
            f"{model_arch or '<unknown>'}). "
            f"For now supported model architectures: {supported_archs}"
        )

    from atom.plugin.config import generate_atom_config_for_plugin_mode

    atom_config = generate_atom_config_for_plugin_mode(config)
    if getattr(atom_config, "enable_tbo", False):
        from atom.plugin.sglang.tbo import install_sglang_tbo_compat_patches

        install_sglang_tbo_compat_patches()

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    logger.info("ATOM model class for %s is %s", model_arch, model_cls)

    if model_adapter.prepare_draft_model_config is not None:
        model_adapter.prepare_draft_model_config(atom_config, config)
    if model_adapter.prepare_config is not None:
        model_adapter.prepare_config(atom_config, model_arch)
    else:
        _remap_quant_config_for_sglang_plugin(atom_config, model_cls)

    # Qwen3-Next/Qwen3.5 model layers are already written against SGLang's
    # RadixAttention contract. Keep SGLang's native attention backend so
    # speculative TARGET_VERIFY uses its proven metadata/cache path.
    if model_arch not in {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        register_ops_to_sglang(atom_config=atom_config)

    # Qwen3.5 is the only family that runs DFLASH on the ATOM plugin today.
    # SGLang's draft greedy sampler assumes a head without `shard_indices` is
    # not vocab-parallel, which is wrong for ATOM's ParallelLMHead and yields
    # garbage draft tokens at TP>1; see dflash_lm_head_bridge for the details.
    if model_arch in {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        from atom.plugin.sglang.dflash_lm_head_bridge import (
            install_dflash_lm_head_patch,
        )

        install_dflash_lm_head_patch()
    set_attn_cls()

    # Init aiter dist for using aiter custom collective ops.
    init_aiter_dist(config=atom_config)

    # Patch SGLang graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in aiter collectives when model uses aiter's
    # custom all_reduce (same fix as atom/plugin/vllm/graph_capture_patch.py).
    from atom.plugin.sglang.patches.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()

    construct_context = (
        model_adapter.construction_context()
        if model_adapter.construction_context is not None
        else nullcontext()
    )
    with construct_context:
        init_params = inspect.signature(model_cls.__init__).parameters
        if "atom_config" in init_params:
            model = model_cls(atom_config=atom_config)
        elif "config" in init_params:
            model = model_cls(config=atom_config)
        else:
            model = model_cls(atom_config)
    if not hasattr(model, "atom_config"):
        model.atom_config = atom_config
    return model


def prepare_model_for_sglang(config: Any):
    """Backward-compatible alias for SGLang plugin model preparation."""
    return prepare_model(config)
