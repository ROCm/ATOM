from __future__ import annotations

import inspect
import logging
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

    model_arch = config.architectures[0]
    if model_arch == "DeepseekV4ForCausalLM":
        from atom.plugin.sglang.deepseek_v4_bridge import (
            install_deepseek_v4_proxy_pool_patch,
        )

        install_deepseek_v4_proxy_pool_patch()

    # Import here to avoid partial initialization while SGLang discovers models.
    from atom.plugin.register import (
        _ATOM_SUPPORTED_MODELS,
        init_aiter_dist,
        register_ops_to_sglang,
        set_attn_cls,
    )

    supported_models = dict(_ATOM_SUPPORTED_MODELS)
    if model_arch == "LlamaForCausalLMEagle3":
        from atom.models.eagle3_llama import Eagle3LlamaModel

        supported_models[model_arch] = Eagle3LlamaModel

    if model_arch not in supported_models:
        supported_archs = list(supported_models.keys())
        raise ValueError(
            f"ATOM does not support the required model architecture: {model_arch}. "
            f"For now supported model architectures: {supported_archs}"
        )

    from atom.plugin.config import generate_atom_config_for_plugin_mode

    atom_config = generate_atom_config_for_plugin_mode(config)
    target_hf_config = getattr(atom_config, "hf_config", None)
    target_hf_config = getattr(target_hf_config, "text_config", target_hf_config)
    target_num_hidden_layers = int(
        getattr(target_hf_config, "num_hidden_layers", 0) or 0
    )
    # `generate_atom_config_for_plugin_mode` is built from SGLang global
    # ServerArgs, which always describe the target model.  External draft
    # runners (for example EAGLE3) pass their own HF config into this wrapper,
    # so overwrite the model-local pieces here before constructing ATOM modules.
    # Otherwise the draft modeling is built with target dimensions while
    # SGLang allocates draft KV pools from the draft config.
    atom_config.hf_config = config
    model_path = getattr(config, "_name_or_path", None) or getattr(
        config, "name_or_path", None
    )
    if model_path:
        atom_config.model = model_path
    from atom.config import QuantizationConfig

    atom_config.quant_config = QuantizationConfig(
        atom_config.hf_config,
        online_quant_config=getattr(atom_config, "online_quant_config", None),
    )

    model_cls = supported_models[model_arch]
    logger.info("ATOM model class for %s is %s", model_arch, model_cls)

    from atom.plugin.sglang.runtime import get_model_arch_spec

    model_adapter = get_model_arch_spec(model_arch)
    if model_adapter.prepare_config is not None:
        model_adapter.prepare_config(atom_config, model_arch)
    else:
        _remap_quant_config_for_sglang_plugin(atom_config, model_cls)

    register_ops_to_sglang(atom_config=atom_config)
    set_attn_cls()

    # Init aiter dist for using aiter custom collective ops.
    init_aiter_dist(config=atom_config)

    # Patch SGLang graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in aiter collectives when model uses aiter's
    # custom all_reduce (same fix as atom/plugin/vllm/graph_capture_patch.py).
    from atom.plugin.sglang.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()

    init_params = inspect.signature(model_cls.__init__).parameters
    init_kwargs = {}
    if model_arch == "LlamaForCausalLMEagle3" and "layer_offset" in init_params:
        init_kwargs["layer_offset"] = target_num_hidden_layers
    if "atom_config" in init_params:
        model = model_cls(atom_config=atom_config, **init_kwargs)
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
