import logging
from typing import Any

logger = logging.getLogger("atom")

# all of the supported frameworks, including server mode and plugin mode
_SUPPORTED_FRAMEWORKS = ["vllm", "sglang", "sgl", "atom", "rtpllm"]

# supported frameworks for plugin mode
_SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE = ["vllm", "sglang", "sgl", "rtpllm"]

# default is atom for server mode
_CURRENT_FRAMEWORK = "atom"


def is_sglang() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in ["sglang", "sgl"])


def is_vllm() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in ["vllm"])


def is_rtpllm() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in ["rtpllm"])


def is_plugin_mode() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in _SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE)


def _set_framework_backbone(framework: str) -> None:
    if framework.lower() not in _SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework {framework} for ATOM to plug in")
    global _CURRENT_FRAMEWORK
    _CURRENT_FRAMEWORK = framework


def _instantiate_prepared_model(config: Any, atom_config: Any, model_cls: Any):
    try:
        model = model_cls(atom_config=atom_config)
    except TypeError as exc:
        # Some SGLang plugin models keep SGLang's native wrapper constructor
        # and only swap their internal language_model with an ATOM model.
        # Those classes accept `config=...` instead of `atom_config=...`.
        if "atom_config" not in str(exc):
            raise
        model = model_cls(config=config)
    if not hasattr(model, "atom_config"):
        model.atom_config = atom_config
    return model


def _prepare_model_atom_sglang(
    config: Any,
    atom_config: Any,
    model_arch: str,
    model_cls: Any,
    register_ops_to_sglang: Any,
    set_attn_cls: Any,
    init_aiter_dist: Any,
):
    if model_arch in {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        from atom.plugin.sglang.models.qwen3_5 import (
            apply_prepare_model_adaptations,
        )

        apply_prepare_model_adaptations(atom_config, model_arch)

    # Qwen3-Next and Qwen3.5 series models keep the upstream attention backend path.
    if model_arch not in {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        register_ops_to_sglang(atom_config=atom_config)
    set_attn_cls()

    # init aiter dist for using aiter custom collective ops
    init_aiter_dist(config=atom_config)

    # Patch SGLang graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in aiter collectives when model uses aiter's
    # custom all_reduce (same fix as atom/plugin/vllm/graph_capture_patch.py)
    from atom.plugin.sglang.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()
    return _instantiate_prepared_model(config, atom_config, model_cls)


def _prepare_model_atom_rtpllm(
    config: Any,
    atom_config: Any,
    model_cls: Any,
    set_attn_cls: Any,
    init_aiter_dist: Any,
):
    # rtp-llm plugin mode uses this entry point for direct model construction.
    # Ensure quant layer name remap/exclude processing is done BEFORE model init,
    # otherwise layer quant_type gets fixed with stale rules.
    conv1d_exclude = "model.layers.*.linear_attn.conv1d"
    if conv1d_exclude not in atom_config.quant_config.exclude_layers:
        atom_config.quant_config.exclude_layers.append(conv1d_exclude)
        logger.info(
            "rtp-llm plugin: add quant exclude for incompatible layer pattern: %s",
            conv1d_exclude,
        )

    atom_config.quant_config.remap_layer_name(
        atom_config.hf_config,
        packed_modules_mapping=getattr(model_cls, "packed_modules_mapping", {}),
        quant_exclude_name_mapping=getattr(model_cls, "quant_exclude_name_mapping", {}),
    )

    set_attn_cls()

    # init aiter dist for using aiter custom collective ops
    init_aiter_dist(config=atom_config)

    return _instantiate_prepared_model(config, atom_config, model_cls)


def prepare_model(config: Any, engine: str):
    """
    Prepare ATOM model for plugin mode upper frameworks.
    """
    logger.info(f"Prepare model for plugin mode, the upper engine is {engine}")

    _set_framework_backbone(engine)

    if not (is_sglang() or is_rtpllm()):
        raise ValueError(
            f"prepare_model does not support engine {engine!r} "
            f"with config type {type(config)}"
        )

    # import here to avoid partial initialization
    from .register import (
        _ATOM_SUPPORTED_MODELS,
        # register_ops_to_vllm,
        register_ops_to_sglang,
        init_aiter_dist,
        set_attn_cls,
    )

    from atom.plugin.config import generate_atom_config_for_plugin_mode

    atom_config = generate_atom_config_for_plugin_mode(config)

    if not hasattr(atom_config.hf_config, "architectures"):
        raise ValueError("Failed to parse model architectures from HF config")
    model_arch = atom_config.hf_config.architectures[0]

    if model_arch not in _ATOM_SUPPORTED_MODELS:
        supported_archs = list(_ATOM_SUPPORTED_MODELS.keys())
        raise ValueError(
            f"ATOM does not support the required model architecture: {model_arch}. "
            f"For now supported model architectures: {supported_archs}"
        )

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    logger.info(f"ATOM model class for {model_arch} is {model_cls}")

    if is_rtpllm():
        return _prepare_model_atom_rtpllm(
            config,
            atom_config,
            model_cls,
            set_attn_cls,
            init_aiter_dist,
        )

    if is_sglang():
        return _prepare_model_atom_sglang(
            config,
            atom_config,
            model_arch,
            model_cls,
            register_ops_to_sglang,
            set_attn_cls,
            init_aiter_dist,
        )

    raise ValueError(f"prepare_model does not support engine {engine!r}")
