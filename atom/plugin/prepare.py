from typing import Any
import logging

logger = logging.getLogger("atom")

# all of the supported frameworks, including server mode and plugin mode
_SUPPORTED_FRAMEWORKS = ["vllm", "sglang", "sgl", "atom"]

# supported frameworks for plugin mode
_SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE = ["vllm", "sglang", "sgl"]

# default is atom for server mode
_CURRENT_FRAMEWORK = "atom"


def is_sglang() -> bool:
    global _CURRENT_FRAMEWORK
    if _CURRENT_FRAMEWORK is None:
        raise ValueError("_CURRENT_FRAMEWORK must be set before use")
    return bool(_CURRENT_FRAMEWORK.lower() in ["sglang", "sgl"])


def is_vllm() -> bool:
    global _CURRENT_FRAMEWORK
    if _CURRENT_FRAMEWORK is None:
        raise ValueError("_CURRENT_FRAMEWORK must be set before use")
    return bool(_CURRENT_FRAMEWORK.lower() in ["vllm"])


def is_plugin_mode() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in _SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE)


def _set_framework_backbone(framework: str) -> None:
    if framework.lower() not in _SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework {framework} for ATOM to plug in")
    global _CURRENT_FRAMEWORK
    _CURRENT_FRAMEWORK = framework


def prepare_model(config: Any, engine: str):
    '''
    Prepare the model to upper framework, including
    register custom ops and init aiter dist
    '''
    logging.info(f'Prepare model for plugin mode, the upper engine is {engine}')

    _set_framework_backbone(engine)

    # different engine passed different config
    if is_vllm():
        model_arch = config.model_config.architectures[0]
    elif is_sglang():
        model_arch = config.architectures[0]

    # import here to avoid partial initialization
    from .register import (
        _ATOM_SUPPORTED_MODELS,
        register_ops_to_vllm,
        register_ops_to_sglang,
        init_aiter_dist,
        set_attn_cls,
    )

    if model_arch not in _ATOM_SUPPORTED_MODELS:
        logger.warning(f"ATOM does not support the required model architecture: {model_arch}")

    from atom.plugin.config import generate_atom_config_for_plugin_mode
    atom_config = generate_atom_config_for_plugin_mode(config)

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    logger.info(f'ATOM model class for {model_arch} is {model_cls}')

    if is_vllm():
        register_ops_to_vllm(atom_config=atom_config)
    elif is_sglang():
        register_ops_to_sglang(atom_config=atom_config)

    set_attn_cls()

    # init aiter dist for using aiter custom collective ops
    init_aiter_dist(config=atom_config)

    return model_cls(atom_config=atom_config)
