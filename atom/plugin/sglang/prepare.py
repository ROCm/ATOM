from __future__ import annotations

import logging
from typing import Any

from atom.plugin.prepare import _set_framework_backbone

logger = logging.getLogger("atom")


def prepare_model(config: Any):
    """Prepare an ATOM model for SGLang plugin mode."""
    logger.info("Prepare model for plugin mode, the upper engine is sglang")
    _set_framework_backbone("sglang")

    model_arch = config.architectures[0]

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
            f"ATOM does not support the required model architecture: {model_arch}. "
            f"For now supported model architectures: {supported_archs}"
        )

    from atom.plugin.config import generate_atom_config_for_plugin_mode

    atom_config = generate_atom_config_for_plugin_mode(config)

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    logger.info("ATOM model class for %s is %s", model_arch, model_cls)

    from atom.plugin.sglang.runtime import get_model_arch_spec

    model_adapter = get_model_arch_spec(model_arch)
    if model_adapter.prepare_config is not None:
        model_adapter.prepare_config(atom_config, model_arch)

    register_ops_to_sglang(atom_config=atom_config)
    set_attn_cls()

    # Init aiter dist for using aiter custom collective ops.
    init_aiter_dist(config=atom_config)

    # Patch SGLang graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in aiter collectives when model uses aiter's
    # custom all_reduce (same fix as atom/plugin/vllm/graph_capture_patch.py).
    from atom.plugin.sglang.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()

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


def prepare_model_for_sglang(config: Any):
    """Backward-compatible alias for SGLang plugin model preparation."""
    return prepare_model(config)
