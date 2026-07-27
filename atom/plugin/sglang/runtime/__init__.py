"""Runtime utilities for ATOM's SGLang plugin integration."""

from atom.plugin.sglang.runtime.context import (
    SGLangForwardBatchMetadata,
    bind_current_forward_batch,
    get_current_forward_batch,
    is_draft_extend_mode,
    plugin_runtime_scope,
)
from atom.plugin.sglang.runtime.forward_context import SGLangPluginRuntime
from atom.plugin.sglang.runtime.load_config_patch import apply_load_config_patch
from atom.plugin.sglang.runtime.model_arch import (
    GLM52_DSA_ARCH,
    MODEL_ADAPTER_SPECS,
    MODEL_ARCH_SPECS,
    SGLangModelAdapterSpec,
    get_model_arch_spec,
    is_glm52_dsa_config,
)

apply_load_config_patch()

__all__ = [
    "GLM52_DSA_ARCH",
    "MODEL_ADAPTER_SPECS",
    "MODEL_ARCH_SPECS",
    "SGLangForwardBatchMetadata",
    "SGLangModelAdapterSpec",
    "SGLangPluginRuntime",
    "apply_load_config_patch",
    "bind_current_forward_batch",
    "get_current_forward_batch",
    "get_model_arch_spec",
    "is_draft_extend_mode",
    "is_glm52_dsa_config",
    "plugin_runtime_scope",
]
