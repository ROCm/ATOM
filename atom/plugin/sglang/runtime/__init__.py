"""Runtime utilities for ATOM's SGLang plugin integration."""

from atom.plugin.sglang.runtime.load_config_patch import apply_load_config_patch
from atom.plugin.sglang.runtime.context import (
    SGLangForwardBatchMetadata,
    bind_current_forward_batch,
    get_current_forward_batch,
    plugin_runtime_scope,
)
from atom.plugin.sglang.runtime.forward_context import SGLangPluginRuntime
from atom.plugin.sglang.runtime.model_arch import (
    MODEL_ADAPTER_SPECS,
    MODEL_ARCH_SPECS,
    SGLangModelAdapterSpec,
    get_model_arch_spec,
)

apply_load_config_patch()

__all__ = [
    "apply_load_config_patch",
    "MODEL_ADAPTER_SPECS",
    "MODEL_ARCH_SPECS",
    "SGLangForwardBatchMetadata",
    "SGLangModelAdapterSpec",
    "SGLangPluginRuntime",
    "bind_current_forward_batch",
    "get_current_forward_batch",
    "get_model_arch_spec",
    "plugin_runtime_scope",
]
