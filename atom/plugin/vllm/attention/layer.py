from typing import Optional

from atom.config import get_current_atom_config
from atom.model_ops.attention_mla import MLAModules
from atom.plugin.vllm.attention.layer_mha import AttentionForVllmMHA
from atom.plugin.vllm.attention.layer_mla import (
    AttentionForVllmMLA,
    AttentionForVllmSparseMLA,
)
from atom.plugin.vllm.attention import ops as _atom_vllm_attention_ops  # noqa: F401


class AttentionForVllm:
    """Factory for ATOM-owned attention layers running under vLLM."""

    def __new__(
        cls,
        *args,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        **kwargs,
    ):
        atom_config = get_current_atom_config()
        if atom_config is None:
            raise RuntimeError("atom_config is required for vLLM plugin attention")

        if use_mla:
            if mla_modules is not None and mla_modules.indexer is not None:
                return AttentionForVllmSparseMLA(
                    *args, mla_modules=mla_modules, **kwargs
                )
            return AttentionForVllmMLA(*args, mla_modules=mla_modules, **kwargs)
        return AttentionForVllmMHA(*args, **kwargs)
