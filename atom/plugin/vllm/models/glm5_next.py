"""GLM-5.3-Flash (glm5_next) bridge for the ATOM vLLM plugin.

GLM-5.3-Flash is Kimi-K3's hybrid shape: KDA linear-attention layers next to
sparse-MLA layers. Native ATOM drives both from its own forward context, which
`vllm serve` never populates, so each half needs the same bridging the other
hybrid models already have:

* KDA: reuse Kimi-K3's vLLM layer wholesale. GLM's KDA *is* Kimi's with the
  output gate factorised (`Glm5NextKDAAttention` subclasses `KimiKDAAttention`),
  so recurrent state, metadata and cudagraph handling carry over unchanged.
* Sparse MLA: the shared `Indexer` already carries a plugin-mode implementation
  (`IndexerDecoratorForPluginMode`). The pooled (kpool) path still reads ATOM's
  `Context`; under `vllm serve` that context is unset on the profile pass, so
  the kpool op treats `context is None` as a dummy run instead of crashing.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import IsHybrid

from atom.models import glm5_next as glm5_next_base
from atom.models.glm5_next import (
    Glm5NextForConditionalGeneration as Glm5NextForConditionalGenerationBase,
)
from atom.models.glm5_next import (
    Glm5NextKDAAttention,
    _normalize_glm5_next_config,
    _text_config,
)
from atom.models.glm5_next_mtp import Glm5NextMTP as Glm5NextMTPBase
from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM
from atom.plugin.vllm.models.kimi_k3 import KimiKDAAttentionVllm


def _glm5_text_config(vllm_config: VllmConfig):
    """The text config with the KDA aliases the state calculators read."""
    config = _text_config(vllm_config.model_config.hf_config)
    _normalize_glm5_next_config(config)
    return config


def _get_glm5_state_shape(
    vllm_config: VllmConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    config = _glm5_text_config(vllm_config)
    num_spec = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    return MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
        num_heads=config.linear_num_value_heads,
        head_dim=config.linear_value_head_dim,
        num_k_heads=config.linear_num_key_heads,
        head_k_dim=config.linear_key_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        num_spec=num_spec,
    )


def _get_glm5_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    return MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
    )


class Glm5NextKDAAttentionVllm(Glm5NextKDAAttention, KimiKDAAttentionVllm):
    """GLM-5.3 KDA layer backed by vLLM-owned recurrent state.

    The MRO runs GLM's gate fold over Kimi's vLLM plumbing: GLM's `__init__`
    adds the low-rank gate factors and its `process_weights_after_loading`
    folds them into `in_proj`, while everything that talks to vLLM -- state
    shape/dtype, the KDA metadata backend, the splitting-op forward -- comes
    from Kimi's layer.
    """

    def process_weights_after_loading(self, *args, **kwargs) -> None:
        """vLLM passes an activation dtype here; GLM's fold takes no arguments."""
        return Glm5NextKDAAttention.process_weights_after_loading(self)


class Glm5NextForConditionalGeneration(Glm5NextForConditionalGenerationBase):
    """Native GLM-5.3-Flash body with the vLLM-backed KDA layer swapped in."""

    def __init__(self, *args, **kwargs):
        original_kda_cls = glm5_next_base.Glm5NextKDAAttention
        glm5_next_base.Glm5NextKDAAttention = Glm5NextKDAAttentionVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            glm5_next_base.Glm5NextKDAAttention = original_kda_cls


class Glm5NextForConditionalGenerationVllm(ATOMMoEForCausalLM, IsHybrid):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return _get_glm5_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return _get_glm5_state_shape(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()


class Glm5NextMTPVllm(Glm5NextMTPBase):
    """Build GLM-5.3-Flash NextN MTP blocks with the vLLM-backed KDA layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from atom.plugin.vllm.model_wrapper import _generate_atom_config_from_vllm_config
        self.atom_config = _generate_atom_config_from_vllm_config(vllm_config)
        self.vllm_config = vllm_config
        original_kda_cls = glm5_next_base.Glm5NextKDAAttention
        glm5_next_base.Glm5NextKDAAttention = Glm5NextKDAAttentionVllm
        try:
            super().__init__(atom_config=self.atom_config, prefix=prefix)
        finally:
            glm5_next_base.Glm5NextKDAAttention = original_kda_cls

    def load_weights(self, weights):
        from atom.model_loader.loader import load_model_in_plugin_mode
        from atom.plugin.vllm.model_wrapper import _MTP_DRAFT_MODEL_ARCHES
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
            spec_decode="Glm5NextMTPModel" in _MTP_DRAFT_MODEL_ARCHES,
            hf_config_override=None,
            model_name_or_path_override=None,
        )
