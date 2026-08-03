from types import SimpleNamespace

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context as get_vllm_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    SupportsMambaPrefixCaching,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from atom.models import kimi_k3 as kimi_k3_base
from atom.models.kimi_k3 import (
    KimiK3ForCausalLM as KimiK3ForCausalLMBase,
)
from atom.models.kimi_k3 import (
    KimiKDAAttention,
    _normalize_kimi_config,
)
from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM
from atom.utils.forward_context import get_forward_context as get_atom_forward_context


def _get_k3_state_shape(
    vllm_config: VllmConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    config = vllm_config.model_config.hf_text_config
    _normalize_kimi_config(config)
    num_spec = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    return MambaStateShapeCalculator.gated_delta_net_state_shape(
        vllm_config.parallel_config.tensor_parallel_size,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.linear_conv_kernel_dim,
        num_spec,
    )


def _get_k3_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    conv_dtype, _ = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
        vllm_config.cache_config.mamba_ssm_cache_dtype,
    )
    # FLA KDA accumulates the recurrent matrix in fp32 and rejects lower
    # precision initial state. Native ATOM uses the same fp32 state contract.
    return conv_dtype, torch.float32


class KimiKDAAttentionVllm(KimiKDAAttention, MambaBase):
    """Kimi-K3 KDA layer backed by vLLM-owned recurrent state."""

    def __init__(self, atom_config, quant_config=None, prefix: str = "") -> None:
        super().__init__(
            atom_config=atom_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        vllm_config = atom_config.plugin_config.vllm_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.num_k_heads = self.num_heads
        self.num_v_heads = self.num_heads
        self.head_k_dim = self.head_dim
        self.head_v_dim = self.head_dim
        self.num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self._atom_metadata = SimpleNamespace(gdn_metadata=None)
        self._atom_cache = SimpleNamespace(k_cache=None, v_cache=None)
        self._atom_kv_cache_data = {f"layer_{self.layer_num}": self._atom_cache}

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return _get_k3_state_dtype(self.vllm_config)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        vllm_context = get_vllm_forward_context()
        attn_metadata = vllm_context.attn_metadata
        if attn_metadata is None:
            return hidden_states.new_zeros(hidden_states.shape)

        if not isinstance(attn_metadata, dict):
            raise TypeError("Kimi-K3 vLLM attention metadata must be layer-indexed")
        gdn_metadata = attn_metadata[self.layer_name]
        if not isinstance(gdn_metadata, GDNAttentionMetadata):
            raise TypeError(
                f"Expected GDNAttentionMetadata for {self.layer_name}, "
                f"got {type(gdn_metadata).__name__}"
            )

        vllm_layer = vllm_context.no_compile_layers[self.layer_name]
        conv_state, ssm_state = vllm_layer.kv_cache
        self._atom_metadata.gdn_metadata = gdn_metadata
        self._atom_cache.k_cache = conv_state
        self._atom_cache.v_cache = ssm_state

        atom_context = get_atom_forward_context()
        previous_metadata = atom_context.attn_metadata
        previous_kv_cache_data = atom_context.kv_cache_data
        atom_context.attn_metadata = self._atom_metadata
        atom_context.kv_cache_data = self._atom_kv_cache_data
        try:
            output = super()._forward_impl(hidden_states)
        finally:
            atom_context.attn_metadata = previous_metadata
            atom_context.kv_cache_data = previous_kv_cache_data

        # vLLM pads token rows to the selected piecewise/full graph bucket,
        # while GDN metadata tracks only real tokens. The native KDA path
        # intentionally slices to num_actual_tokens; restore the graph bucket
        # width so this custom op matches its empty_like fake implementation.
        if output.shape[0] < hidden_states.shape[0]:
            output = torch.nn.functional.pad(
                output,
                (0, 0, 0, hidden_states.shape[0] - output.shape[0]),
            )
        return output


class KimiK3ForCausalLM(KimiK3ForCausalLMBase):
    def __init__(self, *args, **kwargs):
        original_kda_cls = kimi_k3_base.KimiKDAAttention
        kimi_k3_base.KimiKDAAttention = KimiKDAAttentionVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            kimi_k3_base.KimiKDAAttention = original_kda_cls


class KimiK3ForCausalLMVllm(ATOMMoEForCausalLM, IsHybrid, SupportsMambaPrefixCaching):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return _get_k3_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return _get_k3_state_shape(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
