import torch

from aiter.dist.parallel_state import get_tensor_model_parallel_rank
from atom.model_config.qwen3_next import Qwen3NextConfig
from atom.models import qwen3_next as qwen3_next_base
from atom.models.qwen3_next import (
    Qwen3NextForCausalLM as Qwen3NextForCausalLMBase,
    Qwen3NextGatedDeltaNet,
)
from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class Qwen3NextGatedDeltaNetVllm(Qwen3NextGatedDeltaNet, MambaBase):
    def __init__(
        self,
        atom_config: Qwen3NextConfig,
        quant_config=None,
        speculative_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            atom_config=atom_config,
            quant_config=quant_config,
            speculative_config=speculative_config,
            prefix=prefix,
        )
        self.model_config = atom_config.plugin_config.vllm_config.model_config
        self.cache_config = atom_config.plugin_config.vllm_config.cache_config
        self.tp_rank = get_tensor_model_parallel_rank()
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

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


class Qwen3NextForCausalLM(Qwen3NextForCausalLMBase):
    def __init__(self, *args, **kwargs):
        original_gdn_cls = qwen3_next_base.Qwen3NextGatedDeltaNet
        qwen3_next_base.Qwen3NextGatedDeltaNet = Qwen3NextGatedDeltaNetVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            qwen3_next_base.Qwen3NextGatedDeltaNet = original_gdn_cls


class Qwen3NextForCausalLMVllm(ATOMMoEForCausalLM, IsHybrid):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )

        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
