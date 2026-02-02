# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from flash_attn import flash_attn_with_kvcache
from typing import Optional

import torch
from torch import nn


from atom.utils import mark_spliting_op
from .attention_mla import MLAModules
from atom.config import get_current_atom_config
from atom.utils.selector import get_attn_backend


def fake_(
    q: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
    qkv: torch.Tensor,
) -> torch.Tensor:
    output_shape = list(q.shape)
    if use_mla:
        output_shape[-1] = 7168
    # If we fusion rmsnorm and quant, the input dtype is fp8, but actually we use bf16 for output.
    atom_config = get_current_atom_config()
    output_dtype = atom_config.torch_dtype
    output = torch.zeros(output_shape, dtype=output_dtype, device=q.device)

    return output


# Dynamo will not try to inspect any of the internal operations for prefill or decode
# This way, although attention operation is complicated,
# we can still capture the model's computation graph as a full-graph
@mark_spliting_op(is_custom=True, gen_fake=fake_, mutates_args=[])
def unified_attention_with_output_base(
    q: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
    qkv: torch.Tensor,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.impl.forward(q, k, v, positions, q_scale, qkv)


def linear_attention_with_output_base_fask(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return

@mark_spliting_op(is_custom=True, gen_fake=linear_attention_with_output_base_fask, mutates_args=[])
def linear_attention_with_output_base(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.impl.forward(mixed_qkv, b, a, core_attn_out)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        sinks: Optional[nn.Parameter] = None,
        per_layer_sliding_window: Optional[int] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = None,
        q_norm: Optional[torch.nn.Module] = None,
        k_norm: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.layer_num = layer_num
        self.mla_modules = mla_modules
        self.use_mla = use_mla
        self.base_attention = None
        self.kv_cache = torch.tensor([])
        self.indexer = mla_modules.indexer if mla_modules is not None else None
        self.sinks = sinks

        atom_config = get_current_atom_config()
        dtype = atom_config.torch_dtype
        block_size = atom_config.kv_cache_block_size
        self.attn_backend = get_attn_backend(
            block_size,
            use_mla=self.use_mla,
        )
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            layer_num,
            mla_modules,
            sinks=sinks,
            sliding_window=per_layer_sliding_window,
            rotary_emb=rotary_emb,
            dtype=dtype,
            q_norm=q_norm,
            k_norm=k_norm,
        )

        compilation_config = atom_config.compilation_config
        default_name = f"MLA_{layer_num}" if self.use_mla else f"MHA_{layer_num}"
        self.layer_name = prefix if prefix is not None else default_name
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer: {}".format(self.layer_name))
        compilation_config.static_forward_context[self.layer_name] = self

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor = None,
        q_scale: Optional[torch.Tensor] = None,
        qkv: torch.Tensor = None,
    ):
        output = torch.ops.aiter.unified_attention_with_output_base(
            q, q_scale, k, v, positions, self.layer_name, self.use_mla, qkv
        )
        return output


class LinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        key_dim,
        value_dim,
        dt_bias = None,
        A_log = None,
        conv1d = None,
        activation = None,
        layer_num = 0,
        prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dt_bias = dt_bias
        self.A_log = A_log
        self.conv1d = conv1d
        self.activation = activation
        self.layer_num = layer_num
        self.base_linear_attention = None

        atom_config = get_current_atom_config()
        dtype = atom_config.torch_dtype
        block_size = atom_config.kv_cache_block_size
        self.attn_backend = get_attn_backend(
            block_size,
            use_gdn=True,
        )
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            self.hidden_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.key_dim,
            self.value_dim,
            dt_bias,
            A_log,
            conv1d,
            activation,
            layer_num,
            **kwargs,
        )

        compilation_config = atom_config.compilation_config
        default_name = f"Linear_{layer_num}"
        self.layer_name = prefix if prefix is not None else default_name
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer: {}".format(self.layer_name))
        compilation_config.static_forward_context[self.layer_name] = self


    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        output = torch.ops.aiter.linear_attention_with_output_base(
            mixed_qkv, b, a, core_attn_out, self.layer_name
        )
        return output