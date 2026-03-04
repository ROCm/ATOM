# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from flash_attn import flash_attn_with_kvcache
from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch import nn
import triton
import triton.language as tl


from atom.utils import mark_spliting_op
from .attention_mla import MLAModules
from atom.config import get_current_atom_config
from atom.utils.selector import get_attn_backend


# frontend interface class for constructing attention
# op in model file
class Attention:
    def __new__(cls, *args, **kwargs):
        from atom.model_ops import Attention

        return Attention(*args, **kwargs)


# this triton kernel is used to fetch the stored kv in
# kv cache for computing the extend path(chunked prefill)
# and it can be used for both server mode and plugin mode
@triton.jit
def cp_mha_gather_cache_kernel(
    key_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
    value_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    block_table_ptr,  # [num_batches, max_block_num]
    cu_seqlens_kv_ptr,  # [num_batches + 1]
    token_to_batch_ptr,  # [max_cum_tokens]
    seq_start_ptr,  # [num_batches]
    k_scale_ptr,  # [1] / [num_blocks, num_kv_heads, page_size]
    v_scale_ptr,
    num_heads,
    head_size,
    x,
    max_block_num,
    DEQUANT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_FORMAT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    key_ptr_offset = key_ptr + token_id * head_size * num_heads + head_id * head_size
    value_ptr_offset = (
        value_ptr + token_id * head_size * num_heads + head_id * head_size
    )
    batch_idx = tl.load(token_to_batch_ptr + token_id)
    batch_start = tl.load(seq_start_ptr + batch_idx)
    token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
    batch_offset = token_id - token_start + batch_start
    block_offset = batch_offset // PAGE_SIZE
    block_id = tl.load(block_table_ptr + max_block_num * batch_idx + block_offset).to(
        tl.int64
    )
    slot_id = batch_offset % PAGE_SIZE

    if CACHE_FORMAT == "NHD":
        # for kv cache layout as
        # K: [num_blocks, page_size, num_head, head_dim]
        # V: [num_blocks, page_size, num_head, head_dim]
        key_cache_ptr_offset = (
            key_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + slot_id * num_heads * head_size
            + head_id * head_size
        )
        value_cache_ptr_offset = (
            value_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + slot_id * num_heads * head_size
            + head_id * head_size
        )
        k_reg = tl.load(key_cache_ptr_offset + col_offsets)
        v_reg = tl.load(value_cache_ptr_offset + col_offsets)
        if DEQUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)
            k_dtype = k_reg.dtype
            v_dtype = v_reg.dtype
            k_reg = (k_reg.to(tl.float32) * k_scale).to(k_dtype)
            v_reg = (v_reg.to(tl.float32) * v_scale).to(v_dtype)
        tl.store(key_ptr_offset + col_offsets, k_reg)
        tl.store(value_ptr_offset + col_offsets, v_reg)

    elif CACHE_FORMAT == "SHUFFLE":
        # for kv cache layout as
        # K: [num_blocks, num_head, head_dim // x, page_size, x]
        # V: [num_blocks, num_head, page_size // x, head_dim, x]
        key_cache_ptr_offset = (
            key_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + head_id * head_size * PAGE_SIZE
            + slot_id * x
        )
        value_cache_ptr_offset = (
            value_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + head_id * head_size * PAGE_SIZE
            + (slot_id // x) * head_size * x
            + slot_id % x
        )
        k_reg_offset = col_offsets // x * PAGE_SIZE * x + col_offsets % x
        v_reg_offset = col_offsets * x
        k_reg = tl.load(key_cache_ptr_offset + k_reg_offset)
        v_reg = tl.load(value_cache_ptr_offset + v_reg_offset)
        if DEQUANT:
            k_scale = 1.0
            v_scale = 1.0
            k_reg = k_reg.to(tl.float32) * k_scale
            v_reg = v_reg.to(tl.float32) * v_scale
        tl.store(key_ptr_offset + col_offsets, k_reg)
        tl.store(value_ptr_offset + col_offsets, v_reg)


def cp_mha_gather_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_tables: torch.Tensor,
    k_scales: Optional[torch.Tensor],
    v_scales: Optional[torch.Tensor],
    cu_seqlens_kv: torch.Tensor,
    token_to_batch: torch.Tensor,
    seq_starts: torch.Tensor,
    dequant: bool,
    kv_cache_layout: str,
    total_tokens: int,
):
    assert kv_cache_layout in [
        "NHD",
        "SHUFFLE",
    ], "kv_cache_layout only support NHD, SHUFFLE"
    if dequant:
        assert k_scales is not None and v_scales is not None
    head_dim = key.shape[2]
    x = 16 // key_cache.element_size()
    # For k cache layout: [num_blocks, num_heads, page_size, head_dim]
    assert head_dim == key_cache.shape[3], (
        "We assume your kv cache layout is [num_blocks, "
        "page_size, num_heads, head_dim], but got otherwise"
    )
    page_size = key_cache.shape[1]
    num_heads = key_cache.shape[2]

    grid = lambda meta: (total_tokens, num_heads)  # noqa: E731
    cp_mha_gather_cache_kernel[grid](
        key_cache,
        value_cache,
        key,
        value,
        block_tables,
        cu_seqlens_kv,
        token_to_batch,
        seq_starts,
        k_scales,
        v_scales,
        num_heads,
        head_dim,
        x,
        block_tables.size(1),
        DEQUANT=dequant,
        PAGE_SIZE=page_size,
        CACHE_FORMAT=kv_cache_layout,
        BLOCK_SIZE=head_dim,
    )


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
    if use_mla:
        return self.impl.forward(q, k, v, positions, q_scale, qkv)
    else:
        return self.impl.forward(
            layer=self,
            query=q,
            key=k,
            value=v,
            position=positions,
            q_scale=q_scale,
            qkv=qkv,
        )


def linear_attention_with_output_base_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return core_attn_out


@mark_spliting_op(
    is_custom=True, gen_fake=linear_attention_with_output_base_fake, mutates_args=[]
)
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


class BaseAttention(nn.Module, ABC):
    """
    Abstract base class for attention

    This class defines the interface that all attention implementations must follow
    """

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
        **kwargs,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the forward() method"
        )


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
        dt_bias=None,
        A_log=None,
        conv1d=None,
        activation=None,
        layer_num=0,
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
