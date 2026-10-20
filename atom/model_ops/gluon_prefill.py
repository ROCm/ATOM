# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon cached-prefill kernel for the isolated M3 C32 experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from atom.utils.forward_context import ForwardContext


def _scale_cache_view(scale, shape, device):
    """Keep writer scales in place, broadcasting scalar storage without a copy.

    The pinned Gluon BF16/FP8 loader accepts physical per-token scale views.
    Draft rope_cache instead returns kv_scale, leaving its pool scale planes
    unused. Zero strides let that same loader read the scalar for every token;
    this also preserves the gather path's BF16 dequantized operand precision.
    """
    if scale is None or scale.dtype != torch.float32 or scale.device != device:
        raise ValueError("Gluon C32 requires float32 KV scales on the query device")
    if scale.numel() == 1:
        return scale.view(()).expand(shape)
    if tuple(scale.shape) != shape:
        raise ValueError(f"Gluon C32 requires scalar or per-token KV scales {shape}")
    return scale


def run_gluon_prefill(
    attention: Callable,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    fwd_ctx: ForwardContext,
    *,
    softmax_scale: float,
    sliding_window: int = -1,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use real block tables and pool aliases for target and draft prefill.

    The native MHA builder starts requests at logical position zero, and its
    context_lens includes both cached and new tokens. rope_cache has already
    written new KV on this stream. No pool conversion, gather or CPU readback
    occurs here. Only output is allocated; unified attention may use split-K
    scratch proportional to queries rather than a full-context K/V expansion.
    """
    if (
        q.dtype != torch.bfloat16
        or q.shape[-1] != 128
        or k_cache.dtype != torch.float8_e4m3fn
        or k_cache.ndim != 5
        or v_cache.ndim != 5
        or k_cache.shape[-2] != 128
        or sliding_window != -1
    ):
        raise ValueError("Gluon C32 requires native M3 BF16/FP8 page-128 KV")
    meta = fwd_ctx.attn_metadata
    n_seqs = fwd_ctx.context.scheduled_bs
    if not (fwd_ctx.context.is_prefill and meta.has_cached):
        raise ValueError("Paged cached prefill requires a cached prefill batch")
    if k_cache.dtype != v_cache.dtype:
        raise ValueError("Paged prefill K/V dtypes must match")
    blocks, heads, d_chunks, page_size, pack = k_cache.shape
    head_dim = d_chunks * pack
    expected_v = (blocks, heads, page_size // pack, head_dim, pack)
    if (
        pack != 16
        or tuple(v_cache.shape) != expected_v
        or not k_cache.is_contiguous()
        or not v_cache.is_contiguous()
    ):
        raise ValueError("Gluon C32 requires native contiguous shuffled KV")
    if head_dim != q.shape[-1] or q.shape[1] % heads or q.stride(-1) != 1:
        raise ValueError("Paged prefill query/KV head geometry mismatch")
    if (
        n_seqs <= 0
        or meta.block_tables is None
        or meta.block_tables.shape[0] < n_seqs
        or meta.cu_seqlens_q is None
        or meta.cu_seqlens_q.numel() < n_seqs + 1
        or meta.context_lens is None
        or meta.context_lens.numel() < n_seqs
    ):
        raise ValueError("Paged prefill requires real per-request KV metadata")

    expected_scales = (blocks, heads, page_size)
    k_scale = _scale_cache_view(k_scale, expected_scales, q.device)
    v_scale = _scale_cache_view(v_scale, expected_scales, q.device)
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    attention(
        q,
        k_cache,
        v_cache,
        output,
        cu_seqlens_q=meta.cu_seqlens_q[: n_seqs + 1],
        seqused_k=meta.context_lens[:n_seqs],
        max_seqlen_q=meta.max_seqlen_q,
        max_seqlen_k=meta.max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        alibi_slopes=None,
        window_size=(sliding_window - 1, 0) if sliding_window > 0 else (-1, -1),
        block_table=meta.block_tables[:n_seqs],
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
        shuffled_kv_cache=True,
        backend="gluon",
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
    )
    return output
