# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cached MHA prefill reading the existing pool without dense KV materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from atom.utils.forward_context import ForwardContext


def run_paged_prefill(
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
    meta = fwd_ctx.attn_metadata
    n_seqs = fwd_ctx.context.scheduled_bs
    if not (fwd_ctx.context.is_prefill and meta.has_cached):
        raise ValueError("Paged cached prefill requires a cached prefill batch")
    if k_cache.dtype != v_cache.dtype:
        raise ValueError("Paged prefill K/V dtypes must match")
    shuffled = k_cache.ndim == 5
    if shuffled:
        blocks, heads, d_chunks, page_size, pack = k_cache.shape
        head_dim = d_chunks * pack
        shuffle_v = (blocks, heads, page_size // pack, head_dim, pack)
        nhd_v = (blocks, heads, head_dim, page_size)
        if (
            pack != 16 // k_cache.element_size()
            or page_size < pack
            or page_size & (page_size - 1)
            or head_dim & (head_dim - 1)
            or tuple(v_cache.shape) not in (shuffle_v, nhd_v)
            or not k_cache.is_contiguous()
            or not v_cache.is_contiguous()
        ):
            raise ValueError("Paged prefill requires native contiguous paged KV")
    elif k_cache.ndim == 4 and v_cache.shape == k_cache.shape:
        blocks, page_size, heads, head_dim = k_cache.shape
    else:
        raise ValueError("Paged prefill requires native paged or flash-layout KV")
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

    scale_args = {}
    # BF16 pools have no quantization scales. FP8 pools carry either one scale
    # for the cache or separate scales at each physical (page, head, token).
    if k_cache.element_size() == 1:
        if k_scale is None or v_scale is None:
            raise ValueError("FP8 paged prefill requires K and V scales")
        if k_scale.numel() > 1 or v_scale.numel() > 1:
            expected = (blocks, heads, page_size)
            if tuple(k_scale.shape) != expected or tuple(v_scale.shape) != expected:
                raise ValueError(f"Per-token FP8 scales must have shape {expected}")
            scale_args = {"k_scale_cache": k_scale, "v_scale_cache": v_scale}
            k_scale = v_scale = None
    else:
        k_scale = v_scale = None
    output_dtype = q.dtype if q.element_size() > 1 else torch.bfloat16
    output = torch.empty(q.shape, dtype=output_dtype, device=q.device)
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
        k_descale=k_scale,
        v_descale=v_scale,
        sinks=sinks,
        shuffled_kv_cache=shuffled,
        backend="triton",
        **scale_args,
    )
    return output
