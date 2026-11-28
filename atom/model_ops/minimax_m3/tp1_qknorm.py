# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MiniMax-M3.1 at TP1: aiter's qknorm_v3 writes what ``fused_qknorm_idxrqknorm``
writes, byte for byte, for one layout and from ``MIN_TOKENS`` rows (slower below)."""

import functools
import logging

import torch

logger = logging.getLogger("atom")

# q / kv / index heads, head dim, index head dim, rotary dim
LAYOUT = (64, 4, 4, 128, 128, 64)
# measured faster from 640 rows, prefill and verify steps alike (slower at 512)
MIN_TOKENS = 640


@functools.cache
def _qknorm_v3():
    """aiter's qknorm_v3, or None where aiter lacks it or the GPU is not gfx950."""
    try:
        from aiter.ops import m3_tp1_qknorm
    except ImportError:
        m3_tp1_qknorm = None
    if m3_tp1_qknorm is None or not m3_tp1_qknorm.is_supported():
        logger.info(
            "aiter qknorm_v3 is unavailable: the MiniMax-M3.1 TP1 q/k norm, RoPE "
            "and cache write stay on fused_qknorm_idxrqknorm"
        )
        return None
    return m3_tp1_qknorm.qknorm_v3


def serves(impl, qkv: torch.Tensor) -> bool:
    """Whether qknorm_v3 writes this step of ``impl``, a SparseMHAPagedAttentionImpl."""
    return (
        qkv.shape[0] >= MIN_TOKENS
        and qkv.dtype == torch.bfloat16
        and impl.kv_cache_dtype == "fp8"
        and impl.index_cache_dtype == "fp8"
        and not impl.skip_index_topk
        and (
            impl.num_heads,
            impl.num_kv_heads,
            impl.num_idx_heads,
            impl.head_dim,
            impl.index_head_dim,
            impl.rotary_emb.rotary_dim,
        )
        == LAYOUT
        and _qknorm_v3() is not None
    )


def run(
    impl,
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The writes of a step ``serves`` takes. ``kv``: the page-16 SHUFFLE K and V
    caches and their per-token scales."""
    k_cache, v_cache, k_scale, v_scale = kv
    _qknorm_v3()(
        qkv=qkv,
        q_out=q_out,
        index_q_out=index_q,
        q_norm_weight=impl.q_norm.weight,
        k_norm_weight=impl.k_norm.weight,
        index_q_norm_weight=impl.index_q_norm.weight,
        index_k_norm_weight=impl.index_k_norm.weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        index_slot_mapping=slot_mapping,
        k_cache=k_cache,
        v_cache=v_cache,
        index_cache=impl.index_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        eps=impl.q_norm.variance_epsilon,
        block_size=k_cache.shape[3],
    )
