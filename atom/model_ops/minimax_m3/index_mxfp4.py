# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block scores over MiniMax-M3's MXFP4 index cache (``--index-cache-dtype mxfp4``).

aiter's FlyDSL scorers (``score_flydsl_fp4`` for decode, ``score_prefill_flydsl_fp4``
for prefill) read the cache as the two views of ``kv_mxfp4.page_views`` and write
the same ``[heads, rows, max_block]`` fp32 scores the Triton scorers do, so the
selection and the block-table emit after them are index_topk's own, unchanged.

The query is bf16, or e4m3 as the fused writer emits it with
``ATOM_M3_INDEX_Q_FP8=1`` (cast as stored, no per-tensor scale).

The decode scorer wants a per-step work map. It is built once per step in the
metadata (outside any graph) into a persistent buffer, ``index_score_work_map``;
its row count is the grid, so every call of a step -- and every graph replay --
must pass the same ``max_block`` the map was built against. The scores' -inf
sentinel is refilled the same way, once a step (``fresh_index_score``), not once a layer.
"""

import torch

from atom.model_ops.minimax_m3.kv_mxfp4 import page_views


def _flydsl(name):
    from aiter.ops import flydsl

    try:
        return getattr(flydsl, name)
    except AttributeError as e:
        raise RuntimeError(
            f"MiniMax-M3 MXFP4 index scoring needs aiter.ops.flydsl.{name}, which "
            "this aiter does not have (aiter ganyi/m3-mxfp4-all or later)"
        ) from e


def index_work_map_capacity(max_batch, max_block, max_query_len, num_idx_heads):
    """Rows of a persistent work-map buffer serving every batch up to
    ``max_batch`` (the exact grid is not monotonic in the batch)."""
    if max_batch <= 0:
        return 0
    return _flydsl("minimax_m3_index_score_work_map_capacity")(
        max_batch, max_block, max_query_len, num_idx_heads
    )


def build_index_work_map(
    seq_lens, *, max_block, max_query_len, num_idx_heads, out=None
):
    """``[grid, 2]`` int32 dispatch order of the decode scorer; None on an empty
    batch. ``out`` is the persistent buffer to fill (sliced to the grid)."""
    if seq_lens.shape[0] <= 0:
        return None
    return _flydsl("minimax_m3_index_score_work_map")(
        seq_lens, max_block, max_query_len, num_idx_heads, out=out
    )


def index_score_buffer(max_rows, num_idx_heads, max_block, device):
    """Flat fp32 storage for the decode scorer's scores of up to ``max_rows`` rows,
    ``max_block`` wide: the persistent ``out`` of every layer of a captured step."""
    return torch.empty(
        num_idx_heads * max_rows * max_block, dtype=torch.float32, device=device
    )


def fresh_index_score(buf, rows, num_idx_heads, max_block, max_query_len):
    """``[H, rows, max_block]`` view of ``buf``, refilled with the scorer's -inf
    sentinel, for ``decode_score_mxfp4(out=...)``; call it once a step, outside the
    graph. A page a row never reaches is never stored, so without the refill it
    would keep an earlier step's score. None when the scorer lays its own scores out
    transposed (S * H >= its TRANSPOSE_MIN_F, not TP4's S=8, H=1): the caller then
    lets the scorer allocate."""
    from aiter.ops.flydsl.kernels.minimax_m3_index_score import (
        NEG_INF,
        TRANSPOSE_MIN_F,
    )

    if max_query_len * num_idx_heads >= TRANSPOSE_MIN_F:
        return None
    n = num_idx_heads * rows * max_block
    if n > buf.numel():
        raise ValueError(f"{rows} rows x {max_block} blocks outgrow the score buffer")
    return buf[:n].view(num_idx_heads, rows, max_block).fill_(NEG_INF)


def decode_score_mxfp4(
    idx_q,  # [batch * S, H, 128] bf16 or e4m3
    index_cache,  # uint8 [pages, 8704]
    block_table,  # [batch, >= max_block] int32
    seq_lens,  # [batch] int32
    max_block,
    max_query_len,
    num_idx_heads,
    sm_scale,
    work_map=None,
    out=None,
):
    """``[H, batch * S, max_block]`` fp32 block scores of a uniform decode (S query
    rows a request, causal per row). ``work_map`` is the step's; omitted, the
    scorer builds its own (~120 us -- fine in a test, ruinous per layer). ``out`` is
    the step's ``fresh_index_score``; omitted, the scorer allocates and fills its
    own with -inf (a launch a layer)."""
    k_data, k_scale = page_views(index_cache)
    return _flydsl("minimax_m3_index_score_fp4_flydsl")(
        idx_q,
        k_data,
        k_scale,
        block_table,
        seq_lens,
        max_query_len,
        num_idx_heads,
        sm_scale,
        max_block,
        out=out,
        work_map=work_map,
    )


def prefill_score_mxfp4(
    idx_q,  # [total_q, H, 128] bf16 or e4m3
    index_cache,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    max_seq_len,
    sm_scale,
):
    """``[H, total_q, ceil(max_seq_len / 128)]`` fp32 block scores of a ragged
    prefill batch. The scorer stages the referenced pages as fp8 for the call."""
    k_data, k_scale = page_views(index_cache)
    return _flydsl("minimax_m3_index_score_prefill_fp4_flydsl")(
        idx_q,
        k_data,
        k_scale,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        max_seq_len,
        sm_scale,
    )


def require_supported(idx_q, index_cache, S, H, max_block, block_table, seq_lens):
    """Refuse a decode shape the scorer cannot serve, by name, instead of letting
    the call fail somewhere in its validation."""
    k_data, k_scale = page_views(index_cache)
    ok = _flydsl("minimax_m3_index_score_fp4_flydsl_supported")(
        idx_q,
        k_data,
        k_scale,
        S,
        H,
        max_block,
        block_table,
        seq_lens=seq_lens,
    )
    if not ok:
        raise RuntimeError(
            f"the MXFP4 index scorer does not serve S={S} H={H} max_block={max_block} "
            f"batch={seq_lens.shape[0]} (idx_q {tuple(idx_q.shape)} {idx_q.dtype})"
        )
