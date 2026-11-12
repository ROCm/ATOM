# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block-sparse attention over the MXFP4 KV cache (``--kv-cache-dtype fp4``).

A row is a decode or prefill query token: one sparse selection of at most ``topk``
128-token pages, the tail page last, with its exact context length (what index_topk
emits at ``pages_per_block=1``). Each row is a length-1 "sequence" for ``pa_decode``
(MXFP4 K/V, e4m3 query), the fold ``sparse_pa_per_row`` makes for the fp8 caches, with
the (token, kv head) pair collapsed into the row: the emitted page id is
``page * Hkv + head``, the page index of the collapsed ``[pages * Hkv, 1, 128, .]``
views taken here.

Decode (a captured graph, fixed row counts) runs planned, in segments of at most
``_FLYDSL_PLAN_MAX_BATCH`` rows (aiter's planner limit; no gluon fallback exists for
MXFP4 pages), each with a plan from the shared cache. Prefill runs plan-free
(``planned=False``): a chunk's row count is new almost every forward and
``plan_pa_decode`` JIT-compiles per batch size, so it takes one task a row.
"""

import torch
from aiter.ops.flydsl.pa_decode import pa_decode
from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits

from atom.model_ops.base_attention import (
    _FLYDSL_PLAN_MAX_BATCH,
    _flydsl_plan_scratch,
    _flydsl_prepare_explicit_plan,
)
from atom.model_ops.minimax_m3.geometry import SPARSE_BLOCK_SIZE
from atom.model_ops.minimax_m3.mxfp4_pages import HEAD_DIM, page_views


@torch.no_grad()
def sparse_pa_mxfp4(
    q: torch.Tensor,  # [rows, num_heads, 128] float8_e4m3fn, contiguous
    k_pages: torch.Tensor,  # uint8 [pages, Hkv, 8704]
    v_pages: torch.Tensor,  # uint8 [pages, Hkv, 8704]
    sparse_bt: torch.Tensor,  # [rows * Hkv, topk] int32, page * Hkv + head
    sparse_ctx: torch.Tensor,  # [rows * Hkv] int32
    num_kv_heads: int,
    sm_scale: float,
    out_dtype: torch.dtype,
    *,
    planned: bool = False,
    plan_step_owner=None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention of every row over its selected pages; ``[rows, num_heads, 128]``.

    ``out`` is that tensor, of ``out_dtype`` and contiguous, when the caller owns the
    output (a DP step's pad rows past ``rows`` live in a larger buffer it slices);
    else it is allocated here. Returned either way.

    ``planned`` runs the decode path, with plans from the shared cache. Not planned
    runs plan-free, one task a row, written directly (prefill); every row then needs
    a context of at most ``sparse_bt.shape[1]`` pages, which index_topk's emit gives.
    ``plan_step_owner`` is the sparse metadata, which lives one forward, when every
    layer's rows have the same lengths (the tail block is always selected), so a
    plan is refreshed once a step; None refreshes it on every call.
    """
    rows, num_heads, head_dim = q.shape
    if head_dim != HEAD_DIM or not q.is_contiguous():
        raise ValueError(
            f"MXFP4 sparse attention wants contiguous q [rows, H, {HEAD_DIM}]"
        )
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError(f"the MXFP4 path takes an e4m3 query, got {q.dtype}")
    if k_pages.shape[1] != num_kv_heads or v_pages.shape[1] != num_kv_heads:
        raise ValueError(
            f"num_kv_heads is {num_kv_heads}, but the K / V pages hold "
            f"{k_pages.shape[1]} / {v_pages.shape[1]} heads"
        )
    group = num_heads // num_kv_heads
    pages = k_pages.shape[0] * num_kv_heads
    # [pages, Hkv, 8704] -> [pages * Hkv, 1, 8704]: the pool field is contiguous in
    # (page, head), which is what the emitted page ids index.
    k_flat = k_pages.view(pages, 1, k_pages.shape[-1])
    v_flat = v_pages.view(pages, 1, v_pages.shape[-1])
    k_data, k_scale = page_views(k_flat)
    v_data, v_scale = page_views(v_flat)
    n = rows * num_kv_heads
    q_rows = q.view(n, group, HEAD_DIM)
    if out is None:
        out = torch.empty((rows, num_heads, HEAD_DIM), dtype=out_dtype, device=q.device)
    elif out.shape != q.shape or out.dtype != out_dtype or not out.is_contiguous():
        raise ValueError(
            f"out must be a contiguous {tuple(q.shape)} {out_dtype} tensor, got "
            f"{tuple(out.shape)} {out.dtype}"
        )
    out_rows = out.view(n, group, HEAD_DIM)
    bound = sparse_bt.shape[1] * SPARSE_BLOCK_SIZE
    if not planned:
        pa_decode(
            out_rows,
            q_rows,
            k_data,
            v_data,
            sparse_ctx,
            sparse_bt,
            sm_scale,
            1,
            key_scale=k_scale,
            value_scale=v_scale,
            work_plan=None,
            max_context_length=bound,
        )
        return out
    for seg, start in enumerate(range(0, n, _FLYDSL_PLAN_MAX_BATCH)):
        end = min(start + _FLYDSL_PLAN_MAX_BATCH, n)
        ctx = sparse_ctx[start:end]
        plan = _flydsl_prepare_explicit_plan(
            q=q_rows[start:end],
            k_cache=k_data,
            context_lens=ctx,
            num_seqs=end - start,
            query_length=1,
            # the static split count, as the fp8 sparse path caps its plans
            plan_partition_cap=get_recommended_splits(end - start, 1),
            context_bound=bound,
            existing_plan=None,
            plan_step_owner=plan_step_owner,
            plan_segment=seg,
        )
        exp_sums, max_logits, tmp = _flydsl_plan_scratch(
            plan, 1, group, HEAD_DIM, out_dtype, q.device
        )
        pa_decode(
            out_rows[start:end],
            q_rows[start:end],
            k_data,
            v_data,
            ctx,
            sparse_bt[start:end],
            sm_scale,
            1,
            key_scale=k_scale,
            value_scale=v_scale,
            work_plan=plan,
            max_context_length=bound,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=tmp,
        )
    return out
