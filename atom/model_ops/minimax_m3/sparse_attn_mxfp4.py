# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block-sparse attention over the MXFP4 KV cache (``ATOM_M3_SPARSE_KV_MXFP4=1``).

A row is a decode query token or a prefill query token: one sparse selection of
at most ``topk`` 128-token pages, the tail page last, with its exact context
length -- what index_topk emits at ``pages_per_block=1``. Each row is a length-1
"sequence" for ``pa_decode_mxfp4`` (query e4m3 as the fused writer emitted it,
K and V MXFP4 pages), the same fold ``sparse_pa_per_row`` makes for the fp8
caches, with the (token, kv head) pair collapsed into the row so the kernel runs
one kv head per call: the emitted page id is ``page * Hkv + head``, which is the
page index of the collapsed ``[pages * Hkv, 1, 128, .]`` views taken here.

Decode (a captured graph, a fixed set of row counts) runs planned: aiter's planner
takes at most ``_FLYDSL_PLAN_MAX_BATCH`` sequences, so more rows are run in
segments, each with a plan of its own, refreshed from the rows' context lengths
once per step (they do not depend on the layer: the tail block is always
selected) and recorded in the graph when one is being captured.

Prefill runs plan-free (``plan_owner=None``): a chunk has a new row count nearly
every forward, and ``plan_pa_decode`` JIT-compiles a kernel per batch size (~300 ms,
stalling every rank), so one task a row -- a row reads at most ``topk`` pages -- and
one specialization whatever the rows.
"""

import torch

from atom.model_ops.base_attention import _FLYDSL_PLAN_MAX_BATCH, _flydsl_plan_scratch
from atom.model_ops.minimax_m3.kv_mxfp4 import HEAD_DIM, PAGE_TOKENS, page_views

# Plans outlive a step -- a captured graph keeps reading its plan's buffers -- so
# they are keyed by what shapes them, never by a step's objects.
_PLANS: dict[tuple, object] = {}


def _refreshed(owner) -> set:
    marks = getattr(owner, "_mxfp4_pa_refreshed", None)
    if marks is None:
        marks = set()
        owner._mxfp4_pa_refreshed = marks
    return marks


def _plan(owner, context, segment, num_kv_heads):
    """The segment's plan, refreshed from its context lengths once per step."""
    from aiter.ops.flydsl.pa_decode import plan_pa_decode

    device = context.device
    key = (
        int(torch.cuda.current_stream(device).cuda_stream),
        segment,
        int(context.shape[0]),
        num_kv_heads,
        device.index,
    )
    mark = (key, torch.cuda.is_current_stream_capturing())
    plan = _PLANS.get(key)
    marks = _refreshed(owner)
    if plan is not None and mark in marks:
        return plan
    marks.add(mark)
    if plan is None:
        plan = _PLANS[key] = plan_pa_decode(context, num_kv_heads, query_length=1)
    else:
        plan_pa_decode(context, num_kv_heads, query_length=1, plan=plan)
    return plan


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
    plan_owner=None,
) -> torch.Tensor:
    """Attention of every row over its selected pages; ``[rows, num_heads, 128]``.

    ``plan_owner`` is an object that lives exactly one forward (the sparse
    metadata): the plans are refreshed once per owner and capture state. None runs
    plan-free, one task a row, written directly (prefill); every row then needs a
    context of at most ``sparse_bt.shape[1]`` pages, which index_topk's emit gives.
    """
    from aiter.ops.flydsl.pa_decode_mxfp4 import pa_decode_mxfp4

    rows, num_heads, head_dim = q.shape
    if head_dim != HEAD_DIM or not q.is_contiguous():
        raise ValueError(
            f"MXFP4 sparse attention wants contiguous q [rows, H, {HEAD_DIM}]"
        )
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError(f"the MXFP4 path takes an e4m3 query, got {q.dtype}")
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
    out = torch.empty((n, group, HEAD_DIM), dtype=out_dtype, device=q.device)
    bound = sparse_bt.shape[1] * PAGE_TOKENS
    if plan_owner is None:
        pa_decode_mxfp4(
            out,
            q_rows,
            k_data,
            k_scale,
            v_data,
            v_scale,
            sparse_ctx,
            sparse_bt,
            sm_scale,
            work_plan=None,
            max_partitions=1,
            max_context_length=bound,
        )
        return out.view(rows, num_heads, HEAD_DIM)
    for seg, start in enumerate(range(0, n, _FLYDSL_PLAN_MAX_BATCH)):
        end = min(start + _FLYDSL_PLAN_MAX_BATCH, n)
        ctx = sparse_ctx[start:end]
        plan = _plan(plan_owner, ctx, seg, 1)
        exp_sums, max_logits, tmp = _flydsl_plan_scratch(
            plan, 1, group, HEAD_DIM, out_dtype, q.device
        )
        pa_decode_mxfp4(
            out[start:end],
            q_rows[start:end],
            k_data,
            k_scale,
            v_data,
            v_scale,
            ctx,
            sparse_bt[start:end],
            sm_scale,
            work_plan=plan,
            max_context_length=bound,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=tmp,
        )
    return out.view(rows, num_heads, HEAD_DIM)
