# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3 FP8 sparse prefill on aiter's token-tiled kernel (gfx950, TP1): a tile of
12 consecutive tokens of one request and one KV head reads the blocks they share once.
"""

from __future__ import annotations

import functools
import importlib
import logging

import torch

from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import prefill_row_positions
from atom.utils import envs

logger = logging.getLogger("atom")

# the TP1 layer it was measured on: 4 KV heads x 16 query heads of 128, top-16
_KV_HEADS, _HEADS_PER_KV, _HEAD_DIM, _TOPK = 4, 16, 128, 16
# fewer rows leave most CUs idle (0.5-0.97x of ASM in situ)
_MIN_ROWS = 1536
# past it, deep layers of code-like prompts share fewer blocks a tile and lose to ASM
_MAX_SEQ_LEN = 65_536


@functools.cache
def _ops():
    """aiter's op on a gfx950, else None; an aiter without it is logged once."""
    try:
        ops = importlib.import_module("aiter.ops.m3_tt_sparse_prefill")
    except ImportError as exc:
        logger.warning("MiniMax-M3 token-tiled prefill unavailable, ASM kept: %s", exc)
        return None
    if not ops.is_supported():
        return None
    logger.info(
        "MiniMax-M3 sparse prefill: token-tiled kernel on steps of >= %d tokens and "
        "context <= %d, ASM otherwise",
        _MIN_ROWS,
        _MAX_SEQ_LEN,
    )
    return ops


def layer_eligible(
    *, num_heads: int, num_kv_heads: int, head_dim: int, topk: int, kv_cache_dtype: str
) -> bool:
    """The flag, the TP1 FP8 layer shape and aiter's op on a gfx950."""
    return (
        envs.ATOM_M3_TT_PREFILL
        and kv_cache_dtype == "fp8"
        and (num_kv_heads, num_heads, head_dim, topk)
        == (_KV_HEADS, _KV_HEADS * _HEADS_PER_KV, _HEAD_DIM, _TOPK)
        and _ops() is not None
    )


def step_takes(prefill_md, rows: int) -> bool:
    """Whether the step's `rows` prefill rows run the token-tiled kernel: no pad rows
    past the scheduled tokens (its tiles cover only those), enough rows, short context.
    """
    return (
        rows == prefill_md.qo_indptr.shape[0]
        and rows >= _MIN_ROWS
        and prefill_md.max_seq_len <= _MAX_SEQ_LEN
    )


def _tile_map(prefill_md, total_q: int):
    """Each row's request and causal key count, and tiles of at most TT_TQ rows of one
    request padded to a host-known count: built once a step, without a host sync."""
    cached = getattr(prefill_md, "_tt_tile_map", None)
    if cached is not None:
        return cached
    cu, prefix = prefill_md.cu_seqlens_q, prefill_md.context_lens
    nreq = prefix.shape[0]
    req, kv_lens = prefill_row_positions(cu, prefix, total_q)
    tq = _ops().TT_TQ
    lens = (cu[1 : nreq + 1] - cu[:nreq]).to(torch.int64)
    tiles = (lens + tq - 1) // tq
    ends = torch.cumsum(tiles, 0)
    ntiles = (total_q + tq - 1) // tq + nreq  # at most one partial tile a request
    k = torch.arange(ntiles, device=cu.device, dtype=torch.int64)
    b = torch.searchsorted(ends, k, right=True)
    live = b < nreq
    b = b.clamp(max=nreq - 1)
    local = k - (ends[b] - tiles[b])
    t0 = torch.where(live, cu[b].to(torch.int64) + local * tq, 0)
    n = torch.where(live, torch.clamp(lens[b] - local * tq, 0, tq), 0)
    tile_map = (req, kv_lens, t0.to(torch.int32), n.to(torch.int32), ntiles)
    prefill_md._tt_tile_map = tile_map
    return tile_map


def run(
    q,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    topk_idx,
    block_table,
    prefill_md,
    sm_scale,
    out,
) -> None:
    """The step's prefill rows `q` [T, H, 128] into bf16 `out` [T, H, 128]."""
    req, kv_lens, t0, n, ntiles = _tile_map(prefill_md, q.shape[0])
    _ops().tt_sparse_prefill(
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        topk_idx,
        block_table,
        req,
        kv_lens,
        t0,
        n,
        ntiles,
        sm_scale,
        prefill_md.max_seq_len,
        out=out,
    )
