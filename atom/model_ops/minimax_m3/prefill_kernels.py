# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel choice for the NVFP4 sparse prefill of MiniMax-M3 (``--kv-cache-dtype nvfp4``).

``ATOM_M3_PREFILL_ATTN``: ``aiter`` (default, aiter's ``sparse_pa_nvfp4_prefill``),
``ak16`` / ``ak16lazy`` (amd-kernels' fp16 kernels: bitwise aiter's / the lazy-rescale rule),
``v2`` / ``v2lazy`` (``kernels.sparse_pa_prefill``).
``ATOM_M3_PREFILL_INDEX``: ``aiter`` (default: aiter's FlyDSL scorer, then ATOM's selector) or
``amdk`` (amd-kernels' scorer and ``index_prefill_top16``, aiter's per-row selector where ATOM
runs it: the same ids, item by item).
The amd-kernels options need ``/shared_nfs/ygan/work/amd-kernels`` (or a checkout of it) on
PYTHONPATH. Every option reads ATOM's cache pages as they are (the page layouts are the same).
"""

import functools
import os

import numpy as np
import torch

ATTN = os.environ.get("ATOM_M3_PREFILL_ATTN", "aiter")
INDEX = os.environ.get("ATOM_M3_PREFILL_INDEX", "aiter")
INSERT = os.environ.get("ATOM_M3_PREFILL_INSERT", "aiter")
_ATTN_CHOICES = ("aiter", "ak16", "ak16lazy", "v2", "v2lazy")
if ATTN not in _ATTN_CHOICES:
    raise ValueError(f"ATOM_M3_PREFILL_ATTN must be one of {_ATTN_CHOICES}, got {ATTN!r}")
if INDEX not in ("aiter", "amdk"):
    raise ValueError(f"ATOM_M3_PREFILL_INDEX must be aiter or amdk, got {INDEX!r}")
if INSERT not in ("aiter", "amdk"):
    raise ValueError(f"ATOM_M3_PREFILL_INSERT must be aiter or amdk, got {INSERT!r}")

_HEAD_SIDE = 9216  # one KV head's K (or V) of one 128-token page


def _owned(md, name, build):
    """A per-step value cached on the step's prefill metadata (every layer reads it)."""
    v = getattr(md, name, None)
    if v is None:
        v = build()
        setattr(md, name, v)
    return v


def attention(prefill_md, q, kv_cache, topk_idx, block_table, row_req_id, row_kv_lens, sm_scale,
              out, k_scale, v_scale, aiter_fn):
    """The NVFP4 sparse prefill attention of one layer into ``out``."""
    if ATTN == "aiter":
        aiter_fn(q, kv_cache, topk_idx, block_table, row_req_id, row_kv_lens, sm_scale,
                 out=out, k_scale=k_scale, v_scale=v_scale)
    elif ATTN in ("v2", "v2lazy"):
        from atom.model_ops.minimax_m3.kernels.sparse_pa_prefill import sparse_pa_prefill

        sparse_pa_prefill(q, kv_cache, topk_idx, block_table, row_req_id, row_kv_lens, sm_scale,
                          out=out, k_scale=k_scale, v_scale=v_scale, lazy=ATTN == "v2lazy")
    else:
        from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.host import (
            nvfp4_sparse_prefill,
        )

        hkv = topk_idx.shape[0]
        pages = torch.as_strided(
            kv_cache, (kv_cache.shape[0], 2 * hkv, 128, 72), (kv_cache.stride(0), _HEAD_SIDE, 72, 1)
        )
        abs_pos = _owned(prefill_md, "_ak_abs_pos", lambda: (row_kv_lens - 1).contiguous())
        nvfp4_sparse_prefill(q, pages, topk_idx[:, : q.shape[0]], block_table, row_req_id,
                             abs_pos, hkv, sm_scale, out, k_scale, v_scale,
                             compute_dtype="fp16", lazy_rescale=ATTN == "ak16lazy")


@functools.cache
def _ak_insert():
    from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128.host import (
        norm_rope_nvfp4_insert,
    )

    return norm_rope_nvfp4_insert


def insert_amdk(qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions,
                slot_mapping, kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out,
                num_heads, num_kv_heads, num_index_heads, eps):
    """``aiter.fused_qknorm_idxrqknorm`` (NVFP4 K / V / index-K insert mode) through amd-kernels'
    ``norm_rope_nvfp4_insert``: the same page bytes and q / index-q outputs (checked by
    ``scripts/bench_insert.py``)."""
    pages = torch.as_strided(
        kv_cache,
        (kv_cache.shape[0], 2 * num_kv_heads, 128, 72),
        (kv_cache.stride(0), _HEAD_SIDE, 72, 1),
    )
    _ak_insert()(qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions,
                 slot_mapping, slot_mapping, pages, index_cache.view(-1, 128, 72), k_scale,
                 v_scale, q_out, index_q_out.view(index_q_out.shape[0], -1), num_heads,
                 num_kv_heads, num_index_heads, eps)


@functools.cache
def _ak_index():
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.prefill_top16 import (
        index_prefill_top16,
    )
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width

    return index_prefill_top16, score_width


def index_topk_amdk(index_q, index_cache, block_table, cu_seqlens_q, seq_lens, max_query_len,
                    max_seq_len, topk, init_blocks, local_blocks, n_valid_column_per_row,
                    index_scorer, sm_scale):
    """``minimax_m3_index_topk`` (NVFP4 index, no emitted table) through amd-kernels: the top-16 ids
    ``[H, rows, 16]``. The fp32 scores go to the step's score workspace."""
    index_prefill_top16, score_width = _ak_index()
    rows, heads, _ = index_q.shape
    assert topk == 16, topk
    width = score_width(max_seq_len)
    score = index_scorer.scores.logits(heads * rows, width).view(heads, rows, width)
    out = torch.empty((heads, rows, topk), dtype=torch.int32, device=index_q.device)
    pages = index_cache.view(-1, 128, 72)
    from aiter.ops.flydsl.kernels.minimax_m3_index_score import LOG2E

    kw = dict(max_query_len=max_query_len, max_seq_len=max_seq_len,
              num_valid_pages=n_valid_column_per_row, score=score)
    try:  # aiter's scores (see prefill_fast._score_plan)
        index_prefill_top16(index_q, pages, block_table, cu_seqlens_q, seq_lens, init_blocks,
                            local_blocks, out, **kw,
                            out_scale=float(np.float32(sm_scale * LOG2E)))
    except TypeError:
        index_prefill_top16(index_q, pages, block_table, cu_seqlens_q, seq_lens, init_blocks,
                            local_blocks, out, **kw)
    # where ATOM's selector hands the rows to aiter's (other ties at the cut), so does this
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.model_ops.minimax_m3 import index_topk as it

    max_block = -(-max_seq_len // 128)
    dev = index_q.device.index if index_q.device.index is not None else torch.cuda.current_device()
    if it._aiter_selector_wins(max_block, False, arch=get_gfx_runtime(),
                               selector_rows=heads * rows, compute_units=it._compute_units(dev)):
        select, supported = it._small_k_selector()
        flat = score.view(heads * rows, width)[:, :max_block]
        flat_idx = out.view(heads * rows, topk)
        if supported(flat, n_valid_column_per_row, flat_idx, topk):
            select(flat, n_valid_column_per_row, flat_idx, topk, init_blocks, local_blocks)
    return out
