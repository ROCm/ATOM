# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AQ1: the MiniMax-M3.1 TP4 decode QKV chain + CP index score in one gfx1250 launch
(kernels/aq_gfx1250.py). Writes exactly what ``mxfp8_qkv_norm_rope_nvfp4_insert``
followed by ATOM's ``cp_scores_fp4`` write (q, index q, the NVFP4 K / V / index-K
rows, the fp32 scores [heads, tokens, local]), for one request of ``ql`` rows.
With ``exchange=`` (full AQ) the launch also runs ATOM's ``select_exchanged_candidates``
(the CP top-16 exchange through the candidate mailbox) and writes its indices, sparse
block table and ctx instead of the scores."""

from __future__ import annotations

import struct

import numpy as np
import torch

D = 128
NQ, NKV, ROT = 16, 1, 64
K_DIM = 6144
PAGE = 128
LOG2E = 1.4426950409
# the 8-row decode chain config (rF6 / PFX): tile_m, ns, ks, kw, wm, wn, depth, astage, wcpol
AQ1_CONFIG = (16, 4, 4, 6, 1, 2, 2, 1, 1)
N_SCORE = 64  # score workgroups (a multiple of 16)
# KiB of the exchange role's code prefetched into the instruction cache at the role's entry (0: none). A module
# attribute read at each (eager / capture) call, for the one-server A/B; the released value is the default.
IPF_KIB = 0
# 1: the owner merges the keys its poll read and carries each candidate's page through the merge (no second
# mailbox read, no table read after the merge). 0: the previous order, kept for the one-server A/B.
OWNER_FAST = 1
# the exchange's first list-merge level on several waves (1) or on wave 0 alone (0, the previous order): a module
# attribute read at each eager / capture call, for the one-server A/B
MERGE_PAR = 1
N_SCORE_X = 112  # with the exchange: 368 QKV + 112 score + 32 exchange = 512 workgroups, all co-resident
# key-ring slots a score wave: 6 holds a 500K workgroup's 5 blocks a compute group, all issued at entry (4: the
# previous depth, for the one-server A/B). LDS 124 KB a workgroup at 6, 2 a CU still fit
RING = 6
RING_AQ1 = 4  # AQ1 (no exchange) keeps its depth
# the score role's compute groups: 3 = waves 4-7 (the idx-q converters) join waves 0-3 and 8-11 once the q is in LDS,
# with RING_S3 key-ring slots a wave (3 groups x 4 waves x RING_S3 slots must stay <= 160 KB of LDS a workgroup);
# 2 = the previous split (RING), for the one-server A/B
SC_GROUPS = 3
RING_S3 = 4
# the owner's merge (11-27, MONO): 1 = each shard's 16 keys merged into the running top-16 as soon as they arrive (the
# last shard's merge is the only one after the last arrival) and the winners' pages read from the request's block
# table, staged in LDS while the scores are computed; 0 = all four shards polled, then decoded and merged, pages read
# from global memory (the previous order, for the one-server A/B). Same top-16 (a total order), same bytes.
OWNER_V2 = 1
MAXIT = 20_000_000
_SCRATCH: dict = {}


# local blocks a CP rank the exchange's score hand-off holds: 1M tokens at 4 ranks (decode graphs are captured at
# max_model_len, so the launch must take the model's whole context)
LOCAL_MAX = 2048


def aq1_scratch(device, niq: int, ql: int, n_score: int = N_SCORE, xchg: bool = False):
    """Per (device, shape): launch counts [GRID], the index-q hand-off (3 copies, armed 0xff), the index-K flags,
    the timeout counter. Made outside any capture (the eager warm-up call)."""
    from .kernels.aq import aq1_grid, aq_xchg_wgs

    key = (device.index if device.index is not None else torch.cuda.current_device(), niq, ql, n_score, bool(xchg))
    s = _SCRATCH.get(key)
    if s is None:
        grid = aq1_grid(niq, n_score) + (aq_xchg_wgs(niq, ql) if xchg else 0)
        s = dict(
            cnt=torch.zeros(grid, dtype=torch.int32, device=device),
            q=torch.full((3 * ql * niq * D * 2,), 0xFF, dtype=torch.uint8, device=device),
            flag=torch.zeros(max(ql, 16), dtype=torch.int32, device=device),
            err=torch.zeros(4, dtype=torch.int32, device=device),
            # exchange mode: the scores [niq, ql, LOCAL_MAX] as data-as-flag words (armed 0xffffffff, re-armed by the
            # exchange workgroup that reads them)
            scb=torch.full((niq * ql * LOCAL_MAX if xchg else 1,), -1, dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = s
    return s


def _pc_slot(device, spec):
    """One int64 a compiled kernel (device, spec): the end PC its code prefetch is bounded by (0 until recorded)."""
    key = ("pc", device.index if device.index is not None else torch.cuda.current_device(), spec)
    t = _SCRATCH.get(key)
    if t is None:
        t = _SCRATCH[key] = torch.zeros(2, dtype=torch.int32, device=device)
    return t


def ql_ok(ql, ex) -> bool:
    return ql <= ex["max_rows"] and ex["indices"].shape[-1] == ex["topk"] and ex["sparse_ctx"].numel() == ql


def aq1_supported(m, niq, batch, max_query_len, heads_score) -> str | None:
    if batch != 1 or m != max_query_len or heads_score != niq:
        return f"one request of max_query_len rows (m {m}, batch {batch}, ql {max_query_len})"
    if niq * max_query_len > 32 or m > 16:
        return f"{niq} index heads x {max_query_len} rows"
    return None


def aq_exchange_supported(niq, world, topk, max_seq_len, max_query_len, max_rows, slot_keys, pages_per_block):
    """None when the full AQ (``exchange=``: the CP candidate exchange in the launch) takes the call: one index head a
    rank, top-16 of 4 shards, the NVFP4 page-128 table (one page a block), at most LOCAL_MAX local blocks."""
    if niq != world or topk != 16 or world * topk != 64 or slot_keys < topk:
        return f"{niq} index heads, {world} ranks, top-{topk}, {slot_keys}-key slots"
    if pages_per_block != 1:
        return f"{pages_per_block} pages a block"
    if max_query_len > max_rows:
        return f"{max_query_len} rows, mailbox {max_rows}"
    local = -(-(-(-max_seq_len // PAGE)) // world)
    if local > LOCAL_MAX:
        return f"{local} local blocks"
    return None


def flydsl_aq1(
    x, x_scale, w, w_scale, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
    cos_sin_cache, positions, slot_mapping, index_slot_mapping, k_cache, v_cache, k_scale, v_scale,
    index_cache, index_scale, q_out, index_q_out, eps, *,
    num_index_heads, block_table, seq_lens, max_seq_len, rank, world, max_query_len, sm_scale, scores,
    n_score: int | None = None, maxit: int = MAXIT, scratch=None, timeline=None, exchange=None,
    k_chain: bool = False,
):  # fmt: skip
    """One AQ1 launch. Arguments as ``mxfp8_qkv_norm_rope_nvfp4_insert`` + ``cp_scores_fp4`` (``scores``: fp32
    [heads, tokens, ceil(blocks / world)], slots past the live blocks left unwritten).

    ``k_chain``: the QKV GEMM in the tuned ``gemm_a8w8 ..._sk4_fsk``'s K order (each K split one WMMA chain from
    zero, splits summed in split order) instead of the fused chain's (its K-waves summed in wave order): the bytes of
    a layer whose QKV production runs that GEMM."""
    import flydsl.expr as fx

    from .kernels.aq import launch_aq1
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
    from aiter.ops.flydsl.mxfp8_qkv_rope_nvfp4_gfx1250 import _slab

    m = x.shape[0]
    niq = num_index_heads
    n = (NQ + 2 * NKV + niq + 1) * D
    why = aq1_supported(m, niq, seq_lens.numel(), max_query_len, niq if scores is None else scores.shape[0])
    assert why is None, f"aq1: {why}"
    assert tuple(w.shape) == (n, K_DIM) and x.shape[1] == K_DIM and x.stride(1) == 1 and x.stride(0) % 16 == 0
    assert index_q_out.is_contiguous() and tuple(index_q_out.shape) == (m, niq * D)
    assert scores is None or (scores.dtype == torch.float32 and scores.is_contiguous() and scores.shape[1] == m)
    assert block_table.dtype == torch.int32 and seq_lens.dtype == torch.int32 and block_table.stride(1) == 1
    assert index_cache.stride()[1:] == (D // 2, 1) and index_scale.stride()[1:] == (D // 16, 1)
    assert k_cache.stride() == v_cache.stride() and k_scale.stride() == v_scale.stride()
    tile_m, ns, ks, kw, wm, wn, depth, astage, wcpol = AQ1_CONFIG
    mpad = (m + tile_m - 1) // tile_m * tile_m
    slab = _slab(x.device, ks * mpad * n)
    xchg = exchange is not None
    n_score = n_score or (N_SCORE_X if xchg else N_SCORE)
    sc = scratch or aq1_scratch(x.device, niq, max_query_len, n_score, xchg)
    if xchg:
        # exchange = dict(peers, send_calls, recv_calls, timeouts, max_rows, slot_keys, topk, init_blocks, local_keep,
        #                 indices [1, ql, topk] int32, sparse_bt [ql, topk] int32, sparse_ctx [ql] int32)
        ex = exchange
        why = aq_exchange_supported(niq, world, ex["topk"], max_seq_len, max_query_len, ex["max_rows"],
                                    ex["slot_keys"], 1)
        assert why is None, f"aq: {why}"
        assert ex["sparse_bt"].stride(0) == ex["topk"] and ex["indices"].is_contiguous()
        assert ex["sparse_ctx"].is_contiguous() and ql_ok(max_query_len, ex)
    blocks = -(-max_seq_len // PAGE)
    local = -(-blocks // world)
    assert scores is None or scores.shape[2] == local
    assert not xchg or local <= LOCAL_MAX, f"aq: {local} local blocks"
    dummy = sc["err"]
    eps_bits = struct.unpack("<i", struct.pack("<f", float(eps)))[0]
    stream = torch.cuda.current_stream(x.device)
    spec = (tile_m, ns, ks, kw, wm, wn, depth, niq, astage, wcpol, max_query_len, int(rank), int(world),
            (RING_S3 if SC_GROUPS == 3 else RING) if xchg else RING_AQ1, n_score,
            int(timeline is not None), int(xchg),
            *((exchange["topk"], exchange["init_blocks"], exchange["local_keep"], exchange["max_rows"], LOCAL_MAX,
               exchange["slot_keys"]) if xchg else (16, 0, 0, 64, LOCAL_MAX, 64)), IPF_KIB if xchg else 0,
            OWNER_FAST, int(MERGE_PAR) if xchg else 0, int(SC_GROUPS) if xchg else 2,
            int(OWNER_V2) if xchg and OWNER_FAST else 0, int(bool(k_chain)))
    _run_compiled(
        launch_aq1,
        ptr_arg(x), ptr_arg(x_scale), ptr_arg(w), ptr_arg(w_scale), ptr_arg(slab),
        ptr_arg(q_out), ptr_arg(index_q_out), ptr_arg(q_norm_weight), ptr_arg(k_norm_weight),
        ptr_arg(index_q_norm_weight), ptr_arg(index_k_norm_weight), ptr_arg(cos_sin_cache), ptr_arg(positions),
        ptr_arg(slot_mapping), ptr_arg(index_slot_mapping), ptr_arg(k_cache), ptr_arg(v_cache), ptr_arg(k_scale),
        ptr_arg(v_scale), ptr_arg(index_cache), ptr_arg(index_scale),
        ptr_arg(block_table, fx.Int32), ptr_arg(seq_lens, fx.Int32),
        ptr_arg(scores if scores is not None else dummy, fx.Float32 if scores is not None else fx.Int32),
        ptr_arg(sc["q"]), ptr_arg(sc["flag"], fx.Int32), ptr_arg(sc["cnt"], fx.Int32), ptr_arg(sc["err"], fx.Int32),
        ptr_arg(timeline if timeline is not None else sc["err"], fx.Int32),
        ptr_arg(sc["scb"], fx.Int32),
        ptr_arg(exchange["peers"] if xchg else dummy, fx.Int32),
        ptr_arg(exchange["send_calls"] if xchg else dummy, fx.Int32),
        ptr_arg(exchange["recv_calls"] if xchg else dummy, fx.Int32),
        ptr_arg(exchange["timeouts"] if xchg else dummy, fx.Int32),
        ptr_arg(exchange["indices"] if xchg else dummy, fx.Int32),
        ptr_arg(exchange["sparse_bt"] if xchg else dummy, fx.Int32),
        ptr_arg(exchange["sparse_ctx"] if xchg else dummy, fx.Int32),
        ptr_arg(_pc_slot(x.device, spec), fx.Int32),
        m, x.stride(0), mpad, cos_sin_cache.shape[0],
        k_cache.stride(0), k_scale.stride(0), index_cache.stride(0), index_scale.stride(0), eps_bits,
        block_table.stride(0), local, blocks, float(np.float32(sm_scale * LOG2E)), int(maxit),
        fx.Stream(stream),
        *spec,
        specialization_key=("aq1", x.device.index) + spec,
    )  # fmt: skip
    return scores
