# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AQ1: the MiniMax-M3.1 TP4 decode QKV chain + CP index score in one gfx1250 launch
(kernels/aq_gfx1250.py). Writes exactly what ``mxfp8_qkv_norm_rope_nvfp4_insert``
followed by ATOM's ``cp_scores_fp4`` write (q, index q, the NVFP4 K / V / index-K
rows, the fp32 scores [heads, tokens, local]), for one request of ``ql`` rows."""

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
RING = 4
MAXIT = 20_000_000
_SCRATCH: dict = {}


def aq1_scratch(device, niq: int, ql: int, n_score: int = N_SCORE):
    """Per (device, shape): launch counts [GRID], the index-q hand-off (3 copies, armed 0xff), the index-K flags,
    the timeout counter. Made outside any capture (the eager warm-up call)."""
    from .kernels.aq import aq1_grid

    key = (device.index if device.index is not None else torch.cuda.current_device(), niq, ql, n_score)
    s = _SCRATCH.get(key)
    if s is None:
        grid = aq1_grid(niq, n_score)
        s = dict(
            cnt=torch.zeros(grid, dtype=torch.int32, device=device),
            q=torch.full((3 * ql * niq * D * 2,), 0xFF, dtype=torch.uint8, device=device),
            flag=torch.zeros(max(ql, 16), dtype=torch.int32, device=device),
            err=torch.zeros(4, dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = s
    return s


def aq1_supported(m, niq, batch, max_query_len, heads_score) -> str | None:
    if batch != 1 or m != max_query_len or heads_score != niq:
        return f"one request of max_query_len rows (m {m}, batch {batch}, ql {max_query_len})"
    if niq * max_query_len > 32 or m > 16:
        return f"{niq} index heads x {max_query_len} rows"
    return None


def flydsl_aq1(
    x, x_scale, w, w_scale, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
    cos_sin_cache, positions, slot_mapping, index_slot_mapping, k_cache, v_cache, k_scale, v_scale,
    index_cache, index_scale, q_out, index_q_out, eps, *,
    num_index_heads, block_table, seq_lens, max_seq_len, rank, world, max_query_len, sm_scale, scores,
    n_score: int = N_SCORE, maxit: int = MAXIT, scratch=None, timeline=None,
):  # fmt: skip
    """One AQ1 launch. Arguments as ``mxfp8_qkv_norm_rope_nvfp4_insert`` + ``cp_scores_fp4`` (``scores``: fp32
    [heads, tokens, ceil(blocks / world)], slots past the live blocks left unwritten)."""
    import flydsl.expr as fx

    from .kernels.aq import launch_aq1
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
    from aiter.ops.flydsl.mxfp8_qkv_rope_nvfp4_gfx1250 import _slab

    m = x.shape[0]
    niq = num_index_heads
    n = (NQ + 2 * NKV + niq + 1) * D
    why = aq1_supported(m, niq, seq_lens.numel(), max_query_len, scores.shape[0])
    assert why is None, f"aq1: {why}"
    assert tuple(w.shape) == (n, K_DIM) and x.shape[1] == K_DIM and x.stride(1) == 1 and x.stride(0) % 16 == 0
    assert index_q_out.is_contiguous() and tuple(index_q_out.shape) == (m, niq * D)
    assert scores.dtype == torch.float32 and scores.is_contiguous() and scores.shape[1] == m
    assert block_table.dtype == torch.int32 and seq_lens.dtype == torch.int32 and block_table.stride(1) == 1
    assert index_cache.stride()[1:] == (D // 2, 1) and index_scale.stride()[1:] == (D // 16, 1)
    assert k_cache.stride() == v_cache.stride() and k_scale.stride() == v_scale.stride()
    tile_m, ns, ks, kw, wm, wn, depth, astage, wcpol = AQ1_CONFIG
    mpad = (m + tile_m - 1) // tile_m * tile_m
    slab = _slab(x.device, ks * mpad * n)
    sc = scratch or aq1_scratch(x.device, niq, max_query_len, n_score)
    blocks = -(-max_seq_len // PAGE)
    local = -(-blocks // world)
    assert scores.shape[2] == local
    eps_bits = struct.unpack("<i", struct.pack("<f", float(eps)))[0]
    stream = torch.cuda.current_stream(x.device)
    spec = (tile_m, ns, ks, kw, wm, wn, depth, niq, astage, wcpol, max_query_len, int(rank), int(world), RING, n_score,
            int(timeline is not None))
    _run_compiled(
        launch_aq1,
        ptr_arg(x), ptr_arg(x_scale), ptr_arg(w), ptr_arg(w_scale), ptr_arg(slab),
        ptr_arg(q_out), ptr_arg(index_q_out), ptr_arg(q_norm_weight), ptr_arg(k_norm_weight),
        ptr_arg(index_q_norm_weight), ptr_arg(index_k_norm_weight), ptr_arg(cos_sin_cache), ptr_arg(positions),
        ptr_arg(slot_mapping), ptr_arg(index_slot_mapping), ptr_arg(k_cache), ptr_arg(v_cache), ptr_arg(k_scale),
        ptr_arg(v_scale), ptr_arg(index_cache), ptr_arg(index_scale),
        ptr_arg(block_table, fx.Int32), ptr_arg(seq_lens, fx.Int32), ptr_arg(scores, fx.Float32),
        ptr_arg(sc["q"]), ptr_arg(sc["flag"], fx.Int32), ptr_arg(sc["cnt"], fx.Int32), ptr_arg(sc["err"], fx.Int32),
        ptr_arg(timeline if timeline is not None else sc["err"], fx.Int32),
        m, x.stride(0), mpad, cos_sin_cache.shape[0],
        k_cache.stride(0), k_scale.stride(0), index_cache.stride(0), index_scale.stride(0), eps_bits,
        block_table.stride(0), local, blocks, float(np.float32(sm_scale * LOG2E)), int(maxit),
        fx.Stream(stream),
        *spec,
        specialization_key=("aq1", x.device.index) + spec,
    )  # fmt: skip
    return scores
