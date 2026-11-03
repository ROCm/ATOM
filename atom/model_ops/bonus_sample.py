# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`ATOM_SPEC_BONUS_FUSED`: the unfiltered temperature sample in two vocab-tiled
launches instead of five.

`Sampler._temperature_sample` + `_apply_greedy_tokens` run, per call,
`mixed_sample_outer_exponential` (one program per row walking the whole vocab),
then `topk_select` (two launches), `<=` and `where`. Here:

1. `_bonus_tile_kernel` [(row, vocab tile)]: the tile's max of x = logit / T and
   its best race key exp(x - tile_max) / (E + eps) with the lowest id among equal
   keys, plus the tile's raw-logit argmax (lowest id) for greedy rows.
2. `_bonus_pick_kernel` [(row,)]: greedy rows (T <= the sampler's greedy
   epsilon) take the global raw-logit argmax, lowest id on ties -- exactly
   `topk_select(tie="low")`; sampling rows rescale every tile key to the global
   max and take the best.

The noise is the same `exponential_` draw `_temperature_sample` makes (same
generator, same tensor shape), and the race is the same: argmax_i p_i / (E_i + eps)
with p = softmax(logit / max(T, 1e-5)) (aiter's clamp), so the drawn token has
exactly the same distribution. The fp32 rounding of the keys differs from the
aiter kernel's per-thread running-max rescale, so on a near-exact tie of the two
best keys (relative gap ~1e-7) the pick can differ; greedy rows are bit-identical.
"""

import torch
import triton
import triton.language as tl

_BONUS_BLOCK_V = 4096


@triton.jit
def _bonus_tile_kernel(
    logits_ptr,
    stride_l,
    exp_ptr,
    stride_e,
    temps_ptr,
    eps,
    part_m_ptr,  # [n_tiles, rows] fp32
    part_key_ptr,  # [n_tiles, rows] fp32
    part_idx_ptr,  # [n_tiles, rows] int32
    part_gmax_ptr,  # [n_tiles, rows] fp32
    part_gidx_ptr,  # [n_tiles, rows] int32
    rows,
    vocab,
    BLOCK_V: tl.constexpr,
):
    r = tl.program_id(0)
    t = tl.program_id(1)
    offs = t * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < vocab
    x = tl.load(
        logits_ptr + r.to(tl.int64) * stride_l + offs, mask=mask, other=-float("inf")
    ).to(tl.float32)
    temp = tl.load(temps_ptr + r).to(tl.float32)
    inv_t = 1.0 / tl.maximum(temp, 1e-5)
    xs = x * inv_t
    m = tl.max(xs, axis=0)
    e = tl.load(exp_ptr + r.to(tl.int64) * stride_e + offs, mask=mask, other=1.0) + eps
    # A tile the required channel masked to -inf has m = -inf, and
    # exp(-inf - -inf) is NaN. Shifting by 0 instead keys every id 0: the tile
    # loses to any tile with a finite logit and its idx stays in range.
    m_safe = tl.where(m == -float("inf"), 0.0, m)
    key = tl.where(mask, tl.exp(xs - m_safe) / e, -1.0)
    best = tl.max(key, axis=0)
    idx = tl.min(tl.where((key == best) & mask, offs, vocab), axis=0)
    gmax = tl.max(x, axis=0)
    gidx = tl.min(tl.where((x == gmax) & mask, offs, vocab), axis=0)
    o = t * rows + r
    tl.store(part_m_ptr + o, m)
    tl.store(part_key_ptr + o, best)
    tl.store(part_idx_ptr + o, idx)
    tl.store(part_gmax_ptr + o, gmax)
    tl.store(part_gidx_ptr + o, gidx)


@triton.jit
def _bonus_pick_kernel(
    out_ptr,  # [rows] int32
    temps_ptr,
    greedy_eps,
    part_m_ptr,
    part_key_ptr,
    part_idx_ptr,
    part_gmax_ptr,
    part_gidx_ptr,
    rows,
    n_tiles,
    vocab,
    BLOCK_T: tl.constexpr,
):
    r = tl.program_id(0)
    ti = tl.arange(0, BLOCK_T)
    tm = ti < n_tiles
    o = ti * rows + r
    temp = tl.load(temps_ptr + r).to(tl.float32)
    if temp <= greedy_eps:
        g = tl.load(part_gmax_ptr + o, mask=tm, other=-float("inf"))
        gi = tl.load(part_gidx_ptr + o, mask=tm, other=vocab)
        gm = tl.max(g, axis=0)
        tok = tl.min(tl.where((g == gm) & tm, gi, vocab), axis=0)
    else:
        m = tl.load(part_m_ptr + o, mask=tm, other=-float("inf"))
        k = tl.load(part_key_ptr + o, mask=tm, other=-1.0)
        ki = tl.load(part_idx_ptr + o, mask=tm, other=vocab)
        mm = tl.max(m, axis=0)
        # An all -inf row would otherwise compute exp(-inf - -inf) here too.
        mm = tl.where(mm == -float("inf"), 0.0, mm)
        ks = tl.where(tm, k * tl.exp(m - mm), -1.0)
        best = tl.max(ks, axis=0)
        tok = tl.min(tl.where((ks == best) & tm, ki, vocab), axis=0)
    tl.store(out_ptr + r, tok.to(tl.int32))


def bonus_temperature_sample(
    logits: torch.Tensor,  # [rows, V]
    temperatures: torch.Tensor,  # [rows] fp32, pre-clamped to SAMPLER_EPS
    exponential: torch.Tensor,  # [rows, V] fp32 Exp(1) noise (unit inner stride)
    eps: float,
    greedy_eps: float,
) -> torch.Tensor:
    rows, vocab = logits.shape
    out = torch.empty(rows, dtype=torch.int, device=logits.device)
    if rows == 0:
        return out
    assert logits.stride(1) == 1 and exponential.stride(1) == 1
    n_tiles = triton.cdiv(vocab, _BONUS_BLOCK_V)
    f32 = {"dtype": torch.float32, "device": logits.device}
    i32 = {"dtype": torch.int32, "device": logits.device}
    part_m = torch.empty((n_tiles, rows), **f32)
    part_key = torch.empty((n_tiles, rows), **f32)
    part_idx = torch.empty((n_tiles, rows), **i32)
    part_gmax = torch.empty((n_tiles, rows), **f32)
    part_gidx = torch.empty((n_tiles, rows), **i32)
    temps = temperatures.contiguous()
    _bonus_tile_kernel[(rows, n_tiles)](
        logits,
        logits.stride(0),
        exponential,
        exponential.stride(0),
        temps,
        eps,
        part_m,
        part_key,
        part_idx,
        part_gmax,
        part_gidx,
        rows,
        vocab,
        BLOCK_V=_BONUS_BLOCK_V,
        num_warps=8,
    )
    _bonus_pick_kernel[(rows,)](
        out,
        temps,
        greedy_eps,
        part_m,
        part_key,
        part_idx,
        part_gmax,
        part_gidx,
        rows,
        n_tiles,
        vocab,
        BLOCK_T=max(16, triton.next_power_of_2(n_tiles)),
        num_warps=1,
    )
    return out
