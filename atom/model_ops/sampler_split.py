# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-vocabulary Gumbel-max sampling for small batches (ATOM_SAMPLER_SPLIT_V).

``Sampler._temperature_sample`` scores ``exp(logit / T - max) / u`` with aiter's
``mixed_sample_outer_exponential``, one 1024-thread workgroup per row, and then
``_apply_greedy_tokens`` re-reads every row for the greedy (T <= eps) argmax. At
decode with MTP the call has 1-64 rows of a 200K vocabulary, so 1-64 workgroups
walk 200K elements each (~39 us on gfx1250 whatever the row count) while the
rest of the GPU idles.

Here each row is split into ``BLOCK``-wide chunks, one program per (chunk, row):
a chunk keeps its max ``m``, its softmax mass ``s`` at that max and its best
``exp(v - m) / u`` candidate (lowest index on ties); a second launch per row
rescales the chunk winners to the row max, divides by the row mass (as the
aiter kernel does) and keeps the best one, lowest chunk on ties. Rows with
T <= ``greedy_eps`` take the plain argmax of the raw logits (lowest index on
ties), which is exactly ``topk_select(logits, 1, tie="low")`` -- so this one
pair of launches replaces both the aiter kernel and ``_apply_greedy_tokens``.

Same noise, same score: a sampled row picks the same token as the aiter kernel
unless two candidates of the row are within float rounding of each other (the
two kernels round the running-max rescale in a different order). Greedy rows are
bitwise the same.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 4096
# Per (device, rows, chunks) partials, reused across calls (stream-ordered: each
# call's two launches finish with them before the next call's first one starts).
_WORKSPACES: dict = {}


def _workspace(device, M: int, S: int):
    key = (device, M, S)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = (
            torch.empty((M, 3, S), dtype=torch.float32, device=device),
            torch.empty((M, S), dtype=torch.int32, device=device),
        )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # A graph's private pool must not back eager calls later.
            return ws
        _WORKSPACES[key] = ws
    return ws


@triton.jit
def _split_sample_partial_kernel(
    logits_ptr,
    logits_stride,
    exp_ptr,
    exp_stride,
    temp_ptr,
    part_ptr,  # fp32 [M, 3, S]: chunk max, chunk mass, chunk best score
    idx_ptr,  # int32 [M, S]: chunk best token
    V,
    S,
    eps,
    greedy_eps,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    m = tl.program_id(1)
    offs = s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(
        logits_ptr + m.to(tl.int64) * logits_stride + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    t = tl.load(temp_ptr + m)
    greedy = t <= greedy_eps
    # Sampling score: exp(x / T - chunk max) / (u + eps), as the aiter kernel.
    inv_t = 1.0 / tl.maximum(t, 1e-5)
    v = tl.where(mask, x * inv_t, float("-inf"))
    mb = tl.max(v, axis=0)
    p = tl.where(mask, tl.exp(v - mb), 0.0)
    sb = tl.sum(p, axis=0)
    u = tl.load(exp_ptr + m.to(tl.int64) * exp_stride + offs, mask=mask, other=1.0)
    q = tl.where(mask, p / (u + eps), -1.0)
    # Greedy: the raw logit (masked lanes are -inf and never win).
    key = tl.where(greedy, x, q)
    best = tl.max(key, axis=0)
    ib = tl.argmax(key, axis=0, tie_break_left=True)
    base = part_ptr + m.to(tl.int64) * 3 * S + s
    tl.store(base, mb)
    tl.store(base + S, sb)
    tl.store(base + 2 * S, best)
    tl.store(idx_ptr + m.to(tl.int64) * S + s, (s * BLOCK + ib).to(tl.int32))


@triton.jit
def _split_sample_final_kernel(
    part_ptr,
    idx_ptr,
    temp_ptr,
    out_ptr,
    S,
    greedy_eps,
    SB: tl.constexpr,
):
    m = tl.program_id(0)
    o = tl.arange(0, SB)
    mask = o < S
    base = part_ptr + m.to(tl.int64) * 3 * S + o
    mb = tl.load(base, mask=mask, other=float("-inf"))
    sb = tl.load(base + S, mask=mask, other=0.0)
    vb = tl.load(base + 2 * S, mask=mask, other=float("-inf"))
    ib = tl.load(idx_ptr + m.to(tl.int64) * S + o, mask=mask, other=0)
    t = tl.load(temp_ptr + m)
    gmax = tl.max(mb, axis=0)
    sc = tl.where(mask, tl.exp(mb - gmax), 0.0)
    total = tl.sum(sb * sc, axis=0)
    sampled = tl.where(mask, vb * sc / total, -1.0)
    key = tl.where(t <= greedy_eps, vb, sampled)
    j = tl.argmax(key, axis=0, tie_break_left=True)
    tok = tl.sum(tl.where(o == j, ib, 0), axis=0)
    tl.store(out_ptr + m, tok.to(tl.int32))


def split_sample(
    logits: torch.Tensor,
    exponentials: torch.Tensor,
    temperatures: torch.Tensor,
    eps: float,
    greedy_eps: float,
    block: int = _BLOCK,
) -> torch.Tensor:
    """Sampled token ids (int32 ``[M]``) of ``logits`` ``[M, V]`` (row stride
    free, unit stride along V) with exponential noise ``exponentials`` ``[M, V]``
    (row stride 0 = one shared row) and per-row ``temperatures`` ``[M]`` fp32;
    rows at or below ``greedy_eps`` take the lowest-index argmax."""
    M, V = logits.shape
    out = torch.empty(M, dtype=torch.int32, device=logits.device)
    if M == 0:
        return out
    assert logits.stride(1) == 1 and exponentials.stride(1) == 1
    assert exponentials.shape[-1] == V and temperatures.numel() == M
    temps = temperatures.reshape(-1)
    if temps.dtype != torch.float32:
        temps = temps.float()
    S = triton.cdiv(V, block)
    part, idx = _workspace(logits.device, M, S)
    _split_sample_partial_kernel[(S, M)](
        logits,
        logits.stride(0),
        exponentials,
        exponentials.stride(0),
        temps,
        part,
        idx,
        V,
        S,
        float(eps),
        float(greedy_eps),
        BLOCK=block,
        num_warps=4,
    )
    _split_sample_final_kernel[(M,)](
        part,
        idx,
        temps,
        out,
        S,
        float(greedy_eps),
        SB=max(16, triton.next_power_of_2(S)),
        num_warps=1,
    )
    return out
