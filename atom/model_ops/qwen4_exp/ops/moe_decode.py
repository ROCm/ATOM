# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-token Qwen3.8-Flash-Next MoE over FP8 per-channel experts.

At batch 1 each of the top-k experts (the fused shared expert included) is a
GEMV against its own weights. The stock path runs two sort kernels, an
activation quantization and the 1-stage asm MoE, which at this size launches
too few workgroups to use the memory bandwidth. Here it is two launches:

* `up`: one program per (expert slot, N tile) multiplies the BF16 activation
  by the gate and up rows and stores `silu(gate) * up`;
* `down`: one program per output tile runs the down projection for every slot
  and sums them with the routing weights.

Both unroll their reductions so every weight load is in flight at once.
Weights stay in AITER's `shuffle_weight((16, 16))` layout: within each
16-row x 32-byte block, bytes are ordered `[k // 16][n % 16][k % 16]`.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _shuffled_offsets(rows, k0, K: tl.constexpr, BK: tl.constexpr):
    """[rows, BK // 16, 16] byte offsets; the last axis is contiguous."""
    kc = k0 // 16 + tl.arange(0, BK // 16)
    row_part = (rows // 16) * (16 * K) + (rows % 16) * 16
    col_part = (kc // 2) * 512 + (kc % 2) * 256
    return (
        row_part[:, None, None]
        + col_part[None, :, None]
        + tl.arange(0, 16)[None, None, :]
    )


@triton.jit
def _route_slot(logits_ptr, k, N_ROUTED: tl.constexpr, TOP_K: tl.constexpr):
    """Expert id and weight of routing slot k.

    Slots 0..TOP_K-1 are the top-k routed experts, weighted by the softmax
    over the selected logits (softmax -> top-k -> renormalize); slot TOP_K is
    the fused shared expert, weighted by the sigmoid of the tail logit.
    """
    offs = tl.arange(0, N_ROUTED)
    vals = tl.load(logits_ptr + offs).to(tl.float32)
    top = tl.max(vals, axis=0)
    denom = 0.0
    my_val = 0.0
    my_id = 0
    for i in tl.static_range(TOP_K):
        m = tl.max(vals, axis=0)
        idx = tl.min(tl.where(vals == m, offs, N_ROUTED), axis=0)
        denom += tl.exp(m - top)
        my_val = tl.where(k == i, m, my_val)
        my_id = tl.where(k == i, idx, my_id)
        vals = tl.where(offs == idx, -float("inf"), vals)
    routed_w = tl.exp(my_val - top) / denom
    shared_w = tl.sigmoid(tl.load(logits_ptr + N_ROUTED).to(tl.float32))
    is_shared = k == TOP_K
    return tl.where(is_shared, N_ROUTED, my_id), tl.where(is_shared, shared_w, routed_w)


@triton.jit
def _moe_decode_up_kernel(
    x_ptr,
    ids_ptr,
    w13_ptr,
    s13_ptr,
    out_ptr,
    logits_ptr,
    wts_ptr,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ROUTE: tl.constexpr = False,
    N_ROUTED: tl.constexpr = 512,
):
    k = tl.program_id(0)
    nt = tl.program_id(1)
    j = nt * BN + tl.arange(0, BN)
    if ROUTE:
        e, wk = _route_slot(logits_ptr, k, N_ROUTED, TOPK - 1)
        if nt == 0:
            tl.store(ids_ptr + k, e)
            tl.store(wts_ptr + k, wk)
        e = e.to(tl.int64)
    else:
        e = tl.load(ids_ptr + k).to(tl.int64)
    w_base = w13_ptr + e * (2 * INTER * HIDDEN)
    s_base = s13_ptr + e * (2 * INTER)
    acc_g = tl.zeros((BN,), dtype=tl.float32)
    acc_u = tl.zeros((BN,), dtype=tl.float32)
    for k0 in tl.static_range(0, HIDDEN, BK):
        x = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
        x = tl.reshape(x, (BK // 16, 16))
        wg = tl.load(w_base + _shuffled_offsets(j, k0, HIDDEN, BK)).to(tl.float32)
        wu = tl.load(w_base + _shuffled_offsets(j + INTER, k0, HIDDEN, BK)).to(
            tl.float32
        )
        acc_g += tl.sum(tl.sum(wg * x[None, :, :], axis=2), axis=1)
        acc_u += tl.sum(tl.sum(wu * x[None, :, :], axis=2), axis=1)
    g = acc_g * tl.load(s_base + j)
    u = acc_u * tl.load(s_base + INTER + j)
    tl.store(
        out_ptr + k * INTER + j, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty)
    )


@triton.jit
def _down_dot(h_ptr, w_ptr, n, INTER: tl.constexpr):
    """sum_j W[n, j] * h[j] over INTER = 256 + 64 columns."""
    ha = tl.reshape(tl.load(h_ptr + tl.arange(0, 256)).to(tl.float32), (16, 16))
    hb = tl.reshape(tl.load(h_ptr + 256 + tl.arange(0, 64)).to(tl.float32), (4, 16))
    wa = tl.load(w_ptr + _shuffled_offsets(n, 0, INTER, 256)).to(tl.float32)
    wb = tl.load(w_ptr + _shuffled_offsets(n, 256, INTER, 64)).to(tl.float32)
    return tl.sum(tl.sum(wa * ha[None, :, :], axis=2), axis=1) + tl.sum(
        tl.sum(wb * hb[None, :, :], axis=2), axis=1
    )


@triton.jit
def _moe_decode_down_kernel(
    inter_ptr,
    ids_ptr,
    wts_ptr,
    w2_ptr,
    s2_ptr,
    out_ptr,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    BN: tl.constexpr,
):
    tl.static_assert(INTER == 320)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        e = tl.load(ids_ptr + k).to(tl.int64)
        wk = tl.load(wts_ptr + k).to(tl.float32)
        d = _down_dot(inter_ptr + k * INTER, w2_ptr + e * (HIDDEN * INTER), n, INTER)
        acc += wk * d * tl.load(s2_ptr + e * HIDDEN + n)
    tl.store(out_ptr + n, acc.to(out_ptr.dtype.element_ty))


def moe_decode_single_token(
    x: torch.Tensor,
    topk_ids: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    router_logits: torch.Tensor | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
    """`sum_k w_k * expert_k(x)` for one token.

    `w13 [E, 2I, H]` (gate rows then up rows) and `w2 [E, H, I]` are FP8 in
    the (16, 16) shuffled layout; `s13 [E, 2I]` / `s2 [E, H]` are per-channel.
    Either pass `topk_ids` / `topk_weights`, or `router_logits [1, E]` (routed
    logits then the shared expert's gate logit) and `top_k`, in which case the
    routing runs inside the first kernel.
    """
    if x.shape[0] != 1:
        raise ValueError("moe_decode_single_token expects exactly one token")
    hidden = x.shape[1]
    inter = w2.shape[-1]
    route = router_logits is not None
    if route:
        topk = top_k + 1
        topk_ids = torch.empty((1, topk), dtype=torch.int32, device=x.device)
        topk_weights = torch.empty((1, topk), dtype=torch.float32, device=x.device)
    else:
        topk = topk_ids.shape[1]
    inter_buf = torch.empty((topk, inter), dtype=x.dtype, device=x.device)
    bn1, bk1 = 16, 256
    _moe_decode_up_kernel[(topk, inter // bn1)](
        x,
        topk_ids,
        w13,
        s13,
        inter_buf,
        router_logits if route else x,
        topk_weights,
        TOPK=topk,
        HIDDEN=hidden,
        INTER=inter,
        BN=bn1,
        BK=bk1,
        ROUTE=route,
        N_ROUTED=w13.shape[0] - 1 if route else 512,
        num_warps=4,
    )
    out = torch.empty_like(x)
    bn2 = 16
    _moe_decode_down_kernel[(hidden // bn2,)](
        inter_buf,
        topk_ids,
        topk_weights,
        w2,
        s2,
        out,
        TOPK=topk,
        HIDDEN=hidden,
        INTER=inter,
        BN=bn2,
        num_warps=4,
    )
    return out
