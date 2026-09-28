# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared vision routing from ROCm/ATOM#2149 (35c9259501b9).

One kernel selects text/image bias, top-k experts and unbiased routing weights.
V4 identifies images by above-vocabulary IDs and optionally hashes text tokens;
V4.1 supplies its explicit image mask. Expert execution belongs to FusedMoE.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx
from triton.language.extra.hip import libdevice


@triton.jit
def _compare_swap(a, b, ia, ib):
    swap = a < b
    return (
        tl.where(swap, b, a),
        tl.where(swap, a, b),
        tl.where(swap, ib, ia),
        tl.where(swap, ia, ib),
    )


@triton.jit
def _join_expert_groups(a, b, c, d, e, f, g, h):
    # Inverse of the register-group split in _sort_v4_experts.
    even = tl.reshape(tl.join(tl.join(a, e), tl.join(c, g)), (64, 4))
    odd = tl.reshape(tl.join(tl.join(b, f), tl.join(d, h)), (64, 4))
    return tl.reshape(tl.trans(tl.reshape(tl.join(even, odd), (64, 8))), (512,))


@triton.jit
def _sort_v4_experts(sel):
    # AITER topk_gating_kernel_opt: six experts per 64-lane wave, sorted by
    # sort_network_desc<6>. Strict compare-swaps also determine ties within a
    # lane; a stable expert-ID sort is not equivalent. Keep the groups in
    # registers: a generic gather-based network is slower at large batches.
    even, odd = tl.split(tl.reshape(tl.trans(tl.reshape(sel, (8, 64))), (64, 4, 2)))
    v04, v26 = tl.split(tl.reshape(even, (64, 2, 2)))
    v15, v37 = tl.split(tl.reshape(odd, (64, 2, 2)))
    v0, v4 = tl.split(v04)
    v2, v6 = tl.split(v26)
    v1, v5 = tl.split(v15)
    v3, v7 = tl.split(v37)
    lane = tl.arange(0, 64)
    i0, i1, i2, i3, i4, i5 = (
        lane,
        lane + 64,
        lane + 128,
        lane + 192,
        lane + 256,
        lane + 320,
    )
    v0, v1, i0, i1 = _compare_swap(v0, v1, i0, i1)
    v2, v3, i2, i3 = _compare_swap(v2, v3, i2, i3)
    v4, v5, i4, i5 = _compare_swap(v4, v5, i4, i5)
    v0, v2, i0, i2 = _compare_swap(v0, v2, i0, i2)
    v1, v4, i1, i4 = _compare_swap(v1, v4, i1, i4)
    v3, v5, i3, i5 = _compare_swap(v3, v5, i3, i5)
    v0, v1, i0, i1 = _compare_swap(v0, v1, i0, i1)
    v2, v3, i2, i3 = _compare_swap(v2, v3, i2, i3)
    v4, v5, i4, i5 = _compare_swap(v4, v5, i4, i5)
    v1, v2, i1, i2 = _compare_swap(v1, v2, i1, i2)
    v3, v4, i3, i4 = _compare_swap(v3, v4, i3, i4)
    v2, v3, i2, i3 = _compare_swap(v2, v3, i2, i3)
    return _join_expert_groups(v0, v1, v2, v3, v4, v5, v6, v7), _join_expert_groups(
        i0, i1, i2, i3, i4, i5, lane + 384, lane + 448
    )


@triton.jit
def _mm_topk_kernel(
    ids_ptr,  # [N] token ids; only needed for sentinels or hash routing
    image_mask_ptr,
    stride_ids,
    stride_mask,
    gating_ptr,  # [N, n_routed] router logits
    bias_ptr,  # [n_routed] fp32 text router bias
    bias_alt_ptr,  # [n_routed] fp32 image router bias
    tid2eid_ptr,  # [vocab, topk] int32 (hash layers only; else aliases bias)
    out_ids_ptr,  # [N, topk] int32
    out_w_ptr,  # [N, topk] fp32
    stride_g_row,
    stride_g_col,
    stride_tid_row,
    stride_oid_row,
    stride_ow_row,
    vocab,
    n_routed,
    scaling,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
    IS_HASH: tl.constexpr,
    HAS_IMAGE_MASK: tl.constexpr,
    V4_TIES: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    t = tl.program_id(0)
    if IS_HASH or not HAS_IMAGE_MASK:
        tok = tl.load(ids_ptr + t * stride_ids).to(tl.int64)
    if HAS_IMAGE_MASK:
        is_image = tl.load(image_mask_ptr + t * stride_mask)
    else:
        is_image = tok >= vocab

    offs_k = tl.arange(0, BLOCK_TOPK)
    k_mask = offs_k < TOPK
    offs_e = tl.arange(0, BLOCK_E)
    e_mask = offs_e < n_routed

    # ---- scores over every routed expert: sqrt(softplus(logit)) ----
    g = tl.load(
        gating_ptr + t * stride_g_row + offs_e * stride_g_col, mask=e_mask, other=0.0
    ).to(tl.float32)
    # Match AITER compute_score<SCORE_SQRTSOFTPLUS>, including the rounded
    # sqrt; an approximate sqrt can change downstream FP4 quantization.
    sp = tl.where(
        g > 20.0, g, tl.log2(1.0 + tl.exp2(g * 1.4426950408889634)) * 0.6931471805599453
    )
    scores = libdevice.sqrt(sp)

    # ---- selection bias: per-token choice between the two vectors ----
    b_base = tl.load(bias_ptr + offs_e, mask=e_mask, other=0.0).to(tl.float32)
    b_alt = tl.load(bias_alt_ptr + offs_e, mask=e_mask, other=0.0).to(tl.float32)
    sel = scores + tl.where(is_image, b_alt, b_base)
    sel = tl.where(e_mask, sel, float("-inf"))

    if V4_TIES:
        sel, expert_ids = _sort_v4_experts(sel)
        priority = (offs_e % 64) * 8 + offs_e // 64

    # ---- top-k by repeated argmax (TOPK is ~6; a full sort would cost more) ----
    top_ids = tl.zeros([BLOCK_TOPK], dtype=tl.int32)
    top_w = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
    # V4 sums in selection order and multiplies by scaling / sum once.
    weight_sum = tl.full((), 0.0, tl.float32)
    for k in tl.static_range(TOPK):
        if V4_TIES:
            # AITER resolves equal maxima with ballot + first set lane.
            winner = tl.min(tl.where(sel == tl.max(sel, 0), priority, 2147483647), 0)
            selected = priority == winner
            idx = tl.sum(tl.where(selected, expert_ids, 0), 0)
        else:
            idx = tl.argmax(sel, axis=0)
            selected = offs_e == idx
        # Weight comes from the UNBIASED score at the selected expert.
        wk = tl.sum(tl.where(offs_e == idx, scores, 0.0), axis=0)
        top_ids = tl.where(offs_k == k, idx.to(tl.int32), top_ids)
        top_w = tl.where(offs_k == k, wk, top_w)
        weight_sum += wk
        sel = tl.where(selected, float("-inf"), sel)

    if IS_HASH:
        # Text tokens on a hash layer bypass the gate entirely: their experts
        # come from the token-id table. Computed unconditionally and selected,
        # so the kernel stays branch-free.
        tokc = tl.minimum(tl.maximum(tok, 0), vocab - 1)
        eid = tl.load(
            tid2eid_ptr + tokc * stride_tid_row + offs_k, mask=k_mask, other=0
        )
        # A dummy-loaded or corrupt table would otherwise send the downstream
        # expert-weight gather out of bounds and fault the GPU.
        eid = tl.minimum(tl.maximum(eid, 0), n_routed - 1)
        hg = tl.load(
            gating_ptr + t * stride_g_row + eid.to(tl.int64) * stride_g_col,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        hsp = tl.where(
            hg > 20.0,
            hg,
            tl.log2(1.0 + tl.exp2(hg * 1.4426950408889634)) * 0.6931471805599453,
        )
        hash_w = tl.where(k_mask, libdevice.sqrt(hsp), 0.0)
        out_ids = tl.where(is_image, top_ids, eid.to(tl.int32))
        out_w = tl.where(is_image, top_w, hash_w)
    else:
        out_ids = top_ids
        out_w = top_w

    out_w = tl.where(k_mask, out_w, 0.0)
    if RENORM:
        if IS_HASH:
            weight_sum = tl.sum(out_w, axis=0)
        scaling = tl.div_rn(scaling, tl.maximum(weight_sum, 1e-20))
    out_w = out_w * scaling

    tl.store(out_ids_ptr + t * stride_oid_row + offs_k, out_ids, mask=k_mask)
    tl.store(out_w_ptr + t * stride_ow_row + offs_k, out_w, mask=k_mask)


def mm_topk_triton(
    ids: torch.Tensor | None,  # [N] token ids, optional with a non-hash mask
    gating_output: torch.Tensor,  # [N, n_routed]
    bias: torch.Tensor,  # [n_routed] fp32
    bias_alt: torch.Tensor,  # [n_routed] fp32
    tid2eid: torch.Tensor | None,  # [vocab, topk] int32 on hash layers
    vocab_size: int,
    renormalize: bool,
    scaling: float,
    out_ids: torch.Tensor,  # [N, topk] int32 destination
    out_weights: torch.Tensor,  # [N, topk] fp32 destination
    *,
    image_mask: torch.Tensor | None = None,
) -> None:
    """Fill ``out_ids`` / ``out_weights`` in place with V4 vision routing.

    Destinations may be standalone ``[N, topk]`` tensors or ``[:, :topk]`` views
    of a wider preallocated buffer (row stride is read from the tensor; column
    stride is assumed 1).
    """
    num_tokens, n_routed = gating_output.shape
    topk = out_ids.shape[1]
    is_hash = tid2eid is not None
    assert 0 < topk <= n_routed
    assert out_ids.shape == out_weights.shape == (num_tokens, topk)
    assert out_ids.dtype == torch.int32 and out_weights.dtype == torch.float32
    assert out_ids.stride(1) == out_weights.stride(1) == 1
    assert bias.shape == bias_alt.shape == (n_routed,)
    assert bias.stride(0) == bias_alt.stride(0) == 1
    if image_mask is not None:
        assert image_mask.dtype == torch.bool and image_mask.shape == (num_tokens,)
    if image_mask is None or is_hash:
        assert ids is not None and ids.shape == (num_tokens,) and vocab_size > 0
    if is_hash:
        assert tid2eid.shape == (vocab_size, topk) and tid2eid.stride(1) == 1
    if num_tokens == 0:
        return
    gfx = get_gfx()
    # Measured on gfx950 at 1/256/2048/8192 rows. Keep the PR's launch
    # configuration for hash routing and other expert/device geometries.
    warps = (
        1 if not is_hash and n_routed == 384 and topk == 6 and gfx == "gfx950" else 4
    )
    _mm_topk_kernel[(num_tokens,)](
        ids,
        image_mask,
        ids.stride(0) if ids is not None else 0,
        image_mask.stride(0) if image_mask is not None else 0,
        gating_output,
        bias,
        bias_alt,
        # The pointer is unused when IS_HASH is false, but Triton still needs a
        # real tensor to take an address from.
        tid2eid if is_hash else bias,
        out_ids,
        out_weights,
        gating_output.stride(0),
        gating_output.stride(1),
        tid2eid.stride(0) if is_hash else 0,
        out_ids.stride(0),
        out_weights.stride(0),
        vocab_size,
        n_routed,
        scaling,
        TOPK=topk,
        RENORM=renormalize,
        IS_HASH=is_hash,
        HAS_IMAGE_MASK=image_mask is not None,
        V4_TIES=n_routed == 384 and gfx.startswith("gfx9"),
        BLOCK_E=triton.next_power_of_2(n_routed),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=warps,
    )
