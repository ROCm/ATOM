# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused Triton MoE routing for DeepSeek-V4 vision checkpoints.

The vision checkpoint carries a SECOND router bias per layer, ``gate.bias_vl``,
used in place of ``gate.bias`` for image tokens. Reference (`inference/model.py`
``Gate.forward``)::

    scores    = sqrt(softplus(gate_logits))       # weights come from here, unbiased
    is_image  = input_ids >= vocab_size
    # hash layers (the first `n_hash_layers`):
    #   text  -> tid2eid[input_ids]
    #   image -> topk(scores + bias_vl)
    # every other layer:
    #   topk(scores + where(is_image, bias_vl, bias))
    weights   = scores.gather(indices) / sum      # renormalized, then * route_scale

The bias steers *selection* only; the returned weights are always gathered from
the unbiased scores. ``FusedMoE.select_experts`` takes a single ``[E]``
correction bias and cannot express a per-token choice between two of them, so
DeepSeek-V4 vision models route through here via the ``custom_routing_function``
hook instead — the same hook the hash layers already use.

Everything is branch-free on the per-token ``is_image`` predicate: both bias
vectors are loaded and selected with ``tl.where``, and on hash layers both the
table gather and the top-k are computed and selected. Only ``IS_HASH`` branches,
and that is a compile-time constant.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mm_topk_kernel(
    ids_ptr,  # [N] token ids; >= vocab marks an image sentinel
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
    BLOCK_E: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    t = tl.program_id(0)
    tok = tl.load(ids_ptr + t).to(tl.int64)
    # Image sentinels are the only ids at or above the vocabulary.
    is_image = tok >= vocab

    offs_k = tl.arange(0, BLOCK_TOPK)
    k_mask = offs_k < TOPK
    offs_e = tl.arange(0, BLOCK_E)
    e_mask = offs_e < n_routed

    # ---- scores over every routed expert: sqrt(softplus(logit)) ----
    g = tl.load(
        gating_ptr + t * stride_g_row + offs_e * stride_g_col, mask=e_mask, other=0.0
    ).to(tl.float32)
    # Numerically stable softplus: log1p(exp(x)) ~= x for large x.
    sp = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(g)))
    scores = tl.sqrt(sp)

    # ---- selection bias: per-token choice between the two vectors ----
    b_base = tl.load(bias_ptr + offs_e, mask=e_mask, other=0.0).to(tl.float32)
    b_alt = tl.load(bias_alt_ptr + offs_e, mask=e_mask, other=0.0).to(tl.float32)
    sel = scores + tl.where(is_image, b_alt, b_base)
    sel = tl.where(e_mask, sel, float("-inf"))

    # ---- top-k by repeated argmax (TOPK is ~6; a full sort would cost more) ----
    top_ids = tl.zeros([BLOCK_TOPK], dtype=tl.int32)
    top_w = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        idx = tl.argmax(sel, axis=0)
        # Weight comes from the UNBIASED score at the selected expert.
        wk = tl.sum(tl.where(offs_e == idx, scores, 0.0), axis=0)
        top_ids = tl.where(offs_k == k, idx.to(tl.int32), top_ids)
        top_w = tl.where(offs_k == k, wk, top_w)
        sel = tl.where(offs_e == idx, float("-inf"), sel)

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
        hsp = tl.where(hg > 20.0, hg, tl.log(1.0 + tl.exp(hg)))
        hash_w = tl.where(k_mask, tl.sqrt(hsp), 0.0)
        out_ids = tl.where(is_image, top_ids, eid.to(tl.int32))
        out_w = tl.where(is_image, top_w, hash_w)
    else:
        out_ids = top_ids
        out_w = top_w

    out_w = tl.where(k_mask, out_w, 0.0)
    if RENORM:
        out_w = out_w / tl.maximum(tl.sum(out_w, axis=0), 1e-20)
    out_w = out_w * scaling

    tl.store(out_ids_ptr + t * stride_oid_row + offs_k, out_ids, mask=k_mask)
    tl.store(out_w_ptr + t * stride_ow_row + offs_k, out_w, mask=k_mask)


def mm_topk_triton(
    ids: torch.Tensor,  # [N] token ids
    gating_output: torch.Tensor,  # [N, n_routed]
    bias: torch.Tensor,  # [n_routed] fp32
    bias_alt: torch.Tensor,  # [n_routed] fp32
    tid2eid: torch.Tensor | None,  # [vocab, topk] int32 on hash layers
    vocab_size: int,
    renormalize: bool,
    scaling: float,
    out_ids: torch.Tensor,  # [N, topk] int32 destination
    out_weights: torch.Tensor,  # [N, topk] fp32 destination
) -> None:
    """Fill ``out_ids`` / ``out_weights`` in place with V4 vision routing.

    Destinations may be standalone ``[N, topk]`` tensors or ``[:, :topk]`` views
    of a wider preallocated buffer (row stride is read from the tensor; column
    stride is assumed 1).
    """
    num_tokens, n_routed = gating_output.shape
    if num_tokens == 0:
        return
    topk = out_ids.shape[1]
    is_hash = tid2eid is not None
    _mm_topk_kernel[(num_tokens,)](
        ids,
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
        BLOCK_E=triton.next_power_of_2(n_routed),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=4,
    )
