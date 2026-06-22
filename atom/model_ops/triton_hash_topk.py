# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused Triton kernel for DeepSeek-V4 hash-routing topk (first hash layers).

Replaces the multi-op PyTorch path in ``MoE._hash_topk``:

    ids        = input_ids.clamp(0, vocab - 1)
    topk_ids   = tid2eid[ids]                              # [N, topk] gather
    scores     = sqrt(softplus(gating_output.float()))     # over ALL experts
    topk_w     = scores.gather(-1, topk_ids)               # keep only topk
    topk_w     = topk_w / topk_w.sum(-1, keepdim=True)      # optional renorm
    topk_w     = topk_w * routed_scaling_factor

The PyTorch version computes ``softplus``+``sqrt`` over every routed expert
(``n_routed_experts`` ~256-384) but keeps only ``topk`` (~6) of them. This
kernel computes the activation for the ``topk`` selected experts only and
fuses the id clamp, tid2eid gather, gating gather, renorm and scaling into a
single launch (one program per token).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hash_topk_kernel(
    ids_ptr,  # [N] input token ids (int)
    gating_ptr,  # [N, n_routed] router logits
    tid2eid_ptr,  # [vocab, topk] int32 token-id -> expert-id table
    out_ids_ptr,  # [N, topk] int32 (or first topk cols of a wider buffer)
    out_w_ptr,  # [N, topk] fp32 (or first topk cols of a wider buffer)
    stride_g_row,
    stride_g_col,
    stride_tid_row,
    stride_oid_row,
    stride_ow_row,
    vocab,
    topk,
    scaling,
    RENORM: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    t = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_TOPK)
    mask = offs_k < topk

    # Clamp the token id into the valid tid2eid range (guards garbage ids).
    tok = tl.load(ids_ptr + t).to(tl.int64)
    tok = tl.minimum(tl.maximum(tok, 0), vocab - 1)

    # tid2eid[tok, :topk] -> selected expert ids.
    eid = tl.load(tid2eid_ptr + tok * stride_tid_row + offs_k, mask=mask, other=0)
    eid64 = eid.to(tl.int64)

    # Gather gating logits at the selected experts, compute sqrt(softplus(.)).
    g = tl.load(
        gating_ptr + t * stride_g_row + eid64 * stride_g_col, mask=mask, other=0.0
    ).to(tl.float32)
    # Numerically stable softplus: log1p(exp(x)) ~= x for large x.
    sp = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(g)))
    w = tl.sqrt(sp)
    w = tl.where(mask, w, 0.0)

    if RENORM:
        s = tl.sum(w, axis=0)
        w = w / tl.maximum(s, 1e-20)
    w = w * scaling

    tl.store(out_ids_ptr + t * stride_oid_row + offs_k, eid, mask=mask)
    tl.store(out_w_ptr + t * stride_ow_row + offs_k, w, mask=mask)


def hash_topk_triton(
    ids: torch.Tensor,  # [N] input token ids
    gating_output: torch.Tensor,  # [N, n_routed]
    tid2eid: torch.Tensor,  # [vocab, topk] int32
    renormalize: bool,
    scaling: float,
    out_ids: torch.Tensor,  # [N, topk] int32 destination
    out_weights: torch.Tensor,  # [N, topk] fp32 destination
) -> None:
    """Fill ``out_ids`` / ``out_weights`` in place with the hash-routing result.

    ``out_ids`` / ``out_weights`` may be standalone ``[N, topk]`` tensors or
    ``[:, :topk]`` slices of a wider preallocated buffer (their row stride is
    read from the tensors, column stride is assumed 1).
    """
    num_tokens = gating_output.shape[0]
    if num_tokens == 0:
        return
    vocab, topk = tid2eid.shape
    grid = (num_tokens,)
    _hash_topk_kernel[grid](
        ids,
        gating_output,
        tid2eid,
        out_ids,
        out_weights,
        gating_output.stride(0),
        gating_output.stride(1),
        tid2eid.stride(0),
        out_ids.stride(0),
        out_weights.stride(0),
        vocab,
        topk,
        scaling,
        RENORM=renormalize,
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=1,
    )
