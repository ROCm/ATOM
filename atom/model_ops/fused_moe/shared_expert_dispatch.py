# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared-expert ID placement for dispatch and gathered rows.

The backend derives the destination rank from the raw expert id (MoRI:
`destExpert / numExpertPerRank`), so a shared expert pinned on every rank costs
each rank a slot and pushes the routed ids out of alignment:

    rank r owns dispatch slots [r*S, r*S + S)
    routed physical p -> p + num_fused_shared_experts * (p // routed_per_rank)
    this rank's shared ->  r*S + routed_per_rank

The dispatch remap is only for SharedExpertMode.LOCAL_REPLICA; under EPLB the
shared expert is an ordinary routed logical expert and placement handles it.
Gathered LEGACY_AITER rows instead copy their destination rank's shared IDs,
keeping padding invalid.
"""

import torch

from atom.model_ops.fused_moe.expert_layout import topk_padding_mask


def _remap_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_routed_physical: int,
    routed_per_rank: int,
    shared_base: int,
    num_fused_shared_experts: int,
    shared_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for dispatch-space placement."""
    num_tokens = topk_ids.shape[0]
    shifted = topk_ids + num_fused_shared_experts * (topk_ids // routed_per_rank)
    # Out-of-range ids pass through: floor division would send -1 to -2.
    in_range = (topk_ids >= 0) & (topk_ids < num_routed_physical)
    routed = torch.where(in_range, shifted, topk_ids)

    shared_ids = torch.arange(
        shared_base,
        shared_base + num_fused_shared_experts,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    ).expand(num_tokens, num_fused_shared_experts)
    # A pad row (routed to -1 throughout) takes no shared expert either.
    pad = topk_padding_mask(topk_ids)
    shared_ids = torch.where(pad, -1, shared_ids)
    shared_w = torch.where(pad, 0.0, shared_weight).to(topk_weights.dtype)
    return (
        torch.cat(
            (topk_weights, shared_w.expand(num_tokens, num_fused_shared_experts)),
            dim=1,
        ),
        torch.cat((routed, shared_ids), dim=1),
    )


def copy_shared_expert_ids_(
    topk_ids: torch.Tensor, shared_template: torch.Tensor, top_k: int
) -> None:
    """Copy shared IDs in place, preserving invalid routed rows.

    One GPU launch, with no temporary mask/output tensor. The template is
    read-only, including across a real -> idle -> real graph replay.
    """
    rows, width = topk_ids.shape
    num_shared = width - top_k
    assert shared_template.shape == (rows, num_shared)
    n_out = rows * num_shared
    if n_out == 0:
        return
    if not topk_ids.is_cuda:
        shared_ids = topk_ids[:, top_k:]
        shared_ids.copy_(shared_template)
        shared_ids.masked_fill_(topk_padding_mask(topk_ids), -1)
        return

    from atom.model_ops.fused_moe.shared_expert_dispatch_triton import (
        _copy_shared_ids_kernel,
    )

    _copy_shared_ids_kernel[((n_out + 255) // 256,)](
        topk_ids,
        shared_template,
        n_out,
        *topk_ids.stride(),
        *shared_template.stride(),
        top_k,
        num_shared,
        BLOCK=256,
    )


def remap_topk_to_dispatch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_routed_physical: int,
    routed_per_rank: int,
    shared_base: int,
    num_fused_shared_experts: int,
    shared_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Widen routed ids and append this rank's shared column. -> (weights, ids).

    The torch form is several elementwise kernels plus two `torch.cat` per layer
    per step, inside the `aiter.moe_forward` custom op that inductor cannot fuse
    into -- hence the hand-written kernel.
    """
    args = (
        num_routed_physical,
        routed_per_rank,
        shared_base,
        num_fused_shared_experts,
        shared_weight,
    )
    ntok, topk = topk_ids.shape
    out_width = topk + num_fused_shared_experts
    n_out = ntok * out_width
    if not topk_ids.is_cuda or n_out == 0:
        return _remap_torch(topk_weights, topk_ids, *args)

    from atom.model_ops.fused_moe.shared_expert_dispatch_triton import _remap_kernel

    out_ids = torch.empty(
        (ntok, out_width), dtype=topk_ids.dtype, device=topk_ids.device
    )
    out_w = torch.empty(
        (ntok, out_width), dtype=topk_weights.dtype, device=topk_weights.device
    )

    def grid(meta_kw):
        return ((n_out + meta_kw["BLOCK"] - 1) // meta_kw["BLOCK"],)

    _remap_kernel[grid](
        topk_ids.contiguous(),
        topk_weights.contiguous(),
        out_ids,
        out_w,
        num_routed_physical,
        routed_per_rank,
        shared_base,
        shared_weight,
        topk,
        n_out,
        out_width,
        NUM_SHARED=num_fused_shared_experts,
        BLOCK=256,
    )
    return out_w, out_ids
