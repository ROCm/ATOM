# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Remap a routed topk into the all2all backend's id space.

The backend derives the destination rank from the raw expert id (MoRI:
`destExpert / numExpertPerRank`), so a shared expert pinned on every rank costs
each rank a slot and pushes the routed ids out of alignment:

    rank r owns dispatch slots [r*S, r*S + S)
    routed physical p -> p + num_fused_shared_experts * (p // routed_per_rank)
    this rank's shared ->  r*S + routed_per_rank

Only for SharedExpertMode.LOCAL_REPLICA; under EPLB the shared expert is an
ordinary routed logical expert and placement handles it.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only builds
    _HAS_TRITON = False


def _remap_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_routed_physical: int,
    routed_per_rank: int,
    shared_base: int,
    num_fused_shared_experts: int,
    shared_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference path, and the fallback without triton."""
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
    shared_w = torch.full(
        (num_tokens, num_fused_shared_experts),
        shared_weight,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    return (
        torch.cat((topk_weights, shared_w), dim=1),
        torch.cat((routed, shared_ids), dim=1),
    )


if _HAS_TRITON:

    @triton.jit
    def _remap_kernel(
        topk_ids_ptr,  # [ntok, topk]      routed physical ids
        topk_w_ptr,  # [ntok, topk]      routed weights
        out_ids_ptr,  # [ntok, out_width] dispatch-space ids
        out_w_ptr,  # [ntok, out_width] weights
        num_routed_physical,
        routed_per_rank,
        shared_base,
        shared_weight,
        topk,
        n_out,
        out_width,
        NUM_SHARED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Iterates over output elements, so the shared column is written in
        place instead of concatenated."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out

        row = offs // out_width
        col = offs % out_width
        is_routed = col < topk
        rmask = mask & is_routed

        in_off = row * topk + col
        phys = tl.load(topk_ids_ptr + in_off, mask=rmask, other=-1).to(tl.int64)

        in_range = (phys >= 0) & (phys < num_routed_physical)
        shifted = phys + NUM_SHARED * (phys // routed_per_rank)
        disp = tl.where(in_range, shifted, phys)

        sid = shared_base + (col - topk)
        tl.store(out_ids_ptr + offs, tl.where(is_routed, disp, sid), mask=mask)

        w = tl.load(topk_w_ptr + in_off, mask=rmask, other=0.0)
        tl.store(out_w_ptr + offs, tl.where(is_routed, w, shared_weight), mask=mask)


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
    if not _HAS_TRITON or n_out == 0:
        return _remap_torch(topk_weights, topk_ids, *args)

    out_ids = torch.empty(
        (ntok, out_width), dtype=topk_ids.dtype, device=topk_ids.device
    )
    out_w = torch.empty(
        (ntok, out_width), dtype=topk_weights.dtype, device=topk_weights.device
    )

    def grid(meta_kw):
        return (triton.cdiv(n_out, meta_kw["BLOCK"]),)

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
