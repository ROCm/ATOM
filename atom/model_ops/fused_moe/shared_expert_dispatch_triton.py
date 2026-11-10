# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""GPU kernels for shared-expert ID placement."""

import triton
import triton.language as tl


@triton.jit
def _copy_shared_ids_kernel(
    ids_ptr,
    template_ptr,
    n_out,
    IDS_ROW_STRIDE: tl.constexpr,
    IDS_COL_STRIDE: tl.constexpr,
    TEMPLATE_ROW_STRIDE: tl.constexpr,
    TEMPLATE_COL_STRIDE: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_SHARED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    row, col = offs // NUM_SHARED, offs % NUM_SHARED
    # Same padding convention as topk_padding_mask: the first routed ID.
    routed = tl.load(ids_ptr + row * IDS_ROW_STRIDE, mask=mask, other=-1)
    shared = tl.load(
        template_ptr + row * TEMPLATE_ROW_STRIDE + col * TEMPLATE_COL_STRIDE,
        mask=mask,
        other=-1,
    )
    tl.store(
        ids_ptr + row * IDS_ROW_STRIDE + (TOP_K + col) * IDS_COL_STRIDE,
        tl.where(routed < 0, -1, shared),
        mask=mask,
    )


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

    # A pad row (routed to -1 throughout) takes no shared expert either.
    pad = tl.load(topk_ids_ptr + row * topk, mask=mask & ~is_routed, other=0) < 0
    sid = tl.where(pad, -1, shared_base + (col - topk))
    tl.store(out_ids_ptr + offs, tl.where(is_routed, disp, sid), mask=mask)

    w = tl.load(topk_w_ptr + in_off, mask=rmask, other=0.0)
    sw = tl.where(pad, 0.0, shared_weight)
    tl.store(out_w_ptr + offs, tl.where(is_routed, w, sw), mask=mask)
