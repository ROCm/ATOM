# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared-expert ID reassignment for LEGACY_AITER gathered rows."""

import torch
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
    # Routing writes -1 to every routed column of an invalid row.
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


def copy_shared_expert_ids_(
    topk_ids: torch.Tensor, shared_template: torch.Tensor, top_k: int
) -> None:
    """Copy the gathered batch's shared IDs, keeping invalid routed rows invalid.

    Only shared columns in the gathered output are written. Routed columns
    and the prefilled template stay unchanged, including on graph replay.
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
        shared_ids.masked_fill_(topk_ids[:, :1] < 0, -1)
        return

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
