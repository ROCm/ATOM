# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Copy and mask shared-expert IDs after a LEGACY_AITER gather."""

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
