# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Point a native DCP sparse MLA work plan at the current layer's keys.

The metadata builder plans the forward's query rows once, one whole row a work
item. Decode templates also reserve an item for every empty row. Only each row's
rank-owned key range changes from one DSA layer to the next, so a layer rewrites
the items' KV start and end instead of planning again.
"""

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _patch_work_kv_ranges(work_info, items, kv_indptr, BLOCK: tl.constexpr):
    # aiter's MlaWorkInfo (csrc/include/mla.h) is eight int32: batch_idx (the
    # row) at 0, kv_start and kv_end at 4, 5.
    item = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = item < tl.load(items)
    info = work_info + item * 8
    row = tl.load(info, mask=live, other=0)
    # The planner counts KV from the indptr's first entry.
    first = tl.load(kv_indptr)
    tl.store(info + 4, tl.load(kv_indptr + row, mask=live) - first, mask=live)
    tl.store(info + 5, tl.load(kv_indptr + row + 1, mask=live) - first, mask=live)


def patch_work_kv_ranges(
    work_info_set: torch.Tensor,
    work_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    rows: int,
) -> None:
    """Give each work item of a whole-row plan over ``rows`` query rows its
    row's range in ``kv_indptr``.

    The plan must hold at most one unsplit work item a row, as a plan at a KV
    granularity no row exceeds does. A row with no keys can get no item, so
    the plan's own item count (``work_indptr[-1]``) bounds the patch. aiter's
    persistent MLA kernel takes a row's keys from its work item alone, not
    from the indptr it is passed.
    """
    assert work_info_set.dtype == torch.int32 and work_info_set.shape[-1] == 8, (
        "aiter's MlaWorkInfo layout changed: work_info_set is "
        f"{tuple(work_info_set.shape)} {work_info_set.dtype}, the patch writes [*, 8] int32"
    )
    _patch_work_kv_ranges[(-(-rows // BLOCK),)](
        work_info_set, work_indptr[-1:], kv_indptr, BLOCK=BLOCK
    )
