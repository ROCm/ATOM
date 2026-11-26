# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pack chosen rows' top-k selections in sequence order, one program a
sequence."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_selections_kernel(
    rows_ptr,
    indptr_ptr,
    indices_ptr,
    out_indptr_ptr,
    out_indices_ptr,
    WIDTH: tl.constexpr,
    LOOK: tl.constexpr,
):
    seq = tl.program_id(0)
    row = tl.load(rows_ptr + seq)
    src = tl.load(indptr_ptr + row)
    length = tl.load(indptr_ptr + row + 1) - src
    # Lands after every earlier sequence's selection.
    dst = 0
    for base in range(0, seq, LOOK):
        earlier = base + tl.arange(0, LOOK)
        mask = earlier < seq
        earlier_row = tl.load(rows_ptr + earlier, mask=mask, other=0)
        earlier_len = tl.load(indptr_ptr + earlier_row + 1, mask=mask, other=0)
        earlier_len -= tl.load(indptr_ptr + earlier_row, mask=mask, other=0)
        dst += tl.sum(earlier_len)
    offs = tl.arange(0, WIDTH)
    keep = offs < length
    selection = tl.load(indices_ptr + src + offs, mask=keep)
    tl.store(out_indices_ptr + dst + offs, selection, mask=keep)
    tl.store(out_indptr_ptr + seq + 1, dst + length)
    if seq == 0:
        tl.store(out_indptr_ptr, 0)


def pack_row_selections(
    rows: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    out_indptr: torch.Tensor,
    out_indices: torch.Tensor,
    width: int,
) -> None:
    """Write each of ``rows``' selection (``indices`` under the per-row
    ``indptr``, at most ``width`` long) to ``out_indices``, packed in sequence
    order, and ``out_indptr[: len(rows) + 1]`` addressing them per sequence.

    The outputs are other storage than the sources, so no program reads what
    another writes. A padding sequence's row (0) packs that row's selection
    again."""
    num_seqs = rows.numel()
    assert out_indptr.numel() > num_seqs
    _pack_selections_kernel[(num_seqs,)](
        rows,
        indptr,
        indices,
        out_indptr,
        out_indices,
        WIDTH=triton.next_power_of_2(width),
        LOOK=256,
    )
