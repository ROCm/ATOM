# SPDX-License-Identifier: Apache-2.0
"""Triton implementation for zeroing selected MegaMoE output rows."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_pad_rows_kernel(
    out_ptr,
    pad_ptr,
    stride_row,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(pad_ptr + row) != 0:
        offsets = tl.arange(0, BLOCK)
        for start in range(0, HIDDEN, BLOCK):
            tl.store(
                out_ptr + row * stride_row + start + offsets,
                0.0,
                mask=start + offsets < HIDDEN,
            )


def zero_pad_rows_(out: torch.Tensor, pad_rows: torch.Tensor) -> torch.Tensor:
    """Launch the GPU-only in-place row-zeroing kernel."""
    _zero_pad_rows_kernel[(out.shape[0],)](
        out,
        pad_rows.view(torch.uint8),
        out.stride(0),
        HIDDEN=out.shape[1],
        BLOCK=1024,
    )
    return out
