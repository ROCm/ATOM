# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dtype-preserving packing of the Ulysses send buffer."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["TOKENS", "STRIDE"])
def _pack_columns_kernel(
    source,
    columns,
    output,
    TOKENS,
    STRIDE,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Tensor offsets can pass 2**31 on long prefills: keep the block's base in
    # int64 and the per-element index math in int32, relative to that base.
    peer = tl.program_id(1)
    first = tl.program_id(0).to(tl.int64) * BLOCK
    first_row = first // WIDTH
    local = (first % WIDTH).to(tl.int32) + tl.arange(0, BLOCK)
    row, col = local // WIDTH, local % WIDTH
    rows_left = (TOKENS - first_row).to(tl.int32)
    source_col = tl.load(columns + peer * WIDTH + col)
    source += first_row * STRIDE
    value = tl.load(source + row * STRIDE + source_col, row < rows_left, other=0)
    output += peer.to(tl.int64) * TOKENS * WIDTH + first
    tl.store(output + tl.arange(0, BLOCK), value, row < rows_left)


def pack_columns(source: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """``[tokens, *] -> [W, tokens, columns]``: ``source[:, columns[r]]`` for
    each rank r, a pure copy. The source may have a padded row stride."""
    assert source.ndim == 2 and source.stride(1) == 1
    assert columns.dtype == torch.int32 and columns.is_contiguous()
    world, width = columns.shape
    output = source.new_empty((world, source.shape[0], width))
    if output.numel():
        _pack_columns_kernel[(triton.cdiv(source.shape[0] * width, 1024), world)](
            source,
            columns,
            output,
            source.shape[0],
            source.stride(0),
            width,
            1024,
        )
    return output
