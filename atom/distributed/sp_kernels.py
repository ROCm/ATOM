# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dtype-preserving packing of the Ulysses send buffer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_fields_kernel(
    source,
    output,
    TOKENS,
    STRIDE,
    WIDTH: tl.constexpr,
    WORLD: tl.constexpr,
    SPEC: tl.constexpr,
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
    source_col = tl.full((BLOCK,), 0, tl.int32)
    start = 0
    for field in tl.static_range(len(SPEC)):
        offset = SPEC[field][0]
        full_width = SPEC[field][1]
        shards = SPEC[field][2] or WORLD
        size = full_width // shards
        base = offset + (peer // (WORLD // shards)) * size
        source_col = tl.where(
            (col >= start) & (col < start + size), base + col - start, source_col
        )
        start += size
    source += first_row * STRIDE
    value = tl.load(source + row * STRIDE + source_col, row < rows_left, other=0)
    output += peer.to(tl.int64) * TOKENS * WIDTH + first
    tl.store(output + tl.arange(0, BLOCK), value, row < rows_left)


def pack_fields(source: torch.Tensor, spec: tuple, world: int) -> torch.Tensor:
    """Pack [tokens, fields] into [destination, tokens, local fields].

    ``spec`` contains (column offset, full width, head shards) triples. Zero
    shards means world; one replicates the field. Fewer shards than ranks
    replicate each KV head to its consecutive query-head owners. This is a
    pure copy, including indexer keys; no payload arithmetic or cast occurs.
    The source may have a padded row stride.
    """
    assert source.ndim == 2 and source.stride(1) == 1
    width = sum(size // (shards or world) for _, size, shards in spec)
    output = source.new_empty((world, source.shape[0], width))
    if output.numel():
        _pack_fields_kernel[(triton.cdiv(source.shape[0] * width, 1024), world)](
            source,
            output,
            source.shape[0],
            source.stride(0),
            width,
            world,
            tuple(spec),
            1024,
        )
    return output
