# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which KV slot each of this step's tokens is written to.

One expression for both sides of the forward. The two differ only in how the
token axis is shaped -- `.prefill` builds a ragged one, `.decode` a rectangle
-- and neither difference reaches the gather, which reads positions and a
packed table. Keeping it in one place is not tidiness: a slot computed two ways
is a KV write landing on another request's row, and nothing faults.
"""

from __future__ import annotations

import numpy as np


def slot_mapping(
    positions: np.ndarray,
    seqlens_q: np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    out: np.ndarray | None = None,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    """`block_table[pos // block_size] * block_size + pos % block_size`.

    `block_tables` is the packed 2-D buffer as `prepare_block_tables` leaves it,
    so a row's base is its index times the buffer's stride and no second marshal
    of the same rows happens here -- which is also why a decode step marshals
    BEFORE it asks for slots, and why a caller that trimmed its own ragged rows
    (rejected drafts) may still pass the untrimmed table: a token's block index
    stays under its own row's trimmed length, so the entries read are the same
    either way. Flattening the table has two ways to go wrong and
    they need separate guards: `.cast` refuses a non-contiguous buffer, where
    `reshape(-1)` would hand back a copy and the gather silently read zeros, but
    it reinterprets a wider dtype rather than rejecting it -- an int64 table
    reads as twice as many int32s and answers without a word.

    `out` and `scratch` are two int64 buffers of the token axis. Taking the
    offset first lets the block index land in `scratch` and be overwritten in
    place by the slot it gathers, so the only allocations left are the repeat
    and the gather itself.

    Widening at the multiply rather than after it: a block id is int32 and
    `id * block_size` overflows that for a large enough pool, which numpy would
    wrap without a word.
    """
    if block_tables.dtype != np.int32:
        # Raised, not asserted, to match `pack_rows` on the other side of this
        # buffer and to survive `python -O` as the contiguity check does.
        raise TypeError(f"block_tables must be int32, got {block_tables.dtype}")
    n = positions.shape[0]
    stride = block_tables.shape[1]
    flat = np.asarray(memoryview(block_tables).cast("B").cast("i"))
    out = np.empty(n, dtype=np.int64) if out is None else out
    blk = np.empty(n, dtype=np.int64) if scratch is None else scratch[:n]
    # Splitting the position is the step's largest array op, and `np.divmod` on
    # int64 costs 6x a shift and a mask. Positions are non-negative, so the two
    # agree whenever the block size is a power of two -- which every shipped
    # config is, though nothing rejects one that is not.
    if block_size & (block_size - 1):
        np.remainder(positions, block_size, out=out)
        np.floor_divide(positions, block_size, out=blk)
    else:
        np.bitwise_and(positions, block_size - 1, out=out)
        np.right_shift(positions, int(block_size).bit_length() - 1, out=blk)
    blk += np.repeat(np.arange(len(seqlens_q), dtype=np.int64) * stride, seqlens_q)
    np.multiply(flat[blk], block_size, out=blk, dtype=np.int64)
    return np.add(blk, out, out=out)
