# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two per-token index arrays a prefill step has to derive.

Sequence `i` forwards tokens `[num_cached_tokens[i], context_lens[i])`, so both
arrays live on that ragged token axis flattened in sequence order -- the layout
attention reads through `cu_seqlens_q`. Everything per-sequence is an input:
this file derives nothing the step already knows, and marshals nothing --
`prepare_block_tables` owns that, and hands the packed table straight over.

Both take the caller's buffers. A step runs these once between a great deal of
unrelated allocation, so a fresh temporary is not the free list hit it looks
like in a tight loop: dropping two of them off the slot mapping is worth ~2x on
that function at the shapes a real step runs (`/app/logs_claude/
prefill_token_layout_perf.py`, which alternates arms -- timing them one after
the other hides the whole effect).
"""

from __future__ import annotations

import numpy as np


def prefill_positions(
    token_offsets: np.ndarray,
    cached_lens: np.ndarray,
    cu_seqlens_q: np.ndarray,
    seqlens_q: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Each token's absolute index in its OWN sequence.

    Token `t` of sequence `i` is at `cached_lens[i] + (t - cu_seqlens_q[i])`, so
    a chunked prefill resumes at the cached prefix rather than at zero. The
    whole per-sequence part is one repeat, leaving the token axis read once and
    written once. `token_offsets` is the flat axis, `arange(sum(seqlens_q))`,
    passed in because the caller already owns a resident one.
    """
    starts = np.repeat(cached_lens - cu_seqlens_q[:-1], seqlens_q)
    return np.add(token_offsets, starts, out=out)


def prefill_slot_mapping(
    positions: np.ndarray,
    seqlens_q: np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    out: np.ndarray | None = None,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    """Per-token KV slot: `block_table[pos // block_size] * block_size + pos %`.

    `block_tables` is the packed 2-D buffer as `prepare_block_tables` leaves it,
    so a row's base is its index times the buffer's stride and no second marshal
    of the same rows happens here. Flattening it has two ways to go wrong and
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
