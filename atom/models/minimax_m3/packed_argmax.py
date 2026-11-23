# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A ``(value, id)`` argmax folded across programs with one 64-bit ``atomic_max``.

The fp32 value's bits are remapped to signed-int order in the high word and the
id's complement sits in the low word, so the signed order of the int64 is
(value, then lowest id): the pick `tile_argmax.argmax_lowest_id` makes within a
tile, now across tiles (and, after an all-gather of the packed words, across
ranks) without a reduction launch. ``-0.0`` is packed as ``+0.0`` (they compare
equal, so the lowest id wins). ``EMPTY`` is the fold's identity; a row no
program reached reads back as id 0, as `argmax_of_tiles` answers an all-NaN row.
NaNs are not ordered like `argmax_lowest_id` does: a row holding a NaN alongside
numbers is outside the same-ids guarantee.
"""

import triton
import triton.language as tl

EMPTY = -(2**63)
# Triton kernels can only read module globals that are constexpr.
EMPTY_K = tl.constexpr(EMPTY)


@triton.jit
def pack_max_id(val, idx):
    val = tl.where(val == 0.0, 0.0, val)
    bits = val.to(tl.int32, bitcast=True)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return (ordered.to(tl.int64) << 32) | (0xFFFFFFFF - idx.to(tl.int64))


@triton.jit
def unpack_id(packed):
    idx = (0xFFFFFFFF - (packed & 0xFFFFFFFF)).to(tl.int32)
    return tl.where(packed == -9223372036854775808, 0, idx)


@triton.jit
def ranks_best(
    gathered_ptr, rows, mask, num_rows, W: tl.constexpr, STRIDE: tl.constexpr
):
    """The largest packed word of each row over the W ranks' copies, laid out
    ``[W, num_rows, STRIDE]`` (column 0 read)."""
    best = tl.load(gathered_ptr + rows * STRIDE, mask=mask, other=-9223372036854775808)
    for j in tl.static_range(1, W):
        best = tl.maximum(
            best,
            tl.load(
                gathered_ptr + (j * num_rows + rows) * STRIDE,
                mask=mask,
                other=-9223372036854775808,
            ),
        )
    return best
