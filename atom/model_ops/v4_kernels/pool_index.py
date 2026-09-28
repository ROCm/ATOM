# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The unified V4 pool's row formulas, as Triton device functions.

`v4_pool_geometry` owns the arithmetic; this is its device-side half. Five
kernels address the pool — the two index builders, the ring write, the DSpark
window gather and the CSA topk translator — and every one of them would
otherwise carry its own transcription of the same two expressions. They call
these instead, and `tests/test_pool_index.py` pins them to the Python side, so
a layout change has one place to land per side rather than six.

`ring_start` is an ordinary argument and the rest `constexpr`, but nothing here
varies once the pool is allocated — `swa_write` and `csa_translate_pack` run
inside the captured decode graph, where a by-value scalar is frozen exactly as
hard as a `constexpr`. The split between the two is about sparing a Triton
recompile on a reallocation, and buys nothing at replay. What makes the layout
safe under capture is that moving the compress/window boundary changes no
address at all; see `UnifiedPoolGeometry.window_params`.
"""

import triton
import triton.language as tl

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    UnifiedPoolGeometry,
    WindowParams,
)


def served_window_params(geometry: UnifiedPoolGeometry) -> dict[int, WindowParams]:
    """Window parameters for the classes this geometry actually has.

    A class with no layers is not in `UnifiedPoolGeometry.classes` and has no
    address to give: asking for one raises. That is easy to forget for DENSE,
    which every V4 config used to have — until a layer whose window moved into
    a state field took the last one with it, which is what a DSpark draft does
    to a trunk that is all CSA and HCA. The index builders serve one output
    buffer per class, so an absent class simply has no output; they read
    presence from here rather than each deciding what "absent" means.
    """
    return {ratio: geometry.window_params(ratio) for ratio in geometry.classes}


def window_constexprs(params: WindowParams, prefix: str = "") -> dict:
    """The `constexpr` half of `window_row`, ready to splat into a launch.

    Launch as `kernel[grid](..., params.ring_start, **window_constexprs(params))`
    so the four values are never re-listed by hand at a call site. A kernel that
    addresses more than one compress class — the two index builders do, since
    their three output buffers each serve one class — passes a `prefix` per
    class and keeps one set of parameters per output.
    """
    return {
        f"{prefix}RING_SLOTS": params.ring_slots,
        f"{prefix}SLOT_ROWS": params.slot_rows,
        f"{prefix}RING_STRIDE": params.ring_stride,
        f"{prefix}RUN_ROWS": params.run_rows,
    }


@triton.jit
def window_row(
    slot,
    pos,
    ring_start,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
):
    """Row of absolute position `pos` in physical slot `slot`'s window.

    Relative to the calling layer's own view base, and layer-independent
    within a compress class — which is what lets one index buffer serve the
    whole class. See `UnifiedPoolGeometry.window_params`.

    `pos` is the absolute token position; `pos % RING_SLOTS` is the ring
    position. Triton follows C remainder semantics, so a negative `pos` yields
    a negative row here rather than wrapping — callers must not pass the -1
    padding sentinels.
    """
    ring_pos = pos % RING_SLOTS
    return (
        slot * SLOT_ROWS
        + ring_start
        + (ring_pos // RING_STRIDE) * RUN_ROWS
        + (ring_pos % RING_STRIDE)
    )


@triton.jit
def row_offset(row, row_stride):
    """Element offset of a plane row — in 64 bits, which is not optional.

    A row index fits comfortably in 32 bits (a plane is ~150M rows) and the
    index buffers are int32 by ABI, so the row itself never needs widening.
    The product does: at 512 elements per row the plane passes 2^31 elements
    2.8% of the way in, and a window row is deliberately at the far end. A
    32-bit multiply here wraps for every window write and for all but the
    first few thousand blocks, silently.

    Every kernel that turns a row into an address goes through this rather
    than writing the multiply, so the widening cannot be dropped one site at
    a time.
    """
    return row.to(tl.int64) * row_stride


@triton.jit
def compress_row(block, row, ENVELOPE_ROWS: tl.constexpr):
    """Row of a compressed entry, relative to the calling layer's view base.

    `block` is a physical block id and `row` its index within the block for
    this layer's class (`0 <= row < block_size // ratio`). No layer term: the
    layer's offset inside the envelope is already in the view base.
    """
    return block * ENVELOPE_ROWS + row
