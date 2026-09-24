# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The device-side row formulas must agree with the host-side geometry.

`v4_pool_geometry` decides where a row lives and `pool_index` is what the
kernels actually execute; they are two transcriptions of one layout, and this
is the only place the two ever meet. Everything else — allocation, view
binding, the index builders — is downstream of whichever one it happens to
call, so a disagreement here would show up as silent corruption rather than as
a failure.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "runs the Triton device functions against their host-side source; "
        "needs a real GPU",
        allow_module_level=True,
    )

import triton
import triton.language as tl

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
    UnifiedPoolGeometry,
)
from atom.model_ops.v4_kernels.pool_index import (
    compress_row,
    row_offset,
    window_constexprs,
    window_row,
)

DEV = "cuda"
RATIOS = [0, 0] + [4, 128] * 20 + [4] + [0, 0, 0]
BLOCK_SIZE = 256
RING_SLOTS = 131


@triton.jit
def _window_row_probe(
    slot_ptr,
    pos_ptr,
    out_ptr,
    n,
    ring_start,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    mask = i < n
    slot = tl.load(slot_ptr + i, mask=mask, other=0)
    pos = tl.load(pos_ptr + i, mask=mask, other=0)
    tl.store(
        out_ptr + i,
        window_row(slot, pos, ring_start, RING_SLOTS, SLOT_ROWS, RING_STRIDE, RUN_ROWS),
        mask=mask,
    )


@triton.jit
def _compress_row_probe(
    block_ptr, row_ptr, out_ptr, n, ENVELOPE_ROWS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    mask = i < n
    block = tl.load(block_ptr + i, mask=mask, other=0)
    row = tl.load(row_ptr + i, mask=mask, other=0)
    tl.store(out_ptr + i, compress_row(block, row, ENVELOPE_ROWS), mask=mask)


@pytest.fixture(scope="module")
def geometry():
    # A plane wider than the split needs, so `ring_start` carries a gap and a
    # formula that quietly assumed a tight plane would drift.
    return UnifiedPoolGeometry(
        RATIOS,
        num_blocks=7,
        num_slots=4,
        ring_slots=RING_SLOTS,
        block_size=BLOCK_SIZE,
        plane_rows=80_000,
    )


def _run_window(params, slots, positions):
    n = len(slots)
    block = triton.next_power_of_2(n)
    out = torch.zeros(block, dtype=torch.int64, device=DEV)
    _window_row_probe[(1,)](
        torch.tensor(slots, dtype=torch.int64, device=DEV),
        torch.tensor(positions, dtype=torch.int64, device=DEV),
        out,
        n,
        params.ring_start,
        BLOCK=block,
        **window_constexprs(params),
    )
    return out[:n].tolist()


@pytest.mark.parametrize("ratio", [DENSE_RATIO, CSA_RATIO, HCA_RATIO])
def test_window_row_matches_the_geometry(geometry, ratio):
    params = geometry.window_params(ratio)
    layer = geometry.classes[ratio].layers[-1]
    slots, positions, want = [], [], []
    for slot in range(geometry.num_slots):
        # Absolute positions, not ring positions: the wrap is the kernel's job
        # and an off-by-one in it would otherwise never be exercised.
        for pos in (0, 1, 63, 64, 129, 130, 131, 262, 1000):
            slots.append(slot)
            positions.append(pos)
            want.append(geometry.window_index(layer, slot, pos % RING_SLOTS))
    assert _run_window(params, slots, positions) == want


def test_window_row_is_layer_independent_on_device(geometry):
    """Same launch, every layer of the class — the property the three per-class
    index buffers rest on, checked against what the kernel computes rather than
    against the host formula that was written to have it."""
    for ratio, cls in geometry.classes.items():
        params = geometry.window_params(ratio)
        got = _run_window(params, [2] * 3, [5, 200, 1000])
        for layer in cls.layers:
            want = [
                geometry.window_index(layer, 2, p % RING_SLOTS) for p in (5, 200, 1000)
            ]
            assert got == want, (ratio, layer)


def test_compress_row_matches_the_geometry(geometry):
    for ratio in (CSA_RATIO, HCA_RATIO):
        cls = geometry.classes[ratio]
        layer = cls.layers[-1]
        blocks, rows, want = [], [], []
        for block in range(geometry.num_blocks):
            for row in (0, cls.block_rows // 2, cls.block_rows - 1):
                blocks.append(block)
                rows.append(row)
                want.append(geometry.compress_index(layer, block, row))
        n = len(blocks)
        size = triton.next_power_of_2(n)
        out = torch.zeros(size, dtype=torch.int64, device=DEV)
        _compress_row_probe[(1,)](
            torch.tensor(blocks, dtype=torch.int64, device=DEV),
            torch.tensor(rows, dtype=torch.int64, device=DEV),
            out,
            n,
            ENVELOPE_ROWS=geometry.envelope_rows,
            BLOCK=size,
        )
        assert out[:n].tolist() == want


def test_device_rows_never_collide(geometry):
    """The host side proves this by enumeration; repeat it on what the kernels
    compute, since only their agreement makes that proof binding."""
    seen = {}
    for ratio, cls in geometry.classes.items():
        params = geometry.window_params(ratio)
        base = geometry.layer_base_row(cls.layers[-1])
        for slot in range(geometry.num_slots):
            rows = _run_window(params, [slot] * RING_SLOTS, list(range(RING_SLOTS)))
            for pos, row in enumerate(rows):
                key = base + row
                assert key not in seen, (key, seen[key], (ratio, slot, pos))
                seen[key] = (ratio, slot, pos)


@triton.jit
def _row_offset_probe(row_ptr, out_ptr, n, row_stride, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    mask = i < n
    row = tl.load(row_ptr + i, mask=mask, other=0)
    tl.store(out_ptr + i, row_offset(row, row_stride), mask=mask)


def test_row_offset_does_not_wrap_at_two_gigaelements():
    """The one arithmetic every pool kernel shares, at the width where it broke.

    A row index fits 32 bits and the index buffers carry it as int32; the byte
    offset it turns into does not. This feeds `row_offset` int32 rows — the
    same dtype the kernels load — and demands the exact product, so a widening
    dropped from any one call site shows up as a wrong number here rather than
    as a fault three kernels downstream.
    """
    row_stride = 512
    rows = [0, 1, 4_194_303, 4_194_304, 4_194_305, 100_000_000, 148_610_071]
    n = len(rows)
    block = triton.next_power_of_2(n)
    out = torch.zeros(block, dtype=torch.int64, device=DEV)
    _row_offset_probe[(1,)](
        torch.tensor(rows, dtype=torch.int32, device=DEV),
        out,
        n,
        row_stride,
        BLOCK=block,
    )
    assert out[:n].tolist() == [r * row_stride for r in rows]
