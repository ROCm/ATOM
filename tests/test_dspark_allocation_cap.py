# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`AllocationCap`, the allocation staging the paged block drafts share
(Kimi-K3 DSpark GQA, MiniMax-M3.1 DSpark), on CPU."""

import types

import pytest


@pytest.fixture
def host_staging(monkeypatch):
    """`AllocationCap`'s pinned staging and CUDA event, on a host without a
    GPU: unpinned buffers and an event that has nothing to wait for."""
    import torch

    if torch.cuda.is_available():
        return
    zeros = torch.zeros

    def unpinned(*args, pin_memory=False, **kwargs):
        return zeros(*args, **kwargs)

    class _Event:
        def synchronize(self):
            pass

        def record(self):
            pass

    monkeypatch.setattr(torch, "zeros", unpinned)
    monkeypatch.setattr(torch.cuda, "Event", _Event)


@pytest.mark.parametrize("unit", [128, 128 * 8])
def test_allocation_cap_masks_writes_past_each_rows_allocation(host_staging, unit):
    """Rows past `scheduled_bs` and rows holding no page read one page; slots
    at or past the allocation (in tokens, `unit` per page) are dropped."""
    import torch

    from atom.models.dspark_draft import AllocationCap

    cap = AllocationCap(4, "cpu")
    # Row 0 holds one page and its block crosses into a second; row 1 holds
    # two and stays inside; row 2 is a pad row.
    pos = torch.tensor([[unit - 2, unit - 1, unit], [5, 6, 7], [1, 2, 3]])
    slots = torch.tensor([[10, 11, 12], [20, 21, 22], [-1, -1, -1]])
    alloc = cap.apply([1, 2, 9], 2, pos, slots, unit)
    assert alloc.tolist() == [unit, 2 * unit, unit]
    assert slots.tolist() == [[10, 11, -1], [20, 21, 22], [-1, -1, -1]]
    # The staging is reused: a later, smaller step sees its own pages.
    slots = torch.tensor([[30, 31, 32]])
    alloc = cap.apply([0], 1, pos[:1], slots, unit)
    assert alloc.tolist() == [unit]
    assert slots.tolist() == [[30, 31, -1]]


def test_allocation_cap_leaves_the_minimum_to_each_caller(host_staging):
    """MiniMax-M3.1 shape: page-128 units, rows capped into their own buffer
    while `ctx_lens` stays as the proposer built it. Kimi-K3 DSpark GQA shape: one
    pool entry (block size x DCP) per block id, `ctx_lens` capped in place."""
    import torch

    from atom.models.dspark_draft import AllocationCap

    T = 3
    pos = torch.tensor([[126, 127, 128], [5, 6, 7], [1, 2, 3]])
    ctx = torch.tensor([129, 8, 1 + T], dtype=torch.int32)

    cap = AllocationCap(4, "cpu")
    ctx_eff = torch.zeros(4, dtype=torch.int32)
    alloc = cap.apply([1, 2], 2, pos, pos.clone(), 128)
    torch.minimum(ctx, alloc, out=ctx_eff[:3])
    assert ctx.tolist() == [129, 8, 1 + T]
    assert ctx_eff[:3].tolist() == [128, 8, 1 + T]

    entry = 128 * 8
    pos = torch.tensor([[entry - 2, entry - 1, entry], [5, 6, 7], [1, 2, 3]])
    ctx = torch.tensor([entry + 1, 8, 1 + T], dtype=torch.int32)
    alloc = cap.apply([1, 2], 2, pos, pos.clone(), entry)
    torch.minimum(ctx, alloc, out=ctx)
    assert ctx.tolist() == [entry, 8, 1 + T]


def test_minimax_m31_dspark_caps_the_rows_but_not_ctx_lens(host_staging):
    """MiniMax-M3.1's block rows read at most the allocation; the proposer's
    `ctx_lens` (`attn_metadata.context_lens`) is left as it built it."""
    import torch

    m3_dspark = pytest.importorskip(
        "atom.models.minimax_m3.dspark",
        reason="the MiniMax-M3.1 draft imports aiter at module load",
        exc_type=ImportError,
    )
    MiniMaxM3DSpark = m3_dspark.MiniMaxM3DSpark

    draft = MiniMaxM3DSpark.__new__(MiniMaxM3DSpark)
    draft.max_num_seqs, draft.block_size, draft._bt16 = 4, 128, None
    draft.sliding_window = 0
    T = 3
    pos = torch.tensor([[126, 127, 128], [5, 6, 7], [1, 2, 3]])
    slots = torch.tensor([[10, 11, 12], [20, 21, 22], [-1, -1, -1]])
    ctx = torch.tensor([129, 8, 1 + T], dtype=torch.int32)
    md = types.SimpleNamespace()
    block_tables = torch.tensor([[3, 0], [4, 5], [0, 0]], dtype=torch.int32)
    draft.prepare_paged_block(md, block_tables, pos, slots, ctx, [1, 2], 2)
    assert slots.tolist() == [[10, 11, -1], [20, 21, 22], [-1, -1, -1]]
    assert ctx.tolist() == [129, 8, 1 + T]
    assert md.dspark_ctx_rows.tolist() == [128] * T + [8] * T + [1 + T] * T
