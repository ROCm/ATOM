# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`csa_translate_pack` on a striped split's rank-major selection: reading the
first `T` rows of the gathered slots through the split packs exactly what the
same rows in row order pack. Needs a GPU."""

import pytest
import torch

pytest.importorskip(
    "atom.model_ops.indexer.rows",
    reason="the indexer's rows import aiter at load",
    exc_type=ImportError,
)

from atom.model_ops.indexer.rows import RowSplit
from atom.model_ops.v4_kernels.csa_translate_pack import (
    csa_translate_pack,
    csa_translate_pack_reference,
)

ENVELOPE_ROWS = 8
CSA_BLOCK_CAPACITY = 64
INDEX_TOPK = 96


def _prefill_batch(tokens: int, world: int):
    """A prefill of two requests, ragged per-token spans, and its selection
    both in row order and as the striped split gathers it."""
    g = torch.Generator().manual_seed(tokens * 10 + world)
    batch_id = (torch.arange(tokens) >= tokens // 2).to(torch.int32)
    skip = torch.randint(0, 5, (tokens,), generator=g, dtype=torch.int32)
    valid_k = torch.randint(1, INDEX_TOPK + 1, (tokens,), generator=g)
    indptr = torch.zeros(tokens + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(skip + valid_k, 0).to(torch.int32)
    in_order = torch.randint(
        0, CSA_BLOCK_CAPACITY * 16, (tokens, INDEX_TOPK), generator=g, dtype=torch.int32
    )
    block_tables = torch.randint(1, 5000, (2, 16), generator=g, dtype=torch.int32)

    split = RowSplit.striped(tokens, 0, world)
    # Slot r holds rows r, r + world, ...; its tail is never read.
    slots = torch.full((world * split.stride, INDEX_TOPK), -9, dtype=torch.int32)
    for r in range(world):
        mine = in_order[r::world]
        slots[r * split.stride : r * split.stride + len(mine)] = mine
    positions = torch.zeros(tokens, dtype=torch.int32)
    meta = tuple(t.cuda() for t in (block_tables, positions, indptr, batch_id, skip))
    return in_order.cuda(), slots.cuda(), split, meta, int(indptr[-1])


def _pack(fn, topk_local, meta, total, stripe=None):
    block_tables, positions, indptr, batch_id, skip = meta
    dest = torch.full((total,), -7, dtype=torch.int32, device="cuda")
    fn(
        topk_local,
        block_tables,
        positions,
        indptr,
        batch_id,
        skip,
        dest,
        envelope_rows=ENVELOPE_ROWS,
        csa_block_capacity=CSA_BLOCK_CAPACITY,
        stripe=stripe,
    )
    return dest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "tokens, world", [(1000, 8), (1001, 8), (13, 4), (5, 2), (3, 4)]
)
def test_striped_selection_packs_as_rows_in_order(tokens, world):
    in_order, slots, split, meta, total = _prefill_batch(tokens, world)
    want = _pack(csa_translate_pack, in_order, meta, total)
    torch.testing.assert_close(
        _pack(csa_translate_pack_reference, in_order, meta, total), want
    )
    rank_major = slots[:tokens]
    for fn in (csa_translate_pack, csa_translate_pack_reference):
        torch.testing.assert_close(_pack(fn, rank_major, meta, total, split), want)
    # Not vacuous: the rank-major rows read in place pack something else,
    # unless every slot is one row (rank-major is then row order).
    assert split.stride == 1 or not torch.equal(
        _pack(csa_translate_pack, rank_major, meta, total), want
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_striped_selection_must_hold_every_slot():
    """A copy of the step's rows alone has lost the slots the split reads."""
    _, slots, split, meta, total = _prefill_batch(13, 4)
    with pytest.raises(ValueError):
        _pack(csa_translate_pack, slots[:13].clone(), meta, total, split)
