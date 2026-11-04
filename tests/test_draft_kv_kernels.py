# SPDX-License-Identifier: MIT
"""The one-launch draft addressing kernels against the torch chains they replace.

Runs on a GPU, or on the CPU with ``TRITON_INTERPRET=1``; skipped where triton
is absent (ATOM's CPU job).
"""

import os

import pytest

pytest.importorskip("triton", reason="the kernels are triton")

import torch

from atom.spec_decode.draft_kv import draft_ctx_slots, entry_slots
from atom.spec_decode.draft_kv_kernels import (
    block_addressing_one_launch,
    draft_ctx_slots_one_launch,
)

if torch.cuda.is_available():
    DEV = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEV = "cpu"
else:
    pytestmark = pytest.mark.skip(reason="needs a GPU or TRITON_INTERPRET=1")
    DEV = "cpu"


def _tables(rows, n_cols, gen):
    return torch.randint(0, 5000, (rows, n_cols), generator=gen, dtype=torch.int32)


def _cu_ends(seqlens, rows):
    """`cu_seqlens_q[1 : rows + 1]`, flat over the padded tail."""
    ends = torch.tensor(seqlens, dtype=torch.int64).cumsum(0)
    pad = ends[-1:].expand(rows - len(seqlens))
    return torch.cat([ends, pad]).to(torch.int32)


def _ctx_slots_both(positions, seqlens, rows, tables, vblock):
    n = positions.shape[0]
    cu_ends = _cu_ends(seqlens, rows)
    ref = draft_ctx_slots(
        torch.full((n,), 7, dtype=torch.int64),
        positions,
        cu_ends,
        tables,
        torch.arange(n, dtype=torch.int32),
        vblock,
    )
    got = draft_ctx_slots_one_launch(
        torch.full((n,), 7, dtype=torch.int64, device=DEV),
        positions.to(DEV),
        cu_ends.to(DEV),
        tables.to(DEV),
        vblock,
    )
    return ref, got.cpu()


@pytest.mark.parametrize("rows", [1, 3, 8, 70])
@pytest.mark.parametrize("vblock", [16, 128, 1024])
def test_ctx_slots_match_the_torch_chain(rows, vblock):
    """Random batches, several kernel blocks of tokens, more rows than one
    `BLOCK_R` tile, positions past the table, empty requests."""
    gen = torch.Generator().manual_seed(rows * 7919 + vblock)
    n_cols = 6
    tables = _tables(rows, n_cols, gen)
    real = max(1, rows - 2)
    seqlens = torch.randint(0, 90, (real,), generator=gen).tolist()
    seqlens[0] = max(seqlens[0], 2)
    n = sum(seqlens)
    positions = torch.randint(0, (n_cols + 2) * vblock, (n,), generator=gen)
    positions[0], positions[1] = 0, n_cols * vblock  # one in, one past the table
    ref, got = _ctx_slots_both(positions, seqlens, rows, tables, vblock)
    assert torch.equal(got, ref)
    assert ref[0] >= 0 and ref[1] == -1


def test_ctx_slots_tokens_past_the_last_end_take_the_last_row():
    """`searchsorted` past every end is clamped to the last row, as in torch."""
    tables = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    ref, got = _ctx_slots_both(positions, [2, 2], 2, tables, 4)
    assert torch.equal(got, ref)
    # Tokens 4 and 5 are past both requests: row 1, page 1 (entry 6).
    assert got.tolist()[4:] == [6 * 4 + 0, 6 * 4 + 1]


def test_ctx_slots_of_an_empty_forward_launch_nothing():
    out = torch.full((0,), 7, dtype=torch.int64, device=DEV)
    tables = torch.zeros(1, 2, dtype=torch.int32, device=DEV)
    cu = torch.zeros(1, dtype=torch.int32, device=DEV)
    pos = torch.zeros(0, dtype=torch.int64, device=DEV)
    assert draft_ctx_slots_one_launch(out, pos, cu, tables, 16).numel() == 0


def _eager_block_addressing(anchor, tables, T, scheduled_bs, pad_len, entry_tokens):
    """The unsharded branch of `_build_paged_block_metadata`, op for op."""
    running_bs = anchor.shape[0]
    positions = torch.empty(running_bs, T, dtype=torch.int64)
    offsets = torch.arange(1, T + 1, dtype=torch.int64)
    torch.add(anchor.view(running_bs, 1), offsets.view(1, T), out=positions)
    slots = torch.empty(running_bs, T, dtype=torch.int64)
    entry_slots(slots, tables, None, positions, entry_tokens)
    ctx_lens = torch.empty(running_bs, dtype=torch.int32)
    ctx_lens.copy_(anchor + (1 + T))
    ctx_lens.clamp_(max=tables.shape[1] * entry_tokens)
    if running_bs > scheduled_bs:
        slots[scheduled_bs:].fill_(-1)
        ctx_lens[scheduled_bs:].fill_(pad_len)
    return positions, slots, ctx_lens


@pytest.mark.parametrize("T", [1, 3, 4, 7])
@pytest.mark.parametrize("running_bs,scheduled_bs", [(1, 1), (5, 5), (8, 3), (4, 0)])
@pytest.mark.parametrize("entry_tokens", [16, 1024])
def test_block_addressing_matches_the_torch_chain(
    T, running_bs, scheduled_bs, entry_tokens
):
    """Anchors anywhere in the table, including its final page (positions
    past the table) and a pad tail; every output, every element."""
    gen = torch.Generator().manual_seed(T * 131 + running_bs * 17 + entry_tokens)
    n_cols = 5
    tables = _tables(running_bs, n_cols, gen)
    span = n_cols * entry_tokens
    anchor = torch.randint(0, span, (running_bs,), generator=gen)
    anchor[0] = span - 2  # its block runs off the table
    pad_len = min(1 + T, entry_tokens)
    want = _eager_block_addressing(
        anchor, tables, T, scheduled_bs, pad_len, entry_tokens
    )

    positions = torch.full((running_bs, T), -9, dtype=torch.int64, device=DEV)
    slots = torch.full((running_bs, T), -9, dtype=torch.int64, device=DEV)
    ctx_lens = torch.full((running_bs,), -9, dtype=torch.int32, device=DEV)
    block_addressing_one_launch(
        positions,
        slots,
        ctx_lens,
        anchor.to(DEV),
        tables.to(DEV),
        scheduled_bs,
        pad_len,
        entry_tokens,
    )
    for got, ref in zip((positions, slots, ctx_lens), want):
        assert torch.equal(got.cpu(), ref)


def test_block_addressing_takes_a_row_stride_wider_than_the_table():
    """Rows are addressed by the table's row stride, not by its width."""
    gen = torch.Generator().manual_seed(3)
    full = _tables(8, 40, gen)
    tables = full.to(DEV)[:3, :30]
    assert tables.stride(0) == 40
    anchor = torch.tensor([5, 300, 478], dtype=torch.int64)
    want = _eager_block_addressing(anchor, full[:3, :30], 4, 3, 5, 16)
    positions = torch.empty(3, 4, dtype=torch.int64, device=DEV)
    slots = torch.empty(3, 4, dtype=torch.int64, device=DEV)
    ctx_lens = torch.empty(3, dtype=torch.int32, device=DEV)
    block_addressing_one_launch(
        positions, slots, ctx_lens, anchor.to(DEV), tables, 3, 5, 16
    )
    for got, ref in zip((positions, slots, ctx_lens), want):
        assert torch.equal(got.cpu(), ref)
