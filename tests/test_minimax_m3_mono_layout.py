# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MiniMax-M3 mono layer kernel's build-time layout (no GPU, no AITER)."""

import pytest

from atom.models.minimax_m3.mono import layout
from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    M31_MAX_TOKENS,
    MAX_TOKENS,
    MOE_SLOTS,
    N_ROUTED,
    TOKEN_TILE,
    TOP_K,
    IndexHeads,
    tile_rows,
    token_mask_i32,
    token_tiles,
)
from atom.mono.plan.execution import BLOCKS, WAVES

TOKEN_COUNTS = range(1, MAX_TOKENS + 1)


def _regions(table):
    """(name, start, end) of a ``_layout`` table in address order."""
    starts = sorted((off, name) for name, off in table.items() if name != "_bytes")
    ends = [off for off, _ in starts[1:]] + [table["_bytes"]]
    return [(name, off, end) for (off, name), end in zip(starts, ends)]


@pytest.mark.parametrize(
    "table",
    [
        layout.SCRATCH,
        layout.M31_SMALL_SCRATCH,
        layout.M31_SCRATCH,
        layout.sym_layout(4),
        layout.sym_layout(4, M31_MAX_TOKENS),
    ],
    ids=["scratch", "scratch_m31_s16", "scratch_s32", "sym", "sym_s32"],
)
def test_regions_are_disjoint_nonempty_and_aligned(table):
    for name, start, end in _regions(table):
        assert start % 256 == 0, name
        assert end > start, name


@pytest.mark.parametrize("capacity", [MAX_TOKENS, M31_MAX_TOKENS])
@pytest.mark.parametrize("npes", [1, 2, 4, 8])
def test_symmetric_regions_hold_every_rank_and_token(npes, capacity):
    table = {
        name: end - start
        for name, start, end in _regions(layout.sym_layout(npes, capacity))
    }
    # (value, tag) pairs of two bf16 each: 4 bytes of payload per 8
    row_bytes = HIDDEN * 4
    assert table["attn"] >= npes * capacity * row_bytes
    assert table["ffn"] >= npes * capacity * row_bytes
    assert table["a_ag"] >= capacity * row_bytes
    assert table["ffn_ag"] >= capacity * row_bytes


def test_native_attention_scales_do_not_widen_legacy_scratch():
    assert "attnq_sc" not in layout.SCRATCH
    assert "attnq_sc" in layout.M31_SMALL_SCRATCH


def test_s32_uses_sixteen_row_tiles_and_signed_i32_mask():
    assert token_tiles(17) == [(0, 16), (16, 1)]
    assert token_tiles(24) == [(0, 16), (16, 8)]
    assert token_tiles(32) == [(0, 16), (16, 16)]
    assert tile_rows(M31_MAX_TOKENS) == TOKEN_TILE
    assert token_mask_i32(31) == 0x7FFFFFFF
    assert token_mask_i32(32) == -1


def test_s32_pool_holds_one_activation_tile():
    metadata_words = 2 * (TOP_K * M31_MAX_TOKENS + 1) * TOKEN_TILE
    assert layout.pool_words(M31_MAX_TOKENS) == (
        TOKEN_TILE * (layout.XN8_ROW + layout.XSC_ROW) + WAVES * 64 * 4 + metadata_words
    )


@pytest.mark.parametrize("tokens", [layout.WIDE_FROM, MAX_TOKENS])
def test_legacy_wide_pool_layout_is_unchanged(tokens):
    rows = MOE_SLOTS * tokens + 1
    assert layout.pool_words(tokens) == max(
        WAVES * layout.O_K,
        tokens * (layout.XN8_ROW + layout.XSC_ROW + N_ROUTED) + WAVES * 64 * 4,
        rows * (layout.MID_ROW + layout.MIDSC_ROW),
    )


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_every_routed_up_gate_task_runs_once(tokens):
    total = sum(layout.ug_tasks_of(cta, tokens) for cta in range(BLOCKS))
    assert total == TOP_K * layout.UG_PER_SLOT * tokens


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_shared_fp8_mailboxes_cover_narrow_and_wide_widths(tokens):
    names = [
        d.name for d in layout.mailbox_regions(tokens, 1, fuse_k1=True, shared_fp8=True)
    ]
    assert "mid" in names
    assert ("ugdone" in names) == layout.wide_moe(tokens)
    assert layout.diag_region_names(tokens, 1, shared_fp8=True)[: len(names)] == names


@pytest.mark.parametrize(
    "row", ["XN8_ROW", "XSC_ROW", "MID_ROW", "MIDSC_ROW", "O_ROW", "OSC_ROW"]
)
def test_padded_lds_rows_put_sixteen_rows_in_distinct_bank_groups(row):
    """The 16 rows an MFMA operand gathers (16 B a lane) hit 16 different groups
    of 4 of the 64 LDS banks; an unpadded pitch puts them all in one."""
    pitch = getattr(layout, row)
    assert len({(r * pitch) % 64 // 4 for r in range(16)}) == 16


@pytest.mark.parametrize("count, own", [(1, 0), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_index_heads_locate_the_fused_projection_segments(count, own):
    # the fused projection's segments in order (q heads, k, v, index q, index k),
    # as MinimaxM3QKVParallelLinearWithIndexer emits them
    sizes = [LOCAL_Q_HEADS * HEAD_DIM, HEAD_DIM, HEAD_DIM, count * HEAD_DIM, HEAD_DIM]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    heads = IndexHeads(count, own)
    assert heads.rows == sum(sizes)
    assert heads.iq_off == starts[3] + own * HEAD_DIM
    assert heads.ik_off == starts[4]


@pytest.mark.parametrize("count, own", [(2, 0), (4, 4), (1, 1)])
def test_index_heads_reject_other_head_sets(count, own):
    with pytest.raises(AssertionError):
        IndexHeads(count, own)
