# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A DSpark draft's KV kept only as far back as its sliding window reads.

The window is where two components meet without sharing code: the block
pass's own read arithmetic (`_window_rows` in the MiniMax-M3 draft) and the
scheduler's pool that frees what the window has passed. They have to agree to
the row, and a disagreement does not fault: the draft reads another request's
rows and only its acceptance falls. So the geometry is checked against the read
arithmetic row by row, and the pool against the geometry.

Runs on CPU, without aiter.
"""

from __future__ import annotations

import dataclasses

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.draft_window import (
    DRAFT_WINDOW_CLASS,
    DraftWindowGeometry,
)

# MiniMax-M3.1's DSpark draft: a 4096-key window read in pages of 16, in the
# scheduler's 128-token blocks, verifying 7 drafts.
M31 = DraftWindowGeometry(window=4096, read_page=16, block_size=128, spec_tokens=7)
# The same shape at a size a test can walk: 8 keys, pages of 2, blocks of 4.
SMALL = DraftWindowGeometry(window=8, read_page=2, block_size=4, spec_tokens=1)


def _first_key_read(geo: DraftWindowGeometry, anchor: int, block_tokens: int) -> int:
    """The lowest key a block pass anchored at `anchor` reads, row by row, in
    `_window_rows`' own arithmetic: a row at p starts at the page holding
    `p - window + 1`."""
    return min(
        max(0, p - (geo.window - 1)) // geo.read_page * geo.read_page
        for p in range(anchor + 1, anchor + 1 + block_tokens)
    )


# ── geometry ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("geo", [M31, SMALL])
def test_first_read_pos_is_the_lowest_key_the_block_pass_reads(geo):
    for anchor in range(geo.window + 3 * geo.block_size):
        assert geo.first_read_pos(anchor) == _first_key_read(
            geo, anchor, max(1, geo.spec_tokens)
        ), anchor


@pytest.mark.parametrize("geo", [M31, SMALL])
def test_the_live_window_only_moves_right(geo):
    blocks = [geo.first_live_block(f) for f in range(1, 4 * geo.window)]
    assert blocks == sorted(blocks)


def test_blocks_per_req_bounds_every_run_a_sequence_holds():
    """What one sequence holds: from its window's first block to the end of a
    table the scheduler extended `scheduled_slack + lookahead` tokens past its
    finalized length. The maximum over every finalized length is the bound."""
    geo = M31
    reach = geo.scheduled_slack + geo.spec_tokens - 1
    held = [
        -(-(f + reach) // geo.block_size) - geo.first_live_block(f)
        for f in range(1, 3 * geo.window)
    ]
    assert max(held) == geo.blocks_per_req
    # A 32-block window, plus alignment and the verify look-ahead.
    assert geo.blocks_per_req == 34


def test_pages_must_tile_the_blocks():
    with pytest.raises(ValueError, match="do not tile"):
        DraftWindowGeometry(window=64, read_page=48, block_size=128, spec_tokens=1)


# ── block manager ──────────────────────────────────────────────────────────


def _bm(*, blocks=64, draft_blocks=32, prefix=True, geo=SMALL):
    cfg = MockConfig(
        num_kvcache_blocks=blocks,
        kv_cache_block_size=geo.block_size,
        enable_prefix_caching=prefix,
        pool_entries={DRAFT_WINDOW_CLASS: draft_blocks},
        draft_window_geometry=dataclasses.asdict(geo),
    )
    return BlockManager(cfg)


def _admit(bm, seq):
    hit = bm.can_allocate(seq)
    assert hit >= 0
    assert bm.allocate(seq, hit)
    return hit


def _prefill(bm, seq):
    """Admit and run a whole prompt: hit, allocate, publish what it computed."""
    hit = _admit(bm, seq)
    bm.hash_blocks(seq, len(seq) - seq.num_cached_tokens)
    seq.num_cached_tokens = len(seq)
    return hit


def test_without_a_draft_window_nothing_changes(seq_factory):
    cfg = MockConfig(num_kvcache_blocks=16, kv_cache_block_size=4)
    bm = BlockManager(cfg)
    assert not bm.draft_window.enabled
    seq = seq_factory(list(range(20)))
    _admit(bm, seq)
    assert seq.draft_block_table == []


def test_a_prompt_holds_the_window_its_first_block_pass_reads(seq_factory):
    """20 tokens: the pass anchored at 19 reads from key 12 (row 20 starts at
    13, on the page holding 12), so blocks 3 and 4 are held and 0-2 never are.
    """
    bm = _bm()
    seq = seq_factory(list(range(20)))
    _admit(bm, seq)
    assert seq.draft_block_table[:3] == [-1, -1, -1]
    assert seq.draft_window_start == 3
    held = seq.draft_block_table[3:]
    assert len(held) == len(seq.block_table) - 3 and min(held) >= 0
    assert bm.draft_window.pool.num_used == len(held)


def test_decode_hands_back_what_the_window_passed(seq_factory):
    bm = _bm()
    seq = seq_factory(list(range(20)))
    _prefill(bm, seq)
    before = list(seq.draft_block_table)
    # Generate 8 tokens, one decode step each: the next pass anchors at 27 or
    # later and reads from key 20, so blocks 3 and 4 are dead.
    for t in range(8):
        seq.append_token(100 + t)
        bm.may_append(seq)
        seq.num_finalized_tokens = len(seq)
    bm.may_append(seq)
    assert seq.draft_window_start == 5
    assert seq.draft_block_table[:5] == [-1] * 5
    assert seq.draft_block_table[5:] and min(seq.draft_block_table[5:]) >= 0
    assert len(seq.draft_block_table) == len(seq.block_table)
    assert not any(bm.draft_window.pool.is_used(b) for b in before[3:5])


def test_placeholders_do_not_slide_the_window(seq_factory):
    """Tokens appended for a verify are not finalized until it reports; the
    window stays where the finalized length puts it."""
    bm = _bm()
    seq = seq_factory(list(range(20)))
    _prefill(bm, seq)
    for t in range(8):
        seq.append_token(100 + t)
    bm.may_append(seq)  # num_finalized_tokens still 20
    assert seq.draft_window_start == 3


def test_next_turn_reuses_the_window_the_last_one_left(seq_factory):
    bm = _bm()
    first = seq_factory(list(range(20)))
    _prefill(bm, first)
    window = first.draft_block_table[4]
    bm.deallocate(first)
    # The follow-up turn: the whole first prompt plus four new tokens. Its
    # pass anchors at 23 and reads from key 16 -- block 4, which the first
    # turn wrote and published.
    second = seq_factory(list(range(20)) + [90, 91, 92, 93])
    hit = _admit(bm, second)
    assert hit == 5
    assert second.num_cached_tokens == 20
    assert second.draft_window_start == 4
    assert second.draft_block_table[4] == window


def test_a_hit_ends_where_the_window_rows_are_gone(seq_factory):
    """The target still holds all five blocks, the draft none: the prefill
    resumes at the window's first block, one window below the prompt's end,
    rather than reading draft rows nobody holds."""
    bm = _bm()
    first = seq_factory(list(range(20)))
    _prefill(bm, first)
    bm.deallocate(first)
    bm.draft_window.clear_index()
    second = seq_factory(list(range(20)) + [90, 91, 92, 93])
    hit = _admit(bm, second)
    assert hit == 4
    assert second.num_cached_tokens == 16
    assert second.num_compressed_hit_blocks == 5
    # Nothing below the resume point is claimed from the draft, and the four
    # target blocks it kept are the cached ones.
    assert second.draft_block_table[:4] == [-1] * 4
    assert all(bm.kv.block(b).hash != -1 for b in second.block_table[:4])


def test_a_window_above_the_hit_needs_nothing_from_the_cache(seq_factory):
    """A long new suffix puts the whole window above the hit: the target keeps
    its full hit whatever the draft's cache holds."""
    bm = _bm()
    first = seq_factory(list(range(20)))
    _prefill(bm, first)
    bm.deallocate(first)
    bm.draft_window.clear_index()
    second = seq_factory(list(range(20)) + list(range(90, 102)))
    assert _admit(bm, second) == 5
    assert second.draft_window_start == 6


def test_admission_waits_for_the_window_blocks(seq_factory):
    bm = _bm(draft_blocks=3)
    first = seq_factory(list(range(20)))  # holds blocks 3-4 of 5
    _admit(bm, first)
    second = seq_factory(list(range(100, 120)))
    assert bm.can_allocate(second) < 0
    bm.deallocate(first)
    assert bm.can_allocate(second) >= 0


def test_deallocate_returns_every_draft_block(seq_factory):
    bm = _bm()
    seq = seq_factory(list(range(20)))
    _admit(bm, seq)
    bm.deallocate(seq)
    assert seq.draft_block_table == [] and seq.draft_window_start == 0
    assert bm.draft_window.pool.num_used == 0


def test_an_offload_load_resumes_where_the_window_starts(seq_factory):
    """A tier holds no draft rows. Loaded KV past the window's first block is
    recomputed; below it, it is kept."""
    bm = _bm()
    seq = seq_factory(list(range(40)))  # window from key 32 -> block 8
    _admit(bm, seq)
    assert seq.num_cached_tokens == 0
    assert bm.draft_window_resume(seq, 36) == 32
    assert bm.draft_window_resume(seq, 24) == 24


def test_without_a_window_an_offload_load_resumes_where_it_ended(seq_factory):
    cfg = MockConfig(num_kvcache_blocks=16, kv_cache_block_size=4)
    bm = BlockManager(cfg)
    seq = seq_factory(list(range(40)))
    assert bm.draft_window_resume(seq, 36) == 36
