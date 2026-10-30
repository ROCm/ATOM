# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A DSpark draft's KV kept only as far back as its sliding window reads.

The window is where three components meet without sharing code: the block
pass's own read arithmetic (`_window_rows` in the MiniMax-M3 draft), the
scheduler's pool that frees what the window has passed, and the runner's
staging that addresses the pool. Each has to agree with the others to the row,
and a disagreement does not fault: the draft reads another request's rows and
only its acceptance falls. So the geometry is checked against the read
arithmetic row by row, and the pool and the staging against the geometry.

Runs on CPU, without aiter.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.draft_window import (
    DRAFT_WINDOW_CLASS,
    DraftWindowGeometry,
    DraftWindowPool,
)
from atom.model_ops.attentions.pool_layout.sub_pool_spec import Pool
from atom.spec_decode.draft_kv import (
    DraftKvBuilder,
    draft_context_slots,
    refuse_unsupported_draft_window,
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


def test_geometry_survives_the_trip_to_the_engine():
    assert DraftWindowGeometry.from_wire(M31.to_wire()) == M31


def test_pages_must_tile_the_blocks():
    with pytest.raises(ValueError, match="do not tile"):
        DraftWindowGeometry(window=64, read_page=48, block_size=128, spec_tokens=1)


def test_a_pool_without_its_geometry_is_refused():
    with pytest.raises(ValueError, match="both a pool and its geometry"):
        DraftWindowPool(8, None)


# ── block manager ──────────────────────────────────────────────────────────


def _bm(*, blocks=64, draft_blocks=32, prefix=True, geo=SMALL):
    cfg = MockConfig(
        num_kvcache_blocks=blocks,
        kv_cache_block_size=geo.block_size,
        enable_prefix_caching=prefix,
        pool_entries={DRAFT_WINDOW_CLASS: draft_blocks},
        draft_window_geometry=geo.to_wire(),
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


def test_a_received_prompt_publishes_its_window(seq_factory):
    """A P/D consumer runs no prefill: the window the producer wrote is
    published with the KV, so the next turn on this node can reuse it."""
    bm = _bm()
    seq = seq_factory(list(range(20)))
    _admit(bm, seq)
    held = {i: b for i, b in enumerate(seq.draft_block_table) if b >= 0}
    bm.register_received_prefix(seq)
    for i, block_id in held.items():
        if i < 5:
            assert bm.draft_window.pool.block(block_id).hash != -1


# ── runner staging ─────────────────────────────────────────────────────────


def test_context_rows_below_the_window_are_dropped():
    """A prefill chunk of 12 rows from position 8, one request holding blocks
    3-5 (ids 7, 9, 2): the rows at 8-11 are below the window."""
    positions = np.arange(8, 20)
    seqlens = np.array([12])
    tables = np.full((1, 4), -1, dtype=np.int32)
    tables[0, :3] = [7, 9, 2]
    rows = np.empty(12, dtype=np.int64)
    slots = np.empty(12, dtype=np.int64)
    n = draft_context_slots(
        positions,
        seqlens,
        np.array([3]),
        np.array([6]),
        tables,
        4,
        rows_out=rows,
        slots_out=slots,
    )
    assert n == 8
    assert rows[:n].tolist() == list(range(4, 12))
    want = [7 * 4 + p % 4 for p in (12, 13, 14, 15)]
    want += [9 * 4 + p % 4 for p in (16, 17, 18, 19)]
    assert slots[:n].tolist() == want


def _staging_builder(geo=SMALL, max_bs=4, tokens=32):
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(max_num_seqs=max_bs, max_num_batched_tokens=tokens),
        forward_vars={},
    )
    builder = DraftKvBuilder(runner, make_pool=None, window=geo)
    runner.forward_vars.update(builder.step_buffers())
    return builder, runner.forward_vars


def test_a_forward_stages_each_request_run_and_pads_the_rest():
    builder, var = _staging_builder()
    # Two decoding requests, one verify row each: runs from blocks 3 and 0.
    batch = SimpleNamespace(
        draft_windows=(
            np.array([3, 0], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
            np.array([7, 9, 4, 5, 6], dtype=np.int32),
        )
    )
    md = SimpleNamespace()
    builder.prepare_step(
        batch,
        md,
        positions=np.array([19, 10]),
        seqlens=np.array([1, 1]),
        running_bs=3,
        decode=True,
    )
    assert var["draft_window_tables"].gpu[:3, :3].tolist() == [
        [7, 9, 0],
        [4, 5, 6],
        [0, 0, 0],
    ]
    assert var["draft_window_first"].gpu[:3].tolist() == [3, 0, 0]
    assert var["draft_window_extent"].np[:3].tolist() == [5, 3, 1]
    rows, n = builder.kept_context(md, 2)
    assert n == 2 and rows is None
    assert md.draft_slot_mapping.tolist() == [9 * 4 + 3, 6 * 4 + 2]
    # Padded to more rows than it kept, the forward gathers the kept ones.
    rows, n = builder.kept_context(md, 4)
    assert n == 2 and rows.tolist() == [0, 1]


def test_a_forward_that_staged_nothing_writes_nothing():
    """The kept rows live on the metadata they were staged for: a forward
    whose metadata carries none is refused, not handed the last one's."""
    builder, _ = _staging_builder()
    with pytest.raises(RuntimeError, match="staged no draft window"):
        builder.kept_context(SimpleNamespace(), 1)


def test_a_dummy_forward_keeps_no_context_rows():
    builder, var = _staging_builder()
    md = SimpleNamespace()
    builder.prepare_step(
        SimpleNamespace(draft_windows=None),
        md,
        positions=np.zeros(0, dtype=np.int64),
        seqlens=np.zeros(0, dtype=np.int64),
        running_bs=2,
        decode=True,
    )
    assert builder.kept_context(md, 2)[1] == 0
    assert var["draft_window_extent"].np[:2].tolist() == [1, 1]


def test_a_decode_that_would_read_below_its_run_is_refused():
    builder, _ = _staging_builder()
    # The verify row at 19 anchors at 19 or later and reads from key 12,
    # block 3; a run from block 4 has nothing there.
    batch = SimpleNamespace(
        draft_windows=(
            np.array([4], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([7], dtype=np.int32),
        )
    )
    with pytest.raises(RuntimeError, match="reads block 3"):
        builder.prepare_step(
            batch,
            SimpleNamespace(),
            positions=np.array([19]),
            seqlens=np.array([1]),
            running_bs=1,
            decode=True,
        )


def test_a_run_longer_than_the_table_is_refused():
    builder, var = _staging_builder()
    width = var["draft_window_tables"].np.shape[1]
    batch = SimpleNamespace(
        draft_windows=(
            np.array([0], dtype=np.int32),
            np.array([width + 1], dtype=np.int32),
            np.arange(width + 1, dtype=np.int32),
        )
    )
    with pytest.raises(RuntimeError, match="more than the"):
        builder.prepare_step(
            batch,
            SimpleNamespace(),
            positions=np.array([0]),
            seqlens=np.array([1]),
            running_bs=1,
            decode=False,
        )


def test_the_windowed_pool_is_a_state_class_sized_to_the_running_windows():
    runner = SimpleNamespace(
        config=SimpleNamespace(max_num_seqs=8),
        attn_metadata_builder=SimpleNamespace(block_size=4),
    )
    builder = DraftKvBuilder(runner, make_pool=None, window=SMALL)
    builder._kv_pool = SimpleNamespace(entry_bytes=1000)
    (spec,) = builder.sub_pool_specs()
    assert spec.pool is Pool.STATE and spec.name == DRAFT_WINDOW_CLASS
    assert spec.entries_per_req == SMALL.blocks_per_req
    # One cached window per sequence slot by default.
    assert spec.extra_entries == 8 * SMALL.blocks_per_req
    assert builder.paged_pool_bytes(1000) == 0


@pytest.mark.parametrize(
    "overrides, why",
    [
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"enable_rapidserve": True}, "RapidServe"),
        ({"kv_transfer_config": {"kv_connector": "moriio"}}, "MoRI-IO"),
        (
            {
                "kv_transfer_config": {
                    "kv_connector": "multi",
                    "connectors": [
                        {
                            "kv_connector": "mooncake",
                            "kv_role": "kv_producer",
                            "enable_chunked_transfer": True,
                        }
                    ],
                }
            },
            "chunked",
        ),
    ],
)
def test_deployments_without_a_window_path_are_refused(overrides, why):
    config = SimpleNamespace(
        pipeline_parallel_size=1, enable_rapidserve=False, kv_transfer_config={}
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    with pytest.raises(ValueError, match=why):
        refuse_unsupported_draft_window(config)


def test_a_plain_mooncake_pair_is_accepted():
    config = SimpleNamespace(
        pipeline_parallel_size=1,
        enable_rapidserve=False,
        kv_transfer_config={"kv_connector": "mooncake", "kv_role": "kv_consumer"},
    )
    refuse_unsupported_draft_window(config)
