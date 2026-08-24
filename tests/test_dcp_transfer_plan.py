# SPDX-License-Identifier: MIT
"""CPP prefill -> DCP decode KV relayout: the block/token transfer plan.

The mooncake connector's DCP planner decides which producer bytes land in
which consumer slot when a non-DCP prefill node pushes KV to a DCP
decode node. A wrong mapping does not crash and does not produce NaN -- decode
simply attends to the wrong tokens, which only shows up as garbage on long
contexts. So the checks here go past "it runs":

  * the expected destination slots come from the WRITE side -- a transcription
    of ``aiter_mla._dcp_round_robin_slot`` (KV) and of the plain sequential row
    an unsharded indexer writes (index) -- not from a second copy of the
    planner's own arithmetic, so a wrong formula cannot agree with itself;
  * across all DCP ranks the delivered source tokens must form a PARTITION of
    the sequence: every global token delivered exactly once, none twice;
  * run coalescing must preserve the token pairing exactly, since it is what
    keeps the RDMA batch small and is the easiest place to lose a token.

The simulation copies token *ids* rather than bytes, so an off-by-one lands on
a different id instead of on plausible-looking data.
"""

import numpy as np
import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
        build_token_runs,
        plan_replicated,
        plan_sharded,
    )

# (block_size, dcp_size, interleave_size)
LAYOUTS = [
    (16, 4, 16),  # block-level interleave: the production path
    (16, 4, 1),  # token-level interleave: sglang-compatible, correctness only
    (16, 4, 4),  # intermediate S, exercises the general run form
    (16, 2, 16),
    (16, 8, 1),
    (4, 4, 2),
    (8, 1, 1),  # degenerate: DCP off, must reduce to a straight block copy
]
LENGTHS = [1, 15, 16, 17, 63, 64, 65, 200, 1024]


def _cdiv(a, b):
    return -(-a // b)


def _writer_slot_kv(block_table, pos, block_size, dcp_size, dcp_rank, interleave):
    """Slot rank ``dcp_rank`` writes global token ``pos`` into, or -1.

    Transcribed from ``AiterMLAMetadataBuilder._dcp_round_robin_slot``.
    """
    if (pos // interleave) % dcp_size != dcp_rank:
        return -1
    local = (pos // (interleave * dcp_size)) * interleave + (pos % interleave)
    return block_table[pos // (block_size * dcp_size)] * block_size + (
        local % block_size
    )


def _writer_row_index(block_table, pos, block_size, dcp_size):
    """Row an unsharded (DCP-replicated) index cache writes global token ``pos``
    into: virtual blocks hold ``block_size * dcp_size`` tokens in plain order."""
    wide = block_size * dcp_size
    return block_table[pos // wide] * wide + (pos % wide)


def _block_ids(count, seed, spacing=3):
    """Deliberately non-consecutive physical block ids, so any test that passes
    only because ``id == ordinal`` fails here."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(count * spacing)[:count] + 1
    return [int(x) for x in ids]


def _fill_source(src_ids, block_size, num_src_blocks):
    """Flat producer region holding the global token id in every slot."""
    flat = np.full((max(src_ids) + 1) * block_size, -1, dtype=np.int64)
    for ordinal in range(num_src_blocks):
        base = src_ids[ordinal] * block_size
        flat[base : base + block_size] = np.arange(
            ordinal * block_size, (ordinal + 1) * block_size
        )
    return flat


def _apply(runs, src_flat, dst_flat):
    src_off, dst_off, length = runs
    for s, d, n in zip(src_off, dst_off, length):
        dst_flat[d : d + n] = src_flat[s : s + n]


def _expand(runs):
    """Runs -> the exact set of (source token, destination token) pairs."""
    src_off, dst_off, length = runs
    pairs = set()
    for s, d, n in zip(src_off, dst_off, length):
        for k in range(int(n)):
            pairs.add((int(s) + k, int(d) + k))
    return pairs


def _setup(num_tokens, block_size, dcp_size, seed=0):
    n_src = _cdiv(num_tokens, block_size)
    n_dst = _cdiv(num_tokens, block_size * dcp_size)
    src_ids = _block_ids(n_src, seed)
    dst_ids = _block_ids(n_dst, seed + 977)
    return n_src, n_dst, src_ids, dst_ids


# ── KV (sharded) ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("block_size,dcp_size,interleave", LAYOUTS)
@pytest.mark.parametrize("num_tokens", LENGTHS)
def test_sharded_plan_lands_on_the_writer_slots(
    block_size, dcp_size, interleave, num_tokens
):
    if interleave > block_size:
        pytest.skip("interleave must divide the block")
    n_src, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    src_flat = _fill_source(src_ids, block_size, n_src)

    for rank in range(dcp_size):
        plan = plan_sharded(
            n_dst, n_src, block_size, dcp_size, rank, interleave_size=interleave
        )
        dst_flat = np.full((max(dst_ids) + 1) * block_size, -1, dtype=np.int64)
        _apply(build_token_runs(plan, src_ids, dst_ids), src_flat, dst_flat)

        for pos in range(num_tokens):
            slot = _writer_slot_kv(dst_ids, pos, block_size, dcp_size, rank, interleave)
            if slot < 0:
                continue
            assert dst_flat[slot] == pos, (
                f"rank {rank} slot {slot} holds token {dst_flat[slot]}, "
                f"expected {pos} (bs={block_size}, W={dcp_size}, S={interleave})"
            )


@pytest.mark.parametrize("block_size,dcp_size,interleave", LAYOUTS)
@pytest.mark.parametrize("num_tokens", LENGTHS)
def test_sharded_plan_partitions_the_sequence(
    block_size, dcp_size, interleave, num_tokens
):
    """Every global token delivered exactly once across the whole DCP group.

    Tokens past the end of the sequence ride along inside the last block (the
    transfer is block-granular on the source side), so only real tokens are
    counted.
    """
    n_src, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    src_flat = _fill_source(src_ids, block_size, n_src)

    delivered = np.zeros(num_tokens, dtype=np.int64)
    for rank in range(dcp_size):
        plan = plan_sharded(
            n_dst, n_src, block_size, dcp_size, rank, interleave_size=interleave
        )
        for s, _, n in zip(*build_token_runs(plan, src_ids, dst_ids)):
            ids = src_flat[s : s + n]
            ids = ids[ids < num_tokens]
            np.add.at(delivered, ids, 1)

    assert np.array_equal(delivered, np.ones(num_tokens, dtype=np.int64))


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_block_interleave_is_one_whole_block_per_descriptor(dcp_size):
    """S == block_size is the production setting; it must not degenerate into
    per-token descriptors, which is the whole reason it is preferred."""
    block_size, num_tokens = 16, 4096
    n_src, n_dst, _, _ = _setup(num_tokens, block_size, dcp_size)
    for rank in range(dcp_size):
        plan = plan_sharded(
            n_dst, n_src, block_size, dcp_size, rank, interleave_size=block_size
        )
        assert len(plan) == n_dst
        assert np.all(plan.num_tokens == block_size)
        assert np.all(plan.src_token == 0)
        assert np.all(plan.dst_token == 0)
        # dst block b <- src block b*W + r
        assert np.array_equal(plan.src_block, np.arange(n_dst) * dcp_size + rank)


def test_dcp_size_one_is_a_straight_block_copy():
    block_size, num_tokens = 16, 500
    n_src, n_dst, _, _ = _setup(num_tokens, block_size, 1)
    plan = plan_sharded(n_dst, n_src, block_size, 1, 0, interleave_size=block_size)
    assert n_dst == n_src
    assert np.array_equal(plan.src_block, plan.dst_block)


def test_tail_virtual_block_without_a_source_is_dropped():
    """The block manager sizes every rank's table from rank 0's share, so a
    higher rank can own a trailing virtual block with no source tokens."""
    block_size, dcp_size, interleave = 16, 4, 16
    num_tokens = 16 * 4 + 1  # 5 source blocks -> rank 3's block 1 has no source
    n_src, n_dst, _, _ = _setup(num_tokens, block_size, dcp_size)
    assert (n_src, n_dst) == (5, 2)

    counts = [
        len(plan_sharded(n_dst, n_src, block_size, dcp_size, r, interleave))
        for r in range(dcp_size)
    ]
    assert counts == [2, 1, 1, 1]


# ── index (replicated) ────────────────────────────────────────────────────


@pytest.mark.parametrize("block_size,dcp_size,interleave", LAYOUTS)
@pytest.mark.parametrize("num_tokens", LENGTHS)
def test_replicated_plan_lands_on_the_index_rows(
    block_size, dcp_size, interleave, num_tokens
):
    """An unsharded index cache: same content on every rank, plain order."""
    n_src, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    src_flat = _fill_source(src_ids, block_size, n_src)
    wide = block_size * dcp_size

    plan = plan_replicated(n_dst, n_src, block_size, dcp_size)
    assert plan.dst_block_tokens == wide

    dst_flat = np.full((max(dst_ids) + 1) * wide, -1, dtype=np.int64)
    _apply(build_token_runs(plan, src_ids, dst_ids), src_flat, dst_flat)
    for pos in range(num_tokens):
        row = _writer_row_index(dst_ids, pos, block_size, dcp_size)
        assert dst_flat[row] == pos


def test_replicated_plan_is_rank_and_interleave_independent():
    """It takes neither, by construction -- pinned so a future 'optimisation'
    that shards it has to fail a test rather than silently halve the index."""
    plan_a = plan_replicated(8, 32, 16, 4)
    plan_b = plan_replicated(8, 32, 16, 4)
    assert np.array_equal(plan_a.src_block, plan_b.src_block)
    assert len(plan_a) == 8 * 4


# ── run coalescing ────────────────────────────────────────────────────────


@pytest.mark.parametrize("block_size,dcp_size,interleave", LAYOUTS)
def test_coalescing_preserves_every_token_pair(block_size, dcp_size, interleave):
    num_tokens = 1024
    n_src, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    for rank in range(dcp_size):
        plan = plan_sharded(n_dst, n_src, block_size, dcp_size, rank, interleave)
        merged = build_token_runs(plan, src_ids, dst_ids)

        src_lin = np.asarray(src_ids)[plan.src_block] * block_size + plan.src_token
        dst_lin = (
            np.asarray(dst_ids)[plan.dst_block] * plan.dst_block_tokens + plan.dst_token
        )
        unmerged = (src_lin, dst_lin, plan.num_tokens)

        assert _expand(merged) == _expand(unmerged)
        assert merged[2].sum() == plan.num_tokens.sum()


def test_coalescing_collapses_consecutively_allocated_blocks():
    """The payoff case: a virtual block's W replicated sources become one
    descriptor whenever the allocator handed out consecutive ids."""
    block_size, dcp_size, n_dst = 16, 4, 8
    n_src = n_dst * dcp_size
    src_ids = list(range(100, 100 + n_src))
    dst_ids = list(range(7, 7 + n_dst))

    plan = plan_replicated(n_dst, n_src, block_size, dcp_size)
    assert len(plan) == n_dst * dcp_size
    src_off, _, length = build_token_runs(plan, src_ids, dst_ids)
    # Consecutive on both sides throughout -> a single descriptor.
    assert len(src_off) == 1
    assert length[0] == n_src * block_size


def test_coalescing_does_not_merge_across_a_block_id_gap():
    block_size, dcp_size, n_dst = 16, 4, 2
    n_src = n_dst * dcp_size
    src_ids = [0, 1, 2, 3, 50, 51, 52, 53]
    dst_ids = [0, 9]
    plan = plan_replicated(n_dst, n_src, block_size, dcp_size)
    src_off, _, length = build_token_runs(plan, src_ids, dst_ids)
    assert len(src_off) == 2
    assert list(length) == [4 * block_size, 4 * block_size]


# ── PD incremental slicing ────────────────────────────────────────────────


@pytest.mark.parametrize("block_size,dcp_size,interleave", LAYOUTS)
@pytest.mark.parametrize("skip_virtual_blocks", [1, 2])
def test_incremental_transfer_is_the_tail_of_the_full_transfer(
    block_size, dcp_size, interleave, skip_virtual_blocks
):
    """The consumer skips ``off`` virtual blocks it already has cached and the
    producer must skip ``off * W`` source blocks -- not ``off``. Slicing both
    lists that way has to reproduce exactly the tail of the full plan.
    """
    num_tokens = 40 * block_size * dcp_size
    n_src, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    off = skip_virtual_blocks
    if off >= n_dst:
        pytest.skip("sequence too short to skip that many virtual blocks")

    for rank in range(dcp_size):
        full = plan_sharded(n_dst, n_src, block_size, dcp_size, rank, interleave)
        full_pairs = {
            p
            for p, b in zip(_ordered_pairs(full, src_ids, dst_ids), full.dst_block)
            if b >= off
        }

        sliced_src = src_ids[off * dcp_size :]
        sliced_dst = dst_ids[off:]
        part = plan_sharded(
            len(sliced_dst),
            len(sliced_src),
            block_size,
            dcp_size,
            rank,
            interleave,
        )
        part_pairs = set(_ordered_pairs(part, sliced_src, sliced_dst))
        assert part_pairs == full_pairs


def _ordered_pairs(plan, src_ids, dst_ids):
    """One (src token, dst token) pair per run start, in flat token space."""
    src = np.asarray(src_ids)[plan.src_block] * plan.block_size + plan.src_token
    dst = np.asarray(dst_ids)[plan.dst_block] * plan.dst_block_tokens + plan.dst_token
    return [(int(s), int(d)) for s, d in zip(src, dst)]


# ── rejected inputs ───────────────────────────────────────────────────────


def test_planner_rejects_impossible_layouts():
    with pytest.raises(ValueError, match="interleave_size"):
        plan_sharded(4, 16, 16, 4, 0, interleave_size=5)
    with pytest.raises(ValueError, match="dcp_rank"):
        plan_sharded(4, 16, 16, 4, 4, interleave_size=1)
    with pytest.raises(IndexError):
        plan = plan_sharded(4, 16, 16, 4, 0, interleave_size=16)
        build_token_runs(plan, [0, 1], [0, 1, 2, 3])


def test_empty_plans_are_empty_not_broken():
    plan = plan_sharded(0, 0, 16, 4, 2, interleave_size=16)
    assert len(plan) == 0
    src, dst, length = build_token_runs(plan, [], [])
    assert src.size == dst.size == length.size == 0


# ── cross-check against the engine's own DCP math (GPU image only) ────────


@pytest.mark.parametrize("block_size,dcp_size,interleave", LAYOUTS)
def test_agrees_with_dcp_ops(block_size, dcp_size, interleave):
    """Same mapping the attention backend uses, from the other direction."""
    pytest.importorskip("triton")
    pytest.importorskip("aiter")
    from atom.model_ops.dcp_ops import dcp_local_index, dcp_owner_rank

    n_src, n_dst = 64, _cdiv(64, dcp_size)
    for rank in range(dcp_size):
        plan = plan_sharded(n_dst, n_src, block_size, dcp_size, rank, interleave)
        for sb, so, db, dt, n in zip(
            plan.src_block,
            plan.src_token,
            plan.dst_block,
            plan.dst_token,
            plan.num_tokens,
        ):
            for k in range(int(n)):
                g = int(sb) * block_size + int(so) + k
                assert dcp_owner_rank(g, dcp_size, interleave) == rank
                local = dcp_local_index(g, dcp_size, interleave)
                assert local == int(db) * block_size + int(dt) + k
