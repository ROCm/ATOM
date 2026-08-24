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
    n_src, _, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    src_flat = _fill_source(src_ids, block_size, n_src)

    for rank in range(dcp_size):
        runs = plan_sharded(
            src_ids, dst_ids, block_size, dcp_size, rank, interleave_size=interleave
        )
        dst_flat = np.full((max(dst_ids) + 1) * block_size, -1, dtype=np.int64)
        _apply(runs, src_flat, dst_flat)

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
    n_src, _, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    src_flat = _fill_source(src_ids, block_size, n_src)

    delivered = np.zeros(num_tokens, dtype=np.int64)
    for rank in range(dcp_size):
        runs = plan_sharded(
            src_ids, dst_ids, block_size, dcp_size, rank, interleave_size=interleave
        )
        for s, _, n in zip(*runs):
            ids = src_flat[s : s + n]
            ids = ids[ids < num_tokens]
            np.add.at(delivered, ids, 1)

    assert np.array_equal(delivered, np.ones(num_tokens, dtype=np.int64))


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_block_interleave_is_one_whole_block_per_descriptor(dcp_size):
    """S == block_size is the production setting; it must not degenerate into
    per-token descriptors, which is the whole reason it is preferred."""
    block_size, num_tokens = 16, 4096
    _, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    src_ids, dst_ids = np.asarray(src_ids), np.asarray(dst_ids)
    for rank in range(dcp_size):
        src, dst, length = plan_sharded(
            src_ids, dst_ids, block_size, dcp_size, rank, interleave_size=block_size
        )
        assert len(src) == n_dst
        assert np.all(length == block_size)
        # dst block b <- the whole of src block b*W + r
        expect = src_ids[np.arange(n_dst) * dcp_size + rank] * block_size
        assert np.array_equal(src, expect)
        assert np.array_equal(dst, dst_ids * block_size)


def test_dcp_size_one_is_a_straight_block_copy():
    block_size, num_tokens = 16, 500
    n_src, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, 1)
    src, dst, length = plan_sharded(
        src_ids, dst_ids, block_size, 1, 0, interleave_size=block_size
    )
    assert n_dst == n_src
    assert np.array_equal(src, np.asarray(src_ids) * block_size)
    assert np.array_equal(dst, np.asarray(dst_ids) * block_size)
    assert np.all(length == block_size)


def test_tail_virtual_block_without_a_source_is_dropped():
    """The block manager sizes every rank's table from rank 0's share, so a
    higher rank can own a trailing virtual block with no source tokens."""
    block_size, dcp_size, interleave = 16, 4, 16
    num_tokens = 16 * 4 + 1  # 5 source blocks -> rank 3's block 1 has no source
    n_src, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    assert (n_src, n_dst) == (5, 2)

    counts = [
        len(plan_sharded(src_ids, dst_ids, block_size, dcp_size, r, interleave)[2])
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
    n_src, _, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    src_flat = _fill_source(src_ids, block_size, n_src)
    wide = block_size * dcp_size

    dst_flat = np.full((max(dst_ids) + 1) * wide, -1, dtype=np.int64)
    _apply(plan_replicated(src_ids, dst_ids, block_size, dcp_size), src_flat, dst_flat)
    for pos in range(num_tokens):
        row = _writer_row_index(dst_ids, pos, block_size, dcp_size)
        assert dst_flat[row] == pos


def test_replicated_plan_is_rank_and_interleave_independent():
    """It takes neither, by construction -- pinned so a future 'optimisation'
    that shards it has to fail a test rather than silently halve the index."""
    block_size, dcp_size, n_dst = 16, 4, 8
    n_src = n_dst * dcp_size
    src_ids, dst_ids = _block_ids(n_src, 3), _block_ids(n_dst, 11)
    src, _, length = plan_replicated(src_ids, dst_ids, block_size, dcp_size)
    # One run per source block, and every source block carried in full: a plan
    # that sharded the index would move 1/W of this.
    assert len(src) == n_src
    assert length.sum() == n_src * block_size


# ── run coalescing ────────────────────────────────────────────────────────


@pytest.mark.parametrize("block_size,dcp_size,interleave", LAYOUTS)
def test_descriptor_count_under_a_consecutive_allocator(
    block_size, dcp_size, interleave
):
    """What the RDMA batch actually costs once the allocator hands out
    consecutive ids, which is the case the descriptor budget is sized for.

    Coalescing cannot help the sharded region above W == 1: rank r owns source
    blocks r, W+r, 2W+r, ... so the source side is strided by W while the
    destination side is dense, and a merge needs both. The sharded cost is
    therefore the closed form ``n_dst * (block_size / S)`` -- dropping S from
    block_size to 1 really does multiply the batch by block_size. The
    replicated region is where merging pays: its W sources per virtual block
    are consecutive, so the whole request collapses to one descriptor.
    """
    num_tokens = 4096
    n_src = _cdiv(num_tokens, block_size)
    n_dst = _cdiv(num_tokens, block_size * dcp_size)
    src_ids = list(range(100, 100 + n_src))
    dst_ids = list(range(7, 7 + n_dst))

    for rank in range(dcp_size):
        runs = plan_sharded(src_ids, dst_ids, block_size, dcp_size, rank, interleave)
        expect = 1 if dcp_size == 1 else n_dst * (block_size // interleave)
        assert len(runs[2]) == expect
        assert runs[2].sum() == n_dst * block_size

    assert len(plan_replicated(src_ids, dst_ids, block_size, dcp_size)[2]) == 1


def test_coalescing_collapses_consecutively_allocated_blocks():
    """The payoff case: a virtual block's W replicated sources become one
    descriptor whenever the allocator handed out consecutive ids."""
    block_size, dcp_size, n_dst = 16, 4, 8
    n_src = n_dst * dcp_size
    src_ids = list(range(100, 100 + n_src))
    dst_ids = list(range(7, 7 + n_dst))

    src_off, _, length = plan_replicated(src_ids, dst_ids, block_size, dcp_size)
    # Consecutive on both sides throughout -> a single descriptor.
    assert len(src_off) == 1
    assert length[0] == n_src * block_size


def test_coalescing_does_not_merge_across_a_block_id_gap():
    block_size, dcp_size = 16, 4
    src_ids = [0, 1, 2, 3, 50, 51, 52, 53]
    dst_ids = [0, 9]
    src_off, _, length = plan_replicated(src_ids, dst_ids, block_size, dcp_size)
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
    _, n_dst, src_ids, dst_ids = _setup(num_tokens, block_size, dcp_size)
    off = skip_virtual_blocks
    if off >= n_dst:
        pytest.skip("sequence too short to skip that many virtual blocks")

    skipped = set(dst_ids[:off])
    for rank in range(dcp_size):
        full = _expand(
            plan_sharded(src_ids, dst_ids, block_size, dcp_size, rank, interleave)
        )
        tail = {(s, d) for s, d in full if d // block_size not in skipped}

        part = _expand(
            plan_sharded(
                src_ids[off * dcp_size :],
                dst_ids[off:],
                block_size,
                dcp_size,
                rank,
                interleave,
            )
        )
        assert part == tail


# ── degenerate inputs ─────────────────────────────────────────────────────


def test_empty_plans_are_empty_not_broken():
    src, dst, length = plan_sharded([], [], 16, 4, 2, interleave_size=16)
    assert src.size == dst.size == length.size == 0


# ── cross-check against the engine's own DCP math (GPU image only) ────────


@pytest.mark.parametrize("block_size,dcp_size,interleave", LAYOUTS)
def test_agrees_with_dcp_ops(block_size, dcp_size, interleave):
    """Same mapping the attention backend uses, from the other direction."""
    pytest.importorskip("triton")
    pytest.importorskip("aiter")
    from atom.model_ops.dcp_ops import dcp_local_index, dcp_owner_rank

    src_ids = list(range(64))
    dst_ids = list(range(_cdiv(64, dcp_size)))
    for rank in range(dcp_size):
        runs = plan_sharded(src_ids, dst_ids, block_size, dcp_size, rank, interleave)
        for g, slot in _expand(runs):
            assert dcp_owner_rank(g, dcp_size, interleave) == rank
            assert dcp_local_index(g, dcp_size, interleave) == slot
