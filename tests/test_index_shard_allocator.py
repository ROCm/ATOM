# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The block allocator under MiniMax-M3 index sharding (`dcp_config.index_shard`).

Rank r keeps the index rows of block id b iff ``b % P == r``, and it scores
logical block j iff ``j % P == r``. Both meet only if every block table holds,
at logical position j, a block id ``== j mod P`` -- for fresh blocks, prefix
hits, evictions under LRU and SLRU, lazy-offload demotion pins, copies a disown
makes and the blocks a P/D consumer receives into. The pool keeps one free lane
per residue (`BlockPool(residues=P)`); admission and append have to count each
lane, since a full lane fails an allocation however much the others hold.
"""

import array
import random

import pytest
from conftest import MockConfig

from atom.config import DCPConfig
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

P = 4
BLOCK = 4  # tokens a block in these tests


def toks(*ids: int) -> array.array:
    return array.array("i", ids)


def published(pool: BlockPool, block_id: int, h: int) -> int:
    pool.allocate(block_id)
    pool.publish(block_id, h, toks(h))
    pool.free(block_id)
    return block_id


def sharded_bm(blocks: int, *, policy: str = "lru", prefix: bool = True):
    cfg = MockConfig(
        num_kvcache_blocks=blocks,
        kv_cache_block_size=BLOCK,
        enable_prefix_caching=prefix,
        max_num_seqs=64,
        dcp_config=DCPConfig(indexer_dcp_only=True, index_shard=True),
        tensor_parallel_size=P,
    )
    bm = BlockManager(cfg)
    if policy != "lru":
        bm.kv = BlockPool(
            blocks,
            on_evict=bm._record_evicted,
            cache_policy=policy,
            on_remap=bm._note_index_change,
            residues=P,
        )
    return bm


def seq(tokens) -> Sequence:
    return Sequence(list(tokens), BLOCK, sampling_params=SamplingParams())


def assert_positions(bm: BlockManager, *seqs: Sequence) -> None:
    for s in seqs:
        wrong = [(j, b) for j, b in enumerate(s.block_table) if b % P != j % P]
        assert not wrong, f"seq {s.id}: (position, block) {wrong[:4]} off their lanes"


def assert_accounting(pool: BlockPool) -> None:
    """The per-lane free counts are the free set, split by lane, and every free
    block of a lane sits in that lane's containers only."""
    by_lane = [0] * pool.residues
    for b in pool._free:
        by_lane[b % pool.residues] += 1
    assert by_lane == pool._free_count
    for lane in range(pool.residues):
        for queue in (pool._cached[lane], pool._protected[lane], pool._vacant[lane]):
            assert all(b % pool.residues == lane for b in queue)


# ── BlockPool lanes ────────────────────────────────────────────────────────


class TestPoolLanes:
    def test_a_lane_hands_out_its_own_ids_until_it_is_empty(self):
        pool = BlockPool(8, residues=P)
        assert [pool.pop(1), pool.pop(1)] == [1, 5]
        assert pool.has_free_per_residue([2, 0, 2, 2])
        assert not pool.has_free_per_residue([0, 1, 0, 0])
        with pytest.raises(AssertionError, match="lane 1"):
            pool.pop(1)
        assert pool.pop(3) == 3

    def test_eviction_takes_the_coldest_block_of_the_lane_asked(self):
        pool = BlockPool(8, residues=P)
        # Freed in this order: 4 then 0 in lane 0, 1 then 5 in lane 1.
        for b, h in ((4, 104), (1, 101), (0, 100), (5, 105)):
            published(pool, b, h)
        for b in (2, 3, 6, 7):  # the vacant rest, taken first
            pool.allocate(pool.pop(b % P))
        assert pool.pop(0) == 4  # the coldest of lane 0, not lane 1's 1
        pool.allocate(4)
        assert pool.lookup(104) == -1 and pool.lookup(101) == 1
        assert pool.pop(1) == 1
        pool.allocate(1)
        assert_accounting(pool)

    def test_the_protected_bound_is_a_lanes_share(self):
        pool = BlockPool(16, cache_policy="slru", protected_ratio=0.5, residues=P)
        lane0 = [0, 4, 8, 12]
        for b in lane0:
            published(pool, b, 1000 + b)
        for b in lane0:  # reused: protected on release
            pool.claim(b)
        for b in lane0:
            pool.free(b)
        # int(16 // 4 * 0.5) = 2 a lane: the two coldest fall back to probation.
        assert list(pool._protected[0]) == [8, 12]
        assert list(pool._cached[0]) == [0, 4]
        assert not any(pool._protected[lane] for lane in (1, 2, 3))

    def test_next_evicted_interleaves_the_lanes_coldest_first(self):
        pool = BlockPool(8, residues=P)
        for b in (0, 4, 1, 5):
            published(pool, b, 200 + b)
        for b in (2, 3, 6, 7):
            pool.allocate(pool.pop(b % P))
        # 4 allocations, one a lane: lane 0 and lane 1 each evict one.
        assert list(pool.next_evicted(4)) == [0, 1]
        assert list(pool.next_evicted(8)) == [0, 1, 4, 5]

    def test_a_demotion_pin_returns_to_the_head_of_its_lane(self):
        pool = BlockPool(8, residues=P)
        for b in (0, 4, 1):
            published(pool, b, 300 + b)
        for b in (2, 3, 5, 6, 7):
            pool.allocate(pool.pop(b % P))
        # One allocation a lane: lane 0's coldest, and lane 1's only.
        assert list(pool.next_evicted(4)) == [0, 1]
        pool.claim_unreused(0)  # pinned while a store copies it out
        assert pool.pop(0) == 4
        pool.allocate(4)
        pool.free(0, coldest=True)
        assert list(pool._cached[0]) == [0] and list(pool._cached[1]) == [1]
        assert_accounting(pool)

    def test_clear_index_redraws_every_lane(self):
        pool = BlockPool(8, residues=P)
        for b in range(8):
            published(pool, b, 400 + b)
        pool.clear_index()
        assert [pool.pop(lane) for lane in range(P)] == [0, 1, 2, 3]
        assert_accounting(pool)

    def test_a_laned_pool_is_fixed(self):
        with pytest.raises(ValueError, match="multiple of 4"):
            BlockPool(10, residues=P)
        with pytest.raises(ValueError, match="fixed pool"):
            BlockPool(8, max_blocks=16, residues=P)
        pool = BlockPool(8, residues=P)
        for grow in (lambda: pool.extend(4), pool.retire_top):
            with pytest.raises(NotImplementedError, match="fixed"):
                grow()
        with pytest.raises(NotImplementedError, match="raw units"):
            pool.reserve_units(1, "owner")


# ── BlockManager: positions decide lanes ──────────────────────────────────


class TestManagerPositions:
    def test_fresh_blocks_take_their_positions_lane(self):
        bm = sharded_bm(32)
        s = seq(range(37))  # 10 blocks
        assert bm.can_allocate(s) == 0
        bm.allocate(s)
        assert len(s.block_table) == 10
        assert_positions(bm, s)
        for _ in range(12):  # decode past two block boundaries
            s.append_token(1)
            assert bm.can_append(s)
            bm.may_append(s)
        assert_positions(bm, s)
        assert_accounting(bm.kv)

    def test_a_prefix_hit_reuses_the_blocks_at_their_positions(self):
        bm = sharded_bm(32)
        first = seq(range(40))
        bm.allocate(first)
        bm.hash_blocks(first, first.num_tokens)
        cached = list(first.block_table[:7])
        bm.deallocate(first)
        second = seq(list(range(28)) + [999] * 9)
        hit = bm.can_allocate(second)
        assert hit == 7
        bm.allocate(second, hit)
        assert list(second.block_table[:7]) == cached
        assert second.num_cached_tokens == 28
        assert_positions(bm, second)

    def test_admission_counts_each_lane(self):
        bm = sharded_bm(8)  # two blocks a lane
        holder = seq(range(4 * BLOCK))  # one block of every lane
        bm.allocate(holder)
        other = seq(range(100, 100 + BLOCK))  # lane 0 only
        bm.allocate(other)
        assert bm.kv.num_free == 3  # lanes 1, 2, 3
        late = seq(range(200, 200 + BLOCK))  # needs lane 0, which is empty
        assert bm.can_allocate(late) == -1
        bm.deallocate(other)
        assert bm.can_allocate(late) == 0

    def test_append_counts_the_lane_of_the_next_position(self):
        bm = sharded_bm(8)
        s = seq(range(4 * BLOCK))  # positions 0..3; the next block is lane 0
        bm.allocate(s)
        blocker = seq(range(100, 100 + BLOCK))
        bm.allocate(blocker)  # the last lane-0 block
        s.append_token(1)
        assert not bm.can_append(s)
        assert bm.kv.num_free == 3
        bm.deallocate(blocker)
        assert bm.can_append(s)
        bm.may_append(s)
        assert_positions(bm, s)

    def test_a_disown_copies_into_the_same_lanes(self):
        bm = sharded_bm(32)
        first = seq(range(24))
        bm.allocate(first)
        bm.hash_blocks(first, first.num_tokens)
        second = seq(list(range(24)) + [7])
        bm.allocate(second, bm.can_allocate(second))
        shared = list(second.block_table[:5])
        assert shared == list(first.block_table[:5])
        assert bm.disown_claimed_prefix(second)
        assert all(a != b for a, b in zip(second.block_table[:5], shared))
        assert_positions(bm, first, second)

    def test_a_received_prefix_is_hit_at_its_positions(self):
        """The P/D consumer allocates a remote prefill through `allocate` and
        registers the received blocks for the next turn."""
        bm = sharded_bm(32)
        turn1 = seq(range(33))
        turn1.kv_transfer_params = {"do_remote_prefill": True}
        bm.allocate(turn1, bm.can_allocate(turn1))
        assert bm.register_received_prefix(turn1) == 8
        bm.deallocate(turn1)
        turn2 = seq(list(range(33)) + list(range(500, 520)))
        hit = bm.can_allocate(turn2)
        assert hit == 8
        bm.allocate(turn2, hit)
        assert_positions(bm, turn2)

    @pytest.mark.parametrize("policy", ["lru", "slru"])
    def test_a_churning_cache_keeps_every_table_on_its_lanes(self, policy):
        """Shared prefixes, evictions, appends and releases in random order."""
        rng = random.Random(7)
        bm = sharded_bm(64, policy=policy)
        prefixes = [list(range(base, base + 24)) for base in (0, 1000, 2000)]
        live: list[Sequence] = []
        admitted = evicted_before = 0
        for step in range(400):
            action = rng.random()
            if action < 0.45:
                tokens = rng.choice(prefixes)[: rng.randint(4, 24)] + [
                    rng.randint(5000, 9000) for _ in range(rng.randint(1, 30))
                ]
                s = seq(tokens)
                hit = bm.can_allocate(s)
                if hit >= 0:
                    bm.allocate(s, hit)
                    bm.hash_blocks(s, s.num_tokens - s.num_cached_tokens)
                    live.append(s)
                    admitted += 1
            elif action < 0.75 and live:
                s = rng.choice(live)
                for _ in range(rng.randint(1, 6)):
                    s.append_token(rng.randint(0, 99))
                    if not bm.can_append(s):
                        break
                    bm.may_append(s)
            elif live:
                s = live.pop(rng.randrange(len(live)))
                bm.deallocate(s)
            assert_positions(bm, *live)
            assert_accounting(bm.kv)
            assert bm.kv.num_free + bm.kv.num_used == bm.kv.num_blocks
        assert admitted > 50
        assert bm.kv.blocks_evicted > evicted_before, "the run never evicted"
