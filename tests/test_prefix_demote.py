# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_PREFIX_DEMOTE_DECODE: a finished request's generated blocks go to
the evict-first end of the free list.

What has to hold with the flag on, besides the ordering itself: a prefix hit
still names the very blocks that hold the content (demoted ones included), a
block some live sequence holds is never handed out, and the tail-first
(reversed) free keeps reference counts exact when requests share a prefix.
"""

import array

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.sequence import SequenceStatus
from atom.utils import envs

BS = 4


def make_bm(num_blocks: int, demote: bool) -> BlockManager:
    bm = BlockManager(
        MockConfig(
            num_kvcache_blocks=num_blocks,
            kv_cache_block_size=BS,
            enable_prefix_caching=True,
        )
    )
    bm.demote_generated = demote
    return bm


def start(bm: BlockManager, seq_factory, prompt):
    """Admit `prompt` with whatever prefix hit the pool offers, prefill it."""
    s = seq_factory(list(prompt))
    n = bm.can_allocate(s)
    assert n >= 0
    assert bm.allocate(s, n)
    bm.hash_blocks(s, s.num_tokens - s.num_cached_tokens)
    return s


def decode(bm: BlockManager, s, gen):
    for t in gen:
        s.append_token(t)
        bm.may_append(s)
    bm.hash_decode_blocks(s, len(s), snapshot_step=-1)


def finish(bm: BlockManager, s):
    s.status = SequenceStatus.FINISHED
    table = list(s.block_table)
    bm.deallocate(s)
    return table


def hit_tokens(bm: BlockManager, seq_factory, prompt) -> int:
    probe = seq_factory(list(prompt))
    return bm.can_allocate(probe, record=False) * BS


A_PROMPT, A_GEN = list(range(100, 108)), [150, 151, 152, 153]
B_PROMPT, B_GEN = list(range(200, 208)), [250, 251, 252, 253]


class TestFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("ATOM_PREFIX_DEMOTE_DECODE", raising=False)
        assert envs.ATOM_PREFIX_DEMOTE_DECODE is False
        bm = BlockManager(MockConfig(enable_prefix_caching=True))
        assert bm.demote_generated is False

    def test_env_turns_it_on(self, monkeypatch):
        monkeypatch.setenv("ATOM_PREFIX_DEMOTE_DECODE", "1")
        assert BlockManager(MockConfig(enable_prefix_caching=True)).demote_generated


class TestPoolOrder:
    def _pool(self):
        pool = BlockPool(num_blocks=6)
        for h, demote in ((10, False), (11, True), (12, False), (13, True)):
            b = pool.pop()
            pool.allocate(b)
            pool.publish(b, h, array.array("i", [h]))
            pool.free(b, demote)
        return pool

    def test_vacant_then_demoted_then_cached(self):
        pool = self._pool()
        order = []
        while pool.num_free:
            b = pool.pop()
            order.append(pool.block(b).hash)
            pool.allocate(b)
        # two vacant, then the demoted in demotion order, then LRU
        assert order == [-1, -1, 11, 13, 10, 12]
        assert pool.blocks_evicted == 4

    def test_claimed_demoted_block_leaves_the_queue(self):
        pool = self._pool()
        b = pool.lookup(11)
        pool.claim(b)
        assert b not in pool._demoted and pool.is_used(b)
        pool.free(b)  # a plain release files it as ordinary LRU content
        assert b in pool._cached and b not in pool._demoted
        assert pool.num_reusable_free == 4

    def test_demote_only_acts_at_ref_zero(self):
        pool = BlockPool(num_blocks=2)
        b = pool.pop()
        pool.allocate(b)
        pool.publish(b, 7, array.array("i", [7]))
        pool.claim(b)  # second holder
        pool.free(b, True)
        assert pool.is_used(b) and b not in pool._demoted
        assert pool.block(b).ref_count == 1
        pool.free(b)
        assert b in pool._cached and pool.block(b).ref_count == 0

    def test_clear_index_empties_the_demoted_half(self):
        pool = self._pool()
        pool.clear_index()
        assert not pool._demoted and pool.num_reusable_free == 0
        assert pool.num_free == 6


class TestManager:
    @pytest.mark.parametrize("demote", [False, True])
    def test_same_hash_prefix_hit_returns_the_same_blocks(self, seq_factory, demote):
        bm = make_bm(16, demote)
        a = start(bm, seq_factory, A_PROMPT)
        decode(bm, a, A_GEN)
        table = finish(bm, a)
        assert len(table) == 3
        # prompt again plus something new: the two prompt blocks, same ids
        s = start(bm, seq_factory, A_PROMPT + [9, 9, 9])
        assert s.num_cached_tokens == 8
        assert list(s.block_table[:2]) == table[:2]
        # prompt + the generated text + more: the generated block hits too,
        # by its id, although it sat in the demoted half
        t = start(bm, seq_factory, A_PROMPT + A_GEN + [8, 8])
        assert t.num_cached_tokens == 12
        assert list(t.block_table[:3]) == table
        assert table[2] not in bm.kv._demoted
        assert bm.kv.block(table[0]).ref_count == 2  # s and t share it

    @pytest.mark.parametrize("demote,a_hit", [(False, 4), (True, 8)])
    def test_generated_blocks_are_evicted_before_older_prompts(
        self, seq_factory, demote, a_hit
    ):
        bm = make_bm(8, demote)
        a = start(bm, seq_factory, A_PROMPT)
        decode(bm, a, A_GEN)
        finish(bm, a)  # A finishes first: its prompt is the LRU-oldest
        b = start(bm, seq_factory, B_PROMPT)
        decode(bm, b, B_GEN)
        finish(bm, b)
        assert bm.kv.num_free == 8 and bm.kv.num_reusable_free == 6
        # a 4-block prompt: 2 vacant + 2 evicted
        c = start(bm, seq_factory, list(range(300, 316)))
        assert c.num_cached_tokens == 0
        # LRU spent A's generated block and A's second prompt block; demotion
        # spends the two generated blocks and keeps both prompts whole
        assert hit_tokens(bm, seq_factory, A_PROMPT + [1, 1]) == a_hit
        assert hit_tokens(bm, seq_factory, B_PROMPT + [1, 1]) == 8
        assert bm.kv.num_free == 4

    def test_running_sequence_blocks_are_never_evicted(self, seq_factory):
        bm = make_bm(7, True)
        a = start(bm, seq_factory, A_PROMPT)
        decode(bm, a, A_GEN)  # running, holds 3 hashed blocks
        held = list(a.block_table)
        hashes = [bm.kv.block(x).hash for x in held]
        b = start(bm, seq_factory, B_PROMPT)
        decode(bm, b, B_GEN)
        finish(bm, b)  # 3 free hashed blocks (1 demoted) + 1 vacant
        others = []
        while True:
            s = seq_factory(list(range(400 + 10 * len(others), 404 + 10 * len(others))))
            if bm.can_allocate(s) < 0:
                break
            bm.allocate(s)
            others.append(s)
        assert len(others) == 4 and bm.kv.num_free == 0
        for s in others:
            assert not set(s.block_table) & set(held)
        assert [bm.kv.lookup(h) for h in hashes] == held
        assert all(bm.kv.block(x).ref_count == 1 for x in held)
        assert bm.kv._take_free() == -1  # nothing left that is not held

    def test_reversed_free_keeps_refcounts_with_a_shared_prefix(self, seq_factory):
        bm = make_bm(16, True)
        a = start(bm, seq_factory, A_PROMPT + [1, 2, 3, 4])
        b = start(bm, seq_factory, A_PROMPT + [5, 6, 7, 8])  # shares 2 blocks live
        assert b.num_cached_tokens == 8
        shared = list(a.block_table[:2])
        assert list(b.block_table[:2]) == shared
        decode(bm, a, [60, 61, 62, 63])
        decode(bm, b, [70, 71, 72, 73])
        ta, tb = finish(bm, a), list(b.block_table)
        # b still holds the shared prefix: one reference each, not free
        for x in shared:
            assert bm.kv.block(x).ref_count == 1 and bm.kv.is_used(x)
            assert x not in bm.kv._cached and x not in bm.kv._demoted
        # a's private prompt block is LRU content, its generated block demoted
        assert ta[2] in bm.kv._cached and ta[3] in bm.kv._demoted
        finish(bm, b)
        assert bm.kv.num_used == 0 and bm.kv.num_free == 16
        assert all(bm.kv.block(x).ref_count == 0 for x in ta + tb)
        # tail first: the shared root is the most recently freed (evicted last)
        cached = list(bm.kv._cached)
        assert cached[-2:] == [shared[1], shared[0]]
        assert list(bm.kv._demoted) == [ta[3], tb[3]]

    def test_preempted_sequence_is_not_demoted(self, seq_factory):
        bm = make_bm(16, True)
        a = start(bm, seq_factory, A_PROMPT)
        decode(bm, a, A_GEN)
        table = list(a.block_table)
        bm.deallocate(a)  # RUNNING: preemption, it will re-prefill
        assert table[2] in bm.kv._cached and not bm.kv._demoted


def test_demotion_boundary_uses_hash_block_size(seq_factory):
    """Under decode context parallelism one block-table entry spans
    `hash_block_size` (= block_size * dcp) tokens, not `block_size`: the first
    generated entry is `num_prompt_tokens // hash_block_size`."""
    bm = make_bm(32, demote=True)
    s = start(bm, seq_factory, A_PROMPT)  # 8 prompt tokens
    decode(bm, s, A_GEN)
    assert len(s.block_table) == 3
    bm.hash_block_size = 2 * BS  # as if dcp = 2: the prompt is entry 0 only
    table = list(s.block_table)
    calls = []
    free = bm.kv.free

    def record(block_id, demote=False):
        calls.append((block_id, demote))
        free(block_id, demote)

    bm.kv.free = record
    s.status = SequenceStatus.FINISHED
    bm.deallocate(s)
    assert calls == [(table[2], True), (table[1], True), (table[0], False)]


def test_pool_without_demote_argument_disables_the_policy(monkeypatch):
    # Connector pools (UMBP) implement free(block_id) only; the policy must be
    # switched off at init, not raise TypeError when a request finishes.
    from atom.model_engine import block_manager as bmod

    class NoDemotePool(BlockPool):
        def free(self, block_id):
            super().free(block_id)

    monkeypatch.setenv("ATOM_PREFIX_DEMOTE_DECODE", "1")
    monkeypatch.setattr(
        bmod,
        "make_block_pool",
        lambda config, num_blocks, **kw: NoDemotePool(
            num_blocks, on_evict=kw["on_evict"]
        ),
    )
    bm = BlockManager(
        MockConfig(
            num_kvcache_blocks=16, kv_cache_block_size=BS, enable_prefix_caching=True
        )
    )
    assert bm.demote_generated is False
