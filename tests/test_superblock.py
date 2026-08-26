# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import array

import pytest

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.superblock import KV, STATE, UNTYPED, SuperblockMap


def pooled(num_blocks=16, per=4, on_evict=None):
    sb = SuperblockMap(num_blocks, per)
    return BlockPool(num_blocks, on_evict=on_evict, superblocks=sb), sb


def fill(pool, block_id, h):
    """Give a block content so it lands in the cached tier when freed."""
    pool.allocate(block_id)
    # `array("i")`, not a list: `Block.update` pins the type, because a list
    # never compares equal to what the real publish paths store, so every hit
    # on the block would read as a hash collision.
    pool.publish(block_id, h, array.array("i", [h]))


class TestMapping:
    def test_super_of_is_arithmetic_not_a_table(self):
        sb = SuperblockMap(16, 4)
        assert [sb.super_of(b) for b in (0, 3, 4, 15)] == [0, 0, 1, 3]
        assert list(sb.block_range(2)) == [8, 9, 10, 11]

    def test_a_pool_must_divide_evenly_into_superblocks(self):
        with pytest.raises(ValueError, match="whole number"):
            SuperblockMap(15, 4)


class TestLiveness:
    def test_a_superblock_is_reclaimable_only_with_no_live_block(self):
        pool, sb = pooled()
        pool.allocate(0)
        assert not sb.is_reclaimable(0)
        pool.free(0)
        assert sb.is_reclaimable(0)

    def test_cached_content_does_not_keep_a_superblock_live(self):
        """The distinction a live *count* captures and an occupancy flag misses.

        A freed-but-hashed block still holds reusable content, so it is not
        vacant — but it is spendable, exactly as `_take_free` spends `_cached`.
        Treating it as live would make superblocks unreclaimable for as long as
        their content stayed interesting, which is precisely backwards.
        """
        pool, sb = pooled()
        fill(pool, 0, h=101)
        pool.free(0)
        assert pool.lookup(101) == 0, "content still reachable by hash"
        assert sb.is_reclaimable(0), "but the superblock is still claimable"

    def test_sharing_keeps_a_superblock_live_until_the_last_reader(self):
        pool, sb = pooled()
        fill(pool, 0, h=101)
        pool.claim(0)  # a second request hits the same block
        pool.free(0)
        assert not sb.is_reclaimable(0), "one reader left"
        pool.free(0)
        assert sb.is_reclaimable(0)

    def test_freeing_below_zero_is_an_assertion_not_a_silent_wrap(self):
        _pool, sb = SuperblockMap(16, 4), None
        sb = _pool
        with pytest.raises(AssertionError, match="below zero"):
            sb.on_block_free(0)


class TestPacking:
    def test_fresh_blocks_pack_into_one_superblock(self):
        """The safeguard against one live block per superblock, pool-wide.

        Scattering would leave every superblock pinned and none reclaimable
        whole, while the pool sat nearly empty — the catastrophic case this
        design exists to avoid.
        """
        pool, sb = pooled(num_blocks=16, per=4)
        got = [pool.pop() for _ in range(4)]
        for b in got:
            pool.allocate(b)
        assert {sb.super_of(b) for b in got} == {0}

    def test_packing_never_promotes_cached_over_vacant(self):
        """Tier order is the policy; packing only chooses within a tier.

        Preferring a cached block in the open superblock over a vacant one
        elsewhere would spend reusable content while empty blocks waited.
        """
        pool, sb = pooled(num_blocks=8, per=4)
        # Superblock 0: one cached block, three live so it stays open.
        fill(pool, 0, h=101)
        pool.free(0)
        for b in (1, 2, 3):
            pool.allocate(b)
        assert sb._open == 0
        # Superblock 1 is entirely vacant. The vacant tier must win.
        assert pool.blocks[0].hash != -1, "block 0 is cached, not vacant"
        taken = pool.pop()
        assert sb.super_of(taken) == 1, "vacant elsewhere beats cached in-super"

    def test_a_cache_hit_does_not_move_the_packing_frontier(self):
        """`claim` lands wherever content already is; following it would scatter."""
        pool, sb = pooled(num_blocks=16, per=4)
        pool.allocate(0)
        assert sb._open == 0
        fill(pool, 8, h=202)
        pool.free(8)
        sb._open = 0  # reset after the fill's own allocate
        pool.claim(8)
        assert sb._open == 0, "the writer's frontier stayed put"


class TestClaimSuperblock:
    def test_an_untyped_superblock_is_preferred_over_a_cached_one(self):
        """Taking an untyped superblock evicts nothing; prefer it."""
        pool, sb = pooled(num_blocks=8, per=4)
        for b in range(4):
            fill(pool, b, h=100 + b)
            pool.free(b)
        assert sb.is_reclaimable(0) and sb.kind[0] == KV
        assert pool.claim_superblock() == 1, "the untyped one, not the cached one"
        assert pool.blocks_evicted == 0

    def test_claiming_spends_cached_content_through_the_normal_path(self):
        """`blocks_evicted` and `on_evict` must fire as for any other eviction.

        A checkpoint filed under one of these hashes has to learn its prefix is
        gone by the route it always did.
        """
        seen = []
        pool, sb = pooled(num_blocks=4, per=4, on_evict=seen.append)
        for b in range(4):
            fill(pool, b, h=100 + b)
            pool.free(b)
        assert pool.claim_superblock() == 0
        assert pool.blocks_evicted == 4
        assert sorted(seen) == [100, 101, 102, 103]
        assert sb.kind[0] == STATE

    def test_a_live_block_makes_a_superblock_unclaimable(self):
        pool, _sb = pooled(num_blocks=4, per=4)
        pool.allocate(0)
        assert pool.can_claim_superblock() is False
        assert pool.claim_superblock() == -1

    def test_no_claimable_superblock_returns_minus_one_not_a_raise(self):
        """A full pool is a scheduling decision, not a crash.

        This is what the first implementation got wrong in the other direction:
        it returned -1 without ever attempting eviction, so it reported
        starvation while spendable superblocks sat there.
        """
        pool, _sb = pooled(num_blocks=4, per=4)
        pool.allocate(0)
        assert pool.claim_superblock() == -1

    def test_releasing_returns_every_block_to_the_vacant_tier(self):
        pool, sb = pooled(num_blocks=4, per=4)
        index = pool.claim_superblock()
        assert index == 0
        pool.release_superblock(index)
        assert sb.kind[0] == UNTYPED
        assert pool.num_free == 4
        assert pool.pop() >= 0

    def test_release_does_not_leave_duplicate_ids_in_the_vacant_heap(self):
        """`pop` must never hand the same id to two callers.

        `claim_superblock` takes its blocks off `_free` but cannot purge
        `_vacant`, which tolerates stale entries by design. Re-adding on
        release without purging leaves each id in the heap twice, and once the
        block is freed again *both* copies pass `_take_free`'s staleness test.
        The second `allocate` then trips its ref-count assertion — which is how
        the trace replay found this, several hundred requests in.
        """
        pool, _sb = pooled(num_blocks=8, per=4)
        index = pool.claim_superblock()
        pool.release_superblock(index)
        assert sorted(pool._vacant) == list(range(8)), "no duplicates"

        first = [pool.pop() for _ in range(8)]
        for b in first:
            pool.allocate(b)
        for b in first:
            pool.free(b)
        second = [pool.pop() for _ in range(8)]
        assert len(set(second)) == len(second), "an id was handed out twice"

    def test_a_state_superblock_is_never_handed_out_as_kv(self):
        pool, sb = pooled(num_blocks=8, per=4)
        pool.claim_superblock()
        assert sb.kind[0] == STATE
        for _ in range(4):
            assert sb.super_of(pool.pop()) == 1
        assert not sb.is_reclaimable(0), "state superblocks are not KV-reclaimable"


class TestStatePoolBacking:
    """`StateSlotPool` with its slots backed by superblocks from the KV pool.

    This is where the carve becomes dynamic: `num_slots` stops being a
    reservation and becomes a ceiling, and capacity a request is not using sits
    in the paged pool instead of idling.
    """

    def backed(self, num_blocks=32, per=4, slots=4):
        from atom.model_engine.state_pool import StateSlotPool
        from atom.model_engine.state_runtime import StateTransfer

        sb = SuperblockMap(num_blocks, per)
        pool = BlockPool(num_blocks, superblocks=sb)
        state = StateSlotPool(
            slots,
            transfer=StateTransfer.fork(1),
            hash_block_size=4,
            superblock_source=pool,
        )
        return state, pool, sb

    def test_a_slot_takes_a_superblock_only_when_handed_out(self):
        state, pool, sb = self.backed()
        assert sb.occupancy()["supers_state"] == 0, "nothing reserved up front"
        slot = state.pop()
        assert sb.occupancy()["supers_state"] == 1
        assert state._slot_super[slot] >= 0

    def test_a_vacant_slot_gives_its_superblock_back(self):
        state, pool, sb = self.backed()
        slot = state.pop()
        state.release(slot)
        assert sb.occupancy()["supers_state"] == 0, "capacity returned to KV"
        assert slot not in state._slot_super

    def test_a_slot_holding_a_checkpoint_keeps_its_superblock(self):
        """The checkpoint *is* those bytes.

        Handing the backing back on release would leave the index pointing at
        memory the paged pool had already reused — findable, and wrong.
        """
        state, pool, sb = self.backed()
        slot = state.pop()
        state._index(101, slot)
        state.release(slot)
        assert sb.occupancy()["supers_state"] == 1
        assert state.lookup(101) == slot

    def test_admission_asks_the_paged_pool_for_the_shortfall(self):
        """`has_free` must agree with what `pop` can actually back."""
        state, pool, sb = self.backed(num_blocks=8, per=4, slots=4)
        first = state.pop()
        second = state.pop()
        assert not state.has_free(), "both superblocks are spoken for"
        state.release(first)
        assert state.has_free()
        assert state.pop() == first or second >= 0

    def test_slots_already_backed_need_no_further_superblock(self):
        state, pool, sb = self.backed(num_blocks=8, per=4, slots=4)
        slot = state.pop()
        state._index(101, slot)
        state.release(slot)  # keeps its backing, being a checkpoint
        # One superblock left free, and the checkpointed slot needs none.
        assert state.has_free(2), "a backed slot plus one claimable superblock"


class TestClaimOrder:
    def test_a_cached_superblock_is_chosen_by_lru_not_by_index(self):
        """Spending the lowest index would ignore how hot its content is.

        `_cached` is the pool's own release order over blocks, so a superblock
        inherits the coldness of its coldest block — the one a plain
        `_take_free` would have spent next anyway. Keeping a second order per
        superblock would let the two drift.
        """
        pool, sb = pooled(num_blocks=16, per=4)
        # Release superblock 2 first, so it is coldest despite not being lowest.
        for index in (2, 0, 1, 3):
            for b in sb.block_range(index):
                fill(pool, b, h=1000 + b)
            for b in sb.block_range(index):
                pool.free(b)
        assert pool.claim_superblock() == 2

    def test_a_used_but_never_checkpointed_superblock_is_free_to_take(self):
        """What a claim costs is decided by hashes, not by typing.

        A request's own working blocks are allocated, written for its whole
        life, and released without ever being published under a hash. They hold
        nothing anyone can resume from, so the superblock is free to take —
        but it is typed KV forever, and keying on `kind == UNTYPED` made every
        one of them invisible. A pool that had been used once and fully drained
        then refused every claim while being entirely free.
        """
        pool, sb = pooled(num_blocks=8, per=4)
        for index in (0, 1):
            for b in sb.block_range(index):
                pool.allocate(b)  # no publish: nothing findable
            for b in sb.block_range(index):
                pool.free(b)
        assert sb.kind == ["kv", "kv"], "typed KV, and it stays that way"
        assert not any(pool.blocks[b].hash != -1 for b in range(8))

        assert pool.claim_superblock() == 0
        assert pool.superblock_claims_refused == 0
        assert pool.superblocks_evicted_cached == 0, "destroyed nothing"

    def test_a_free_superblock_is_preferred_over_a_cached_one(self):
        pool, sb = pooled(num_blocks=8, per=4)
        for b in sb.block_range(0):  # superblock 0 holds reusable content
            fill(pool, b, h=900 + b)
        for b in sb.block_range(0):
            pool.free(b)
        for b in sb.block_range(1):  # superblock 1 holds nothing findable
            pool.allocate(b)
        for b in sb.block_range(1):
            pool.free(b)

        assert pool.claim_superblock() == 1, "the one that costs nothing"
        assert pool.superblocks_evicted_cached == 0
        assert pool.lookup(900) >= 0, "the cached prefix survived"

        assert pool.claim_superblock() == 0, "only now spend the cache"
        assert pool.superblocks_evicted_cached == 1

    def test_a_kv_floor_is_held_back_from_bulk_claims(self):
        """Spending the pool to nothing for slots leaves nothing to prefill with.

        Admission would then refuse for want of blocks it had just given away.
        """
        pool, sb = pooled(num_blocks=40, per=4)  # 10 superblocks
        floor = pool._kv_superblock_floor()
        assert floor >= 1
        assert pool.can_claim_superblocks(sb.num_supers - floor)
        assert not pool.can_claim_superblocks(sb.num_supers - floor + 1)


class TestElasticSlots:
    """`num_slots` is a ceiling from the pool, not a count fixed at startup.

    Without a source it is `max_num_seqs * entries_per_req + extra_entries`,
    computed before the pool existed — which caps checkpoints at roughly the
    concurrency however much room there is. That is the static carve in index
    form, and it is what held Kimi-K3 to 32 slots against a pool with hundreds
    of superblocks free.
    """

    def elastic(self, num_blocks=320, per=32, slots=2, max_slots=8):
        """A pool starting below the state tensor's slot dimension.

        `max_slots` models that tensor, which is what growth is really bounded
        by: `mamba_k_cache[layer, slot]` is allocated once at startup, so an
        index past its slot dimension is an out-of-bounds GPU read. The
        superblock supply governs the bytes; the tensor governs how many slot
        indices may exist at all, and on Kimi-K3 it is the lower of the two.
        """
        from atom.model_engine.state_pool import StateSlotPool
        from atom.model_engine.state_runtime import StateTransfer

        sb = SuperblockMap(num_blocks, per)
        pool = BlockPool(num_blocks, superblocks=sb)
        state = StateSlotPool(
            slots,
            transfer=StateTransfer.fork(1),
            hash_block_size=4,
            superblock_source=pool,
            max_slots=max_slots,
        )
        return state, pool, sb

    def test_growth_stops_at_the_state_tensor_dimension(self):
        """The ceiling a `hipErrorLaunchFailure` taught, the hard way.

        Growth was bounded only by the superblock supply, so the pool minted
        indices past the tensor the KDA kernels read through. That is not a
        refused allocation — it is an out-of-bounds GPU read, surfacing eleven
        minutes into a benchmark as a launch failure inside a kernel, with
        nothing in the traceback pointing back at the allocator.

        Sizing the tensor for the supply instead is not an option either: 627
        slots is 32.8 GiB at Kimi-K3's geometry, which is the whole paged pool.
        """
        state, pool, sb = self.elastic(num_blocks=320, per=32, slots=2, max_slots=5)
        while state.has_free() and state.num_slots < 50:
            state.pop()
        assert state.num_slots == 5, "minted past the tensor"
        assert sb.occupancy()["supers_reclaimable"] > 0, "supply was not the limit"

    def test_the_index_space_grows_past_its_initial_ceiling(self):
        state, pool, sb = self.elastic(slots=2)
        taken = []
        while state.has_free() and len(taken) < 50:
            taken.append(state.pop())
        assert state.num_slots > 2, "stayed at the startup ceiling"
        assert len(taken) == state.num_slots

    def test_growth_stops_at_the_kv_floor(self):
        state, pool, sb = self.elastic(num_blocks=320, per=32, slots=2)
        while state.has_free() and state.num_slots < 50:
            state.pop()
        assert sb.occupancy()["supers_reclaimable"] >= pool._kv_superblock_floor()

    def test_growth_runs_while_checkpoints_are_resident(self):
        """The reachability test, which three separate bugs would have failed.

        Every previous defect had the same shape: the code was correct and
        never ran, so a benchmark measured the old path and reported a clean
        number. Asserting behaviour under *pressure with checkpoints resident*
        is what catches that, where asserting it on an empty pool does not.

        Here the growth gate was `not self._free`, and `_free` holds
        checkpointed slots as well as vacant ones — so a pool with any
        checkpoint resident never looked empty, growth never fired, and `pop`
        fell through to spending checkpoints. On hardware that was 114
        evictions against 627 free superblocks.
        """
        state, pool, sb = self.elastic(num_blocks=320, per=32, slots=2)
        # Fill both starting slots and file a checkpoint on each, so every
        # free slot is checkpointed and none is vacant — the state the server
        # actually ran in (`8/32 used, 24 checkpointed, 0 vacant`).
        for h in (101, 202):
            slot = state.pop()
            state._index(h, slot)
            state.release(slot)
        assert state.num_slots == 2
        assert not any(state.slot_hash[s] == -1 for s in state._free), "none vacant"

        before = state.checkpoints_evicted
        state.pop()

        assert state.num_slots > 2, "grew instead of evicting"
        assert state.checkpoints_evicted == before, "spent a checkpoint anyway"
        assert state.lookup(101) >= 0 and state.lookup(202) >= 0

    def test_a_checkpoint_is_spent_only_once_growth_is_exhausted(self):
        """Minting is second in the order, not a replacement for eviction."""
        state, pool, sb = self.elastic(num_blocks=128, per=32, slots=1)
        slot = state.pop()
        state._index(101, slot)
        state.release(slot)
        # Grow until the KV floor stops it, then the only option left is the
        # checkpoint — which `pop` must still be willing to take.
        while state._can_mint(1):
            state.pop()
        before = state.checkpoints_evicted
        state.pop()
        assert state.checkpoints_evicted == before + 1

    def test_a_pool_without_a_source_keeps_its_fixed_ceiling(self):
        from atom.model_engine.state_pool import StateSlotPool
        from atom.model_engine.state_runtime import StateTransfer

        state = StateSlotPool(2, transfer=StateTransfer.fork(1), hash_block_size=4)
        state.pop()
        state.pop()
        assert not state.has_free(), "no source, so no growth"
        assert state.num_slots == 2


class TestDiagnostics:
    def test_partially_pinned_is_reported_for_accumulation_tracking(self):
        """The standing count, not the per-event rate, is the accumulation risk."""
        pool, sb = pooled(num_blocks=8, per=4)
        pool.allocate(0)
        pool.allocate(1)
        occ = sb.occupancy()
        assert occ["supers_partially_pinned"] == 1
        assert occ["supers_total"] == 2


class TestAdmissionAgreesWithAllocation:
    """`has_free(n)` must promise exactly what `pop()` n times can deliver.

    An overcount is not a refused admission -- it is an `AssertionError` from
    `_ensure_backed` partway through, after earlier slots have already been
    handed out.
    """

    @staticmethod
    def _pool(num_blocks=40, per=4, slots=3, spend=9):
        from atom.model_engine.state_pool import StateSlotPool

        sb = SuperblockMap(num_blocks, per)
        pool = BlockPool(num_blocks, superblocks=sb)
        state = StateSlotPool(
            num_slots=slots, superblock_source=pool, max_slots=slots * 3
        )
        for _ in range(spend):  # leave exactly one claimable superblock
            assert pool.claim_superblock() >= 0
        return state

    def test_a_multi_slot_answer_counts_every_superblock_it_needs(self):
        """One claimable superblock does not make three slots available.

        `can_claim_superblock` answers for a single claim, so asking it on
        behalf of a shortfall of three said yes on the strength of one -- the
        first `pop` took it and the second asserted.
        """
        state = self._pool()
        assert not any(s in state._slot_super for s in state._free), "none backed"
        assert state.has_free(3) is False

    def test_a_single_slot_answer_is_not_held_to_the_kv_floor(self):
        """The plural check reserves a KV floor; `claim_superblock` does not.

        So a shortfall of one has to keep asking the singular question, or
        admission refuses slots the pool would in fact have handed over.
        """
        state = self._pool()
        assert state.has_free(1) is True
        assert state.pop() >= 0, "and the claim it promised actually succeeds"


class TestRetireTopBacking:
    def test_a_rehomed_checkpoint_takes_its_superblock_with_it(self):
        """Otherwise `_unback` finds nothing and the bytes never come back.

        `_slot_super` is keyed by slot index, so leaving the entry under the
        retired index strands it: that index no longer exists, and the slot
        now holding the checkpoint has no backing to release.
        """
        from atom.model_engine.state_pool import StateSlotPool

        sb = SuperblockMap(32, 4)
        pool = BlockPool(32, superblocks=sb)
        state = StateSlotPool(num_slots=2, superblock_source=pool, max_slots=4)

        # Drain the pool so the checkpoint lands in the TOP slot, which is the
        # one `retire_top` takes -- only `pop` attaches backing, so the slot
        # has to be reached that way.
        held = [state.pop() for _ in range(state.num_slots)]
        top = state.num_slots - 1
        for slot in held:
            if slot != top:
                state.release(slot)
        state._index(900, top)
        state.release(top)
        assert top in state._slot_super, "the checkpoint's slot is backed"

        moved = state.retire_top()
        assert moved is not None and moved.relocated_to >= 0
        assert moved.retired not in state._slot_super, "no entry left behind"
        assert moved.relocated_to in state._slot_super, "backing followed"
