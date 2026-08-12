# SPDX-License-Identifier: MIT
# The scheduler-side half of the state offload tier: a hash set and a bounded
# spill queue. No device work happens here, so this runs anywhere.

from atom.model_engine.state_offload import StateOffloadIndex
from atom.model_engine.state_pool import StateGroupPool, StateTransfer


def index(depth=2):
    return StateOffloadIndex(staging_depth=depth, kv_offload_enabled=False)


def test_a_spill_reserves_a_distinct_staging_slot():
    idx = index(depth=2)
    a = idx.request_spill(11, group=3)
    b = idx.request_spill(22, group=4)
    assert a >= 0 and b >= 0 and a != b


def test_spills_beyond_depth_k_are_dropped():
    """Dropping is not a regression: checkpoints_evicted counts them either
    way, which is exactly today's behaviour."""
    idx = index(depth=1)
    assert idx.request_spill(11, group=3) >= 0
    assert idx.request_spill(22, group=4) == -1
    assert idx.spills_dropped == 1


def test_a_released_slot_is_reusable():
    idx = index(depth=1)
    slot = idx.request_spill(11, group=3)
    idx.take_pending()
    idx.release_staging(slot)
    assert idx.request_spill(22, group=4) == slot


def test_take_pending_drains():
    idx = index(depth=2)
    idx.request_spill(11, group=3)
    assert [h for h, _ in idx.take_pending()] == [11]
    assert idx.take_pending() == []


def test_only_a_confirmed_spill_enters_the_index():
    idx = index()
    idx.request_spill(11, group=3)
    assert 11 not in idx.hashes
    idx.confirm_spill(11)
    assert 11 in idx.hashes


def test_forget_drops_a_hash_that_failed_to_load():
    idx = index()
    idx.confirm_spill(11)
    idx.forget(11)
    assert 11 not in idx.hashes


def test_resumable_from_is_hbm_or_tier():
    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = index()
    assert not pool._resumable_from(99)
    pool.offload.confirm_spill(99)
    assert pool._resumable_from(99)


def test_resumable_from_without_a_tier_is_the_plain_lookup():
    """Zero cost when disabled is a stated constraint, so the None path must
    behave exactly like the original `h in self.hash_to_group`."""
    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4
    )
    assert pool.offload is None
    assert not pool._resumable_from(99)
    pool.hash_to_group[99] = 0
    assert pool._resumable_from(99)


def test_a_spilled_hash_still_takes_the_fork_test():
    """min_fork_tokens is not relaxed for spilled hashes: a boundary too close
    to the end of the prompt leaves GDN's replacement group unfilled, which is
    a wrong state, not a slow one."""

    class Seq:
        has_per_req_cache = True
        num_tokens = 8

    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.fork(tokens=64), hash_block_size=4
    )
    pool.offload = index()
    pool.offload.confirm_spill(7)
    assert pool.resumable_hit(Seq(), 2, [3, 7]) == 0
