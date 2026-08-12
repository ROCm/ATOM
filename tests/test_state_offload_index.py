# SPDX-License-Identifier: MIT
# The scheduler-side half of the state offload tier: a hash set and a bounded
# spill queue. No device work happens here, so this runs anywhere.

import logging

from atom.model_engine.state_offload import (
    _STARVATION_DROP_THRESHOLD,
    StateOffloadIndex,
)
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


# --------------------- the undrained-consumer detectors --------------------- #
# `take_spill_copies` has no in-tree consumer yet. A tier attached before one
# exists leaks every staging slot and silently stops spilling, so both halves of
# that failure have to announce themselves rather than degrade quietly.


def test_a_negative_hash_is_not_counted_as_a_dropped_spill():
    """`_spill` already refuses a group with no checkpoint, so h<0 is a caller
    bug. Counting it as a drop would inflate the backpressure signal."""
    idx = index(depth=2)
    assert idx.request_spill(-1, group=3) == -1
    assert idx.spills_dropped == 0
    assert idx.spills_requested == 0


def test_a_starved_ring_warns_once_naming_the_undrained_consumer(caplog):
    idx = index(depth=1)
    assert idx.request_spill(11, group=3) >= 0  # takes the only slot, never freed
    with caplog.at_level(logging.WARNING, logger="atom"):
        for i in range(_STARVATION_DROP_THRESHOLD + 5):
            assert idx.request_spill(100 + i, group=4) == -1
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    msg = warnings[0].getMessage()
    assert "take_spill_copies" in msg and "release_staging" in msg


def test_drain_and_release_never_warns(caplog):
    """The detector must not fire on healthy traffic, or it is just noise:
    many more spills than the threshold, but each slot comes back."""
    pool = StateGroupPool(
        num_groups=4, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = index(depth=2)
    with caplog.at_level(logging.WARNING, logger="atom"):
        for i in range(_STARVATION_DROP_THRESHOLD * 4):
            pool.group_hash[0] = 1000 + i
            pool._spill(0)
            for _group, slot in pool.take_spill_copies():
                pool.offload.confirm_spill(1000 + i)
                pool.offload.release_staging(slot)
            pool.offload.take_pending()
    assert pool.offload.spills_dropped == 0
    assert [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ] == []


def test_undrained_spill_copies_past_the_staging_depth_warns_once(caplog):
    """The exact detector: a slot must be released before it is handed out
    again, so more outstanding copies than `staging_depth` is proof, not a
    heuristic, that nobody is calling `take_spill_copies`."""
    pool = StateGroupPool(
        num_groups=4, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = StateOffloadIndex(staging_depth=2, kv_offload_enabled=False)
    # Hand the ring back its slots without ever draining `_spill_copies`, so
    # spills keep succeeding while the pool-side list grows past the depth.
    with caplog.at_level(logging.WARNING, logger="atom"):
        for i in range(6):
            pool.group_hash[0] = 2000 + i
            pool._spill(0)
            pool.offload.release_staging(i % 2)
    assert len(pool._spill_copies) > pool.offload.staging_depth
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "take_spill_copies" in warnings[0].getMessage()
