# SPDX-License-Identifier: MIT
# The scheduler-side half of the state offload tier: a hash set and a bounded
# spill queue. No device work happens here, so this runs anywhere.

import logging

import pytest

from atom.model_engine.state_offload import (
    _STARVATION_DROP_THRESHOLD,
    StateOffloadIndex,
    state_offload_staging_groups,
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


def test_disabled_by_default_costs_nothing():
    """Zero cost when disabled is a stated constraint: depth 0 means every
    request_spill is refused and `hashes` stays empty, so `_resumable_from`
    degenerates to the original `in`."""
    idx = StateOffloadIndex(staging_depth=0, kv_offload_enabled=False)
    assert idx.enabled is False
    assert idx.request_spill(11, group=1) == -1
    assert idx.hashes == set()


def test_kv_offload_flag_is_carried_for_the_orphan_decision():
    assert StateOffloadIndex(1, kv_offload_enabled=True).kv_offload_enabled is True
    assert StateOffloadIndex(1, kv_offload_enabled=False).kv_offload_enabled is False


def test_staging_groups_is_zero_unless_the_tier_is_switched_on(monkeypatch):
    monkeypatch.delenv("OFFLOAD_STATE", raising=False)
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "4")
    assert state_offload_staging_groups() == 0


def test_staging_groups_reads_its_depth_when_on(monkeypatch):
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "4")
    assert state_offload_staging_groups() == 4


def test_staging_groups_defaults_to_one(monkeypatch):
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.delenv("OFFLOAD_STATE_STAGING_GROUPS", raising=False)
    assert state_offload_staging_groups() == 1


def test_garbage_depth_falls_back_rather_than_crashing_model_load(monkeypatch):
    """This runs inside model load. A typo in an env var must not be fatal."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "banana")
    assert state_offload_staging_groups() == 1


def test_garbage_depth_is_not_silent(monkeypatch, caplog):
    """Falling back to 1 without a word is worse than the typo: a mistyped
    depth looks exactly like a deliberate one until the ring starves."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "2O")
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert state_offload_staging_groups() == 1
    assert "2O" in caplog.text


@pytest.mark.parametrize("value", ["true", "TRUE", "yes", "on"])
def test_the_usual_spellings_of_on_turn_the_tier_on(monkeypatch, value):
    """`OFFLOAD_STATE=true` means on. Reading only the literal "1" would give
    that user a healthy-looking server that never spills."""
    monkeypatch.setenv("OFFLOAD_STATE", value)
    monkeypatch.delenv("OFFLOAD_STATE_STAGING_GROUPS", raising=False)
    assert state_offload_staging_groups() == 1


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_the_usual_spellings_of_off_keep_it_off(monkeypatch, value):
    monkeypatch.setenv("OFFLOAD_STATE", value)
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "4")
    assert state_offload_staging_groups() == 0


def test_a_negative_depth_is_floored_to_zero(monkeypatch):
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "-3")
    assert state_offload_staging_groups() == 0


def test_spills_for_batch_joins_the_copy_and_the_hash_on_the_slot():
    """The pool knows (group, slot); the ring knows (hash, slot). A spill
    needs all three, and the only thing that relates them is the slot."""
    from atom.model_engine.state_pool import StateGroupPool, StateTransfer

    pool = StateGroupPool(
        num_groups=4, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = StateOffloadIndex(staging_depth=2, kv_offload_enabled=False)
    pool.group_hash[1] = 111
    pool.group_hash[2] = 222
    pool._spill(1)
    pool._spill(2)

    copies = dict(pool.take_spill_copies())  # group -> slot
    pending = {s: h for h, s in pool.offload.take_pending()}  # slot -> hash
    joined = sorted((g, pool.num_groups + s, s, pending[s]) for g, s in copies.items())
    assert [t[0] for t in joined] == [1, 2]
    assert [t[3] for t in joined] == [111, 222]
    # The destination is addressed in the same space state_entry_views uses.
    assert all(dst == pool.num_groups + slot for _, dst, slot, _ in joined)


def test_a_slot_returns_only_after_the_report_comes_back():
    """The ring must not free a slot when the copy is issued -- only when the
    worker says its D2H landed. Freeing early hands the same staging entry to
    a second spill while the first is still being read."""
    idx = index(depth=1)
    slot = idx.request_spill(11, group=3)
    idx.take_pending()
    assert idx.request_spill(22, group=4) == -1  # still busy
    idx.release_staging(slot)  # the report arrives
    assert idx.request_spill(33, group=5) == slot


def test_a_slot_is_released_only_when_every_rank_reports():
    """Each TP rank D2Hs its own shard out of the same staging entry. The
    entry is reusable only once the last rank is done with it."""
    from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
    from atom.kv_transfer.disaggregation.types import KVConnectorOutput

    agg = KVOutputAggregator(world_size=2)
    out = agg.aggregate(
        [
            KVConnectorOutput(state_staging_released={1}, state_indexed={99}),
            KVConnectorOutput(),
        ]
    )
    assert out.state_staging_released == set() and out.state_indexed == set()
    out = agg.aggregate(
        [
            KVConnectorOutput(),
            KVConnectorOutput(state_staging_released={1}, state_indexed={99}),
        ]
    )
    assert out.state_staging_released == {1} and out.state_indexed == {99}
