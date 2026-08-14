# SPDX-License-Identifier: MIT
# The scheduler-side half of the state offload tier: a hash set and a bounded
# spill queue. No device work happens here, so this runs anywhere.

import logging

import pytest

from atom.model_engine import state_pool
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.state_offload import (
    _STARVATION_DROP_THRESHOLD,
    StateOffloadIndex,
    should_load_state,
    state_offload_min_load_tokens,
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


# ------------------------------ the load leg ------------------------------- #
# The engine reserves a load, the worker moves the bytes, the engine settles it.
# Everything here is the engine's half: the worker cannot reach this object.


def test_a_load_is_only_offered_for_a_hash_the_tier_believes_in():
    """`request_load` is the guard between the two index spaces. Asking for a
    hash the tier never stored would park a request against bytes no `get` can
    return, and the park is only ever resolved by a report."""
    idx = index()
    assert idx.request_load("r1", 11) is False
    assert idx.loads_attempted == 0
    idx.confirm_spill(11)
    assert idx.request_load("r1", 11) is True
    assert idx.loads_attempted == 1


def test_one_request_may_only_have_one_load_in_flight(caplog):
    """Reports are keyed by request id and nothing tells two of them apart, so
    the first completion would unpark the request while the second is still
    writing its group. Refusing costs a disown, which is always safe."""
    idx = index()
    idx.confirm_spill(11)
    idx.confirm_spill(22)
    assert idx.request_load("r1", 11) is True
    with caplog.at_level(logging.WARNING):
        assert idx.request_load("r1", 22) is False
    assert "already has a load in flight" in caplog.text
    assert idx.pending_loads == {"r1": 11}
    assert idx.loads_attempted == 1


def test_a_completed_load_leaves_the_hash_in_the_index():
    """A hit does not consume the bytes: LMCache still holds them and the next
    request over the same prefix must find them."""
    idx = index()
    idx.confirm_spill(11)
    idx.request_load("r1", 11)
    idx.complete_load("r1")
    assert idx.loads_completed == 1
    assert 11 in idx.hashes
    assert idx.request_load("r2", 11) is True


def test_a_failed_load_forgets_the_hash():
    """The index is optimistic by construction -- LMCache's LRU can drop bytes
    under a hash it still advertises. The miss is the only signal that happened,
    so it has to be the one that retracts the claim; otherwise every subsequent
    request over that prefix parks, misses, and recomputes."""
    idx = index()
    idx.confirm_spill(11)
    idx.request_load("r1", 11)
    idx.fail_load("r1")
    assert idx.loads_failed == 1
    assert 11 not in idx.hashes
    assert idx.request_load("r2", 11) is False


def test_an_abandoned_load_does_not_retract_the_hash():
    """A request aborted mid-flight says nothing about the bytes. Counting it
    as a failure would forget a hash that is still perfectly loadable, and the
    next request would recompute a prefix the tier is holding."""
    idx = index()
    idx.confirm_spill(11)
    idx.request_load("r1", 11)
    idx.abandon_load("r1")
    assert idx.loads_failed == 0
    assert 11 in idx.hashes
    assert idx.pending_loads == {}


def test_settling_an_unknown_request_is_a_no_op():
    """Reports are keyed by request id and arrive from every rank through an
    aggregator; a duplicate or a stale one must not move a counter."""
    idx = index()
    idx.complete_load("ghost")
    idx.fail_load("ghost")
    idx.abandon_load("ghost")
    assert (idx.loads_completed, idx.loads_failed) == (0, 0)


def test_the_load_counters_read_as_the_index_false_positive_rate():
    """`attempted - completed - failed` is what is still in flight or was
    abandoned, which is the only way to read the three together."""
    idx = index()
    for h in (11, 22, 33):
        idx.confirm_spill(h)
        idx.request_load(f"r{h}", h)
    idx.complete_load("r11")
    idx.fail_load("r22")
    stats = idx.stats()
    assert stats["loads_attempted"] == 3
    assert stats["loads_completed"] == 1
    assert stats["loads_failed"] == 1
    assert len(idx.pending_loads) == 1


def test_resumable_from_ignores_the_tier_while_loads_are_unwired(monkeypatch):
    """A spilled hash is indexed but not *reachable*, so it must not vote.

    `resumable_hit` stops its right-to-left scan at the first hash this
    accepts, so accepting one no load path can deliver hides every shorter
    checkpoint still in HBM. While `STATE_OFFLOAD_LOADS_WIRED` is False the
    predicate is exactly the HBM lookup.
    """
    monkeypatch.setattr(state_pool, "STATE_OFFLOAD_LOADS_WIRED", False)
    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = index()
    pool.offload.confirm_spill(99)
    assert not pool._resumable_from(99, 64)
    pool.hash_to_group[99] = 0
    assert pool._resumable_from(99, 64), "HBM must still answer"


def test_resumable_from_is_hbm_or_tier_once_loads_are_wired(monkeypatch):
    """The re-widening is this one flag and nothing else. Both tiers are keyed
    by the same integer, so once a load can act on the tier its hashes count."""
    monkeypatch.setattr(state_pool, "STATE_OFFLOAD_LOADS_WIRED", True)
    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = index()
    assert not pool._resumable_from(99, 64)
    pool.offload.confirm_spill(99)
    assert pool._resumable_from(99, 64)


def test_resumable_from_without_a_tier_is_the_plain_lookup():
    """Zero cost when disabled is a stated constraint, so the None path must
    behave exactly like the original `h in self.hash_to_group`."""
    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4
    )
    assert pool.offload is None
    assert not pool._resumable_from(99, 64)
    pool.hash_to_group[99] = 0
    assert pool._resumable_from(99, 64)


# ------------------------------- the floor --------------------------------- #


def test_a_hit_at_or_above_the_floor_loads():
    assert should_load_state(8192, 8192) is True


def test_a_short_hit_does_not_repay_the_transfer():
    assert should_load_state(4096, 8192) is False


def test_a_zero_floor_loads_anything_positive():
    assert should_load_state(1, 0) is True


def test_a_zero_hit_never_loads():
    """A floor of 0 means "no minimum", not "load a boundary of nothing": a
    0-token boundary is a cold start, with no state to restore."""
    assert should_load_state(0, 0) is False


def floored_pool(floor, monkeypatch):
    """A loads-wired pool whose tier declines boundaries under `floor` tokens."""
    monkeypatch.setattr(state_pool, "STATE_OFFLOAD_LOADS_WIRED", True)
    monkeypatch.setenv("OFFLOAD_STATE_MIN_LOAD_TOKENS", str(floor))
    pool = StateGroupPool(
        num_groups=4, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = StateOffloadIndex(staging_depth=2, kv_offload_enabled=True)
    return pool


def test_a_boundary_under_the_floor_does_not_repay_the_load(monkeypatch):
    pool = floored_pool(64, monkeypatch)
    pool.offload.confirm_spill(99)
    assert not pool._resumable_from(99, 32)
    assert pool._resumable_from(99, 64)


def test_the_floor_never_gates_a_resident_checkpoint(monkeypatch):
    """The floor is about whether an H2D repays itself. An HBM checkpoint costs
    no transfer at all, so applying the floor there would decline a free hit."""
    pool = floored_pool(1 << 20, monkeypatch)
    pool.hash_to_group[99] = 0
    assert pool._resumable_from(99, 4)


def test_a_short_spilled_rung_lets_the_scan_reach_a_resident_one(monkeypatch):
    """Why the floor lives inside the predicate and not at the load site.

    `resumable_hit` stops at the first boundary the predicate accepts. Accepting
    a too-short spilled rung and declining the transfer afterwards is the §6
    shadowing bug all over again -- the shorter *resident* rung the walk-back
    would have reached is never tried. Declining inside the predicate is what
    lets the scan keep walking.
    """

    class Seq:
        has_per_req_cache = True
        num_tokens = 64

    pool = floored_pool(32, monkeypatch)
    # Rungs at 8 tokens (resident, hash 1) and 12 tokens (spilled, hash 2).
    pool.hash_to_group[1] = 0
    pool.offload.confirm_spill(2)
    assert pool.resumable_hit(Seq(), 3, [0, 1, 2]) == 2, "took the short spilled rung"


def test_the_floor_is_off_by_default(monkeypatch):
    """Unlike KV's OFFLOAD_MIN_LOAD_TOKENS: a state load moves one flat entry
    whatever the boundary is, while the prefill it replaces grows with the
    boundary. There is no length below which the transfer is the expensive
    half, so the default declines nothing."""
    monkeypatch.delenv("OFFLOAD_STATE_MIN_LOAD_TOKENS", raising=False)
    assert state_offload_min_load_tokens() == 0
    monkeypatch.setattr(state_pool, "STATE_OFFLOAD_LOADS_WIRED", True)
    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = StateOffloadIndex(staging_depth=1, kv_offload_enabled=True)
    pool.offload.confirm_spill(99)
    assert pool._resumable_from(99, 4)


@pytest.mark.parametrize("raw", ["-5", "banana"])
def test_a_bad_floor_falls_back_to_zero_loudly(monkeypatch, caplog, raw):
    """Model load must not die on a typo, and must not swallow it either: a
    mistyped floor that silently became huge would turn every load off and read
    as a broken tier."""
    monkeypatch.setenv("OFFLOAD_STATE_MIN_LOAD_TOKENS", raw)
    with caplog.at_level(logging.WARNING):
        assert state_offload_min_load_tokens() == 0
    assert "OFFLOAD_STATE_MIN_LOAD_TOKENS" in caplog.text


def test_a_spilled_hash_still_takes_the_fork_test(monkeypatch):
    """min_fork_tokens is not relaxed for spilled hashes: a boundary too close
    to the end of the prompt leaves GDN's replacement group unfilled, which is
    a wrong state, not a slow one. Asked in the loads-wired world, where the
    tier's hashes are candidates at all and the fork test is what still
    excludes this one."""
    monkeypatch.setattr(state_pool, "STATE_OFFLOAD_LOADS_WIRED", True)

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


def test_the_kv_offload_flag_is_stored():
    assert StateOffloadIndex(1, kv_offload_enabled=True).kv_offload_enabled is True
    assert StateOffloadIndex(1, kv_offload_enabled=False).kv_offload_enabled is False


def orphaning_pool(kv_offload_enabled):
    """A pool holding one checkpoint whose KV blocks have just been dropped."""
    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = StateOffloadIndex(
        staging_depth=2, kv_offload_enabled=kv_offload_enabled
    )
    pool.claim(0)
    pool._index(555, 0)
    pool.release(0)
    return pool


@pytest.mark.parametrize("kv_offload_enabled", [True, False])
def test_an_orphan_is_spilled_only_when_kv_offload_can_bring_the_blocks_back(
    kv_offload_enabled,
):
    """`unindex` gates its `_spill` on the flag (`state_pool.py`), and the gate
    is the whole reason the flag is plumbed this far down.

    A checkpoint is a joint claim on state *and* KV. When the KV blocks are
    gone and KV offload is off, they can never come back, so the hash is
    unreachable forever -- spilling it would spend LMCache capacity on bytes no
    load could ever use. With KV offload on, the blocks can be fetched again,
    so the state is worth keeping. Without this gate every orphan spills.
    """
    pool = orphaning_pool(kv_offload_enabled)
    assert pool.unindex(555) == 0

    copies = pool.take_spill_copies()
    pending = pool.offload.take_pending()
    if kv_offload_enabled:
        assert copies == [(0, 0)]
        assert pending == [(555, 0)]
    else:
        assert copies == []
        assert pending == []


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


@pytest.mark.parametrize(
    "value",
    ["0", "false", "no", "off", "OFF", "", " ", "off ", " off", "\tOFF\n"],
)
def test_the_usual_spellings_of_off_keep_it_off(monkeypatch, value):
    """Empty and padded spellings included, and they are the dangerous ones.

    `OFFLOAD_STATE=` is how a shell script clears a flag inline, and a bare
    `not in ("0", "false", ...)` test reads the empty string as ON. On a
    default-off feature that fails in the wrong direction: the operator who
    just wrote the flag off gets a server that spills.
    """
    monkeypatch.setenv("OFFLOAD_STATE", value)
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "4")
    assert state_offload_staging_groups() == 0


def test_a_negative_depth_is_floored_to_zero(monkeypatch):
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "-3")
    assert state_offload_staging_groups() == 0


def test_a_negative_depth_is_not_silent(monkeypatch, caplog):
    """Louder than the `banana` case deserves to be, not quieter. `banana`
    costs you 1 group instead of 20; `-3` returns 0, which is the same value
    `OFFLOAD_STATE=0` returns -- the tier is off entirely while the flag says
    on, and the only symptom is a server that never spills."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "-3")
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert state_offload_staging_groups() == 0
    assert "-3" in caplog.text


class OnlyStateCaches:
    """`state_spills_for_batch` reads `self.state_caches` and nothing else."""

    def __init__(self, caches):
        self.state_caches = caches

    spills_for_batch = BlockManager.state_spills_for_batch


def spilling_pool(num_groups=4):
    pool = StateGroupPool(
        num_groups=num_groups, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = StateOffloadIndex(staging_depth=2, kv_offload_enabled=False)
    return pool


def test_spills_for_batch_joins_the_copy_and_the_hash_on_the_slot():
    """The pool knows (group, slot); the ring knows (hash, slot). A spill
    needs all three, and the only thing that relates them is the slot.

    Driving the real `BlockManager.state_spills_for_batch`, because the join
    and the `num_groups + slot` addressing both live there -- re-deriving them
    in the test would assert the expression against itself.
    """
    pool = spilling_pool()
    pool.group_hash[1] = 111
    pool.group_hash[2] = 222
    pool._spill(1)
    pool._spill(2)

    # (src_group, dst_entry, staging_slot, hash). The destination is addressed
    # in the same space `state_entry_views` uses: past the pool's own groups.
    assert sorted(OnlyStateCaches([pool]).spills_for_batch()) == [
        (1, 4, 0, 111),
        (2, 5, 1, 222),
    ]


def test_a_copy_with_no_pending_hash_is_dropped_and_its_slot_released(caplog):
    """The two lists are appended by one `_spill()` and drained together, so a
    slot in one and not the other means something already went wrong. Guessing
    which half is right would store bytes under a hash they do not belong to;
    dropping the spill costs one later prefix hit. The slot must still come
    back, or the ring leaks a staging entry per occurrence.
    """
    pool = spilling_pool()
    pool.group_hash[1] = 111
    pool._spill(1)
    pool.offload.take_pending()  # the hash half vanishes; the copy remains

    free_before = len(pool.offload._free_slots)
    with caplog.at_level(logging.WARNING):
        assert OnlyStateCaches([pool]).spills_for_batch() == []
    assert len(pool.offload._free_slots) == free_before + 1
    assert "no pending hash" in caplog.text


def test_a_pending_hash_with_no_copy_releases_its_slot_too():
    """The mirror image: nothing to feed the staging entry, so nothing is
    spilled, but the slot is still the ring's to reclaim."""
    pool = spilling_pool()
    pool.group_hash[1] = 111
    pool._spill(1)
    pool.take_spill_copies()  # the copy half vanishes; the hash remains

    free_before = len(pool.offload._free_slots)
    assert OnlyStateCaches([pool]).spills_for_batch() == []
    assert len(pool.offload._free_slots) == free_before + 1


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


def test_the_offload_counters_reach_the_periodic_stats_line():
    """`stats()` had no caller, so `spills_dropped` was invisible below the
    256-consecutive-drop starvation warning -- and a ring that drops one spill
    in three never trips that. `checkpoint_fates` is what the scheduler reads
    every 100 ticks, so the counters have to arrive there.

    Non-vacuousness: drop the `self.offload is not None` merge in
    `checkpoint_fates` and this fails on the missing key.
    """
    pool = spilling_pool()
    pool.group_hash[1] = 111
    pool._spill(1)
    pool._spill(1)  # depth is 2, so both land
    pool.group_hash[2] = 222
    pool._spill(2)  # ring is full: dropped

    fates = pool.checkpoint_fates()
    assert fates["state_offload_spills_dropped"] == 1
    assert fates["state_offload_spills_requested"] == 2
    # The pool's own counters must still be there -- this merges, not replaces.
    assert "checkpoints_evicted" in fates


def test_the_offload_keys_are_namespaced():
    """`state_checkpoint_fates` sums by key across every state class, so an
    unprefixed `indexed` would silently add itself to any future pool's."""
    pool = spilling_pool()
    assert "state_offload_indexed" in pool.checkpoint_fates()
    assert "indexed" not in pool.checkpoint_fates()


def test_a_pool_with_no_tier_reports_only_its_own_fates():
    """Zero cost when disabled: with no tier attached the dict is exactly what
    it was before the merge, so the log line does not grow five zero columns
    on every server that leaves OFFLOAD_STATE off."""
    pool = StateGroupPool(
        num_groups=4, transfer=StateTransfer.copy(), hash_block_size=4
    )
    assert pool.offload is None
    assert not any(k.startswith("state_offload_") for k in pool.checkpoint_fates())
