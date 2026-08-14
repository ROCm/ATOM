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


def test_resumable_from_ignores_the_tier_while_loads_are_unwired(monkeypatch):
    """A spilled hash is indexed but not *reachable*, so it must not vote.

    `resumable_hit` stops its right-to-left scan at the first hash this
    accepts, so accepting one no load path can deliver hides every shorter
    checkpoint still in HBM. While `STATE_OFFLOAD_LOADS_WIRED` is False the
    predicate is exactly the HBM lookup.
    """
    assert state_pool.STATE_OFFLOAD_LOADS_WIRED is False, "the branch ships unwired"
    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4
    )
    pool.offload = index()
    pool.offload.confirm_spill(99)
    assert not pool._resumable_from(99)
    pool.hash_to_group[99] = 0
    assert pool._resumable_from(99), "HBM must still answer"


def test_resumable_from_is_hbm_or_tier_once_loads_are_wired(monkeypatch):
    """The re-widening is this one flag and nothing else. Both tiers are keyed
    by the same integer, so once a load can act on the tier its hashes count."""
    monkeypatch.setattr(state_pool, "STATE_OFFLOAD_LOADS_WIRED", True)
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
