# SPDX-License-Identifier: MIT
# The worker-side driver: its own executor, and completions reported by req_id.

from atom.kv_transfer.offload.state_tier import StateOffloadTier
from atom.model_engine.state_offload import StateOffloadIndex


class FakeCodec:
    entry_bytes = 4096

    def __init__(self, put_ok=True, get_ok=True):
        self.put_ok, self.get_ok = put_ok, get_ok
        self.puts, self.gets = [], []

    def put(self, h, entry_index):
        self.puts.append((h, entry_index))
        return self.put_ok

    def get(self, h, entry_index):
        self.gets.append((h, entry_index))
        return self.get_ok


def tier(codec):
    return StateOffloadTier(codec)


def test_a_successful_spill_reports_the_hash():
    codec = FakeCodec()
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    indexed, _, index_failed = t.take_spill_reports()
    assert indexed == {11}
    assert index_failed == set()
    # Reported, not applied. The tier holds no index at all -- that object
    # lives in the engine process, and a worker-side copy of it would be a
    # second opinion about what is stored.
    assert not hasattr(t, "index")


def test_the_codec_packs_the_staging_entry_not_the_pool_group():
    """The hook copied the group's bytes into staging precisely so `pop()`
    could hand the original out immediately; packing the group would race the
    new owner."""
    codec = FakeCodec()
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    assert codec.puts == [(11, 64)]


def test_a_failed_spill_is_not_reported_as_indexed():
    """The report is the only record the engine gets, so reporting a spill that
    did not land is a guaranteed false positive on every later request."""
    codec = FakeCodec(put_ok=False)
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    indexed, _, index_failed = t.take_spill_reports()
    assert indexed == set()
    assert index_failed == {11}


def test_a_spill_always_reports_its_staging_slot():
    """A slot the worker never reports is one the engine never refills, so the
    ring shrinks permanently and the feature quietly stops spilling with no
    error anywhere.  Both slots must come back even though both `put`s failed.
    """
    codec = FakeCodec(put_ok=False)
    t = tier(codec)
    # Every slot the ring can hand out, in flight at once: nothing can come
    # back until the reports do.
    ring = StateOffloadIndex(2, kv_offload_enabled=False)
    slot0 = ring.request_spill(11, group=0)
    slot1 = ring.request_spill(22, group=1)
    assert slot0 >= 0 and slot1 >= 0
    assert ring.request_spill(33, group=2) == -1, "ring not empty before transfers"
    t.submit_spill(11, entry_index=64, staging_slot=slot0)
    t.submit_spill(22, entry_index=65, staging_slot=slot1)
    t.drain()
    _, released, _ = t.take_spill_reports()
    assert released == {slot0, slot1}
    for slot in released:
        ring.release_staging(slot)
    assert ring.request_spill(33, group=2) >= 0, "the ring never recovered"


def test_a_successful_load_reports_done():
    codec = FakeCodec()
    t = tier(codec)
    t.submit_load("req-a", 11, group=3)
    t.drain()
    assert t.get_finished() == ({"req-a"}, set())
    # Mirror of test_the_codec_packs_the_staging_entry_not_the_pool_group:
    # the load path must receive a real pool group, not a staging entry.
    assert codec.gets == [(11, 3)]


def test_a_failed_load_reports_failed_by_request():
    """Three triggers funnel here — LMCache's own LRU, a spill that never
    landed, a transfer error — and all three are the same normal path.

    The report names the request and nothing else. Retracting the hash from
    the index is the engine's job: it owns that object, and a worker that
    edited its own copy would be a second opinion nobody reads.
    """
    t = tier(FakeCodec(get_ok=False))
    t.submit_load("req-a", 11, group=3)
    t.drain()
    assert t.get_finished() == (set(), {"req-a"})


def test_a_load_that_was_never_attempted_can_still_be_failed():
    """The no-tier and refused-tier paths need somewhere to send a load they
    cannot serve. Silence is the one answer the engine cannot recover from:
    the request is already parked and only a report unparks it."""
    t = tier(FakeCodec())
    t.fail_loads(["req-a", "req-b"])
    assert t.get_finished() == (set(), {"req-a", "req-b"})


def test_get_finished_drains():
    t = tier(FakeCodec())
    t.submit_load("req-a", 11, group=3)
    t.drain()
    t.get_finished()
    assert t.get_finished() == (set(), set())


def test_inflight_is_empty_after_drain():
    """After drain(), _inflight must be empty — every completed future must
    have been discarded, not accumulated for the lifetime of the process."""
    t = tier(FakeCodec())
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.submit_load("req-a", 11, group=3)
    t.drain()
    with t._lock:
        assert len(t._inflight) == 0


def test_inflight_does_not_grow_on_get_finished_path():
    """The production serving path calls submit → get_finished, never drain.
    _inflight must not accumulate completed futures across repeated cycles.

    Non-vacuousness: revert the add_done_callback fix (use a list that only
    drains in drain()) and this test fails because _inflight keeps growing.
    """
    t = tier(FakeCodec())
    for req_id in ("req-1", "req-2", "req-3"):
        t.submit_load(req_id, 11, group=0)
        # Allow the worker thread to finish so the callback fires before we
        # check; we use the executor directly rather than drain() to mirror
        # the real serving path (no drain call in production).
        t._executor.submit(lambda: None).result()  # fence: wait for prior work
        t.get_finished()

    with t._lock:
        assert len(t._inflight) == 0


# ------------------------ the report/apply boundary ------------------------ #
# The worker half of the state tier runs in a different process from the
# StateOffloadIndex, so it may not mutate it -- it reports and the engine
# applies. These tests pin that boundary.
#
# `ReportingCodec` rather than reusing `FakeCodec` above: a second
# module-level `class FakeCodec` would rebind the name for the whole module
# and silently retarget every test defined before it.


class ReportingCodec:
    def __init__(self, ok=True):
        self.ok, self.calls = ok, []

    def put(self, h, entry_index):
        self.calls.append((h, entry_index))
        return self.ok


def test_a_landed_spill_is_reported_not_applied():
    tier = StateOffloadTier(ReportingCodec(ok=True))
    tier.submit_spill(11, entry_index=5, staging_slot=1)
    tier.drain()
    indexed, released, index_failed = tier.take_spill_reports()
    assert indexed == {11} and released == {1} and index_failed == set()


def test_a_refused_spill_still_releases_its_slot():
    """`put` returning False is normal backpressure (LMCache allocator under
    pressure), not an error -- but the slot must come back either way or the
    ring shrinks permanently."""
    tier = StateOffloadTier(ReportingCodec(ok=False))
    tier.submit_spill(11, entry_index=5, staging_slot=1)
    tier.drain()
    indexed, released, index_failed = tier.take_spill_reports()
    assert indexed == set() and released == {1} and index_failed == {11}


def test_a_throwing_codec_still_releases_its_slot():
    class Boom:
        def put(self, h, entry_index):
            raise RuntimeError("storage down")

    tier = StateOffloadTier(Boom())
    tier.submit_spill(11, entry_index=5, staging_slot=1)
    tier.drain()
    indexed, released, index_failed = tier.take_spill_reports()
    assert indexed == set() and released == {1} and index_failed == {11}


def test_reports_are_drained_once():
    tier = StateOffloadTier(ReportingCodec())
    tier.submit_spill(11, entry_index=5, staging_slot=1)
    tier.drain()
    tier.take_spill_reports()
    assert tier.take_spill_reports() == (set(), set(), set())


def test_the_ready_event_is_waited_on_before_the_pack():
    """The D2D staging copy is issued on the forward's compute stream; the
    pack runs on a worker thread with its own stream. Without the event the
    pack races the copy and stores whatever was in the staging entry before."""

    class RecordingEvent:
        def __init__(self):
            self.synchronized = False

        def synchronize(self):
            self.synchronized = True

    ev = RecordingEvent()
    codec = ReportingCodec()
    tier = StateOffloadTier(codec)
    tier.submit_spill(11, entry_index=5, staging_slot=1, ready_event=ev)
    tier.drain()
    assert ev.synchronized, "spill packed without waiting on the producer event"


# --------------------------------------------------------------------------- #
# Task-9i: partial-store failure channel                                        #
# --------------------------------------------------------------------------- #
# The tier must report failures so the aggregator can take quorum on the union
# of stored and failed sets.  Without this channel, a hash where one rank's put
# fails sits in the aggregator forever (pinned with one worker in its set,
# never reaching world_size).


def test_failed_put_returns_false_lands_hash_in_index_failed():
    """A codec whose `put` returns False reports the hash in index_failed.

    Non-vacuousness: remove the `else: self._index_failed.add(int(h))` branch
    from _do_spill and this test fails because index_failed is empty.
    """
    tier = StateOffloadTier(ReportingCodec(ok=False))
    tier.submit_spill(11, entry_index=5, staging_slot=1)
    tier.drain()
    indexed, released, index_failed = tier.take_spill_reports()
    assert indexed == set(), "a failed store must not appear as indexed"
    assert index_failed == {11}, "hash must appear in index_failed on put=False"
    assert released == {1}, "slot must be released regardless"


def test_raising_codec_lands_hash_in_index_failed():
    """A codec whose `put` raises reports the hash in index_failed.

    Non-vacuousness: remove the `else` branch and this test fails because
    index_failed is empty even though the exception path ran.
    """

    class RaisingCodec:
        def put(self, h, entry_index):
            raise OSError("storage unavailable")

    tier = StateOffloadTier(RaisingCodec())
    tier.submit_spill(22, entry_index=7, staging_slot=3)
    tier.drain()
    indexed, released, index_failed = tier.take_spill_reports()
    assert indexed == set()
    assert index_failed == {22}
    assert released == {3}


def test_state_staging_released_unconditional_on_failed_spill():
    """Slots must be released even when the spill fails.

    This is the load-bearing invariant from the brief: a leaked slot shrinks
    the staging ring permanently and the feature quietly stops spilling.

    Non-vacuousness: condition the `_released.add` on `stored` and this test
    fails because released is empty after a put=False run.
    """
    tier = StateOffloadTier(ReportingCodec(ok=False))
    for slot in range(4):
        tier.submit_spill(slot, entry_index=slot, staging_slot=slot)
    tier.drain()
    _, released, _ = tier.take_spill_reports()
    assert released == {0, 1, 2, 3}, "all slots must be released even on failure"
