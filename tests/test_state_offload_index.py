# SPDX-License-Identifier: MIT
"""StateOffloadIndex must stop advertising images the worker codec dropped."""

from math import inf
from types import SimpleNamespace

import pytest

from atom.model_engine import state_offload as so
from atom.model_engine.state_offload import StateOffloadIndex, state_tier_cpu_bytes


def _stored(idx, hashes):
    for h in hashes:
        idx.note_stored(h)


class TestStateOffloadIndexCap:
    def test_no_codec_bound_keeps_the_leak_cap(self):
        assert StateOffloadIndex()._hash_cap == 1 << 20

    def test_cap_tracks_codec_entries_minus_margin(self):
        assert StateOffloadIndex(max_cpu_entries=1183)._hash_cap == 1183 - 36
        assert StateOffloadIndex(max_cpu_entries=100)._hash_cap == 100 - 16
        assert StateOffloadIndex(max_cpu_entries=4)._hash_cap == 1

    def test_oldest_store_is_forgotten_past_the_cap(self):
        idx = StateOffloadIndex(max_cpu_entries=116)  # cap 100
        _stored(idx, range(150))
        assert len(idx.hashes) == 100
        assert idx.hashes_evicted == 50
        assert not idx.could_serve(0)
        assert idx.could_serve(149)

    def test_completed_load_refreshes_like_codec_get(self):
        idx = StateOffloadIndex(max_cpu_entries=116)  # cap 100
        _stored(idx, range(100))
        assert idx.request_load("r", 0)
        idx.complete_load("r")
        idx.note_stored(100)
        # Hash 1 is now least recent, as it is in the codec; 0 survives.
        assert idx.could_serve(0)
        assert not idx.could_serve(1)

    def test_failed_load_does_not_refresh(self):
        idx = StateOffloadIndex(max_cpu_entries=116)
        _stored(idx, range(10))
        assert idx.request_load("r", 3)
        idx.fail_load("r")
        assert not idx.could_serve(3)


def test_state_tier_cpu_bytes_reads_gib(monkeypatch):
    monkeypatch.delenv("OFFLOAD_STATE_CPU_SIZE", raising=False)
    assert state_tier_cpu_bytes() == 32 << 30
    monkeypatch.setenv("OFFLOAD_STATE_CPU_SIZE", "96")
    assert state_tier_cpu_bytes() == 96 << 30


# ── the index: votes queue touches, verdicts keep it in step ───────────────


#: Past every confirmation window `tier_verified_for_s` can give.
STALE = so.tier_verified_for_s(inf) + 1


def _index(*hashes, verified_for_s=5.0, **kw):
    idx = StateOffloadIndex(verified_for_s=verified_for_s, **kw)
    for h in hashes:
        idx.note_stored(h)
    return idx


class TestIndexTouches:
    def test_only_a_registered_hit_queues_a_touch(self, offload_clock):
        idx = _index(7)
        assert idx.could_serve(7)
        assert idx.take_touches() == [], "could_serve is pure"
        assert idx.request_load("r", 7)
        assert idx.take_touches() == [], "a load is its own verdict"
        idx.want_touch(7)
        idx.want_touch(8)  # not indexed: nothing to touch
        assert idx.take_touches() == [7]
        assert idx.take_touches() == []

    def test_no_touches_without_refresh(self, offload_clock):
        idx = _index(7, verified_for_s=None)
        idx.want_touch(7)
        assert idx.take_touches() == []
        offload_clock[0] += STALE
        assert not idx.hold_admission(7)
        # Up to the hash cap of entries nothing would read.
        assert idx._verified_at == {}

    def test_a_hash_voted_every_pass_is_touched_at_an_interval(self, offload_clock):
        idx = _index(7)
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        offload_clock[0] += so._TIER_TOUCH_INTERVAL_S / 2
        idx.want_touch(7)
        assert idx.take_touches() == []
        offload_clock[0] += so._TIER_TOUCH_INTERVAL_S
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        assert idx.touches_sent == 2

    @pytest.mark.parametrize(
        "before_touch, after_touch, dropped",
        [
            ((), (), True),
            ((), ("note_stored",), False),
            ((), ("forget", "note_stored"), False),
            (("note_stored",), (), True),
        ],
        ids=["gone", "stored-since", "restored-since", "stored-before"],
    )
    def test_a_gone_verdict_drops_the_hash_unless_a_store_landed_since(
        self, offload_clock, before_touch, after_touch, dropped
    ):
        idx = _index(7)
        for op in before_touch:
            offload_clock[0] += 1
            getattr(idx, op)(7)
        offload_clock[0] += 1
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        for op in after_touch:
            offload_clock[0] += 0.01
            getattr(idx, op)(7)
        idx.apply_touch_verdict(7, False)
        assert idx.could_serve(7) is not dropped
        assert idx.stats()["touch_misses"] == int(dropped)

    def test_a_resident_verdict_refreshes_the_cap_order(self, offload_clock):
        idx = _index(1, 2, 3, max_cpu_entries=19)  # cap 3
        idx.apply_touch_verdict(1, True)
        idx.note_stored(4)
        assert idx.could_serve(1) and not idx.could_serve(2)

    def test_a_verdict_for_a_forgotten_hash_is_ignored(self, offload_clock):
        idx = _index(7)
        idx.forget(7)
        idx.apply_touch_verdict(7, True)
        assert not idx.could_serve(7)

    def test_touch_state_leaves_with_its_hash(self, offload_clock):
        idx = _index(1, 2, 3, max_cpu_entries=19)  # cap 3
        for h in (1, 2, 3):
            idx.want_touch(h)
        idx.take_touches()
        for h in (1, 2, 3):
            idx.want_touch(h)
        offload_clock[0] += STALE
        assert idx.hold_admission(3)
        idx.note_stored(4)
        idx.note_stored(5)  # evicts 1 and 2
        idx.forget(3)
        for table in (
            idx._touch_wanted,
            idx._touch_sent_at,
            idx._verified_at,
            idx._hold_started_at,
        ):
            assert set(table) <= idx.hashes


class TestHoldAdmission:
    def test_a_fresh_store_needs_no_verdict(self, offload_clock):
        assert not _index(7).hold_admission(7)

    def test_a_store_only_role_holds_nothing(self, offload_clock):
        """A `kv_producer` indexes its own stores but never loads them."""
        idx = _index(7, can_load=False)
        offload_clock[0] += STALE
        assert not idx.hold_admission(7)
        assert idx.take_touches() == []

    def test_a_stale_hash_waits_for_its_touch(self, offload_clock):
        idx = _index(7)
        offload_clock[0] += STALE
        assert not idx.hold_admission(-1), "nothing to load"
        assert idx.hold_admission(7)
        assert idx.take_touches() == [7], "the hold asks for its own touch"
        assert idx.hold_admission(7), "still out"
        assert idx.stats()["loads_deferred_for_verify"] == 2
        idx.apply_touch_verdict(7, True)
        assert not idx.hold_admission(7)

    def test_a_gone_verdict_ends_the_hold_by_dropping_the_hash(self, offload_clock):
        idx = _index(7)
        offload_clock[0] += STALE
        assert idx.hold_admission(7)
        idx.take_touches()
        idx.apply_touch_verdict(7, False)
        assert not idx.hold_admission(7) and not idx.could_serve(7)

    def test_a_lost_verdict_holds_no_longer_than_the_lookup_defer(self, offload_clock):
        idx = _index(7)
        offload_clock[0] += STALE
        # Re-probed every pass, re-touched every interval, never answered:
        # the re-sends must not extend the hold.
        held_for, step = 0.0, 0.05
        while idx.hold_admission(7):
            assert held_for <= idx.defer_s + 1e-6
            idx.take_touches()
            offload_clock[0] += step
            held_for += step
        assert held_for >= idx.defer_s
        assert idx.touches_sent > 1, "the touch was re-sent while held"
        assert not idx.hold_admission(7), "later admissions are not held again"
        assert idx.stats()["verify_holds_expired"] == 1

    def test_evidence_after_an_expired_hold_rearms_it(self, offload_clock):
        idx = _index(7)
        offload_clock[0] += STALE
        assert idx.hold_admission(7)
        offload_clock[0] += idx.defer_s + 1
        assert not idx.hold_admission(7)
        idx.apply_touch_verdict(7, True)
        offload_clock[0] += STALE
        assert idx.hold_admission(7), "a fresh hold, timed from now"
        offload_clock[0] += idx.defer_s / 2
        assert idx.hold_admission(7)

    def test_a_verdict_ends_the_hold_timer(self, offload_clock):
        idx = _index(7)
        offload_clock[0] += STALE
        assert idx.hold_admission(7)
        offload_clock[0] += idx.defer_s * 0.75
        idx.apply_touch_verdict(7, True)
        offload_clock[0] += STALE
        assert idx.hold_admission(7)
        offload_clock[0] += idx.defer_s * 0.75
        assert idx.hold_admission(7), "the earlier wait does not count"

    def test_the_lookup_defer_is_read_once_at_construction(
        self, offload_clock, monkeypatch
    ):
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "0")
        idx = _index(7)
        offload_clock[0] += STALE
        assert not idx.hold_admission(7), "0 disables the hold"
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "2")
        assert not idx.hold_admission(7), "not re-read on the admission path"

    def test_an_index_without_refresh_never_reads_the_defer(self, monkeypatch):
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "not-a-number")
        assert StateOffloadIndex().defer_s == 0

    def test_a_completed_load_counts_as_verified(self, offload_clock):
        idx = _index(7)
        offload_clock[0] += STALE
        assert idx.request_load("r", 7)
        idx.complete_load("r")
        assert not idx.hold_admission(7)


@pytest.mark.parametrize(
    "pool_gib, window", [(0, 5.0), (8, 5.0), (32, 8.0), (208.6, 52.15), (512, 60.0)]
)
def test_the_verified_window_scales_with_the_pool_within_bounds(pool_gib, window):
    assert so.tier_verified_for_s(pool_gib) == pytest.approx(window)


def test_the_index_holds_by_its_own_window(offload_clock):
    idx = _index(7, verified_for_s=8.0)
    offload_clock[0] += 7.5
    assert not idx.hold_admission(7)
    offload_clock[0] += 1
    assert idx.hold_admission(7)


class TestTouchPlumbing:
    """The scheduler's two calls a pass: `publish_touches` before the metadata
    is built, `collect_touch_verdicts` before the next admission."""

    def test_a_connector_without_touches_turns_the_refresh_off(self, offload_clock):
        idx = _index(7)
        idx.want_touch(7)
        idx.publish_touches(SimpleNamespace())
        idx.want_touch(7)
        assert idx.take_touches() == []
        offload_clock[0] += STALE
        assert not idx.hold_admission(7), "nothing would ever answer the hold"
