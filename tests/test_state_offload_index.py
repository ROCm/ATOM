# SPDX-License-Identifier: MIT
"""StateOffloadIndex must stop advertising images the worker codec dropped,
and keeps state images as warm in LMCache's LRU as the KV they belong to
(hit touches, their verdicts, and the admission hold)."""

from types import SimpleNamespace

import pytest

from atom.model_engine import state_offload as so
from atom.model_engine.state_offload import (
    StateOffloadIndex,
    state_tier_budget_warning,
    state_tier_cpu_bytes,
)


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


@pytest.fixture
def clock(monkeypatch):
    """`state_offload`'s monotonic clock, advanced by hand."""
    now = [1000.0]
    monkeypatch.setattr(so, "monotonic", lambda: now[0])
    return now


# ── the index: votes queue touches, verdicts keep it in step ───────────────


#: Past every confirmation window `tier_verified_for_s` can give.
STALE = so._TIER_VERIFIED_MAX_S + 1


def _index(*hashes, refresh=True, **kw):
    idx = StateOffloadIndex(refresh_hits=refresh, **kw)
    for h in hashes:
        idx.note_stored(h)
    return idx


class TestIndexTouches:
    def test_could_serve_queues_nothing(self, clock):
        idx = _index(7)
        assert idx.could_serve(7) and not idx.could_serve(8)
        assert idx.take_touches() == []

    def test_a_registered_hit_queues_a_touch(self, clock):
        idx = _index(7)
        idx.want_touch(7)
        idx.want_touch(8)  # not indexed: nothing to touch
        assert idx.take_touches() == [7]
        assert idx.take_touches() == []

    def test_no_touches_without_refresh(self, clock):
        idx = _index(7, refresh=False)
        assert idx.could_serve(7)
        idx.want_touch(7)
        assert idx.take_touches() == []

    def test_a_hash_voted_every_pass_is_touched_at_an_interval(self, clock):
        idx = _index(7)
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        clock[0] += so._TIER_TOUCH_INTERVAL_S / 2
        idx.want_touch(7)
        assert idx.take_touches() == []
        clock[0] += so._TIER_TOUCH_INTERVAL_S
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        assert idx.touches_sent == 2

    def test_a_gone_verdict_drops_the_hash(self, clock):
        idx = _index(7)
        clock[0] += 1
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        idx.apply_touch_verdict(7, False)
        assert not idx.could_serve(7)
        assert idx.touch_misses == 1
        assert idx.stats()["touch_misses"] == 1

    def test_a_gone_verdict_older_than_a_store_is_ignored(self, clock):
        """The touch looked before the store's bytes landed; its verdict must
        not erase the image that store just indexed."""
        idx = _index(7)
        clock[0] += 1
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        clock[0] += 0.01
        idx.note_stored(7)
        idx.apply_touch_verdict(7, False)
        assert idx.could_serve(7)
        assert idx.touch_misses == 0

    def test_a_gone_verdict_for_a_forgotten_then_restored_hash_is_ignored(self, clock):
        idx = _index(7)
        clock[0] += 1
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        idx.forget(7)
        clock[0] += 0.01
        idx.note_stored(7)
        idx.apply_touch_verdict(7, False)
        assert idx.could_serve(7)

    def test_a_gone_verdict_after_a_store_and_a_newer_touch_applies(self, clock):
        idx = _index(7)
        clock[0] += 1
        idx.note_stored(7)
        clock[0] += 1
        idx.want_touch(7)
        assert idx.take_touches() == [7]
        idx.apply_touch_verdict(7, False)
        assert not idx.could_serve(7)

    def test_a_resident_verdict_refreshes_the_cap_order(self, clock):
        idx = StateOffloadIndex(max_cpu_entries=19, refresh_hits=True)  # cap 3
        for h in (1, 2, 3):
            idx.note_stored(h)
        idx.apply_touch_verdict(1, True)
        idx.note_stored(4)
        assert idx.could_serve(1) and not idx.could_serve(2)

    def test_a_verdict_for_a_forgotten_hash_is_ignored(self, clock):
        idx = _index(7)
        idx.forget(7)
        idx.apply_touch_verdict(7, True)
        assert not idx.could_serve(7)

    def test_stop_refreshing_drops_pending_touches(self, clock):
        idx = _index(7)
        idx.want_touch(7)
        idx.stop_refreshing()
        assert idx.take_touches() == []
        assert not idx.needs_verify(7)


class TestNeedsVerify:
    def test_a_fresh_store_needs_no_verdict(self, clock):
        assert not _index(7).needs_verify(7)

    def test_a_store_only_role_holds_nothing(self, clock):
        """A `kv_producer` indexes its own stores but never loads them."""
        idx = _index(7, can_load=False)
        clock[0] += STALE
        assert not idx.needs_verify(7)
        assert idx.take_touches() == []

    def test_a_stale_hash_waits_for_its_touch(self, clock):
        idx = _index(7)
        clock[0] += STALE
        assert idx.needs_verify(7)
        assert idx.take_touches() == [7], "the hold asks for its own touch"
        assert idx.needs_verify(7), "still out"
        idx.apply_touch_verdict(7, True)
        assert not idx.needs_verify(7)

    def test_a_gone_verdict_ends_the_hold_by_dropping_the_hash(self, clock):
        idx = _index(7)
        clock[0] += STALE
        assert idx.needs_verify(7)
        idx.take_touches()
        idx.apply_touch_verdict(7, False)
        assert not idx.needs_verify(7) and not idx.could_serve(7)

    def test_a_lost_verdict_holds_no_longer_than_the_lookup_defer(self, clock):
        idx = _index(7, defer_s=2)
        clock[0] += STALE
        # A held request re-probes every pass and its touch is re-sent every
        # interval; verdicts never come back. Re-sends must not extend the hold.
        held_for, step = 0.0, 0.05
        while idx.needs_verify(7):
            assert held_for <= 2 + 1e-6, "the hold outlived OFFLOAD_LOOKUP_DEFER_S"
            assert idx.could_serve(7)
            idx.take_touches()
            clock[0] += step
            held_for += step
        assert held_for >= 2
        assert idx.touches_sent > 1, "the touch was re-sent while held"
        assert idx.verify_holds_expired == 1
        assert idx.stats()["verify_holds_expired"] == 1
        assert not idx.needs_verify(7), "later admissions are not held again"
        assert idx.verify_holds_expired == 1

    def test_evidence_after_an_expired_hold_rearms_it(self, clock):
        idx = _index(7, defer_s=2)
        clock[0] += STALE
        assert idx.needs_verify(7)
        clock[0] += 3
        assert not idx.needs_verify(7)
        idx.apply_touch_verdict(7, True)
        clock[0] += STALE
        assert idx.needs_verify(7), "a fresh hold, timed from now"
        clock[0] += 1
        assert idx.needs_verify(7)

    def test_a_verdict_ends_the_hold_timer(self, clock):
        idx = _index(7, defer_s=2)
        clock[0] += STALE
        assert idx.needs_verify(7)
        clock[0] += 1.5
        idx.apply_touch_verdict(7, True)
        clock[0] += STALE
        assert idx.needs_verify(7)
        clock[0] += 1.5
        assert idx.needs_verify(7), "the earlier 1.5 s does not count"
        assert idx.verify_holds_expired == 0

    def test_a_zero_defer_never_holds(self, clock, monkeypatch):
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "0")
        idx = _index(7)
        assert idx.defer_s == 0
        clock[0] += STALE
        assert not idx.needs_verify(7)

    def test_the_lookup_defer_is_read_once_at_construction(self, clock, monkeypatch):
        """`needs_verify` is on the admission path; it must not re-read and
        re-parse `OFFLOAD_LOOKUP_DEFER_S` on every call."""
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "2")
        idx = _index(7)
        assert idx.defer_s == 2.0

        def unread():
            raise AssertionError("OFFLOAD_LOOKUP_DEFER_S read on the admission path")

        monkeypatch.setattr(so, "verify_defer_s", unread)
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "0")
        clock[0] += STALE
        assert idx.needs_verify(7), "the value read at construction still holds"

    def test_an_index_without_refresh_never_reads_the_defer(self, monkeypatch):
        def unread():
            raise AssertionError("read for an index that never holds")

        monkeypatch.setattr(so, "verify_defer_s", unread)
        assert StateOffloadIndex().defer_s == 0

    def test_a_completed_load_counts_as_verified(self, clock):
        idx = _index(7)
        clock[0] += STALE
        assert idx.request_load("r", 7)
        idx.complete_load("r")
        assert not idx.needs_verify(7)


class TestVerifiedWindow:
    """How long a confirmed image is trusted scales with the LMCache pool it
    has to age out of, within [5 s, 60 s]."""

    GIB = 1 << 30

    def test_the_agentx_pool_keeps_about_the_measured_window(self):
        assert so.tier_verified_for_s(int(208.6 * self.GIB)) == pytest.approx(52.15)

    def test_the_window_grows_with_the_pool_between_its_bounds(self):
        sizes = [8, 20, 32, 64, 100, 208.6, 240, 512, 2048]
        windows = [so.tier_verified_for_s(int(g * self.GIB)) for g in sizes]
        assert windows == sorted(windows)
        assert windows[0] == windows[1] == so._TIER_VERIFIED_MIN_S
        assert windows[2] == pytest.approx(8.0)
        assert windows[-3] == windows[-1] == so._TIER_VERIFIED_MAX_S

    def test_an_unknown_pool_gets_the_floor(self):
        assert so.tier_verified_for_s(0) == so._TIER_VERIFIED_MIN_S

    def test_the_index_holds_by_its_own_window(self, clock):
        idx = _index(7, verified_for_s=so.tier_verified_for_s(32 * self.GIB), defer_s=2)
        assert idx.verified_for_s == pytest.approx(8.0)
        clock[0] += 7.5
        assert not idx.needs_verify(7)
        clock[0] += 1
        assert idx.needs_verify(7)


class TestTouchPlumbing:
    """The index's two calls on the connector, made by the scheduler once a
    pass (`publish_touches` before the metadata is built,
    `collect_touch_verdicts` before the next admission)."""

    def test_voted_hashes_reach_the_connector(self, clock):
        idx = _index(7)
        idx.want_touch(7)
        sent = []
        idx.publish_touches(
            SimpleNamespace(enqueue_state_touches=lambda hs: sent.extend(hs) or True)
        )
        assert sent == [7] and idx.refresh_hits

    def test_a_connector_without_touches_turns_the_refresh_off(self, clock):
        idx = _index(7)
        idx.want_touch(7)
        idx.publish_touches(SimpleNamespace())
        assert not idx.refresh_hits
        clock[0] += STALE
        assert not idx.needs_verify(7), "nothing would ever answer the hold"

    def test_verdicts_are_applied_to_the_index(self, clock):
        idx = _index(7, 8)
        clock[0] += 1
        idx.want_touch(7)
        idx.want_touch(8)
        assert sorted(idx.take_touches()) == [7, 8]
        idx.collect_touch_verdicts(
            SimpleNamespace(take_state_touch_verdicts=lambda: {7: False, 8: True})
        )
        assert not idx.could_serve(7) and idx.could_serve(8)

    def test_presence_only_touches_reach_the_stats_line(self, clock):
        idx = _index(7)
        connector = SimpleNamespace(
            take_state_touch_verdicts=lambda: {7: True},
            take_state_touches_unrefreshed=lambda: 3,
        )
        assert idx.stats()["touches_unrefreshed"] == 0
        idx.collect_touch_verdicts(connector)
        idx.collect_touch_verdicts(connector)
        assert idx.stats()["touches_unrefreshed"] == 6

    def test_an_offered_load_registers_a_touch(self, clock):
        idx = _index(7)
        assert idx.request_load("r", 7)
        assert idx.take_touches() == [7]

    def test_a_hold_is_counted(self, clock):
        idx = _index(7)
        assert not idx.hold_admission(-1), "nothing to load"
        assert not idx.hold_admission(7), "confirmed by its store"
        clock[0] += STALE
        assert idx.hold_admission(7)
        assert idx.loads_deferred_for_verify == 1
        assert idx.stats()["loads_deferred_for_verify"] == 1


# ── the startup budget check ───────────────────────────────────────────────


GIB = 1 << 30


class TestBudgetWarning:
    def test_a_cap_below_half_the_pool_is_fine(self):
        assert state_tier_budget_warning(59 * GIB, 208 * GIB, 112 << 20) is None

    def test_a_cap_the_size_of_the_pool_bounds_nothing(self):
        """#2562's accuracy run: a 32 GiB pool with the 32 GiB default cap."""
        msg = state_tier_budget_warning(32 * GIB, 32 * GIB, 112 << 20)
        assert "bounds nothing" in msg
        assert "LMCACHE_MAX_LOCAL_CPU_SIZE=32.0" in msg

    def test_a_cap_over_half_the_pool_is_called_out(self):
        msg = state_tier_budget_warning(40 * GIB, 64 * GIB, 112 << 20)
        assert msg is not None and "more than half" in msg

    def test_unknown_sizes_are_not_judged(self):
        assert state_tier_budget_warning(32 * GIB, 0, 1) is None
        assert state_tier_budget_warning(0, 32 * GIB, 1) is None
