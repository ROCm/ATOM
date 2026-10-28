# SPDX-License-Identifier: MIT
"""State images are refreshed in LMCache's LRU like the KV they belong to.

State images and paged KV share one LocalCPU pool. A KV lookup refreshes every
chunk it touches (`contains(pin=True)` then `touch_cache()` ->
`cache_policy.update_on_hit`), but a state hit is answered engine-side by
`StateOffloadIndex`, so LMCache never learned an image was used: a prefix hit
on every turn kept its KV at the young end and its image at the cold end, where
pool pressure evicted it first, and the index kept advertising it until a joint
load parked on it, missed, and recomputed the whole prompt.

These tests model LocalCPU's hot cache and LRU policy, and cover the three
pieces of the fix: the codec touch, the engine-side touch/verdict loop that
keeps the index in step with evictions, and the admission hold that lets a
verdict land before a request parks on an image.
"""

import dataclasses
import sys
import threading
import types
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import ConnectorCompletion, StateTouchId
from atom.model_engine import state_offload as so
from atom.model_engine.state_offload import (
    StateOffloadIndex,
    state_tier_budget_warning,
)


@pytest.fixture
def lmcache_key(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class CacheEngineKey:
        model_name: str
        world_size: int
        worker_id: int
        chunk_hash: int
        dtype: object

    utils_module = types.ModuleType("lmcache.utils")
    utils_module.CacheEngineKey = CacheEngineKey
    lmcache_module = types.ModuleType("lmcache")
    lmcache_module.__path__ = []
    lmcache_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "lmcache", lmcache_module)
    monkeypatch.setitem(sys.modules, "lmcache.utils", utils_module)
    return CacheEngineKey


@pytest.fixture
def clock(monkeypatch):
    """`state_offload`'s monotonic clock, advanced by hand."""
    now = [1000.0]
    monkeypatch.setattr(so, "monotonic", lambda: now[0])
    return now


class _Obj:
    """A MemoryObj's reference count, and nothing else."""

    def __init__(self, size: int = 1024) -> None:
        self.refs = 1
        self.size = size

    def ref_count_up(self) -> None:
        self.refs += 1

    def ref_count_down(self) -> None:
        assert self.refs > 0, "reference discharged twice"
        self.refs -= 1

    def get_ref_count(self) -> int:
        return self.refs

    def get_size(self) -> int:
        return self.size


class _LRUPolicy:
    """LMCache's `LRUCachePolicy.update_on_hit`."""

    def update_on_hit(self, key, cache) -> None:
        cache.move_to_end(key)


class _LocalCPU:
    """`storage_manager` over LocalCPU's rules.

    `batched_put` adopts an object with its own reference unless the key is
    already resident (LocalCPU refuses duplicates), then drops the caller's.
    `get` does not touch the LRU (`get_blocking` only takes a reference).
    `evict_oldest` is the pool making room for KV: oldest first, skipping
    anything another holder still references.
    """

    def __init__(self, refuse_all: bool = False) -> None:
        self.hot: OrderedDict = OrderedDict()
        self.refuse_all = refuse_all
        self.fail_put = False
        self.local_cpu_backend = SimpleNamespace(
            cpu_lock=threading.Lock(), hot_cache=self.hot, cache_policy=_LRUPolicy()
        )

    def batched_put(self, keys, objs) -> None:
        if self.fail_put:
            raise RuntimeError("put failed")
        for k, o in zip(keys, objs, strict=True):
            if not self.refuse_all and k not in self.hot:
                o.ref_count_up()
                self.hot[k] = o
            o.ref_count_down()

    def get(self, key):
        o = self.hot.get(key)
        if o is not None:
            o.ref_count_up()
        return o

    def contains(self, key):
        return "LocalCPUBackend" if key in self.hot else None

    def remove(self, key) -> int:
        o = self.hot.pop(key, None)
        if o is None:
            return 0
        o.ref_count_down()
        return 1

    def evict_oldest(self):
        for k, o in self.hot.items():
            if o.get_ref_count() == 1:
                self.hot.pop(k).ref_count_down()
                return k
        return None


def _codec(storage, max_entries):
    from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec

    codec = StateByteCodec(
        backend=SimpleNamespace(
            page_unit_views=lambda units: [], state_entry_views=lambda slot: []
        ),
        staged=SimpleNamespace(pack=lambda views, obj: None, unpack=lambda o, v: None),
        entry_bytes=1024,
        model_name="qwen",
        world_size=1,
        worker_id=0,
        layout_id="L",
        max_cache_bytes=None if max_entries is None else max_entries * 1024,
    )
    codec.bind_storage_manager(storage)
    allocated = []

    def allocate(nbytes):
        allocated.append(_Obj(nbytes))
        return allocated[-1]

    codec._allocate = allocate
    return codec, allocated


def _kv(pool, name):
    pool.hot[name] = _Obj()


# ── the codec: touch, get and re-store refresh LMCache's LRU ───────────────


class TestCodecRefresh:
    def test_an_untouched_image_is_the_first_thing_pressure_evicts(self, lmcache_key):
        """The bug: KV written after the image is younger, however often the
        prefix hits."""
        pool = _LocalCPU()
        codec, _ = _codec(pool, max_entries=None)
        assert codec.put(1, [])
        _kv(pool, "kv-chunk")
        assert pool.evict_oldest() == codec.key(1)

    def test_a_touch_moves_the_image_past_older_kv(self, lmcache_key):
        pool = _LocalCPU()
        codec, _ = _codec(pool, max_entries=None)
        assert codec.put(1, [])
        _kv(pool, "kv-chunk")
        assert codec.touch(1) is True
        assert pool.evict_oldest() == "kv-chunk"
        assert codec.key(1) in pool.hot

    def test_a_touch_reports_an_evicted_image_gone(self, lmcache_key):
        pool = _LocalCPU()
        codec, _ = _codec(pool, max_entries=2)
        assert codec.put(1, [])
        assert pool.evict_oldest() == codec.key(1)
        assert codec.touch(1) is False
        assert codec.key(1) not in codec._cache_keys, "no longer counts against the cap"

    def test_a_resume_refreshes_the_image(self, lmcache_key):
        """`get_blocking` takes a reference and nothing else."""
        pool = _LocalCPU()
        codec, objs = _codec(pool, max_entries=None)
        assert codec.put(1, [])
        _kv(pool, "kv-chunk")
        assert codec.get(1, 0)
        assert objs[0].refs == 1, "the load's reference is discharged"
        assert pool.evict_oldest() == "kv-chunk"

    def test_a_missed_get_leaves_the_cap_lru(self, lmcache_key):
        pool = _LocalCPU()
        codec, _ = _codec(pool, max_entries=2)
        assert codec.put(1, [])
        pool.evict_oldest()
        assert not codec.get(1, 0)
        assert codec._cache_keys == OrderedDict()

    def test_a_resident_image_is_refreshed_not_stored_twice(self, lmcache_key):
        pool = _LocalCPU()
        codec, objs = _codec(pool, max_entries=2)
        released = []
        assert codec.put(1, [])
        _kv(pool, "kv-chunk")
        assert codec.put(1, [], on_source_released=lambda: released.append(True))

        assert len(objs) == 1, "no second image-sized allocation and D2H"
        assert released == [True], "the source units must still be handed back"
        assert pool.evict_oldest() == "kv-chunk"

    def test_an_evicted_image_is_stored_again(self, lmcache_key):
        pool = _LocalCPU()
        codec, objs = _codec(pool, max_entries=2)
        assert codec.put(1, [])
        pool.evict_oldest()
        assert codec.put(1, [])
        assert len(objs) == 2 and codec.key(1) in pool.hot

    def test_the_cap_retires_the_least_recently_used_image(self, lmcache_key):
        """Not the oldest store: the image a touch kept warm survives."""
        pool = _LocalCPU()
        codec, objs = _codec(pool, max_entries=2)
        assert codec.put(1, []) and codec.put(2, [])
        assert codec.touch(1)
        assert codec.put(3, [])

        assert codec.key(2) not in pool.hot
        assert objs[1].refs == 0, "a retired image goes back to the allocator"
        assert list(codec._cache_keys) == [codec.key(1), codec.key(3)]

    def test_images_hold_no_extra_reference(self, lmcache_key):
        """No reservation: KV pressure may evict an image like any chunk."""
        pool = _LocalCPU()
        codec, objs = _codec(pool, max_entries=2)
        assert codec.put(1, [])
        assert objs[0].refs == 1
        assert pool.evict_oldest() == codec.key(1)

    def test_a_refused_put_keeps_no_reference(self, lmcache_key):
        pool = _LocalCPU(refuse_all=True)
        codec, objs = _codec(pool, max_entries=2)
        assert codec.put(1, [])
        assert objs[0].refs == 0, "holding a refused object would strand its block"

    def test_a_failed_put_discharges_its_reference(self, lmcache_key):
        pool = _LocalCPU()
        pool.fail_put = True
        codec, objs = _codec(pool, max_entries=2)
        with pytest.raises(RuntimeError):
            codec.put(1, [])
        assert objs[0].refs == 0

    def test_without_a_local_cpu_backend_a_touch_asks_presence(self, lmcache_key):
        pool = _LocalCPU()
        codec, _ = _codec(pool, max_entries=None)
        assert codec.put(1, [])
        pool.local_cpu_backend = None
        assert codec.touch(1) is True
        pool.hot.clear()
        assert codec.touch(1) is False


# ── the index: votes queue touches, verdicts keep it in step ───────────────


def _index(*hashes, refresh=True):
    idx = StateOffloadIndex(refresh_hits=refresh)
    for h in hashes:
        idx.note_stored(h)
    return idx


class TestIndexTouches:
    def test_a_hit_vote_queues_a_touch(self, clock):
        idx = _index(7)
        assert idx.could_serve(7) and not idx.could_serve(8)
        assert idx.take_touches() == [7]
        assert idx.take_touches() == []

    def test_no_touches_without_refresh(self, clock):
        idx = _index(7, refresh=False)
        assert idx.could_serve(7)
        assert idx.take_touches() == []

    def test_a_hash_voted_every_pass_is_touched_at_an_interval(self, clock):
        idx = _index(7)
        idx.could_serve(7)
        assert idx.take_touches() == [7]
        clock[0] += so._TIER_TOUCH_INTERVAL_S / 2
        idx.could_serve(7)
        assert idx.take_touches() == []
        clock[0] += so._TIER_TOUCH_INTERVAL_S
        idx.could_serve(7)
        assert idx.take_touches() == [7]
        assert idx.touches_sent == 2

    def test_a_gone_verdict_drops_the_hash(self, clock):
        idx = _index(7)
        idx.apply_touch_verdict(7, False)
        assert not idx.could_serve(7)
        assert idx.touch_misses == 1
        assert idx.stats()["touch_misses"] == 1

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
        idx.could_serve(7)
        idx.stop_refreshing()
        assert idx.take_touches() == []
        assert not idx.needs_verify(7)


class TestNeedsVerify:
    def test_a_fresh_store_needs_no_verdict(self, clock):
        assert not _index(7).needs_verify(7)

    def test_a_stale_hash_waits_for_its_touch(self, clock):
        idx = _index(7)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert idx.needs_verify(7)
        assert idx.take_touches() == [7], "the hold asks for its own touch"
        assert idx.needs_verify(7), "still out"
        idx.apply_touch_verdict(7, True)
        assert not idx.needs_verify(7)

    def test_a_gone_verdict_ends_the_hold_by_dropping_the_hash(self, clock):
        idx = _index(7)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert idx.needs_verify(7)
        idx.take_touches()
        idx.apply_touch_verdict(7, False)
        assert not idx.needs_verify(7) and not idx.could_serve(7)

    def test_a_lost_verdict_holds_no_longer_than_the_lookup_defer(
        self, clock, monkeypatch
    ):
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "2")
        idx = _index(7)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
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

    def test_evidence_after_an_expired_hold_rearms_it(self, clock, monkeypatch):
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "2")
        idx = _index(7)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert idx.needs_verify(7)
        clock[0] += 3
        assert not idx.needs_verify(7)
        idx.apply_touch_verdict(7, True)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert idx.needs_verify(7), "a fresh hold, timed from now"
        clock[0] += 1
        assert idx.needs_verify(7)

    def test_a_verdict_ends_the_hold_timer(self, clock, monkeypatch):
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "2")
        idx = _index(7)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert idx.needs_verify(7)
        clock[0] += 1.5
        idx.apply_touch_verdict(7, True)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert idx.needs_verify(7)
        clock[0] += 1.5
        assert idx.needs_verify(7), "the earlier 1.5 s does not count"
        assert idx.verify_holds_expired == 0

    def test_a_zero_defer_never_holds(self, clock, monkeypatch):
        monkeypatch.setenv("OFFLOAD_LOOKUP_DEFER_S", "0")
        idx = _index(7)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert not idx.needs_verify(7)

    def test_a_completed_load_counts_as_verified(self, clock):
        idx = _index(7)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert idx.request_load("r", 7)
        idx.complete_load("r")
        assert not idx.needs_verify(7)


# ── the connector: touches out in the metadata, verdicts back by quorum ────


def _k3_worker(tier):
    from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
        KimiK3OffloadConnector,
    )

    c = KimiK3OffloadConnector.__new__(KimiK3OffloadConnector)
    c._state_tier = tier
    c._touch_verdicts = {}
    return c


class TestConnectorTouches:
    def test_the_worker_touches_and_records_each_verdict(self):
        touched = []

        def touch(h):
            touched.append(h)
            return h != 2

        worker = _k3_worker(SimpleNamespace(codec=SimpleNamespace(touch=touch)))
        ids = [StateTouchId(1, 0), StateTouchId(2, 1)]
        worker._touch_states(SimpleNamespace(state_touches=ids))
        assert touched == [1, 2]
        assert worker._touch_verdicts == {ids[0]: True, ids[1]: False}

    def test_with_no_tier_every_image_is_gone(self):
        worker = _k3_worker(None)
        worker._touch_states(SimpleNamespace(state_touches=[StateTouchId(1, 0)]))
        assert worker._touch_verdicts == {StateTouchId(1, 0): False}

    def test_a_failing_touch_does_not_erase_the_index(self):
        def touch(h):
            raise RuntimeError("backend hiccup")

        worker = _k3_worker(SimpleNamespace(codec=SimpleNamespace(touch=touch)))
        worker._touch_states(SimpleNamespace(state_touches=[StateTouchId(1, 0)]))
        assert worker._touch_verdicts == {StateTouchId(1, 0): True}

    def _scheduler(self, monkeypatch):
        from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
        from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
            KimiK3OffloadScheduler,
        )
        from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata

        monkeypatch.setattr(
            DenseOffloadScheduler,
            "build_connector_meta",
            lambda self: LMCacheOffloadMetadata(),
        )
        s = KimiK3OffloadScheduler.__new__(KimiK3OffloadScheduler)
        s._pending_state_loads = []
        s._pending_state_stores = []
        s._pending_state_touches = []
        s._touch_round = 0
        s._state_touch_verdicts = {}
        s._refresh_save_stall = lambda: None
        return s

    def test_each_touch_is_a_distinct_identity_drained_once(self, monkeypatch):
        s = self._scheduler(monkeypatch)
        assert s.enqueue_state_touches([]) is False
        assert s.enqueue_state_touches([5, 5])
        meta = s.build_connector_meta()
        assert meta.state_touches == [StateTouchId(5, 0), StateTouchId(5, 1)]
        assert s.build_connector_meta().state_touches == []

    def test_one_rank_missing_the_image_makes_it_gone(self, monkeypatch):
        from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
            STATE_TOUCH_CHANNEL,
        )

        s = self._scheduler(monkeypatch)
        for op, ok in ((StateTouchId(5, 0), True), (StateTouchId(5, 1), False)):
            handled = s.connector_completion(
                ConnectorCompletion(STATE_TOUCH_CHANNEL, op, ok)
            )
            assert handled is None, "handled, but not a terminal save"
        s.connector_completion(
            ConnectorCompletion(STATE_TOUCH_CHANNEL, StateTouchId(6, 2), True)
        )
        assert s.take_state_touch_verdicts() == {5: False, 6: True}
        assert s.take_state_touch_verdicts() == {}

    def test_a_negative_round_is_refused(self):
        with pytest.raises(ValueError):
            StateTouchId(1, -1)


# ── the engine: touches published, verdicts applied, admission held ────────


class TestSchedulerPlumbing:
    def _sched(self, connector, idx):
        from atom.model_engine.scheduler import Scheduler

        s = Scheduler.__new__(Scheduler)
        s.kv_connector = connector
        s.block_manager = SimpleNamespace(state_offload=idx)
        return s

    def test_voted_hashes_reach_the_connector(self, clock):
        idx = _index(7)
        idx.could_serve(7)
        sent = []
        s = self._sched(
            SimpleNamespace(enqueue_state_touches=lambda hs: sent.extend(hs) or True),
            idx,
        )
        s._publish_state_touches()
        assert sent == [7] and idx.refresh_hits

    def test_a_connector_without_touches_turns_the_refresh_off(self, clock):
        idx = _index(7)
        idx.could_serve(7)
        s = self._sched(SimpleNamespace(), idx)
        s._publish_state_touches()
        assert not idx.refresh_hits
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert not idx.needs_verify(7), "nothing would ever answer the hold"

    def test_verdicts_are_applied_to_the_index(self, clock):
        idx = _index(7, 8)
        s = self._sched(
            SimpleNamespace(take_state_touch_verdicts=lambda: {7: False, 8: True}),
            idx,
        )
        s._apply_state_touch_verdicts(idx)
        assert not idx.could_serve(7) and idx.could_serve(8)


def _paged_bm():
    from test_state_checkpoint import (
        PAGED_COPY_RUNTIME,
        ckpt_config,
        make_block_manager,
    )

    bm = make_block_manager(
        ckpt_config(num_kvcache_blocks=160, pool_entries={"state": 8}),
        state_runtime=PAGED_COPY_RUNTIME,
    )
    bm.state_offload = StateOffloadIndex(refresh_hits=True)
    return bm


def _seq():
    from test_state_checkpoint import stateful_seq

    return stateful_seq(list(range(40)))


class TestAdmissionHold:
    """`record_allocation(defer_unverified=True)` holds an admission that
    would load an unconfirmed image, and commits nothing while it does."""

    def _joint(self, bm, monkeypatch, *, h=77, from_hbm=False):
        from atom.model_engine.block_manager import JointBoundaryDecision

        decision = JointBoundaryDecision(
            boundary_tokens=16,
            boundary_hash=h,
            kv_tokens=16,
            claim_tokens=16,
            state_from_hbm=from_hbm,
        )
        commits = []
        monkeypatch.setattr(bm, "_joint_kv_boundary", lambda *a: decision)
        monkeypatch.setattr(
            bm, "_commit_joint_boundary", lambda seq, d: commits.append(d)
        )
        return commits

    def test_an_unconfirmed_joint_image_is_held(self, clock, monkeypatch):
        bm = _paged_bm()
        commits = self._joint(bm, monkeypatch)
        bm.state_offload.note_stored(77)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        seq = _seq()

        assert not bm.record_allocation(seq, 2, [1, 2], defer_unverified=True)
        assert commits == [] and bm.state_offload.loads_deferred_for_verify == 1
        assert bm.state_offload.take_touches() == [77]

        bm.state_offload.apply_touch_verdict(77, True)
        assert bm.record_allocation(seq, 2, [1, 2], defer_unverified=True)
        assert len(commits) == 1

    def test_a_gone_image_is_not_parked_on(self, clock, monkeypatch):
        """The verdict drops the hash; the retry's boundary search no longer
        sees the tier rung (`could_serve` is False), so it plans a lower one."""
        bm = _paged_bm()
        self._joint(bm, monkeypatch)
        bm.state_offload.note_stored(77)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert not bm.record_allocation(_seq(), 2, [1, 2], defer_unverified=True)
        bm.state_offload.take_touches()
        bm.state_offload.apply_touch_verdict(77, False)
        assert not bm._tier_can_serve(77)

    def test_a_recently_confirmed_image_goes_straight_through(self, clock, monkeypatch):
        bm = _paged_bm()
        commits = self._joint(bm, monkeypatch)
        bm.state_offload.note_stored(77)
        assert bm.record_allocation(_seq(), 2, [1, 2], defer_unverified=True)
        assert len(commits) == 1

    def test_a_state_resident_in_hbm_is_never_held(self, clock, monkeypatch):
        bm = _paged_bm()
        self._joint(bm, monkeypatch, from_hbm=True)
        bm.state_offload.note_stored(77)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert bm.record_allocation(_seq(), 2, [1, 2], defer_unverified=True)

    def test_the_fit_probe_path_never_holds(self, clock, monkeypatch):
        bm = _paged_bm()
        self._joint(bm, monkeypatch)
        bm.state_offload.note_stored(77)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        assert bm.record_allocation(_seq(), 2, [1, 2])

    def test_an_hbm_hit_holds_only_for_a_tier_sourced_state(self, clock, monkeypatch):
        from atom.model_engine.block_manager import JointBoundaryDecision

        bm = _paged_bm()
        monkeypatch.setattr(
            bm,
            "_joint_kv_boundary",
            lambda *a: JointBoundaryDecision(skip_reason="no_kv_reach"),
        )
        monkeypatch.setattr(bm, "_commit_joint_boundary", lambda seq, d: None)
        bm.state_offload.note_stored(2)
        clock[0] += so._TIER_VERIFIED_FOR_S + 1
        resident = {2}
        monkeypatch.setattr(
            bm.paged_state_checkpoints, "contains", lambda h: h in resident
        )
        assert bm.record_allocation(_seq(), 2, [1, 2], defer_unverified=True)
        resident.clear()
        assert not bm.record_allocation(_seq(), 2, [1, 2], defer_unverified=True)


def test_the_scheduler_retries_a_held_admission(monkeypatch):
    """A held seq stays waiting and is admitted on a later pass."""
    from test_state_checkpoint import ckpt_config, make_scheduler, stateful_seq

    sched = make_scheduler(ckpt_config())
    bm = sched.block_manager
    real = bm.record_allocation
    holds = [True]

    def record(seq, n, hashes, *, defer_unverified=False):
        assert defer_unverified, "admission asks to hold unverified loads"
        if holds.pop() if holds else False:
            return False
        return real(seq, n, hashes, defer_unverified=defer_unverified)

    monkeypatch.setattr(bm, "record_allocation", record)
    seq = stateful_seq(list(range(12)))
    sched.add(seq)
    _, scheduled = sched.schedule()
    assert scheduled == {}
    assert list(sched.waiting) == [seq] and len(seq.block_table) == 0
    _, scheduled = sched.schedule()
    assert seq.id in scheduled and not sched.waiting


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
