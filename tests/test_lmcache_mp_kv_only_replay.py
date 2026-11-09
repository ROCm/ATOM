# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache MP native-state loads that bring the PAGE KV alone, for a replay.

With a bounded replay the native-state layout stores no state image
(`stores_state_images`): it registers PAGE alone, so a stock
LMCache server's hit is how far the PAGE KV reaches. No image ends it, so the
state-carrying hit is 0 and the reach is the replay's (`Sequence.replay_kv_load`).
These tests drive that lookup through its whole life: answered, weighed, turned
into a load with no image or released, closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from weakref import WeakKeyDictionary

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    LoadOperationId,
)
from atom.kv_transfer.offload.mp import deployment
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp.native_state_scheduler import (
    NativeStateLMCacheMPConnectorScheduler,
)
from atom.kv_transfer.offload.mp.native_state_worker import (
    NATIVE_STATE_MP_STORE_CHANNEL,
)
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
)
from atom.model_engine.sequence import Sequence

CHUNK = 8
REPLAY = 6
SESSION = "atom-offload-dp0:1"


@dataclass(frozen=True)
class _Key:
    token_ids: tuple
    start: int
    end: int
    request_id: str
    worker_id: int | None
    request_configs: dict | None = field(default=None, compare=False)


class _Future:
    def __init__(self, value=None):
        self.value = value

    def query(self):
        return True

    def result(self, timeout=None):
        return self.value


class _Client:
    """A stock LMCache server: answers its hit, `chunks` chunks.

    With the PAGE-only layout that is how far the PAGE KV reaches; with images,
    the longest prefix with an image at its end.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.keys = []

    def lookup(self, key, tp_size):
        self.keys.append(key)
        return _Future()

    def query_prefetch_status(self, request_id):
        return _Future(self.chunks)


class _Adapter:
    lmcache_tokens_per_chunk = CHUNK

    def __init__(self, client):
        self._client = client
        self._parallel = SimpleNamespace(tp_size=1)
        self._pending_lookups: set[str] = set()
        self._lookup_results: dict[str, int] = {}
        self.freed = []
        self.ended = []

    def _create_key(self, token_ids, start, end, request_id, worker_id):
        return _Key(tuple(token_ids[start:end]), start, end, request_id, worker_id)

    def maybe_submit_lookup_request(self, request_id, token_ids):
        # A lookup with no asynchronous answer waiting: the same hit, in tokens.
        self._client.keys.append(
            _Key(tuple(token_ids), 0, len(token_ids), request_id, None)
        )
        self._lookup_results[request_id] = self._client.chunks * CHUNK

    def check_lookup_result(self, request_id):
        return self._lookup_results.get(request_id)

    def free_lookup_locks(self, **kwargs):
        self.freed.append(kwargs)

    def cleanup_lookup_result(self, request_id):
        self._pending_lookups.discard(request_id)
        self._lookup_results.pop(request_id, None)

    def end_session(self, request_id):
        self.ended.append(request_id)

    def shutdown(self):
        pass


def _freed(adapter):
    """`(start, end)` per lock release, in order."""
    return [(f["start"], f["end"]) for f in adapter.freed]


def _scheduler(monkeypatch, *, chunks, replay=REPLAY, min_load=0):
    """`replay`: the engine's `Config.state_replay_tokens`; with one the layout
    stores no images, without one (0) it does."""
    monkeypatch.setenv("OFFLOAD_MIN_SAVE_TOKENS", "0")
    monkeypatch.setattr(
        deployment.offcfg,
        "build_lmcache_config",
        lambda kvc: SimpleNamespace(
            chunk_size=kvc["kv_connector_extra_config"]["lmcache.chunk_size"]
        ),
    )
    adapter = _Adapter(_Client(chunks))

    def connect(config, *, checkpoint_spec=None, images=True):
        adapter.images = images  # the namespace this scheduler registers under
        return adapter

    monkeypatch.setattr(mp_scheduler, "_make_scheduler_adapter", connect)
    config = SimpleNamespace(
        kv_cache_block_size=4,
        kv_transfer_config={
            "kv_role": "offload",
            "kv_connector_extra_config": {"lmcache.chunk_size": CHUNK},
        },
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        state_replay_tokens=replay,  # what `Config.__post_init__` sets
    )
    scheduler = NativeStateLMCacheMPConnectorScheduler(config)
    checkpoints = PagedStateCheckpointCoordinator(
        BlockPool(30),
        PagedStateCheckpointSpec(10, 50, "kv-only-test", image_bytes=25),
        enabled=True,
    )
    fallbacks = []
    manager = SimpleNamespace(
        paged_state_checkpoints=checkpoints,
        enable_prefix_caching=True,
        hash_block_size=4,
        compute_hash=BlockManager.compute_hash,
        state_replay_tokens=replay,
        fall_back_to_hbm_replay=fallbacks.append,
    )
    manager._hash_block_tokens = lambda seq, index: BlockManager._hash_block_tokens(
        manager, seq, index
    )
    manager._prompt_hash_chains = WeakKeyDictionary()
    manager._prompt_hashes = lambda seq: BlockManager._prompt_hashes(manager, seq)
    manager._adopt_shipped_chain = lambda seq: BlockManager._adopt_shipped_chain(
        manager, seq
    )
    manager._block_hash = lambda seq, chain, i, prev: BlockManager._block_hash(
        manager, seq, chain, i, prev
    )
    manager.prefix_hash_chain = lambda seq, hashes, blocks: BlockManager._chain_to(
        manager, seq, hashes, blocks
    )
    scheduler.bind_block_manager(manager)
    scheduler._min_load_tokens = min_load
    return scheduler, checkpoints, adapter, fallbacks


def _seq(count=40):
    seq = Sequence(list(range(count)), 4, id=1, has_per_req_cache=True)
    seq.state_slots = [5]
    seq.block_table = list(range((count + 3) // 4))
    return seq


def _look_up(scheduler, seq):
    scheduler.prefetch_lookups(iter([seq]))  # the scheduler passes an iterator
    assert not scheduler.lookup_pending(seq)
    return scheduler.get_num_new_matched_tokens(seq)


def _admit_kv_only(scheduler, seq, *, hbm):
    """What `BlockManager._replay_hit` and `allocate` leave for a KV-only load:
    the hit replays, past the HBM hit, and the claim stops at the HBM hit."""
    _look_up(scheduler, seq)
    reach = seq.offload_joint.kv_only_tokens
    seq.num_cached_tokens = hbm
    seq.replays_hit = seq.replay_kv_load = True
    return reach


def test_without_images_the_server_hit_is_the_kv_only_reach(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=4)
    assert adapter.images is False
    seq = _seq()
    # No image ends the PAGE KV, so nothing is matched as a state-carrying hit.
    assert _look_up(scheduler, seq) == (0, False)
    assert seq.offload_joint.kv_only_tokens == 4 * CHUNK
    # One lookup, with no option a stock server would not know.
    [key] = adapter._client.keys
    assert key.request_id == SESSION
    assert not key.request_configs


def test_with_images_the_server_hit_carries_one(monkeypatch):
    # No replay, so images: the hit is the image's, there is no KV-only
    # reach, and the hit loads as before.
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=2, replay=0)
    assert adapter.images is True
    seq = _seq()
    assert _look_up(scheduler, seq) == (2 * CHUNK, True)
    assert seq.offload_joint.kv_only_tokens == 0


def test_a_kv_only_load_reads_page_alone_and_restores_nothing(monkeypatch):
    scheduler, checkpoints, adapter, fallbacks = _scheduler(monkeypatch, chunks=4)
    seq = _seq()
    reach = _admit_kv_only(scheduler, seq, hbm=8)
    assert reach == 32
    free_before = checkpoints.store.pool.num_free
    scheduler.update_state_after_alloc(seq)
    # The lookup's locks back the load: nothing went back at admission.
    assert adapter.freed == []
    assert scheduler.should_park_for_load_after_alloc(seq)
    assert fallbacks == []
    # No image units are reserved for it.
    assert checkpoints.store.pool.num_free == free_before
    [request] = scheduler.build_connector_meta().requests
    assert request.native_state.kv_only
    assert request.native_state.unit_ids == ()
    assert request.native_state.boundary_tokens == reach
    assert request.load_spec.hbm_cached_tokens == 8
    assert request.load_spec.lmcache_cached_tokens == reach
    # HBM covers [0, 8); the retrieve reads and releases PAGE [8, 32).
    assert _freed(adapter) == [(0, 8)]
    operation = seq._load_operation
    assert isinstance(operation, LoadOperationId)
    assert scheduler.load_finished(operation) is True
    # Nothing was adopted as a checkpoint: there was no image.
    assert checkpoints.store.pool.num_free == free_before
    scheduler.request_finished(seq)
    assert len(adapter.freed) == 1


def test_a_refused_kv_only_load_falls_back_and_returns_its_locks(monkeypatch):
    scheduler, _, adapter, fallbacks = _scheduler(
        monkeypatch, chunks=4, min_load=10_000
    )
    seq = _seq()
    _admit_kv_only(scheduler, seq, hbm=8)
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    assert fallbacks == [seq]
    assert _freed(adapter) == [(0, 32)]


def test_an_unused_reach_returns_its_locks_at_admission(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=4)
    seq = _seq()
    _look_up(scheduler, seq)
    assert adapter.freed == []
    seq.num_cached_tokens = 0  # an ordinary admission: no replay chosen
    scheduler.update_state_after_alloc(seq)
    assert _freed(adapter) == [(0, 32)]


def test_the_reach_stays_locked_until_admission_weighs_it(monkeypatch):
    # HBM already holds part of it, so no load is armed and the base would
    # release the lookup at once; the PAGE KV past HBM is exactly what a replay
    # can still use, so its locks stay until admission.
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=4)
    seq = _seq()
    seq.num_cached_tokens = CHUNK
    assert _look_up(scheduler, seq) == (0, False)
    assert seq.offload_joint.kv_only_tokens == 4 * CHUNK
    assert adapter.freed == []
    scheduler.update_state_after_alloc(seq)
    assert _freed(adapter) == [(0, 32)]


def test_a_replay_from_hbm_never_loads(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=4)
    seq = _seq()
    _look_up(scheduler, seq)
    seq.num_cached_tokens = 16  # the HBM hit the replay ends at
    seq.replays_hit = True
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    assert scheduler.build_connector_meta().requests == []
    assert _freed(adapter) == [(0, 32)]


def test_asking_again_first_returns_the_earlier_locks(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=4)
    seq = _seq()
    _look_up(scheduler, seq)
    assert adapter.freed == []
    # A fresh lookup for the same request (its remembered answer ran out, say)
    # would replace the server's record of what the first one locked.
    scheduler._fresh_tier_lookup(seq, str(seq.id))
    assert _freed(adapter) == [(0, 32)]
    assert len(adapter._client.keys) == 2


def _admit_on_a_remembered_hit(scheduler, seq):
    """Asked in one step, admitted in a later one on the remembered answer."""
    assert _look_up(scheduler, seq) == (0, False)
    # Not admitted this step (no room, say).
    assert scheduler.build_connector_meta().requests == []
    return _admit_kv_only(scheduler, seq, hbm=8)


def test_a_kv_only_load_admitted_on_a_remembered_hit_still_loads(monkeypatch):
    scheduler, _, adapter, fallbacks = _scheduler(monkeypatch, chunks=4)
    seq = _seq()
    assert _admit_on_a_remembered_hit(scheduler, seq) == 32
    assert len(adapter._client.keys) == 1  # answered from memory, not asked again
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    # The dispatch confirms the reach its lookup still holds; asking again
    # would weigh a state-carrying hit of 0 and drop the load.
    [request] = scheduler.build_connector_meta().requests
    assert request.native_state.kv_only
    assert request.load_spec.lmcache_cached_tokens == 32
    assert _freed(adapter) == [(0, 8)]
    assert fallbacks == []
    output = scheduler.process_completions(KVConnectorOutput())
    assert output.failed_loading == set()


def test_a_parked_load_dropped_at_dispatch_wakes_its_request(monkeypatch):
    scheduler, _, _, _ = _scheduler(monkeypatch, chunks=4)
    seq = _seq()
    _admit_on_a_remembered_hit(scheduler, seq)
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    # The lookup no longer holds the reach the load was aimed at.
    monkeypatch.setattr(scheduler._lookup_client, "hit_tokens", lambda _id: 16)
    assert scheduler.build_connector_meta().requests == []
    assert scheduler._native_loads == {}
    # Parked, with no transfer left to report: woken as a failed load, which
    # the scheduler turns into a replay below the HBM hit.
    output = scheduler.process_completions(KVConnectorOutput())
    assert output.failed_loading == {seq.id}
    assert scheduler.process_completions(KVConnectorOutput()).failed_loading == set()


def test_an_abort_while_waiting_returns_the_reach(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=4)
    seq = _seq()
    _look_up(scheduler, seq)
    scheduler.request_finished(seq)
    assert (0, 32) in _freed(adapter)


# -- Images or PAGE alone, by the replay ------------------------------------


def test_a_replay_stores_no_images(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=0)
    assert scheduler._page_only
    assert adapter.images is False


def test_no_replay_keeps_the_images(monkeypatch):
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=0, replay=0)
    assert not scheduler._page_only
    assert adapter.images is True


def test_a_page_only_save_stores_page_alone_with_no_checkpoint(monkeypatch):
    scheduler, checkpoints, _, _ = _scheduler(monkeypatch, chunks=0)
    seq = _seq()
    seq.num_cached_tokens = 36  # computed so far; no checkpoint anywhere
    free = checkpoints.store.pool.num_free
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    # Up to the computed chunk boundary, with no image waited for or pinned.
    assert request.native_state.kv_only
    assert request.native_state.unit_ids == ()
    assert request.native_state.boundary_tokens == 32
    assert request.token_ids == list(seq.token_ids[:32])
    assert request.save_spec.skip_leading_tokens == 0
    assert checkpoints.store.pool.num_free == free
    operation = request.save_operation
    scheduler.process_completions(
        KVConnectorOutput(
            finished_saving={operation},
            connector_completions={
                ConnectorCompletion(NATIVE_STATE_MP_STORE_CHANNEL, operation, True)
            },
        )
    )
    assert scheduler._native_saves == {}
    assert str(seq.id) not in scheduler._save_inflight
    assert checkpoints.store.pool.num_free == free


def test_page_only_never_restores_an_image(monkeypatch):
    scheduler, checkpoints, _, _ = _scheduler(monkeypatch, chunks=2)
    seq = _seq()
    _look_up(scheduler, seq)
    seq.num_cached_tokens = 0  # no replay chosen: nothing to load the KV for
    free = checkpoints.store.pool.num_free
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    assert scheduler._native_loads == {}
    assert checkpoints.store.pool.num_free == free


def test_a_page_only_save_starts_at_the_page_reach(monkeypatch):
    # The base saves from the state-carrying hit, 0 here; that would store the
    # whole prompt again. LMCache holds the PAGE KV to the reach.
    scheduler, _, adapter, _ = _scheduler(monkeypatch, chunks=3)
    seq = _seq()
    _look_up(scheduler, seq)
    assert seq.offload_joint.kv_only_tokens == 24
    seq.num_cached_tokens = 36  # HBM covers the reach: nothing loads
    scheduler.update_state_after_alloc(seq)
    assert not scheduler.should_park_for_load_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert request.load_spec is None
    assert request.save_spec.skip_leading_tokens == 24
    assert request.native_state.boundary_tokens == 32
    # The lookup's locks, to the reach, went back at admission.
    assert (0, 24) in _freed(adapter)


def test_a_page_only_readmission_keeps_the_earlier_reach_as_its_floor(monkeypatch):
    # A request admitted on a remembered answer asks nothing new.
    scheduler, _, _, _ = _scheduler(monkeypatch, chunks=0)
    seq = _seq()
    seq._mp_page_reach = 16
    seq.num_cached_tokens = 36
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert request.save_spec.skip_leading_tokens == 16
