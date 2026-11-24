# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_OFFLOAD_PREHASHED_SAVE: scheduler-hashed LMCache saves for M3.

The keys a prehashed save stores must be exactly the keys LMCache's own
token path stores and looks up, so these tests drive the real
``ChunkedTokenDatabase`` and compare against ``process_tokens``.
"""

import array
import random
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lmcache")

from lmcache.v1.token_database import ChunkedTokenDatabase  # noqa: E402

from atom.kv_transfer.disaggregation.types import SaveOperationId  # noqa: E402
from atom.kv_transfer.offload import config as offcfg  # noqa: E402
from atom.kv_transfer.offload.dense.connector import (  # noqa: E402
    DenseOffloadConnector,
)
from atom.kv_transfer.offload.hybrid.m3.connector import (  # noqa: E402
    M3OffloadConnector,
    M3OffloadScheduler,
)
from atom.kv_transfer.offload.metadata import (  # noqa: E402
    LMCacheReqMeta,
    PrehashedSave,
    SaveSpec,
)
from atom.kv_transfer.offload._offload_common import tokens_to_tensor  # noqa: E402

CHUNK = 8


def _tdb():
    cfg = SimpleNamespace(
        chunk_size=CHUNK,
        save_unfull_chunk=False,
        pre_caching_hash_algorithm="builtin",
        remote_url=None,
        enable_pd=False,
        get_extra_config_value=lambda _name, default=None: default,
    )
    meta = SimpleNamespace(
        model_name="m3",
        world_size=4,
        worker_id=1,
        kv_dtype=torch.uint8,
        use_mla=False,
        is_first_rank=lambda: False,
    )
    return ChunkedTokenDatabase(cfg, meta)


class _Transport:
    world_size = 4

    def __init__(self, hit=0):
        self.messages = []
        self.hit = hit

    def send_and_recv_all(self, message):
        self.messages.append(message)
        return [int(self.hit).to_bytes(4, "big")] * self.world_size


class _LookupClient:
    def __init__(self, hit=0):
        self.token_database = _tdb()
        self.transport = _Transport(hit)
        self.enable_blending = False
        self.reqs_status = {}

    def lookup(self, token_ids, lookup_id):  # the legacy path, for comparison
        hashes, offsets = [], []
        for start, end, h in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(h)
            offsets.append(end - start)
        if not hashes:
            return 0
        responses = self.transport.send_and_recv_all([hashes, offsets, lookup_id, ""])
        return min(int.from_bytes(r, "big") for r in responses)


def _scheduler(monkeypatch, *, enabled=True, hit=0):
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_SAVE", "1" if enabled else "0")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _c=None: SimpleNamespace(chunk_size=CHUNK),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_a: object())
    monkeypatch.setattr(offcfg, "lmcache_replica_world_size", lambda _c: 4)
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "offload"},
        kv_cache_block_size=4,
        decode_context_parallel_size=1,
        tensor_parallel_size=4,
    )
    scheduler = M3OffloadScheduler(config)
    scheduler._lookup_client = _LookupClient(hit)
    scheduler._init_prehashed_save()
    return scheduler


def _seq(n_tokens, seq_id=7):
    rng = random.Random(seq_id)
    tokens = [rng.randrange(200000) for _ in range(n_tokens)]
    return SimpleNamespace(
        id=seq_id,
        token_ids=array.array("i", tokens),
        num_prompt_tokens=n_tokens,
        num_cached_tokens=0,
        block_table=list(range(n_tokens // 4 + 1)),
    )


def test_chain_matches_lmcache_process_tokens(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    assert scheduler._prehashed
    seq = _seq(10 * CHUNK + 3)
    chain = scheduler._prefix_chunk_hashes(seq, 10)
    tdb = scheduler._lookup_client.token_database
    expected = [
        h for _s, _e, h in tdb.process_tokens(list(seq.token_ids), make_key=False)
    ]
    assert chain[:10] == expected
    # Extending reuses the chain object and hashes only what is missing.
    again = scheduler._prefix_chunk_hashes(seq, 4)
    assert again is chain


def test_lookup_sends_lmcache_message(monkeypatch):
    scheduler = _scheduler(monkeypatch, hit=3 * CHUNK)
    seq = _seq(5 * CHUNK + 1)
    hit = scheduler._tier_lookup(seq, "7")
    assert hit == 3 * CHUNK
    sent = scheduler._lookup_client.transport.messages[-1]
    legacy = _LookupClient(hit=3 * CHUNK)
    legacy.lookup(list(seq.token_ids), "7")
    assert sent == legacy.transport.messages[-1]
    assert scheduler._lookup_client.reqs_status["7"] == 3 * CHUNK


def test_save_request_ships_new_chunk_hashes(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    seq = _seq(6 * CHUNK)
    op = SaveOperationId(seq.id, 0)
    req = scheduler._build_save_request(
        seq, 2 * CHUNK, 5 * CHUNK, op, list(seq.block_table), False
    )
    assert req.token_ids == []
    assert isinstance(req.prehashed_save, PrehashedSave)
    assert req.prehashed_save.start == 2 * CHUNK
    tdb = scheduler._lookup_client.token_database
    expected = [
        h
        for s, _e, h in tdb.process_tokens(
            list(seq.token_ids[: 5 * CHUNK]), make_key=False
        )
        if s >= 2 * CHUNK
    ]
    assert list(req.prehashed_save.hashes) == expected
    assert req.save_spec.skip_leading_tokens == 2 * CHUNK


def test_flag_off_is_the_token_path(monkeypatch):
    scheduler = _scheduler(monkeypatch, enabled=False)
    assert not scheduler._prehashed
    seq = _seq(4 * CHUNK)
    req = scheduler._build_save_request(
        seq, 0, 4 * CHUNK, SaveOperationId(seq.id, 0), [0, 1, 2, 3, 4, 5, 6, 7], True
    )
    assert req.prehashed_save is None
    assert req.token_ids == list(seq.token_ids)


def test_mismatched_chain_disables(monkeypatch):
    from atom.kv_transfer.offload import prehashed

    scheduler = _scheduler(monkeypatch)
    real = prehashed._chunk_list
    # Corrupt only the per-chunk slices the chain hashes, not the whole-prefix
    # slice the one-time check hands to LMCache's `process_tokens`.
    monkeypatch.setattr(
        prehashed,
        "_chunk_list",
        lambda t, s, e: (
            real(t, s, e) if e - s > CHUNK else [x + 1 for x in real(t, s, e)]
        ),
    )
    seq = _seq(3 * CHUNK)
    assert scheduler._prefix_chunk_hashes(seq, 3) is None
    assert not scheduler._prehashed
    req = scheduler._build_save_request(
        seq, 0, 3 * CHUNK, SaveOperationId(seq.id, 0), [0] * 6, True
    )
    assert req.prehashed_save is None


# ---------------------------------------------------------------- worker side
class _MemoryObj:
    def __init__(self):
        self.refs = 1

    def ref_count_down(self):
        self.refs -= 1


class _StorageManager:
    def __init__(self, capacity=1 << 30):
        self.capacity = capacity
        self.allocated = []
        self.put = []

    def allocate(self, shapes, dtypes, busy_loop=True, fmt=None):
        if len(self.allocated) >= self.capacity:
            return None
        obj = _MemoryObj()
        self.allocated.append(obj)
        return obj

    def batched_put(self, keys, memory_objs, transfer_spec=None, location=None):
        self.put.append((list(keys), list(memory_objs)))

    def contains(self, key, search_range=None, pin=False):
        """LMCache's: the backend that holds `key`, or None."""
        held = any(key in keys for keys, _objs in self.put)
        return "LocalCPUBackend" if held else None


class _GpuConnector:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        if self.fail:
            raise RuntimeError("pack failed")
        self.calls.append((list(starts), list(ends), kwargs))


def _engine(capacity=1 << 30, fail=False):
    return SimpleNamespace(
        token_database=_tdb(),
        storage_manager=_StorageManager(capacity),
        gpu_connector=_GpuConnector(fail),
        metadata=SimpleNamespace(
            get_shapes=lambda n: [torch.Size((n,))], get_dtypes=lambda: [torch.uint8]
        ),
        config=SimpleNamespace(get_extra_config_value=lambda _n, d=None: d),
        fmt=None,
        store_location=None,
        is_healthy=lambda: True,
        is_frozen=lambda: False,
        _is_passive=lambda: False,
    )


def _worker(engine):
    worker = M3OffloadConnector.__new__(M3OffloadConnector)
    worker._init_worker_common(
        SimpleNamespace(kv_transfer_config={"kv_role": "offload"})
    )
    worker.chunk_size = CHUNK
    worker._engine = engine
    worker._early_release = False
    worker._rank = 0
    return worker


def _legacy_keys(tdb, tokens, skip, end):
    mask = torch.ones(end, dtype=torch.bool)
    mask[:skip] = False
    return [
        (s, e, k)
        for s, e, k in tdb.process_tokens(tokens_to_tensor(tokens[:end]), mask=mask)
    ]


def test_worker_stores_the_token_path_keys():
    engine = _engine()
    worker = _worker(engine)
    tokens = list(range(1000, 1000 + 6 * CHUNK))
    tdb = engine.token_database
    hashes = [h for _s, _e, h in tdb.process_tokens(tokens, make_key=False)]
    req = LMCacheReqMeta(
        req_id=3,
        token_ids=[],
        block_ids=list(range(12)),
        save_spec=SaveSpec(skip_leading_tokens=2 * CHUNK),
        is_last_prefill=False,
        save_operation=SaveOperationId(3, 0),
        prehashed_save=PrehashedSave(start=2 * CHUNK, hashes=tuple(hashes[2:5])),
    )
    worker._do_save_req(req, producer_event=None)
    keys, objs = engine.storage_manager.put[-1]
    legacy = _legacy_keys(tdb, tokens, 2 * CHUNK, 5 * CHUNK)
    assert keys == [k for _s, _e, k in legacy]
    starts, ends, kwargs = engine.gpu_connector.calls[-1]
    assert starts == [s for s, _e, _k in legacy]
    assert ends == [e for _s, e, _k in legacy]
    assert kwargs["block_ids"] == list(range(12))
    assert worker.get_finished().finished_saving == {SaveOperationId(3, 0)}


def test_worker_stops_at_first_refused_allocation():
    engine = _engine(capacity=1)
    worker = _worker(engine)
    req = LMCacheReqMeta(
        req_id=4,
        token_ids=[],
        block_ids=list(range(8)),
        save_spec=SaveSpec(skip_leading_tokens=0),
        prehashed_save=PrehashedSave(start=0, hashes=(11, 12, 13)),
    )
    worker._do_save_req(req)
    keys, _objs = engine.storage_manager.put[-1]
    assert [k.chunk_hash for k in keys] == [11]


def _store_all(req_id, hashes):
    """A lazy demotion's save: chunk i is block i, all of it or a failure."""
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=list(range(len(hashes))),
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(req_id, 0),
        prehashed_save=PrehashedSave(
            start=0, hashes=tuple(hashes), must_store_all=True
        ),
    )


def _store_reports(worker):
    from atom.kv_transfer.offload.chunked_scheduler import DENSE_PAGE_STORE_CHANNEL

    return {
        (c.operation_id.req_id, c.succeeded)
        for c in worker.get_finished().connector_completions
        if c.channel == DENSE_PAGE_STORE_CHANNEL
    }


@pytest.mark.parametrize("must_store_all", [False, True])
def test_a_refused_chunk_fails_only_a_save_that_must_store_all(must_store_all):
    """The tier refusing the second of three chunks: an eager save reports the
    prefix it stored as done, as `LMCacheEngine.store` does; a demotion, whose
    blocks a success marks saved, reports a failure (its stored chunk stays)."""
    engine = _engine(capacity=1)
    worker = _worker(engine)
    worker._early_release = True  # as the M3 worker runs
    req = _store_all("lmcache-lazy-demotion-0", (11, 12, 13))
    req.prehashed_save = PrehashedSave(
        start=0, hashes=(11, 12, 13), must_store_all=must_store_all
    )
    worker._do_save_req(req)
    keys, _objs = engine.storage_manager.put[-1]
    assert [k.chunk_hash for k in keys] == [11]
    assert _store_reports(worker) == {("lmcache-lazy-demotion-0", not must_store_all)}


def test_a_save_that_must_store_all_skips_chunks_the_tier_holds():
    engine = _engine(capacity=2)
    worker = _worker(engine)
    worker._early_release = True
    worker._do_save_req(_store_all("a", (11,)))
    # 11 is held, so the one allocation left goes to 12, the second chunk:
    # block 1.
    worker._do_save_req(_store_all("b", (11, 12)))
    keys, _objs = engine.storage_manager.put[-1]
    assert [k.chunk_hash for k in keys] == [12]
    starts, _ends, _kwargs = engine.gpu_connector.calls[-1]
    assert starts == [CHUNK]
    assert _store_reports(worker) == {("a", True), ("b", True)}


@pytest.mark.parametrize("exclusive", ["0", "1"])
def test_a_lazy_exclusive_demotion_makes_the_chunks_the_tier_kept_recent(
    monkeypatch, exclusive
):
    """HBM is evicting its copy of a chunk the tier kept (cold since HBM loaded
    it): skipped as before, and, with exclusive tiers, counted as hit."""
    monkeypatch.setenv("ATOM_OFFLOAD_LAZY_EXCLUSIVE", exclusive)
    from atom.kv_transfer.offload import prehashed

    engine = _engine()
    worker = _worker(engine)
    worker._early_release = True
    calls = []
    monkeypatch.setattr(
        worker, "_tier_backends_do", lambda hashes, op: calls.append((hashes, op))
    )
    worker._do_save_req(_store_all("a", (11,)))
    worker._do_save_req(_store_all("b", (11, 12)))
    keys, _objs = engine.storage_manager.put[-1]
    assert [k.chunk_hash for k in keys] == [12]
    assert calls == ([([11], prehashed._make_recent)] if exclusive == "1" else [])
    assert _store_reports(worker) == {("a", True), ("b", True)}


@pytest.mark.parametrize("state", ["unhealthy", "frozen", "passive"])
def test_an_engine_that_stores_nothing_fails_a_save_that_must_store_all(state):
    engine = _engine()
    setattr(
        engine,
        {"unhealthy": "is_healthy", "frozen": "is_frozen", "passive": "_is_passive"}[
            state
        ],
        (lambda: False) if state == "unhealthy" else (lambda: True),
    )
    worker = _worker(engine)
    worker._early_release = True
    worker._do_save_req(_store_all("c", (31, 32)))
    assert engine.storage_manager.put == []
    assert _store_reports(worker) == {("c", False)}


def test_worker_releases_allocations_when_pack_fails():
    engine = _engine(fail=True)
    worker = _worker(engine)
    req = LMCacheReqMeta(
        req_id=5,
        token_ids=[],
        block_ids=list(range(8)),
        save_spec=SaveSpec(skip_leading_tokens=0),
        prehashed_save=PrehashedSave(start=0, hashes=(21, 22)),
    )
    worker._guard("save", worker._do_save_req, req)
    assert [o.refs for o in engine.storage_manager.allocated] == [0, 0]
    assert engine.storage_manager.put == []
    # `_guard` reports the failed save terminal.
    assert worker.get_finished().finished_saving


def test_worker_without_prehash_takes_token_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        DenseOffloadConnector,
        "_do_save_req",
        lambda self, req, producer_event=None: calls.append(req.req_id),
    )
    worker = _worker(_engine())
    req = LMCacheReqMeta(
        req_id=6,
        token_ids=list(range(CHUNK)),
        block_ids=[0, 1],
        save_spec=SaveSpec(skip_leading_tokens=0),
    )
    worker._do_save_req(req)
    assert calls == [6]


# ------------------------------------------------- ATOM_OFFLOAD_ASYNC_TIER_LOOKUP
def _async_scheduler(monkeypatch, hit=0):
    monkeypatch.setenv("ATOM_OFFLOAD_ASYNC_TIER_LOOKUP", "1")
    return _scheduler(monkeypatch, hit=hit)


def test_async_sync_lookup_runs_on_the_lookup_thread(monkeypatch):
    scheduler = _async_scheduler(monkeypatch, hit=2 * CHUNK)
    assert scheduler._lookup_pool is not None
    seq = _seq(4 * CHUNK)
    assert scheduler._tier_lookup(seq, "7") == 2 * CHUNK
    assert len(scheduler._lookup_client.transport.messages) == 1


def test_detach_when_hbm_covers_and_answer_lowers_floor(monkeypatch):
    scheduler = _async_scheduler(monkeypatch, hit=2 * CHUNK)
    seq = _seq(6 * CHUNK)
    seq.num_tokens = 6 * CHUNK
    hbm = 5 * CHUNK
    assert scheduler.can_detach_lookup(seq, hbm)
    assert scheduler.detach_lookup(seq, hbm)
    assert scheduler._hit_save_floors["7"] == 5 * CHUNK
    scheduler._lookup_pool.submit(lambda: None).result()  # the job has run
    # the lookup message is LMCache's, for the whole prompt
    sent = scheduler._lookup_client.transport.messages[-1]
    assert sent[1] == [CHUNK] * 6 and sent[2] == "7"
    scheduler.update_state_after_alloc(seq)
    assert scheduler._save_tracker["7"][1] == 5 * CHUNK
    meta = scheduler.build_connector_meta()
    assert "7" in meta.lookup_requests_in_step  # its pins are released
    assert scheduler._save_tracker["7"][1] == 2 * CHUNK  # tier holds less


def test_no_detach_when_a_load_could_help(monkeypatch):
    scheduler = _async_scheduler(monkeypatch)
    scheduler._min_load_tokens = 2 * CHUNK
    seq = _seq(6 * CHUNK)
    seq.num_tokens = 6 * CHUNK
    assert not scheduler.can_detach_lookup(seq, 2 * CHUNK)
    assert scheduler.can_detach_lookup(seq, 5 * CHUNK)


def test_async_flag_alone_is_ignored(monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_ASYNC_TIER_LOOKUP", "1")
    scheduler = _scheduler(monkeypatch, enabled=False)
    assert scheduler._lookup_pool is None
    seq = _seq(6 * CHUNK)
    seq.num_tokens = 6 * CHUNK
    # the base answer: the in-process client cannot detach
    assert not scheduler.can_detach_lookup(seq, 5 * CHUNK)
