# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler-hashed LMCache saves for the M3 offload layout.

The keys a prehashed save stores must be exactly the keys LMCache's own
token path stores and looks up, so these tests drive the real
``ChunkedTokenDatabase`` and compare against ``process_tokens``.
"""

import array
import json
import random
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lmcache")

from lmcache.v1.token_database import ChunkedTokenDatabase

from atom.kv_transfer.disaggregation.types import SaveOperationId
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import (
    CACHE_SALT_TAG,
    tokens_to_tensor,
)
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
)
from atom.kv_transfer.offload.hybrid.m3.connector import (
    M3OffloadConnector,
    M3OffloadScheduler,
)
from atom.kv_transfer.offload.lookup_client import as_lookup_client
from atom.kv_transfer.offload.metadata import (
    LMCacheReqMeta,
    PrehashedSave,
    SaveSpec,
)

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

    def lookup(self, token_ids, lookup_id, request_configs=None):
        """The legacy path, for comparison (`LMCacheLookupClient.lookup`)."""
        hashes, offsets = [], []
        for start, end, h in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(h)
            offsets.append(end - start)
        if not hashes:
            return 0
        configs = json.dumps(request_configs) if request_configs else ""
        responses = self.transport.send_and_recv_all(
            [hashes, offsets, lookup_id, configs]
        )
        return min(int.from_bytes(r, "big") for r in responses)


def _scheduler(monkeypatch, *, hash_seed="0", hit=0, kv_events=False):
    monkeypatch.setenv("PYTHONHASHSEED", hash_seed)
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _c=None: SimpleNamespace(chunk_size=CHUNK, enable_kv_events=kv_events),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_a: object())
    monkeypatch.setattr(offcfg, "lmcache_replica_world_size", lambda _c: 4)
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "offload"},
        kv_cache_block_size=4,
        decode_context_parallel_size=1,
        tensor_parallel_size=4,
        offload_min_load_tokens=8192,
    )
    scheduler = M3OffloadScheduler(config)
    # As the scheduler wraps LMCache's in-process client (`as_lookup_client`).
    scheduler._lookup_client = as_lookup_client(_LookupClient(hit))
    scheduler._init_prehashed_save()
    return scheduler


def _seq(n_tokens, seq_id=7, cache_seed=-1):
    rng = random.Random(seq_id)
    tokens = [rng.randrange(200000) for _ in range(n_tokens)]
    return SimpleNamespace(
        id=seq_id,
        token_ids=array.array("i", tokens),
        num_prompt_tokens=n_tokens,
        num_cached_tokens=0,
        block_table=list(range(n_tokens // 4 + 1)),
        cache_seed=cache_seed,
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


@pytest.mark.parametrize("cache_seed", [-1, 1234])
def test_lookup_sends_lmcache_message(monkeypatch, cache_seed):
    scheduler = _scheduler(monkeypatch, hit=3 * CHUNK)
    seq = _seq(5 * CHUNK + 1, cache_seed=cache_seed)
    assert scheduler._tier_lookup(seq, "7") == 3 * CHUNK
    sent = scheduler._lookup_client.transport.messages[-1]
    legacy = _LookupClient(hit=3 * CHUNK)
    salt = {CACHE_SALT_TAG: str(cache_seed)} if cache_seed >= 0 else None
    legacy.lookup(list(seq.token_ids), "7", request_configs=salt)
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
    assert req.cache_salt == ""
    salted = scheduler._build_save_request(
        _seq(6 * CHUNK, seq_id=8, cache_seed=99),
        0,
        2 * CHUNK,
        SaveOperationId(8, 0),
        list(seq.block_table),
        False,
    )
    assert salted.cache_salt == "99" and salted.prehashed_save is not None


def test_builtin_hash_without_seed_ships_tokens(monkeypatch):
    scheduler = _scheduler(monkeypatch, hash_seed="random")
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

    def get_size(self):
        return CHUNK


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


class _Fence:
    """A recorded producer fence: `wait_save_fence` synchronizes it."""

    def __init__(self):
        self.waited = 0

    def synchronize(self):
        self.waited += 1


class _GpuConnector:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        if self.fail:
            raise RuntimeError("pack failed")
        self.calls.append((list(starts), list(ends), kwargs))


class _StoreStats:
    process_tokens_time = from_gpu_time = put_time = 0.0

    def __getattr__(self, name):  # profile_process_tokens / _from_gpu / _put
        return nullcontext

    def time_to_store(self):
        return 1.0


class _StatsMonitor:
    def __init__(self):
        self.calls = []

    def on_store_request(self, tokens):
        self.calls.append(("request", tokens))
        return _StoreStats()

    def on_store_finished(self, stats, tokens):
        self.calls.append(("finished", tokens))


def _engine(capacity=1 << 30, fail=False):
    return SimpleNamespace(
        stats_monitor=_StatsMonitor(),
        kv_events_enabled=True,
        kv_events=[],
        _get_req_id=lambda kw: kw.get("req_id"),
        _log_kvcache_for_check=lambda **kw: None,
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


def _legacy_keys(tdb, tokens, skip, end, request_configs=None):
    mask = torch.ones(end, dtype=torch.bool)
    mask[:skip] = False
    return [
        (s, e, k)
        for s, e, k in tdb.process_tokens(
            tokens_to_tensor(tokens[:end]), mask=mask, request_configs=request_configs
        )
    ]


@pytest.mark.parametrize("salt", ["", "42"])
def test_worker_stores_the_token_path_keys(salt):
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
        cache_salt=salt,
        prehashed_save=PrehashedSave(start=2 * CHUNK, hashes=tuple(hashes[2:5])),
    )
    fence = _Fence()
    worker._do_save_req(req, producer_event=fence)
    assert fence.waited == 1
    keys, _objs = engine.storage_manager.put[-1]
    configs = {CACHE_SALT_TAG: salt} if salt else None
    legacy = _legacy_keys(tdb, tokens, 2 * CHUNK, 5 * CHUNK, configs)
    assert keys == [k for _s, _e, k in legacy]
    starts, ends, kwargs = engine.gpu_connector.calls[-1]
    assert starts == [s for s, _e, _k in legacy]
    assert ends == [e for _s, e, _k in legacy]
    assert kwargs["block_ids"] == list(range(12))
    assert kwargs["producer_event"] is fence
    assert worker.get_finished().finished_saving == {SaveOperationId(3, 0)}


def test_worker_save_without_fence_fails_closed():
    engine = _engine()
    worker = _worker(engine)
    req = LMCacheReqMeta(
        req_id=10,
        token_ids=[],
        block_ids=list(range(4)),
        save_spec=SaveSpec(skip_leading_tokens=0),
        prehashed_save=PrehashedSave(start=0, hashes=(1, 2)),
    )
    with pytest.raises(ValueError):
        worker._do_save_req(req, producer_event=None)
    assert engine.storage_manager.put == []


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
    worker._do_save_req(req, producer_event=_Fence())
    keys, _objs = engine.storage_manager.put[-1]
    assert [k.chunk_hash for k in keys] == [11]


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
    worker._guard("save", worker._do_save_req, req, producer_event=_Fence())
    assert [o.refs for o in engine.storage_manager.allocated] == [0, 0]
    assert engine.storage_manager.put == []
    # `_guard` reports the failed save terminal.
    assert worker.get_finished().finished_saving


def test_worker_without_prehash_takes_token_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        DenseOffloadConnector,
        "_do_save_req",
        lambda self, req, producer_event: calls.append(req.req_id),
    )
    worker = _worker(_engine())
    req = LMCacheReqMeta(
        req_id=6,
        token_ids=list(range(CHUNK)),
        block_ids=[0, 1],
        save_spec=SaveSpec(skip_leading_tokens=0),
    )
    worker._do_save_req(req, producer_event=_Fence())
    assert calls == [6]


def test_save_ships_new_chunk_tokens_only_with_kv_events(monkeypatch):
    seq = _seq(6 * CHUNK)
    op = SaveOperationId(seq.id, 0)
    for kv_events in (False, True):
        scheduler = _scheduler(monkeypatch, kv_events=kv_events)
        req = scheduler._build_save_request(
            seq, 2 * CHUNK, 5 * CHUNK, op, list(seq.block_table), False
        )
        want = tuple(seq.token_ids[2 * CHUNK : 5 * CHUNK]) if kv_events else None
        assert req.prehashed_save.token_ids == want


def _lmcache_store():
    """`LMCacheEngine.store` from the installed LMCache source, compiled against
    the names it reads (the module itself needs LMCache's native ops)."""
    import ast
    import importlib.util
    import logging
    import pathlib
    import typing

    import lmcache.utils as lmc_utils

    path = importlib.util.find_spec("lmcache.v1.cache_engine").origin
    tree = ast.parse(pathlib.Path(path).read_text())
    engine_cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheEngine"
    )
    store = next(
        n
        for n in engine_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "store"
    )
    store.decorator_list = []
    scope = {
        **vars(lmc_utils),
        "torch": torch,
        "torch_device_type": "cpu",
        "logger": logging.getLogger("lmcache-test"),
        "List": list,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "MemoryObj": object,
    }
    exec(compile(ast.Module([store], []), path, "exec"), scope)  # noqa: S102
    return scope["store"]


def test_worker_records_the_token_path_stats_and_kv_events():
    """The same store statistics and KV events as LMCache's `store` on the token
    path, for a save past an already-stored prefix."""
    tokens = list(range(1000, 1000 + 6 * CHUNK))
    skip, end = 2 * CHUNK, 5 * CHUNK
    token_engine = _engine()
    mask = torch.ones(end, dtype=torch.bool)
    mask[:skip] = False
    _lmcache_store()(
        token_engine,
        tokens_to_tensor(tokens[:end]),
        mask=mask,
        block_ids=list(range(12)),
        req_id="3",
        producer_event=_Fence(),
    )
    engine = _engine()
    tdb = engine.token_database
    hashes = [h for _s, _e, h in tdb.process_tokens(tokens[:end], make_key=False)]
    req = LMCacheReqMeta(
        req_id=3,
        token_ids=[],
        block_ids=list(range(12)),
        save_spec=SaveSpec(skip_leading_tokens=skip),
        save_operation=SaveOperationId(3, 0),
        prehashed_save=PrehashedSave(
            start=skip,
            hashes=tuple(hashes[2:5]),
            token_ids=tuple(tokens[skip:end]),
        ),
    )
    _worker(engine)._do_save_req(req, producer_event=_Fence())
    assert engine.stats_monitor.calls == token_engine.stats_monitor.calls
    assert engine.kv_events == token_engine.kv_events and len(engine.kv_events) == 3
