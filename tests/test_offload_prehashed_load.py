# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_OFFLOAD_PREHASHED_LOAD and OFFLOAD_BATCHED_HOST_COPY.

A prehashed load must hand the GPU connector exactly what LMCache's own
``retrieve`` hands it (objects, token ranges, kwargs), leave every reference
and pin as ``retrieve`` does, and report the request loaded only after the
copy returned. The worker tests run a real LMCache engine on the CPU.
"""

import threading
import uuid
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lmcache")

from lmcache.v1.cache_engine import LMCacheEngineBuilder  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.memory_management import MemoryFormat  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from test_offload_prehashed_save import CHUNK, _scheduler, _seq  # noqa: E402

from atom.kv_transfer.disaggregation.types import LoadOperationId  # noqa: E402
from atom.kv_transfer.offload import _block_gpu_connector as bgc  # noqa: E402
from atom.kv_transfer.offload.atom_lmcache_staging import (  # noqa: E402
    memory_object_as_uint8,
    memory_object_bytes,
)
from atom.kv_transfer.offload.hybrid.m3.connector import (  # noqa: E402
    M3OffloadConnector,
)
from atom.kv_transfer.offload.metadata import (  # noqa: E402
    ATOMRawBytesLMCacheMetadata,
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    PrehashedLoad,
)

BLOCK = 4
BYTES_PER_BLOCK = 32


# --------------------------------------------------------------- scheduler
def _spec(hbm, lmc):
    return LoadSpec(hbm_cached_tokens=hbm, lmcache_cached_tokens=lmc, can_load=True)


def test_load_request_ships_the_hashes_of_the_loaded_chunks(monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_LOAD", "1")
    scheduler = _scheduler(monkeypatch)
    seq = _seq(6 * CHUNK + 3)
    op = LoadOperationId(seq.id, 0)
    req = scheduler._build_load_request(seq, _spec(2 * CHUNK, 5 * CHUNK), op, 5 * CHUNK)
    assert req.token_ids == []
    assert req.block_ids == list(seq.block_table)
    assert req.load_operation == op
    tdb = scheduler._lookup_client.token_database
    expected = [
        h
        for s, _e, h in tdb.process_tokens(list(seq.token_ids), make_key=False)
        if 2 * CHUNK <= s < 5 * CHUNK
    ]
    assert req.prehashed_load == PrehashedLoad(start=2 * CHUNK, hashes=tuple(expected))


@pytest.mark.parametrize(
    ("flag", "hbm", "lmc", "end"),
    [
        ("0", 0, 4 * CHUNK, 4 * CHUNK),  # off
        ("1", 1, 4 * CHUNK, 4 * CHUNK),  # unaligned HBM frontier
        ("1", 0, 4 * CHUNK - 1, 4 * CHUNK - 1),  # unaligned hit
        ("1", 0, 4 * CHUNK, 5 * CHUNK),  # a transfer past the hit
    ],
)
def test_load_request_keeps_the_token_path(monkeypatch, flag, hbm, lmc, end):
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_LOAD", flag)
    scheduler = _scheduler(monkeypatch)
    seq = _seq(6 * CHUNK)
    req = scheduler._build_load_request(
        seq, _spec(hbm, lmc), LoadOperationId(seq.id, 0), end
    )
    assert req.prehashed_load is None
    assert req.token_ids == list(seq.token_ids[:end])


def test_load_request_needs_the_prehashed_chain(monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_LOAD", "1")
    scheduler = _scheduler(monkeypatch, enabled=False)
    seq = _seq(4 * CHUNK)
    req = scheduler._build_load_request(
        seq, _spec(0, 4 * CHUNK), LoadOperationId(seq.id, 0), 4 * CHUNK
    )
    assert req.prehashed_load is None


# ------------------------------------------------------- worker, real LMCache
class _BytesConnector:
    """Fills a stored chunk with its start token, records what a load delivers."""

    def __init__(self):
        self.loads = []
        self.during_load = None

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for obj, start in zip(memory_objs, starts):
            obj.tensor.fill_(start % 251)

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        if self.during_load is not None:
            self.during_load()
        self.loads.append(
            (
                list(starts),
                list(ends),
                [int(obj.tensor.reshape(-1)[0]) for obj in memory_objs],
                [obj.get_ref_count() for obj in memory_objs],
                kwargs,
            )
        )


@pytest.fixture
def engines(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    made = []

    def make(worker_id=1):
        iid = f"atom-prehashed-load-{uuid.uuid4().hex}"
        cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=CHUNK, local_cpu=True, max_local_cpu_size=0.01
        )
        base = LMCacheMetadata(
            model_name="m3",
            world_size=4,
            local_world_size=4,
            worker_id=worker_id,
            local_worker_id=worker_id,
            kv_dtype=torch.uint8,
            kv_shape=(1, 2, CHUNK, 1, 1),
            chunk_size=CHUNK,
            engine_id=iid,
        )
        meta = ATOMRawBytesLMCacheMetadata(
            base, atom_block_size=BLOCK, bytes_per_block=BYTES_PER_BLOCK
        )
        engine = LMCacheEngineBuilder.get_or_create(
            iid, cfg, meta, _BytesConnector(), lambda t, s: None, lambda o, s: o
        )
        engine.fmt = MemoryFormat.KV_2LTD
        engine.post_init()
        made.append(iid)
        return engine

    yield make
    for iid in made:
        LMCacheEngineBuilder.destroy(iid)


def _worker(engine):
    worker = M3OffloadConnector.__new__(M3OffloadConnector)
    worker._init_worker_common(
        SimpleNamespace(kv_transfer_config={"kv_role": "offload"})
    )
    worker.chunk_size = CHUNK
    worker.block_size = worker.virtual_block_size = BLOCK
    worker._engine = engine
    worker._early_release = False
    worker._rank = 0
    worker._lookup_server = None
    return worker


TOKENS = list(range(1000, 1000 + 6 * CHUNK))
BLOCKS = list(range(100, 100 + 6 * CHUNK // BLOCK))


def _store(engine, tokens=TOKENS):
    engine.store(torch.tensor(tokens), block_ids=BLOCKS)


def _hashes(engine, tokens=TOKENS):
    return [
        h for _s, _e, h in engine.token_database.process_tokens(tokens, make_key=False)
    ]


def _token_req(req_id, hbm, lmc):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=TOKENS[:lmc],
        block_ids=BLOCKS,
        load_spec=_spec(hbm, lmc),
        load_operation=LoadOperationId(req_id, 0),
    )


def _hashed_req(engine, req_id, hbm, lmc):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=BLOCKS,
        load_spec=_spec(hbm, lmc),
        load_operation=LoadOperationId(req_id, 0),
        prehashed_load=PrehashedLoad(
            start=hbm, hashes=tuple(_hashes(engine)[hbm // CHUNK : lmc // CHUNK])
        ),
    )


def _ref_counts(engine):
    backend = engine.storage_manager.storage_backends["LocalCPUBackend"]
    return sorted(
        (obj.get_ref_count(), obj.meta.pin_count) for obj in backend.hot_cache.values()
    )


@pytest.mark.parametrize("pinned", [True, False])
def test_prehashed_load_delivers_what_retrieve_delivers(engines, pinned):
    """Same objects, ranges and kwargs to `batched_to_gpu`, same references
    held during the copy and released after, with and without a lookup pin."""
    engine = engines()
    _store(engine)
    worker = _worker(engine)
    before = _ref_counts(engine)
    results = []
    mapped = []
    real_mapping = engine.storage_manager.get_block_mapping

    def mapping(infos):
        mapped.append(len(infos))
        return real_mapping(infos)

    engine.storage_manager.get_block_mapping = mapping
    for req in (
        _token_req(7, 2 * CHUNK, 5 * CHUNK),
        _hashed_req(engine, 7, 2 * CHUNK, 5 * CHUNK),
    ):
        if pinned:
            hit = engine.lookup(TOKENS[: 5 * CHUNK], lookup_id="7", pin=True)
            assert hit == 5 * CHUNK
        results.append(worker._retrieve(req, 2 * CHUNK, 5 * CHUNK))
        worker._lookup_unpin(7)
        assert _ref_counts(engine) == before
    # A pinned lookup names the location; only an unpinned load maps blocks.
    assert mapped == ([] if pinned else [3, 3])
    token_load, hashed_load = engine.gpu_connector.loads
    assert hashed_load == token_load
    assert token_load[0] == [2 * CHUNK, 3 * CHUNK, 4 * CHUNK]
    assert token_load[4] == {"block_ids": BLOCKS, "req_id": "7"}
    assert results == [(True, 3 * CHUNK), (True, 3 * CHUNK)]


def test_a_missing_chunk_ends_the_load_where_retrieve_ends_it(engines):
    engine = engines()
    _store(engine)
    key = engine.token_database._make_key_by_hash(_hashes(engine)[3])
    engine.storage_manager.remove(key)
    worker = _worker(engine)
    before = _ref_counts(engine)
    token = worker._retrieve(_token_req(8, CHUNK, 5 * CHUNK), CHUNK, 5 * CHUNK)
    hashed = worker._retrieve(
        _hashed_req(engine, 8, CHUNK, 5 * CHUNK), CHUNK, 5 * CHUNK
    )
    assert token == hashed == (False, 2 * CHUNK)
    token_load, hashed_load = engine.gpu_connector.loads
    assert hashed_load == token_load
    assert token_load[0] == [CHUNK, 2 * CHUNK]
    assert _ref_counts(engine) == before


def test_chunks_past_a_missing_one_in_another_location_are_dropped(engines):
    """Two tiers (e.g. CPU then disk): a chunk the first cannot read ends the
    run, so the second tier's later chunks are released, not loaded."""
    engine = engines()
    _store(engine)
    manager = engine.storage_manager
    real_get = manager.batched_get
    missing = engine.token_database._make_key_by_hash(_hashes(engine)[1])
    manager.get_block_mapping = lambda infos: {"A": infos[:3], "B": infos[3:]}

    def batched_get(keys, location=None):
        objs = real_get(keys, location="LocalCPUBackend")
        for index, key in enumerate(keys):
            if key == missing:  # unreadable: as if never got
                objs[index].ref_count_down()
                objs[index] = None
        return objs

    manager.batched_get = batched_get
    worker = _worker(engine)
    before = _ref_counts(engine)
    assert worker._retrieve(_hashed_req(engine, 10, 0, 4 * CHUNK), 0, 4 * CHUNK) == (
        False,
        CHUNK,
    )
    assert engine.gpu_connector.loads[-1][0] == [0]
    assert _ref_counts(engine) == before


def test_each_pp_stage_loads_its_own_chunks_from_the_same_hashes(engines):
    """Under PP every stage's worker gets the same metadata; the key carries the
    stage's own worker id, so the same hashes find that stage's bytes."""
    stages = [engines(worker_id=w) for w in (0, 1)]
    for index, engine in enumerate(stages):
        engine.gpu_connector.batched_from_gpu = (
            lambda objs, starts, ends, _i=index, **kw: [
                obj.tensor.fill_(10 * _i + s // CHUNK) for obj, s in zip(objs, starts)
            ]
        )
        _store(engine)
    req = _hashed_req(stages[0], 9, 0, 3 * CHUNK)
    for index, engine in enumerate(stages):
        assert _worker(engine)._retrieve(req, 0, 3 * CHUNK) == (True, 3 * CHUNK)
        assert engine.gpu_connector.loads[-1][2] == [10 * index + c for c in range(3)]


def test_the_load_is_reported_only_after_the_copy_returned(engines):
    engine = engines()
    _store(engine)
    worker = _worker(engine)
    seen = []
    engine.gpu_connector.during_load = lambda: seen.append(
        set(worker.get_finished().finished_loading or ())
    )
    worker._do_load_req(_hashed_req(engine, 11, 0, 4 * CHUNK))
    assert seen == [set()]
    done = worker.get_finished()
    assert done.finished_loading == {LoadOperationId(11, 0)}
    assert not done.failed_loading


def test_a_short_load_fails_and_names_its_blocks(engines):
    engine = engines()
    _store(engine, TOKENS[: 2 * CHUNK])
    worker = _worker(engine)
    worker._do_load_req(_hashed_req(engine, 12, 0, 4 * CHUNK))
    done = worker.get_finished()
    assert done.failed_loading == {LoadOperationId(12, 0)}
    assert worker.take_load_error_blocks() == set(BLOCKS[: 4 * CHUNK // BLOCK])


def test_a_load_whose_hashes_do_not_cover_its_range_fails_closed(engines):
    engine = engines()
    _store(engine)
    worker = _worker(engine)
    req = _hashed_req(engine, 13, 0, 4 * CHUNK)
    req.load_spec.lmcache_cached_tokens = 5 * CHUNK  # one chunk more than shipped
    worker._guard("load", worker._do_load_req, req)
    assert engine.gpu_connector.loads == []
    assert worker.get_finished().failed_loading == {LoadOperationId(13, 0)}


def test_a_preempting_fence_waits_for_an_in_flight_prehashed_load(engines):
    """`wait_for_requests` (abort / preemption before block reuse) must not
    return while the load thread is still writing the request's blocks."""
    engine = engines()
    _store(engine)
    worker = _worker(engine)
    worker._do_load = True
    entered, release = threading.Event(), threading.Event()

    def hold():
        entered.set()
        assert release.wait(5)

    engine.gpu_connector.during_load = hold
    meta = LMCacheOffloadMetadata()
    meta.add_request(_hashed_req(engine, 14, 0, 2 * CHUNK))
    worker.start_load_kv(meta)
    assert entered.wait(5)
    fenced = threading.Event()
    waiter = threading.Thread(
        target=lambda: (worker.wait_for_requests([14]), fenced.set())
    )
    waiter.start()
    assert not fenced.wait(0.2)
    release.set()
    waiter.join(5)
    assert fenced.is_set()
    assert worker.get_finished().finished_loading == {LoadOperationId(14, 0)}
    worker.close()


@pytest.mark.parametrize(
    "flag", ["async_loading", "save_only_first_rank", "remove_after_retrieve"]
)
def test_an_engine_retrieve_does_not_mirror_is_refused(monkeypatch, flag):
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_LOAD", "1")
    worker = _worker(SimpleNamespace(**{flag: True}))
    with pytest.raises(ValueError, match=flag):
        worker._check_prehashed_load_engine()
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_LOAD", "0")
    worker._check_prehashed_load_engine()


# ------------------------------------------------- OFFLOAD_BATCHED_HOST_COPY
def test_memory_object_bytes_are_the_validated_bytes(engines):
    engine = engines()
    shapes = engine.metadata.get_shapes(CHUNK)
    obj = engine.storage_manager.allocate(
        shapes, engine.metadata.get_dtypes(), fmt=engine.fmt
    )
    nbytes = obj.get_size()
    obj.tensor.reshape(-1).copy_(torch.arange(nbytes) % 251)
    fast = memory_object_bytes(obj, nbytes)
    assert fast.data_ptr() == memory_object_as_uint8(obj, nbytes).data_ptr()
    assert torch.equal(fast, memory_object_as_uint8(obj, nbytes))
    # A shorter request is not the whole buffer: the validated slice.
    assert torch.equal(memory_object_bytes(obj, 8), memory_object_as_uint8(obj, 8))
    with pytest.raises(ValueError):
        memory_object_bytes(obj, nbytes + 1)
    # A narrowed object's buffer is longer than its bytes: never the buffer.
    obj.set_used_size(nbytes - 8)
    narrowed = memory_object_bytes(obj, nbytes - 8)
    assert narrowed.numel() == nbytes - 8
    assert torch.equal(narrowed, memory_object_as_uint8(obj, nbytes - 8))
    obj.ref_count_down()


def _group(chunks):
    return SimpleNamespace(
        chunks=[SimpleNamespace(tensor=t, nbytes=int(t.numel())) for t in chunks]
    )


def _connector(monkeypatch):
    monkeypatch.delenv("OFFLOAD_GPU_STAGING_MAX_BYTES", raising=False)
    return bgc.BlockGPUConnector(
        SimpleNamespace(device=torch.device("cpu"), bytes_per_block=4),
        block_size=4,
        chunk_size=4,
    )


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_batched_copy_moves_the_bytes_the_per_chunk_copy_moves(monkeypatch, direction):
    connector = _connector(monkeypatch)
    sizes = (4, 12, 8)
    results = []
    for batched in (False, True):
        monkeypatch.setattr(bgc, "_BATCHED_HOST_COPY", batched)
        hosts = [
            torch.full((n,), i + 1, dtype=torch.uint8) for i, n in enumerate(sizes)
        ]
        staging = torch.arange(sum(sizes), dtype=torch.uint8)
        group = _group(hosts)
        if direction == "h2d":
            connector._memory_objs_to_slice(group, staging, prepared_ids_active=True)
            result = staging.clone()
        else:
            connector._slice_to_memory_objs(group, staging, prepared_ids_active=True)
            result = torch.cat(hosts)
        results.append(result)
    assert torch.equal(results[0], results[1])


class _Host:
    """A host tensor stand-in: device, backing allocation, physical pinning."""

    def __init__(self, allocation, pinned):
        self.device = SimpleNamespace(type="cpu")
        self._allocation = allocation
        self.pinned = pinned
        self.pin_queries = 0

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self._allocation)

    def is_pinned(self):
        self.pin_queries += 1
        return self.pinned


class _Staging:
    device = SimpleNamespace(type="cuda")

    def split(self, sizes):
        return [object() for _ in sizes]


@pytest.mark.parametrize("prepared_ids_active", [False, True])
@pytest.mark.parametrize("pinned", [(True, True), (True, False), (False, True)])
def test_batched_copy_policy(monkeypatch, prepared_ids_active, pinned):
    """Non-blocking only as the per-chunk policy would make every chunk's copy,
    asking pinning once per backing allocation."""
    connector = _connector(monkeypatch)
    calls = []
    monkeypatch.setattr(
        bgc.torch,
        "_foreach_copy_",
        lambda dst, src, non_blocking: calls.append(non_blocking),
    )
    # Pinning is a property of the allocation: chunks 0 and 1 share one.
    hosts = [_Host(a, pinned[a - 1]) for a in (1, 1, 2)]
    group = SimpleNamespace(chunks=[SimpleNamespace(tensor=h, nbytes=4) for h in hosts])
    connector._copy_group(
        group, _Staging(), to_staging=True, prepared_ids_active=prepared_ids_active
    )
    assert calls == [prepared_ids_active and all(pinned)]
    if prepared_ids_active:
        assert hosts[0].pin_queries == 1 and hosts[1].pin_queries == 0
        assert hosts[2].pin_queries == int(pinned[0])
    else:
        assert all(h.pin_queries == 0 for h in hosts)
