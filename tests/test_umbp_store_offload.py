# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The UMBP offload connector (``umbp_store``), on CPU.

UMBP is faked by a ``UMBPClient`` stand-in that moves bytes by address
(ctypes) between registered host buffers and one dict per standalone server,
answering one bool per key as the real binding does. The block GPU connector
is faked by one that packs a byte pattern per KV block. Self-contained: no
helper of the Mooncake Store offload's tests is used.
"""

from __future__ import annotations

import array
import ctypes
import logging
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL,
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, SaveSpec
from atom.kv_transfer.offload.umbp_store import client as umbp_client
from atom.kv_transfer.offload.umbp_store import keys, taint
from atom.kv_transfer.offload.umbp_store import pool as pool_mod
from atom.kv_transfer.offload.umbp_store import worker as worker_mod
from atom.kv_transfer.offload.umbp_store.config import (
    UMBPStoreOffloadConfig,
    check_engine_compatibility,
    parse_umbp_store_config,
)
from atom.kv_transfer.offload.umbp_store.pool import TransferSlotPool
from atom.kv_transfer.offload.umbp_store.scheduler import UMBPStoreOffloadScheduler
from atom.kv_transfer.offload.umbp_store.worker import UMBPStoreOffloadConnector

ADDRESS = "unix:///tmp/umbp-test/server.sock"
OTHER_ADDRESS = "unix:///tmp/umbp-test/other.sock"
BLOCK = 4  # tokens per KV block
CHUNK = 8  # tokens per chunk: two blocks
BYTES_PER_BLOCK = 16
CHUNK_BYTES = BYTES_PER_BLOCK * CHUNK // BLOCK
SLOT = 4096  # one slot's stride for CHUNK_BYTES
NAMESPACE = "atomumbp1-test"


# --- fakes -------------------------------------------------------------------


class FakeServer:
    """Standalone servers by address: their objects, and their clients."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, bytes]] = {}
        self.clients: list[FakeUMBPClient] = []
        # Keys whose put or get answers False.
        self.fail_puts: set[str] = set()
        self.fail_gets: set[str] = set()
        # A call every client raises from.
        self.raise_on: str | None = None
        self.connect_error: Exception | None = None

    def at(self, address: str = ADDRESS) -> dict[str, bytes]:
        return self.objects.setdefault(address, {})

    def connect(self, address: str, startup_timeout_ms: int) -> FakeUMBPClient:
        if self.connect_error is not None:
            raise self.connect_error
        client = FakeUMBPClient(self, address, startup_timeout_ms)
        self.clients.append(client)
        return client


class FakeUMBPClient:
    """``mori.umbp.UMBPClient`` in standalone mode, over host memory."""

    def __init__(self, server: FakeServer, address: str, timeout_ms: int) -> None:
        self.server = server
        self.address = address
        self.timeout_ms = timeout_ms
        self.regions: dict[int, int] = {}
        self.calls: list[tuple] = []

    @property
    def objects(self) -> dict[str, bytes]:
        return self.server.at(self.address)

    def _maybe_raise(self, call):
        if call == self.server.raise_on:
            raise RuntimeError(f"{call} blew up")

    def _is_registered(self, ptr, size):
        return any(
            base <= ptr and ptr + size <= base + n for base, n in self.regions.items()
        )

    def register_memory(self, ptr, size):
        self.calls.append(("register", ptr, size))
        self.regions[ptr] = size
        return True

    def deregister_memory(self, ptr):
        # Standalone: one call drops every region of the client.
        self.calls.append(("deregister", ptr))
        self.regions.clear()

    def batch_exists(self, keys):
        self.calls.append(("exists", list(keys)))
        self._maybe_raise("exists")
        return [key in self.objects for key in keys]

    def batch_put_from_ptr(self, keys, ptrs, sizes):
        self.calls.append(("put", list(keys)))
        self._maybe_raise("put")
        answers = []
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            if key in self.server.fail_puts or not self._is_registered(ptr, size):
                answers.append(False)
                continue
            # Immutable keys: an existing one keeps its bytes and succeeds.
            self.objects.setdefault(key, ctypes.string_at(ptr, size))
            answers.append(True)
        return answers

    def batch_get_into_ptr(self, keys, ptrs, sizes):
        self.calls.append(("get", list(keys)))
        self._maybe_raise("get")
        answers = []
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            data = self.objects.get(key)
            if (
                key in self.server.fail_gets
                or data is None
                or len(data) != size
                or not self._is_registered(ptr, size)
            ):
                answers.append(False)
                continue
            ctypes.memmove(ptr, data, size)
            answers.append(True)
        return answers


class FakeGPUConnector:
    """Packs each KV block as a byte pattern, as the block GPU connector would.

    ``kv`` is the GPU KV as block id -> bytes; a block never written holds a
    pattern derived from its id. A save packs tail-to-head and publishes one
    source-safe group per chunk inside ``track_save_source``.
    """

    gpu_staging_chunk_bytes = CHUNK_BYTES
    gpu_staging_buffer_bytes = 4 * CHUNK_BYTES
    block_size = BLOCK
    _ranges_to_block_ids = BlockGPUConnector._ranges_to_block_ids
    _range_block_ids = BlockGPUConnector._range_block_ids

    def __init__(self, source_safe=None) -> None:
        self.kv: dict[int, bytes] = {}
        self.calls: list[tuple] = []
        self.source_safe = source_safe
        self.closed = False
        self._operation = None

    @contextmanager
    def track_save_source(self, operation):
        previous, self._operation = self._operation, operation
        try:
            yield
        finally:
            self._operation = previous

    @staticmethod
    def pattern(block: int) -> bytes:
        return bytes((block * 7 + i) % 251 for i in range(BYTES_PER_BLOCK))

    def batched_from_gpu(self, slots, starts, ends, *, block_ids, producer_event=None):
        self.calls.append(("from", list(starts)))
        for slot, start, end in sorted(
            zip(slots, starts, ends, strict=True), key=lambda item: -item[1]
        ):
            blocks = block_ids[start // BLOCK : end // BLOCK]
            data = b"".join(self.kv.get(b, self.pattern(b)) for b in blocks)
            slot.tensor.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
            if self._operation is not None and self.source_safe is not None:
                self.source_safe(SaveSourceGroupId(self._operation, ((start, end),)))

    def batched_to_gpu(self, slots, starts, ends, *, block_ids):
        self.calls.append(("to", list(starts)))
        for slot, start, end in zip(slots, starts, ends, strict=True):
            data = slot.tensor.numpy().tobytes()
            for i, block in enumerate(block_ids[start // BLOCK : end // BLOCK]):
                self.kv[block] = data[i * BYTES_PER_BLOCK : (i + 1) * BYTES_PER_BLOCK]

    def close(self):
        self.closed = True


class _FakeAiterDtypes:
    def __init__(self) -> None:
        self.d_dtypes = {"fp8": torch.float8_e4m3fn, "bf16": torch.bfloat16}


@pytest.fixture(autouse=True)
def aiter_arch(monkeypatch):
    """AITER's dtype table and arch detection, as on gfx950.

    The namespace asks AITER what ``fp8`` is stored as; CPU runners have no
    AITER.
    """
    package = types.ModuleType("aiter")
    package.dtypes = _FakeAiterDtypes()
    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx = lambda: "gfx950"
    for name, module in (
        ("aiter", package),
        ("aiter.jit", types.ModuleType("aiter.jit")),
        ("aiter.jit.utils", types.ModuleType("aiter.jit.utils")),
        ("aiter.jit.utils.chip_info", chip_info),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(autouse=True)
def fresh_process_state(monkeypatch):
    """What a process keeps for its life (marks, retained memory), per test."""
    monkeypatch.setattr(taint, "_TAINTED", {})
    monkeypatch.setattr(umbp_client, "_RETAINED_CLIENTS", [])
    monkeypatch.setattr(pool_mod, "_UNSETTLED_ALLOCATIONS", [])


@pytest.fixture
def server(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(umbp_client, "_new_umbp_client", server.connect)
    return server


@pytest.fixture
def one_rank(monkeypatch):
    monkeypatch.setattr(
        worker_mod,
        "_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )


def _extra(**overrides):
    extra = {"umbp_store.address": ADDRESS, "umbp_store.chunk_tokens": CHUNK}
    for name, value in overrides.items():
        if value is None:
            extra.pop(f"umbp_store.{name}", None)
        else:
            extra[f"umbp_store.{name}"] = value
    return extra


def _config(role="offload", *, pp=1, tp=1, layers=4, extra=None, **fields):
    config = SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "umbp_store",
            "kv_role": role,
            "kv_connector_extra_config": _extra() if extra is None else extra,
        },
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=1,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        hf_config=SimpleNamespace(num_hidden_layers=layers, model_type="glm_moe_dsa"),
        model="test-model",
        kv_cache_dtype="fp8",
        speculative_config=None,
    )
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def _seq(req_id, num_prompt_tokens, *, offset=0):
    return SimpleNamespace(
        id=req_id,
        token_ids=array.array("i", range(offset, offset + num_prompt_tokens)),
        num_prompt_tokens=num_prompt_tokens,
        num_cached_tokens=0,
        block_table=list(range(-(-num_prompt_tokens // BLOCK))),
    )


def _hashes(num_tokens, offset=0):
    return keys.chunk_hash_chain(list(range(offset, offset + num_tokens)), CHUNK)


def _key(hashes, index, *, rank=0, world=1, namespace=NAMESPACE):
    [key] = keys.chunk_keys(namespace, rank, world, hashes, index, index + 1)
    return key


def _save_req(req_id, num_tokens, *, generation=0, offset=0):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=list(range(num_tokens // BLOCK)),
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(req_id, generation),
        chunk_hashes=_hashes(num_tokens, offset),
    )


def _load_req(req_id, *, hbm, lmc, block_ids, generation=0):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=block_ids,
        load_spec=LoadSpec(
            hbm_cached_tokens=hbm,
            lmcache_cached_tokens=lmc,
            can_load=True,
            transfer_end_tokens=None,
        ),
        load_operation=LoadOperationId(req_id, generation),
        chunk_hashes=_hashes(lmc),
    )


def _store(operation, succeeded=True):
    return ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, operation, succeeded)


def _quiescent(operation):
    return ConnectorCompletion(DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL, operation, True)


def _safe(operation, index):
    return ConnectorCompletion(
        DENSE_PAGE_SOURCE_SAFE_CHANNEL,
        SaveSourceGroupId(operation, ((index * CHUNK, (index + 1) * CHUNK),)),
        True,
    )


def _kv_caches(num_blocks=8):
    # MLA-like: token-major rows, one plane per layer.
    return {
        f"layer_{i}": SimpleNamespace(
            k_cache=torch.zeros((num_blocks * BLOCK, 1, 8), dtype=torch.uint8),
            index_cache=None,
        )
        for i in range(2)
    }


@pytest.fixture
def make_worker(server):
    """A worker whose GPU connector and pool are fakes over a UMBP client."""
    built = []

    def build(*, address=ADDRESS, save_slots=16, load_slots=16):
        worker = UMBPStoreOffloadConnector(_config(extra=_extra(address=address)))
        gpu = FakeGPUConnector(worker._source_group_safe)
        client = umbp_client.UMBPStoreClient(address=address)
        pool = TransferSlotPool(
            device="cpu",
            chunk_bytes=CHUNK_BYTES,
            save_bytes=save_slots * SLOT,
            load_bytes=load_slots * SLOT,
            client=client,
        )
        worker._gpu_connector, worker._client, worker._pool = gpu, client, pool
        worker._namespace = NAMESPACE
        worker._rank, worker._world = 0, 1
        worker._chunk_bytes = CHUNK_BYTES
        built.append(worker)
        return worker, gpu

    yield build
    for worker in built:
        worker.close()


# --- config and registration -------------------------------------------------


def test_config_defaults():
    cfg = parse_umbp_store_config({"kv_connector_extra_config": _extra()})
    assert cfg == UMBPStoreOffloadConfig(
        address=ADDRESS,
        startup_timeout_ms=10000,
        chunk_tokens=CHUNK,
        load_pool_mib=1024,
        save_pool_mib=256,
        save_abandon_timeout_s=300.0,
    )
    assert cfg.load_pool_bytes == 1024 << 20
    # Without kv_connector_extra_config, the top-level dict.
    assert parse_umbp_store_config(_extra()).address == ADDRESS


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({}, "needs umbp_store.address"),
        # No Mooncake setting is read, nor a Mooncake-only one accepted.
        ({"mooncake_store.master": "x:1"}, "needs umbp_store.address"),
        ({**_extra(), "umbp_store.master": "x:1"}, "unknown UMBP offload setting"),
        ({**_extra(), "umbp_store.protocol": "rdma"}, "unknown UMBP offload setting"),
        (_extra(chunk_tokens=0), "chunk_tokens"),
        (_extra(startup_timeout_ms="5"), "startup_timeout_ms"),
        (_extra(save_abandon_timeout_s=0), "save_abandon_timeout_s"),
        (_extra(address=" "), "non-empty string"),
    ],
)
def test_config_refuses_bad_settings(extra, match):
    with pytest.raises(ValueError, match=match):
        parse_umbp_store_config({"kv_connector_extra_config": extra})


def test_config_ignores_mooncake_settings():
    extra = {**_extra(), "mooncake_store.chunk_tokens": 2 * CHUNK}
    cfg = parse_umbp_store_config({"kv_connector_extra_config": extra})
    assert cfg.chunk_tokens == CHUNK


def test_engine_compatibility_takes_m3_and_refuses_the_rest():
    cfg = parse_umbp_store_config(_extra())
    m3 = _config()
    m3.hf_config.model_type = "minimax_m3"
    check_engine_compatibility(cfg, m3)
    check_engine_compatibility(cfg, _config())
    with pytest.raises(ValueError, match="decode context parallelism"):
        check_engine_compatibility(cfg, _config(decode_context_parallel_size=2))
    with pytest.raises(ValueError, match="multiple of the KV cache block size"):
        check_engine_compatibility(
            parse_umbp_store_config(_extra(chunk_tokens=CHUNK + 1)), _config()
        )


def test_registered_as_an_offload_backend_with_both_halves(server):
    assert KVConnectorFactory.canonical_name("umbp_store") == "umbp_store"
    assert KVConnectorFactory.is_offload_backend("umbp_store")
    scheduler = KVConnectorFactory.create_connector(_config(), role="scheduler")
    worker = KVConnectorFactory.create_connector(_config(), role="worker")
    try:
        assert isinstance(scheduler, UMBPStoreOffloadScheduler)
        assert scheduler.is_offload and not scheduler.is_producer
        assert isinstance(worker, UMBPStoreOffloadConnector)
        assert worker.chunk_size == CHUNK and worker._early_release
        assert server.clients == []  # nobody connects before it has to
    finally:
        worker.close()


def test_m3_reads_the_page_regions_and_dense_does_not():
    def topology(model_type):
        return SimpleNamespace(
            kv_transfer_config={"kv_connector": "umbp_store"},
            hf_config=SimpleNamespace(model_type=model_type, architectures=[]),
        )

    assert KVConnectorFactory.topology_reads_block_regions(topology("minimax_m3"))
    assert not KVConnectorFactory.topology_reads_block_regions(topology("llama"))


@pytest.mark.parametrize("model_type", ["minimax_m3", "llama"])
def test_multi_mooncake_producer_with_umbp_offload(model_type):
    """The P side of a P/D deployment: Mooncake transfer plus UMBP offload.

    Reads the region map as the same topology with the Mooncake Store
    offload does; M3 needs it for its PAGE codec either way.
    """

    def multi(offload):
        return SimpleNamespace(
            kv_transfer_config={
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                    {"kv_connector": offload, "kv_role": "offload"},
                ],
            },
            hf_config=SimpleNamespace(model_type=model_type, architectures=[]),
        )

    reads = KVConnectorFactory.topology_reads_block_regions
    assert reads(multi("umbp_store")) == reads(multi("mooncake_store"))
    if model_type == "minimax_m3":
        assert reads(multi("umbp_store"))


def test_namespace_is_umbps_own():
    namespace = keys.store_namespace(_config(), CHUNK)
    assert namespace.startswith("atomumbp1-")
    assert not namespace.startswith("atomkv1-")
    m3 = _config()
    m3.hf_config.model_type = "minimax_m3"
    assert keys.store_namespace(m3, CHUNK) != namespace  # codec id folded in
    assert not hasattr(keys, "STORE_TAINT_KEY")  # no server-side marker


# --- client mapping ----------------------------------------------------------


def _host_buffer(nbytes, fill=0):
    return torch.full((nbytes,), fill, dtype=torch.uint8)


def test_client_answers_one_bool_per_key(server):
    client = umbp_client.UMBPStoreClient(address=ADDRESS, startup_timeout_ms=1234)
    [fake] = server.clients
    assert (fake.address, fake.timeout_ms) == (ADDRESS, 1234)
    src, dst = _host_buffer(64, 7), _host_buffer(64)
    client.register(src.data_ptr(), 64)
    client.register(dst.data_ptr(), 64)

    server.fail_puts.add("b")
    assert client.put(["a", "b"], [src.data_ptr()] * 2, [32, 32]) == [True, False]
    assert client.exists(["b", "a"]) == [False, True]
    assert client.get(["a", "b"], [dst.data_ptr(), dst.data_ptr() + 32], [32, 32]) == [
        True,
        False,
    ]
    assert dst[:32].tolist() == [7] * 32 and dst[32:].tolist() == [0] * 32

    stats = client.stats()
    assert stats["counts"]["put_keys"] == 2 and stats["counts"]["put_bytes"] == 32
    assert stats["counts"]["get_bytes"] == 32
    assert stats["failures"] == {"put:false": 1, "get:false": 1}
    # No remove, no marker: the client has no such calls to pretend with.
    assert not hasattr(client, "remove") and not hasattr(client, "put_marker")


def test_a_repeated_put_succeeds_and_keeps_the_first_bytes(server):
    client = umbp_client.UMBPStoreClient(address=ADDRESS)
    src = _host_buffer(32, 7)
    client.register(src.data_ptr(), 32)
    assert client.put(["a"], [src.data_ptr()], [32]) == [True]
    src.fill_(9)
    assert client.put(["a"], [src.data_ptr()], [32]) == [True]
    assert server.at()["a"] == bytes([7] * 32)


def test_client_batches_lookups_and_refuses_duplicates(server, monkeypatch):
    monkeypatch.setattr(umbp_client, "_LOOKUP_BATCH_KEYS", 2)
    client = umbp_client.UMBPStoreClient(address=ADDRESS)
    server.at().update({"k1": b"", "k3": b""})
    assert client.exists(["k0", "k1", "k2", "k3", "k4"]) == [
        False,
        True,
        False,
        True,
        False,
    ]
    [fake] = server.clients
    assert [len(c[1]) for c in fake.calls if c[0] == "exists"] == [2, 2, 1]
    with pytest.raises(umbp_client.CallRefused, match="unique"):
        client.put(["x", "x"], [0, 0], [1, 1])
    with pytest.raises(umbp_client.CallRefused, match="unique"):
        client.get(["x", "x"], [0, 0], [1, 1])
    assert [c for c in fake.calls if c[0] in ("put", "get")] == []


def test_client_connect_failure_names_the_server(server):
    server.connect_error = RuntimeError("server is not ready")
    with pytest.raises(RuntimeError, match="umbp_standalone_server") as info:
        umbp_client.UMBPStoreClient(address=ADDRESS)
    assert ADDRESS in str(info.value)


def test_deregister_waits_for_the_last_region_and_close_drops_the_client(server):
    client = umbp_client.UMBPStoreClient(address=ADDRESS)
    [fake] = server.clients
    a, b = _host_buffer(16), _host_buffer(16)
    client.register(a.data_ptr(), 16)
    client.register(b.data_ptr(), 16)
    client.unregister(a.data_ptr())
    # A standalone deregister drops every region: b is still in use.
    assert ("deregister", a.data_ptr()) not in fake.calls
    assert b.data_ptr() in fake.regions
    client.unregister(b.data_ptr())
    assert fake.calls[-1] == ("deregister", b.data_ptr()) and fake.regions == {}
    client.close()
    assert umbp_client._RETAINED_CLIENTS == []
    with pytest.raises(umbp_client.CallRefused, match="closed"):
        client.exists(["a"])
    client.close()  # idempotent


# --- the worker --------------------------------------------------------------


def test_save_then_load_round_trips(server, make_worker):
    worker, gpu = make_worker(save_slots=8, load_slots=8)
    request = _save_req(11, 40)
    operation = request.save_operation
    worker._do_save_req(request, producer_event="fence")
    for index in range(5):
        expected = gpu.pattern(2 * index) + gpu.pattern(2 * index + 1)
        assert server.at()[_key(request.chunk_hashes, index)] == expected
    assert worker.get_finished().connector_completions == {
        _store(operation),
        _quiescent(operation),
        *(_safe(operation, index) for index in range(5)),
    }

    destination = list(range(100, 110))
    load = _load_req(11, hbm=0, lmc=40, block_ids=destination)
    worker._do_load_req(load)
    assert [gpu.kv[block] for block in destination] == [
        gpu.pattern(block) for block in range(10)
    ]
    assert worker.get_finished().finished_loading == {load.load_operation}
    assert worker._pool.quarantined("save") == worker._pool.quarantined("load") == 0


def test_saving_chunks_the_server_already_holds_succeeds(server, make_worker):
    worker, _ = make_worker()
    worker._do_save_req(_save_req(14, 16))
    worker.get_finished()
    again = _save_req(15, 16, generation=1)
    worker._do_save_req(again)
    assert _store(again.save_operation) in worker.get_finished().connector_completions
    assert len(server.at()) == 2


def test_a_put_answered_false_fails_its_save_and_frees_its_slots(server, make_worker):
    worker, _ = make_worker(save_slots=8)
    request = _save_req(16, 40)
    server.fail_puts.add(_key(request.chunk_hashes, 3))
    worker._do_save_req(request)
    assert worker.get_finished().connector_completions >= {
        _store(request.save_operation, False),
        _quiescent(request.save_operation),
    }
    # The call returned: nothing quarantined, nothing marked, loads go on.
    assert worker._pool.quarantined("save") == 0
    assert worker._store_distrusted is None and taint.reason(ADDRESS) is None


def test_a_load_miss_fails_all_or_nothing(server, make_worker):
    worker, gpu = make_worker(load_slots=8)
    worker._do_save_req(_save_req(34, 40))
    request = _load_req(34, hbm=8, lmc=40, block_ids=list(range(100, 110)))
    del server.at()[_key(request.chunk_hashes, 3)]  # evicted
    worker._do_load_req(request)
    assert gpu.calls[-1] == ("to", [8, 16])
    assert worker.get_finished().failed_loading == {request.load_operation}
    assert worker.take_load_error_blocks() == set(range(102, 110))
    assert worker._pool.quarantined("load") == 0


def test_a_get_that_raises_quarantines_its_slots_only(server, make_worker):
    worker, _ = make_worker(load_slots=8)
    worker._do_save_req(_save_req(40, 16))
    server.raise_on = "get"
    load = _load_req(40, hbm=0, lmc=16, block_ids=[8, 9, 10, 11])
    worker._do_load_req(load)
    server.raise_on = None
    assert load.load_operation in worker.get_finished().failed_loading
    assert worker._pool.quarantined("load") == 2
    # A get writes no object: the server stays trusted.
    assert worker._store_distrusted is None and taint.reason(ADDRESS) is None


def test_a_put_that_raises_marks_the_server_in_this_process_only(
    server, make_worker, caplog
):
    worker, _ = make_worker(save_slots=8)
    same_process, _ = make_worker()
    elsewhere, _ = make_worker(address=OTHER_ADDRESS)
    stored = _save_req(50, 16)
    worker._do_save_req(stored)
    elsewhere._do_save_req(stored)
    server.raise_on = "put"
    with caplog.at_level(logging.ERROR, logger="atom"):
        worker._do_save_req(_save_req(51, 16, offset=100))
    server.raise_on = None
    worker.get_finished()
    assert worker._pool.quarantined("save") == 2
    assert "marks it in this process only" in caplog.text
    assert taint.reason(ADDRESS) is not None
    # Nothing was put on the server to mark it.
    assert set(server.at()) == {_key(stored.chunk_hashes, i) for i in range(2)}

    load = _load_req(50, hbm=0, lmc=16, block_ids=[8, 9, 10, 11])
    worker._do_load_req(load)
    assert worker.get_finished().failed_loading == {load.load_operation}
    # Another worker of this process sees the process's mark, and stops too.
    same_process._do_load_req(load)
    assert same_process.get_finished().failed_loading == {load.load_operation}
    # One of another server does not; nor would another process (not told).
    elsewhere._do_load_req(load)
    assert elsewhere.get_finished().finished_loading == {load.load_operation}
    # Saves go on.
    later = _save_req(52, 16, offset=200)
    worker._do_save_req(later)
    assert _store(later.save_operation) in worker.get_finished().connector_completions


# --- startup and close -------------------------------------------------------


def _registering_config(**overrides):
    return _config(extra=_extra(save_pool_mib=1, load_pool_mib=1, **overrides))


@pytest.mark.usefixtures("one_rank")
def test_registration_probes_without_removing_and_close_deregisters_last(
    server, caplog
):
    worker = UMBPStoreOffloadConnector(_registering_config())
    with caplog.at_level(logging.WARNING, logger="atom"):
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
    [fake] = server.clients
    pool = worker._pool
    assert fake.regions == {pool.base_ptr: pool.nbytes}
    assert [call[0] for call in fake.calls] == ["register", "put", "exists", "get"]
    # No remove by key: the probe's object stays until evicted, and says so.
    [probe] = [key for key in server.at() if "/probe/w0/" in key]
    assert "cannot remove by key" in caplog.text and probe in caplog.text
    assert worker._namespace == keys.store_namespace(worker._config, CHUNK)
    assert worker._store_distrusted is None

    worker.close()
    # Executors joined first, then the one region, then the client.
    assert fake.calls[-1] == ("deregister", pool.base_ptr) and fake.regions == {}
    assert umbp_client._RETAINED_CLIENTS == []
    assert worker._client is None and probe in server.at()


@pytest.mark.usefixtures("one_rank")
def test_a_probe_put_answered_false_stops_startup_and_releases_everything(server):
    server.fail_puts = _AnyKey()
    worker = UMBPStoreOffloadConnector(_registering_config())
    with pytest.raises(RuntimeError, match="did not store"):
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
    [fake] = server.clients
    assert fake.calls[-1][0] == "deregister" and fake.regions == {}
    assert worker._client is None and taint.reason(ADDRESS) is None


@pytest.mark.usefixtures("one_rank")
def test_a_probe_put_that_raises_keeps_the_pool_and_marks_the_server(server):
    server.raise_on = "put"
    worker = UMBPStoreOffloadConnector(_registering_config())
    with pytest.raises(RuntimeError, match="put blew up"):
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
    [fake] = server.clients
    # The quarantined slot keeps the pool registered and the client open.
    assert ("deregister",) not in [call[:1] for call in fake.calls]
    assert umbp_client._RETAINED_CLIENTS == [fake]
    assert "probe" in taint.reason(ADDRESS)


@pytest.mark.usefixtures("one_rank")
def test_a_worker_started_after_the_mark_refuses_loads(server):
    taint.mark(ADDRESS, "an earlier put raised")
    worker = UMBPStoreOffloadConnector(_registering_config())
    try:
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
        assert worker._store_distrusted == "an earlier put raised"
    finally:
        worker.close()


def test_close_keeps_a_client_whose_pool_may_still_be_in_use(server, make_worker):
    worker, _ = make_worker()
    client, pool = worker._client, worker._pool
    [fake] = server.clients
    pool.quarantine(pool.acquire("load", 1), reason="test")
    worker.close()
    # The pool stayed registered, so the client was not dropped.
    assert ("deregister", pool.base_ptr) not in fake.calls
    assert fake.regions == {pool.base_ptr: pool.nbytes}
    assert umbp_client._RETAINED_CLIENTS == [fake]
    assert client.__dict__.get("_client") is None


@pytest.mark.usefixtures("one_rank")
def test_m3_registration_packs_every_page_plane(server, monkeypatch):
    from atom.kv_transfer.disaggregation.page_region import page_region
    from atom.kv_transfer.disaggregation.types import KVTransferTensors
    from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec

    monkeypatch.setattr(
        DSV4PageSlotCodec, "has_fused_chunk_major_staging", property(lambda _: True)
    )
    config = _registering_config()
    config.hf_config.model_type = "minimax_m3"
    fields = {
        name: torch.zeros((8, width), dtype=torch.uint8)
        for name, width in [
            ("k", 256),
            ("v", 256),
            ("k_scale", 8),
            ("v_scale", 8),
            ("index", 144),
        ]
    }
    transfer = KVTransferTensors(
        pages=[page_region(t, semantic_role=f"m3.{name}") for name, t in fields.items()]
    )
    transfer.set_block_count(8)
    worker = UMBPStoreOffloadConnector(config)
    try:
        worker.register_kv_caches({}, transfer, num_blocks=8)
        per_block = sum(t.shape[1] for t in fields.values())
        assert worker._codec.bytes_per_block == per_block  # index cache included
        assert worker._chunk_bytes == per_block * CHUNK // BLOCK
        assert worker._namespace != keys.store_namespace(_registering_config(), CHUNK)
    finally:
        worker.close()


# --- the scheduler -----------------------------------------------------------


def test_scheduler_lookups_ask_the_server(server):
    scheduler = UMBPStoreOffloadScheduler(_config(pp=2))
    lookup = scheduler._lookup_client
    assert server.clients == []  # opened on the first lookup
    namespace = keys.store_namespace(_config(pp=2), CHUNK)
    hashes = _hashes(48)
    for rank, chunks in ((0, range(6)), (1, [0, 1, 3, 4, 5])):
        for index in chunks:
            server.at()[
                _key(hashes, index, rank=rank, world=2, namespace=namespace)
            ] = b""
    assert lookup.lookup(_seq(0, 48)) == 2 * CHUNK  # rank 1 lacks chunk 2
    [fake] = server.clients
    assert fake.address == ADDRESS and fake.regions == {}
    [asked] = [call[1] for call in fake.calls if call[0] == "exists"]
    assert len(asked) == 12  # no marker key asked for
    assert (lookup.uneven_lookups, lookup.stranded_objects) == (1, 4)


def test_a_lookup_that_raises_answers_nothing_and_pauses(server):
    scheduler = UMBPStoreOffloadScheduler(_config())
    lookup = scheduler._lookup_client
    server.raise_on = "exists"
    assert lookup.lookup(_seq(0, 16)) is None
    assert lookup.paused_until > 0
    server.raise_on = None
    assert lookup.lookup(_seq(0, 16)) is None  # still paused
    lookup.paused_until = 0.0
    assert lookup.lookup(_seq(0, 16)) == 0


def test_an_unreachable_server_answers_nothing(server):
    scheduler = UMBPStoreOffloadScheduler(_config())
    server.connect_error = RuntimeError("no server")
    assert scheduler._lookup_client.lookup(_seq(0, 16)) is None
    assert scheduler._lookup_client.client is None


def test_lookups_in_a_process_that_marked_the_server_hold_nothing(server):
    scheduler = UMBPStoreOffloadScheduler(_config())
    hashes = _hashes(16)
    namespace = keys.store_namespace(_config(), CHUNK)
    for index in range(2):
        server.at()[_key(hashes, index, namespace=namespace)] = b""
    assert scheduler._lookup_client.lookup(_seq(0, 16)) == 2 * CHUNK
    taint.mark(ADDRESS, "a put raised")
    assert scheduler._lookup_client.lookup(_seq(0, 16)) == 0


def test_a_load_is_dispatched_without_asking_again(server):
    scheduler = UMBPStoreOffloadScheduler(_config())
    spec = LoadSpec(
        hbm_cached_tokens=0,
        lmcache_cached_tokens=16,
        can_load=True,
        transfer_end_tokens=None,
    )
    assert scheduler._ensure_lookup_pin(_seq(0, 16), "0", spec) is True
    assert server.clients == []


class _AnyKey:
    """A key set that holds every key."""

    def __contains__(self, key):
        return True
