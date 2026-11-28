# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The UMBP offload connector (``umbp_store``), on CPU.

What is UMBP's own: its settings, the client's bool-per-key answers, settling
a call by its answer, the process-local taint mark, no remove or groups, and
lookups against one server, and copies staged on the forward's stream as
the shared layer stages them. The windowed put/get loop, the completion
protocol and held-chunk skipping are the shared ``kv_store`` layer's and are
tested through ``mooncake_store`` (``test_mooncake_store_offload.py``); the
round trips here run that loop over UMBP's answers.

UMBP is faked by a ``UMBPClient`` stand-in that moves bytes by address
(ctypes) between registered host buffers and one dict per standalone server,
answering one bool per key as the real binding does. The block GPU connector
is faked by one that packs a byte pattern per KV block.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
import types
from types import SimpleNamespace

import kv_store_test_support
import pytest
import torch
from kv_store_test_support import (
    BLOCK,
    BYTES_PER_BLOCK,
    CHUNK,
    CHUNK_BYTES,
    SLOT,
    FakeGPUConnector,
    all_slots_free,
    chunk_key,
    engine_config,
    install_fake_aiter,
    use_one_rank,
)
from kv_store_test_support import chunk_hashes as _hashes
from kv_store_test_support import kv_caches as _kv_caches
from kv_store_test_support import load_req as _load_req
from kv_store_test_support import quiescent as _quiescent
from kv_store_test_support import save_req as _save_req
from kv_store_test_support import seq as _seq
from kv_store_test_support import source_safe as _safe
from kv_store_test_support import stored as _store

from atom.config import ParallelConfig
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload.kv_store import keys
from atom.kv_transfer.offload.kv_store import pool as pool_mod
from atom.kv_transfer.offload.metadata import LoadSpec
from atom.kv_transfer.offload.umbp_store import client as umbp_client
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store import worker as umbp_worker_mod
from atom.kv_transfer.offload.umbp_store.config import (
    UMBPStoreOffloadConfig,
    check_engine_compatibility,
    parse_umbp_store_config,
    store_namespace,
)
from atom.kv_transfer.offload.umbp_store.scheduler import UMBPStoreOffloadScheduler
from atom.kv_transfer.offload.umbp_store.worker import (
    TransferSlotPool,
    UMBPStoreOffloadConnector,
)

ADDRESS = "unix:///tmp/umbp-test/server.sock"
OTHER_ADDRESS = "unix:///tmp/umbp-test/other.sock"
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
        # A call every client answers only after this many seconds.
        self.slow_on: tuple[str, float] | None = None
        # Puts answered False before any is answered as usual.
        self.refuse_first_puts = 0
        # Results every client answers short of: (call, how many fewer).
        self.short_on: tuple[str, int] | None = None
        self.refuse_registration = False
        self.deregister_error: Exception | None = None

    def at(self, address: str = ADDRESS) -> dict[str, bytes]:
        return self.objects.setdefault(address, {})

    def connect(
        self, address: str, startup_timeout_ms: int, lookup_timeout_ms: int
    ) -> FakeUMBPClient:
        if self.connect_error is not None:
            raise self.connect_error
        client = FakeUMBPClient(self, address, startup_timeout_ms)
        client.lookup_timeout_ms = lookup_timeout_ms
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
        if self.server.slow_on is not None and self.server.slow_on[0] == call:
            time.sleep(self.server.slow_on[1])

    def _shorten(self, call, answers):
        if self.server.short_on is not None and self.server.short_on[0] == call:
            return answers[: len(answers) - self.server.short_on[1]]
        return answers

    def _is_registered(self, ptr, size):
        return any(
            base <= ptr and ptr + size <= base + n for base, n in self.regions.items()
        )

    def register_memory(self, ptr, size):
        self.calls.append(("register", ptr, size))
        if self.server.refuse_registration:
            return False
        self.regions[ptr] = size
        return True

    def deregister_memory(self, ptr):
        # Standalone: one call drops every region of the client.
        self.calls.append(("deregister", ptr))
        if self.server.deregister_error is not None:
            raise self.server.deregister_error
        self.regions.clear()

    def batch_exists(self, keys):
        self.calls.append(("exists", list(keys)))
        self._maybe_raise("exists")
        return self._shorten("exists", [key in self.objects for key in keys])

    def batch_put_from_ptr(self, keys, ptrs, sizes):
        self.calls.append(("put", list(keys)))
        self._maybe_raise("put")
        if self.server.refuse_first_puts:
            self.server.refuse_first_puts -= 1
            return [False] * len(keys)
        answers = []
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            if key in self.server.fail_puts or not self._is_registered(ptr, size):
                answers.append(False)
                continue
            # Immutable keys: an existing one keeps its bytes and succeeds.
            self.objects.setdefault(key, ctypes.string_at(ptr, size))
            answers.append(True)
        return self._shorten("put", answers)

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


# Every test once per save path (the shared worker's serial save and the
# opt-in save pipeline); see `kv_store_test_support.save_mode`.
save_mode = kv_store_test_support.save_mode


@pytest.fixture(autouse=True)
def aiter_arch(monkeypatch):
    """AITER's dtype table and arch detection, as on gfx950 to start; setting
    ``aiter_arch.gfx = "gfx942"`` makes it an MI300X."""
    return install_fake_aiter(monkeypatch)


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
    use_one_rank(monkeypatch)


def _extra(**overrides):
    extra = {"umbp_store.address": ADDRESS, "umbp_store.chunk_tokens": CHUNK}
    for name, value in overrides.items():
        if value is None:
            extra.pop(f"umbp_store.{name}", None)
        else:
            extra[f"umbp_store.{name}"] = value
    return extra


def _config(role="offload", *, pp=1, tp=1, layers=4, extra=None, **fields):
    return engine_config(
        "umbp_store",
        _extra() if extra is None else extra,
        role,
        pp=pp,
        tp=tp,
        layers=layers,
        **fields,
    )


def _key(hashes, index, *, rank=0, world=1, namespace=NAMESPACE):
    return chunk_key(namespace, hashes, index, rank=rank, world=world)


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
        lookup_timeout_ms=1000,
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
        # 0 is Mori's "wait for ever", which would stop the scheduler thread.
        (_extra(lookup_timeout_ms=0), "lookup_timeout_ms"),
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
    # The stateful layouts and V4.1's replay PAGEs, which the Mooncake Store
    # offload moves, are not UMBP's.
    hybrid = _config()
    hybrid.hf_config.compress_ratios = [4, 128]
    with pytest.raises(ValueError, match="resolves to 'hybrid'"):
        check_engine_compatibility(cfg, hybrid)
    v41 = _config(enable_encoder_swa_bounded_replay=True)
    v41.hf_config.model_type = "deepseek_v41"
    v41.hf_config.compress_ratios = [4, 128]
    with pytest.raises(ValueError, match="resolves to 'dsv41'"):
        check_engine_compatibility(cfg, v41)
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


@pytest.mark.parametrize("producer", ["mooncake", "moriio"])
def test_multi_refuses_a_pd_producer_beside_umbp_offload(producer):
    """PD with UMBP is not supported: the prefill side is refused in `Config`,
    where `validate_multi_config` runs before the model loads."""
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        validate_multi_config,
    )

    def multi(offload):
        return {
            "kv_connector": "multi",
            "connectors": [
                {"kv_connector": producer, "kv_role": "kv_producer"},
                {"kv_connector": offload, "kv_role": "offload"},
            ],
        }

    with pytest.raises(ValueError, match="umbp_store beside a P/D connector"):
        validate_multi_config(multi("umbp_store"))
    # The Mooncake Store offload keeps running beside a producer.
    assert len(validate_multi_config(multi("mooncake_store"))) == 2


def test_loads_hits_from_the_mooncake_store_floor():
    # The same 256-token chunks as mooncake_store, so the same smallest load.
    assert _config().offload_min_load_tokens == 512


@pytest.mark.parametrize("dp_local", [2, 1])
def test_shared_prefix_routing_with_umbp_is_single_node_only(dp_local):
    # What `--dp-load-balance share_cache|prefix_spread` asks at startup, on
    # the engine Config: the DP placement lives in its `parallel_config`.
    config = _config(
        parallel_config=ParallelConfig(
            data_parallel_size=2, data_parallel_size_local=dp_local
        )
    )
    if dp_local == 2:  # both DP ranks on this node
        assert KVConnectorFactory.offload_tier(config).loads
        return
    with pytest.raises(ValueError, match="single-node only"):
        KVConnectorFactory.offload_tier(config)


def test_a_tp_group_over_several_nodes_is_refused():
    cfg = parse_umbp_store_config(_extra())
    check_engine_compatibility(cfg, _config(tp=4))
    two_nodes = _config(
        tp=4, parallel_config=ParallelConfig(tensor_parallel_size_local=2)
    )
    assert two_nodes.is_multinode_tp
    with pytest.raises(ValueError, match="several nodes"):
        check_engine_compatibility(cfg, two_nodes)


def test_namespace_is_umbps_own():
    namespace = store_namespace(_config(), CHUNK)
    assert namespace.startswith("atomumbp1-")
    assert not namespace.startswith("atomkv1-")
    m3 = _config()
    m3.hf_config.model_type = "minimax_m3"
    assert store_namespace(m3, CHUNK) != namespace  # codec id folded in
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


def test_a_deregister_the_server_fails_is_logged_and_teardown_goes_on(server, caplog):
    # The server was stopped first: the RPC fails, the close must not.
    client = umbp_client.UMBPStoreClient(address=ADDRESS)
    buffer = _host_buffer(16)
    client.register(buffer.data_ptr(), 16)
    server.deregister_error = RuntimeError("standalone DeregisterMemory RPC failed")
    with caplog.at_level(logging.WARNING, logger="atom"):
        client.unregister(buffer.data_ptr())
    assert "deregistering the buffer" in caplog.text
    client.close()
    assert umbp_client._RETAINED_CLIENTS == []


@pytest.mark.parametrize("call", ["exists", "put"])
def test_an_answer_of_the_wrong_length_raises(server, call):
    client = umbp_client.UMBPStoreClient(address=ADDRESS)
    buffer = _host_buffer(2 * CHUNK_BYTES)
    client.register(buffer.data_ptr(), 2 * CHUNK_BYTES)
    server.short_on = (call, 1)
    with pytest.raises(RuntimeError, match="returned 1 results for 2 keys"):
        if call == "exists":
            client.exists(["a", "b"])
        else:
            client.put(
                ["a", "b"],
                [buffer.data_ptr(), buffer.data_ptr() + CHUNK_BYTES],
                [CHUNK_BYTES] * 2,
            )


def test_clients_bound_their_lookups(server):
    umbp_client.UMBPStoreClient(address=ADDRESS, lookup_timeout_ms=250)
    [fake] = server.clients
    assert fake.lookup_timeout_ms == 250


@pytest.mark.usefixtures("one_rank")
def test_a_worker_client_takes_the_configured_lookup_deadline(server):
    worker = UMBPStoreOffloadConnector(_registering_config(lookup_timeout_ms=250))
    try:
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
        [fake] = server.clients
        assert fake.lookup_timeout_ms == 250
    finally:
        worker.close()


class _FakeMoriProcessConfig:
    """`UMBPStandaloneProcessConfig`: a fixed set of fields, as the binding's
    `def_readwrite` attributes are; an unknown one raises."""

    __slots__ = ("address", "auto_start", "lookup_timeout_ms", "startup_timeout_ms")


class _FakeMoriOldProcessConfig:
    """The same, from a Mori without ROCm/mori#949 (no lookup deadline)."""

    __slots__ = ("address", "auto_start", "startup_timeout_ms")


def _fake_mori(monkeypatch, process_config):
    made = []

    class UMBPConfig:
        standalone_process = None

    class UMBPClient:
        def __init__(self, config):
            made.append(config)

    module = types.ModuleType("mori.umbp")
    module.UMBPClient = UMBPClient
    module.UMBPConfig = UMBPConfig
    module.UMBPStandaloneProcessConfig = process_config
    monkeypatch.setitem(sys.modules, "mori", types.ModuleType("mori"))
    monkeypatch.setitem(sys.modules, "mori.umbp", module)
    return made


def test_the_mori_client_is_configured_for_a_standalone_server(monkeypatch):
    made = _fake_mori(monkeypatch, _FakeMoriProcessConfig)
    umbp_client.UMBPStoreClient(
        address=ADDRESS, startup_timeout_ms=1234, lookup_timeout_ms=250
    )
    [config] = made
    process = config.standalone_process
    assert process.address == ADDRESS
    assert process.auto_start is False  # the server is started outside ATOM
    assert (process.startup_timeout_ms, process.lookup_timeout_ms) == (1234, 250)


def test_a_mori_without_the_lookup_deadline_is_refused(monkeypatch):
    made = _fake_mori(monkeypatch, _FakeMoriOldProcessConfig)
    with pytest.raises(RuntimeError, match="lookup_timeout_ms"):
        umbp_client.UMBPStoreClient(address=ADDRESS)
    assert made == []


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
    assert all_slots_free(worker._pool, "save")
    assert all_slots_free(worker._pool, "load")


class FakeStridedCodec:
    """A PAGE codec's in-place API over host bytes: restores padded slots."""

    has_fused_chunk_major_staging = True
    supports_strided_restore = True

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.kv: dict[int, bytes] = {}
        self.calls: list[tuple] = []

    def prepare_block_id_groups(self, groups, *, device, stream):
        return SimpleNamespace(groups=groups, stream=stream)

    def chunk_major_strided_buffer_to_gpu_prepared(
        self, buf, owner, index, *, stream, items_per_chunk, chunk_stride_bytes
    ):
        assert stream is owner.stream
        data = buf.numpy().tobytes()
        for chunk, blocks in enumerate(owner.groups[index]):
            assert len(blocks) == items_per_chunk
            for item, block in enumerate(blocks):
                start = chunk * chunk_stride_bytes + item * BYTES_PER_BLOCK
                self.kv[block] = data[start : start + BYTES_PER_BLOCK]
        self.calls.append(("strided", items_per_chunk, chunk_stride_bytes))


def test_a_load_restores_padded_slots_in_place(server, make_worker, monkeypatch):
    from atom.kv_transfer.offload import atom_lmcache_staging as staging

    worker, gpu = make_worker(save_slots=8, load_slots=8)
    codec = FakeStridedCodec()
    stream = SimpleNamespace(synchronize=lambda: None)
    worker._codec, worker._in_place = codec, True
    monkeypatch.setattr(worker, "_in_place_stream", lambda: stream)
    monkeypatch.setattr(staging, "fence_submitted_work", lambda s: s.synchronize())
    assert worker._pool.slot_stride == SLOT != CHUNK_BYTES

    request = _save_req(12, 40)
    worker._do_save_req(request, producer_event="fence")
    # A padded slot is no chunk-major buffer: the save stages.
    assert [call[0] for call in gpu.calls] == ["from"] * len(gpu.calls)
    worker.get_finished()

    destination = list(range(100, 110))
    load = _load_req(12, hbm=0, lmc=40, block_ids=destination)
    worker._do_load_req(load)
    # The load reads the padded run where it lies, never through staging.
    assert not any(call[0] == "to" for call in gpu.calls)
    assert codec.calls and set(codec.calls) == {("strided", CHUNK // BLOCK, SLOT)}
    assert [codec.kv[block] for block in destination] == [
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


def _counting_source_safe(worker, gpu):
    """Count every source-safe report of ``worker``, by chunk index."""
    reported = []
    report = worker._source_group_safe

    def counting(identity):
        [(start, _end)] = identity.ranges
        reported.append(start // CHUNK)
        report(identity)

    worker._source_group_safe = counting
    gpu.source_safe = counting
    return reported


def _puts(server):
    return [call[1] for c in server.clients for call in c.calls if call[0] == "put"]


def test_a_save_puts_only_the_chunks_the_server_lacks(server, make_worker):
    worker, gpu = make_worker(save_slots=8)
    request = _save_req(19, 48)
    hashes = request.chunk_hashes
    for index in (1, 3, 4):
        server.at()[_key(hashes, index)] = b"held" * (CHUNK_BYTES // 4)
    reported = _counting_source_safe(worker, gpu)
    worker._do_save_req(request)
    # Packed and put tail first, only the chunks missing, in runs.
    assert gpu.calls == [("from", [40]), ("from", [16]), ("from", [0])]
    assert _puts(server)[-3:] == [
        [_key(hashes, 5)],
        [_key(hashes, 2)],
        [_key(hashes, 0)],
    ]
    for index in (0, 2, 5):
        expected = gpu.pattern(2 * index) + gpu.pattern(2 * index + 1)
        assert server.at()[_key(hashes, index)] == expected
    for index in (1, 3, 4):
        assert server.at()[_key(hashes, index)] == b"held" * (CHUNK_BYTES // 4)
    assert sorted(reported) == list(range(6))
    operation = request.save_operation
    assert worker.get_finished().connector_completions == {
        _store(operation),
        _quiescent(operation),
        *(_safe(operation, index) for index in range(6)),
    }
    # One lookup per window; UMBP's True is a hit, its False a chunk to put.
    assert worker._save_counts == {"lookups": 3, "put_chunks": 3, "held_chunks": 3}
    assert worker._pool.quarantined("save") == 0


def test_a_put_answered_false_fails_its_save_and_frees_its_slots(server, make_worker):
    worker, _ = make_worker(save_slots=8)
    request = _save_req(16, 40)
    server.fail_puts.add(_key(request.chunk_hashes, 3))
    worker._do_save_req(request)
    assert worker.get_finished().connector_completions >= {
        _store(request.save_operation, False),
        _quiescent(request.save_operation),
    }
    # The call returned: nothing quarantined, every slot back, nothing
    # marked, loads go on.
    assert worker._pool.quarantined("save") == 0
    assert all_slots_free(worker._pool, "save")
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
    assert all_slots_free(worker._pool, "load")


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
    assert worker._namespace == store_namespace(worker._config, CHUNK)
    assert worker._store_distrusted is None

    worker.close()
    # Executors joined first, then the one region, then the client.
    assert fake.calls[-1] == ("deregister", pool.base_ptr) and fake.regions == {}
    assert umbp_client._RETAINED_CLIENTS == []
    assert worker._client is None and probe in server.at()


@pytest.fixture
def fast_probe_retries(monkeypatch):
    monkeypatch.setattr(umbp_worker_mod, "_PROBE_RETRY_INTERVAL_S", 0.001)


@pytest.mark.usefixtures("one_rank", "fast_probe_retries")
def test_a_probe_put_answered_false_stops_startup_and_releases_everything(server):
    server.fail_puts = _AnyKey()
    worker = UMBPStoreOffloadConnector(_registering_config(startup_timeout_ms=30))
    with pytest.raises(RuntimeError, match="did not store .*startup_timeout_ms=30"):
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
    [fake] = server.clients
    # Asked again until the startup timeout, then given up.
    assert [call[0] for call in fake.calls].count("put") > 1
    assert fake.calls[-1][0] == "deregister" and fake.regions == {}
    assert worker._client is None and taint.reason(ADDRESS) is None


@pytest.mark.usefixtures("one_rank", "fast_probe_retries")
def test_a_probe_put_refused_while_the_server_starts_is_asked_again(server):
    # A server still registering its memory answers False to early puts.
    server.refuse_first_puts = 3
    worker = UMBPStoreOffloadConnector(_registering_config())
    try:
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
        [fake] = server.clients
        assert [call[0] for call in fake.calls] == [
            "register",
            *["put"] * 4,
            "exists",
            "get",
        ]
        assert worker._store_distrusted is None
        pool = worker._pool
        assert pool.quarantined("save") == pool.quarantined("load") == 0
        # Every slot is back: all of them can be taken again.
        pool.release(pool.acquire("save", pool.capacity("save")))
    finally:
        worker.close()


def _probe_not_listed(fake, keys):
    return [False] * len(keys)


def _probe_reads_zeros(fake, keys, ptrs, sizes):
    for ptr, size in zip(ptrs, sizes, strict=True):
        ctypes.memset(ptr, 0, size)
    return [True] * len(keys)


@pytest.mark.parametrize(
    ("call", "answer", "match"),
    [
        ("batch_exists", _probe_not_listed, "not on the server"),
        ("batch_get_into_ptr", None, "did not return"),
        ("batch_get_into_ptr", _probe_reads_zeros, "differs from the one written"),
    ],
)
@pytest.mark.usefixtures("one_rank")
def test_a_probe_that_cannot_read_its_chunk_back_stops_startup(
    server, monkeypatch, call, answer, match
):
    if answer is None:
        server.fail_gets = _AnyKey()
    else:
        monkeypatch.setattr(FakeUMBPClient, call, answer)
    worker = UMBPStoreOffloadConnector(_registering_config())
    with pytest.raises(RuntimeError, match=match):
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
    [fake] = server.clients
    # Nothing raised inside UMBP: every slot was released, the pool and the
    # client closed, and the server is not marked.
    assert fake.calls[-1][0] == "deregister" and fake.regions == {}
    assert umbp_client._RETAINED_CLIENTS == []
    assert worker._client is None and taint.reason(ADDRESS) is None


@pytest.mark.usefixtures("one_rank")
def test_a_pool_the_server_will_not_register_stops_startup(server):
    server.refuse_registration = True
    worker = UMBPStoreOffloadConnector(_registering_config())
    with pytest.raises(RuntimeError, match="could not register"):
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
    [fake] = server.clients
    assert "put" not in [call[0] for call in fake.calls]
    assert worker._client is None


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
        assert worker._namespace != store_namespace(_registering_config(), CHUNK)
    finally:
        worker.close()


# --- the scheduler -----------------------------------------------------------


def test_scheduler_lookups_ask_the_server(server):
    scheduler = UMBPStoreOffloadScheduler(_config(pp=2))
    lookup = scheduler._lookup_client
    assert server.clients == []  # opened on the first lookup
    namespace = store_namespace(_config(pp=2), CHUNK)
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


def test_a_lookup_that_takes_its_deadline_answers_nothing_and_pauses(server):
    # Past the deadline Mori answers every key absent; that is not a miss.
    scheduler = UMBPStoreOffloadScheduler(_config(extra=_extra(lookup_timeout_ms=5)))
    lookup = scheduler._lookup_client
    hashes = _hashes(16)
    namespace = store_namespace(_config(), CHUNK)
    for index in range(2):
        server.at()[_key(hashes, index, namespace=namespace)] = b""
    server.slow_on = ("exists", 0.02)
    lookup._client_to_ask()
    [fake] = server.clients
    assert fake.lookup_timeout_ms == 5
    assert lookup.lookup(_seq(0, 16)) is None
    assert lookup.paused_until > time.monotonic()
    server.slow_on = None
    assert lookup.lookup(_seq(0, 16)) is None  # still paused
    lookup.paused_until = 0.0
    assert lookup.lookup(_seq(0, 16)) == 2 * CHUNK


@pytest.mark.parametrize(("blocked_s", "answered"), [(0.999, True), (1.0, False)])
def test_a_lookup_that_blocked_for_the_whole_deadline_is_no_answer(
    server, monkeypatch, blocked_s, answered
):
    """The boundary: Mori gives up at the deadline, so a call that took all of
    it was answered by the deadline, not by the server."""
    from atom.kv_transfer.offload.umbp_store import scheduler as umbp_scheduler

    clock = [100.0]
    monkeypatch.setattr(
        umbp_scheduler,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter),
    )
    original = FakeUMBPClient.batch_exists

    def blocking_exists(self, keys):
        clock[0] += blocked_s
        return original(self, keys)

    monkeypatch.setattr(FakeUMBPClient, "batch_exists", blocking_exists)
    scheduler = UMBPStoreOffloadScheduler(_config(extra=_extra(lookup_timeout_ms=1000)))
    hashes = _hashes(16)
    namespace = store_namespace(_config(), CHUNK)
    for index in range(2):
        server.at()[_key(hashes, index, namespace=namespace)] = b""
    answer = scheduler._lookup_client.lookup(_seq(0, 16))
    assert answer == (2 * CHUNK if answered else None)


def test_an_unreachable_server_answers_nothing(server):
    scheduler = UMBPStoreOffloadScheduler(_config())
    server.connect_error = RuntimeError("no server")
    assert scheduler._lookup_client.lookup(_seq(0, 16)) is None
    assert scheduler._lookup_client.client is None


def test_lookups_in_a_process_that_marked_the_server_hold_nothing(server):
    scheduler = UMBPStoreOffloadScheduler(_config())
    hashes = _hashes(16)
    namespace = store_namespace(_config(), CHUNK)
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


# --- copy streams --------------------------------------------------------------


@pytest.mark.usefixtures("one_rank")
def test_registration_stages_on_the_forward_stream(server):
    """No side streams: the copies follow the stream the runner reports, in
    place ones too, as the shared layer stages them by default."""
    assert UMBPStoreOffloadConnector.STAGES_ON_FORWARD_STREAM
    worker = UMBPStoreOffloadConnector(_registering_config())
    try:
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
        gpu = worker._gpu_connector
        assert gpu._compute_stream_provider is not None
        worker.set_forward_stream("forward-stream")
        assert gpu._compute_stream_provider() == "forward-stream"
        worker._pin_in_place_stream()
        assert worker._in_place_stream() == "forward-stream"
    finally:
        worker.close()
