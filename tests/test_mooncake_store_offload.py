# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The native Mooncake Store offload connector (``mooncake_store``), on CPU.

Mooncake is faked by a ``MooncakeDistributedStore`` stand-in that moves bytes
by address (ctypes) between registered host buffers and one dict per master,
so puts and gets are zero-copy here as they are on the NIC. The block GPU
connector is faked by one that packs a byte pattern per KV block. sysfs is a
temporary tree shaped like a rail-optimized MI355X node (pit2-p03-g40), where
HIP numbers the GPUs in KFD order, not PCI order.
"""

from __future__ import annotations

import array
import copy
import ctypes
import json
import logging
import sys
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import MockConfig

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL,
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store import keys, nic
from atom.kv_transfer.offload.mooncake_store import pool as pool_mod
from atom.kv_transfer.offload.mooncake_store import scheduler as scheduler_mod
from atom.kv_transfer.offload.mooncake_store import worker as worker_mod
from atom.kv_transfer.offload.mooncake_store.config import (
    check_engine_compatibility,
    parse_mooncake_store_config,
)
from atom.kv_transfer.offload.mooncake_store.pool import (
    SlotPoolExhausted,
    TransferSlotPool,
)
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    MooncakeStoreOffloadScheduler,
    _PromptView,
    prompt_chunk_hashes,
)
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)
from atom.model_engine.arg_utils import (
    compose_kv_offload_config,
    kv_offload_connector_config,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams

MASTER = "10.0.0.1:26051"
METADATA = "http://10.0.0.1:26080/metadata"
BLOCK = 4  # tokens per KV block
CHUNK = 8  # tokens per Store chunk: two blocks
BYTES_PER_BLOCK = 16
CHUNK_BYTES = BYTES_PER_BLOCK * CHUNK // BLOCK
SLOT = 4096  # one slot's stride for CHUNK_BYTES
NAMESPACE = "atomkv1-test"


# --- fakes -------------------------------------------------------------------


class FakeCluster:
    """The masters a test's Store clients reach, each holding its objects."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, bytes]] = {}
        # Per master: the Mooncake group each grouped key was put in.
        self.groups: dict[str, dict[str, str]] = {}
        self.stores: list[FakeStore] = []
        self.setup_rc = 0
        # A get that reports success but never writes: the probe must notice.
        self.silent_gets = False
        # The result of every put not in a store's put_codes, if set.
        self.put_code: int | None = None
        # A call every store raises from, as `FakeStore.raise_on`.
        self.raise_on: str | None = None

    def new_store(self) -> FakeStore:
        store = FakeStore(self)
        self.stores.append(store)
        return store

    def store_of(self, master: str) -> FakeStore:
        return next(
            s for s in self.stores if s.setup_args and s.setup_args[6] == master
        )


class FakeStore:
    """``MooncakeDistributedStore`` over host memory, zero-copy by address."""

    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster
        self.objects: dict[str, bytes] = {}
        self.groups: dict[str, str] = {}
        # Each put's group ids; None for a put without a ReplicateConfig.
        self.put_group_ids: list[list[str] | None] = []
        self.setup_args: tuple | None = None
        self.registered: dict[int, int] = {}
        self.calls: list[tuple] = []
        self.put_codes: dict[str, int] = {}
        self.get_codes: dict[str, int] = {}
        self.exist_codes: dict[str, int] = {}
        self.register_rc = 0
        self.raise_on: str | None = None
        self.closed = False

    def setup(self, *args):
        self.setup_args = args
        if self.cluster.setup_rc:
            return self.cluster.setup_rc
        self.objects = self.cluster.objects.setdefault(args[6], {})
        self.groups = self.cluster.groups.setdefault(args[6], {})
        return 0

    def register_buffer(self, ptr, size):
        self.calls.append(("register", ptr, size))
        if self.register_rc:
            return self.register_rc
        self.registered[ptr] = size
        return 0

    def unregister_buffer(self, ptr):
        self.calls.append(("unregister", ptr))
        return 0 if self.registered.pop(ptr, None) is not None else -600

    def _is_registered(self, ptr, size):
        return any(
            base <= ptr and ptr + size <= base + n
            for base, n in self.registered.items()
        )

    def _maybe_raise(self, call):
        if call in (self.raise_on, self.cluster.raise_on):
            raise RuntimeError(f"{call} blew up")

    def batch_is_exist(self, keys):
        self.calls.append(("exists", list(keys)))
        self._maybe_raise("exists")
        return [self.exist_codes.get(k, int(k in self.objects)) for k in keys]

    def batch_put_from(self, keys, ptrs, sizes, config=None):
        self.calls.append(("put", list(keys)))
        group_ids = None if config is None else list(config.group_ids)
        self.put_group_ids.append(group_ids)
        self._maybe_raise("put")
        if group_ids is not None and len(group_ids) != len(keys):
            return [store_client.INVALID_PARAMS] * len(keys)
        codes = []
        for index, (key, ptr, size) in enumerate(zip(keys, ptrs, sizes, strict=True)):
            code = self.put_codes.get(key, self.cluster.put_code)
            if code is None:
                if not self._is_registered(ptr, size):
                    code = store_client.TRANSFER_FAIL
                else:
                    # An existing key is a success and keeps its bytes and group.
                    if key not in self.objects:
                        self.objects[key] = ctypes.string_at(ptr, size)
                        if group_ids and group_ids[index]:
                            self.groups[key] = group_ids[index]
                    code = 0
            codes.append(code)
        return codes

    def batch_get_into(self, keys, ptrs, sizes):
        self.calls.append(("get", list(keys)))
        self._maybe_raise("get")
        codes = []
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            code = self.get_codes.get(key)
            if code is None:
                data = self.objects.get(key)
                if data is None:
                    code = store_client.OBJECT_NOT_FOUND
                elif len(data) > size:
                    code = store_client.INVALID_PARAMS
                elif not self._is_registered(ptr, len(data)):
                    code = store_client.TRANSFER_FAIL
                else:
                    if not self.cluster.silent_gets:
                        ctypes.memmove(ptr, data, len(data))
                    code = len(data)
            codes.append(code)
        return codes

    def remove(self, key, force=False):
        self.calls.append(("remove", key, force))
        return 0 if self.objects.pop(key, None) is not None else -704

    def close(self):
        self.closed = True
        return 0


class FakeGPUConnector:
    """Packs each KV block as a byte pattern, as the block GPU connector would.

    ``kv`` is the GPU KV as block id -> bytes; a block never written holds a
    pattern derived from its id. A save packs tail-to-head and publishes one
    source-safe group per chunk inside ``track_save_source``, as the real
    connector does once a group's copy has fenced.
    """

    gpu_staging_chunk_bytes = CHUNK_BYTES
    gpu_staging_buffer_bytes = 4 * CHUNK_BYTES

    def __init__(self, source_safe=None) -> None:
        self.kv: dict[int, bytes] = {}
        self.calls: list[tuple] = []
        self.source_safe = source_safe
        self.fail_from = False
        self.fail_to = False
        self.stats: dict = {}
        self.events: list = []
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
        self.events.append(producer_event)
        if self.fail_from:
            raise RuntimeError("pack failed")
        for slot, start, end in sorted(
            zip(slots, starts, ends, strict=True), key=lambda item: -item[1]
        ):
            blocks = block_ids[start // BLOCK : end // BLOCK]
            data = b"".join(self.kv.get(b, self.pattern(b)) for b in blocks)
            slot.tensor.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
            if self._operation is not None and self.source_safe is not None:
                self.source_safe(SaveSourceGroupId(self._operation, ((start, end),)))
        self.stats = {"producer_fenced": int(producer_event is not None)}

    def batched_to_gpu(self, slots, starts, ends, *, block_ids):
        self.calls.append(("to", list(starts)))
        if self.fail_to:
            raise RuntimeError("unpack failed")
        for slot, start, end in zip(slots, starts, ends, strict=True):
            data = slot.tensor.numpy().tobytes()
            for i, block in enumerate(block_ids[start // BLOCK : end // BLOCK]):
                self.kv[block] = data[i * BYTES_PER_BLOCK : (i + 1) * BYTES_PER_BLOCK]

    def last_transfer_stats(self):
        return dict(self.stats)

    def reset_transfer_stats(self):
        self.stats = {}

    def close(self):
        self.closed = True


_AITER_FP8 = {"gfx942": torch.float8_e4m3fnuz, "gfx950": torch.float8_e4m3fn}


class _FakeAiterDtypes:
    """``aiter.dtypes`` as the arch in ``state`` resolves it."""

    def __init__(self, state) -> None:
        self._state = state

    @property
    def d_dtypes(self):
        return {"fp8": _AITER_FP8[self._state.gfx], "bf16": torch.bfloat16}


@pytest.fixture(autouse=True)
def aiter_arch(monkeypatch):
    """AITER's dtype table and arch detection, as on gfx950 to start.

    The namespace asks AITER what ``fp8`` is stored as; CPU runners have no
    AITER. Setting ``aiter_arch.gfx = "gfx942"`` makes it an MI300X.
    """
    state = SimpleNamespace(gfx="gfx950")
    package = types.ModuleType("aiter")
    package.dtypes = _FakeAiterDtypes(state)
    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx = lambda: state.gfx
    for name, module in (
        ("aiter", package),
        ("aiter.jit", types.ModuleType("aiter.jit")),
        ("aiter.jit.utils", types.ModuleType("aiter.jit.utils")),
        ("aiter.jit.utils.chip_info", chip_info),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return state


@pytest.fixture
def cluster(monkeypatch):
    cluster = FakeCluster()
    monkeypatch.setattr(store_client, "_new_distributed_store", cluster.new_store)
    monkeypatch.setattr(
        store_client,
        "_new_replicate_config",
        lambda group_ids: SimpleNamespace(group_ids=list(group_ids)),
    )
    # Pinned host memory needs a GPU runtime; plain host memory holds the same
    # bytes and has an address the fake Store reads and writes.
    monkeypatch.setattr(
        pool_mod,
        "_allocate_pool_tensor",
        lambda nbytes, device: torch.empty((nbytes,), dtype=torch.uint8),
    )
    # What closed pools kept for the life of the process, per test.
    monkeypatch.setattr(pool_mod, "_UNSETTLED_ALLOCATIONS", [])
    return cluster


def _extra(**overrides):
    extra = {
        "mooncake_store.master": MASTER,
        "mooncake_store.metadata": METADATA,
        "mooncake_store.local_hostname": "10.0.0.2",
        "mooncake_store.protocol": "tcp",
        "mooncake_store.chunk_tokens": CHUNK,
    }
    for name, value in overrides.items():
        if value is None:
            extra.pop(f"mooncake_store.{name}", None)
        else:
            extra[f"mooncake_store.{name}"] = value
    return extra


def _config(role="offload", *, pp=1, tp=1, layers=4, extra=None, **fields):
    config = SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "mooncake_store",
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


def _seq(req_id, num_prompt_tokens, *, offset=0, cached=0, **fields):
    seq = SimpleNamespace(
        id=req_id,
        token_ids=array.array("i", range(offset, offset + num_prompt_tokens)),
        num_prompt_tokens=num_prompt_tokens,
        num_cached_tokens=cached,
        block_table=list(range(-(-num_prompt_tokens // BLOCK))),
    )
    for name, value in fields.items():
        setattr(seq, name, value)
    return seq


def _hashes(num_tokens, offset=0):
    return keys.chunk_hash_chain(list(range(offset, offset + num_tokens)), CHUNK)


def _key(hashes, index, *, rank=0, world=1, namespace=NAMESPACE):
    return keys.chunk_key(namespace, rank, world, keys.chunk_digest(hashes, index))


def _save_req(req_id, num_tokens, *, skip=0, generation=0, offset=0, block_ids=None):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=(
            list(range(num_tokens // BLOCK)) if block_ids is None else block_ids
        ),
        save_spec=SaveSpec(skip_leading_tokens=skip),
        save_operation=SaveOperationId(req_id, generation),
        chunk_hashes=_hashes(num_tokens, offset),
    )


def _load_req(req_id, *, hbm, lmc, block_ids, end=None, generation=0, num_tokens=None):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=block_ids,
        load_spec=LoadSpec(
            hbm_cached_tokens=hbm,
            lmcache_cached_tokens=lmc,
            can_load=True,
            transfer_end_tokens=end,
        ),
        load_operation=LoadOperationId(req_id, generation),
        chunk_hashes=_hashes(num_tokens or (end or lmc)),
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


@pytest.fixture
def make_worker(cluster):
    """A worker whose GPU connector, pool and Store client are the fakes."""
    built = []

    def build(
        *, rank=0, world=1, save_slots=16, load_slots=16, config=None, namespace=None
    ):
        worker = MooncakeStoreOffloadConnector(config or _config())
        gpu = FakeGPUConnector(worker._source_group_safe)
        client = store_client.MooncakeStoreClient(
            local_hostname="10.0.0.2",
            metadata_server=METADATA,
            master_server_addr=MASTER,
            protocol="tcp",
            rdma_devices="",
        )
        pool = TransferSlotPool(
            device="cpu",
            chunk_bytes=CHUNK_BYTES,
            save_bytes=save_slots * SLOT,
            load_bytes=load_slots * SLOT,
            client=client,
        )
        worker._gpu_connector, worker._client, worker._pool = gpu, client, pool
        worker._namespace = namespace or NAMESPACE
        worker._rank, worker._world = rank, world
        worker._chunk_bytes = CHUNK_BYTES
        built.append(worker)
        return worker, gpu

    yield build
    for worker in built:
        worker.close()


@pytest.fixture
def slow_calls(monkeypatch):
    """Every Store call takes as long as Mooncake's batch wait."""
    monkeypatch.setattr(
        store_client.CallClock, "seconds", lambda _self: store_client.BATCH_WAIT_S
    )


@pytest.fixture
def scheduler_clock(monkeypatch):
    """The scheduler module's monotonic clock, advanced by hand.

    Its wall clock, which only stamps a save's dispatch time for the workers,
    stays the real one the workers read.
    """
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        scheduler_mod,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            perf_counter=time.perf_counter,
            time=time.time,
        ),
    )
    return clock


# --- config ------------------------------------------------------------------


def test_config_defaults_and_shared_master():
    cfg = parse_mooncake_store_config(
        {
            "kv_connector_extra_config": {
                "mooncake_store.master": MASTER,
                "mooncake_store.metadata": METADATA,
                "mooncake_store.local_hostname": "10.0.0.2",
                "lmcache.chunk_size": 256,  # another backend's key: ignored
                "max_pending_saves": 4,
            }
        }
    )
    assert (cfg.master, cfg.metadata, dict(cfg.pools)) == (MASTER, METADATA, {})
    assert cfg.protocol == "rdma" and cfg.pool_device == "gpu"
    assert cfg.chunk_tokens == 256 and cfg.lookup_batch_keys == 8192
    assert (cfg.load_pool_bytes, cfg.save_pool_bytes) == (1024 << 20, 256 << 20)
    assert cfg.save_abandon_timeout_s == 300.0
    assert cfg.startup_probe and cfg.direct_copy and cfg.chunk_groups
    assert cfg.owner_rdma_devices == cfg.rdma_devices == ()
    assert cfg.store_masters() == [nic.StorePool(MASTER, METADATA)]


def test_config_falls_back_to_the_top_level_dict_and_resolves_the_host(monkeypatch):
    monkeypatch.setenv("ATOM_HOST_IP", "10.9.9.9")
    cfg = parse_mooncake_store_config(
        {
            "kv_connector": "mooncake_store",
            "mooncake_store.master": MASTER,
            "mooncake_store.metadata": "P2PHANDSHAKE",
            "mooncake_store.owner_rdma_devices": "rdma4, rdma5,",
        }
    )
    assert cfg.local_hostname == "10.9.9.9"
    assert cfg.metadata == "P2PHANDSHAKE"
    assert cfg.owner_rdma_devices == ("rdma4", "rdma5")


def test_config_per_nic_pools_from_json_text():
    pools = {
        "rdma0": {"master": "10.0.0.1:26051", "metadata": METADATA},
        "rdma1": {"master": "10.0.0.1:26151", "metadata": METADATA},
        "rdma2": {"master": "10.0.0.1:26051", "metadata": METADATA},
    }
    cfg = parse_mooncake_store_config(
        _extra(master=None, metadata=None, pools=json.dumps(pools))
    )
    assert cfg.master is None and cfg.pools["rdma1"].master == "10.0.0.1:26151"
    # Two NICs naming one master make one lookup target.
    assert cfg.store_masters() == [
        nic.StorePool("10.0.0.1:26051", METADATA),
        nic.StorePool("10.0.0.1:26151", METADATA),
    ]


@pytest.mark.parametrize(
    "launcher_json",
    [
        # What the atomesh launcher sets as ATOM_KV_OFFLOAD_EXTRA_CONFIG: one
        # shared master ...
        (
            '{"mooncake_store.local_hostname":"10.1.1.1","mooncake_store.protocol":'
            '"rdma","mooncake_store.master":"10.1.1.1:26051","mooncake_store.'
            'metadata":"http://10.1.1.1:26080/metadata","mooncake_store.'
            'owner_rdma_devices":"rdma4,rdma5,rdma6,rdma7","max_pending_saves":8}'
        ),
        # ... or one master per NIC.
        (
            '{"mooncake_store.local_hostname":"10.1.1.1","mooncake_store.protocol":'
            '"rdma","mooncake_store.pools":{"rdma0":{"master":"10.1.1.1:26051",'
            '"metadata":"http://10.1.1.1:26080/metadata"},"rdma1":{"master":'
            '"10.1.1.1:26151","metadata":"http://10.1.1.1:26180/metadata"}}}'
        ),
    ],
)
def test_config_takes_the_launcher_worker_config(launcher_json):
    offload = kv_offload_connector_config("mooncake_store", launcher_json)
    cfg = parse_mooncake_store_config(offload)
    assert cfg.protocol == "rdma" and cfg.local_hostname == "10.1.1.1"
    assert len(cfg.store_masters()) in (1, 2)
    assert cfg.pools or cfg.owner_rdma_devices == ("rdma4", "rdma5", "rdma6", "rdma7")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"masterr": MASTER}, "unknown Mooncake Store offload setting"),
        ({"master": None}, "needs mooncake_store.master"),
        ({"pools": {"rdma0": {"master": MASTER, "metadata": METADATA}}}, "not both"),
        ({"master": "10.0.0.1"}, "host:port"),
        ({"master": "10.0.0.1:0"}, "host:port"),
        ({"metadata": ""}, "non-empty string"),
        ({"protocol": "ib"}, "protocol must be one of"),
        ({"pool_device": "nvme"}, "pool_device must be one of"),
        ({"chunk_tokens": 256.0}, "chunk_tokens must be an integer"),
        ({"chunk_tokens": 0}, "chunk_tokens must be positive"),
        ({"load_pool_mib": True}, "load_pool_mib must be an integer"),
        ({"lookup_batch_keys": 0}, "lookup_batch_keys must be positive"),
        ({"save_abandon_timeout_s": 0}, "finite and > 0"),
        ({"save_abandon_timeout_s": float("inf")}, "finite and > 0"),
        ({"save_abandon_timeout_s": "300"}, "number of seconds"),
        # Gone: switching it off left every block after a load unhashed.
        ({"publish_loaded_prefix": False}, "unknown Mooncake Store offload"),
        ({"startup_probe": 1}, "true or false"),
        ({"direct_copy": "false"}, "true or false"),
        ({"chunk_groups": "yes"}, "true or false"),
        ({"rdma_devices": ["rdma0"]}, "comma-separated string"),
    ],
)
def test_config_refuses_bad_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        parse_mooncake_store_config(_extra(**overrides))


def test_config_refuses_a_bad_pool_table():
    with pytest.raises(ValueError, match="not JSON"):
        parse_mooncake_store_config(_extra(master=None, metadata=None, pools="{x"))
    with pytest.raises(ValueError, match="host:port"):
        parse_mooncake_store_config(
            _extra(
                master=None,
                metadata=None,
                pools={"rdma0": {"master": "nohost", "metadata": METADATA}},
            )
        )


def test_engine_compatibility_refuses_what_phase_one_does_not_move():
    cfg = parse_mooncake_store_config(_extra())
    check_engine_compatibility(cfg, _config())
    hybrid = _config()
    hybrid.hf_config.compress_ratios = [4, 128]
    with pytest.raises(ValueError, match="only the dense KV layout"):
        check_engine_compatibility(cfg, hybrid)
    with pytest.raises(ValueError, match="decode context parallelism"):
        check_engine_compatibility(cfg, _config(decode_context_parallel_size=4))
    odd = parse_mooncake_store_config(_extra(chunk_tokens=6))
    with pytest.raises(ValueError, match="multiple of the KV cache block size 4"):
        check_engine_compatibility(odd, _config())


@pytest.mark.parametrize("role_side", ["scheduler", "worker"])
def test_both_sides_refuse_dcp(cluster, role_side):
    config = _config(decode_context_parallel_size=2)
    with pytest.raises(ValueError, match="decode context parallelism"):
        KVConnectorFactory.create_connector(config, role=role_side)


# --- keys --------------------------------------------------------------------


def test_chain_is_the_same_for_list_array_and_ndarray():
    tokens = list(range(1000, 1100))
    expected = keys.chunk_hash_chain(tokens, CHUNK)
    assert len(expected) == (100 // CHUNK) * keys.DIGEST_BYTES
    assert keys.chunk_hash_chain(array.array("i", tokens), CHUNK) == expected
    assert keys.chunk_hash_chain(np.array(tokens, dtype=np.int64), CHUNK) == expected
    assert keys.chunk_hash_chain(tokens, CHUNK) == expected  # deterministic


def test_chain_digests_name_their_whole_prefix():
    tokens = list(range(64))
    full = keys.chunk_hash_chain(tokens, CHUNK)
    # A prefix's chain is a prefix of the chain; a partial chunk is left out.
    assert keys.chunk_hash_chain(tokens[:20], CHUNK) == full[: 2 * keys.DIGEST_BYTES]
    assert keys.chunk_hash_chain(tokens, CHUNK, num_tokens=20) == full[:32]
    # Extending from previous digests hashes only the new chunks.
    assert keys.chunk_hash_chain(tokens, CHUNK, previous=full[:32]) == full
    # Changing one token changes its chunk's digest and every later one.
    edited = tokens.copy()
    edited[17] += 1
    other = keys.chunk_hash_chain(edited, CHUNK)
    assert other[:32] == full[:32]
    assert all(
        keys.chunk_digest(other, i) != keys.chunk_digest(full, i) for i in range(2, 8)
    )
    assert keys.chunk_hash_chain(tokens, 2 * CHUNK)[:16] != full[:16]
    assert keys.chunk_hash_chain(tokens[:7], CHUNK) == b""


def test_chain_seed_separates_media_prompts():
    tokens = list(range(32))
    text = keys.chunk_hash_chain(tokens, CHUNK)
    assert keys.chain_seed(None) == keys.chain_seed(-1) == bytes(16)
    assert keys.chunk_hash_chain(tokens, CHUNK, seed=keys.chain_seed(-1)) == text
    image_a = keys.chunk_hash_chain(tokens, CHUNK, seed=keys.chain_seed(12345))
    image_b = keys.chunk_hash_chain(tokens, CHUNK, seed=keys.chain_seed(54321))
    assert len({text[:16], image_a[:16], image_b[:16]}) == 3


def test_chain_refuses_bad_input():
    with pytest.raises(ValueError, match="fit in int32"):
        keys.chunk_hash_chain([2**31] * CHUNK, CHUNK)
    with pytest.raises(ValueError, match="whole 16-byte digests"):
        keys.chunk_hash_chain(list(range(16)), CHUNK, previous=b"x")
    with pytest.raises(ValueError, match="cover 3 chunks"):
        keys.chunk_hash_chain(list(range(16)), CHUNK, previous=bytes(48))


def test_chain_does_not_pin_the_sequence_array():
    tokens = array.array("i", range(40))
    keys.chunk_hash_chain(tokens, CHUNK)
    tokens.append(7)  # BufferError if a numpy view outlived the call


def test_namespace_fingerprints_the_layout(monkeypatch):
    base = keys.store_namespace(_config(), CHUNK)
    assert base == keys.store_namespace(_config(), CHUNK)
    assert base.startswith("atomkv1-") and len(base) == len("atomkv1-") + 24
    assert keys.store_namespace(_config(), 2 * CHUNK) != base
    assert keys.store_namespace(_config(kv_cache_dtype="bf16"), CHUNK) != base
    assert (
        keys.store_namespace(
            _config(online_quant_config={"global_quant_config": "fp8"}), CHUNK
        )
        != base
    )
    # PP4 x TP1 and PP1 x TP4 have the same world but different layer slices.
    pp4 = keys.store_namespace(_config(pp=4, layers=8), CHUNK)
    assert pp4 != keys.store_namespace(_config(tp=4, layers=8), CHUNK)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "3,2,2,1")
    assert keys.store_namespace(_config(pp=4, layers=8), CHUNK) != pp4
    assert keys.pp_stage_layer_spans(_config(pp=4, layers=8)) == [
        (0, 3),
        (3, 5),
        (5, 7),
        (7, 8),
    ]


def test_namespace_fingerprints_the_formats_the_gpu_stores(aiter_arch):
    mi355 = keys.store_namespace(_config(), CHUNK)
    assert keys.kv_storage_formats(_config()) == {
        "gfx": "gfx950",
        "kv_cache": "torch.float8_e4m3fn",
        "index_cache": "torch.float8_e4m3fn",
    }
    assert keys.kv_storage_formats(_config(index_cache_dtype="fp4"))["index_cache"] == (
        "fp4"
    )
    bf16_mi355 = keys.store_namespace(_config(kv_cache_dtype="bf16"), CHUNK)
    aiter_arch.gfx = "gfx942"
    # "fp8" is e4m3fnuz here: one byte either way, so only the namespace
    # keeps an MI300X from loading an MI355X's chunk.
    assert keys.kv_storage_formats(_config())["kv_cache"] == "torch.float8_e4m3fnuz"
    assert keys.store_namespace(_config(), CHUNK) != mi355
    # A DSA index cache is fp8 under a bf16 KV cache too.
    assert keys.store_namespace(_config(kv_cache_dtype="bf16"), CHUNK) != bf16_mi355


def test_namespace_fingerprints_rope_overrides_and_revision():
    base = keys.store_namespace(_config(), CHUNK)
    assert keys.store_namespace(_config(), CHUNK) == base
    yarn = _config()
    yarn.hf_config.rope_parameters = {"rope_type": "yarn", "factor": 4.0}
    assert keys.store_namespace(yarn, CHUNK) != base
    theta = _config()
    theta.hf_config.rope_theta = 1_000_000.0
    assert keys.store_namespace(theta, CHUNK) != base
    # Any --hf-overrides field, e.g. DeepSeek-V3.2's index sharing.
    overridden = _config(hf_overrides={"use_index_cache": True, "index_topk_freq": 4})
    assert keys.store_namespace(overridden, CHUNK) != base
    revised = _config()
    revised.hf_config._commit_hash = "0123abcd"
    assert keys.store_namespace(revised, CHUNK) != base


def test_namespace_hashes_the_rope_the_config_snapshotted():
    shipped = _config()
    shipped.hf_config.rope_parameters = {"rope_type": "default", "rope_theta": 1e4}
    shipped.offload_rope_config = offcfg.snapshot_rope_config(shipped.hf_config)
    scheduler, worker = copy.deepcopy(shipped), copy.deepcopy(shipped)
    # What llama.py writes into the live config while a worker builds it.
    worker.hf_config.rope_parameters["original_max_position_embeddings"] = 8192
    assert keys.store_namespace(scheduler, CHUNK) == keys.store_namespace(worker, CHUNK)
    # Without the snapshot the two diverge and every lookup would miss.
    del scheduler.offload_rope_config, worker.offload_rope_config
    assert keys.store_namespace(scheduler, CHUNK) != keys.store_namespace(worker, CHUNK)


def test_the_rope_snapshot_never_raises():
    hf = SimpleNamespace(
        rope_scaling={"factor": float("nan"), "type": object()},
        text_config=SimpleNamespace(rope_theta=5e5),
    )
    snapshot = json.loads(offcfg.snapshot_rope_config(hf))
    assert snapshot["text_config"]["rope_theta"] == 5e5
    assert snapshot["rope_scaling"]["factor"] != snapshot["rope_scaling"]["factor"]


_KV_LAYOUT_ENV = (
    "ATOM_MLA_PAGE_SIZE",
    "ATOM_USE_TRITON_MLA",
    "ATOM_USE_TRITON_MLA_SHUFFLE_KV",
    "ATOM_USE_UNIFIED_ATTN",
    "ATOM_FORCE_ATTN_TRITON",
)


@pytest.mark.parametrize(
    "env",
    [
        # Segmented MLA: every token's nope, then every token's pe, per block.
        {"ATOM_MLA_PAGE_SIZE": "64"},
        # The Triton MLA backend's shuffled view of the same allocation.
        {"ATOM_USE_TRITON_MLA": "1", "ATOM_USE_TRITON_MLA_SHUFFLE_KV": "1"},
        # The MHA kernels' block becomes the scheduler's.
        {"ATOM_USE_UNIFIED_ATTN": "1"},
        # fp8 MHA KV under one fixed scale instead of per-token scales.
        {"ATOM_FORCE_ATTN_TRITON": "1"},
    ],
)
def test_namespace_fingerprints_the_kv_layout_the_environment_selects(monkeypatch, env):
    # Same block size and bytes per block, arranged differently: a load checks
    # only the size, so only the namespace keeps the layouts apart.
    for name in _KV_LAYOUT_ENV:
        monkeypatch.delenv(name, raising=False)
    config = _config(kv_cache_block_size=64)
    base = keys.store_namespace(config, 256)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert keys.store_namespace(config, 256) != base


def test_key_spellings():
    digest = bytes(range(16))
    assert keys.chunk_key("ns", 2, 4, digest) == f"ns/w2of4/{digest.hex()}"
    assert (
        keys.chunk_keys("ns", 1, 2, digest * 3, 1, 3)
        == [f"ns/w1of2/{digest.hex()}"] * 2
    )
    probe = keys.probe_key("ns", 2, "abc")
    assert probe == "ns/probe/w2/abc" and not probe.startswith("ns/w")


# --- client ------------------------------------------------------------------


def _client(**overrides):
    fields = {
        "local_hostname": "10.0.0.2",
        "metadata_server": METADATA,
        "master_server_addr": MASTER,
        "protocol": "rdma",
        "rdma_devices": "rdma3",
    }
    fields.update(overrides)
    return store_client.MooncakeStoreClient(**fields)


def test_client_is_a_pure_zero_copy_client(cluster):
    client = _client()
    # global_segment_size=0 and local_buffer_size=0, positionally.
    assert cluster.stores[0].setup_args == (
        "10.0.0.2",
        METADATA,
        0,
        0,
        "rdma",
        "rdma3",
        MASTER,
    )
    client.close()
    client.close()
    assert cluster.stores[0].closed


def test_client_setup_and_registration_failures_name_the_cause(cluster):
    cluster.setup_rc = store_client.RPC_FAIL
    with pytest.raises(RuntimeError, match=f"RPC_FAIL.*master={MASTER}.*rdma3"):
        _client()
    cluster.setup_rc = 0
    client = _client()
    client._store.register_rc = store_client.INVALID_PARAMS
    with pytest.raises(RuntimeError, match="transparent huge pages"):
        client.register(0x1000, 4096)


def test_client_lookups_are_batched_in_order(cluster):
    client = _client(lookup_batch_keys=3)
    store = cluster.stores[0]
    store.objects.update({"k1": b"x", "k4": b"x"})
    store.exist_codes["k5"] = store_client.RPC_FAIL
    assert client.exists([f"k{i}" for i in range(7)]) == [0, 1, 0, 0, 1, -900, 0]
    assert [len(call[1]) for call in store.calls if call[0] == "exists"] == [3, 3, 1]
    assert client.stats()["counts"]["exists_calls"] == 3
    assert client.stats()["failures"] == {"exists:RPC_FAIL(-900)": 1}


def test_client_refuses_duplicate_keys_and_short_answers(cluster):
    client = _client()
    with pytest.raises(ValueError, match="unique"):
        client.put(["a", "a"], [1, 2], [8, 8])
    with pytest.raises(ValueError, match="unique"):
        client.get(["a", "a"], [1, 2], [8, 8])
    client._store.batch_get_into = lambda *args: []
    with pytest.raises(RuntimeError, match="0 results for 1 keys"):
        client.get(["a"], [1], [8])


def test_result_codes():
    assert store_client.describe(-704) == "OBJECT_NOT_FOUND(-704)"
    assert store_client.describe(-12345) == "rc=-12345"
    wait = store_client.BATCH_WAIT_S
    for seconds in (0.001, wait - 0.01, wait, 3 * wait):
        assert store_client.buffer_settled(0, seconds)
        assert store_client.buffer_settled(4096, seconds)
        for code in (-200, -600, -703, -704, -705, -707, -900, -901):
            assert store_client.buffer_settled(code, seconds)
    # Before the batch wait runs out a failed key has no transfer left: every
    # piece of it completed or failed, or none was posted (a dead owner fails
    # a get in 0.2-1.2 s) ...
    for code in (store_client.TRANSFER_FAIL, -1, -702):
        assert store_client.buffer_settled(code, 1.2)
        assert store_client.buffer_settled(code, wait - 0.01)
    # ... after it, Mooncake may have given up without cancelling posted RDMA.
    for code in (store_client.TRANSFER_FAIL, -1, -702):
        assert not store_client.buffer_settled(code, wait)
        assert not store_client.buffer_settled(code, 3 * wait)


def test_call_clock_counts_a_wall_clock_step(monkeypatch):
    clock = store_client.CallClock()
    assert 0 <= clock.seconds() < 1
    # Mooncake's batch wait runs on the wall clock: a step forward shortens it.
    wall = time.time()
    monkeypatch.setattr(store_client.time, "time", lambda: wall + 61)
    assert clock.seconds() >= 61


# --- pool --------------------------------------------------------------------


def _pool(cluster, *, save=2, load=3):
    client = _client(protocol="tcp", rdma_devices="")
    return TransferSlotPool(
        device="cpu",
        chunk_bytes=CHUNK_BYTES,
        save_bytes=save * SLOT,
        load_bytes=load * SLOT + SLOT - 1,
        client=client,
    )


def test_pool_geometry_and_registration(cluster):
    pool = _pool(cluster)
    assert pool.slot_stride == SLOT
    assert (pool.capacity("save"), pool.capacity("load")) == (2, 3)
    assert pool.base_ptr % (2 << 20) == 0
    assert cluster.stores[0].registered == {pool.base_ptr: 5 * SLOT}
    slots = pool.acquire("save", 2) + pool.acquire("load", 3)
    assert [s.ptr - pool.base_ptr for s in slots] == [i * SLOT for i in range(5)]
    assert all(s.tensor.numel() == CHUNK_BYTES for s in slots)
    pool.release(slots)
    pool.close()
    pool.close()
    assert cluster.stores[0].registered == {}
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire("save", 1)


def test_pool_refuses_a_region_without_a_slot(cluster):
    with pytest.raises(ValueError, match="save_pool_mib is smaller than one"):
        TransferSlotPool(
            device="cpu",
            chunk_bytes=CHUNK_BYTES,
            save_bytes=SLOT - 1,
            load_bytes=SLOT,
            client=_client(),
        )


def test_pool_acquire_waits_for_a_release(cluster):
    pool = _pool(cluster)
    held = pool.acquire("save", 2)
    got = []
    waiter = threading.Thread(target=lambda: got.extend(pool.acquire("save", 1)))
    waiter.start()
    time.sleep(0.05)
    assert got == []
    pool.release(held[:1])
    waiter.join(timeout=5)
    assert got == held[:1]
    with pytest.raises(RuntimeError, match="not held"):
        pool.release(held[:1] + held[:1])
    with pytest.raises(ValueError, match="3 slots of the 2-slot save region"):
        pool.acquire("save", 3)


def test_a_window_is_at_most_a_quarter_of_its_region(cluster):
    pool = _pool(cluster, save=2, load=16)
    assert [pool.window("load", threads) for threads in (1, 2, 4, 8, 32)] == [
        4,
        4,
        4,
        2,
        1,
    ]
    assert pool.window("save", 1) == 1
    # A transfer that never settled leaves the rest of the region working.
    pool.quarantine(pool.acquire("load", 4))
    assert pool.usable("load") == 12 and pool.window("load", 1) == 3


def test_quarantined_slots_never_come_back(cluster, caplog):
    pool = _pool(cluster)
    held = pool.acquire("save", 2)
    with caplog.at_level(logging.INFO, logger="atom"):
        pool.quarantine(held, reason="test")
        assert (pool.usable("save"), pool.quarantined("save")) == (0, 2)
        with pytest.raises(SlotPoolExhausted, match="0 usable slots of 2"):
            pool.acquire("save", 1)
        # Reported once, as an error: every save of this worker now fails.
        pool.quarantine(pool.acquire("load", 1), reason="test")
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == [
            (
                "Mooncake Store offload: every save slot of this worker is out "
                "of use; its saves fail at once for good"
            )
        ]
    # No time brings them back: nothing bounds how late an RDMA access lands.
    assert (pool.usable("save"), pool.quarantined("save")) == (0, 2)
    assert (pool.usable("load"), pool.quarantined("load")) == (2, 1)
    assert not hasattr(held[0], "held_until")


def test_closing_a_pool_with_a_quarantined_slot_keeps_its_memory(cluster, caplog):
    pool = _pool(cluster)
    [store] = cluster.stores
    backing = pool._backing
    pool.quarantine(pool.acquire("load", 1), reason="test")
    with caplog.at_level(logging.WARNING, logger="atom"):
        pool.close()
    # Registered and referenced for the life of the process: a late RDMA
    # transfer into a quarantined slot lands in memory nothing else is given.
    assert store.registered == {pool.base_ptr: pool.nbytes}
    assert ("unregister", pool.base_ptr) not in store.calls
    assert pool_mod._UNSETTLED_ALLOCATIONS == [backing]
    assert any(
        "stay registered and allocated" in r.getMessage() for r in caplog.records
    )
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire("save", 1)


def test_closing_a_pool_with_a_leased_slot_keeps_its_memory(cluster):
    pool = _pool(cluster)
    pool.acquire("save", 1)  # a transfer that never came back
    pool.close()
    assert cluster.stores[0].registered == {pool.base_ptr: pool.nbytes}
    assert len(pool_mod._UNSETTLED_ALLOCATIONS) == 1


def test_quarantine_retires_slots_and_wakes_waiters(cluster):
    pool = _pool(cluster)
    held = pool.acquire("save", 2)
    outcome = []

    def wait_for_two():
        try:
            pool.acquire("save", 2)
        except SlotPoolExhausted as exc:
            outcome.append(exc)

    waiter = threading.Thread(target=wait_for_two)
    waiter.start()
    time.sleep(0.05)
    pool.quarantine(held[:1], reason="test")
    waiter.join(timeout=5)
    assert outcome and "1 usable slots of 2 (1 quarantined)" in str(outcome[0])
    assert (pool.usable("save"), pool.quarantined("save")) == (1, 1)
    assert pool.window("save", 1) == 1
    pool.release(held[1:])
    assert pool.acquire("save", 1) == held[1:]


# --- NIC per worker (fake sysfs of pit2-p03-g40) ------------------------------

# HIP ordinal -> GPU BDF, and the BDF of the NIC behind the same PCIe switch.
GPU_BDF = ["75", "05", "65", "15", "f5", "85", "e5", "95"]
NIC_BDF = {
    f"rdma{i}": bus
    for i, bus in enumerate(["09", "19", "69", "79", "89", "99", "e9", "f9"])
}
POOLS = {
    "rdma2": {"master": "10.0.0.1:26251", "metadata": "http://10.0.0.1:26280/metadata"},
    "rdma3": {"master": "10.0.0.1:26351", "metadata": "http://10.0.0.1:26380/metadata"},
}


def _root_complex(bus: str) -> str:
    return f"pci0000:{bus[0]}0"


def _make_node(root: Path, *, inactive=()):
    """A sysfs tree: per GPU one host bridge and one switch shared with a NIC."""
    devices = root / "devices"
    pci = root / "bus/pci/devices"
    ib = root / "class/infiniband"
    pci.mkdir(parents=True)
    ib.mkdir(parents=True)
    for bus in GPU_BDF:
        top = bus[0]
        gpu = devices / _root_complex(bus) / f"0000:{top}0:01.1/0000:{top}1:00.0"
        gpu = gpu / f"0000:{top}2:00.0/0000:{bus}:00.0"
        gpu.mkdir(parents=True)
        (pci / f"0000:{bus}:00.0").symlink_to(gpu)
    for name, bus in NIC_BDF.items():
        top = bus[0]
        nic_dir = devices / _root_complex(bus) / f"0000:{top}0:01.1/0000:{top}1:00.0"
        nic_dir = nic_dir / f"0000:{top}2:01.0/0000:{bus}:00.0"
        nic_dir.mkdir(parents=True)
        node = ib / name
        (node / "ports/1").mkdir(parents=True)
        state = "1: DOWN" if name in inactive else "4: ACTIVE"
        (node / "ports/1/state").write_text(state + "\n")
        (node / "device").symlink_to(nic_dir)
    return pci, ib


def _nic_cfg(rdma_devices="", owners="", pools=None):
    return SimpleNamespace(
        rdma_devices=tuple(nic.parse_device_list(rdma_devices)),
        owner_rdma_devices=tuple(nic.parse_device_list(owners)),
        pools=nic.parse_store_pools(pools),
    )


def test_rail_rdma_device_follows_the_pci_tree_not_the_gpu_ordinal(tmp_path):
    pci, ib = _make_node(tmp_path)
    picked = [
        nic.rail_rdma_device(f"0000:{bus}:00.0", ib_root=ib, pci_root=pci)
        for bus in GPU_BDF
    ]
    # HIP GPU 0 is bus 0x75, whose switch holds rdma3, not rdma0.
    assert picked == [f"rdma{i}" for i in (3, 0, 2, 1, 7, 4, 6, 5)]


def test_rail_rdma_device_skips_inactive_nics_and_refuses_guesses(tmp_path):
    pci, ib = _make_node(tmp_path, inactive={"rdma3"})
    with pytest.raises(ValueError, match="cannot tell GPU 0000:75:00.0's NIC"):
        nic.rail_rdma_device("0000:75:00.0", ib_root=ib, pci_root=pci)
    with pytest.raises(ValueError, match="not in"):
        nic.rail_rdma_device("0000:aa:00.0", ib_root=ib, pci_root=pci)


def test_rail_rdma_device_refuses_two_equally_close_nics(tmp_path):
    pci, ib = _make_node(tmp_path)
    twin = ib / "rdma8"
    (twin / "ports/1").mkdir(parents=True)
    (twin / "ports/1/state").write_text("4: ACTIVE\n")
    (twin / "device").symlink_to((ib / "rdma3/device").resolve())
    with pytest.raises(ValueError, match=r"rdma3:\d+, rdma4:\d+.*rdma8:\d+"):
        nic.rail_rdma_device("0000:75:00.0", ib_root=ib, pci_root=pci)


@pytest.fixture
def node(tmp_path, monkeypatch):
    pci, ib = _make_node(tmp_path)
    monkeypatch.setattr(nic, "_IB_SYSFS_ROOT", ib)
    monkeypatch.setattr(nic, "_PCI_SYSFS_ROOT", pci)
    monkeypatch.setattr(nic, "gpu_pci_bdf", lambda index: f"0000:{GPU_BDF[index]}:00.0")
    real = nic.rail_rdma_device
    # Keyword defaults were bound at import; rebind to the fake tree.
    monkeypatch.setattr(
        nic, "rail_rdma_device", lambda bdf: real(bdf, ib_root=ib, pci_root=pci)
    )
    return ib


def test_requester_device_defaults_to_the_topology(node):
    cfg = _nic_cfg()
    assert [nic.requester_rdma_device(i, cfg) for i in range(4)] == [
        "rdma3",
        "rdma0",
        "rdma2",
        "rdma1",
    ]


def test_requester_device_override_is_indexed_by_gpu_ordinal(node):
    table = "rdma0,rdma1,rdma2,rdma3,rdma0,rdma1,rdma2,rdma3"
    cfg = _nic_cfg(table)
    assert [nic.requester_rdma_device(i, cfg) for i in range(8)] == table.split(",")
    assert nic.requester_rdma_device(3, _nic_cfg(" rdma5 ")) == "rdma5"
    with pytest.raises(ValueError, match="has none for GPU 2"):
        nic.requester_rdma_device(2, _nic_cfg("rdma0,rdma1"))
    with pytest.raises(ValueError, match="'mlx5_9'.*does not exist"):
        nic.requester_rdma_device(0, _nic_cfg("mlx5_9"))


def test_requester_device_never_shares_an_owner_nic(node):
    cfg = _nic_cfg(owners="rdma4,rdma5,rdma6,rdma7")
    assert nic.requester_rdma_device(0, cfg) == "rdma3"
    with pytest.raises(ValueError, match="'rdma7'.*Store owner"):
        nic.requester_rdma_device(4, cfg)


def test_requester_device_shares_the_owner_nic_of_its_own_pool(node):
    cfg = _nic_cfg(owners="rdma0,rdma1,rdma2,rdma3", pools=json.dumps(POOLS))
    assert nic.requester_rdma_device(0, cfg) == "rdma3"


def test_parse_store_pools():
    assert nic.parse_store_pools(" ") == nic.parse_store_pools(None) == {}
    assert nic.parse_store_pools(json.dumps(POOLS)) == {
        "rdma2": nic.StorePool("10.0.0.1:26251", "http://10.0.0.1:26280/metadata"),
        "rdma3": nic.StorePool("10.0.0.1:26351", "http://10.0.0.1:26380/metadata"),
    }
    assert nic.parse_store_pools(POOLS) == nic.parse_store_pools(json.dumps(POOLS))
    for bad, match in (
        ("rdma0=10.0.0.1:50051", "not JSON"),
        ("{}", "non-empty JSON object"),
        ('["rdma0"]', "non-empty JSON object"),
        ('{"rdma0": {"metadata": "http://h/metadata"}}', "'rdma0'"),
        ('{"rdma0": "10.0.0.1:50051"}', "'rdma0'"),
    ):
        with pytest.raises(ValueError, match=match):
            nic.parse_store_pools(bad)


def test_store_pool_of():
    pools = nic.parse_store_pools(POOLS)
    assert nic.store_pool_of(None, {}) is None
    assert nic.store_pool_of("rdma3", pools).master == "10.0.0.1:26351"
    with pytest.raises(ValueError, match="tcp"):
        nic.store_pool_of(None, pools)
    with pytest.raises(ValueError, match="no pool for RDMA device 'rdma0'"):
        nic.store_pool_of("rdma0", pools)


def test_parse_device_list():
    assert nic.parse_device_list(" rdma0, ,rdma1,") == ["rdma0", "rdma1"]
    assert nic.parse_device_list("") == []


# --- lookup ------------------------------------------------------------------


def _store_chunks(cluster, hashes, chunks, *, rank=0, world=1, master=MASTER):
    namespace = keys.store_namespace(_config(pp=world), CHUNK)
    objects = cluster.objects.setdefault(master, {})
    for index in chunks:
        objects[_key(hashes, index, rank=rank, world=world, namespace=namespace)] = (
            bytes(CHUNK_BYTES)
        )


def test_scheduler_opens_its_store_client_on_the_first_lookup(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_config())
    assert cluster.stores == []  # every PP stage builds one; only the head asks
    hashes = _hashes(40)
    _store_chunks(cluster, hashes, range(5))
    assert scheduler.get_num_new_matched_tokens(_seq(1, 41)) == (40, True)
    [store] = cluster.stores
    # Lookups are master RPCs: always tcp, no NIC.
    assert store.setup_args == ("10.0.0.2", METADATA, 0, 0, "tcp", "", MASTER)


def test_lookup_is_the_prefix_every_rank_holds(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_config(pp=2))
    lookup = scheduler._lookup_client
    hashes = _hashes(48)
    _store_chunks(cluster, hashes, range(6), rank=0, world=2)
    _store_chunks(cluster, hashes, [0, 1, 3, 4, 5], rank=1, world=2)  # hole at 2
    assert lookup.lookup(list(range(48))) == 2 * CHUNK
    _store_chunks(cluster, hashes, [2], rank=1, world=2)
    assert lookup.lookup(list(range(48))) == 6 * CHUNK
    assert lookup.lookup(list(range(52))) == 6 * CHUNK  # partial tail not asked
    assert lookup.lookup(list(range(7))) == 0  # shorter than a chunk
    [store] = cluster.stores
    # One RPC per lookup, rank-major, the partial chunk never asked about.
    asked = [call[1] for call in store.calls if call[0] == "exists"]
    assert [len(batch) for batch in asked] == [12, 12, 12]


def test_lookup_counts_rank_objects_past_the_shared_prefix(
    cluster, scheduler_clock, caplog
):
    scheduler = MooncakeStoreOffloadScheduler(_config(pp=2))
    lookup = scheduler._lookup_client
    hashes = _hashes(48)
    _store_chunks(cluster, hashes, range(6), rank=0, world=2)
    _store_chunks(cluster, hashes, [0, 1, 3, 4, 5], rank=1, world=2)  # hole at 2
    assert lookup.lookup(list(range(48))) == 2 * CHUNK
    # Rank 0's chunks 2-5 are stranded: stored, unusable while rank 1 lacks 2.
    assert (lookup.lookups, lookup.uneven_lookups, lookup.stranded_objects) == (
        1,
        1,
        4,
    )
    _store_chunks(cluster, hashes, [2], rank=1, world=2)
    assert lookup.lookup(list(range(48))) == 6 * CHUNK
    assert (lookup.lookups, lookup.uneven_lookups, lookup.stranded_objects) == (
        2,
        1,
        4,
    )
    with caplog.at_level(logging.INFO, logger="atom"):
        lookup.lookup(list(range(48)))
        assert not any("LOOKUP-STATS" in r.getMessage() for r in caplog.records)
        scheduler_clock.now += scheduler_mod._LOOKUP_STATS_INTERVAL_S
        lookup.lookup(list(range(48)))
    assert [
        r.getMessage() for r in caplog.records if "LOOKUP-STATS" in r.getMessage()
    ] == ["[OFFLOAD-LOOKUP-STATS] lookups=4 uneven_lookups=1 stranded_objects=4"]


def test_lookup_answers_nothing_on_any_store_error(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_config())
    hashes = _hashes(24)
    _store_chunks(cluster, hashes, range(3))
    lookup = scheduler._lookup_client
    assert lookup.lookup(list(range(24))) == 24
    [store] = cluster.stores
    store.exist_codes[next(iter(cluster.objects[MASTER]))] = store_client.RPC_FAIL
    assert lookup.lookup(list(range(24))) is None
    store.exist_codes.clear()
    store.raise_on = "exists"
    lookup._retry_lookup_at = 0.0  # past the pause the failure started
    assert lookup.lookup(list(range(24))) is None
    assert store.calls[-1][0] == "exists"
    # None is a non-answer: the scheduler does not remember it as a miss.
    lookup._retry_lookup_at = 0.0
    assert scheduler.get_num_new_matched_tokens(_seq(3, 25)) == (0, False)
    assert scheduler._lookup_results == {}


def test_lookup_splits_large_prompts_into_batches(cluster):
    scheduler = MooncakeStoreOffloadScheduler(
        _config(pp=2, extra=_extra(lookup_batch_keys=4))
    )
    hashes = _hashes(40)
    for rank in range(2):
        _store_chunks(cluster, hashes, range(5), rank=rank, world=2)
    assert scheduler._lookup_client.lookup(list(range(40))) == 40
    [store] = cluster.stores
    assert [len(c[1]) for c in store.calls if c[0] == "exists"] == [4, 4, 2]


def test_lookup_with_per_nic_pools_asks_every_pool(cluster):
    pools = {
        "rdma0": {"master": "10.0.0.1:26051", "metadata": METADATA},
        "rdma1": {"master": "10.0.0.1:26151", "metadata": METADATA},
    }
    config = _config(pp=2, extra=_extra(master=None, metadata=None, pools=pools))
    lookup = MooncakeStoreOffloadScheduler(config)._lookup_client
    hashes = _hashes(32)
    # Each stage's NIC routes its chunks to its own pool.
    _store_chunks(cluster, hashes, range(4), rank=0, world=2, master="10.0.0.1:26051")
    _store_chunks(cluster, hashes, range(4), rank=1, world=2, master="10.0.0.1:26151")
    assert lookup.lookup(list(range(32))) == 32
    assert sorted(s.setup_args[6] for s in cluster.stores) == [
        "10.0.0.1:26051",
        "10.0.0.1:26151",
    ]
    # A pool that errs is outvoted where another pool holds the key ...
    cluster.store_of("10.0.0.1:26051").exist_codes = {
        key: store_client.RPC_FAIL for key in cluster.objects["10.0.0.1:26151"]
    }
    assert lookup.lookup(list(range(32))) == 32
    # ... and makes the answer a non-answer where no pool does.
    cluster.objects["10.0.0.1:26151"].clear()
    assert lookup.lookup(list(range(32))) is None
    lookup.close()


def test_a_failed_lookup_pauses_the_next_ones(cluster, scheduler_clock):
    scheduler = MooncakeStoreOffloadScheduler(_config())
    lookup = scheduler._lookup_client
    prompt = list(range(16))
    assert lookup.lookup(prompt) == 0
    [store] = cluster.stores

    def asked():
        return sum(1 for call in store.calls if call[0] == "exists")

    for failure in ("raise", "code"):
        lookup._retry_lookup_at = 0.0
        if failure == "raise":
            store.raise_on = "exists"
        else:
            store.exist_codes = {
                _key(_hashes(16), 0, namespace=scheduler._namespace): (
                    store_client.RPC_FAIL
                )
            }
        assert lookup.lookup(prompt) is None
        store.raise_on, store.exist_codes = None, {}
        calls = asked()
        # A healthy Store again, but each call could block the scheduler
        # thread: inside the pause no lookup reaches it.
        scheduler_clock.now += scheduler_mod._LOOKUP_BACKOFF_S - 0.5
        assert lookup.lookup(prompt) is None
        assert asked() == calls
        scheduler_clock.now += 1.0
        assert lookup.lookup(prompt) == 0
        assert asked() == calls + 1


def test_a_lookup_that_blocked_pauses_ten_times_as_long(cluster, scheduler_clock):
    lookup = MooncakeStoreOffloadScheduler(_config())._lookup_client
    assert lookup.lookup(list(range(16))) == 0
    [store] = cluster.stores

    def dead_master(keys):
        scheduler_clock.now += 4.3  # measured: a dead master holds a call this long
        return [store_client.RPC_FAIL] * len(keys)

    store.batch_is_exist = dead_master
    assert lookup.lookup(list(range(16))) is None
    assert lookup._retry_lookup_at == pytest.approx(scheduler_clock.now + 43.0)


def test_lookup_retries_an_unreachable_store_only_after_a_pause(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_config())
    lookup = scheduler._lookup_client
    cluster.setup_rc = store_client.RPC_FAIL
    assert lookup.lookup(list(range(16))) is None
    assert lookup.lookup(list(range(16))) is None
    assert len(cluster.stores) == 1  # no reconnect on every step
    cluster.setup_rc = 0
    lookup._retry_connect_at = 0.0
    assert lookup.lookup(list(range(16))) == 0
    assert len(cluster.stores) == 2


@pytest.mark.parametrize("blocked_s", [31.0, 88.0])
def test_a_connect_that_blocked_pauses_ten_times_as_long(
    cluster, scheduler_clock, monkeypatch, blocked_s
):
    lookup = MooncakeStoreOffloadScheduler(_config())._lookup_client
    setup = FakeStore.setup

    def unanswered_setup(store, *args):
        # Mooncake's setup retries a master that does not answer, holding the
        # scheduler thread (and the GIL) all along, then fails.
        scheduler_clock.now += blocked_s
        store.setup_args = args
        return store_client.RPC_TIMEOUT

    monkeypatch.setattr(FakeStore, "setup", unanswered_setup)
    assert lookup.lookup(list(range(16))) is None
    # Paused from the failure, not from before the setup that blocked.
    assert lookup._retry_connect_at == pytest.approx(
        scheduler_clock.now + 10 * blocked_s
    )
    scheduler_clock.now += 0.01  # the next request's lookup
    assert lookup.lookup(list(range(16))) is None
    assert len(cluster.stores) == 1
    monkeypatch.setattr(FakeStore, "setup", setup)
    scheduler_clock.now += 10 * blocked_s
    assert lookup.lookup(list(range(16))) == 0
    assert len(cluster.stores) == 2


def test_lookup_hashes_a_sequence_prompt_once(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_config())
    seq = _seq(4, 20)
    seq.token_ids.extend([999] * 5)  # generated tokens are not prompt
    view = scheduler._lookup_token_ids(seq)
    assert isinstance(view, _PromptView) and len(view) == 20
    assert scheduler._lookup_client.lookup(view) == 0
    assert seq._mooncake_store_chunk_hashes == _hashes(16)
    seq._mooncake_store_chunk_hashes = b"\x01" * 32  # cached: not hashed again
    assert prompt_chunk_hashes(seq, CHUNK) == b"\x01" * 32
    media = _seq(5, 20, cache_seed=77)
    assert prompt_chunk_hashes(media, CHUNK) != _hashes(16)


# --- scheduler metadata -----------------------------------------------------


def _arm_load(scheduler, seq, *, hbm, lmc, end=None):
    sid = str(seq.id)
    scheduler._min_load_tokens = 0
    seq.num_cached_tokens = hbm
    scheduler._load_specs[sid] = LoadSpec(
        hbm_cached_tokens=hbm,
        lmcache_cached_tokens=lmc,
        can_load=True,
        transfer_end_tokens=end,
    )
    scheduler._reqs_need_recv[sid] = seq
    scheduler._lookup_results[sid] = (seq, lmc)
    if isinstance(scheduler, MooncakeStoreOffloadScheduler):
        # As this step's lookup leaves it: fresh, so not confirmed again.
        scheduler._looked_up_at[sid] = scheduler_mod.time.monotonic()


def test_save_requests_carry_digests_not_tokens(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_config("kv_producer"))
    seq = _seq(6, 30)
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 16
    before = time.time()
    [first] = scheduler.build_connector_meta().requests
    assert first.token_ids == [] and first.save_spec.skip_leading_tokens == 0
    assert first.chunk_hashes == _hashes(16)
    # Where the reclaim clock starts, for the workers' source read deadline.
    assert before <= first.dispatched_at <= time.time()
    scheduler.save_finished(first.save_operation)
    seq.num_cached_tokens = 30
    [second] = scheduler.build_connector_meta().requests
    assert second.save_spec.skip_leading_tokens == 16
    assert second.chunk_hashes == _hashes(24)  # whole chunks of the prompt only


def test_load_requests_carry_digests_and_publish_the_loaded_prefix(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_config("kv_consumer"))
    seq = _seq(7, 50)
    _arm_load(scheduler, seq, hbm=16, lmc=40)
    [request] = scheduler.build_connector_meta().requests
    assert request.token_ids == [] and request.block_ids == seq.block_table
    assert request.chunk_hashes == _hashes(40)
    assert request.load_operation == LoadOperationId(7, 0)
    assert seq.offload_load_start_tokens == 16
    assert seq.offload_loaded_tokens == 40

    full = _seq(8, 48)
    _arm_load(scheduler, full, hbm=0, lmc=40, end=48)
    [request] = scheduler.build_connector_meta().requests
    assert request.chunk_hashes == _hashes(48)


def test_a_hit_that_waited_for_blocks_is_looked_up_again_before_its_load(
    cluster, scheduler_clock
):
    scheduler = MooncakeStoreOffloadScheduler(_config("offload"))
    scheduler._min_load_tokens = 0
    hashes = _hashes(40)
    _store_chunks(cluster, hashes, range(5))
    seq = _seq(50, 41)
    assert scheduler.get_num_new_matched_tokens(seq) == (40, True)
    [store] = cluster.stores

    def asked():
        return sum(1 for call in store.calls if call[0] == "exists")

    # No KV blocks for 100 steps: the hit is reused, the Store not asked again.
    for _ in range(100):
        scheduler_clock.now += 0.5
        assert scheduler.get_num_new_matched_tokens(seq) == (40, True)
        assert scheduler.build_connector_meta().requests == []
    assert asked() == 1
    # The lookup's lease ran out long ago, and the Store evicted a chunk.
    namespace = scheduler._namespace
    del cluster.objects[MASTER][_key(hashes, 3, namespace=namespace)]
    scheduler.update_state_after_alloc(seq)
    # Admitted, so parked: asked again right before the get, which renews the
    # leases, and the load dispatched whatever the answer -- only its report
    # wakes the request.
    [request] = scheduler.build_connector_meta().requests
    assert request.load_spec.lmcache_cached_tokens == 40
    assert asked() == 2
    # The worker's get misses chunk 3 and fails the load: the request prefills
    # and its prompt is saved again from the HBM frontier.
    assert scheduler._save_tracker["50"][1] == 40
    assert scheduler.load_failed(request.load_operation)
    assert scheduler._save_tracker["50"][1] == 0
    assert not scheduler.has_pending_work()


def test_a_fresh_hit_is_loaded_without_asking_again(cluster, scheduler_clock):
    scheduler = MooncakeStoreOffloadScheduler(_config("kv_consumer"))
    scheduler._min_load_tokens = 0
    hashes = _hashes(40)
    _store_chunks(cluster, hashes, range(5))
    seq = _seq(51, 41)
    assert scheduler.get_num_new_matched_tokens(seq) == (40, True)
    scheduler_clock.now += 0.5
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert request.load_spec.lmcache_cached_tokens == 40
    [store] = cluster.stores
    assert sum(1 for call in store.calls if call[0] == "exists") == 1
    # A stale one that the Store still holds whole is loaded after one more ask.
    again = _seq(52, 41)
    assert scheduler.get_num_new_matched_tokens(again) == (40, True)
    scheduler_clock.now += 5.0
    scheduler.update_state_after_alloc(again)
    [request] = scheduler.build_connector_meta().requests
    assert request.load_operation.req_id == 52
    assert sum(1 for call in store.calls if call[0] == "exists") == 3
    scheduler.request_finished(again)
    assert "52" not in scheduler._looked_up_at


@pytest.fixture
def store_engine(cluster, monkeypatch):
    """The engine's scheduler, prefix caching on, offloading to the Store.

    ``withheld.on`` refuses every allocation, as a full KV cache does: a
    request then waits at the head of the queue on the hit of its first lookup.
    """
    engine = Scheduler(
        MockConfig(
            num_kvcache_blocks=40,
            max_model_len=128,
            max_num_batched_tokens=128,
            enable_prefix_caching=True,
        )
    )
    offload = MooncakeStoreOffloadScheduler(_config("offload"))
    offload._min_load_tokens = 0
    offload.bind_block_manager(engine.block_manager)
    engine.kv_connector = offload
    withheld = SimpleNamespace(on=False)
    can_allocate = engine.block_manager.can_allocate
    monkeypatch.setattr(
        engine.block_manager,
        "can_allocate",
        lambda seq, **kwargs: -1 if withheld.on else can_allocate(seq, **kwargs),
    )
    return SimpleNamespace(engine=engine, offload=offload, withheld=withheld)


def _park_after_a_wait_for_blocks(cluster, scheduler_clock, store_engine, meanwhile):
    """Admit a 41-token prompt with a 40-token hit after 3 s without KV blocks.

    ``meanwhile`` happens while it waits. Returns the sequence and the load
    the admitting step dispatched.
    """
    engine, offload = store_engine.engine, store_engine.offload
    hashes = _hashes(40)
    _store_chunks(cluster, hashes, range(5))
    seq = Sequence(list(range(41)), BLOCK, sampling_params=SamplingParams())
    engine.add(seq)
    store_engine.withheld.on = True
    engine.schedule()  # looked up: a 40-token hit, but no KV blocks
    assert seq.status == SequenceStatus.WAITING
    [store] = cluster.stores
    asked = sum(1 for call in store.calls if call[0] == "exists")
    scheduler_clock.now += 3.0  # its lookup's leases have run out
    key = _key(hashes, 3, namespace=offload._namespace)
    if meanwhile == "evicted":
        del cluster.objects[MASTER][key]
    elif meanwhile == "store_error":
        store.exist_codes[key] = store_client.RPC_FAIL
    elif meanwhile == "lookups_paused":  # another prompt's lookup just failed
        offload._lookup_client._retry_lookup_at = scheduler_clock.now + 10.0
    store_engine.withheld.on = False
    batch, _ = engine.schedule()
    # Admitted and parked, and its load dispatched in the same step.
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    loads = [r for r in batch.connector_meta_output.requests if r.load_spec]
    assert [r.load_spec.lmcache_cached_tokens for r in loads] == [40]
    # Asked again right before the get, unless lookups are paused.
    again = int(meanwhile != "lookups_paused")
    assert sum(1 for call in store.calls if call[0] == "exists") == asked + again
    return seq, loads[0]


@pytest.mark.parametrize(
    "meanwhile", ["nothing", "evicted", "store_error", "lookups_paused"]
)
def test_a_parked_request_is_woken_by_its_load_whatever_the_store_answers(
    cluster, scheduler_clock, store_engine, make_worker, meanwhile
):
    engine, offload = store_engine.engine, store_engine.offload
    seq, request = _park_after_a_wait_for_blocks(
        cluster, scheduler_clock, store_engine, meanwhile
    )
    worker, _ = make_worker(namespace=offload._namespace)
    worker._do_load_req(request)
    engine._update_from_kv_xfer_finished(worker.get_finished())
    batch, _ = engine.schedule()
    assert seq.status == SequenceStatus.RUNNING
    assert engine._num_parked_remote_kv == 0
    if meanwhile == "evicted":
        # The get missed chunk 3: prefilled from the start, and the prompt is
        # saved again once computed.
        assert batch.num_cached_tokens == [0]
        assert offload._save_tracker[str(seq.id)][1] == 0
    else:
        assert batch.num_cached_tokens == [40]


def test_an_aborted_parked_load_frees_its_blocks_when_it_reports(
    cluster, scheduler_clock, store_engine, make_worker
):
    engine, offload = store_engine.engine, store_engine.offload
    free = engine.block_manager.kv.num_free
    seq, request = _park_after_a_wait_for_blocks(
        cluster, scheduler_clock, store_engine, "evicted"
    )
    assert engine.abort_request(seq.id)
    assert seq.id in engine.deferred_free_blocks  # the worker may be writing
    worker, _ = make_worker(namespace=offload._namespace)
    worker._do_load_req(request)
    engine._update_from_kv_xfer_finished(worker.get_finished())
    assert seq.id not in engine.deferred_free_blocks
    assert engine.block_manager.kv.num_free == free
    assert engine._num_parked_remote_kv == 0
    assert not offload.has_pending_work()


def test_a_loaded_prefix_is_published_and_the_suffix_hashes_onto_it(
    cluster, store_engine, make_worker, caplog
):
    engine, offload = store_engine.engine, store_engine.offload
    _store_chunks(cluster, _hashes(40), range(5))
    seq = Sequence(list(range(49)), BLOCK, sampling_params=SamplingParams())
    engine.add(seq)
    batch, _ = engine.schedule()
    [request] = [r for r in batch.connector_meta_output.requests if r.load_spec]
    worker, _ = make_worker(namespace=offload._namespace)
    worker._do_load_req(request)
    engine._update_from_kv_xfer_finished(worker.get_finished())
    kv = engine.block_manager.kv
    with caplog.at_level(logging.ERROR, logger="atom"):
        batch, _ = engine.schedule()  # the suffix, [40, 49)
        assert batch.num_cached_tokens == [40]
        # Woken: the ten loaded blocks are indexed before any forward.
        assert all(kv.block(block).hash != -1 for block in seq.block_table[:10])
        engine.postprocess(
            [seq],
            ScheduledBatchOutput(
                req_ids=[seq.id],
                token_ids=[(7,)],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )
    # The two blocks the suffix filled chain their hashes onto the loaded ones.
    assert all(kv.block(block).hash != -1 for block in seq.block_table[:12])
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
    # A later turn finds the prefix in HBM and loads nothing.
    again = Sequence(list(range(49)), BLOCK, sampling_params=SamplingParams())
    engine.add(again)
    batch, _ = engine.schedule()
    assert again.status == SequenceStatus.RUNNING
    assert [r for r in batch.connector_meta_output.requests if r.load_spec] == []
    assert again.num_cached_tokens >= 40


def test_dense_scheduler_load_request_is_unchanged(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=CHUNK),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "kv_consumer"},
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=1,
        tensor_parallel_size=1,
    )
    scheduler = DenseOffloadScheduler(config)
    seq = _seq(10, 40)
    _arm_load(scheduler, seq, hbm=8, lmc=32)
    [request] = scheduler.build_connector_meta().requests
    assert request.token_ids == list(range(32))
    assert request.chunk_hashes is None
    assert request.block_ids == seq.block_table
    assert not hasattr(seq, "offload_load_start_tokens")


def test_scheduler_settings(cluster):
    scheduler = MooncakeStoreOffloadScheduler(
        _config(extra=_extra(save_abandon_timeout_s=120))
    )
    assert scheduler.save_abandon_timeout_s() == 120.0
    assert scheduler._early_release and scheduler.chunk_size == CHUNK
    assert scheduler.max_pending_saves is None and scheduler._may_emit_save()


def test_max_pending_saves_bounds_saves_in_flight(cluster):
    extra = _extra()
    extra["max_pending_saves"] = 1
    scheduler = MooncakeStoreOffloadScheduler(_config("kv_producer", extra=extra))
    assert scheduler.max_pending_saves == 1
    seqs = [_seq(20 + i, 16) for i in range(2)]
    for seq in seqs:
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 16
    assert len(scheduler.build_connector_meta().requests) == 1
    assert len(scheduler.build_connector_meta().requests) == 0
    [operation] = scheduler._save_inflight.values()
    scheduler.save_finished(operation)
    assert len(scheduler.build_connector_meta().requests) == 1


# --- worker: save -------------------------------------------------------------


def test_save_puts_every_chunk_tail_first_with_one_terminal(cluster, make_worker):
    worker, gpu = make_worker(save_slots=8)
    request = _save_req(11, 40)
    operation = request.save_operation
    worker._do_save_req(request, producer_event="fence")

    # 8 save slots: windows of a quarter, 2 chunks, highest first.
    assert gpu.calls == [("from", [24, 32]), ("from", [8, 16]), ("from", [0])]
    assert gpu.events == ["fence"] * 3
    objects = cluster.objects[MASTER]
    for index in range(5):
        expected = gpu.pattern(2 * index) + gpu.pattern(2 * index + 1)
        assert objects[_key(request.chunk_hashes, index)] == expected
    output = worker.get_finished()
    assert output.finished_saving == {operation}
    assert output.connector_completions == {
        _store(operation),
        _quiescent(operation),
        *(_safe(operation, index) for index in range(5)),
    }
    pool = worker._pool
    assert pool.quarantined("save") == 0 and len(pool.acquire("save", 2)) == 2


def test_every_stage_puts_a_chunk_in_the_same_group(cluster, make_worker):
    stages = [make_worker(rank=rank, world=2, save_slots=8)[0] for rank in range(2)]
    request = _save_req(51, 40)
    for stage in stages:
        stage._do_save_req(copy.deepcopy(request))
    groups = cluster.groups[MASTER]
    for index in range(5):
        digest = keys.chunk_digest(request.chunk_hashes, index)
        group = keys.chunk_group_id(NAMESPACE, digest)
        assert group == f"{NAMESPACE}/group/{digest.hex()}"
        for rank in range(2):
            assert (
                groups[_key(request.chunk_hashes, index, rank=rank, world=2)] == group
            )
    assert len(groups) == 10 and len(set(groups.values())) == 5


def test_chunk_groups_can_be_turned_off(cluster, make_worker):
    worker, _ = make_worker(config=_config(extra=_extra(chunk_groups=False)))
    worker._do_save_req(_save_req(52, 40))
    [store] = cluster.stores
    assert store.put_group_ids and all(ids is None for ids in store.put_group_ids)
    assert cluster.groups[MASTER] == {}


def test_put_refuses_group_ids_that_do_not_match_its_keys(cluster):
    client = store_client.MooncakeStoreClient(
        local_hostname="10.0.0.2",
        metadata_server=METADATA,
        master_server_addr=MASTER,
        protocol="tcp",
        rdma_devices="",
    )
    try:
        with pytest.raises(ValueError, match="one per key"):
            client.put(["a", "b"], [1, 2], [8, 8], group_ids=["g"])
        [store] = cluster.stores
        assert store.put_group_ids == []
    finally:
        client.close()


def test_save_skips_what_is_already_stored(cluster, make_worker):
    worker, gpu = make_worker()
    request = _save_req(12, 40, skip=16)
    worker._do_save_req(request)
    assert gpu.calls == [("from", [16, 24, 32])]
    assert len(cluster.objects[MASTER]) == 3
    assert _key(request.chunk_hashes, 1) not in cluster.objects[MASTER]

    done = _save_req(13, 16, skip=16)
    worker._do_save_req(done)
    assert gpu.calls[-1] == ("from", [16, 24, 32])  # nothing new read
    assert worker.get_finished().connector_completions >= {
        _store(done.save_operation),
        _quiescent(done.save_operation),
    }


def test_putting_an_existing_chunk_succeeds(cluster, make_worker):
    worker, _ = make_worker()
    first = _save_req(14, 16)
    worker._do_save_req(first)
    worker.get_finished()
    again = _save_req(15, 16, generation=1)
    worker._do_save_req(again)
    assert _store(again.save_operation) in worker.get_finished().connector_completions


def test_put_failure_is_quiescent_and_reports_every_chunk(cluster, make_worker):
    worker, gpu = make_worker(save_slots=8)
    request = _save_req(16, 40)
    operation = request.save_operation
    worker._client._store.put_codes[_key(request.chunk_hashes, 3)] = (
        store_client.NO_AVAILABLE_HANDLE
    )
    worker._do_save_req(request)

    assert gpu.calls == [("from", [24, 32])]  # the first window failed: stop
    output = worker.get_finished()
    assert output.finished_saving == {operation}
    # Chunks 3-4 were read (their copies fenced); 0-2 never will be. Both are
    # source-safe, so every PP stage reports the same per-chunk set.
    assert output.connector_completions == {
        _store(operation, False),
        _quiescent(operation),
        *(_safe(operation, index) for index in range(5)),
    }
    assert worker._pool.quarantined("save") == 0  # no transfer left behind


def test_a_put_transfer_failure_before_the_batch_wait_frees_its_slots(
    cluster, make_worker
):
    # A dead owner fails a put in about a second: nothing is left posted.
    worker, _ = make_worker(save_slots=8)
    request = _save_req(17, 16)
    worker._client._store.put_codes[_key(request.chunk_hashes, 1)] = (
        store_client.TRANSFER_FAIL
    )
    worker._do_save_req(request)
    assert worker._pool.quarantined("save") == 0
    assert _store(request.save_operation, False) in (
        worker.get_finished().connector_completions
    )


def test_a_put_the_batch_wait_gave_up_on_quarantines_only_its_slot(
    cluster, make_worker, slow_calls, monkeypatch
):
    worker, _ = make_worker(save_slots=8)
    request = _save_req(17, 16)
    worker._client._store.put_codes[_key(request.chunk_hashes, 1)] = (
        store_client.TRANSFER_FAIL
    )
    worker._do_save_req(request)
    pool = worker._pool
    # The chunk that went through settled; the one whose RDMA read of the slot
    # may still be posted is out of use for good.
    assert (pool.quarantined("save"), pool.usable("save")) == (1, 7)
    assert _store(request.save_operation, False) in (
        worker.get_finished().connector_completions
    )
    # However long ago: nothing bounds how late that RDMA work can land.
    later = time.monotonic() + 3600
    monkeypatch.setattr(time, "monotonic", lambda: later)
    assert (pool.quarantined("save"), pool.usable("save")) == (1, 7)


def test_a_put_that_raises_quarantines_its_window(cluster, make_worker):
    worker, _ = make_worker(save_slots=8)
    worker._client._store.raise_on = "put"
    request = _save_req(18, 16)
    worker._do_save_req(request)
    pool = worker._pool
    assert (pool.quarantined("save"), pool.usable("save")) == (2, 6)
    output = worker.get_finished()
    assert output.connector_completions >= {
        _store(request.save_operation, False),
        _quiescent(request.save_operation),
    }
    # The rest of the region still saves.
    worker._client._store.raise_on = None
    later = _save_req(19, 16, generation=1)
    worker._do_save_req(later)
    assert _store(later.save_operation) in worker.get_finished().connector_completions


def test_a_region_with_no_usable_slot_fails_saves_at_once(cluster, make_worker):
    worker, gpu = make_worker(save_slots=2)
    worker._pool.quarantine(worker._pool.acquire("save", 2))
    request = _save_req(19, 16)
    worker._do_save_req(request)
    assert gpu.calls == []
    # Nothing was read, so the failure is quiescent and every chunk safe.
    assert worker.get_finished().connector_completions == {
        _store(request.save_operation, False),
        _quiescent(request.save_operation),
        _safe(request.save_operation, 0),
        _safe(request.save_operation, 1),
    }


def test_gpu_copy_failure_claims_nothing(cluster, make_worker):
    worker, gpu = make_worker(save_slots=2)
    gpu.fail_from = True
    request = _save_req(20, 16)
    worker._guard("save", worker._do_save_req, request, producer_event=None)
    output = worker.get_finished()
    assert output.finished_saving == {request.save_operation}
    # A failed copy may still be reading: no quiescent claim, no source-safe.
    assert output.connector_completions == {_store(request.save_operation, False)}
    # The device fenced (a CPU no-op here), so the slots are reusable.
    assert worker._pool.quarantined("save") == 0
    assert len(worker._pool.acquire("save", 2)) == 2


def test_gpu_copy_failure_without_a_fence_retires_its_slots(
    cluster, make_worker, monkeypatch
):
    worker, gpu = make_worker(save_slots=8)
    gpu.fail_from = True

    def no_fence(_device):
        raise RuntimeError("sticky HIP error")

    monkeypatch.setattr(worker_mod, "_synchronize", no_fence)
    worker._guard("save", worker._do_save_req, _save_req(21, 16))
    # A copy that may still run has no time bound: out of use for good.
    assert worker._pool.quarantined("save") == 2


def test_a_save_queued_past_its_deadline_reads_nothing(cluster, make_worker):
    worker, gpu = make_worker()
    request = _save_req(22, 24)
    request._mooncake_store_dispatched_at = time.monotonic() - 10_000
    worker._do_save_req(request)
    assert gpu.calls == []
    assert cluster.objects.get(MASTER, {}) == {}
    operation = request.save_operation
    assert worker.get_finished().connector_completions == {
        _store(operation, False),
        _quiescent(operation),
        *(_safe(operation, index) for index in range(3)),
    }
    # The deadline leaves a margin inside the scheduler's reclaim window.
    assert worker._source_read_window_s == 240.0


@pytest.mark.parametrize(("runs_after_s", "stored"), [(7.5, True), (8.5, False)])
def test_a_saves_deadline_counts_from_its_dispatch_not_its_receipt(
    cluster, make_worker, monkeypatch, runs_after_s, stored
):
    clock = SimpleNamespace(monotonic=5000.0, wall=1.75e9)
    monkeypatch.setattr(
        worker_mod,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.monotonic,
            time=lambda: clock.wall,
            perf_counter=time.perf_counter,
        ),
    )
    # Enqueue nothing: the test runs the save when its thread would get to it.
    monkeypatch.setattr(DenseOffloadConnector, "start_load_kv", lambda *_: None)
    worker, gpu = make_worker(config=_config(extra=_extra(save_abandon_timeout_s=10)))
    # The scheduler may reclaim the source 10 s after the dispatch.
    assert worker._source_read_window_s == 8.0
    request = _save_req(26, 24)
    request.dispatched_at = clock.wall
    # A downstream PP stage receives it 3 s later, after the forwards queued
    # ahead of it.
    clock.monotonic += 3.0
    clock.wall += 3.0
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(request)
    worker.start_load_kv(metadata)
    assert request._mooncake_store_dispatched_at == clock.monotonic - 3.0
    clock.monotonic += runs_after_s - 3.0
    worker._do_save_req(request)
    operation = request.save_operation
    completions = worker.get_finished().connector_completions
    if stored:
        assert len(cluster.objects[MASTER]) == 3
        assert _store(operation) in completions
    else:
        # Past the window counted from the dispatch, though not from receipt.
        assert gpu.calls == [] and cluster.objects.get(MASTER, {}) == {}
        assert completions == {
            _store(operation, False),
            _quiescent(operation),
            *(_safe(operation, index) for index in range(3)),
        }


def test_a_late_exception_still_leaves_exactly_one_terminal(
    cluster, make_worker, monkeypatch
):
    worker, _ = make_worker()

    def broken_stats():
        raise RuntimeError("logging broke")

    monkeypatch.setattr(worker, "_maybe_log_store_stats", broken_stats)
    save = _save_req(25, 16)
    worker._guard("save", worker._do_save_req, save)
    completions = worker.get_finished().connector_completions
    stores = [c for c in completions if c.channel == DENSE_PAGE_STORE_CHANNEL]
    assert stores == [_store(save.save_operation, False)]

    load = _load_req(25, hbm=0, lmc=16, block_ids=list(range(4)))
    worker._guard("load", worker._do_load_req, load)
    output = worker.get_finished()
    assert output.failed_loading == {load.load_operation}
    assert output.finished_loading == set()


def test_start_load_kv_stamps_saves_and_shares_one_fence(
    cluster, make_worker, monkeypatch
):
    events = []

    class Event:
        def __init__(self):
            events.append(self)

        def record(self, _stream):
            pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", object)
    worker, gpu = make_worker()
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(_save_req(23, 16))
    metadata.add_request(_save_req(24, 16, offset=100, block_ids=[8, 9, 10, 11]))
    before = time.monotonic()
    worker.start_load_kv(metadata)
    worker._save_executor.shutdown(wait=True)
    # No dispatch time: the deadline counts from receipt.
    assert all(r._mooncake_store_dispatched_at >= before for r in metadata.requests)
    assert len(events) == 1 and gpu.events == [events[0], events[0]]
    output = worker.get_finished()
    assert output.finished_saving == {r.save_operation for r in metadata.requests}
    assert len(cluster.objects[MASTER]) == 4


# --- worker: load ------------------------------------------------------------


def test_load_round_trips_into_other_blocks(cluster, make_worker):
    worker, gpu = make_worker(load_slots=8)
    worker._do_save_req(_save_req(30, 40))
    worker.get_finished()
    destination = list(range(100, 110))
    request = _load_req(30, hbm=0, lmc=40, block_ids=destination)
    worker._do_load_req(request)

    # 8 load slots: windows of a quarter, 2 chunks, lowest first.
    assert gpu.calls[-3:] == [("to", [0, 8]), ("to", [16, 24]), ("to", [32])]
    assert [gpu.kv[block] for block in destination] == [
        gpu.pattern(block) for block in range(10)
    ]
    output = worker.get_finished()
    assert output.finished_loading == {request.load_operation}
    assert output.failed_loading == set()
    assert worker._pool.quarantined("load") == 0


def test_load_starts_at_the_hbm_frontier_and_honours_the_transfer_end(
    cluster, make_worker
):
    worker, gpu = make_worker()
    worker._do_save_req(_save_req(31, 40))
    request = _load_req(31, hbm=16, lmc=32, end=40, block_ids=list(range(50, 60)))
    worker._do_load_req(request)
    assert gpu.calls[-1] == ("to", [16, 24, 32])
    [get] = [call for call in worker._client._store.calls if call[0] == "get"]
    assert get[1] == [_key(request.chunk_hashes, i) for i in (2, 3, 4)]
    assert worker.get_finished().finished_loading == {request.load_operation}


def test_load_with_nothing_to_load_is_done(cluster, make_worker):
    worker, gpu = make_worker()
    request = _load_req(32, hbm=16, lmc=16, block_ids=[0, 1, 2, 3])
    worker._do_load_req(request)
    assert gpu.calls == [] and worker._client._store.calls == [
        call for call in worker._client._store.calls if call[0] == "register"
    ]
    assert worker.get_finished().finished_loading == {request.load_operation}


@pytest.mark.parametrize(
    ("hbm", "lmc", "end", "num_tokens"),
    [
        (4, 24, None, None),  # HBM prefix not on a chunk
        (0, 24, 20, 24),  # transfer end short of the hit
        (0, 24, None, 16),  # too few digests
    ],
)
def test_load_refuses_requests_it_cannot_serve(
    cluster, make_worker, hbm, lmc, end, num_tokens
):
    worker, gpu = make_worker()
    request = _load_req(
        33, hbm=hbm, lmc=lmc, end=end, block_ids=list(range(8)), num_tokens=num_tokens
    )
    worker._do_load_req(request)
    assert gpu.calls == []
    output = worker.get_finished()
    assert output.failed_loading == {request.load_operation}
    assert worker.take_load_error_blocks() == set(range(hbm // BLOCK, lmc // BLOCK))


def test_load_miss_fails_all_or_nothing(cluster, make_worker):
    worker, gpu = make_worker(load_slots=8)
    worker._do_save_req(_save_req(34, 40))
    request = _load_req(34, hbm=8, lmc=40, block_ids=list(range(100, 110)))
    del cluster.objects[MASTER][_key(request.chunk_hashes, 3)]  # evicted
    worker._do_load_req(request)
    # The first window landed; the second missed, so the load fails whole.
    assert gpu.calls[-1] == ("to", [8, 16])
    output = worker.get_finished()
    assert output.failed_loading == {request.load_operation}
    assert output.finished_loading == set()
    assert worker.take_load_error_blocks() == set(range(102, 110))
    assert worker._pool.quarantined("load") == 0


def test_load_of_another_layout_fails(cluster, make_worker):
    worker, gpu = make_worker()
    request = _load_req(35, hbm=0, lmc=16, block_ids=list(range(4)))
    objects = cluster.objects.setdefault(MASTER, {})
    objects[_key(request.chunk_hashes, 0)] = bytes(CHUNK_BYTES)
    objects[_key(request.chunk_hashes, 1)] = bytes(CHUNK_BYTES - 8)  # foreign size
    worker._do_load_req(request)
    assert gpu.calls == []
    assert worker.get_finished().failed_loading == {request.load_operation}


def test_load_transfer_failure_before_the_batch_wait_frees_its_slots(
    cluster, make_worker
):
    worker, _ = make_worker()
    worker._do_save_req(_save_req(36, 16))
    request = _load_req(36, hbm=0, lmc=16, block_ids=list(range(4)))
    worker._client._store.get_codes[_key(request.chunk_hashes, 0)] = (
        store_client.TRANSFER_FAIL
    )
    worker._do_load_req(request)
    assert worker._pool.quarantined("load") == 0
    assert worker.get_finished().failed_loading == {request.load_operation}
    # The next load of the same chunks finds every slot.
    del worker._client._store.get_codes[_key(request.chunk_hashes, 0)]
    again = _load_req(36, hbm=0, lmc=16, block_ids=list(range(4)), generation=1)
    worker._do_load_req(again)
    assert worker.get_finished().finished_loading == {again.load_operation}


def test_a_load_the_batch_wait_gave_up_on_quarantines_only_its_slot(
    cluster, make_worker, slow_calls, monkeypatch
):
    worker, _ = make_worker(load_slots=8)
    worker._do_save_req(_save_req(36, 16))
    request = _load_req(36, hbm=0, lmc=16, block_ids=list(range(4)))
    worker._client._store.get_codes[_key(request.chunk_hashes, 0)] = (
        store_client.TRANSFER_FAIL
    )
    worker._do_load_req(request)
    pool = worker._pool
    # The chunk that arrived settled; the one whose RDMA write may still land
    # is out of use for good.
    assert (pool.quarantined("load"), pool.usable("load")) == (1, 7)
    assert worker.get_finished().failed_loading == {request.load_operation}
    later = time.monotonic() + 3600
    monkeypatch.setattr(time, "monotonic", lambda: later)
    assert (pool.quarantined("load"), pool.usable("load")) == (1, 7)


def test_a_get_that_raises_quarantines_its_window(cluster, make_worker):
    worker, _ = make_worker(load_slots=8)
    worker._do_save_req(_save_req(38, 16))
    worker._client._store.raise_on = "get"
    request = _load_req(38, hbm=0, lmc=16, block_ids=list(range(4)))
    worker._do_load_req(request)
    assert (worker._pool.quarantined("load"), worker._pool.usable("load")) == (2, 6)
    assert worker.get_finished().failed_loading == {request.load_operation}


def test_load_gpu_failure_reports_one_failure(cluster, make_worker):
    worker, gpu = make_worker()
    worker._do_save_req(_save_req(37, 16))
    worker.get_finished()
    gpu.fail_to = True
    request = _load_req(37, hbm=0, lmc=16, block_ids=list(range(4)))
    worker._guard("load", worker._do_load_req, request)
    output = worker.get_finished()
    assert output.failed_loading == {request.load_operation}
    assert output.finished_loading == set()
    assert worker._pool.quarantined("load") == 0


def test_worker_close_releases_everything_once(cluster, make_worker):
    worker, gpu = make_worker()
    store = worker._client._store
    worker.close()
    worker.close()
    assert gpu.closed and store.closed and store.registered == {}
    assert worker._pool is None and worker._client is None


def test_worker_close_keeps_a_pool_a_transfer_may_still_reach(
    cluster, make_worker, slow_calls
):
    worker, _ = make_worker(save_slots=8)
    request = _save_req(39, 16)
    worker._client._store.put_codes[_key(request.chunk_hashes, 1)] = (
        store_client.TRANSFER_FAIL
    )
    worker._do_save_req(request)
    store, pool = worker._client._store, worker._pool
    worker.close()
    # The executors are joined, but the abandoned put's RDMA read may still be
    # posted: the client's teardown ends it, and the memory is never handed out.
    assert store.closed and store.registered == {pool.base_ptr: pool.nbytes}
    assert len(pool_mod._UNSETTLED_ALLOCATIONS) == 1


# --- end to end on CPU: scheduler, worker, aggregation ----------------------


def test_the_scheduler_finds_what_the_worker_stored(cluster, make_worker):
    config = _config()
    scheduler = MooncakeStoreOffloadScheduler(config)
    worker, _ = make_worker(
        config=config, namespace=keys.store_namespace(config, CHUNK)
    )
    seq = _seq(40, 44)
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 44
    [request] = scheduler.build_connector_meta().requests
    worker._do_save_req(request)
    scheduler.process_completions(worker.get_finished())
    assert scheduler._save_inflight == {}

    # A later turn of the same conversation finds every stored chunk.
    turn = _seq(41, 60)
    assert scheduler.get_num_new_matched_tokens(turn) == (40, True)


def test_a_failing_save_is_retried_after_a_pause_that_grows(
    cluster, make_worker, scheduler_clock
):
    config = _config()
    scheduler = MooncakeStoreOffloadScheduler(config)
    worker, _ = make_worker(
        config=config, namespace=keys.store_namespace(config, CHUNK)
    )
    seq = _seq(43, 44)
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 44

    def step():
        requests = scheduler.build_connector_meta().requests
        for request in requests:
            worker._do_save_req(request)
        scheduler.process_completions(worker.get_finished())
        return len(requests)

    # A full Store fails every put at once, and the base retries a failed
    # save on the next step: without a pause every PP stage would re-pack it
    # every step.
    cluster.put_code = store_client.NO_AVAILABLE_HANDLE
    assert step() == 1
    assert step() == 0
    scheduler_clock.now += 1.0
    assert step() == 1  # failed again: the next pause is 2 s
    scheduler_clock.now += 1.0
    assert step() == 0
    scheduler_clock.now += 1.0
    cluster.put_code = None
    assert step() == 1
    assert scheduler._save_inflight == {}
    # Stored: the next save goes out at once.
    assert scheduler._failed_saves_in_a_row == 0
    later = _seq(44, 20)
    scheduler.update_state_after_alloc(later)
    later.num_cached_tokens = 20
    assert step() == 1


def test_a_stage_that_stops_early_still_completes_every_quorum(cluster, make_worker):
    stages = [make_worker(rank=rank, world=2, save_slots=8)[0] for rank in range(2)]
    request = _save_req(42, 40)
    stages[1]._client._store.put_codes[
        _key(request.chunk_hashes, 1, rank=1, world=2)
    ] = store_client.NO_AVAILABLE_HANDLE
    aggregator = PPKVAggregator(pp_size=2)
    merged = []
    for rank, stage in enumerate(stages):
        stage._do_save_req(copy.deepcopy(request))
        merged.append(aggregator.ingest(rank, stage.get_finished()))
    operation = request.save_operation
    assert merged[1].finished_saving == {operation}
    assert merged[1].connector_completions == {
        _store(operation, False),
        _quiescent(operation),
        *(_safe(operation, index) for index in range(5)),
    }
    assert not aggregator.has_pending()


# --- registration and startup ------------------------------------------------


def test_registered_under_its_name_and_alias():
    assert KVConnectorFactory.canonical_name("mooncake_store") == "mooncake_store"
    assert (
        KVConnectorFactory.canonical_name("MooncakeStoreOffloadConnector")
        == "mooncake_store"
    )


def test_a_second_offload_connector_beside_it_is_refused_at_startup(cluster):
    assert KVConnectorFactory.is_offload_backend("mooncake_store")
    assert not KVConnectorFactory.is_offload_backend("mooncake")
    store = {
        "kv_connector": "mooncake_store",
        "kv_role": "offload",
        "kv_connector_extra_config": _extra(),
    }
    for other in (store, {"kv_connector": "lmcache_mp", "kv_role": "offload"}):
        config = _config()
        config.kv_transfer_config = {
            "kv_connector": "multi",
            "connectors": [store, other],
        }
        for role in ("scheduler", "worker"):
            with pytest.raises(ValueError, match="at most one offload sub-connector"):
                KVConnectorFactory.create_connector(config, role=role)
    assert cluster.stores == []


def test_factory_builds_both_halves(cluster):
    scheduler = KVConnectorFactory.create_connector(_config(), role="scheduler")
    worker = KVConnectorFactory.create_connector(_config(), role="worker")
    try:
        assert isinstance(scheduler, MooncakeStoreOffloadScheduler)
        assert scheduler.is_offload and not scheduler.is_producer
        assert isinstance(worker, MooncakeStoreOffloadConnector)
        assert worker.chunk_size == CHUNK and worker._early_release
    finally:
        worker.close()


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
def one_rank(monkeypatch):
    monkeypatch.setattr(
        worker_mod,
        "_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )


def _registering_config(**overrides):
    return _config(extra=_extra(save_pool_mib=1, load_pool_mib=1, **overrides))


def test_registration_builds_the_pool_and_probes_the_store(cluster, one_rank):
    worker = MooncakeStoreOffloadConnector(_registering_config())
    try:
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
        [store] = cluster.stores
        pool = worker._pool
        # 2 layers x 4 tokens x 8 bytes per block, two blocks per chunk.
        assert worker._chunk_bytes == pool.chunk_bytes == 128
        assert store.registered == {pool.base_ptr: pool.nbytes}
        assert (pool.capacity("save"), pool.capacity("load")) == (256, 256)
        puts = [call for call in store.calls if call[0] == "put"]
        assert len(puts) == 1 and "/probe/w0/" in puts[0][1][0]
        assert store.put_group_ids == [None]  # the probe's key is not grouped
        assert store.objects == {}  # the probe removed its key
        assert ("remove", puts[0][1][0], True) in store.calls
        assert worker._namespace == keys.store_namespace(worker._config, CHUNK)
    finally:
        worker.close()
    assert store.closed and store.registered == {}


def test_a_failed_probe_stops_startup_and_releases_the_pool(cluster, one_rank):
    cluster.silent_gets = True
    worker = MooncakeStoreOffloadConnector(_registering_config())
    try:
        with pytest.raises(RuntimeError, match="differs from the one written"):
            worker.register_kv_caches(_kv_caches(), num_blocks=8)
        [store] = cluster.stores
        assert store.closed and store.registered == {}
        assert worker._pool is None
        assert pool_mod._UNSETTLED_ALLOCATIONS == []
    finally:
        worker.close()


def test_a_probe_put_failing_at_once_releases_the_pool(cluster, one_rank):
    # A dead owner fails the put before any RDMA work is posted.
    cluster.put_code = store_client.TRANSFER_FAIL
    worker = MooncakeStoreOffloadConnector(_registering_config())
    try:
        with pytest.raises(RuntimeError, match=r"put of 128 bytes .*TRANSFER_FAIL"):
            worker.register_kv_caches(_kv_caches(), num_blocks=8)
        [store] = cluster.stores
        assert store.closed and store.registered == {}
        assert pool_mod._UNSETTLED_ALLOCATIONS == []
    finally:
        worker.close()


@pytest.mark.parametrize("failure", ["batch_wait", "raise"])
def test_a_probe_transfer_left_unsettled_keeps_the_pool(
    cluster, one_rank, monkeypatch, failure
):
    if failure == "batch_wait":
        cluster.put_code = store_client.TRANSFER_FAIL
        monkeypatch.setattr(
            store_client.CallClock, "seconds", lambda _self: store_client.BATCH_WAIT_S
        )
    else:
        cluster.raise_on = "put"
    worker = MooncakeStoreOffloadConnector(_registering_config())
    try:
        with pytest.raises(RuntimeError, match="TRANSFER_FAIL|put blew up"):
            worker.register_kv_caches(_kv_caches(), num_blocks=8)
        [store] = cluster.stores
        # Startup gives up, but the probe's put may still read its slot: the
        # pool stays registered and allocated while the client tears down.
        assert store.closed and len(store.registered) == 1
        assert len(pool_mod._UNSETTLED_ALLOCATIONS) == 1
        assert worker._pool is None
    finally:
        worker.close()


def test_registration_without_the_probe(cluster, one_rank):
    worker = MooncakeStoreOffloadConnector(_registering_config(startup_probe=False))
    try:
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
        assert [c[0] for c in cluster.stores[0].calls] == ["register"]
    finally:
        worker.close()


def test_rdma_workers_need_one_qp_and_use_their_pool(cluster, one_rank, monkeypatch):
    config = _config(
        extra=_extra(
            master=None,
            metadata=None,
            pools=POOLS,
            protocol="rdma",
            save_pool_mib=1,
            load_pool_mib=1,
        )
    )
    monkeypatch.setattr(worker_mod, "_device_index", lambda device: 0)
    monkeypatch.setattr(worker_mod, "requester_rdma_device", lambda index, cfg: "rdma3")
    monkeypatch.delenv("MC_NUM_QP_PER_EP", raising=False)
    worker = MooncakeStoreOffloadConnector(config)
    try:
        with pytest.raises(ValueError, match="MC_NUM_QP_PER_EP=1"):
            worker.register_kv_caches(_kv_caches(), num_blocks=8)
        monkeypatch.setenv("MC_NUM_QP_PER_EP", "1")
        monkeypatch.setenv("MC_MS_AUTO_DISC", "1")
        with pytest.raises(ValueError, match="MC_MS_AUTO_DISC"):
            worker.register_kv_caches(_kv_caches(), num_blocks=8)
        monkeypatch.delenv("MC_MS_AUTO_DISC")
        worker.register_kv_caches(_kv_caches(), num_blocks=8)
        [store] = cluster.stores
        assert store.setup_args[4:] == ("rdma", "rdma3", "10.0.0.1:26351")
        assert store.setup_args[1] == "http://10.0.0.1:26380/metadata"
    finally:
        worker.close()


# --- ATOM_KV_OFFLOAD -----------------------------------------------------------


def test_kv_offload_mode_selects_the_store_connector():
    extra = {"mooncake_store.master": MASTER, "mooncake_store.metadata": METADATA}
    offload = kv_offload_connector_config("mooncake_store", json.dumps(extra))
    assert offload == {
        "kv_connector": "mooncake_store",
        "kv_role": "offload",
        "kv_connector_extra_config": extra,
    }
    producer = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    assert json.loads(compose_kv_offload_config(json.dumps(producer), offload)) == {
        "kv_connector": "multi",
        "connectors": [producer, offload],
    }
    for name in ("mooncake_store", "MooncakeStoreOffloadConnector"):
        existing = {"kv_connector": "multi", "connectors": [{"kv_connector": name}]}
        with pytest.raises(ValueError, match="conflicts with an offload connector"):
            compose_kv_offload_config(
                json.dumps(existing), kv_offload_connector_config("lmcache", "")
            )


# --- in-place copies -----------------------------------------------------------

# A chunk that fills its slot exactly, so a window of slots is one buffer.
PAGE_BLOCK_BYTES = 2048
PAGE_CHUNK_BYTES = PAGE_BLOCK_BYTES * CHUNK // BLOCK  # 4096: no padding


def _page_pattern(block: int) -> bytes:
    return bytes((block * 11 + i) % 253 for i in range(PAGE_BLOCK_BYTES))


class FakeStream:
    def __init__(self) -> None:
        self.waited: list = []
        self.synchronized = 0

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        self.synchronized += 1


class FakeInPlaceCodec:
    """The dense codec's prepared API over host bytes, block by block."""

    has_fused_chunk_major_staging = True

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.kv: dict[int, bytes] = {}
        self.calls: list[tuple] = []
        self.fail = False

    def prepare_block_id_groups(self, groups, *, device, stream):
        self.calls.append(("prepare", [[list(chunk) for chunk in g] for g in groups]))
        return SimpleNamespace(groups=groups, stream=stream)

    def gpu_to_chunk_major_device_buffer_prepared(self, buf, owner, index, *, stream):
        assert stream is owner.stream
        if self.fail:
            raise RuntimeError("pack failed")
        blocks = [block for chunk in owner.groups[index] for block in chunk]
        data = b"".join(self.kv.get(b, _page_pattern(b)) for b in blocks)
        buf[: len(data)].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        self.calls.append(("pack", blocks))

    def chunk_major_device_buffer_to_gpu_prepared(self, buf, owner, index, *, stream):
        assert stream is owner.stream
        blocks = [block for chunk in owner.groups[index] for block in chunk]
        data = buf.numpy().tobytes()
        for position, block in enumerate(blocks):
            start = position * PAGE_BLOCK_BYTES
            self.kv[block] = data[start : start + PAGE_BLOCK_BYTES]
        self.calls.append(("unpack", blocks))


@pytest.fixture
def make_in_place_worker(cluster, monkeypatch):
    """A worker whose windows take the in-place path over a host pool."""
    built = []

    def build(*, save_slots=8, load_slots=8):
        worker = MooncakeStoreOffloadConnector(_config())
        gpu = FakeGPUConnector(worker._source_group_safe)
        gpu.gpu_staging_chunk_bytes = PAGE_CHUNK_BYTES
        codec = FakeInPlaceCodec()
        stream = FakeStream()
        client = store_client.MooncakeStoreClient(
            local_hostname="10.0.0.2",
            metadata_server=METADATA,
            master_server_addr=MASTER,
            protocol="tcp",
            rdma_devices="",
        )
        pool = TransferSlotPool(
            device="cpu",
            chunk_bytes=PAGE_CHUNK_BYTES,
            save_bytes=save_slots * PAGE_CHUNK_BYTES,
            load_bytes=load_slots * PAGE_CHUNK_BYTES,
            client=client,
        )
        worker._gpu_connector, worker._client, worker._pool = gpu, client, pool
        worker._codec = codec
        worker._namespace = NAMESPACE
        worker._rank, worker._world = 0, 1
        worker._chunk_bytes = PAGE_CHUNK_BYTES
        monkeypatch.setattr(worker, "_in_place_copy", lambda _pool: True)
        monkeypatch.setattr(worker, "_in_place_stream", lambda: stream)
        built.append(worker)
        return worker, gpu, codec, stream

    yield build
    for worker in built:
        worker.close()


def test_pool_hands_out_a_window_as_one_run(cluster):
    client = _client()
    pool = TransferSlotPool(
        device="cpu",
        chunk_bytes=PAGE_CHUNK_BYTES,
        save_bytes=6 * PAGE_CHUNK_BYTES,
        load_bytes=PAGE_CHUNK_BYTES,
        client=client,
    )
    first = pool.acquire("save", 2)
    second = pool.acquire("save", 3)
    assert [slot.index for slot in first] == [0, 1]
    assert [slot.index for slot in second] == [2, 3, 4]
    run = pool.contiguous_view(second)
    assert run.numel() == 3 * PAGE_CHUNK_BYTES
    assert run.data_ptr() == second[0].ptr
    pool.release(first)
    # Free slots 0, 1 and 5: the lowest run of two is 0-1.
    assert [slot.index for slot in pool.acquire("save", 2)] == [0, 1]
    # Only 5 is free now; 2-4 are taken, so no run of two exists yet.
    pool.release(second)
    pool.quarantine([pool.acquire("save", 1)[0]], reason="test")
    scattered = pool.acquire("save", 2)
    assert [slot.index for slot in scattered] == [3, 4]
    assert pool.contiguous_view(scattered) is not None
    pool.close()


def test_pool_contiguous_view_refuses_gaps_and_padding(cluster):
    padded = TransferSlotPool(
        device="cpu",
        chunk_bytes=CHUNK_BYTES,
        save_bytes=4 * SLOT,
        load_bytes=SLOT,
        client=_client(),
    )
    assert padded.contiguous_view(padded.acquire("save", 2)) is None
    padded.close()
    exact = TransferSlotPool(
        device="cpu",
        chunk_bytes=PAGE_CHUNK_BYTES,
        save_bytes=4 * PAGE_CHUNK_BYTES,
        load_bytes=2 * PAGE_CHUNK_BYTES,
        client=_client(),
    )
    slots = exact.acquire("save", 4)
    assert exact.contiguous_view([slots[0], slots[2]]) is None
    load = exact.acquire("load", 1)
    assert exact.contiguous_view([slots[3], load[0]]) is None
    exact.close()


def test_in_place_save_packs_each_window_once(cluster, make_in_place_worker):
    worker, gpu, codec, stream = make_in_place_worker(save_slots=8)
    request = _save_req(41, 40)
    operation = request.save_operation
    worker._do_save_req(request, producer_event="fence")

    # Windows of two chunks (a quarter of 8 slots), highest first, one pack each;
    # the staging path is never used.
    assert gpu.calls == []
    packs = [call[1] for call in codec.calls if call[0] == "pack"]
    assert packs == [[6, 7, 8, 9], [2, 3, 4, 5], [0, 1]]
    assert stream.waited == ["fence"] * 3 and stream.synchronized == 3
    objects = cluster.objects[MASTER]
    for index in range(5):
        expected = _page_pattern(2 * index) + _page_pattern(2 * index + 1)
        assert objects[_key(request.chunk_hashes, index)] == expected
    output = worker.get_finished()
    assert output.finished_saving == {operation}
    assert output.connector_completions == {
        _store(operation),
        _quiescent(operation),
        *(_safe(operation, index) for index in range(5)),
    }


def test_in_place_load_unpacks_each_window_once(cluster, make_in_place_worker):
    worker, gpu, codec, stream = make_in_place_worker(save_slots=8, load_slots=8)
    worker._do_save_req(_save_req(42, 40))
    codec.calls.clear()
    destination = list(range(100, 110))
    load = _load_req(42, hbm=8, lmc=40, block_ids=destination)
    synced = stream.synchronized
    worker._do_load_req(load)

    assert gpu.calls == []
    unpacks = [call[1] for call in codec.calls if call[0] == "unpack"]
    assert unpacks == [[102, 103, 104, 105], [106, 107, 108, 109]]
    # Each window's unpack is waited for before its slots return to the pool.
    assert stream.synchronized - synced == 2
    for chunk in range(1, 5):
        for half in range(2):
            source = 2 * chunk + half
            assert codec.kv[destination[source]] == _page_pattern(source)
    output = worker.get_finished()
    assert output.finished_loading == {load.load_operation}
    assert output.failed_loading == set()


def test_scattered_slots_fall_back_to_the_staging_copy(cluster, make_in_place_worker):
    worker, gpu, codec, _stream = make_in_place_worker(save_slots=16)
    gpu.pattern = staticmethod(_page_pattern)
    pool = worker._pool
    # Quarantine every odd slot: 8 usable slots give windows of two, and no two
    # free slots are adjacent, so a window is two scattered slots.
    slots = pool.acquire("save", 16)
    pool.release([slot for slot in slots if slot.index % 2 == 0])
    pool.quarantine([slot for slot in slots if slot.index % 2], reason="test")
    request = _save_req(43, 32)
    worker._do_save_req(request)

    assert not [call for call in codec.calls if call[0] == "pack"]
    assert gpu.calls == [("from", [16, 24]), ("from", [0, 8])]
    objects = cluster.objects[MASTER]
    for index in range(4):
        expected = _page_pattern(2 * index) + _page_pattern(2 * index + 1)
        assert objects[_key(request.chunk_hashes, index)] == expected
    assert _store(request.save_operation) in worker.get_finished().connector_completions


def test_in_place_pack_failure_claims_nothing(cluster, make_in_place_worker):
    worker, _gpu, codec, _stream = make_in_place_worker()
    codec.fail = True
    request = _save_req(44, 16)
    with pytest.raises(RuntimeError, match="pack failed"):
        worker._do_save_req(request)
    completions = worker.get_finished().connector_completions
    assert not any(
        completion.channel == DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL
        for completion in completions
    )
    assert worker._pool.quarantined("save") == 0  # the fence held: released


@pytest.mark.parametrize(
    ("direct_copy", "device", "fused", "expected"),
    [
        (True, "cuda", True, True),
        (False, "cuda", True, False),
        (True, "cpu", True, False),
        (True, "cuda", False, False),
    ],
)
def test_in_place_gate(cluster, direct_copy, device, fused, expected):
    worker = MooncakeStoreOffloadConnector(
        _config(extra=_extra(direct_copy=direct_copy))
    )
    worker._codec = SimpleNamespace(has_fused_chunk_major_staging=fused)
    pool = SimpleNamespace(device=torch.device(device))
    assert worker._in_place_copy(pool) is expected
    worker._codec = None
    assert worker._in_place_copy(pool) is False
    worker.close()


def test_in_place_save_failure_reports_each_chunk_once(cluster, make_in_place_worker):
    worker, _gpu, _codec, _stream = make_in_place_worker(save_slots=8)
    request = _save_req(45, 40)
    operation = request.save_operation
    store = cluster.store_of(MASTER)
    # Windows [3,5), [1,3), [0,1): the second one's put fails.
    store.put_codes[_key(request.chunk_hashes, 2)] = store_client.NO_AVAILABLE_HANDLE
    worker._do_save_req(request)
    completions = worker.get_finished().connector_completions
    safe = [c for c in completions if c.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL]
    assert sorted(safe, key=lambda c: c.operation_id.ranges) == [
        _safe(operation, index) for index in range(5)
    ]
    assert _store(operation, False) in completions
    assert _quiescent(operation) in completions


def test_mixed_windows_report_the_same_chunks(
    cluster, make_in_place_worker, monkeypatch
):
    worker, gpu, codec, _stream = make_in_place_worker(save_slots=8)
    gpu.pattern = staticmethod(_page_pattern)
    pool = worker._pool
    # The first window [2,4) is one run; the second [0,2) is handed out as if
    # scattered, so it takes the staging copy.
    view = pool.contiguous_view
    calls = []

    def first_only(slots):
        calls.append(len(slots))
        return view(slots) if len(calls) == 1 else None

    monkeypatch.setattr(pool, "contiguous_view", first_only)
    request = _save_req(46, 32)
    operation = request.save_operation
    worker._do_save_req(request)
    assert [call[1] for call in codec.calls if call[0] == "pack"] == [[4, 5, 6, 7]]
    assert gpu.calls == [("from", [0, 8])]
    completions = worker.get_finished().connector_completions
    assert completions == {
        _store(operation),
        _quiescent(operation),
        *(_safe(operation, index) for index in range(4)),
    }
    objects = cluster.objects[MASTER]
    for index in range(4):
        expected = _page_pattern(2 * index) + _page_pattern(2 * index + 1)
        assert objects[_key(request.chunk_hashes, index)] == expected
