# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The LMCache MP server entry point with a Mooncake Store L2."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field

import pytest
import torch

from atom.kv_transfer.offload.mooncake_store_l2 import StorePool
from atom.kv_transfer.offload.mp import mooncake_l2_server as server


@pytest.fixture
def ran(monkeypatch):
    """Record the module the entry point runs and the argv it runs it with."""
    runs = []

    def run_module(name, run_name, alter_sys):
        runs.append((name, run_name, alter_sys, list(server.sys.argv)))

    monkeypatch.setattr(server.runpy, "run_module", run_module)
    monkeypatch.setattr(server, "requester_rdma_device", lambda gpu: f"rdma{3 - gpu}")
    monkeypatch.setattr(server, "keep_l1_objects_page_aligned", lambda: None)
    monkeypatch.setattr(server, "make_l1_huge_before_pinning", lambda numa: None)
    monkeypatch.setattr(server, "prefetch_from_lookup_start", lambda: None)
    return runs


def _adapter(argv):
    flag = argv.index("--l2-adapter")
    return json.loads(argv[flag + 1])


def test_server_gets_the_store_l2_of_its_gpus_nic_and_pool(monkeypatch, ran):
    pools = {"rdma2": StorePool("10.0.0.1:26151", "http://10.0.0.1:26180/metadata")}
    monkeypatch.setattr(server, "store_pool_of", lambda device: pools.get(device))

    server.main(
        [
            "--gpu",
            "1",
            "--numa",
            "0",
            "--local-hostname",
            "10.0.0.1",
            "--",
            "--port",
            "25556",
        ]
    )

    name, run_name, alter_sys, argv = ran[0]
    assert (name, run_name, alter_sys) == (
        "lmcache.v1.multiprocess.server",
        "__main__",
        True,
    )
    assert argv[:3] == ["lmcache.v1.multiprocess.server", "--port", "25556"]
    assert _adapter(argv) == {
        "type": "mooncake_store",
        "num_workers": 8,
        "master_server_addr": "10.0.0.1:26151",
        "metadata_server": "http://10.0.0.1:26180/metadata",
        "local_hostname": "10.0.0.1",
        "protocol": "rdma",
        "rdma_devices": "rdma2",
        "global_segment_size": "0",
        "local_buffer_size": "0",
    }
    # One argv word, so a logged command line splits back into the same argv.
    assert " " not in argv[argv.index("--l2-adapter") + 1]


def test_shared_pool_comes_from_the_command_line(monkeypatch, ran):
    monkeypatch.setattr(server, "store_pool_of", lambda device: None)

    server.main(
        [
            "--gpu",
            "0",
            "--numa",
            "0",
            "--local-hostname",
            "10.0.0.1",
            "--master",
            "10.0.0.1:26051",
            "--metadata",
            "http://10.0.0.1:26080/metadata",
            "--",
        ]
    )

    adapter = _adapter(ran[0][3])
    assert adapter["rdma_devices"] == "rdma3"
    assert adapter["master_server_addr"] == "10.0.0.1:26051"


def test_no_pool_at_all_is_refused(monkeypatch, ran):
    monkeypatch.setattr(server, "store_pool_of", lambda device: None)

    with pytest.raises(SystemExit, match="--master and --metadata"):
        server.main(["--gpu", "0", "--numa", "0", "--local-hostname", "10.0.0.1", "--"])
    assert ran == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--gpu", "0", "--numa", "0", "--local-hostname", "h"],
        [
            "--gpu",
            "0",
            "--numa",
            "0",
            "--local-hostname",
            "h",
            "--",
            "--l2-adapter",
            "{}",
        ],
        ["--gpu", "0", "--numa", "0", "--local-hostname", "h", "--", "--l2-adapter={}"],
    ],
)
def test_bad_command_lines_are_refused(argv, ran):
    with pytest.raises(SystemExit):
        server.main(argv)
    assert ran == []


def _lazy_allocator_with(monkeypatch, *, pinned_whole):
    """LMCache's LazyMemoryAllocator, built as its constructor builds it."""
    lazy = pytest.importorskip("lmcache.v1.memory_allocators.lazy_memory_allocator")
    from lmcache.v1.memory_allocators.tensor_memory_allocator import (
        TensorMemoryAllocator,
    )

    size = 8 << 20
    buffer = torch.zeros(size, dtype=torch.uint8)

    def constructor(self, final_size, align_bytes):
        self._final_size = final_size
        self._curr_size = final_size if pinned_whole else final_size // 2
        self._buffer = buffer
        self._allocator = TensorMemoryAllocator(
            tensor=buffer, align_bytes=align_bytes, init_address_space=self._curr_size
        )
        self._address_manager = self._allocator.address_manager

    monkeypatch.setattr(lazy.LazyMemoryAllocator, "__init__", constructor)
    server.keep_l1_objects_page_aligned()
    server.keep_l1_objects_page_aligned()  # idempotent
    return lazy.LazyMemoryAllocator(size, 2 << 20)


def test_l1_pinned_whole_gets_page_aligned_objects(monkeypatch):
    allocator = _lazy_allocator_with(monkeypatch, pinned_whole=True)

    assert allocator._address_manager._align == server.L1_OBJECT_ALIGN_BYTES
    assert allocator._allocator.address_manager is allocator._address_manager
    assert allocator._allocator.buffer.data_ptr() == allocator._buffer.data_ptr()


def test_lazily_growing_l1_keeps_its_allocator(monkeypatch):
    allocator = _lazy_allocator_with(monkeypatch, pinned_whole=False)

    assert allocator._address_manager._align == 2 << 20


def _pinning_allocator(monkeypatch, *, collapsed):
    """LMCache's LazyMemoryAllocator whose pin is wrapped by the THP check."""
    lazy = pytest.importorskip("lmcache.v1.memory_allocators.lazy_memory_allocator")
    calls = []
    buffer = torch.zeros(4 << 20, dtype=torch.uint8)

    def pin(self, offset, size):
        calls.append(("pin", offset, size))

    monkeypatch.setattr(lazy.LazyMemoryAllocator, "_pin_memory_chunk", pin)
    monkeypatch.setattr(
        server,
        "_touch_every_huge_page",
        lambda address, size, node: calls.append(("touch", size, node)),
    )
    monkeypatch.setattr(
        server,
        "_collapse_into_huge_pages",
        lambda address, size: calls.append(("collapse", size)) or collapsed,
    )
    server.make_l1_huge_before_pinning(1)
    server.make_l1_huge_before_pinning(1)  # idempotent
    allocator = lazy.LazyMemoryAllocator.__new__(lazy.LazyMemoryAllocator)
    allocator._buffer = buffer
    return allocator, calls


def test_l1_is_touched_and_collapsed_before_it_is_pinned(monkeypatch):
    allocator, calls = _pinning_allocator(monkeypatch, collapsed=True)

    allocator._pin_memory_chunk(0, 4 << 20)

    assert calls == [("touch", 4 << 20, 1), ("collapse", 4 << 20), ("pin", 0, 4 << 20)]


def test_l1_short_of_huge_pages_fails_before_it_is_pinned(monkeypatch):
    allocator, calls = _pinning_allocator(monkeypatch, collapsed=False)

    with pytest.raises(RuntimeError, match="not all huge pages after MADV_COLLAPSE"):
        allocator._pin_memory_chunk(0, 4 << 20)
    assert ("pin", 0, 4 << 20) not in calls


@dataclass(frozen=True)
class _Key:
    """The fields of LMCache's IPCCacheServerKey the patch reads."""

    token_ids: tuple
    start: int
    end: int
    request_id: str = field(default="req", compare=False)


@dataclass(frozen=True)
class _Row:
    """The field of LMCache's GroupedObjectKeys the patch sets."""

    keys: list
    sliding_window_size: int = -1


@pytest.fixture
def lookups(monkeypatch):
    """LMCache's LookupModule as the LOOKUP path uses it, with the patch on.

    Chunk size 4: a full-attention row and a 3-chunk sliding-window row per
    lookup. ``hit`` is the prefix length in chunks the next prefetch answers;
    ``freed`` records each release's range.
    """
    sessions = {}

    class Session:
        def __init__(self):
            self.lookup_ipc_key = None
            self.prefetch_hit_chunks = -1

    ctx = types.SimpleNamespace(
        chunk_size=4,
        session_manager=types.SimpleNamespace(
            get_or_create=lambda request_id: sessions.setdefault(request_id, Session())
        ),
    )

    class LookupModule:
        def __init__(self, ctx):
            self._ctx = ctx
            self.hit = 0
            self.submitted = []
            self.freed = []

        def lookup(self, key, tp_size):
            hashes = [f"h{i}" for i in range(key.end // self._ctx.chunk_size)]
            session = self._ctx.session_manager.get_or_create(key.request_id)
            session.lookup_ipc_key = key
            session.prefetch_hit_chunks = -1
            self.submitted.append(module.ipc_key_to_grouped_object_keys(key, hashes))

        def query_prefetch_status(self, request_id: str) -> int | None:
            session = self._ctx.session_manager.get_or_create(request_id)
            session.prefetch_hit_chunks = self.hit
            return self.hit

        def free_lookup_locks(self, key, tp_size: int) -> None:
            self.freed.append((key.start, key.end))

    module = types.ModuleType("lmcache.v1.multiprocess.modules.lookup")
    module.LookupModule = LookupModule
    module.ipc_key_to_grouped_object_keys = lambda key, hashes: [
        _Row(list(hashes)),
        _Row(list(hashes), sliding_window_size=3),
    ]
    for name in (
        "lmcache",
        "lmcache.v1",
        "lmcache.v1.multiprocess",
        "lmcache.v1.multiprocess.modules",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["lmcache.v1.multiprocess.modules"].lookup = module
    monkeypatch.setitem(sys.modules, module.__name__, module)

    server.prefetch_from_lookup_start()
    server.prefetch_from_lookup_start()  # idempotent
    return LookupModule(ctx), sessions


def test_lookup_from_chunk_zero_is_unchanged(lookups):
    lookup, _sessions = lookups

    lookup.lookup(_Key(tuple(range(16)), 0, 16), 1)

    assert [row.sliding_window_size for row in lookup.submitted[0]] == [-1, 3]


def test_lookup_looks_every_chunk_up_and_reads_from_its_start(lookups):
    # All four chunks stay in the prefetch -- looking them up renews their
    # Store leases -- but a row reads only the two chunks from the start on.
    lookup, _sessions = lookups

    lookup.lookup(_Key(tuple(range(16)), 8, 16), 1)

    rows = lookup.submitted[0]
    assert [row.keys for row in rows] == [["h0", "h1", "h2", "h3"]] * 2
    assert [row.sliding_window_size for row in rows] == [2, 2]


def test_release_is_clamped_to_the_chunks_the_window_locked(lookups):
    lookup, _sessions = lookups
    lookup.lookup(_Key(tuple(range(16)), 8, 16), 1)
    lookup.hit = 4  # read chunks 2-3
    lookup.query_prefetch_status("req")

    lookup.free_lookup_locks(_Key(tuple(range(16)), 0, 16), 1)
    lookup.free_lookup_locks(_Key(tuple(range(16)), 0, 8), 1)
    lookup.free_lookup_locks(_Key(tuple(range(16)), 12, 16), 1)

    assert lookup.freed == [(8, 16), (12, 16)]


def test_a_hit_short_of_the_end_locks_below_the_start(lookups):
    # A 3-chunk hit with a 2-chunk window read chunks 1-2.
    lookup, _sessions = lookups
    lookup.lookup(_Key(tuple(range(16)), 8, 16), 1)
    lookup.hit = 3
    lookup.query_prefetch_status("req")

    lookup.free_lookup_locks(_Key(tuple(range(16)), 0, 12), 1)

    assert lookup.freed == [(4, 12)]


def test_release_before_the_answer_is_clamped_to_the_lookup_start(lookups):
    lookup, _sessions = lookups
    lookup.lookup(_Key(tuple(range(16)), 8, 16), 1)

    lookup.free_lookup_locks(_Key(tuple(range(16)), 0, 16), 1)

    assert lookup.freed == [(8, 16)]


def test_release_for_a_session_never_looked_up_is_dropped(lookups):
    # LMCache would release the whole range for it, dropping other requests'
    # locks on shared chunks.
    lookup, _sessions = lookups

    lookup.free_lookup_locks(_Key(tuple(range(16)), 0, 16, request_id="other"), 1)

    assert lookup.freed == []


def test_lmcache_still_finds_the_patched_lookup_handlers(monkeypatch):
    lookup_module = pytest.importorskip("lmcache.v1.multiprocess.modules.lookup")
    from lmcache.v1.multiprocess.request_handler import iter_request_handlers

    module = lookup_module.LookupModule
    for name in ("__init__", "free_lookup_locks"):
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(
        lookup_module,
        "ipc_key_to_grouped_object_keys",
        lookup_module.ipc_key_to_grouped_object_keys,
    )
    before = {handler.operation for handler in iter_request_handlers(module)}

    server.prefetch_from_lookup_start()

    after = {handler.operation for handler in iter_request_handlers(module)}
    assert after == before
    assert module.free_lookup_locks._atom_from_lookup_start
