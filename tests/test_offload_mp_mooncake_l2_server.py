# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The LMCache MP server entry point with a Mooncake Store L2."""

from __future__ import annotations

import json

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


def _pinning_allocator(monkeypatch, *, huge_fraction):
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
        lambda address, size: calls.append(("collapse", size)),
    )
    monkeypatch.setattr(
        server,
        "anon_huge_page_bytes",
        lambda start, end: int((end - start) * huge_fraction),
    )
    server.make_l1_huge_before_pinning(1)
    server.make_l1_huge_before_pinning(1)  # idempotent
    allocator = lazy.LazyMemoryAllocator.__new__(lazy.LazyMemoryAllocator)
    allocator._buffer = buffer
    return allocator, calls


def test_l1_is_touched_and_collapsed_before_it_is_pinned(monkeypatch):
    allocator, calls = _pinning_allocator(monkeypatch, huge_fraction=1.0)

    allocator._pin_memory_chunk(0, 4 << 20)

    assert calls == [("touch", 4 << 20, 1), ("collapse", 4 << 20), ("pin", 0, 4 << 20)]


def test_l1_short_of_huge_pages_fails_before_it_is_pinned(monkeypatch):
    allocator, calls = _pinning_allocator(monkeypatch, huge_fraction=0.5)

    with pytest.raises(RuntimeError, match="huge pages after MADV_COLLAPSE"):
        allocator._pin_memory_chunk(0, 4 << 20)
    assert ("pin", 0, 4 << 20) not in calls
