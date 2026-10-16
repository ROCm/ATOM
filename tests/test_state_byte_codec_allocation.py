# SPDX-License-Identifier: MIT
"""State blobs must satisfy LMCache's token accounting without changing bytes."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec


@pytest.fixture
def lmcache_types(monkeypatch):
    # No real backend/allocator construction, accelerator discovery, or device sync.
    guard = Mock(side_effect=AssertionError("this test must remain CPU-only"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", guard)
    monkeypatch.setattr(torch.cuda, "synchronize", guard)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    pytest.importorskip("lmcache", exc_type=ModuleNotFoundError)
    from lmcache.v1.memory_management import (
        MemoryFormat,
        MemoryObjMetadata,
        TensorMemoryObj,
    )
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

    yield SimpleNamespace(
        fmt=MemoryFormat.KV_2LTD,
        metadata=MemoryObjMetadata,
        memory_obj=TensorMemoryObj,
        backend=LocalCPUBackend,
    )
    guard.assert_not_called()


@pytest.fixture
def allocation(lmcache_types):
    allocator = SimpleNamespace(close=Mock(), free=Mock())
    objects = []

    def allocate(shape, dtype, *, fmt, busy_loop):
        raw = torch.empty(shape.numel(), dtype=dtype, device="cpu")
        meta = lmcache_types.metadata(
            shape=shape,
            dtype=dtype,
            address=0,
            phy_size=raw.numel() * raw.element_size(),
            ref_count=1,
            fmt=fmt,
        )
        obj = lmcache_types.memory_obj(raw, meta, allocator)
        allocator.free.side_effect = lambda obj: obj.invalidate()
        objects.append(obj)
        return obj

    codec = object.__new__(StateByteCodec)
    codec._storage = SimpleNamespace(allocate=Mock(side_effect=allocate))
    yield codec, allocator
    # Cleanup must also work on the red test, without invoking real allocator.close.
    for obj in objects:
        while obj.get_ref_count():
            obj.ref_count_down()


@pytest.mark.parametrize("nbytes", [1, 257, 5_276_672, 5_971_968])
def test_snapshot_token_accounting_and_flat_bytes(lmcache_types, allocation, nbytes):
    codec, _ = allocation
    obj = codec._allocate(nbytes)

    # Exercise the real get_num_tokens (the old one-dimensional shape raises).
    assert obj.get_num_tokens() == 1
    assert obj.get_shape() == torch.Size([1, 1, 1, nbytes])
    assert obj.get_dtype() == torch.uint8
    assert obj.get_memory_format() == lmcache_types.fmt
    assert obj.get_size() == obj.get_physical_size() == nbytes
    codec._storage.allocate.assert_called_once_with(
        torch.Size([1, 1, 1, nbytes]),
        torch.uint8,
        fmt=lmcache_types.fmt,
        busy_loop=False,
    )

    payload = torch.arange(nbytes, dtype=torch.int64, device="cpu").to(torch.uint8)
    flat = obj.tensor.view(torch.uint8).reshape(-1)
    flat.copy_(payload)
    assert flat.device.type == "cpu"
    assert flat.shape == torch.Size([nbytes])
    assert flat.data_ptr() == obj.raw_data.data_ptr()
    assert torch.equal(obj.raw_data, payload)
    assert obj.byte_array.tobytes() == payload.numpy().tobytes()
    obj.raw_data[-1] = 123
    assert flat[-1].item() == 123


@pytest.mark.parametrize("operation", ["clear", "close"])
def test_real_backend_cleanup_counts_snapshots(lmcache_types, allocation, operation):
    codec, allocator = allocation
    objects = [codec._allocate(17), codec._allocate(513)]
    backend = object.__new__(lmcache_types.backend)
    backend.hot_cache = dict(enumerate(objects))
    backend.cpu_lock = threading.Lock()
    backend.use_hot = True
    backend.batched_msg_sender = None
    backend.cache_policy = SimpleNamespace(update_on_force_evict=Mock())
    backend.memory_allocator = allocator

    # Real clear/close -> batched_remove -> remove -> ref_count_down.
    result = getattr(backend, operation)()

    assert result == (2 if operation == "clear" else None)
    assert not backend.hot_cache
    assert all(obj.get_ref_count() == 0 for obj in objects)
    assert allocator.free.call_count == 2
    for obj in objects:
        allocator.free.assert_any_call(obj)
    if operation == "close":
        allocator.close.assert_called_once_with()
    else:
        allocator.close.assert_not_called()
    assert backend.clear() == 0
    assert allocator.free.call_count == 2


def test_full_pool_returns_none_without_enabling_busy_loop(lmcache_types):
    codec = object.__new__(StateByteCodec)
    codec._storage = SimpleNamespace(allocate=Mock(return_value=None))

    assert codec._allocate(257) is None
    codec._storage.allocate.assert_called_once()
    assert codec._storage.allocate.call_args.args[1] == torch.uint8
    assert codec._storage.allocate.call_args.kwargs == {
        "fmt": lmcache_types.fmt,
        "busy_loop": False,
    }
