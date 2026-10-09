# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The startup check of the in-process LMCache offload's remote tier.

LMCache's backends are faked: a remote that keeps chunks in a dict, with
switches for each way the real one loses them without an error.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "arange"):
    pytest.skip("real torch is unavailable", allow_module_level=True)

from atom.kv_transfer.offload import remote_check

CHUNK_BYTES = 4096


@dataclass(frozen=True)
class _Key:
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    dtype: object


class _Chunk:
    def __init__(self, data: torch.Tensor):
        self.tensor = data
        self.refs = 1

    def ref_count_down(self):
        self.refs -= 1


class _Local:
    def __init__(self):
        self.allocated = []

    def allocate(self, shapes, dtypes, fmt=None):
        # A pinned host pool, whatever torch's default device.
        chunk = _Chunk(torch.zeros(shapes[0], dtype=torch.uint8, device="cpu"))
        self.allocated.append(chunk)
        return chunk


class _Remote:
    def __init__(self, *, drop_puts=False, corrupt=False, fail_get=False, hang=False):
        self.objects = {}
        self.removed = []
        self.returned = []
        self.drop_puts = drop_puts
        self.corrupt = corrupt
        self.fail_get = fail_get
        self.hang = hang
        self.remote_url = "mooncakestore://fake/"
        self.connection = SimpleNamespace(
            getWrappedConnector=lambda: SimpleNamespace(registered_buffer_ptr=1),
            remove_sync=lambda key: self.removed.append(key),
        )

    def batched_submit_put_task(self, keys, objs, on_complete_callback=None):
        if self.hang:
            return
        for key, obj in zip(keys, objs):
            if not self.drop_puts:
                self.objects[key] = obj.tensor.clone()
            on_complete_callback(key)

    def batched_contains(self, keys):
        hits = 0
        for key in keys:
            if key not in self.objects:
                break
            hits += 1
        return hits

    def batched_get_blocking(self, keys):
        if self.fail_get:
            return [None] * len(keys)
        chunks = []
        for key in keys:
            data = self.objects[key].clone()
            if self.corrupt:
                data[17] ^= 1
            chunks.append(_Chunk(data))
        self.returned.extend(chunks)
        return chunks


@pytest.fixture
def fake_lmcache(monkeypatch):
    utils = types.ModuleType("lmcache.utils")
    utils.CacheEngineKey = _Key
    memory = types.ModuleType("lmcache.v1.memory_management")
    memory.MemoryFormat = SimpleNamespace(KV_2LTD="KV_2LTD")
    for name in ("lmcache", "lmcache.v1"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "lmcache.utils", utils)
    monkeypatch.setitem(sys.modules, "lmcache.v1.memory_management", memory)


METADATA = SimpleNamespace(
    model_name="glm::atom-page-v3-abc",
    world_size=4,
    worker_id=2,
    kv_dtype="fp8",
    get_shapes=lambda: [torch.Size([CHUNK_BYTES])],
    get_dtypes=lambda: [torch.uint8],
)


def _engine(remote=None, local=None, backends=None):
    if backends is None:
        backends = {"LocalCPUBackend": local or _Local(), "RemoteBackend": remote}
    return SimpleNamespace(storage_manager=SimpleNamespace(storage_backends=backends))


def test_round_trip_passes_and_balances_every_reference(fake_lmcache):
    remote, local = _Remote(), _Local()
    remote_check.verify_remote_backend(_engine(remote, local), METADATA)
    (source,) = local.allocated
    (returned,) = remote.returned
    assert source.refs == 0 and returned.refs == 0
    (key,) = remote.removed
    assert key.model_name == "glm::atom-page-v3-abc::atom-remote-check"
    assert (key.world_size, key.worker_id, key.dtype) == (4, 2, "fp8")
    # The payload is not all zeros, so an untouched buffer cannot pass.
    assert remote.objects[key].count_nonzero() > CHUNK_BYTES // 2


def test_round_trip_compares_on_the_cpu_whatever_the_default_device(fake_lmcache):
    # The caller may run under a non-CPU default device; the chunks are host
    # memory either way.
    remote = _Remote()
    with torch.device("meta"):
        remote_check.verify_remote_backend(_engine(remote, _Local()), METADATA)
    (stored,) = remote.objects.values()
    assert stored.device.type == "cpu"
    assert stored.count_nonzero() > CHUNK_BYTES // 2


@pytest.mark.parametrize(
    "remote, message",
    [
        (_Remote(drop_puts=True), "does not hold the chunk"),
        (_Remote(corrupt=True), "different bytes"),
        (_Remote(fail_get=True), "get failed"),
    ],
)
def test_lost_or_altered_chunks_fail_startup(fake_lmcache, remote, message):
    local = _Local()
    with pytest.raises(RuntimeError, match=message):
        remote_check.verify_remote_backend(_engine(remote, local), METADATA)
    assert all(chunk.refs == 0 for chunk in local.allocated + remote.returned)


def test_a_put_that_never_finishes_fails_startup(fake_lmcache):
    local = _Local()
    with pytest.raises(RuntimeError, match="did not finish in 0.05 s"):
        remote_check.verify_remote_backend(
            _engine(_Remote(hang=True), local), METADATA, put_timeout_s=0.05
        )
    assert local.allocated[0].refs == 0


def test_missing_or_unconnected_remote_fails_startup(fake_lmcache):
    with pytest.raises(RuntimeError, match="no RemoteBackend .*LocalCPUBackend"):
        remote_check.verify_remote_backend(
            _engine(backends={"LocalCPUBackend": _Local()}), METADATA
        )
    with pytest.raises(RuntimeError, match="backends: none"):
        remote_check.verify_remote_backend(SimpleNamespace(), METADATA)
    remote = _Remote()
    remote.connection = None
    with pytest.raises(RuntimeError, match="did not connect"):
        remote_check.verify_remote_backend(_engine(remote), METADATA)


def test_unregistered_pool_fails_startup(fake_lmcache):
    remote = _Remote()
    remote.connection = SimpleNamespace(
        getWrappedConnector=lambda: SimpleNamespace(registered_buffer_ptr=None)
    )
    with pytest.raises(RuntimeError, match="could not register the CPU pool"):
        remote_check.verify_remote_backend(_engine(remote), METADATA)


def test_connectors_without_a_registered_buffer_are_not_required_to_have_one(
    fake_lmcache,
):
    remote = _Remote()
    remote.connection = SimpleNamespace(remove_sync=remote.removed.append)
    remote_check.verify_remote_backend(_engine(remote), METADATA)
    assert len(remote.removed) == 1


def test_probe_removal_is_best_effort(fake_lmcache):
    remote = _Remote()

    def refuse(key):
        raise NotImplementedError

    remote.connection = SimpleNamespace(remove_sync=refuse)
    remote_check.verify_remote_backend(_engine(remote), METADATA)
    assert len(remote.objects) == 1
