# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the Kimi-K3 state object codec.

No aiter and no GPU: an ``importorskip`` on either never fires on the CPU CI
runner, which is how a bug in this arithmetic shipped once already.
"""

from __future__ import annotations

import struct
import sys
import types
import zlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import (
    HEADER_BYTES,
    MAGIC,
    StateByteCodec,
)

ENTRY_BYTES = 256
OBJECT_BYTES = HEADER_BYTES + ENTRY_BYTES


@pytest.fixture(autouse=True)
def fake_lmcache(monkeypatch):
    @dataclass(frozen=True)
    class CacheEngineKey:
        model_name: str
        world_size: int
        worker_id: int
        chunk_hash: int
        dtype: object

    kv_2ltd = object()
    utils_module = types.ModuleType("lmcache.utils")
    utils_module.CacheEngineKey = CacheEngineKey
    memory_module = types.ModuleType("lmcache.v1.memory_management")
    memory_module.MemoryFormat = SimpleNamespace(KV_2LTD=kv_2ltd)
    v1_module = types.ModuleType("lmcache.v1")
    v1_module.__path__ = []
    v1_module.memory_management = memory_module
    lmcache_module = types.ModuleType("lmcache")
    lmcache_module.__path__ = []
    lmcache_module.utils = utils_module
    lmcache_module.v1 = v1_module

    for name, module in (
        ("lmcache", lmcache_module),
        ("lmcache.utils", utils_module),
        ("lmcache.v1", v1_module),
        ("lmcache.v1.memory_management", memory_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(CacheEngineKey=CacheEngineKey, KV_2LTD=kv_2ltd)


class _MemoryObj:
    """MemoryObj fake: a flat uint8 buffer plus the reference counter."""

    def __init__(self, nbytes: int, *, size_override: object = ...) -> None:
        self.tensor = torch.zeros(int(nbytes), dtype=torch.uint8)
        self.decref_count = 0
        self._size_override = size_override

    def get_size(self):
        if self._size_override is not ...:
            return self._size_override
        return int(self.tensor.numel())

    def ref_count_down(self) -> None:
        self.decref_count += 1


class _Storage:
    def __init__(self) -> None:
        self.allocate_result: object = ...
        self.put_error: Exception | None = None
        self.get_result: _MemoryObj | None = None
        self.batched_put_calls: list[tuple[list, list]] = []
        self.get_calls: list = []

    def allocate(self, shape, dtype, fmt=None, busy_loop=True):
        assert busy_loop is False, "a store must never busy-loop on a full pool"
        if self.allocate_result is not ...:
            return self.allocate_result
        return _MemoryObj(int(shape[0]))

    def batched_put(self, keys, memory_objs):
        self.batched_put_calls.append((list(keys), list(memory_objs)))
        if self.put_error is not None:
            # Raise before the terminal tail loop, i.e. before adoption.
            raise self.put_error
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()

    def get(self, key):
        self.get_calls.append(key)
        return self.get_result


class _Staged:
    """Fills the payload with a byte pattern instead of driving a GPU."""

    def __init__(self, fill: int = 0xA5) -> None:
        self.fill = fill
        self.unpacked: list[torch.Tensor] = []
        self.pack_error: Exception | None = None

    def pack(self, segments, dst) -> None:
        if self.pack_error is not None:
            raise self.pack_error
        dst.fill_(self.fill)

    def unpack(self, src, segments) -> None:
        self.unpacked.append(src.clone())


class _Backend:
    def page_unit_views(self, unit_ids):
        return []

    def state_entry_views(self, slot):
        return []


def _codec(storage=None, staged=None, *, layout_id="gdn-v1-tp8") -> StateByteCodec:
    codec = StateByteCodec(
        _Backend(),
        staged if staged is not None else _Staged(),
        ENTRY_BYTES,
        model_name="kimi-k3",
        world_size=8,
        worker_id=0,
        layout_id=layout_id,
    )
    if storage is not None:
        codec.bind_storage_manager(storage)
    return codec


def _stored_object(
    storage: _Storage, staged: _Staged, h: int = 0xDEADBEEF
) -> _MemoryObj:
    """Run one successful put and hand back the framed object it produced."""
    assert _codec(storage, staged).put(h, [0, 1]) is True
    return storage.batched_put_calls[-1][1][0]


def test_key_folds_layout_id_so_geometries_never_share_a_key():
    left = _codec(layout_id="gdn-v1-tp8").key(12345)
    right = _codec(layout_id="gdn-v2-tp8").key(12345)
    assert left.chunk_hash != right.chunk_hash
    assert left.model_name == right.model_name


def test_key_accepts_hashes_above_the_signed_range():
    # An ATOM block hash spans the full unsigned 64-bit range.
    assert _codec().key((1 << 64) - 1).chunk_hash >= 0


def test_put_frames_the_payload_and_hands_off_exactly_one_reference():
    storage, staged = _Storage(), _Staged()
    obj = _stored_object(storage, staged)
    assert int(obj.tensor.numel()) == OBJECT_BYTES
    head = bytes(obj.tensor[:HEADER_BYTES].numpy())
    magic, version, flags, payload_bytes, crc = struct.unpack_from("<4sIIQI", head)
    assert magic == MAGIC
    assert (version, flags, payload_bytes) == (1, 0, ENTRY_BYTES)
    payload = bytes(obj.tensor[HEADER_BYTES:].numpy())
    assert crc == zlib.crc32(payload) & 0xFFFFFFFF
    assert payload == bytes([0xA5]) * ENTRY_BYTES
    # batched_put owns the reference on success; the codec must not down it too.
    assert obj.decref_count == 1


def test_put_downs_the_reference_once_when_batched_put_raises_before_adopting():
    storage, staged = _Storage(), _Staged()
    obj = _MemoryObj(OBJECT_BYTES)
    storage.allocate_result = obj
    storage.put_error = RuntimeError("scheduler-role guard")
    with pytest.raises(RuntimeError, match="scheduler-role guard"):
        _codec(storage, staged).put(1, [0])
    assert obj.decref_count == 1


def test_put_downs_the_reference_once_when_pack_raises():
    storage = _Storage()
    staged = _Staged()
    staged.pack_error = RuntimeError("gather failed")
    obj = _MemoryObj(OBJECT_BYTES)
    storage.allocate_result = obj
    with pytest.raises(RuntimeError, match="gather failed"):
        _codec(storage, staged).put(1, [0])
    assert obj.decref_count == 1


def test_put_reports_refusal_without_touching_a_reference():
    storage = _Storage()
    storage.allocate_result = None
    assert _codec(storage, _Staged()).put(1, [0]) is False
    assert storage.batched_put_calls == []


def test_get_round_trips_a_freshly_stored_object():
    storage, staged = _Storage(), _Staged()
    obj = _stored_object(storage, staged)
    storage.get_result = obj
    assert _codec(storage, staged).get(0xDEADBEEF, slot=3) is True
    assert len(staged.unpacked) == 1
    assert int(staged.unpacked[0].numel()) == ENTRY_BYTES
    assert obj.decref_count == 2  # one for the put hand-off, one for this hit


def test_get_treats_a_size_mismatch_as_a_miss_and_counts_it():
    storage, staged = _Storage(), _Staged()
    obj = _MemoryObj(OBJECT_BYTES + 64)
    storage.get_result = obj
    codec = _codec(storage, staged)
    assert codec.get(7, slot=0) is False
    assert codec.misfit_reads == 1
    assert staged.unpacked == []  # never unpack another entry's bytes
    assert obj.decref_count == 1


def test_get_treats_an_unmeasurable_object_as_a_miss():
    storage, staged = _Storage(), _Staged()
    obj = _MemoryObj(OBJECT_BYTES, size_override=None)
    del obj.tensor  # no size accessor and no tensor fallback
    storage.get_result = obj
    codec = _codec(storage, staged)
    assert codec.get(7, slot=0) is False
    assert codec.misfit_reads == 1
    assert obj.decref_count == 1


def test_get_treats_a_corrupt_payload_as_a_miss_and_counts_it():
    storage, staged = _Storage(), _Staged()
    obj = _stored_object(storage, staged)
    obj.tensor[HEADER_BYTES + 5] ^= 0xFF  # a torn write, right size
    storage.get_result = obj
    codec = _codec(storage, staged)
    assert codec.get(0xDEADBEEF, slot=0) is False
    assert codec.corrupt_reads == 1
    assert codec.misfit_reads == 0
    assert staged.unpacked == []
    assert obj.decref_count == 2


@pytest.mark.parametrize(
    ("offset", "value"),
    [
        (0, 0x00),  # magic
        (4, 0x02),  # layout version
        (8, 0x01),  # flags
        (12, 0x01),  # payload_bytes
        (24, 0x01),  # reserved must stay zero
    ],
)
def test_get_rejects_every_corrupted_header_field(offset, value):
    storage, staged = _Storage(), _Staged()
    obj = _stored_object(storage, staged)
    obj.tensor[offset] = value
    storage.get_result = obj
    codec = _codec(storage, staged)
    assert codec.get(0xDEADBEEF, slot=0) is False
    assert codec.corrupt_reads == 1
    assert staged.unpacked == []


def test_get_returns_a_miss_before_the_storage_manager_is_bound():
    assert _codec().get(1, slot=0) is False


def test_put_returns_false_before_the_storage_manager_is_bound():
    assert _codec().put(1, [0]) is False


def test_construction_rejects_a_missing_layout_id():
    with pytest.raises(ValueError, match="layout id"):
        _codec(layout_id="")
