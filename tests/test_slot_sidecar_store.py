# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from atom.kv_transfer.offload.hybrid.sidecar_format import (  # noqa: E402
    SlotSidecarHeader,
    SlotSidecarKey,
    encode_sidecar,
)
from atom.kv_transfer.offload.hybrid.store import (  # noqa: E402
    SlotSidecarCorruptionError,
    SlotSidecarStore,
)

_FINGERPRINT = bytes.fromhex("00112233445566778899aabbccddeeff")
_UNSET = object()


class _MemoryObj:
    def __init__(
        self,
        tensor: object,
        *,
        clear_on_decref: bool = False,
        tensor_via_getter: bool = False,
    ) -> None:
        self.backing_tensor = tensor
        self.tensor = None if tensor_via_getter else tensor
        self.clear_on_decref = clear_on_decref
        self.decref_count = 0
        self.get_tensor_calls: list[int] = []

    def get_tensor(self, index: int):
        self.get_tensor_calls.append(index)
        assert index == 0
        return self.backing_tensor

    def get_num_tokens(self) -> int:
        if not isinstance(self.backing_tensor, torch.Tensor):
            raise RuntimeError("MemoryObj has no backing tensor")
        return int(self.backing_tensor.shape[2])

    def ref_count_down(self) -> None:
        self.decref_count += 1
        if self.clear_on_decref and isinstance(self.backing_tensor, torch.Tensor):
            self.backing_tensor.zero_()


class _StorageManager:
    """Small behavioral fake matching the StorageManager methods used here."""

    def __init__(self) -> None:
        self.allocate_result = _UNSET
        self.allocate_error: Exception | None = None
        self.put_error: Exception | None = None
        self.get_result: _MemoryObj | None = None
        self.get_error: Exception | None = None
        self.contains_result: str | None = "LocalCPUBackend"
        self.contains_error: Exception | None = None
        self.remove_result = 1
        self.remove_error: Exception | None = None

        self.allocate_calls: list[tuple[torch.Size, torch.dtype, object, bool]] = []
        self.allocated_objects: list[_MemoryObj] = []
        self.batched_put_calls: list[
            tuple[list[object], list[_MemoryObj], object, str | None]
        ] = []
        self.clear_num_tokens: list[int] = []
        self.get_calls: list[tuple[object, str | None]] = []
        self.contains_calls: list[tuple[object, list[str] | None, bool]] = []
        self.remove_calls: list[tuple[object, list[str] | None]] = []

    def allocate(
        self,
        shapes: torch.Size,
        dtypes: torch.dtype,
        fmt: object,
        busy_loop: bool = True,
    ) -> _MemoryObj | None:
        self.allocate_calls.append((shapes, dtypes, fmt, busy_loop))
        if self.allocate_error is not None:
            raise self.allocate_error
        if self.allocate_result is _UNSET:
            memory_obj = _MemoryObj(torch.empty(shapes, dtype=dtypes))
        else:
            memory_obj = self.allocate_result
        if memory_obj is not None:
            self.allocated_objects.append(memory_obj)
        return memory_obj

    def batched_put(
        self,
        keys: list[object],
        memory_objs: list[_MemoryObj],
        transfer_spec=None,
        location: str | None = None,
    ) -> None:
        self.batched_put_calls.append(
            (list(keys), list(memory_objs), transfer_spec, location)
        )
        if self.put_error is not None:
            raise self.put_error

        # Real StorageManager.batched_put owns and decrements each submitted
        # object before returning. Simulate LocalCPUBackend.clear consulting
        # KV_2LTD's token dimension before release.
        for memory_obj in memory_objs:
            self.clear_num_tokens.append(memory_obj.get_num_tokens())
            memory_obj.ref_count_down()

    def get(self, key: object, location: str | None = None) -> _MemoryObj | None:
        self.get_calls.append((key, location))
        if self.get_error is not None:
            raise self.get_error
        return self.get_result

    def contains(
        self,
        key: object,
        search_range: list[str] | None = None,
        pin: bool = False,
    ) -> str | None:
        self.contains_calls.append((key, search_range, pin))
        if self.contains_error is not None:
            raise self.contains_error
        if search_range is not None and self.contains_result not in search_range:
            return None
        return self.contains_result

    def remove(self, key: object, locations: list[str] | None = None) -> int:
        self.remove_calls.append((key, locations))
        if self.remove_error is not None:
            raise self.remove_error
        return self.remove_result


@pytest.fixture(autouse=True)
def fake_lmcache(monkeypatch):
    @dataclass(frozen=True)
    class CacheEngineKey:
        model_name: str
        world_size: int
        worker_id: int
        chunk_hash: int
        dtype: torch.dtype

    kv_2ltd = object()

    utils_module = types.ModuleType("lmcache.utils")
    utils_module.CacheEngineKey = CacheEngineKey
    memory_management_module = types.ModuleType("lmcache.v1.memory_management")
    memory_management_module.MemoryFormat = SimpleNamespace(KV_2LTD=kv_2ltd)

    v1_module = types.ModuleType("lmcache.v1")
    v1_module.__path__ = []
    v1_module.memory_management = memory_management_module
    lmcache_module = types.ModuleType("lmcache")
    lmcache_module.__path__ = []
    lmcache_module.utils = utils_module
    lmcache_module.v1 = v1_module

    for name, module in (
        ("lmcache", lmcache_module),
        ("lmcache.utils", utils_module),
        ("lmcache.v1", v1_module),
        ("lmcache.v1.memory_management", memory_management_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(CacheEngineKey=CacheEngineKey, KV_2LTD=kv_2ltd)


def _key(*, tp_rank: int = 1) -> SlotSidecarKey:
    return SlotSidecarKey(
        boundary_block_hash=0x1234ABCD,
        fingerprint=_FINGERPRINT,
        tp_size=4,
        tp_rank=tp_rank,
    )


def _aos1_blob(*, tp_rank: int = 1) -> bytes:
    return encode_sidecar(
        SlotSidecarHeader(
            boundary_tokens=512,
            boundary_block_hash=0x1234ABCD,
            payload_bytes=None,
            payload_crc32=None,
            fingerprint=_FINGERPRINT,
            tp_size=4,
            tp_rank=tp_rank,
        ),
        b"\x07\x08\x09\xff",
    )


def _store(
    storage_manager: _StorageManager,
    *,
    model_name: str = "org/model",
    world_size: int = 4,
    worker_id: int = 1,
    store_location: str | None = None,
    retrieve_locations: list[str] | None = None,
) -> SlotSidecarStore:
    return SlotSidecarStore(
        SimpleNamespace(
            storage_manager=storage_manager,
            store_location=store_location,
            retrieve_locations=retrieve_locations,
        ),
        model_name=model_name,
        world_size=world_size,
        worker_id=worker_id,
    )


def _assert_cache_key(
    cache_key,
    fake_lmcache,
    sidecar_key: SlotSidecarKey,
    *,
    worker_id: int = 1,
) -> None:
    assert isinstance(cache_key, fake_lmcache.CacheEngineKey)
    assert cache_key.model_name == "org/model"
    assert cache_key.world_size == 4
    assert cache_key.worker_id == worker_id
    assert cache_key.chunk_hash == sidecar_key.storage_hash()
    assert cache_key.dtype is torch.uint8


def test_requires_non_none_engine_storage_manager():
    with pytest.raises(ValueError, match="engine.storage_manager"):
        SlotSidecarStore(
            SimpleNamespace(storage_manager=None),
            model_name="org/model",
            world_size=4,
            worker_id=1,
        )

    with pytest.raises(ValueError, match="engine.storage_manager"):
        SlotSidecarStore(
            SimpleNamespace(),
            model_name="org/model",
            world_size=4,
            worker_id=1,
        )


def test_put_allocates_copies_and_submits_exact_aos1_object(fake_lmcache):
    manager = _StorageManager()
    store = _store(manager)
    sidecar_key = _key()
    blob = _aos1_blob()

    assert store.put(sidecar_key, blob) is True

    assert manager.allocate_calls == [
        (
            torch.Size([1, 1, len(blob)]),
            torch.uint8,
            fake_lmcache.KV_2LTD,
            False,
        )
    ]
    assert len(manager.allocated_objects) == 1
    memory_obj = manager.allocated_objects[0]
    assert tuple(memory_obj.tensor.shape) == (1, 1, len(blob))
    assert memory_obj.tensor.dtype is torch.uint8
    assert torch.equal(
        memory_obj.tensor.reshape(-1),
        torch.tensor(list(blob), dtype=torch.uint8),
    )
    assert memory_obj.get_num_tokens() == len(blob)
    assert manager.clear_num_tokens == [len(blob)]

    assert len(manager.batched_put_calls) == 1
    cache_keys, memory_objs, transfer_spec, location = manager.batched_put_calls[0]
    assert memory_objs == [memory_obj]
    assert transfer_spec is None
    assert location is None
    assert len(cache_keys) == 1
    _assert_cache_key(cache_keys[0], fake_lmcache, sidecar_key)

    # The fake manager performs StorageManager's successful-handoff decrement.
    # Exactly one decrement proves the store did not also release the object.
    assert memory_obj.decref_count == 1


def test_put_accepts_contiguous_cpu_uint8_tensor_and_flattens_allocation():
    manager = _StorageManager()
    store = _store(manager)
    blob = torch.arange(12, dtype=torch.uint8).reshape(3, 4)

    assert store.put(_key(), blob) is True

    assert manager.allocate_calls[0][0] == torch.Size([1, 1, 12])
    assert manager.allocate_calls[0][3] is False
    assert tuple(manager.allocated_objects[0].tensor.shape) == (1, 1, 12)
    assert torch.equal(
        manager.allocated_objects[0].tensor.reshape(-1),
        blob.reshape(-1),
    )


def test_put_uses_get_tensor_zero_when_allocated_tensor_property_is_none():
    manager = _StorageManager()
    blob = _aos1_blob()
    memory_obj = _MemoryObj(
        torch.empty((1, 1, len(blob)), dtype=torch.uint8),
        tensor_via_getter=True,
    )
    manager.allocate_result = memory_obj
    store = _store(manager)

    assert store.put(_key(), blob) is True

    assert memory_obj.get_tensor_calls == [0]
    assert torch.equal(
        memory_obj.backing_tensor.reshape(-1),
        torch.tensor(list(blob), dtype=torch.uint8),
    )
    assert memory_obj.get_num_tokens() == len(blob)
    assert memory_obj.decref_count == 1


@pytest.mark.parametrize(
    "blob",
    [
        pytest.param(bytearray(b"AOS1-bytes"), id="bytearray"),
        pytest.param(memoryview(b"AOS1-view"), id="memoryview"),
    ],
)
def test_put_accepts_other_contiguous_bytes_like_objects(blob):
    manager = _StorageManager()
    store = _store(manager)
    expected = torch.tensor(list(memoryview(blob).cast("B")), dtype=torch.uint8)

    assert store.put(_key(), blob) is True

    assert torch.equal(manager.allocated_objects[0].tensor.reshape(-1), expected)


def test_put_returns_false_when_allocation_returns_none():
    manager = _StorageManager()
    manager.allocate_result = None
    store = _store(manager)

    assert store.put(_key(), _aos1_blob()) is False

    assert len(manager.allocate_calls) == 1
    assert manager.batched_put_calls == []


def test_put_catches_allocation_exception():
    manager = _StorageManager()
    manager.allocate_error = RuntimeError("allocator unavailable")
    store = _store(manager)

    assert store.put(_key(), _aos1_blob()) is False

    assert manager.allocated_objects == []
    assert manager.batched_put_calls == []


def test_put_exception_releases_pre_handoff_object_once():
    manager = _StorageManager()
    manager.put_error = RuntimeError("submission failed")
    store = _store(manager)

    assert store.put(_key(), _aos1_blob()) is False

    memory_obj = manager.allocated_objects[0]
    assert len(manager.batched_put_calls) == 1
    assert memory_obj.decref_count == 1


def test_put_copy_failure_releases_pre_handoff_object_once():
    manager = _StorageManager()
    memory_obj = _MemoryObj(torch.empty((1, 1, 1), dtype=torch.uint8))
    manager.allocate_result = memory_obj
    store = _store(manager)

    assert store.put(_key(), _aos1_blob()) is False

    assert manager.batched_put_calls == []
    assert memory_obj.decref_count == 1


def test_put_fails_closed_and_releases_when_get_tensor_is_invalid():
    manager = _StorageManager()
    memory_obj = _MemoryObj(object(), tensor_via_getter=True)
    manager.allocate_result = memory_obj
    store = _store(manager)

    assert store.put(_key(), _aos1_blob()) is False

    assert memory_obj.get_tensor_calls == [0]
    assert manager.batched_put_calls == []
    assert memory_obj.decref_count == 1


def test_get_returns_flat_clone_and_always_releases_fetched_object(fake_lmcache):
    manager = _StorageManager()
    expected = torch.arange(12, dtype=torch.uint8)
    fetched = _MemoryObj(expected.reshape(1, 1, 12).clone(), clear_on_decref=True)
    manager.get_result = fetched
    store = _store(manager)
    sidecar_key = _key()

    result = store.get(sidecar_key)

    assert result is not None
    assert tuple(result.shape) == (12,)
    assert result.dtype is torch.uint8
    assert result.device.type == "cpu"
    assert torch.equal(result, expected)
    assert result.data_ptr() != fetched.tensor.data_ptr()
    assert torch.count_nonzero(fetched.tensor) == 0
    assert fetched.decref_count == 1
    assert len(manager.get_calls) == 1
    _assert_cache_key(manager.get_calls[0][0], fake_lmcache, sidecar_key)


def test_borrow_exposes_storage_view_until_consumer_finishes_then_releases():
    manager = _StorageManager()
    expected = torch.arange(12, dtype=torch.uint8)
    fetched = _MemoryObj(expected.reshape(1, 1, 12).clone(), clear_on_decref=True)
    manager.get_result = fetched
    store = _store(manager)

    with store.borrow(_key()) as result:
        assert result is not None
        assert result.data_ptr() == fetched.tensor.data_ptr()
        assert fetched.decref_count == 0
        assert torch.equal(result, expected)

    assert fetched.decref_count == 1
    assert torch.count_nonzero(fetched.tensor) == 0


def test_get_uses_get_tensor_zero_and_clones_before_release():
    manager = _StorageManager()
    expected = torch.arange(12, dtype=torch.uint8)
    fetched = _MemoryObj(
        expected.reshape(1, 1, 12).clone(),
        clear_on_decref=True,
        tensor_via_getter=True,
    )
    manager.get_result = fetched
    store = _store(manager)

    result = store.get(_key())

    assert result is not None
    assert fetched.get_tensor_calls == [0]
    assert tuple(result.shape) == (12,)
    assert torch.equal(result, expected)
    assert torch.count_nonzero(fetched.backing_tensor) == 0
    assert fetched.decref_count == 1


def test_get_returns_none_on_miss():
    manager = _StorageManager()
    store = _store(manager)

    assert store.get(_key()) is None

    assert len(manager.get_calls) == 1


def test_get_catches_storage_exception():
    manager = _StorageManager()
    manager.get_error = RuntimeError("disk read failed")
    store = _store(manager)

    assert store.get(_key()) is None


def test_get_raises_corruption_and_releases_when_tensor_is_unavailable():
    manager = _StorageManager()
    fetched = _MemoryObj(None, tensor_via_getter=True)
    manager.get_result = fetched
    store = _store(manager)

    with pytest.raises(SlotSidecarCorruptionError, match="expose a tensor"):
        store.get(_key())

    assert fetched.get_tensor_calls == [0]
    assert fetched.decref_count == 1


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param(torch.ones(4, dtype=torch.int16), id="wrong-dtype"),
        pytest.param(torch.empty(0, dtype=torch.uint8), id="empty"),
        pytest.param(
            torch.arange(6, dtype=torch.uint8).reshape(2, 3).t(),
            id="noncontiguous",
        ),
    ],
)
def test_borrow_distinguishes_malformed_objects_from_storage_io(malformed):
    manager = _StorageManager()
    fetched = _MemoryObj(malformed)
    manager.get_result = fetched
    store = _store(manager)

    with pytest.raises(SlotSidecarCorruptionError):
        with store.borrow(_key()):
            pytest.fail("malformed object must not be yielded")

    assert fetched.decref_count == 1


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        pytest.param("LocalCPUBackend", True, id="location"),
        pytest.param(None, False, id="miss"),
    ],
)
def test_contains_uses_non_none_storage_location(
    fake_lmcache,
    location,
    expected,
):
    manager = _StorageManager()
    manager.contains_result = location
    store = _store(manager)
    sidecar_key = _key()

    assert store.contains(sidecar_key) is expected

    assert len(manager.contains_calls) == 1
    _assert_cache_key(manager.contains_calls[0][0], fake_lmcache, sidecar_key)


def test_contains_raises_sanitized_storage_exception():
    manager = _StorageManager()
    manager.contains_error = RuntimeError("private-key=full-storage-key")
    store = _store(manager)

    with pytest.raises(
        RuntimeError,
        match="LMCache SLOT sidecar visibility probe failed",
    ) as exc_info:
        store.contains(_key())
    assert "private-key" not in str(exc_info.value)


def test_invalidate_removes_all_tier_copies_for_republication(fake_lmcache):
    manager = _StorageManager()
    store = _store(manager)
    sidecar_key = _key()

    assert store.invalidate(sidecar_key) is True

    assert len(manager.remove_calls) == 1
    cache_key, locations = manager.remove_calls[0]
    _assert_cache_key(cache_key, fake_lmcache, sidecar_key)
    assert locations is None


def test_failed_invalidation_fences_put_until_corrupt_copy_is_removed():
    manager = _StorageManager()
    manager.remove_result = 0
    store = _store(manager)
    sidecar_key = _key()

    assert store.invalidate(sidecar_key) is False
    assert store.put(sidecar_key, _aos1_blob()) is False

    # put retries the unresolved eviction, but must not allocate or submit
    # replacement bytes while LMCache may still retain the corrupt key.
    assert len(manager.remove_calls) == 2
    assert manager.allocate_calls == []
    assert manager.batched_put_calls == []

    manager.remove_result = 1
    assert store.put(sidecar_key, _aos1_blob()) is True
    assert len(manager.remove_calls) == 3
    assert len(manager.allocate_calls) == 1
    assert len(manager.batched_put_calls) == 1


@pytest.mark.parametrize(
    ("location", "retrieve_locations"),
    [
        ("LocalCPUBackend", ["LocalCPUBackend"]),
        ("LocalDiskBackend", ["LocalDiskBackend"]),
    ],
)
def test_location_policy_applies_to_put_contains_and_get(
    location,
    retrieve_locations,
):
    manager = _StorageManager()
    manager.contains_result = location
    manager.get_result = _MemoryObj(torch.arange(4, dtype=torch.uint8).reshape(1, 1, 4))
    store = _store(
        manager,
        store_location=location,
        retrieve_locations=retrieve_locations,
    )

    assert store.put(_key(), _aos1_blob())
    assert store.contains(_key())
    assert store.get(_key()) is not None

    assert manager.batched_put_calls[0][3] == location
    assert manager.contains_calls[-1][1:] == (retrieve_locations, False)
    assert manager.get_calls == [(manager.contains_calls[-1][0], location)]


def test_get_uses_selected_location_from_multiple_retrieve_locations():
    manager = _StorageManager()
    manager.contains_result = "LocalDiskBackend"
    manager.get_result = _MemoryObj(torch.arange(4, dtype=torch.uint8).reshape(1, 1, 4))
    retrieve_locations = ["LocalCPUBackend", "LocalDiskBackend"]
    store = _store(manager, retrieve_locations=retrieve_locations)

    assert store.get(_key()) is not None

    assert manager.contains_calls[0][1] == retrieve_locations
    assert manager.get_calls[0][1] == "LocalDiskBackend"


def test_get_rejects_allocator_cpu_hit_outside_disk_only_policy():
    manager = _StorageManager()
    manager.contains_result = "LocalCPUBackend"
    manager.get_result = _MemoryObj(torch.arange(4, dtype=torch.uint8).reshape(1, 1, 4))
    store = _store(
        manager,
        store_location="LocalDiskBackend",
        retrieve_locations=["LocalDiskBackend"],
    )

    assert store.contains(_key()) is False
    assert store.get(_key()) is None

    assert manager.contains_calls == [
        (manager.contains_calls[0][0], ["LocalDiskBackend"], False),
        (manager.contains_calls[1][0], ["LocalDiskBackend"], False),
    ]
    assert manager.get_calls == []


@pytest.mark.parametrize("method_name", ["put", "get", "contains"])
def test_operations_reject_non_sidecar_keys_before_storage(method_name):
    manager = _StorageManager()
    store = _store(manager)

    with pytest.raises(TypeError, match="SlotSidecarKey"):
        if method_name == "put":
            store.put(object(), b"AOS1")
        else:
            getattr(store, method_name)(object())

    assert manager.allocate_calls == []
    assert manager.get_calls == []
    assert manager.contains_calls == []


@pytest.mark.parametrize(
    ("blob_factory", "error_type", "message"),
    [
        pytest.param(lambda: b"", ValueError, "nonempty", id="empty-bytes"),
        pytest.param(lambda: "AOS1", TypeError, "bytes-like", id="string"),
        pytest.param(
            lambda: torch.empty(0, dtype=torch.uint8),
            ValueError,
            "nonempty",
            id="empty-tensor",
        ),
        pytest.param(
            lambda: torch.ones(4, dtype=torch.int16),
            ValueError,
            "uint8",
            id="wrong-dtype",
        ),
        pytest.param(
            lambda: torch.arange(6, dtype=torch.uint8).reshape(2, 3).t(),
            ValueError,
            "contiguous",
            id="noncontiguous-tensor",
        ),
        pytest.param(
            lambda: torch.empty(4, dtype=torch.uint8, device="meta"),
            ValueError,
            "CPU",
            id="non-cpu-tensor",
        ),
        pytest.param(
            lambda: memoryview(bytearray(b"abcd"))[::2],
            ValueError,
            "contiguous",
            id="noncontiguous-buffer",
        ),
    ],
)
def test_put_rejects_invalid_blobs_before_allocation(
    blob_factory,
    error_type,
    message,
):
    manager = _StorageManager()
    store = _store(manager)

    with pytest.raises(error_type, match=message):
        store.put(_key(), blob_factory())

    assert manager.allocate_calls == []
    assert manager.batched_put_calls == []


def test_cache_keys_are_deterministic_and_worker_rank_separated():
    manager = _StorageManager()
    manager.contains_result = "LocalDiskBackend"
    store0 = _store(manager, worker_id=0)
    store1 = _store(manager, worker_id=1)
    sidecar_key = _key()

    assert store0.contains(sidecar_key) is True
    assert store0.contains(sidecar_key) is True
    assert store1.contains(sidecar_key) is True

    first_rank0, second_rank0, rank1 = manager.contains_calls
    first_rank0 = first_rank0[0]
    second_rank0 = second_rank0[0]
    rank1 = rank1[0]
    assert first_rank0 == second_rank0
    assert first_rank0 != rank1
    assert first_rank0.chunk_hash == rank1.chunk_hash == sidecar_key.storage_hash()
    assert first_rank0.worker_id == 0
    assert rank1.worker_id == 1
