# SPDX-License-Identifier: MIT
# The state object: one key per checkpoint, ATOM's own hash, no chunking.

import pytest
import torch

from atom.kv_transfer.offload.state_object import StateByteCodec


class FakeStaged:
    """Records what the packer was asked to move; no device involved."""

    def __init__(self):
        self.packed = []
        self.unpacked = []

    def pack(self, segments, dst):
        self.packed.append((list(segments), dst))

    def unpack(self, src, segments):
        self.unpacked.append((src, list(segments)))


class FakeBackend:
    def __init__(self):
        self.views = {g: [f"view-{g}-{i}" for i in range(3)] for g in range(4)}

    def state_entry_views(self, group):
        return self.views[group]


class FakeMemoryObj:
    def __init__(self, shapes, dtypes, fmt):
        self.shapes = shapes
        self.dtypes = dtypes
        self.fmt = fmt


class FakeStorageManager:
    """Mirrors the pinned LMCache StorageManager surface.

    `contains` returns an Optional[str] location, not a bool, and `allocate`
    returns None under memory pressure -- both are load-bearing here.
    """

    def __init__(self, out_of_memory=False):
        self.store = {}
        self.puts = []
        self.out_of_memory = out_of_memory

    def contains(self, key, search_range=None, pin=False):
        return "LocalCPUBackend" if key in self.store else None

    def batched_put(self, keys, memory_objs, transfer_spec=None, location=None):
        for key, obj in zip(keys, memory_objs, strict=True):
            self.store[key] = obj
            self.puts.append(key)

    def get(self, key, location=None):
        return self.store.get(key)

    def allocate(self, shapes, dtypes, fmt=None, eviction=True, busy_loop=True):
        if self.out_of_memory:
            return None
        return FakeMemoryObj(shapes, dtypes, fmt)


def codec(storage=None):
    c = StateByteCodec(
        FakeBackend(),
        FakeStaged(),
        entry_bytes=4096,
        model_name="m",
        world_size=8,
        worker_id=0,
    )
    c.bind_storage_manager(FakeStorageManager() if storage is None else storage)
    return c


def test_one_hash_is_one_key():
    """State is an indivisible whole-prefix snapshot: N keys would be useful
    only all together and would multiply the partial-invalidation chance."""
    c = codec()
    assert c.key(1234) == c.key(1234)
    assert c.key(1234) != c.key(5678)


def test_the_key_carries_the_atom_hash_unmodified():
    c = codec()
    assert c.key(1234).chunk_hash == 1234


def test_the_key_dtype_matches_the_object_we_allocate():
    """The key's dtype is part of its identity; a mismatch against the uint8
    object would make the key describe something we never stored."""
    c = codec()
    assert c.key(1234).dtype is torch.uint8


def test_the_key_is_per_worker():
    """Each TP rank stores its own shard of the state."""
    a = StateByteCodec(
        FakeBackend(), FakeStaged(), 4096, model_name="m", world_size=8, worker_id=0
    )
    b = StateByteCodec(
        FakeBackend(), FakeStaged(), 4096, model_name="m", world_size=8, worker_id=1
    )
    assert a.key(1234) != b.key(1234)


def test_put_packs_every_view_of_the_group():
    c = codec()
    c.put(1234, entry_index=2)
    segments, _ = c._staged.packed[-1]
    assert segments == c._backend.state_entry_views(2)


def test_put_writes_exactly_one_object():
    c = codec()
    c.put(1234, entry_index=2)
    assert len(c._storage.puts) == 1


def test_put_allocates_one_uint8_object_of_entry_bytes():
    c = codec()
    c.put(1234, entry_index=2)
    obj = c._storage.store[c.key(1234)]
    assert obj.dtypes is torch.uint8
    assert tuple(obj.shapes) == (4096,)


def test_get_on_a_miss_is_false_and_moves_nothing():
    c = codec()
    assert c.get(9999, entry_index=1) is False
    assert c._staged.unpacked == []


def test_get_after_put_unpacks_into_the_destination_group():
    c = codec()
    c.put(1234, entry_index=2)
    assert c.get(1234, entry_index=3) is True
    _, segments = c._staged.unpacked[-1]
    assert segments == c._backend.state_entry_views(3)


def test_contains_answers_from_storage_not_from_a_local_set():
    """A local set would go stale the moment LMCache's LRU evicted."""
    c = codec()
    assert c.contains(1234) is False
    c.put(1234, entry_index=2)
    assert c.contains(1234) is True


def test_an_allocation_refusal_is_a_quiet_false_not_an_error():
    """`allocate` returns None under memory pressure. Spilling is best-effort:
    the pool already counted the eviction, so a refusal costs a later prefix
    hit and nothing else."""
    c = codec(FakeStorageManager(out_of_memory=True))
    assert c.put(1234, entry_index=2) is False
    assert c._staged.packed == []
    assert c._storage.puts == []


def test_nothing_moves_before_a_storage_manager_is_bound():
    c = StateByteCodec(
        FakeBackend(), FakeStaged(), 4096, model_name="m", world_size=8, worker_id=0
    )
    assert c.put(1234, entry_index=2) is False
    assert c.get(1234, entry_index=2) is False
    assert c.contains(1234) is False


def test_a_zero_sized_entry_is_a_construction_error():
    """entry_bytes is computed from the backend's layout; zero means the
    caller measured nothing, and every put would then store an empty object."""
    with pytest.raises(ValueError, match="must be > 0"):
        StateByteCodec(
            FakeBackend(), FakeStaged(), 0, model_name="m", world_size=8, worker_id=0
        )
