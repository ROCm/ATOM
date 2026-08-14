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
        # `get` hands out a +1 reference that the caller owns; the count is
        # tracked so a leak is a test failure rather than a production one.
        self.ref_count = 1

    def ref_count_up(self):
        self.ref_count += 1

    def ref_count_down(self):
        self.ref_count -= 1


def _allocatable_formats():
    """Exactly what `MixedMemoryAllocator.allocate` routes to the pinned
    allocator (`memory_management.py`). BINARY is *not* in this list -- it
    falls through to `raise ValueError`, and BINARY_BUFFER, while accepted,
    yields a BytesBufferMemoryObj whose `.tensor` is None and which
    `StagedTransfer.memory_tensor` then rejects. Mirroring the vendor's
    accept-list here is what stops a format regression from shipping green.
    """
    from lmcache.v1.memory_management import MemoryFormat

    return {
        MemoryFormat.KV_2LTD,
        MemoryFormat.KV_2TD,
        MemoryFormat.KV_T2D,
        MemoryFormat.KV_MLA_FMT,
        MemoryFormat.EC_TD,
    }


class FakeStorageManager:
    """Mirrors the pinned LMCache StorageManager surface.

    `contains` returns an Optional[str] location, not a bool; `allocate`
    returns None under memory pressure, raises on a format the real
    `MixedMemoryAllocator` would refuse, and takes the `busy_loop` a store
    must not leave defaulted; `get` hands out a +1 reference the caller must
    discharge. All of them are load-bearing here.
    """

    def __init__(self, out_of_memory=False):
        self.store = {}
        self.puts = []
        self.busy_loops = []
        self.out_of_memory = out_of_memory

    def contains(self, key, search_range=None, pin=False):
        return "LocalCPUBackend" if key in self.store else None

    def batched_put(self, keys, memory_objs, transfer_spec=None, location=None):
        for key, obj in zip(keys, memory_objs, strict=True):
            self.store[key] = obj
            self.puts.append(key)

    def get(self, key, location=None):
        obj = self.store.get(key)
        if obj is None:
            return None
        # `LocalCPUBackend.get_blocking` refs up for the caller.
        obj.ref_count_up()
        return obj

    def allocate(self, shapes, dtypes, fmt=None, eviction=True, busy_loop=True):
        if fmt not in _allocatable_formats():
            raise ValueError(f"Unsupported memory format: {fmt}")
        # Real LMCache accepts either value; we refuse True so that *every*
        # test that reaches a spill is a detector for it. See the assert's
        # message and `test_a_spill_never_waits_for_lmcache_to_find_room`.
        assert busy_loop is False, (
            "the state tier's spill is a store, and a store must pass "
            "busy_loop=False -- LocalCPUBackend.allocate spins `while True` on "
            "0.1s sleeps under the default"
        )
        self.busy_loops.append(busy_loop)
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


def test_a_successful_put_returns_true():
    """`_do_spill` does `stored = bool(self.codec.put(...))` and sends a falsy
    result to `_index_failed`. The refusal paths pin False; without this, a
    regression to an implicit `return None` on the success path would send
    *every* successful spill to the failure report -- permanently disabling
    indexing while the bytes are actually in LMCache -- and nothing would
    fail, because `test_state_tier.py`'s FakeCodec.put returns a literal True.
    """
    c = codec()
    assert c.put(1234, entry_index=2) is True


def test_put_allocates_one_uint8_object_of_entry_bytes():
    c = codec()
    c.put(1234, entry_index=2)
    obj = c._storage.store[c.key(1234)]
    assert obj.dtypes is torch.uint8
    assert tuple(obj.shapes) == (4096,)


def test_put_allocates_under_a_format_the_lmcache_allocator_accepts():
    """`MixedMemoryAllocator.allocate` raises on BINARY and hands back a
    tensor-less BytesBufferMemoryObj for BINARY_BUFFER, so the format has to
    come from its tensor accept-list. The shape/dtype already make the object
    an opaque flat blob, so the value is inert beyond passing that check."""
    from lmcache.v1.memory_management import MemoryFormat

    c = codec()
    c.put(1234, entry_index=2)
    assert c._storage.store[c.key(1234)].fmt is MemoryFormat.KV_2LTD


def test_a_spill_never_waits_for_lmcache_to_find_room():
    """A store must pass `busy_loop=False`, which is not LMCache's default.

    `LocalCPUBackend.allocate`'s own docstring says busy_loop "should only be
    used for retrieve" because "many stores happen concurrently (if they
    busy_loop, deadlock happens)". Under the default, a pool with no eviction
    candidate loops `while True` on 0.1s sleeps with no attempt bound: it never
    returns the None `put` is written to handle, and instead hangs the state
    tier's save worker for as long as CPU memory stays full.

    Stated here as well as in the fake's assert so that deleting one still
    leaves a detector.
    """
    c = codec()
    c.put(1234, entry_index=2)
    assert c._storage.busy_loops == [False]


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


def test_get_releases_the_reference_it_was_handed():
    """`get_blocking` refs up for the caller and LMCache's own callers ref
    down after use. Holding it would keep the count off zero forever: LRU
    eviction drops the block from the index but never returns it to the pinned
    allocator, leaking one entry (53.6 MiB on the real model) per hit until
    `allocate` returns None for good."""
    c = codec()
    c.put(1234, entry_index=2)
    obj = c._storage.store[c.key(1234)]
    before = obj.ref_count
    assert c.get(1234, entry_index=3) is True
    assert obj.ref_count == before


def test_a_failing_unpack_still_releases_the_reference():
    """`finally`, not a trailing statement -- an exception on the unpack path
    would otherwise leak exactly the same way."""
    c = codec()
    c.put(1234, entry_index=2)
    obj = c._storage.store[c.key(1234)]
    before = obj.ref_count

    def _boom(src, segments):
        raise RuntimeError("unpack blew up")

    c._staged.unpack = _boom
    with pytest.raises(RuntimeError, match="unpack blew up"):
        c.get(1234, entry_index=3)
    assert obj.ref_count == before


def test_put_does_not_double_free_the_allocate_reference():
    """`StorageManager.batched_put` discharges the allocate reference itself,
    so a symmetric `ref_count_down` in `put` would be a double free."""
    c = codec()
    c.put(1234, entry_index=2)
    assert c._storage.store[c.key(1234)].ref_count == 1


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
