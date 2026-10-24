# SPDX-License-Identifier: MIT
"""The state tier's CPU budget is a reservation inside the shared LMCache pool.

State images and paged KV share one LocalCPU pool. Before, the codec's budget
only bounded the codec's own LRU: KV writes could still push LMCache's LRU over
the images, while the engine-side `StateOffloadIndex` kept advertising them, and
every such hit parked a joint load that then failed and recomputed the whole
prompt. These tests model LocalCPU's eviction rule (only objects whose ref count
is 1 are evictable) and check that a budgeted image survives KV pressure until
the codec itself retires it.
"""

import dataclasses
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def lmcache_key(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class CacheEngineKey:
        model_name: str
        world_size: int
        worker_id: int
        chunk_hash: int
        dtype: object

    utils_module = types.ModuleType("lmcache.utils")
    utils_module.CacheEngineKey = CacheEngineKey
    lmcache_module = types.ModuleType("lmcache")
    lmcache_module.__path__ = []
    lmcache_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "lmcache", lmcache_module)
    monkeypatch.setitem(sys.modules, "lmcache.utils", utils_module)
    return CacheEngineKey


class _Obj:
    """A MemoryObj's reference count, and nothing else."""

    def __init__(self, size: int = 1024) -> None:
        self.refs = 1
        self.size = size

    def ref_count_up(self) -> None:
        self.refs += 1

    def ref_count_down(self) -> None:
        assert self.refs > 0, "reference discharged twice"
        self.refs -= 1

    def get_ref_count(self) -> int:
        return self.refs

    def get_size(self) -> int:
        return self.size


class _LocalCPU:
    """`storage_manager` over LocalCPU's rules.

    `batched_put` adopts an object with its own reference unless the key is
    already resident, then drops the caller's reference. `evict_lru` is the
    pool making room for KV: it skips anything another holder still references.
    """

    def __init__(self, refuse_all: bool = False) -> None:
        self.hot: dict = {}
        self.refuse_all = refuse_all
        self.fail_put = False

    def batched_put(self, keys, objs) -> None:
        if self.fail_put:
            raise RuntimeError("put failed")
        for k, o in zip(keys, objs, strict=True):
            if not self.refuse_all and k not in self.hot:
                o.ref_count_up()
                self.hot[k] = o
            o.ref_count_down()

    def get(self, key):
        o = self.hot.get(key)
        if o is not None:
            o.ref_count_up()
        return o

    def remove(self, key) -> int:
        o = self.hot.pop(key, None)
        if o is None:
            return 0
        o.ref_count_down()
        return 1

    def evict_lru(self) -> list:
        gone = [k for k, o in self.hot.items() if o.get_ref_count() == 1]
        for k in gone:
            self.hot.pop(k).ref_count_down()
        return gone


def _codec(storage, max_entries):
    from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec

    codec = StateByteCodec(
        backend=SimpleNamespace(
            page_unit_views=lambda units: [], state_entry_views=lambda slot: []
        ),
        staged=SimpleNamespace(pack=lambda views, obj: None, unpack=lambda o, v: None),
        entry_bytes=1024,
        model_name="qwen",
        world_size=1,
        worker_id=0,
        layout_id="L",
        max_cache_bytes=None if max_entries is None else max_entries * 1024,
    )
    codec.bind_storage_manager(storage)
    allocated = []

    def allocate(nbytes):
        allocated.append(_Obj(nbytes))
        return allocated[-1]

    codec._allocate = allocate
    return codec, allocated


def test_kv_pressure_cannot_evict_a_budgeted_image(lmcache_key):
    pool = _LocalCPU()
    codec, objs = _codec(pool, max_entries=2)
    kv = _Obj()
    pool.hot["kv-chunk"] = kv  # a KV chunk only the pool references

    assert codec.put(1, [])
    assert objs[0].refs == 2  # the pool's reference and the codec's
    assert pool.evict_lru() == ["kv-chunk"]
    assert codec.get(1, 0), "the image the index advertises must still load"


def test_the_codec_lru_retires_and_frees_its_oldest_image(lmcache_key):
    pool = _LocalCPU()
    codec, objs = _codec(pool, max_entries=2)
    assert codec.put(1, []) and codec.put(2, []) and codec.put(3, [])

    assert codec.key(1) not in pool.hot
    assert objs[0].refs == 0, "a retired image goes back to the allocator"
    assert [o.refs for o in objs[1:]] == [2, 2]
    assert len(codec._held) == 2


def test_a_resident_image_is_not_stored_twice(lmcache_key):
    pool = _LocalCPU()
    codec, objs = _codec(pool, max_entries=2)
    released = []
    assert codec.put(1, [])
    assert codec.put(1, [], on_source_released=lambda: released.append(True))

    assert len(objs) == 1, "no second image-sized allocation and D2H"
    assert released == [True], "the source units must still be handed back"


def test_a_refused_put_keeps_no_reference(lmcache_key):
    pool = _LocalCPU(refuse_all=True)
    codec, objs = _codec(pool, max_entries=2)
    assert codec.put(1, [])

    assert objs[0].refs == 0, "holding a refused object would strand its block"
    assert codec._held == {}


def test_a_failed_put_discharges_both_references(lmcache_key):
    pool = _LocalCPU()
    pool.fail_put = True
    codec, objs = _codec(pool, max_entries=2)
    with pytest.raises(RuntimeError):
        codec.put(1, [])

    assert objs[0].refs == 0
    assert codec._held == {}


def test_an_unbounded_codec_leaves_eviction_to_lmcache(lmcache_key):
    pool = _LocalCPU()
    codec, objs = _codec(pool, max_entries=None)
    assert codec.put(1, [])

    assert objs[0].refs == 1
    assert pool.evict_lru() == [codec.key(1)]
