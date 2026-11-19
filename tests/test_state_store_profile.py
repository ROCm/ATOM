# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OFFLOAD_PROFILE timing of state-tier stores (`[STATE-STORE-PROF]`).

The profile exists to say where a store's wall time goes, so the tests pin
what each phase covers and that profiling off leaves the store path's calls
unchanged (test doubles without a `prof` argument keep working).
"""

import dataclasses
import logging
import sys
import types
from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import StateStoreOperationId
from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec
from atom.kv_transfer.offload.hybrid.kimi_k3.state_tier import StateOffloadTier


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


class _Storage:
    def __init__(self) -> None:
        self.objects: dict = {}
        self.removed: list = []

    def batched_put(self, keys, objs) -> None:
        for k, o in zip(keys, objs, strict=True):
            self.objects[k] = o

    def remove(self, key) -> None:
        self.removed.append(key)
        self.objects.pop(key, None)


class _Staged:
    """`pack_segments` stand-in that records the phases `_gather` would."""

    def __init__(self) -> None:
        self.prof_seen = []

    def pack_segments(self, ptrs, sizes, obj, prof=None):
        self.prof_seen.append(prof)
        if prof is not None:
            prof["gather_ms"] = 1.0
            prof["d2h_ms"] = 2.0


def _codec(max_entries=None):
    storage = _Storage()
    staged = _Staged()
    codec = StateByteCodec(
        backend=SimpleNamespace(page_unit_segments=lambda units: ([], [])),
        staged=staged,
        entry_bytes=1024,
        model_name="q",
        world_size=4,
        worker_id=0,
        layout_id="gdn",
        max_cache_bytes=None if max_entries is None else 1024 * max_entries,
    )
    codec.bind_storage_manager(storage)
    codec._allocate = lambda nbytes: SimpleNamespace(ref_count_down=lambda: None)
    return codec, storage, staged


def test_codec_records_every_phase_under_the_entry_cap(lmcache_key):
    codec, storage, staged = _codec(max_entries=1)
    assert codec.put(1, [0])
    prof = {}
    released = []
    assert codec.put(2, [0], on_source_released=lambda: released.append(1), prof=prof)
    assert released == [1]
    # The cap forced one eviction, and every phase is timed.
    assert prof["evict_n"] == 1 and len(storage.removed) == 1
    for k in (
        "lock_ms",
        "evict_ms",
        "alloc_ms",
        "gather_ms",
        "d2h_ms",
        "pack_ms",
        "put_ms",
    ):
        assert prof[k] >= 0.0, k
    # The staged transfer got the dict only on the profiled store.
    assert staged.prof_seen == [None, prof]


def test_codec_without_profile_calls_pack_as_before(lmcache_key):
    codec, _, staged = _codec()
    seen = {}

    def pack_segments(ptrs, sizes, obj):  # no `prof` parameter at all
        seen["called"] = True

    staged.pack_segments = pack_segments
    assert codec.put(5, [0])
    assert seen == {"called": True}


class _ProfCodec:
    def __init__(self) -> None:
        self.kwargs = []

    def put(self, h, unit_ids, on_source_released=None, **kw):
        self.kwargs.append(kw)
        if "prof" in kw:
            kw["prof"].update(lock_ms=0.5, put_ms=3.0, evict_n=2)
        if on_source_released is not None:
            on_source_released()
        return True


def test_tier_logs_a_profile_line_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("OFFLOAD_PROFILE", "1")
    codec = _ProfCodec()
    tier = StateOffloadTier(codec, profile_tag="rank=3")
    tier._store_profile._interval_s = 0.0
    try:
        with caplog.at_level(logging.INFO):
            tier.submit_store(StateStoreOperationId(11, 1), (0,))
            tier.drain()
    finally:
        tier.shutdown()
    lines = [
        r.getMessage() for r in caplog.records if "[STATE-STORE-PROF]" in r.getMessage()
    ]
    assert len(lines) == 1
    line = lines[0]
    assert "rank=3 n=1" in line and "stored=1" in line and "evicted=2" in line
    for k in ("queue=", "budget=", "lock=", "put=", "total=", "cpu="):
        assert k in line, k
    assert "prof" in codec.kwargs[0]


def test_tier_passes_no_prof_when_disabled(monkeypatch):
    monkeypatch.delenv("OFFLOAD_PROFILE", raising=False)
    codec = _ProfCodec()
    tier = StateOffloadTier(codec)
    try:
        assert tier._store_profile is None
        tier.submit_store(StateStoreOperationId(12, 1), (0,))
        tier.drain()
    finally:
        tier.shutdown()
    assert codec.kwargs == [{}]
