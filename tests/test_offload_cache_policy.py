# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import OrderedDict
from types import SimpleNamespace

import pytest

pytest.importorskip("lmcache")

from atom.kv_transfer.offload.cache_policy import SLRUCachePolicy, register_slru_policy


def test_reused_cpu_prefix_survives_scan():
    policy = SLRUCachePolicy()
    cache = OrderedDict()
    for key in range(8):
        cache[key] = SimpleNamespace(can_evict=True)
        policy.update_on_put(key)
    # LMCache touches a prefix in reverse order so the head stays newest.
    for key in reversed(range(4)):
        policy.update_on_hit(key, cache)
    for key in range(10, 30):
        (victim,) = policy.get_evict_candidates(cache)
        del cache[victim]
        cache[key] = SimpleNamespace(can_evict=True)
        policy.update_on_put(key)
    assert all(key in cache for key in range(4))
    assert len(cache) == 8


def test_cpu_policy_respects_pins_and_can_evict_protected_data():
    policy = SLRUCachePolicy()
    cache = OrderedDict((key, SimpleNamespace(can_evict=True)) for key in range(4))
    for key in cache:
        policy.update_on_put(key)
    policy.update_on_hit(0, cache)
    policy.update_on_hit(1, cache)
    cache[2].can_evict = cache[3].can_evict = False
    assert policy.get_evict_candidates(cache, 3) == [0, 1]
    for value in cache.values():
        value.can_evict = False
    assert policy.get_evict_candidates(cache) == []


def test_cpu_hot_set_can_change_and_explicit_removal_is_safe():
    policy = SLRUCachePolicy()
    cache = OrderedDict((key, SimpleNamespace(can_evict=True)) for key in range(4))
    for key in cache:
        policy.update_on_put(key)
    for key in range(4):
        policy.update_on_hit(key, cache)
    assert policy.get_evict_candidates(cache, 2) == [0, 1]
    del cache[2]
    policy.update_on_force_evict(2)
    assert set(policy.get_evict_candidates(cache, 10)) == {0, 1, 3}


def test_cpu_policy_is_available_through_lmcache_registry():
    from lmcache.v1.storage_backend.cache_policy import get_cache_policy

    register_slru_policy()
    assert isinstance(get_cache_policy("ATOM_SLRU"), SLRUCachePolicy)


@pytest.mark.parametrize("all_ranks", [False, True])
def test_offload_configuration_keeps_lookup_scope_explicit(monkeypatch, all_ranks):
    from lmcache.v1.config import LMCacheEngineConfig

    from atom.kv_transfer.offload.config import build_lmcache_config

    monkeypatch.setattr(
        LMCacheEngineConfig,
        "from_env",
        lambda: SimpleNamespace(
            cache_policy="LRU",
            lookup_server_worker_ids=None,
            local_cpu=True,
            max_local_cpu_size=256,
            local_disk=None,
            max_local_disk_size=0,
        ),
    )
    extra = {"lmcache.cache_policy": "ATOM_SLRU"}
    if all_ranks:
        extra["lmcache.lookup_server_worker_ids"] = []
    cfg = build_lmcache_config({"kv_connector_extra_config": extra})
    assert cfg.lookup_server_worker_ids == ([] if all_ranks else [0])
    assert cfg.max_local_cpu_size == 256
    assert cfg.local_disk is None


def test_all_rank_lookup_env_reaches_the_lmcache_factory(monkeypatch):
    from atom.kv_transfer.offload.config import build_lmcache_config

    monkeypatch.setenv("LMCACHE_LOOKUP_SERVER_WORKER_IDS", "0,1,2,3")
    cfg = build_lmcache_config()
    assert cfg.get_lookup_server_worker_ids(use_mla=False, world_size=4) == [0, 1, 2, 3]


@pytest.mark.parametrize("source", ["env", "extra"])
def test_async_loading_is_rejected_before_lookup_factory(monkeypatch, source):
    from atom.kv_transfer.offload.config import build_lmcache_config

    monkeypatch.setenv(
        "LMCACHE_ENABLE_ASYNC_LOADING", "true" if source == "env" else "false"
    )
    config = (
        {"kv_connector_extra_config": {"lmcache.enable_async_loading": True}}
        if source == "extra"
        else None
    )
    with pytest.raises(ValueError, match="LMCACHE_ENABLE_ASYNC_LOADING"):
        build_lmcache_config(config)


def test_sync_loading_remains_supported(monkeypatch):
    from atom.kv_transfer.offload.config import build_lmcache_config

    monkeypatch.setenv("LMCACHE_ENABLE_ASYNC_LOADING", "false")
    assert build_lmcache_config().enable_async_loading is False


@pytest.mark.parametrize("source", ["env", "extra"])
def test_slru_config_normalizes_case_and_whitespace(monkeypatch, source):
    from lmcache.v1.storage_backend.cache_policy import get_cache_policy

    from atom.kv_transfer.offload.config import build_lmcache_config

    monkeypatch.setenv("LMCACHE_ENABLE_ASYNC_LOADING", "false")
    monkeypatch.setenv(
        "LMCACHE_CACHE_POLICY", " atom_slru " if source == "env" else "LRU"
    )
    config = (
        {"kv_connector_extra_config": {"lmcache.cache_policy": " Atom_Slru "}}
        if source == "extra"
        else None
    )
    cfg = build_lmcache_config(config)
    assert cfg.cache_policy == "ATOM_SLRU"
    assert isinstance(get_cache_policy(cfg.cache_policy), SLRUCachePolicy)


@pytest.mark.parametrize("size", [8, 1024, 10000])
def test_probationary_eviction_does_not_scan_protected_segment(size):
    class CountedCache(OrderedDict):
        visits = 0

        def items(self):
            for item in super().items():
                self.visits += 1
                yield item

        def get(self, key, default=None):
            self.visits += 1
            return super().get(key, default)

    cache = CountedCache()
    policy = SLRUCachePolicy()
    for key in range(size):
        cache[key] = SimpleNamespace(can_evict=True)
        policy.update_on_put(key)
    for key in range(size // 2):
        policy.update_on_hit(key, cache)
    for key in range(size, size + size // 2):
        (victim,) = policy.get_evict_candidates(cache)
        del cache[victim]  # Normal backend eviction does not issue a callback.
        cache[key] = SimpleNamespace(can_evict=True)
        policy.update_on_put(key)

    # Steady-state puts sit behind the protected half in the backend mapping.
    # Each eviction should only inspect one stale entry and its next victim.
    for key in range(2 * size, 3 * size):
        cache.visits = 0
        (victim,) = policy.get_evict_candidates(cache)
        assert cache.visits <= 2
        assert victim >= size
        del cache[victim]
        cache[key] = SimpleNamespace(can_evict=True)
        policy.update_on_put(key)
    assert all(key in cache for key in range(size // 2))


def test_candidate_selection_survives_retry_pin_and_reinsertion():
    policy = SLRUCachePolicy()
    cache = OrderedDict((key, SimpleNamespace(can_evict=True)) for key in range(4))
    for key in cache:
        policy.update_on_put(key)
    assert policy.get_evict_candidates(cache) == [0]
    # Merely selecting a victim must not forget it if the caller does not evict.
    assert policy.get_evict_candidates(cache) == [0]
    cache[0].can_evict = False
    assert policy.get_evict_candidates(cache) == [1]
    cache[0].can_evict = True
    assert policy.get_evict_candidates(cache) == [0]
    del cache[0]
    cache[0] = SimpleNamespace(can_evict=True)
    policy.update_on_put(0)
    assert policy.get_evict_candidates(cache) == [1]
    del cache[1]
    policy.update_on_force_evict(1)
    assert policy.get_evict_candidates(cache, 10) == [2, 3, 0]
    assert policy.get_evict_candidates(cache, 0) == []
    assert policy.get_evict_candidates(cache, -1) == []


def test_stale_entries_in_both_queues_are_removed_lazily():
    policy = SLRUCachePolicy()
    cache = OrderedDict((key, SimpleNamespace(can_evict=True)) for key in range(6))
    for key in cache:
        policy.update_on_put(key)
    policy.update_on_hit(0, cache)
    policy.update_on_hit(1, cache)
    del cache[0]
    del cache[2]
    for key in (3, 4, 5):
        cache[key].can_evict = False
    assert policy.get_evict_candidates(cache, 10) == [1]
    assert 0 not in policy._protected
    assert 2 not in policy._probationary
