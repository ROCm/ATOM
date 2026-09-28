# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV sleep/wake ownership must include GLM-5.3's k-pool tail."""

from types import SimpleNamespace

import torch
from torch import nn

from atom.rollout import memory_manager
from atom.rollout.memory_manager import MemoryManagerMixin


class _SparseImpl(nn.Module):
    """`SparseMHAPagedAttentionImpl`: holds `index_cache` and nothing else.

    The nesting is the whole point. MiniMax-M3's indexer slice is bound to the
    impl (`module.impl.index_cache`) while K/V go on the `Attention` around it,
    so a release that gates `index_cache` on a sibling `k_cache` never reaches
    the one module that has it.
    """

    def __init__(self):
        super().__init__()
        self.index_cache = torch.empty(1)


class _IndexerCache(nn.Module):
    """`DeepseekV32IndexerCache`: its slice lives in a one-element list."""

    def __init__(self):
        super().__init__()
        self.kv_cache = [torch.empty(1)]


class _Indexer(nn.Module):
    """The MLA indexer, whose `k_cache` is a *module*, not a view.

    Same name as the K view every attention module carries, and the release
    walk visits both. Blanking this one costs no memory and breaks waking:
    rebinding goes through `indexer.k_cache.kv_cache[0]`.
    """

    def __init__(self):
        super().__init__()
        self.k_cache = _IndexerCache()


class _CacheOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_cache = torch.empty(1)
        self.v_cache = torch.empty(1)
        self.kv_cache = torch.empty(1)
        self.kpool_tail_cache = torch.empty(1)
        # Views of the same pool buffer as the caches above, handed over by
        # `build_kv_cache_tensor` on an fp8 / sparse model.
        self.k_scale = torch.empty(1)
        self.v_scale = torch.empty(1)
        self.impl = _SparseImpl()
        self.indexer = _Indexer()


class _ValueScaledAttention(nn.Module):
    """`MiMoV2Attention`: a `v_scale` that is not a dequant plane.

    It is the config's `attention_value_scale`, a float multiplied into V, and
    blanking it does not raise -- it silently stops scaling. The KV views live
    on the `Attention` this module contains, whose own `v_scale` IS the dequant
    plane. Same name one level apart, which is why the release reads the value
    rather than the name.
    """

    def __init__(self):
        super().__init__()
        self.v_scale = 0.5
        self.attn = _CacheOwner()


def test_release_kv_cache_drops_runner_and_module_tail_references(monkeypatch):
    owner = _CacheOwner()
    runner = SimpleNamespace(
        kv_cache=torch.empty(1),
        kv_scale=torch.empty(1),
        index_cache=torch.empty(1),
        mamba_k_cache=torch.empty(1),
        mamba_v_cache=torch.empty(1),
        kpool_tail_cache=torch.empty(1),
        config=SimpleNamespace(num_kvcache_blocks=7),
        model=nn.Sequential(owner),
        label="test",
    )
    runner._get_models_with_kv = lambda: [runner.model]
    monkeypatch.setattr(memory_manager, "set_kv_cache_data", lambda _value: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    MemoryManagerMixin._release_kv_cache(runner)

    assert runner.kv_cache is None
    assert runner._kv_cache_num_blocks == 7
    assert not hasattr(runner, "kpool_tail_cache")
    assert owner.k_cache is owner.v_cache is owner.kv_cache is None
    assert owner.kpool_tail_cache is None


def test_release_kv_cache_drops_every_view_of_the_one_pool(monkeypatch):
    """Scales and indexer slices too, or the pool is not freed at all.

    They used to be separate allocations, so a surviving `k_scale` leaked a
    scale tensor. They are regions of one buffer now, so any one of them left
    on a module pins the whole KV pool -- and the sleep path reports success.
    """
    outer = _ValueScaledAttention()
    owner = outer.attn
    runner = SimpleNamespace(
        kv_cache=torch.empty(1),
        config=SimpleNamespace(num_kvcache_blocks=7),
        model=nn.Sequential(outer),
        label="test",
    )
    runner._get_models_with_kv = lambda: [runner.model]
    monkeypatch.setattr(memory_manager, "set_kv_cache_data", lambda _value: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    MemoryManagerMixin._release_kv_cache(runner)

    assert owner.k_scale is owner.v_scale is None
    # The impl's, which no sibling `k_cache` announces.
    assert owner.impl.index_cache is None
    # And the one a level up, which only shares the name, keeps its value.
    assert outer.v_scale == 0.5


def test_release_kv_cache_leaves_the_indexer_rebindable(monkeypatch):
    """Waking assigns `indexer.k_cache.kv_cache[0] = ...`, so that path has to
    survive the sleep. Two ways it did not, both freeing nothing extra and both
    failing only on the way back: the container replaced by `None`, and the
    `k_cache` *module* blanked because a K view elsewhere shares its name.
    """
    owner = _CacheOwner()
    indexer = owner.indexer.k_cache
    runner = SimpleNamespace(
        kv_cache=torch.empty(1),
        config=SimpleNamespace(num_kvcache_blocks=7),
        model=nn.Sequential(owner),
        label="test",
    )
    runner._get_models_with_kv = lambda: [runner.model]
    monkeypatch.setattr(memory_manager, "set_kv_cache_data", lambda _value: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    MemoryManagerMixin._release_kv_cache(runner)

    assert isinstance(owner.indexer.k_cache, _IndexerCache)
    assert isinstance(indexer.kv_cache, list)
    assert indexer.kv_cache[0].numel() == 0
    owner.indexer.k_cache.kv_cache[0] = torch.empty(1)  # what waking does
