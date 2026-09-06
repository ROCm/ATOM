# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV sleep/wake ownership must include GLM-5.3's k-pool tail."""

from types import SimpleNamespace

import torch
from torch import nn

from atom.rollout import memory_manager
from atom.rollout.memory_manager import MemoryManagerMixin


class _CacheOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_cache = object()
        self.v_cache = object()
        self.kv_cache = object()
        self.kpool_tail_cache = object()


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
