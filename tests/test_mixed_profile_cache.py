# SPDX-License-Identifier: MIT
"""Temporary profiling pools must release both ordinary and registered views."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder,
)


@pytest.mark.parametrize("fail_forward", [False, True])
def test_temporary_pool_restores_registered_buffers(monkeypatch, fail_forward):
    module = torch.nn.Module()
    placeholder = torch.ones(1)
    module.register_buffer("kv_cache", placeholder)
    module.scale = placeholder
    old_plan = SimpleNamespace()
    plan = SimpleNamespace(entries={})
    old_plan.with_paged_entries = lambda blocks: plan
    runner = SimpleNamespace(model=module, pool_plan=old_plan)
    weak_pools = []

    def allocate(**kwargs):
        pool = torch.zeros(64)
        weak_pools.append(weakref.ref(pool))
        return {"test_pool": pool}

    def bind(layer):
        layer.kv_cache = runner.test_pool[:8]
        layer.scale = runner.test_pool[8:16]
        layer.new_view = runner.test_pool[16:24]

    builder = SimpleNamespace(
        model_runner=runner,
        pool_geometry="original",
        num_blocks=0,
        allocate_kv_cache_tensors=allocate,
        allocate_per_req_cache=lambda entries: {},
        warmup_per_req_cache=lambda: None,
        build_kv_cache_tensor=bind,
        _invalidate_pool_caches=lambda: None,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def profile():
        with DeepseekV4AttentionMetadataBuilder.temporary_profile_cache(builder, 64):
            assert module.kv_cache.numel() == 8
            assert module.scale.numel() == 8
            if fail_forward:
                raise RuntimeError("forward failed")

    if fail_forward:
        with pytest.raises(RuntimeError, match="forward failed"):
            profile()
    else:
        profile()
    gc.collect()
    assert module.kv_cache is placeholder
    assert module.scale is placeholder
    assert not hasattr(module, "new_view")
    assert not hasattr(runner, "test_pool")
    assert runner.pool_plan is old_plan
    assert all(ref() is None for ref in weak_pools)
