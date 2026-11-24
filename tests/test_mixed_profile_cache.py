# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Temporary profiling pools must release both ordinary and registered views."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def builder_cls():
    with atom_without_aiter() as load:
        yield load(
            "atom.model_ops.attentions.deepseek_v4_attn"
        ).DeepseekV4AttentionMetadataBuilder


@pytest.mark.parametrize("fail_forward", [False, True])
def test_profile_drops_region_layout_when_serving_reuses_addresses(
    monkeypatch, builder_cls, fail_forward
):
    storage = torch.zeros(128, dtype=torch.uint8)
    plane = torch.zeros(64, dtype=torch.uint8)
    serving = storage.view(2, 4, 16)
    plan = SimpleNamespace(entries={})
    plan.with_paged_entries = lambda blocks: plan
    runner = SimpleNamespace(
        model=torch.nn.Module(),
        pool_plan=plan,
        config=SimpleNamespace(speculative_config=None),
        index_pool=serving,
    )
    builder = SimpleNamespace(
        model_runner=runner,
        pool_geometry=SimpleNamespace(envelope_rows=4),
        num_blocks=4,
        csa_layers=[0, 1],
        _page_unit_region_owners=(),
        _kv_planes=lambda: [plane],
        _plane_row_widths=lambda: [8],
        _indexer_page_pools=lambda: [(runner.index_pool, None)],
        allocate_kv_cache_tensors=lambda **_: {
            "index_pool": storage[:64].view(2, 2, 16)
        },
        allocate_per_req_cache=lambda _: {},
        build_kv_cache_tensor=lambda _: None,
    )
    runner.attn_metadata_builder = builder
    builder._invalidate_pool_caches = lambda: builder_cls._invalidate_pool_caches(
        builder
    )
    builder.warmup_per_req_cache = lambda: builder_cls._page_unit_regions(builder)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def profile():
        with builder_cls.temporary_profile_cache(builder, 2):
            bases, _ = builder_cls._page_unit_regions(builder)
            assert bases[-1] == storage.data_ptr() + 32
            if fail_forward:
                raise RuntimeError("profile failed")

    if fail_forward:
        with pytest.raises(RuntimeError, match="profile failed"):
            profile()
    else:
        profile()
    assert runner.index_pool is serving
    bases, _ = builder_cls._page_unit_regions(builder)
    assert bases[-1] == serving.data_ptr() + serving.stride(0)


@pytest.mark.parametrize("fail_forward", [False, True])
@pytest.mark.parametrize("independent_draft", [False, True])
def test_temporary_pool_restores_registered_buffers(
    monkeypatch, fail_forward, independent_draft, builder_cls
):
    module = torch.nn.Module()
    placeholder = torch.ones(1)
    module.register_buffer("kv_cache", placeholder)
    module.scale = placeholder
    old_plan = SimpleNamespace()
    plan = SimpleNamespace(entries={})
    old_plan.with_paged_entries = lambda blocks: plan
    runner = SimpleNamespace(
        model=module,
        pool_plan=old_plan,
        config=SimpleNamespace(speculative_config=None),
    )
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
    runner.attn_metadata_builder = builder
    draft = torch.nn.Module()
    draft.register_buffer("kv_cache", placeholder)
    if independent_draft:
        runner.config.speculative_config = object()
        runner.drafter = SimpleNamespace(model=draft)

        def bind_draft(layer):
            layer.kv_cache = runner.test_pool[24:32]
            return layer.kv_cache

        runner.draft_kv_builder = SimpleNamespace(build_kv_cache_tensor=bind_draft)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def profile():
        with builder_cls.temporary_profile_cache(builder, 64):
            assert module.kv_cache.numel() == 8
            assert module.scale.numel() == 8
            if independent_draft:
                assert draft.kv_cache.numel() == 8
                assert draft.kv_cache.data_ptr() == runner.test_pool[24:].data_ptr()
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
    assert draft.kv_cache is placeholder
    assert not hasattr(module, "new_view")
    assert not hasattr(runner, "test_pool")
    assert runner.pool_plan is old_plan
    assert all(ref() is None for ref in weak_pools)


def test_binding_falls_back_only_for_unrecognized_draft_modules():
    from atom.model_engine.kv_cache_binding import kv_cache_bindings

    target, draft, shared = object(), object(), object()
    target_calls, draft_calls = [], []

    def target_bind(module):
        target_calls.append(module)
        return "target" if module is target else "shared"

    def draft_bind(module):
        draft_calls.append(module)
        return "draft" if module is draft else None

    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [target]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: [draft, shared])),
        config=SimpleNamespace(speculative_config=object()),
        attn_metadata_builder=SimpleNamespace(build_kv_cache_tensor=target_bind),
        draft_kv_builder=SimpleNamespace(build_kv_cache_tensor=draft_bind),
    )
    assert list(kv_cache_bindings(runner)) == [
        (target, "target"),
        (draft, "draft"),
        (shared, "shared"),
    ]
    assert target_calls == [target, shared]
    assert draft_calls == [draft, shared]
