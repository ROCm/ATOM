# SPDX-License-Identifier: MIT
"""Real CPU cache storage at the vLLM binding/offload boundary, without vLLM."""

import ast
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("triton", reason="the real V4.1 cache imports Triton kernels")

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

PLUGIN = Path(__file__).resolve().parents[2] / "atom/plugin/vllm"


def _load(path, names, **scope):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    scope = {"torch": torch, "np": np, "logger": logging.getLogger(__name__), **scope}
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope
    )
    return SimpleNamespace(**{name: scope[name] for name in names})


@pytest.fixture(params=["bfloat16", "fp8", "nvfp4"])
def bound_cache(request, monkeypatch):
    overrides = _load(
        PLUGIN / "v41_core_overrides.py",
        ["VllmBackedPagedAttentionCache"],
        PagedAttentionCache=PagedAttentionCache,
    )
    module = ModuleType("atom.plugin.vllm.v41_core_overrides")
    module.VllmBackedPagedAttentionCache = overrides.VllmBackedPagedAttentionCache
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # External vLLM orchestration is stubbed; binding, geometry, allocation,
    # registration and prefill reservation are the actual implementations.
    slots_module = ModuleType("atom.plugin.vllm.state_slot_allocator")
    slots_module.StateSlotAllocator = Mock()
    monkeypatch.setitem(sys.modules, slots_module.__name__, slots_module)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=request.param, num_gpu_blocks=7),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )
    dtype = _load(
        PLUGIN / "deepseek_v41_bridge.py", ["v41_kv_cache_dtype"]
    ).v41_kv_cache_dtype(config)
    geometry = V41PoolGeometry(
        3,
        ((0, 2), (1, 1)),
        32,
        8,
        512,
        128,
        kv_dtype=dtype,
        speculative_tokens=5,
        layer_ratios=(0, 1, 2),
        index_topk=4,
    )
    size = geometry.paged_extents(7)[1] + 4 * geometry.state_bytes
    slab = torch.full((size,), 73, dtype=torch.uint8)
    proxy = SimpleNamespace(kv_cache=slab[: 7 * geometry.paged_bytes].view(7, 1, 1, -1))
    config.compilation_config = SimpleNamespace(static_forward_context={"proxy": proxy})
    reservation = (
        torch.empty(
            (4 * geometry.ring_slots + 16) * geometry.window_row_bytes,
            dtype=torch.uint8,
        )
        if geometry.window_fp8
        else None
    )
    builder = SimpleNamespace(
        geometry=geometry,
        device="cpu",
        score_workspace=None,
        _prefill_window_staging=reservation,
    )
    model = SimpleNamespace(_atom_v41_slot_allocator=object())
    bridge = _load(
        PLUGIN / "deepseek_v41_bridge.py",
        ["bind_deepseek_v41_proxy_cache", "v41_page_planes", "v41_kv_cache_dtype"],
        ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME="proxy",
        ATOM_DEEPSEEK_V41_PROXY_ALIGNMENT=256,
        AtomDeepseekV41ProxyAttention=SimpleNamespace,
        v41_proxy_geometry=lambda _: geometry,
        v41_num_state_slots=lambda _: 4,
        _v41_install_state_checkpoint_runtime=Mock(),
    )
    assert bridge.bind_deepseek_v41_proxy_cache(model, builder, config)
    cache = builder.cache
    assert bridge.bind_deepseek_v41_proxy_cache(model, builder, config)
    assert builder.cache is cache
    assert all(
        t.untyped_storage().data_ptr() == slab.data_ptr() for t in cache.allocations
    )
    module = ModuleType("atom.plugin.vllm.deepseek_v41_bridge")
    module.ensure_v41_proxy_bound = lambda _: (cache, builder)
    module.v41_page_planes = bridge.v41_page_planes
    monkeypatch.setitem(sys.modules, module.__name__, module)
    register = _load(
        PLUGIN / "kv_transfer/connector.py", ["_v41_page_tensors"]
    )._v41_page_tensors
    connector = SimpleNamespace(
        _vllm_config=config, _kv_cache_config=SimpleNamespace(num_blocks=7)
    )
    return cache, builder, module, lambda: register(connector)


def test_offload_registration_covers_page_regions_and_all_allocations(bound_cache):
    cache, _, _, register = bound_cache
    tensors, blocks, dimension = register()
    assert blocks == dimension == cache.num_pages
    regions = cache.unit_regions()
    assert len(tensors) == len(regions)
    for tensor, (pointer, size) in zip(tensors, regions):
        assert tensor.k_cache.data_ptr() == pointer
        assert tensor.k_cache.numel() * tensor.k_cache.element_size() == blocks * size
    assert (
        sum(t.numel() * t.element_size() for t in cache.allocations)
        == cache.geometry.paged_extents(blocks)[1]
        + cache.num_slots * cache.geometry.state_bytes
    )
    if cache.window_fp8:
        assert cache.page_bytes is None and len(cache.allocations) > 1
    else:
        assert tensors[0].k_cache is cache.page_bytes
    state_ranges = [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
        for _, t in cache.state_regions
    ]
    for tensor in tensors:
        start = tensor.k_cache.data_ptr()
        end = start + tensor.k_cache.numel() * tensor.k_cache.element_size()
        assert all(end <= low or start >= high for low, high in state_ranges)


def test_serving_prefill_reuses_profiled_reservation(bound_cache, monkeypatch):
    cache, builder, _, _ = bound_cache
    if not cache.window_fp8:
        assert cache.prefill_window_staging is None
        return
    reservation = builder._prefill_window_staging
    assert cache.prefill_window_staging is reservation
    for batches, tokens in [(4, 16), (1, 4), (4, 16)]:
        step = SimpleNamespace(
            decode=False,
            cu_seqlens_q=torch.zeros(batches + 1, dtype=torch.int32),
            running_tokens=tokens,
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                torch,
                "empty",
                Mock(
                    side_effect=AssertionError(
                        "prefill allocated another staging buffer"
                    )
                ),
            )
            cache._reserve_prefill_window(step)
        assert step.prefill_window_staging is reservation
        assert cache.prefill_window_staging.data_ptr() == reservation.data_ptr()


@pytest.mark.parametrize("invalid", ["missing-plane", "extra-allocation"])
def test_offload_registration_rejects_incomplete_storage(
    bound_cache, monkeypatch, invalid
):
    cache, _, module, register = bound_cache
    if invalid == "missing-plane":
        exporter = module.v41_page_planes
        monkeypatch.setattr(
            module, "v41_page_planes", lambda cache: exporter(cache)[1:]
        )
        message = "PAGE unit mismatch"
    else:
        monkeypatch.setattr(
            cache,
            "allocations",
            (
                *cache.allocations,
                torch.zeros(cache.geometry.alignment, dtype=torch.uint8),
            ),
        )
        message = "arena does not balance"
    with pytest.raises(ValueError, match=message):
        register()
