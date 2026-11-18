# SPDX-License-Identifier: MIT
"""Serving warmup follows the live step, index and window-storage contracts."""

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp4", "fp8"])
def test_addressing_warmup_uses_live_step_and_preserves_window_planes(
    atom_modules, monkeypatch, kv_dtype
):
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    warmup = importlib.import_module("atom.model_ops.attentions.deepseek_v41.warmup")
    indices = importlib.import_module("atom.model_ops.attentions.deepseek_v41.indices")
    geometry = V41PoolGeometry(
        2,
        ((0, 2),),
        32,
        4,
        512,
        32,
        kv_dtype=kv_dtype,
        layer_ratios=(2,),
        index_topk=2,
    )
    topology = tuple(
        LayerAttentionSpec(
            i,
            2,
            AttentionMode.REINDEX if i == 0 else AttentionMode.REUSE,
            kv_owner=0,
            index_group_start=0,
            index_group_size=2,
        )
        for i in range(2)
    )
    rings = [
        (
            torch.full((geometry.ring_slots, 512), 17, dtype=torch.uint8),
            torch.full((geometry.ring_slots, 16), 127, dtype=torch.uint8),
        )
        for _ in topology
    ]
    cache = SimpleNamespace(
        indptr_device="cpu",
        geometry=geometry,
        num_pages=6,
        indptr_buffers={2: tuple(torch.empty(18, dtype=torch.int32) for _ in range(2))},
        ring_view=lambda layer: rings[layer],
    )
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((self.name, grid, args, kwargs))

            return launch

    monkeypatch.setattr(indices, "_indptr_scan_all", Kernel("scan"))
    monkeypatch.setattr(indices, "_indices", Kernel("indices"))
    monkeypatch.setattr(
        indices.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length()
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    writes = []
    real_write = warmup.write_window_planes

    def write(values, scales, planes, step, window, dim, cached):
        assert step.running_tokens == values.shape[0] == scales.shape[0]
        assert not step.decode and (step.batch_ids == -1).all()
        assert values.shape[1] == dim and scales.shape[1] == dim // 32
        assert values.is_contiguous() and scales.is_contiguous()
        writes.append(cached is not None)
        return real_write(values, scales, planes, step, window, dim, cached)

    monkeypatch.setattr(warmup, "write_window_planes", write)
    warmup.warmup_prefill_addressing(cache, topology, 17, 9)

    index_calls = [call for call in calls if call[0] == "indices"]
    assert {call[1][0] for call in index_calls} == {1, 2, 4, 8, 15, 16, 17}
    assert all(call[1][1] == (2 if kv_dtype == "bf16" else 1) for call in index_calls)
    assert all(call[3]["SPLIT_PREFILL"] == (kv_dtype == "fp8") for call in index_calls)
    assert all(call[2][9].stride(0) == 9 for call in index_calls)
    assert writes == ([False, True] * 14 if kv_dtype != "bf16" else [])
    assert all((data == 17).all() and (scale == 127).all() for data, scale in rings)
