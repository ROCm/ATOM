# SPDX-License-Identifier: MIT
"""Unplaced V4.1 decoder experts keep the ordinary EP weight allocation."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("enable_eplb", [False, True])
def test_unplaced_layer_allocates_no_redundant_weights(
    atom_modules, monkeypatch, rank, enable_eplb
):
    moe = atom_modules.moe
    parallel = moe.FusedMoEParallelConfig(
        tp_size=1,
        dp_size=8,
        ep_size=8,
        tp_rank=0,
        dp_rank=rank,
        ep_rank=rank,
        use_ep=True,
        local_ep_size=8,
        requested_all2all_backend="none",
    )
    monkeypatch.setattr(moe.FusedMoEParallelConfig, "make", lambda *args: parallel)
    group = SimpleNamespace(world_size=8, rank_in_group=rank)
    monkeypatch.setattr(moe, "get_dp_group", lambda: group)
    monkeypatch.setattr(moe, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        moe,
        "is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config",
        lambda *a, **k: False,
    )
    config = SimpleNamespace(
        eplb_enable=True,
        eplb_config=SimpleNamespace(num_redundant_experts=32),
        torch_dtype=torch.float32,
        fake_eplb=False,
        enable_dp_attention=True,
        max_num_batched_tokens=16,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(moe, "get_current_atom_config", lambda: config)
    layer = moe.FusedMoE(
        num_experts=384,
        top_k=8,
        hidden_size=8,
        intermediate_size=8,
        tp_size=1,
        dp_size=8,
        layer_id=20,
        prefix="layers.20.ffn.experts",
        enable_eplb=enable_eplb,
    )
    slots = 52 if enable_eplb else 48
    assert layer.num_redundant_experts == (32 if enable_eplb else 0)
    assert layer.global_num_experts == slots * 8
    assert layer.w13_weight.shape == (slots, 16, 8)
    assert layer.w2_weight.shape == (slots, 8, 8)
    base_experts = int((layer.expert_map[:384] >= 0).sum())
    expected_base = (52 if rank < 7 else 20) if enable_eplb else 48
    assert base_experts == expected_base
