from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="full dispatch tests require aiter")

import atom.model_ops.eplb as eplb_module
import atom.model_ops.moe as moe_module
import atom.model_ops.topK as topK_module
from atom.model_ops.fused_moe import shared_expert_dispatch as dispatch_module
from atom.model_ops.fused_moe.expert_layout import MoEExpertLayout
from atom.model_ops.moe import FusedMoE, FusedMoEMethodBase, FusedMoEParallelConfig
from atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)


class _FakeMoELayer(SimpleNamespace):
    shared_dispatch_base = FusedMoE.shared_dispatch_base
    shared_expert_weight = FusedMoE.shared_expert_weight


def _dispatch_layer(*, ep_rank: int = 1) -> _FakeMoELayer:
    layout = MoEExpertLayout.make(
        num_routed=8,
        num_fused_shared_experts=1,
        num_configured_redundant=0,
        ep_size=2,
        use_all2all=True,
        eplb_enabled=False,
    )
    ns = _FakeMoELayer(
        expert_layout=layout,
        num_fused_shared_experts=1,
        global_num_experts=10,
        num_redundant_experts=0,
        routed_scaling_factor=2.0,
        ep_size=2,
        ep_rank=ep_rank,
        local_num_experts=5,
        layer_id=None,
        use_ep=True,
        expert_map=torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3, 4, -1]),
        expert_mask=torch.empty(0, dtype=torch.int32),
    )
    return ns


def test_to_dispatch_space_is_backend_neutral(monkeypatch):
    layer = _dispatch_layer(ep_rank=1)
    monkeypatch.setattr(dispatch_module, "_HAS_TRITON", False)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: False
    )
    # No GPU here; pin to the torch reference.
    routed_ids = torch.tensor([[0, 4], [3, -1]], dtype=torch.int32)
    routed_weights = torch.tensor([[0.7, 0.3], [0.6, 0.0]])

    weights, ids = FusedMoE.to_dispatch_space(layer, routed_weights, routed_ids)

    assert ids.tolist() == [[0, 5, 9], [3, -1, 9]]
    assert torch.equal(weights[:, :2], routed_weights)
    assert weights[:, 2].tolist() == [0.5, 0.5]


def test_select_experts_keeps_shared_out_of_router_and_eplb(monkeypatch):
    layer = _dispatch_layer(ep_rank=1)
    monkeypatch.setattr(dispatch_module, "_HAS_TRITON", False)
    captured = {}
    routed_weights = torch.tensor([[0.75, 0.25]])
    routed_ids = torch.tensor([[0, 4]], dtype=torch.int32)

    def fake_select_experts(**kwargs):
        captured["num_fused_shared_experts"] = kwargs["num_fused_shared_experts"]
        return routed_weights, routed_ids

    def fake_map_and_record(received_layer, received_ids):
        # EPLB must see routed ids only.
        assert received_layer is layer
        assert received_ids.shape[1] == 2
        return received_ids

    monkeypatch.setattr(FusedMoE, "select_experts", fake_select_experts)
    monkeypatch.setattr(eplb_module, "eplb_map_and_record_fused", fake_map_and_record)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: True
    )
    # Pins the ordering router -> EPLB -> shared, not the kernel arithmetic.
    layer.to_dispatch_space = lambda weights, ids: (
        FusedMoE.to_dispatch_space(layer, weights, ids)
    )

    weights, ids = FusedMoEMethodBase.select_experts_with_record(
        object(),
        layer=layer,
        hidden_states=torch.empty(1, 4),
        router_logits=torch.empty(1, 8),
        top_k=2,
        renormalize=True,
    )

    assert captured["num_fused_shared_experts"] == 0
    assert ids.tolist() == [[0, 5, 9]]
    assert weights.tolist() == [[0.75, 0.25, 1.0]]


def test_eplb_appends_unscaled_shared_weight(monkeypatch):
    layer = _dispatch_layer()
    captured = {}
    layer.expert_layout = MoEExpertLayout.make(
        num_routed=8,
        num_fused_shared_experts=1,
        num_configured_redundant=0,
        ep_size=2,
        use_all2all=True,
        eplb_enabled=True,
    )
    layer.append_shared_logical_column = lambda weights, ids: (
        FusedMoE.append_shared_logical_column(layer, weights, ids)
    )

    def fake_select_experts(**kwargs):
        captured["scale"] = kwargs["routed_scaling_factor"]
        return torch.tensor([[1.5, 0.5]]), torch.tensor([[0, 4]])

    monkeypatch.setattr(FusedMoE, "select_experts", fake_select_experts)
    monkeypatch.setattr(eplb_module, "eplb_map_and_record_fused", lambda _, ids: ids)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: True
    )

    weights, ids = FusedMoEMethodBase.select_experts_with_record(
        object(),
        layer=layer,
        hidden_states=torch.empty(1, 4),
        router_logits=torch.empty(1, 8),
        top_k=2,
        renormalize=True,
    )

    assert weights.tolist() == [[1.5, 0.5, 1.0]]
    assert ids.tolist() == [[0, 4, 8]]
    assert captured["scale"] == 2.0


def _parallel_config(*, dp_size: int, use_ep: bool) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=8,
        dp_size=dp_size,
        ep_size=8 if use_ep else 1,
        tp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=use_ep,
        local_ep_size=8,
    )


@pytest.mark.parametrize(
    "dp_size, use_ep, has_mori, expected",
    [
        # EP without DP still runs the legacy AITER fusion: no all2all backend,
        # so there is no dispatch space to fold the shared expert into.
        (1, True, True, False),
        (8, True, True, True),
        (8, False, True, False),
        (8, True, False, False),
    ],
)
def test_ep_alone_does_not_imply_all2all(
    monkeypatch, dp_size, use_ep, has_mori, expected
):
    monkeypatch.setattr(moe_module, "_has_module", lambda name: has_mori)

    config = _parallel_config(dp_size=dp_size, use_ep=use_ep)

    assert config.use_all2all_kernels is expected


def _atom_config(*, eplb: bool) -> SimpleNamespace:
    return SimpleNamespace(
        quant_config=SimpleNamespace(exclude_layers=[], quant_dtype=None),
        parallel_config=SimpleNamespace(data_parallel_size=8),
        moe_ep_flatten_tp_across_dp=False,
        eplb_enable=eplb,
    )


@pytest.mark.parametrize(
    "eplb, switch, expected",
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, True),
    ],
)
def test_eplb_overrides_the_local_replica_switch(monkeypatch, eplb, switch, expected):
    monkeypatch.setattr(topK_module.envs, "ATOM_FUSE_SHARED_EXPERT", switch)
    monkeypatch.setattr(
        topK_module,
        "get_current_atom_config",
        lambda: _atom_config(eplb=eplb),
    )

    enabled = is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(None)

    assert enabled is expected
