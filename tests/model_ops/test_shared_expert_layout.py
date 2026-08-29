from types import SimpleNamespace

import torch

from atom.model_ops.fused_moe.config import FusedMoEConfig
from atom.model_ops.fused_moe.expert_layout import MoEExpertLayout, SharedExpertMode
from atom.model_ops.fused_moe.shared_expert_dispatch import _remap_torch


def test_backend_config_widens_every_rank_block_by_the_shared_slot():
    parallel = SimpleNamespace(
        tp_size=1,
        dp_size=8,
        ep_size=8,
        tp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=True,
        use_mori_kernels=True,
    )
    config = FusedMoEConfig(
        num_experts=264,
        experts_per_token=9,
        hidden_dim=7168,
        num_local_experts=33,
        moe_parallel_config=parallel,
        expert_layout=MoEExpertLayout.make(
            num_routed=256,
            num_fused_shared_experts=1,
            num_configured_redundant=0,
            ep_size=8,
            use_all2all=True,
            eplb_enabled=False,
        ),
    )

    assert config.expert_layout.mode is SharedExpertMode.LOCAL_REPLICA
    assert config.num_local_experts == 33
    assert config.expert_layout.num_routed_physical == 256
    assert config.expert_layout.num_physical == 264
    assert config.experts_per_token == 9


def test_eplb_promotes_shared_to_routed_layout():
    layout = MoEExpertLayout.make(
        num_routed=256,
        num_fused_shared_experts=1,
        num_configured_redundant=32,
        ep_size=8,
        use_all2all=True,
        eplb_enabled=True,
    )

    assert layout.mode is SharedExpertMode.EPLB_ROUTED
    assert layout.num_logical == 257
    assert layout.num_physical == 296
    assert layout.physical_per_rank == 37


def test_remap_preserves_routed_owner_and_local_slot():
    """The point of the widening: MoRI must still resolve the same rank."""
    routed_per_rank, num_fused_shared_experts = 4, 1
    stride = routed_per_rank + num_fused_shared_experts
    ids = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    weights = torch.zeros((1, 8))

    _, out_ids = _remap_torch(
        weights,
        ids,
        8,
        routed_per_rank,
        routed_per_rank,
        num_fused_shared_experts,
        0.5,
    )
    dispatch = out_ids[0, :8]

    assert dispatch.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    for physical, disp in zip(range(8), dispatch.tolist()):
        assert disp // stride == physical // routed_per_rank
        assert disp % stride == physical % routed_per_rank


def test_remap_appends_shared_without_rewriting_sentinels():
    ids = torch.tensor([[0, 4], [3, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.0]])

    # ep_rank 1 of 2, 4 routed slots each -> shared base 1*5 + 4 = 9.
    out_weights, out_ids = _remap_torch(weights, ids, 8, 4, 9, 1, 0.5)

    assert out_ids.tolist() == [[0, 5, 9], [3, -1, 9]]
    assert torch.equal(out_weights[:, :2], weights)
    assert out_weights[:, 2].tolist() == [0.5, 0.5]
