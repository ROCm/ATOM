# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py ExpertLocationMetadata

import pytest

torch = pytest.importorskip("torch")

# Load atom.config fully first to avoid the mainline circular import; skip the
# whole module if the full atom import env (aiter/triton) is unavailable.
try:
    import atom.config  # noqa: F401
    from atom.model_ops.eplb import ExpertLocationMetadata
except Exception as _e:  # aiter/triton absent under bare non-GPU pytest
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)


def test_from_rebalance_result_pads_and_derives_rank0():
    # 1 layer, ep=2, 4 physical, 3 logical. logical0 replicated at slots 0 & 3.
    p2l = torch.tensor([[0, 1, 2, 0]], dtype=torch.int32)
    logcnt = torch.tensor([[2, 1, 1]], dtype=torch.int32)
    l2p_var = torch.tensor([[[0, 3], [1, -1], [2, -1]]], dtype=torch.int32)  # cur=2

    meta = ExpertLocationMetadata.from_rebalance_result(
        physical_to_logical_map=p2l,
        logical_to_physical_map=l2p_var,
        logical_replica_count=logcnt,
        ep_size=2,
        ep_rank=0,
        max_num_replicas=2,
    )
    assert meta.num_local_physical_experts == 2
    assert meta.logical_to_physical_map.shape == (1, 3, 2)
    # rank0 owns physical slots [0,1].
    assert meta.expert_map[0].tolist() == [0, 1, -1, -1]
    # dispatch: logical0->local slot0, logical1->local slot1, logical2->remote slot2
    assert meta.logical_to_rank_dispatch_physical_map[0].tolist() == [0, 1, 2]


def test_from_rebalance_result_rank1_dispatch():
    p2l = torch.tensor([[0, 1, 2, 0]], dtype=torch.int32)
    logcnt = torch.tensor([[2, 1, 1]], dtype=torch.int32)
    l2p_var = torch.tensor([[[0, 3], [1, -1], [2, -1]]], dtype=torch.int32)
    meta = ExpertLocationMetadata.from_rebalance_result(
        physical_to_logical_map=p2l,
        logical_to_physical_map=l2p_var,
        logical_replica_count=logcnt,
        ep_size=2,
        ep_rank=1,
        max_num_replicas=2,
    )
    assert meta.expert_map[0].tolist() == [-1, -1, 0, 1]
    # rank1 owns slots [2,3]: logical0->local replica at slot3, logical1->remote slot1,
    # logical2->local slot2
    assert meta.logical_to_rank_dispatch_physical_map[0].tolist() == [3, 1, 2]


def test_pad_widens_to_max_num_replicas():
    p2l = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    logcnt = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32)
    l2p_var = torch.tensor([[[0], [1], [2], [3]]], dtype=torch.int32)  # cur=1
    meta = ExpertLocationMetadata.from_rebalance_result(
        physical_to_logical_map=p2l,
        logical_to_physical_map=l2p_var,
        logical_replica_count=logcnt,
        ep_size=2,
        ep_rank=0,
        max_num_replicas=3,  # pad 1 -> 3
    )
    assert meta.logical_to_physical_map.shape == (1, 4, 3)
    # tail padded with -1
    assert meta.logical_to_physical_map[0, 0].tolist() == [0, -1, -1]


def test_from_trivial_invariants():
    meta = ExpertLocationMetadata.from_trivial(
        num_layers=3,
        num_logical_experts=8,
        num_physical_experts=12,
        ep_size=4,
        ep_rank=0,
    )
    assert meta.num_physical_experts == 12
    assert meta.max_num_replicas == 5  # num_redundant(4) + 1
    assert meta.physical_to_logical_map.shape == (3, 12)
    # every physical slot maps to a valid logical (no -1)
    assert int(meta.physical_to_logical_map.min()) >= 0
    assert int(meta.physical_to_logical_map.max()) < 8
    # slot i → logical i % 8
    assert meta.physical_to_logical_map[0].tolist() == [i % 8 for i in range(12)]
    # replica counts sum to num_physical per layer
    assert meta.logical_replica_count.sum(dim=1).tolist() == [12, 12, 12]
    # first 4 experts get 2 replicas, last 4 get 1
    assert meta.logical_replica_count[0].tolist() == [2, 2, 2, 2, 1, 1, 1, 1]
    # l2p consistent with p2l
    for layer in range(3):
        for e in range(8):
            cnt = int(meta.logical_replica_count[layer, e].item())
            reps = meta.logical_to_physical_map[layer, e, :cnt]
            assert torch.all(
                meta.physical_to_logical_map[layer, reps.to(torch.int64)] == e
            )


def test_update_is_inplace_and_correct():
    live = ExpertLocationMetadata.from_trivial(
        num_layers=2,
        num_logical_experts=6,
        num_physical_experts=8,
        ep_size=2,
        ep_rank=0,
    )
    # A different placement (swap logicals via a skewed load) for layer 0.
    skew = torch.tensor([[100, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 100]], dtype=torch.int32)
    from atom.model_ops.eplb import rebalance_experts

    p2l, l2p, cnt = rebalance_experts(
        skew,
        num_physical=8,
        num_groups=1,
        num_nodes=1,
        num_gpus=2,
        enable_hierarchical=False,
    )
    new = ExpertLocationMetadata.from_rebalance_result(
        physical_to_logical_map=p2l,
        logical_to_physical_map=l2p,
        logical_replica_count=cnt,
        ep_size=2,
        ep_rank=0,
        max_num_replicas=3,
    )
    # capture addresses of all live tensors
    ptrs = {
        name: getattr(live, name).data_ptr()
        for name in (
            "physical_to_logical_map",
            "logical_to_physical_map",
            "logical_replica_count",
            "expert_map",
            "logical_to_rank_dispatch_physical_map",
        )
    }
    layer0_before = live.physical_to_logical_map[0].clone()

    live.update(new, [0])

    # addresses unchanged (in-place copy_)
    for name, p in ptrs.items():
        assert getattr(live, name).data_ptr() == p, f"{name} address changed"
    # layer 0 now equals new's layer 0 for every map
    assert torch.equal(live.physical_to_logical_map[0], new.physical_to_logical_map[0])
    assert torch.equal(live.logical_to_physical_map[0], new.logical_to_physical_map[0])
    assert torch.equal(live.logical_replica_count[0], new.logical_replica_count[0])
    assert torch.equal(
        live.logical_to_rank_dispatch_physical_map[0],
        new.logical_to_rank_dispatch_physical_map[0],
    )
    # layer 1 untouched (still uniform placement, differs from new layer1 generally)
    assert not torch.equal(live.physical_to_logical_map[0], layer0_before)


def test_update_rejects_budget_mismatch():
    a = ExpertLocationMetadata.from_trivial(
        num_layers=1,
        num_logical_experts=4,
        num_physical_experts=4,
        ep_size=2,
        ep_rank=0,
    )
    b = ExpertLocationMetadata.from_trivial(
        num_layers=1,
        num_logical_experts=4,
        num_physical_experts=8,
        ep_size=2,
        ep_rank=0,
    )
    with pytest.raises(AssertionError):
        a.update(b, [0])
