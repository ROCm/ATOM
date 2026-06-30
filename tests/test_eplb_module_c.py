# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-C rebalancing algorithms)

import pytest

torch = pytest.importorskip("torch")

from atom.model_ops.eplb import (
    _build_logical_to_physical_map,
    balanced_packing,
    rebalance_experts,
    replicate_experts,
)


def test_balanced_packing_equal_cardinality():
    weight = torch.tensor([[9, 8, 7, 6]], dtype=torch.int32)
    pack_idx, rank_in_pack = balanced_packing(weight, num_packs=2)
    assert pack_idx.shape == (1, 4)
    assert rank_in_pack.shape == (1, 4)

    counts = [(pack_idx[0] == p).sum().item() for p in range(2)]
    assert counts == [2, 2]
    assert sorted(rank_in_pack[0, pack_idx[0] == 0].tolist()) == [0, 1]
    assert sorted(rank_in_pack[0, pack_idx[0] == 1].tolist()) == [0, 1]


def test_replicate_experts_conservation_and_rank():
    weight = torch.tensor([[10, 1, 1]], dtype=torch.int32)
    phy2log, phyrank, logcnt = replicate_experts(weight, num_physical=5)
    assert phy2log.shape == (1, 5)
    assert phyrank.shape == (1, 5)
    assert logcnt.shape == (1, 3)
    assert int(logcnt.sum().item()) == 5

    for logical_id in range(3):
        mask = phy2log[0] == logical_id
        count = int(mask.sum().item())
        assert count == int(logcnt[0, logical_id].item())
        if count > 0:
            ranks = sorted(phyrank[0, mask].tolist())
            assert ranks == list(range(count))


def test_build_logical_to_physical_map_uses_phyrank():
    p2l = torch.tensor([[0, 0, 0, 1]], dtype=torch.int32)
    phyrank = torch.tensor([[2, 0, 1, 0]], dtype=torch.int32)
    logcnt = torch.tensor([[3, 1]], dtype=torch.int32)
    l2p = _build_logical_to_physical_map(p2l, phyrank, logcnt)
    assert l2p.shape == (1, 2, 3)
    assert l2p[0, 0].tolist() == [1, 2, 0]
    assert l2p[0, 1].tolist() == [3, -1, -1]


def test_build_logical_to_physical_map_rejects_rank_out_of_logical_range():
    p2l = torch.tensor([[0, 1, 1]], dtype=torch.int32)
    phyrank = torch.tensor([[0, 2, 0]], dtype=torch.int32)
    logcnt = torch.tensor([[1, 2]], dtype=torch.int32)
    with pytest.raises(AssertionError, match="physical rank out of logical expert range"):
        _build_logical_to_physical_map(p2l, phyrank, logcnt)


def test_rebalance_experts_global_invariants():
    weight = torch.tensor([[8, 6, 2, 1], [1, 2, 6, 8]], dtype=torch.int32)
    p2l, l2p, logcnt = rebalance_experts(
        weight,
        num_physical=8,
        num_groups=1,
        num_nodes=1,
        num_gpus=4,
        enable_hierarchical=False,
    )
    assert p2l.shape == (2, 8)
    assert logcnt.shape == (2, 4)
    assert l2p.shape[0] == 2 and l2p.shape[1] == 4
    assert l2p.shape[2] == int(logcnt.max().item())

    for layer in range(2):
        for logical_id in range(4):
            expected = int(logcnt[layer, logical_id].item())
            physical_ids = l2p[layer, logical_id]
            valid = physical_ids[physical_ids >= 0]
            assert valid.numel() == expected
            assert torch.all(p2l[layer, valid.to(torch.int64)] == logical_id)


@pytest.mark.parametrize(
    "kwargs,err",
    [
        (
            dict(num_physical=8, num_groups=3, num_nodes=1, num_gpus=4, enable_hierarchical=True),
            "num_logical must be divisible by num_groups",
        ),
        (
            dict(num_physical=8, num_groups=4, num_nodes=3, num_gpus=6, enable_hierarchical=True),
            "num_groups must be divisible by num_nodes",
        ),
        (
            dict(num_physical=8, num_groups=4, num_nodes=2, num_gpus=3, enable_hierarchical=True),
            "num_gpus must be divisible by num_nodes",
        ),
        (
            dict(num_physical=10, num_groups=4, num_nodes=2, num_gpus=4, enable_hierarchical=True),
            "num_physical must be divisible by num_gpus",
        ),
    ],
)
def test_rebalance_experts_constraints(kwargs, err):
    weight = torch.ones((1, 8), dtype=torch.int32)
    with pytest.raises(AssertionError, match=err):
        rebalance_experts(weight, **kwargs)


def test_rebalance_experts_hierarchical_invariants():
    weight = torch.tensor([[100, 80, 1, 1, 60, 40, 1, 1]], dtype=torch.int32)
    p2l, l2p, logcnt = rebalance_experts(
        weight,
        num_physical=8,
        num_groups=4,
        num_nodes=2,
        num_gpus=4,
        enable_hierarchical=True,
    )
    assert p2l.shape == (1, 8)
    assert logcnt.shape == (1, 8)
    assert l2p.shape[0] == 1 and l2p.shape[1] == 8
    assert int(logcnt.sum().item()) == 8
    assert sorted(p2l[0].tolist()) == list(range(8))
