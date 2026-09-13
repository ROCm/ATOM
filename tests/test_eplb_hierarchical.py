# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hierarchical (multi-node) EPLB placement.

`rebalance_experts(enable_hierarchical=True, num_nodes>1)` has never executed in
production: `EPLBManager._nnodes` is hardcoded to 1, so every rebalance so far
took the flat branch. These tests cover the node-aware branch before WideEP
starts depending on it.

CPU-only; no GPU or distributed setup.
"""

import pytest
import torch

from atom.model_ops.eplb import rebalance_experts

# DSv4-Pro-shaped: 384 logical experts in 8 groups, EP16 over 2 nodes,
# num_redundant = ep_size (the TRT "num_slots = experts + EP size" tier).
DSV4 = {
    "num_logical": 384,
    "num_groups": 8,
    "num_physical": 400,
    "num_gpus": 16,
    "num_nodes": 2,
}


def _weights(num_layers, num_logical, *, skew=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand((num_layers, num_logical), generator=g) + 0.1
    if skew is not None:
        w[:, :skew] *= 100.0
    return w


def _run(cfg, *, num_layers=2, weight=None, hierarchical=True):
    w = _weights(num_layers, cfg["num_logical"]) if weight is None else weight
    return rebalance_experts(
        w,
        num_physical=cfg["num_physical"],
        num_groups=cfg["num_groups"],
        num_nodes=cfg["num_nodes"],
        num_gpus=cfg["num_gpus"],
        enable_hierarchical=hierarchical,
    )


def _assert_placement_valid(p2l, logcnt, cfg):
    """Invariants that any placement must satisfy."""
    num_layers = p2l.shape[0]
    num_logical = cfg["num_logical"]
    assert p2l.shape == (num_layers, cfg["num_physical"])
    assert logcnt.shape == (num_layers, num_logical)

    for layer in range(num_layers):
        ids = p2l[layer]
        assert int(ids.min()) >= 0, "unfilled physical slot"
        assert int(ids.max()) < num_logical, "physical slot maps outside logical space"
        # Every logical expert is placed, or its tokens have nowhere to go.
        counted = torch.bincount(ids.to(torch.int64), minlength=num_logical)
        assert int(counted.min()) >= 1, "logical expert with no physical slot"
        assert int(counted.sum()) == cfg["num_physical"]
        # logcnt must agree with raw placement p2l.
        assert torch.equal(counted.to(torch.int32), logcnt[layer])


class TestHierarchicalPlacement:
    def test_dsv4_ep16_two_nodes(self):
        p2l, l2p, logcnt = _run(DSV4)
        _assert_placement_valid(p2l, logcnt, DSV4)
        assert l2p.shape[:2] == (2, DSV4["num_logical"])

    def test_group_stays_on_one_node(self):
        """The whole point of the hierarchical path.

        A group's experts are co-routed, so splitting one across nodes turns
        intra-node XGMI traffic into RDMA traffic for every token hitting it.
        """
        cfg = DSV4
        p2l, _, _ = _run(cfg)
        group_size = cfg["num_logical"] // cfg["num_groups"]
        phy_per_node = cfg["num_physical"] // cfg["num_nodes"]

        for layer in range(p2l.shape[0]):
            for g in range(cfg["num_groups"]):
                members = set(range(g * group_size, (g + 1) * group_size))
                nodes = {
                    int(i) // phy_per_node
                    for i, e in enumerate(p2l[layer].tolist())
                    if e in members
                }
                assert len(nodes) == 1, f"group {g} split across nodes {nodes}"

    def test_each_node_owns_its_share_of_slots(self):
        cfg = DSV4
        p2l, _, _ = _run(cfg)
        phy_per_node = cfg["num_physical"] // cfg["num_nodes"]
        for layer in range(p2l.shape[0]):
            for n in range(cfg["num_nodes"]):
                lo, hi = n * phy_per_node, (n + 1) * phy_per_node
                assert int((p2l[layer][lo:hi] >= 0).sum()) == phy_per_node

    def test_skewed_load_keeps_groups_balanced(self):
        """balanced_packing must still hand each node groups_per_node groups.

        The implementation asserts that; a heavily skewed layer is the case
        most likely to break it.
        """
        cfg = DSV4
        w = _weights(2, cfg["num_logical"], skew=48)  # one whole group is hot
        p2l, _, logcnt = _run(cfg, weight=w)
        _assert_placement_valid(p2l, logcnt, cfg)

    def test_hot_experts_get_more_replicas(self):
        cfg = DSV4
        w = _weights(1, cfg["num_logical"], skew=48)
        _, _, logcnt = _run(cfg, num_layers=1, weight=w)
        hot = logcnt[0, :48].float().mean()
        cold = logcnt[0, 48:].float().mean()
        assert hot > cold, f"hot={hot} not replicated more than cold={cold}"


class TestHierarchicalVsFlat:
    def test_flat_and_hierarchical_both_valid(self):
        """Same inputs, both branches; only the placement should differ."""
        w = _weights(2, DSV4["num_logical"])
        flat_p2l, _, flat_cnt = _run(DSV4, weight=w, hierarchical=False)
        hier_p2l, _, hier_cnt = _run(DSV4, weight=w, hierarchical=True)
        _assert_placement_valid(flat_p2l, flat_cnt, DSV4)
        _assert_placement_valid(hier_p2l, hier_cnt, DSV4)

    def test_flat_branch_does_split_groups(self):
        """Pins the discriminating power of test_group_stays_on_one_node.

        Without this, a hierarchical branch that silently degraded into the
        flat one would still satisfy the co-location test if the flat placement
        happened to keep groups together. Measured: flat splits 16/16
        group-layer pairs, hierarchical splits 0/16.
        """
        cfg = DSV4
        w = _weights(2, cfg["num_logical"])
        p2l, _, _ = _run(cfg, weight=w, hierarchical=False)
        group_size = cfg["num_logical"] // cfg["num_groups"]
        phy_per_node = cfg["num_physical"] // cfg["num_nodes"]
        split = sum(
            len(
                {
                    i // phy_per_node
                    for i, e in enumerate(p2l[layer].tolist())
                    if g * group_size <= e < (g + 1) * group_size
                }
            )
            > 1
            for layer in range(p2l.shape[0])
            for g in range(cfg["num_groups"])
        )
        assert split > 0, "flat placement kept every group on one node by chance"

    def test_single_node_ignores_hierarchical_flag(self):
        cfg = dict(DSV4, num_nodes=1, num_gpus=8, num_physical=392)
        w = _weights(2, cfg["num_logical"])
        a = _run(cfg, weight=w, hierarchical=True)
        b = _run(cfg, weight=w, hierarchical=False)
        assert torch.equal(a[0], b[0]), "nnodes=1 must take the flat branch"
        assert torch.equal(a[1], b[1]), "logical_to_physical must match too"


class TestHierarchicalConstraints:
    @pytest.mark.parametrize(
        "override, match",
        [
            ({"num_groups": 5}, "num_logical must be divisible by num_groups"),
            ({"num_nodes": 3}, "num_groups must be divisible by num_nodes"),
            ({"num_gpus": 15}, "num_gpus must be divisible by num_nodes"),
            ({"num_physical": 401}, "num_physical must be divisible by num_gpus"),
        ],
    )
    def test_rejects_incoherent_topology(self, override, match):
        cfg = dict(DSV4, **override)
        with pytest.raises(AssertionError, match=match):
            _run(cfg)

    @pytest.mark.parametrize("num_nodes, num_gpus", [(2, 16), (4, 16), (8, 16)])
    def test_scales_to_more_nodes(self, num_nodes, num_gpus):
        cfg = dict(DSV4, num_nodes=num_nodes, num_gpus=num_gpus)
        p2l, _, logcnt = _run(cfg, num_layers=1)
        _assert_placement_valid(p2l, logcnt, cfg)
