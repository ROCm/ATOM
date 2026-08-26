# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the cross-node migration guard (M2.2).

The guard decides whether a rearrange is worth its inter-node traffic. It reads
a count derived from the global maps rather than from planning; the first test
here is what keeps that shortcut honest.
"""

import torch
from aiter_stub import stubbed_aiter

# Everything under test is pure index arithmetic. Stub AITER rather than skip
# without it, so a CPU box runs these for real instead of reporting green.
with stubbed_aiter():
    import atom.config  # noqa: F401  (initialize before model_ops' __init__ chain)
    from atom.model_ops.eplb import (
        MigrationCost,
        _plan_single_layer_migration,
        count_migration_transfers,
    )

# DSv4-Pro at the WideEP target: 384 logical experts, num_redundant = ep_size.
NUM_LOGICAL = 384
NUM_PHYSICAL = 400
WORLD_SIZE = 16
NUM_LOCAL = NUM_PHYSICAL // WORLD_SIZE
GPU_PER_NODE = 8  # 2 nodes


def _make_p2l(num_layers: int, seed: int) -> torch.Tensor:
    """A valid placement: every logical present, the spare slots as replicas."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(num_layers):
        base = torch.randperm(NUM_LOGICAL, generator=g)
        spare = torch.randint(
            0, NUM_LOGICAL, (NUM_PHYSICAL - NUM_LOGICAL,), generator=g
        )
        rows.append(torch.cat([base, spare]))
    return torch.stack(rows)


def _planner_transfers(old_p2l, new_p2l, *, gpu_per_node):
    """Replay the real per-rank planner and total up its P2P receives."""
    total = cross = 0
    num_layers = old_p2l.shape[0]
    for layer in range(num_layers):
        for rank in range(WORLD_SIZE):
            _, _, recv_actions, _ = _plan_single_layer_migration(
                old_p2l_layer=old_p2l[layer],
                new_p2l_layer=new_p2l[layer],
                num_local_physical_experts=NUM_LOCAL,
                num_gpu_per_node=gpu_per_node,
                rank=rank,
                world_size=WORLD_SIZE,
            )
            for action in recv_actions:
                total += 1
                if action.peer_rank // gpu_per_node != rank // gpu_per_node:
                    cross += 1
    return total, cross


class TestCountMatchesPlanner:
    def test_migration_cost_matches_planner(self):
        """The shortcut must equal what the planner actually issues.

        count_migration_transfers derives the answer once from the global maps;
        the planner derives it per rank. They encode the same recv rule and the
        same source selection, and this is the only thing stopping them from
        drifting apart.
        """
        old = _make_p2l(4, seed=0)
        new = _make_p2l(4, seed=1)
        assert count_migration_transfers(
            old_p2l=old,
            new_p2l=new,
            num_local_physical_experts=NUM_LOCAL,
            num_gpu_per_node=GPU_PER_NODE,
        ) == _planner_transfers(old, new, gpu_per_node=GPU_PER_NODE)

    def test_matches_planner_on_one_node(self):
        old = _make_p2l(2, seed=2)
        new = _make_p2l(2, seed=3)
        assert count_migration_transfers(
            old_p2l=old,
            new_p2l=new,
            num_local_physical_experts=NUM_LOCAL,
            num_gpu_per_node=WORLD_SIZE,
        ) == _planner_transfers(old, new, gpu_per_node=WORLD_SIZE)


class TestCounts:
    def test_no_change_moves_nothing(self):
        p2l = _make_p2l(3, seed=4)
        assert count_migration_transfers(
            old_p2l=p2l,
            new_p2l=p2l.clone(),
            num_local_physical_experts=NUM_LOCAL,
            num_gpu_per_node=GPU_PER_NODE,
        ) == (0, 0)

    def test_single_node_never_crosses(self):
        old = _make_p2l(3, seed=5)
        new = _make_p2l(3, seed=6)
        transfers, cross = count_migration_transfers(
            old_p2l=old,
            new_p2l=new,
            num_local_physical_experts=NUM_LOCAL,
            num_gpu_per_node=WORLD_SIZE,
        )
        assert transfers > 0
        assert cross == 0

    def test_same_node_holder_keeps_it_local(self):
        # 4 ranks, 2 per node, 1 slot each. Rank 1 wants expert 0; rank 0 (same
        # node) holds it, so the node-aware selector must not reach off-node.
        old = torch.tensor([[0, 9, 0, 8]])
        new = torch.tensor([[0, 0, 0, 8]])
        transfers, cross = count_migration_transfers(
            old_p2l=old,
            new_p2l=new,
            num_local_physical_experts=1,
            num_gpu_per_node=2,
        )
        assert (transfers, cross) == (1, 0)

    def test_only_off_node_holder_forces_a_crossing(self):
        # Same shape, but expert 0 lives only on node 0 and node 1 needs it.
        old = torch.tensor([[0, 9, 7, 8]])
        new = torch.tensor([[0, 9, 0, 8]])
        transfers, cross = count_migration_transfers(
            old_p2l=old,
            new_p2l=new,
            num_local_physical_experts=1,
            num_gpu_per_node=2,
        )
        assert (transfers, cross) == (1, 1)

    def test_replicas_on_one_rank_cost_one_transfer(self):
        # Rank 1 gains two slots of expert 0; the planner sends it once and
        # copies locally for the second (the free-rider path).
        old = torch.tensor([[0, 0, 9, 8, 7, 6]])
        new = torch.tensor([[0, 0, 0, 0, 7, 6]])
        transfers, cross = count_migration_transfers(
            old_p2l=old,
            new_p2l=new,
            num_local_physical_experts=2,
            num_gpu_per_node=1,
        )
        assert (transfers, cross) == (1, 1)


class TestStallEstimate:
    def test_only_cross_node_bytes_count(self):
        cost = MigrationCost(
            transfers=100, cross_node_transfers=10, bytes_per_expert=10**8
        )
        assert cost.total_bytes == 10**10
        assert cost.cross_node_bytes == 10**9
        # 1 GB at 50 GB/s = 20 ms.
        assert cost.cross_node_stall_ms(50.0) == 20.0

    def test_nothing_to_move_is_free(self):
        cost = MigrationCost(
            transfers=5, cross_node_transfers=0, bytes_per_expert=10**8
        )
        assert cost.cross_node_stall_ms(50.0) == 0.0
