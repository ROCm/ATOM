"""Fusing the shared expert into the all2all dispatch, CPU-only.

The whole design rests on one claim: a shared expert replicated per rank is an
ordinary logical expert, so the existing EPLB placement and dispatch remap can
carry it and nothing bespoke is needed. These pin that claim down.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.eplb import ExpertLocationMetadata
from atom.model_ops.moe import FusedMoE

NUM_ROUTED, EP_SIZE, NUM_SHARED = 64, 8, 1
ROUTED_PER_RANK = NUM_ROUTED // EP_SIZE  # 8
PER_RANK = ROUTED_PER_RANK + NUM_SHARED  # 9
NUM_LOGICAL = NUM_ROUTED + NUM_SHARED  # 65
NUM_PHYSICAL = PER_RANK * EP_SIZE  # 72


def _meta(ep_rank: int) -> ExpertLocationMetadata:
    return ExpertLocationMetadata.from_shared_replicated(
        num_layers=2,
        num_routed_experts=NUM_ROUTED,
        num_shared_experts=NUM_SHARED,
        ep_size=EP_SIZE,
        ep_rank=ep_rank,
    )


def test_placement_is_a_bijection_over_the_dispatch_space():
    meta = _meta(0)
    p2l = meta.physical_to_logical_map[0]

    assert p2l.numel() == NUM_PHYSICAL
    assert int(meta.num_logical_experts) == NUM_LOGICAL
    # Every slot is claimed, and every logical expert claims what it should.
    counts = torch.bincount(p2l.to(torch.int64), minlength=NUM_LOGICAL)
    assert torch.equal(counts[:NUM_ROUTED], torch.ones(NUM_ROUTED, dtype=counts.dtype))
    assert torch.all(counts[NUM_ROUTED:] == EP_SIZE)


def test_shared_expert_gets_one_replica_on_every_rank():
    meta = _meta(0)
    p2l = meta.physical_to_logical_map[0]
    shared_slots = (p2l == NUM_ROUTED).nonzero().flatten()

    assert shared_slots.numel() == EP_SIZE
    # Backends resolve the rank as slot // per_rank (MoRI: internode.hpp).
    assert [int(s) // PER_RANK for s in shared_slots] == list(range(EP_SIZE))
    # ...and it sits at the end of each rank's block.
    assert torch.all(shared_slots % PER_RANK == ROUTED_PER_RANK)


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_dispatch_resolves_the_shared_expert_to_this_rank(ep_rank):
    """No pinning needed: the local-preference rule already does this."""
    meta = _meta(ep_rank)
    dispatch = meta.logical_to_rank_dispatch_physical_map[0]

    shared_slot = int(dispatch[NUM_ROUTED])
    assert shared_slot == ep_rank * PER_RANK + ROUTED_PER_RANK
    assert shared_slot // PER_RANK == ep_rank
    # -1 means "forced remote"; a per-rank replica must never need it.
    assert shared_slot >= 0


@pytest.mark.parametrize("ep_rank", [0, 3, 7])
def test_routed_experts_keep_their_owner(ep_rank):
    meta = _meta(ep_rank)
    dispatch = meta.logical_to_rank_dispatch_physical_map[0]

    for e in range(NUM_ROUTED):
        slot = int(dispatch[e])
        assert slot // PER_RANK == e // ROUTED_PER_RANK, (e, slot)
        assert slot % PER_RANK < ROUTED_PER_RANK, (e, slot)


def test_every_layer_shares_the_same_placement():
    meta = _meta(2)
    assert torch.equal(
        meta.physical_to_logical_map[0], meta.physical_to_logical_map[1]
    )


def _layer(*, fused: bool, ep_rank: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        fuse_shared_into_dispatch=fused,
        num_fused_shared_experts=NUM_SHARED if fused else 0,
        num_logical_experts=NUM_LOGICAL if fused else NUM_ROUTED,
        routed_scaling_factor=2.0,
        ep_rank=ep_rank,
    )


def test_shared_column_carries_the_logical_id(monkeypatch):
    import atom.model_ops.moe as moe_module

    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: False
    )
    layer = _layer(fused=True)
    ids = torch.tensor([[0, 9], [3, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.0]])

    out_w, out_ids = FusedMoE.append_shared_logical_column(layer, weights, ids)

    # Logical, not physical: the remap turns it into this rank's slot.
    assert out_ids[:, -1].tolist() == [NUM_ROUTED, NUM_ROUTED]
    assert torch.equal(out_ids[:, :2], ids)
    assert torch.equal(out_w[:, :2], weights)
    assert out_w[:, -1].tolist() == [0.5, 0.5]  # 1 / routed_scaling_factor


def test_append_returns_weights_first():
    """The two tensors differ only by dtype, so a swapped return is silent."""
    layer = _layer(fused=True)
    out_w, out_ids = FusedMoE.append_shared_logical_column(
        layer,
        torch.zeros((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.int32),
    )
    assert out_w.dtype == torch.float32
    assert out_ids.dtype == torch.int32


def test_append_is_a_noop_without_fusion():
    layer = _layer(fused=False)
    ids = torch.tensor([[0, 9]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3]])

    out_w, out_ids = FusedMoE.append_shared_logical_column(layer, weights, ids)

    assert out_ids is ids and out_w is weights


def test_routed_logical_count_excludes_the_shared_expert():
    assert FusedMoE.num_routed_logical_experts.fget(_layer(fused=True)) == NUM_ROUTED
    assert FusedMoE.num_routed_logical_experts.fget(_layer(fused=False)) == NUM_ROUTED
