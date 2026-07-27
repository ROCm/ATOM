# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Expert slot layout for FusedMoE parameters.

Where each expert's weights live inside the fused ``w13_weight`` / ``w2_weight``
buffers, and which of those slots a given rank owns.  Both the weight loader
and the batched staging path need this, and they must agree byte for byte, so
it lives in one place instead of being re-derived at each call site.

Deliberately pure ``torch`` — no AITER, no ``atom.config`` — so the model
loader's unit tests can exercise the real layout on a plain CPU runner.
"""

import torch

# Which dimension of a single expert slot a shard is sharded along, before the
# `is_transposed` flip.  `w1`/`w3` are the two halves of the gate_up projection
# (column-parallel, output dim); `w2` is the down projection (row-parallel).
_SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}


def determine_expert_map(
    ep_size: int, ep_rank: int, global_num_experts: int
) -> tuple[int, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size (int): The size of the expert parallel group
        global_num_experts (int): The total number of experts in the model.

    Returns:
        Tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create a expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts] = (
            torch.arange(0, local_num_experts, dtype=torch.int32)
        )
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = global_num_experts - ep_rank * local_num_experts

        expert_map[-local_num_experts:] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    return (local_num_experts, expert_map)


def count_local_base_experts(
    expert_map: torch.Tensor | None,
    global_num_experts: int,
    num_redundant_experts: int,
    local_num_experts: int,
) -> int:
    """Number of local slots that receive a routed expert from the checkpoint.

    Excludes EPLB redundant replicas, which `fill_redundant` populates after
    loading.  Fused shared experts are *not* excluded here — they sit after the
    routed slots and callers that care subtract `num_fused_shared_experts`
    themselves.

    `determine_expert_map` assigns a rank a contiguous run of global ids
    starting at local index 0, and redundant replicas extend the global range
    past `global_num_experts - num_redundant_experts`, so the returned count is
    also the length of the local slot *prefix* holding routed base experts.
    """
    if expert_map is None:
        return local_num_experts
    num_logical = global_num_experts - num_redundant_experts
    return int((expert_map[:num_logical] != -1).sum().item())


def physical_expert_id(
    expert_id: int,
    global_num_experts: int,
    num_redundant_experts: int,
    num_fused_shared_experts: int,
) -> int:
    """Translate a checkpoint-side expert id into a physical slot index.

    Checkpoints (and `FusedMoE.make_expert_params_mapping`) number a fused
    shared expert immediately after the *logical* routed experts, because that
    is all the HF config knows about.  `expert_map` is indexed by *physical*
    slot, and EPLB inserts `num_redundant_experts` replicas between the two —
    so without this translation a shared expert would be written over a
    redundant replica while its own slot kept its init value.

    A no-op unless EPLB is configured with redundant experts.
    """
    num_logical = global_num_experts - num_redundant_experts
    if num_fused_shared_experts and expert_id >= num_logical:
        return global_num_experts + (expert_id - num_logical)
    return expert_id


def expert_shard_dim(shard_id: str, is_transposed: bool = False) -> int:
    """Dimension of a single expert slot that `shard_id` is sharded along."""
    dim = _SHARD_ID_TO_SHARDED_DIM[shard_id]
    return int(not dim) if is_transposed else dim


def expert_shard_view(
    slot: torch.Tensor, shard_id: str, shard_dim: int
) -> torch.Tensor:
    """The part of one expert slot that `shard_id` owns.

    `w13` stacks the gate and up projections along `shard_dim`, so `w1` owns
    the first half and `w3` the second; `w2` owns the whole slot.
    """
    if shard_id == "w2":
        return slot
    assert shard_id in ("w1", "w3"), f"unexpected shard_id {shard_id!r}"
    half = slot.shape[shard_dim] // 2
    return slot.narrow(shard_dim, 0 if shard_id == "w1" else half, half)


def expert_region(
    tensor: torch.Tensor,
    local_expert_id: int,
    shard_id: str,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Sub-view of a param-shaped `tensor` owned by one (expert, shard) arrival.

    The region is the *full* half-slot, not the width a particular arrival
    copies: `_load_w13` / `_load_w2` narrow further to `loaded_weight`'s size
    when the parameter is padded (MXFP4 alignment), and a flush does not have
    `loaded_weight` to hand.  The difference is the padding tail, which is zero
    on both sides — staging buffers are zero-initialised and MXFP4 parameters
    are zeroed in `create_weights` — so copying the full half matches what a
    whole-parameter flush would have written.  Ownership is therefore tracked
    at (slot, shard) granularity, not per byte.
    """
    return expert_shard_view(
        tensor[local_expert_id], shard_id, expert_shard_dim(shard_id, is_transposed)
    )
