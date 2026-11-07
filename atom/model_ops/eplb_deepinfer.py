# SPDX-License-Identifier: Apache-2.0
# Adapted from DeepSeek's DeepInfer expert replication policy:
# https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/eplb/policy.py
# Modifications: extract replica selection, validate inputs, and return ATOM's
# physical replica ranks instead of constructing placements/inverse mappings.
# The upstream license is in third_party/licenses/DeepInfer.txt.

"""Select redundant experts using DeepInfer's historical bottleneck objective."""

from __future__ import annotations

import torch


def replicate_experts_deepinfer(
    tokens_per_expert: torch.Tensor, num_physical: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate replicas to minimize the sum of per-step peak replica loads.

    Each layer starts with one copy of every logical expert. For each redundant
    slot, the upstream greedy policy chooses the expert whose additional copy
    minimizes ``sum_step(max_expert(tokens / replica_count))``. Keeping the
    historical step dimension allows a different choice from total-load top-k.

    Args:
        tokens_per_expert: Non-negative, finite token counts with shape
            [num_steps, num_layers, num_logical].
        num_physical: Total physical experts per layer, at least num_logical.

    Returns:
        physical_to_logical: [num_layers, num_physical] int32. Original experts
            occupy the first num_logical slots; selected replicas follow.
        physical_rank: [num_layers, num_physical] int32, the zero-based replica
            index of each slot within its logical expert.
        logical_replica_count: [num_layers, num_logical] int32.

        All outputs stay on the input device. Selection does not assign experts
        to GPUs or modify the input history.
    """
    if tokens_per_expert.ndim != 3:
        raise ValueError(
            "tokens_per_expert must be rank-3 [num_steps, num_layers, num_logical]"
        )
    if any(size == 0 for size in tokens_per_expert.shape):
        raise ValueError("tokens_per_expert dimensions must all be positive")
    if isinstance(num_physical, bool) or not isinstance(num_physical, int):
        raise TypeError("num_physical must be an integer")
    _, num_layers, num_logical = tokens_per_expert.shape
    if num_physical < num_logical:
        raise ValueError("num_physical must be >= num_logical for replication")

    device = tokens_per_expert.device
    physical_to_logical = torch.empty(
        (num_layers, num_physical), dtype=torch.int32, device=device
    )
    physical_rank = torch.zeros_like(physical_to_logical)
    logical_count = torch.ones(
        (num_layers, num_logical), dtype=torch.int32, device=device
    )
    layers = torch.arange(num_layers, dtype=torch.int32, device=device)
    logical_experts = torch.arange(num_logical, dtype=torch.int32, device=device)
    physical_to_logical[:, :num_logical] = logical_experts

    # Preserve DeepInfer's perturbation and dtype promotion, including its tie
    # behavior. The candidate changes the maximum only on steps where that
    # candidate currently has the largest per-replica load.
    tokens_all_diff = tokens_per_expert + logical_experts * 1e-4
    for physical_index in range(num_logical, num_physical):
        score = tokens_all_diff / logical_count
        score_added = tokens_per_expert / (logical_count + 1)
        values, indices = score.max(-1, keepdim=True)
        values = values.expand_as(score).contiguous()
        score.scatter_(-1, indices, score_added.gather(-1, indices))
        values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
        selected = values.sum(0).argmin(-1)

        physical_to_logical[:, physical_index] = selected
        physical_rank[:, physical_index] = logical_count[layers, selected]
        logical_count[layers, selected] += 1

    return physical_to_logical, physical_rank, logical_count
