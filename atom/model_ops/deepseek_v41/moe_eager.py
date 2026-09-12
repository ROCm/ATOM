# SPDX-License-Identifier: MIT
"""Eager expert dispatch for the W4A8 correctness baseline.

Tokens are replicated across the tensor-parallel group; each rank computes its
owned experts, then reduces FP32 outputs. This intentionally retains a host
count synchronization. Fused/MORI dispatch requires a separate precision gate.
"""

import torch


def execute_experts(hidden, weights, indices, experts, num_experts, group):
    output = torch.zeros_like(hidden, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=num_experts).tolist()
    for name, expert in experts.items():
        expert_id = int(name)
        if counts[expert_id]:
            rows, choices = torch.where(indices == expert_id)
            output[rows] += expert(hidden[rows], weights[rows, choices, None])
    if group.world_size > 1:
        output = group.all_reduce(output, ca_fp8_quant=False)
    return output
