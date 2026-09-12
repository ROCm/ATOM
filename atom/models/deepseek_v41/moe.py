# SPDX-License-Identifier: MIT
"""V4.1 expert ownership and projection composition, with explicit W4A8 routing."""

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_ops.deepseek_v41.moe import Expert, Router
from atom.model_ops.deepseek_v41.moe_eager import execute_experts
from atom.model_ops.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from .layers import native_quant_config, reduce_output


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group = get_tp_group()
        self.num_experts = config.n_routed_experts
        self.gate = Router(
            config.hidden_size,
            self.num_experts,
            config.num_experts_per_tok,
            route_scale=config.routed_scaling_factor,
        )
        local_count = self.num_experts // self.group.world_size
        begin = self.group.rank_in_group * local_count
        self.experts = nn.ModuleDict(
            {
                str(expert_id): self._expert(config, routed=True)
                for expert_id in range(begin, begin + local_count)
            }
        )
        self.shared_experts = self._expert(config, routed=False)

    @staticmethod
    def _expert(config, *, routed):
        quant = native_quant_config(fp4=routed)
        up_type = ReplicatedLinear if routed else ColumnParallelLinear
        down_type = ReplicatedLinear if routed else RowParallelLinear
        dim, inter = config.hidden_size, config.moe_intermediate_size
        return Expert(
            up_type(dim, inter, quant_config=quant),
            down_type(inter, dim, quant_config=quant, reduce_results=False),
            up_type(dim, inter, quant_config=quant),
            swiglu_limit=config.swiglu_limit,
        )

    def forward(self, hidden, image_mask=None):
        flat = hidden.reshape(-1, hidden.shape[-1])
        weights, indices = self.gate(
            flat, None if image_mask is None else image_mask.flatten()
        )
        output = execute_experts(
            flat, weights, indices, self.experts, self.num_experts, self.group
        )
        # Upstream rounds the complete shared FFN output once. Its TP
        # partials must remain FP32 until after the reduction.
        shared = reduce_output(
            self.shared_experts(flat, output_dtype=torch.float32)
        ).to(hidden.dtype)
        return (output + shared).to(hidden.dtype).view_as(hidden)
