# SPDX-License-Identifier: MIT
"""V4.1 router and expert arithmetic, independent of dispatch/communication.

The oracle for `test_math.py`, and only that: serving routes through V4's
`FusedMoE`. It is the readable statement of the math those kernels compute,
pinned against the published model. It lived under `atom/model_ops/` while it
was written, which read like a second expert implementation a caller might
select; there is no such caller.
"""

import torch
import torch.nn.functional as F
from torch import nn

from atom.model_ops.utils import atom_parameter


class Router(nn.Module):
    def __init__(
        self, hidden_size, num_experts, topk, *, route_scale=1.5, gate_temperature=1.0
    ):
        super().__init__()
        if not 1 <= topk <= num_experts or gate_temperature <= 0:
            raise ValueError("Invalid router top-k or gate temperature")
        self.weight = atom_parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.bfloat16)
        )
        self.bias = atom_parameter(torch.empty(num_experts, dtype=torch.float32))
        self.bias_vl = atom_parameter(torch.empty(num_experts, dtype=torch.float32))
        self.topk = topk
        self.route_scale = route_scale
        self.gate_temperature = gate_temperature

    def forward(self, hidden, image_mask=None):
        scores = F.softplus(
            F.linear(hidden.float(), self.weight.float()) / self.gate_temperature
        ).sqrt()
        bias = self.bias
        if image_mask is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


def weighted_swiglu(
    gate, up, routing_weights=None, *, limit=10.0, dtype=torch.bfloat16
):
    """Weight FP32 SwiGLU BEFORE BF16 rounding and the following A8 QAT."""
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    activated = F.silu(gate) * up
    if routing_weights is not None:
        activated = activated * routing_weights
    return activated.to(dtype)


class Expert(nn.Module):
    """Compose native projections; the executor owns routing and token gathering."""

    def __init__(self, w1, w2, w3, *, swiglu_limit=10.0):
        super().__init__()
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden, routing_weights=None, *, output_dtype=None):
        activation = weighted_swiglu(
            self.w1(hidden),
            self.w3(hidden),
            routing_weights,
            limit=self.swiglu_limit,
            dtype=hidden.dtype,
        )
        if output_dtype is not None:
            return self.w2(activation, otype=output_dtype)
        return self.w2(activation)
