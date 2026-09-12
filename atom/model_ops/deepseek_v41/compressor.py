# SPDX-License-Identifier: MIT
"""CSA2 non-overlapping latent pooling; caller owns the incomplete group."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from atom.model_ops.layernorm import RMSNorm


@dataclass(frozen=True)
class CompressorTail:
    values: torch.Tensor
    scores: torch.Tensor


class Compressor(nn.Module):
    def __init__(self, hidden_size, head_dim, ratio, eps):
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError("CSA2 compressor ratio must be 1 or 2")
        self.ratio = ratio
        self.wkv = nn.Linear(hidden_size, head_dim, bias=False, dtype=torch.bfloat16)
        self.norm = RMSNorm(head_dim, eps)
        if ratio > 1:
            self.wgate = nn.Linear(
                hidden_size, head_dim, bias=False, dtype=torch.bfloat16
            )
            self.register_buffer("pool_weight", None, persistent=False)
            self.register_buffer("gate_weight", None, persistent=False)

    def process_weights_after_loading(self):
        if self.ratio > 1:
            self.pool_weight = self.wkv.weight.float()
            self.gate_weight = self.wgate.weight.float()

    def forward(self, x, start_position, tail=None):
        if self.ratio == 1:
            return self.norm(self.wkv(x)), None
        if self.pool_weight is None or self.gate_weight is None:
            raise RuntimeError("Compressor weights must be processed after loading")
        values = F.linear(x.float(), self.pool_weight)
        scores = F.linear(x.float(), self.gate_weight)
        remainder = start_position % self.ratio
        if remainder:
            expected = (x.shape[0], remainder, values.shape[-1])
            if (
                tail is None
                or tail.values.shape != expected
                or tail.scores.shape != expected
            ):
                raise ValueError(
                    "Incomplete compressor group is missing or has the wrong shape"
                )
            values = torch.cat((tail.values, values), dim=1)
            scores = torch.cat((tail.scores, scores), dim=1)
        elif tail is not None:
            raise ValueError("Unexpected compressor tail at a group boundary")
        cutoff = values.shape[1] // self.ratio * self.ratio
        next_tail = None
        if cutoff < values.shape[1]:
            next_tail = CompressorTail(
                values[:, cutoff:].clone(), scores[:, cutoff:].clone()
            )
        if cutoff == 0:
            return None, next_tail
        grouped_values = values[:, :cutoff].unflatten(1, (-1, self.ratio))
        grouped_scores = scores[:, :cutoff].unflatten(1, (-1, self.ratio))
        latent = (grouped_values * grouped_scores.softmax(dim=2)).sum(dim=2)
        return self.norm(latent.to(x.dtype)), next_tail
