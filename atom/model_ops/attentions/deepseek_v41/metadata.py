# SPDX-License-Identifier: MIT
"""Request spans shared by CSA2 paging, compression and Engram staging."""

from dataclasses import dataclass, field

import torch

from atom.model_ops.attentions.deepseek_v41_state import AttentionStep


@dataclass(frozen=True)
class RequestSpan:
    request_id: int
    position: int
    offset: int
    length: int
    slot: int
    block_ids: tuple[int, ...]

    @property
    def end(self):
        return self.position + self.length

    @property
    def token_slice(self):
        return slice(self.offset, self.offset + self.length)


@dataclass
class BatchStep:
    requests: tuple[RequestSpan, ...]
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    slots: torch.Tensor
    batch_ids: torch.Tensor
    block_tables: torch.Tensor
    request_steps: tuple[AttentionStep, ...]
    selected: dict[int, torch.Tensor] = field(default_factory=dict)
    tentative: bool = False

    @property
    def length(self):
        return self.positions.numel()

    @property
    def decode(self):
        # Verification has ring slack for the entire tentative block. All rows
        # can use the same causal paged-decode kernel as autoregressive decode.
        return self.tentative or all(request.length == 1 for request in self.requests)

    @property
    def max_length(self):
        return max((request.length for request in self.requests), default=0)
