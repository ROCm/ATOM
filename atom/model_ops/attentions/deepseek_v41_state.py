# SPDX-License-Identifier: MIT
"""Bounded eager CSA2 cache ownership; paged request lifecycle is a separate layer."""

from dataclasses import dataclass, field

import torch

from atom.model_ops.deepseek_v41.compressor import CompressorTail
from atom.models.deepseek_v41.config import AttentionMode


@dataclass
class AttentionStep:
    position: int
    length: int
    indices: dict[int, torch.Tensor] = field(default_factory=dict)
    candidates: dict[int, torch.Tensor] = field(default_factory=dict)


class EagerAttentionCache:
    """One fixed batch of equal-position sequences, with one allocation per owner.

    This is the offline correctness baseline. It deliberately has no global
    module state and cannot be shared between independent requests or branches.
    """

    def __init__(self, config, topology, batch_size, max_length, device):
        if batch_size < 1 or not 1 <= max_length <= config.max_position_embeddings:
            raise ValueError("Invalid eager CSA2 cache capacity")
        self.batch_size, self.max_length = batch_size, max_length
        self.position = 0
        self.window = {}
        self.tails: dict[int, CompressorTail | None] = {}
        self.main = {}
        self.index = {}
        for spec in topology:
            if spec.mode == AttentionMode.FULL:
                count = max_length // spec.ratio
                self.main[spec.layer_id] = torch.empty(
                    batch_size,
                    count,
                    config.head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                self.index[spec.layer_id] = torch.empty(
                    batch_size,
                    count,
                    config.index_head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )

    def begin_step(self, position, length, batch_size):
        if position != self.position or batch_size != self.batch_size or length < 1:
            raise ValueError(
                "Eager cache requires the next contiguous step of its fixed batch"
            )
        if position + length > self.max_length:
            raise ValueError("Eager cache capacity exceeded")
        return AttentionStep(position, length)

    def finish_step(self, step):
        if step.position != self.position:
            raise ValueError("Attention step does not match the cache position")
        self.position += step.length

    def append_window(self, layer_id, kv, window_size, step):
        previous = self.window.get(layer_id)
        if step.position and previous is None:
            raise ValueError("Missing sliding-window state")
        combined = kv if previous is None else torch.cat((previous, kv), dim=1)
        past = 0 if previous is None else previous.shape[1]
        end = past + torch.arange(step.length, device=kv.device)
        start = end - window_size + 1
        if step.position == 0:
            count = min(window_size, combined.shape[1])
            start = start.clamp_min(0)
        else:
            # Reference decode retains the leading empty ring slots. Compacting
            # them changes the 64-row BF16 probability-rounding boundaries.
            count = window_size
        indices = start[:, None] + torch.arange(count, device=kv.device)
        indices = indices.masked_fill(
            (indices < 0) | (indices > end[:, None]), -1
        ).int()
        self.window[layer_id] = combined[:, -window_size:].clone()
        return combined, indices[None].expand(kv.shape[0], -1, -1).contiguous()
