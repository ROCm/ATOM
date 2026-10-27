# SPDX-License-Identifier: MIT
"""The draft block's geometry: where a row sits, what it sees, how it rotates.

Plain torch, kept out of `dspark` so that it imports without Triton: these
three are what a CPU-only machine can check, and the kernels beside them are
not.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DraftStep:
    positions: torch.Tensor
    indices: torch.Tensor
    decode: bool = False


def draft_step(context_positions, anchors, width, window):
    """Anchors locate the last processed target token, not the next input ID."""
    positions = anchors[:, None] + torch.arange(
        1, width + 1, device=anchors.device, dtype=anchors.dtype
    )
    valid = (
        (context_positions >= 0)
        & (context_positions <= anchors[:, None])
        & (context_positions > anchors[:, None] - window)
    )
    slots = torch.arange(context_positions.shape[1], device=anchors.device)
    history = torch.where(valid, slots[None], -1)
    draft = context_positions.shape[1] + torch.arange(width, device=anchors.device)
    indices = torch.cat((history, draft[None].expand(anchors.shape[0], -1)), dim=-1)
    return DraftStep(
        positions, indices[:, None].expand(-1, width, -1).int().contiguous()
    )


def rotate_rows(rope, hidden, positions, *, inverse=False):
    """Flatten ragged-request positions onto the existing V4.1 RoPE interface."""
    shape = hidden.shape
    return rope(
        hidden.reshape(1, -1, *shape[2:]), positions.flatten(), inverse=inverse
    ).view(shape)
