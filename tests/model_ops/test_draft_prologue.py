# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The draft block's prologue, fused, against the torch chains it replaces.

Two seams sat in front of every drafting step, both of them launch-floor work:

  * the ring-slot -> position map and the block's index list, ~16 torch ops
    over `[B, ring]` rows, four of them an `arange` rebuilding a constant;
  * the block's id tensor, its embedding, the mHC broadcast and the pre-mix,
    five ops over `[B, width]` -- with an embedding that is a TP all-reduce
    over `B * width` rows of which only `B` ever differ.

Both are integer index math or a straight copy, so both are held to
bit-exactness here rather than the "nearer fp64" standard the quantizing
fusions get: there is nothing to round.

The seed-2 case in the index sweep draws positions below `ring`, which is what
exercises the trap that seam exists to get right: `(position - slot) % ring`
has a negative dividend on a partly-filled ring, Triton's `%` keeps the
dividend's sign and torch's does not, and the difference silently points the
draft at the wrong rows rather than failing.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises Triton kernels; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.deepseek_v41.draft_block import draft_step
from atom.model_ops.deepseek_v41.dspark import build_block_state, draft_step_indices
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState

DEV = "cuda"
HC_MULT = 4  # config.hc_mult


def _torch_index_chain(positions, ring, window, width):
    """What `block_backbone` + `draft_step` spelled out, unchanged."""
    physical = torch.arange(ring, device=positions.device)
    context_positions = positions[:, None] - (positions[:, None] - physical) % ring
    return draft_step(context_positions, positions, width, window), context_positions


# The kernel is one program per row, so the batch axis only varies how many
# rows there are; what the index math turns on is how `ring` sits against
# `window` and how wide the draft is.
@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize(
    "ring,window,width",
    [
        (128, 64, 5),  # ring wider than the window, production width
        (128, 128, 5),  # ring exactly the window
        (64, 128, 5),  # ring narrower than the window
        (256, 64, 1),  # single-column draft
        (64, 64, 8),  # widest draft on the tightest ring
        (256, 128, 8),
    ],
)
def test_index_build_is_bit_exact(batch, ring, window, width):
    gen = torch.Generator(device=DEV).manual_seed(batch * ring + window + width)
    positions = torch.randint(
        0, 4096, (batch,), generator=gen, device=DEV, dtype=torch.int64
    )
    reference, context = _torch_index_chain(positions, ring, window, width)
    fused_positions, fused_indices, fused_context = draft_step_indices(
        positions, ring, window, width
    )

    assert torch.equal(fused_positions, reference.positions)
    assert torch.equal(fused_indices, reference.indices)
    assert torch.equal(fused_context, context)
    assert fused_indices.dtype == torch.int32 and fused_indices.is_contiguous()


@pytest.mark.parametrize("ring", [128, 64])
def test_partly_filled_ring_wraps_the_way_torch_does(ring):
    """Positions below `ring`: the negative-dividend remainder path."""
    positions = torch.arange(ring, device=DEV, dtype=torch.int64)
    reference, context = _torch_index_chain(positions, ring, 64, 5)
    fused_positions, fused_indices, fused_context = draft_step_indices(
        positions, ring, 64, 5
    )
    assert torch.equal(fused_positions, reference.positions)
    assert torch.equal(fused_indices, reference.indices)
    assert torch.equal(fused_context, context)
    # Position 0 sees ring slot 0 and nothing else -- every other slot maps to
    # a negative absolute position, a row the window writer has not filled for
    # this request yet. The list is a broadcast along the draft axis, so each
    # of the `width` rows says the same thing.
    assert (fused_indices[0] >= 0).sum() == 5 * (1 + 5)
    assert torch.equal(fused_indices[0, 0], fused_indices[0, -1])


# `hidden` only sets the tile-loop count; width and batch set the broadcast.
@pytest.mark.parametrize(
    "batch,width,hidden",
    [(1, 5, 5120), (64, 5, 5120), (1, 1, 1024), (64, 8, 1024), (64, 1, 5120)],
)
def test_block_state_matches_embed_then_broadcast(batch, width, hidden):
    gen = torch.Generator(device=DEV).manual_seed(batch * width + hidden)
    anchor = torch.randn(batch, hidden, generator=gen, device=DEV, dtype=torch.bfloat16)
    noise = torch.randn(hidden, generator=gen, device=DEV, dtype=torch.bfloat16)

    # The chain this replaces: embed a [B, width] id block whose columns past
    # zero all carry the noise token, then broadcast over the mHC streams.
    embedded = noise.expand(batch, width, hidden).clone()
    embedded[:, 0, :] = anchor
    reference = SinglePassHCState.from_embeddings(embedded, HC_MULT)

    residual, pre_mix = build_block_state(anchor, noise, width, HC_MULT)
    assert torch.equal(residual, reference.residual)
    assert torch.equal(pre_mix, reference.pre_mix)
    assert residual.shape == (batch, width, HC_MULT, hidden)


def test_prologue_handles_an_empty_batch():
    positions = torch.empty(0, device=DEV, dtype=torch.int64)
    out_positions, out_indices, out_context = draft_step_indices(positions, 128, 64, 5)
    assert out_positions.shape == (0, 5)
    assert out_indices.shape == (0, 5, 133)
    assert out_context.shape == (0, 128)

    anchor = torch.empty(0, 512, device=DEV, dtype=torch.bfloat16)
    noise = torch.zeros(512, device=DEV, dtype=torch.bfloat16)
    residual, pre_mix = build_block_state(anchor, noise, 5, HC_MULT)
    assert residual.shape == (0, 5, HC_MULT, 512)
    assert pre_mix.shape == (0, 5, HC_MULT)
