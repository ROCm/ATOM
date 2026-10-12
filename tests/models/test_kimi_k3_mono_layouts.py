# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``atom.models.kimi_k3.mono.layouts`` against the AITER functions it inverts.

A round trip is the whole claim: these two inverses exist so the K3 mono bind
can dequantize the latent MoE projections ATOM stores shuffled, and an inverse
that is subtly wrong produces a weight that is uncorrelated noise rather than an
error. AITER is imported inside the test so the module still collects where it
is absent.
"""

import pytest
import torch

from atom.models.kimi_k3.mono.layouts import unshuffle_scale, unshuffle_weight

# (rows, packed columns) of the planes K3's bind actually undoes, and two small
# shapes that exercise the tiling on its own
WEIGHT_SHAPES = [(32, 64), (16, 32), (896, 1792), (3584, 3584)]
# (rows, E8M0 groups): the latent up- and down-projections, then a padded one
SCALE_SHAPES = [(896, 112), (3584, 224), (64, 12), (32, 8)]


@pytest.mark.parametrize(("rows", "cols"), WEIGHT_SHAPES)
def test_unshuffle_weight_round_trips(rows, cols):
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    w = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    shuffled = shuffle.shuffle_weight(w, layout=(16, 16))
    assert not torch.equal(shuffled.view(torch.uint8), w), "shuffle was a no-op"
    back = unshuffle_weight(shuffled.view(torch.uint8))
    assert torch.equal(back, w)


@pytest.mark.parametrize(("rows", "groups"), SCALE_SHAPES)
def test_unshuffle_scale_round_trips(rows, groups):
    fp4_utils = pytest.importorskip("aiter.utility.fp4_utils")
    s = torch.randint(0, 256, (rows, groups), dtype=torch.uint8)
    shuffled = fp4_utils.e8m0_shuffle(s)
    back = unshuffle_scale(shuffled, rows, groups)
    assert back.shape == (rows, groups)
    assert torch.equal(back, s)


def test_unshuffle_weight_refuses_an_untiled_shape():
    with pytest.raises(ValueError, match="tiled"):
        unshuffle_weight(torch.zeros(17, 32, dtype=torch.uint8))


def test_unshuffle_scale_refuses_a_wrong_sized_plane():
    with pytest.raises(ValueError, match="bytes"):
        unshuffle_scale(torch.zeros(32 * 7, dtype=torch.uint8), 32, 8)
