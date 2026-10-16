# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``atom.models.kimi_k3.mono.layouts`` against the AITER functions it inverts.

A round trip is the whole claim: these two inverses exist so the K3 mono bind
can dequantize the latent MoE projections ATOM stores shuffled, and an inverse
that is subtly wrong produces a weight that is uncorrelated noise rather than an
error. AITER is imported inside the test so the module still collects where it
is absent.
"""

import types

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


# A tile-aligned plane: 256 rows and 8 groups are exactly what `unshuffle_scale`
# pads up to, so inverse-permuting one that was never permuted raises nothing.
# That is why the un-gated un-shuffle was silent on the routed-expert planes.
GATE_ROWS, GATE_COLS = 256, 256


def _mxfp4_module(packed: torch.Tensor, scale: torch.Tensor, *, shuffled: bool):
    """A stand-in for a loaded latent-MoE projection: a ``weight`` Parameter
    carrying ``is_shuffled`` and the ``weight_scale`` beside it."""
    weight = torch.nn.Parameter(packed, requires_grad=False)
    weight.is_shuffled = shuffled
    return types.SimpleNamespace(
        weight=weight, weight_scale=torch.nn.Parameter(scale, requires_grad=False)
    )


def _reference(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from atom.model_ops.quant_v4 import dequant_fp4_e2m1

    return dequant_fp4_e2m1(
        packed.view(torch.int8),
        scale.view(torch.float8_e8m0fnu),
        32,
        torch.bfloat16,
    )


def test_unshuffled_scale_is_not_inverse_permuted():
    """`ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1` leaves BOTH the weight and its
    E8M0 scale row-major (`linear.py` gates each on the same predicate), so the
    bind must undo neither. Un-gating the scale alone mis-scales every group of
    `w_ld` / `w_up`, which no shape check catches and no output flags.
    """
    from atom.models.kimi_k3.mono.weights import _mxfp4_to_bf16

    torch.manual_seed(0)
    groups = GATE_COLS // 32
    packed = torch.randint(0, 256, (GATE_ROWS, GATE_COLS // 2), dtype=torch.uint8)
    # a spread of E8M0 exponents, so a permuted plane cannot coincide with this one
    scale = torch.randint(120, 136, (GATE_ROWS, groups), dtype=torch.uint8)

    got = _mxfp4_to_bf16(
        _mxfp4_module(packed, scale, shuffled=False), "w_up", (GATE_ROWS, GATE_COLS)
    )
    assert torch.equal(got, _reference(packed, scale))
    # the permuted reading really is a different answer, not an equal one
    assert not torch.equal(
        got, _reference(packed, unshuffle_scale(scale, GATE_ROWS, groups))
    )


def test_shuffled_scale_is_still_inverse_permuted():
    """The gate must not cost the default path: a preshuffled weight still has
    both layouts undone."""
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    fp4_utils = pytest.importorskip("aiter.utility.fp4_utils")
    from atom.models.kimi_k3.mono.weights import _mxfp4_to_bf16

    torch.manual_seed(0)
    groups = GATE_COLS // 32
    packed = torch.randint(0, 256, (GATE_ROWS, GATE_COLS // 2), dtype=torch.uint8)
    scale = torch.randint(120, 136, (GATE_ROWS, groups), dtype=torch.uint8)

    module = _mxfp4_module(
        shuffle.shuffle_weight(packed, layout=(16, 16)).view(torch.uint8),
        fp4_utils.e8m0_shuffle(scale),
        shuffled=True,
    )
    got = _mxfp4_to_bf16(module, "w_up", (GATE_ROWS, GATE_COLS))
    assert torch.equal(got, _reference(packed, scale))


def test_a_bound_band_does_not_retain_the_whole_plane():
    """A row `narrow` of a row-major tensor is already contiguous, so
    `.contiguous()` returns the same view and the full dequantized plane stays
    alive behind it. K3 binds one TP band per rank out of a 49 MiB temporary;
    holding the temporary costs ~43 MiB a layer a rank, which runs the lazy
    bind out of memory and silently drops mono back to the production path."""
    plane = torch.arange(256 * 64, dtype=torch.bfloat16).reshape(256, 64)
    band = plane.narrow(0, 64, 64)
    assert band.is_contiguous() and band.contiguous() is band

    bound = band.clone()
    assert torch.equal(bound, band)
    assert bound.is_contiguous()
    assert bound.untyped_storage().nbytes() == band.numel() * band.element_size()
    assert bound.untyped_storage().nbytes() < plane.untyped_storage().nbytes()
