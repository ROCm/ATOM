# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MXFP4 SR-Philox round-count and dither-quality regressions."""

import importlib
import os

import atomrl.lumen.kernels.mxfp4 as mxfp4_kernels
import pytest
import torch
from aiter.utility.fp4_utils import mxfp4_to_f32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

G = 16
BLOCK = 32
ROUNDS_FLOOR = 4
ROUNDS_BELOW_FLOOR = 2
DRAWS = 96


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@pytest.fixture
def rebuild_at_rounds():
    saved = os.environ.get("LUMEN_SR_PHILOX_ROUNDS")

    def _rebuild(rounds):
        os.environ["LUMEN_SR_PHILOX_ROUNDS"] = str(rounds)
        module = importlib.reload(mxfp4_kernels)
        assert module.SR_PHILOX_ROUNDS == rounds, "reload did not take"
        return module

    yield _rebuild

    if saved is None:
        os.environ.pop("LUMEN_SR_PHILOX_ROUNDS", None)
    else:
        os.environ["LUMEN_SR_PHILOX_ROUNDS"] = saved
    importlib.reload(mxfp4_kernels)


def _dequant(packed, scales, block):
    codes = mxfp4_to_f32(packed)
    mult = (scales.to(torch.int32) << 23).view(torch.float32)
    return codes * mult.repeat_interleave(block, dim=-1)


def _residual_std(draws=DRAWS):
    from atomrl.lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(0)
    shape = (256, BLOCK * 8)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    acc = torch.zeros(shape, dtype=torch.float32, device="cuda")
    for i in range(draws):
        packed, scales = convert_to_mxfp4(
            x,
            BLOCK,
            axis=-1,
            use_sr=True,
            philox_seed=1234 + i,
            philox_offset=0,
        )
        acc += _dequant(packed, scales, BLOCK)
    resid = acc / draws - x.to(torch.float32)
    denom = x.to(torch.float32).abs().mean().clamp_min(1e-6)
    return (resid.std() / denom).item()


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 SR packing")
def test_sr_dither_does_not_repeat_between_tiles():
    from atomrl.lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(9)
    tile = torch.randn((64, 64), dtype=torch.bfloat16, device="cuda")
    x = tile.repeat(2, 1)

    packed, scales = convert_to_mxfp4(
        x,
        BLOCK,
        axis=-1,
        use_sr=True,
        philox_seed=1234,
        philox_offset=0,
    )

    torch.testing.assert_close(scales[:64], scales[64:], atol=0, rtol=0)
    assert not torch.equal(
        packed[:64], packed[64:]
    ), "identical input tiles reused the same stochastic-rounding stream"


def test_default_round_count_is_the_documented_default(rebuild_at_rounds):
    os.environ.pop("LUMEN_SR_PHILOX_ROUNDS", None)
    module = importlib.reload(mxfp4_kernels)

    assert module.SR_PHILOX_ROUNDS == module.SR_PHILOX_ROUNDS_DEFAULT


def test_override_reaches_the_traced_constant(rebuild_at_rounds):
    module = rebuild_at_rounds(ROUNDS_FLOOR)

    assert module.SR_PHILOX_ROUNDS == ROUNDS_FLOOR
    assert module.SR_PHILOX_ROUNDS_C.value == ROUNDS_FLOOR


def test_rounds_at_the_floor_hold_dither_quality(rebuild_at_rounds):
    rebuild_at_rounds(mxfp4_kernels.SR_PHILOX_ROUNDS_DEFAULT)
    default_std = _residual_std()

    rebuild_at_rounds(ROUNDS_FLOOR)
    floor_std = _residual_std()

    assert floor_std == pytest.approx(default_std, rel=0.15), (
        f"rounds={ROUNDS_FLOOR} residual std {floor_std:.6f} vs "
        f"default {default_std:.6f}"
    )


def test_metric_catches_a_dither_below_the_floor(rebuild_at_rounds):
    rebuild_at_rounds(mxfp4_kernels.SR_PHILOX_ROUNDS_DEFAULT)
    default_std = _residual_std()

    rebuild_at_rounds(ROUNDS_BELOW_FLOOR)
    starved_std = _residual_std()

    assert starved_std > 2 * default_std, (
        f"rounds={ROUNDS_BELOW_FLOOR} residual std {starved_std:.6f} is not "
        f"clearly worse than default {default_std:.6f}; the metric has lost "
        "its discriminating power"
    )
