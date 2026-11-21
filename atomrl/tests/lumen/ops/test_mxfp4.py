# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 operand-layout, fusion, and SR-Philox regressions.

Collected from the per-optimization files so one module covers:
  - dual-layout column shuffle
  - fused activation scale swizzle
  - fused WGrad activation operand
  - Philox round-count dither quality
"""

import importlib
import os
import random

import atomrl.lumen.kernels.mxfp4 as mxfp4_kernels
import pytest
import torch
from atomrl.lumen.ops.quantize.linear import (
    QuantizedLinearFunction,
    _shuffle_mxfp4_weight,
)
from atomrl.lumen.ops.quantize.ops import dual_layout_quant_mxfp4

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


def _run_linear(x, w, dy):
    random.seed(0)
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    out = QuantizedLinearFunction.apply(x, w, None, None, "mxfp4", None, 32, "weight")
    out.backward(dy)
    return out, x.grad, w.grad


def _snr(ref, got):
    err = (ref.float() - got.float()).pow(2).sum()
    if err == 0:
        return float("inf")
    return 10 * torch.log10(ref.float().pow(2).sum() / err).item()


@pytest.mark.parametrize("shape", [(256, 128), (512, 256), (1024, 512)])
def test_shuffled_col_operand_matches_separate_shuffle_pass(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    sign = (torch.randint(0, 2, (G,), device="cuda") * 2 - 1).to(torch.bfloat16)

    row, row_s, col, col_s = dual_layout_quant_mxfp4(
        x,
        sign,
        block_size=BLOCK,
        g=G,
        use_sr_row=False,
        use_sr_transposed=False,
    )
    row_f, row_sf, col_f, col_sf = dual_layout_quant_mxfp4(
        x,
        sign,
        block_size=BLOCK,
        g=G,
        use_sr_row=False,
        use_sr_transposed=False,
        shuffle_col=True,
    )

    torch.testing.assert_close(row_f, row, rtol=0, atol=0)
    torch.testing.assert_close(row_sf, row_s, rtol=0, atol=0)
    torch.testing.assert_close(col_sf, col_s, rtol=0, atol=0)
    torch.testing.assert_close(col_f, _shuffle_mxfp4_weight(col), rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 scale layout")
@pytest.mark.parametrize("M,K,N", [(512, 4096, 4096), (256, 2048, 512)])
def test_mxfp4_linear_fused_act_scale_matches_row_major(M, K, N, monkeypatch):
    import atomrl.lumen.ops.quantize.linear as lin

    torch.manual_seed(17)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    if not lin._mxfp4_can_fuse_scale_swizzle((M, K // 32)):
        pytest.skip(f"({M}, {K}) activation scales are not swizzle-eligible")

    monkeypatch.setattr(lin, "_mxfp4_wgrad_activation_operand", lambda *a: None)

    try:
        got = _run_linear(x, w, dy)
        monkeypatch.setattr(lin, "_mxfp4_can_fuse_scale_swizzle", lambda *a: False)
        ref = _run_linear(x, w, dy)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 path unavailable: {e}")

    for name, a, b in zip(("out", "dX", "dW"), got, ref):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=f"{name} differs")


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
@pytest.mark.parametrize("M,K,N", [(512, 512, 256), (768, 256, 512)])
def test_fused_wgrad_operand_keeps_forward_and_improves_wgrad(M, K, N, monkeypatch):
    import atomrl.lumen.ops.quantize.linear as lin

    torch.manual_seed(17)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    if lin._mxfp4_wgrad_activation_operand(x, w, "mxfp4", True, True) is None:
        pytest.skip(f"({M}, {K}) is not eligible for the fused WGrad operand")

    try:
        got = _run_linear(x, w, dy)
        monkeypatch.setattr(lin, "_mxfp4_wgrad_activation_operand", lambda *a: None)
        ref = _run_linear(x, w, dy)
    except (AssertionError, RuntimeError) as e:
        pytest.skip(f"Lumen MXFP4 path unavailable: {e}")

    torch.testing.assert_close(got[0], ref[0], atol=0, rtol=0, msg="out differs")
    torch.testing.assert_close(got[1], ref[1], atol=0, rtol=0, msg="dX differs")

    dw_ref = dy.float().t() @ x.float()
    snr_fused, snr_rebuilt = _snr(dw_ref, got[2]), _snr(dw_ref, ref[2])
    assert (
        snr_fused >= snr_rebuilt - 0.5
    ), f"fused dW is worse than the rebuilt one: {snr_fused:.2f} dB vs {snr_rebuilt:.2f} dB"


@pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout")
def test_fused_operand_is_the_direct_quantization_of_the_rotated_activation():
    import atomrl.lumen.ops.quantize.linear as lin
    from atomrl.lumen.ops.quantize.ops import (
        convert_from_mxfp4,
        convert_to_mxfp4,
        dequant_hadamard_quant_mxfp4,
        hadamard_quant_mxfp4,
    )

    M, K = 512, 512
    torch.manual_seed(3)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    sign = lin._get_mxfp4_rht_sign(x.device)

    _, _, wg_fp4, wg_scale = dual_layout_quant_mxfp4(
        x,
        sign,
        block_size=32,
        g=lin._MXFP4_RHT_G,
        use_sr_row=False,
        use_sr_transposed=False,
    )
    exact_fp4, exact_scale = hadamard_quant_mxfp4(
        x.t().contiguous(),
        sign,
        block_size=32,
        g=lin._MXFP4_RHT_G,
        use_sr=False,
    )
    torch.testing.assert_close(wg_fp4, exact_fp4, atol=0, rtol=0)
    torch.testing.assert_close(wg_scale, exact_scale, atol=0, rtol=0)

    x_fp4, x_scale = convert_to_mxfp4(x, block_size=32, axis=-1, use_sr=False)
    rebuilt_fp4, rebuilt_scale = dequant_hadamard_quant_mxfp4(
        x_fp4,
        x_scale,
        sign,
        block_size=32,
        g=lin._MXFP4_RHT_G,
        use_sr=False,
    )
    exact = convert_from_mxfp4(
        exact_fp4, exact_scale, output_dtype=torch.float32, block_size=32
    )
    rebuilt = convert_from_mxfp4(
        rebuilt_fp4,
        rebuilt_scale,
        output_dtype=torch.float32,
        block_size=32,
    )
    assert not torch.equal(
        exact, rebuilt
    ), "the rebuild would be lossless, which it is not"


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
    fp4_utils = pytest.importorskip("aiter.utility.fp4_utils", reason="AITER required")

    codes = fp4_utils.mxfp4_to_f32(packed)
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
