# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################
"""
Tests for atomrl.lumen.ops.quantize.linear: FP8 quantized linear / GEMM.

Covers:
  - FP8 quantized GEMM forward — all 7 scaling types vs torch.mm reference
  - FP8 quantized linear forward + backward — all 7 scaling types
  - Bias, weight-only quant, BF16 wgrad, edge cases
"""

import atomrl.lumen.ops.quantize.linear as linear_ops
import pytest
import torch
from conftest import LinearConfig, compute_snr

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

LINEAR_SHAPES = [
    LinearConfig(32, 256, 512),
    LinearConfig(128, 4096, 4096),
    LinearConfig(256, 4096, 1024),
]

LINEAR_IDS = [repr(c) for c in LINEAR_SHAPES]

ALL_SCALING_TYPES = [
    "delayed",
    "dynamic",
    "per_token",
    "blockwise",
    "blockwise2d",
    "mxfp8",
    "mxfp4",
    "none",
]

# Backward reuses forward's weight_scale after transposing weight_fp8.
# Only per-tensor (scalar) scales survive transposition; per_token (N,1) block
# scales become misaligned with the transposed layout, so per_token stays on the
# dequant→BF16 fallback.
#
# blockwise2d (Jet-RL §4.2) uses square 128×128 tiles whose scale grid IS
# transpose-symmetric, so DGrad reuses the FProp kernel and WGrad runs a
# (1×128)×(1×128) blockscale GEMM with X re-quantized along the token axis.
# blockwise(1D) reaches full FP8 DGrad/WGrad via a columnwise re-quant copy of
# the weight (1×128 along N); WGrad is shared with blockwise2d.
# Both require M (token count) divisible by 128 — non-aligned M falls back to BF16.
#
# mxfp4 reaches full backward by a different route: DGrad reads a transposed
# packed-FP4 weight and WGrad applies a Hadamard rotation before quantizing, so
# neither leg depends on a scale surviving transposition.
BWD_SCALING_TYPES = [
    "delayed",
    "dynamic",
    "blockwise",
    "blockwise2d",
    "mxfp8",
    "mxfp4",
    "none",
]

# SNR floors per scaling type — coarser quantization ⇒ lower expected SNR.
# mxfp8 uses E8M0 exponent-only scales; blockwise has block granularity.
#
# mxfp4 sits above mxfp8 here despite being the narrower format: its floors are
# set from measurement, and a GEMM averages each operand's quantization error
# over K, so 4-bit elements do not cost 4-bit accuracy in the product. Measured
# 15.3 fwd / 17.4-17.8 dx / 15.9-18.9 dw across LINEAR_SHAPES on gfx950; the
# floors leave room for the run-to-run spread stochastic rounding puts on dw.
_FWD_SNR = {
    "delayed": 12,
    "dynamic": 12,
    "per_token": 12,
    "blockwise": 8,
    "blockwise2d": 8,
    "mxfp8": 5,
    "mxfp4": 12,
    "none": 25,
}
_BWD_DX_SNR = {
    "delayed": 6,
    "dynamic": 6,
    "per_token": 6,
    "blockwise": 4,
    "blockwise2d": 4,
    "mxfp8": 3,
    "mxfp4": 12,
    "none": 20,
}
_BWD_DW_SNR = {
    "delayed": 6,
    "dynamic": 6,
    "per_token": 6,
    "blockwise": 4,
    "blockwise2d": 4,
    "mxfp8": 3,
    "mxfp4": 10,
    "none": 20,
}


def _block_size_for(scaling_type, default=128):
    """Effective block_size used by quantize_input for this scaling type."""
    if scaling_type in ("mxfp8", "mxfp4"):
        return 32 if default > 64 else default
    return default


def _skip_if_unaligned(config, scaling_type, *, bwd=False):
    block_size = _block_size_for(scaling_type)
    if (
        scaling_type in ("blockwise", "blockwise2d", "mxfp8", "mxfp4")
        and config.K % block_size != 0
    ):
        pytest.skip(f"K={config.K} not divisible by block_size={block_size}")
    # blockwise2d backward needs M (token count) % 128 == 0 because the WGrad
    # kernel groups along the M axis with GROUP_K=128. Non-aligned M would
    # silently fall back to BF16 in production but we skip the test here so
    # the FP8 path is what's being exercised.
    if bwd and scaling_type == "blockwise2d" and config.M % block_size != 0:
        pytest.skip(f"blockwise2d bwd needs M={config.M} divisible by {block_size}")


# ===================================================================
# FP8 quantized linear forward — all scaling types
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES, ids=LINEAR_IDS)
@pytest.mark.parametrize("scaling_type", ALL_SCALING_TYPES)
def test_fp8_linear_fwd(config, scaling_type):
    """FP8 quantized GEMM forward: compare output vs torch.mm reference."""
    _skip_if_unaligned(config, scaling_type)
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02

    out_ref = x @ w.T
    try:
        out_lumen = linear_ops.quantized_linear(
            x,
            w,
            scaling_type=scaling_type,
            quantize_activation=True,
        )
    except AssertionError as e:
        if "GROUP_K" in str(e) or "BLOCK_SIZE_K" in str(e):
            pytest.skip(f"AITER kernel config unsupported for {scaling_type}: {e}")
        raise
    torch.cuda.synchronize()

    snr = compute_snr(out_ref, out_lumen)
    floor = _FWD_SNR[scaling_type]
    assert (
        snr > floor
    ), f"FP8 linear fwd ({scaling_type}) SNR: {snr:.1f} dB (expected > {floor})"


def test_mxfp8_forward_slices_padded_output():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith(("gfx950", "gfx1250")):
        pytest.skip(f"AITER native MXFP8 GEMM is unavailable on {arch}")

    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(50, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    reference = x @ weight.T
    result = linear_ops.quantized_linear(x, weight, scaling_type="mxfp8")

    assert result.shape == (17, 50)
    assert compute_snr(reference, result) > _FWD_SNR["mxfp8"]


def test_mxfp8_backward_slices_padded_operands():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith(("gfx950", "gfx1250")):
        pytest.skip(f"AITER native MXFP8 GEMM is unavailable on {arch}")

    x = (
        torch.randn(17, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    weight = (
        torch.randn(50, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    grad = torch.randn(17, 50, device="cuda", dtype=torch.bfloat16)

    result = linear_ops.quantized_linear(x, weight, scaling_type="mxfp8")
    reference = x_ref @ weight_ref.T
    result.backward(grad)
    reference.backward(grad)

    assert x.grad.shape == x.shape
    assert weight.grad.shape == weight.shape
    assert compute_snr(x_ref.grad, x.grad) > _BWD_DX_SNR["mxfp8"]
    assert compute_snr(weight_ref.grad, weight.grad) > _BWD_DW_SNR["mxfp8"]


# ===================================================================
# FP8 quantized linear forward + backward — all scaling types
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
@pytest.mark.parametrize("scaling_type", BWD_SCALING_TYPES)
def test_fp8_linear_fwd_bwd(config, scaling_type):
    """Forward + backward through quantized linear.

    See BWD_SCALING_TYPES comment for which scaling modes support FP8 bwd.
    """
    _skip_if_unaligned(config, scaling_type, bwd=True)
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    x.requires_grad_(True)
    w.requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type=scaling_type,
        quantize_activation=True,
    )
    torch.cuda.synchronize()
    out_lumen.float().mean().backward()
    torch.cuda.synchronize()

    out_snr = compute_snr(out_ref, out_lumen)
    dx_snr = compute_snr(x_ref.grad, x.grad)
    dw_snr = compute_snr(w_ref.grad, w.grad)

    fwd_floor = _FWD_SNR[scaling_type]
    dx_floor = _BWD_DX_SNR[scaling_type]
    dw_floor = _BWD_DW_SNR[scaling_type]
    assert out_snr > fwd_floor, f"fwd ({scaling_type}) SNR: {out_snr:.1f} dB"
    assert dx_snr > dx_floor, f"bwd dx ({scaling_type}) SNR: {dx_snr:.1f} dB"
    assert dw_snr > dw_floor, f"bwd dw ({scaling_type}) SNR: {dw_snr:.1f} dB"


# ===================================================================
# Bias
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
@pytest.mark.parametrize("scaling_type", BWD_SCALING_TYPES)
def test_fp8_linear_bias(config, scaling_type):
    """Forward + backward with bias."""
    _skip_if_unaligned(config, scaling_type, bwd=True)
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    x.requires_grad_(True)
    w.requires_grad_(True)
    b = torch.randn(N, device="cuda", dtype=dtype) * 0.01
    b.requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T + b_ref
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        bias=b,
        scaling_type=scaling_type,
        quantize_activation=True,
    )
    torch.cuda.synchronize()
    out_lumen.float().mean().backward()
    torch.cuda.synchronize()

    fwd_floor = _FWD_SNR[scaling_type]
    assert compute_snr(out_ref, out_lumen) > fwd_floor
    assert compute_snr(x_ref.grad, x.grad) > _BWD_DX_SNR[scaling_type]
    assert compute_snr(w_ref.grad, w.grad) > _BWD_DW_SNR[scaling_type]
    assert compute_snr(b_ref.grad, b.grad) > 20, "bias gradient mismatch"


# ===================================================================
# Bias forward-only — scaling types incompatible with backward
# ===================================================================


FWD_ONLY_SCALING_TYPES = [s for s in ALL_SCALING_TYPES if s not in BWD_SCALING_TYPES]


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
@pytest.mark.parametrize("scaling_type", FWD_ONLY_SCALING_TYPES)
def test_fp8_linear_bias_fwd_only(config, scaling_type):
    """Forward-only bias test for scaling types that don't support backward.

    per_token / blockwise / mxfp8 weight scales become misaligned after
    transposition in the backward pass (see BWD_SCALING_TYPES comment),
    so we only verify the forward output here.
    """
    _skip_if_unaligned(config, scaling_type)
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    b = torch.randn(N, device="cuda", dtype=dtype) * 0.01

    out_ref = x @ w.T + b
    try:
        out_lumen = linear_ops.quantized_linear(
            x,
            w,
            bias=b,
            scaling_type=scaling_type,
            quantize_activation=True,
        )
    except AssertionError as e:
        if "GROUP_K" in str(e) or "BLOCK_SIZE_K" in str(e):
            pytest.skip(f"AITER kernel config unsupported for {scaling_type}: {e}")
        raise
    torch.cuda.synchronize()

    snr = compute_snr(out_ref, out_lumen)
    floor = _FWD_SNR[scaling_type]
    assert (
        snr > floor
    ), f"FP8 linear bias fwd ({scaling_type}) SNR: {snr:.1f} dB (expected > {floor})"


# ===================================================================
# Weight-only quantization (quantize_activation=False)
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
@pytest.mark.parametrize("scaling_type", ["dynamic", "per_token"])
def test_fp8_linear_weight_only(config, scaling_type):
    """Weight-only quant: activation stays BF16, weight is FP8-dequanted."""
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    x.requires_grad_(True)
    w.requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type=scaling_type,
        quantize_activation=False,
    )
    torch.cuda.synchronize()
    out_lumen.float().mean().backward()
    torch.cuda.synchronize()

    assert compute_snr(out_ref, out_lumen) > 10
    assert compute_snr(x_ref.grad, x.grad) > 6
    assert compute_snr(w_ref.grad, w.grad) > 6


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
def test_fp8_stored_blockwise2d_dgrad(config):
    """FP8-stored (frozen) weight blockwise2d DGrad runs as an FP8 blockscale
    GEMM (no dequant->BF16 + weight.t().contiguous()). Only DGrad is produced;
    the weight is frozen (no WGrad). Exercises FP8StoredLinearFunction."""
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N
    if N % 128 or K % 128:
        pytest.skip("blockwise2d FP8 DGrad path needs N, K divisible by 128")

    fp8_dtype = linear_ops._get_float8_e4m3()
    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    x.requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone()

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    # Pre-quantize weight to blockwise2d FP8 (128x128 tiles, 2D scale) and run
    # through the FP8-stored path (pre_quantized_weight -> FP8StoredLinearFunction).
    desc = linear_ops.quantize_input(w, "blockwise2d", fp8_dtype, 128, is_weight=True)
    out = linear_ops.quantized_linear(
        x,
        w,
        scaling_type="blockwise2d",
        pre_quantized_weight=(desc.data, desc.scale),
    )
    torch.cuda.synchronize()
    out.float().mean().backward()
    torch.cuda.synchronize()

    assert compute_snr(out_ref, out) > 8
    assert x.grad is not None
    assert compute_snr(x_ref.grad, x.grad) > 4


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
def test_fp8_blockwise2d_wgrad_fp8_requant(config):
    """WGrad FP8→FP8 requant kernel: activation (1×128 row) re-quantized to (128×1 col)
    without BF16 roundtrip via requant_fp8_row_to_col.  Validates numerical accuracy
    against the BF16-roundtrip reference (dequant→BF16→requant(axis=0))."""
    from atomrl.lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight
    from atomrl.lumen.ops.quantize.ops import (
        quant_fp8_blockwise_impl,
        requant_fp8_row_to_col,
    )

    dtype = torch.bfloat16
    M, K, _N = config.M, config.K, config.N
    if M % 128 or K % 128:
        pytest.skip("requant_fp8_row_to_col needs M, K divisible by 128")

    fp8_dtype = linear_ops._get_float8_e4m3()

    # Build a BF16 activation and row-wise quantize it (as done in forward).
    x_bf16 = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    x_row, x_row_s = quant_fp8_blockwise_impl(x_bf16, fp8_dtype, axis=1, block_size=128)

    # Reference: dequant → BF16 → col-wise requant (existing path).
    x_dequant = _dequant_fp8_weight(x_row, x_row_s, 128).bfloat16()
    x_col_ref, x_col_s_ref = quant_fp8_blockwise_impl(
        x_dequant.contiguous(),
        fp8_dtype,
        axis=0,
        block_size=128,
    )

    # Candidate: direct FP8→FP8 requant kernel.
    x_col, x_col_s = requant_fp8_row_to_col(x_row, x_row_s, fp8_dtype, 128)

    # Dequant both and compare as float32.
    x_col_f32_ref = _dequant_fp8_weight(x_col_ref, x_col_s_ref, 128).float()
    x_col_f32 = _dequant_fp8_weight(x_col, x_col_s, 128).float()
    snr = compute_snr(x_col_f32_ref, x_col_f32)
    assert snr > 20, f"requant_fp8_row_to_col SNR vs BF16-roundtrip ref: {snr:.1f} dB"

    # Secondary check: x_col dequanted should be close to the BF16-roundtrip version.
    # This confirms the kernel's col scales are numerically consistent.
    x_col_f32_ref2 = _dequant_fp8_weight(x_col_ref, x_col_s_ref, 128).float()
    x_col_f32_2 = _dequant_fp8_weight(x_col, x_col_s, 128).float()
    snr2 = compute_snr(x_col_f32_ref2, x_col_f32_2)
    assert (
        snr2 > _BWD_DW_SNR["blockwise2d"]
    ), f"requant_fp8_row_to_col col SNR vs BF16-roundtrip: {snr2:.1f} dB"


# ===================================================================
# BF16 weight gradient (fp8_wgrad=False)
# ===================================================================


@pytest.mark.parametrize("config", LINEAR_SHAPES[:2], ids=LINEAR_IDS[:2])
def test_fp8_linear_bf16_wgrad(config):
    """fp8_wgrad=False: weight gradient computed in BF16 instead of FP8."""
    dtype = torch.bfloat16
    M, K, N = config.M, config.K, config.N

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    x.requires_grad_(True)
    w.requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type="dynamic",
        quantize_activation=True,
        fp8_wgrad=False,
    )
    torch.cuda.synchronize()
    out_lumen.float().mean().backward()
    torch.cuda.synchronize()

    assert compute_snr(out_ref, out_lumen) > 12
    assert compute_snr(x_ref.grad, x.grad) > 6
    # BF16 wgrad should be at least as good as FP8 wgrad
    assert compute_snr(w_ref.grad, w.grad) > 8


# ===================================================================
# Edge case: M=1 (single token)
# ===================================================================


@pytest.mark.parametrize("scaling_type", BWD_SCALING_TYPES)
def test_fp8_linear_m1(scaling_type):
    """Single-row input (M=1) — exercises per-token edge case."""
    dtype = torch.bfloat16
    M, K, N = 1, 256, 512

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02
    x.requires_grad_(True)
    w.requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    out_ref = x_ref @ w_ref.T
    out_ref.float().mean().backward()

    out_lumen = linear_ops.quantized_linear(
        x,
        w,
        scaling_type=scaling_type,
        quantize_activation=True,
    )
    torch.cuda.synchronize()
    out_lumen.float().mean().backward()
    torch.cuda.synchronize()

    fwd_floor = _FWD_SNR[scaling_type]
    assert compute_snr(out_ref, out_lumen) > fwd_floor
    assert x.grad is not None and x.grad.shape == x.shape
    assert w.grad is not None and w.grad.shape == w.shape
    assert (
        compute_snr(x_ref.grad, x.grad) > _BWD_DX_SNR[scaling_type]
    ), f"M=1 bwd dx ({scaling_type}) SNR: {compute_snr(x_ref.grad, x.grad):.1f} dB"
    assert (
        compute_snr(w_ref.grad, w.grad) > _BWD_DW_SNR[scaling_type]
    ), f"M=1 bwd dw ({scaling_type}) SNR: {compute_snr(w_ref.grad, w.grad):.1f} dB"
