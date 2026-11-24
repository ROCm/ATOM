# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################
"""
Tests for atomrl.lumen.ops.normalization: RMSNorm and LayerNorm.

Covers:
  - Forward + backward (BF16, FP32)
  - Fused quantized variants — forward (delayed, dynamic, per_token, blockwise, blockwise2d, mxfp8)
  - Fused quantized variants — backward (FP8 training: norm→quant→dequant→linear→loss)
  - MXFP8 fused quant benchmarked against torchao reference

Reference: PyTorch F.layer_norm / manual RMSNorm (following aiter test patterns).
"""

import pytest
import torch
import torch.nn.functional as F
from conftest import NormConfig, compute_snr, layernorm_ref, rmsnorm_ref
from torchao.prototype.mx_formats.config import ScaleCalculationMode

# --- torchao reference: MXFP8 ---
from torchao.prototype.mx_formats.mx_tensor import to_dtype as torchao_to_dtype
from torchao.prototype.mx_formats.mx_tensor import to_mx as torchao_to_mx

# --- torchao reference: per-tensor / blockwise FP8 quant & dequant ---
from torchao.quantization.quant_primitives import (
    _dequantize_affine_float8 as torchao_dequant_fp8,
)
from torchao.quantization.quant_primitives import (
    _quantize_affine_float8 as torchao_quant_fp8,
)

try:
    from torchao.kernel.blockwise_quantization import fp8_blockwise_act_quant
except ImportError:
    fp8_blockwise_act_quant = None

import atomrl.lumen.ops.normalization as norm_ops

# ---------------------------------------------------------------------------
# Configurations (following aiter get_vals() shapes)
# ---------------------------------------------------------------------------

NORM_SHAPES = [
    NormConfig(1, 128),
    NormConfig(2, 256),
    NormConfig(4, 4096),
    NormConfig(32, 4096),
    NormConfig(256, 4096),
    NormConfig(512, 8192),
    NormConfig(2048, 4096),
]

NORM_IDS = [repr(c) for c in NORM_SHAPES]

QUANT_SCALING_TYPES = [
    "delayed",
    "dynamic",
    "per_token",
    "blockwise",
    "blockwise2d",
    "mxfp8",
]


def _delayed_scale(norm_ref, fp8_dtype=torch.float8_e4m3fn):
    """Pre-compute a delayed (per-tensor) scale from a reference norm output.

    Simulates a ScalingManager that recorded the amax in a previous iteration.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    amax = norm_ref.abs().max().float().clamp(min=1e-12)
    return (amax / fp8_max).unsqueeze(0)


def _dequant_output(x_fp8, scale, scaling_type, block_size=128, norm_ref=None):
    """Dequantize fused norm+quant output using torchao reference primitives."""
    if scaling_type == "delayed":
        # delayed: scale = amax / fp8_max (aiter convention); torchao dequant = x * scale
        return torchao_dequant_fp8(x_fp8, scale.float(), output_dtype=torch.bfloat16)
    elif scaling_type == "dynamic":
        # dynamic: fused kernel does per-token quant internally, returns
        # max(per_row_scales) as a single scalar.  For accurate dequant we
        # need the actual per-row scales; derive them from the reference
        # norm output when available.
        # NOTE: The scalar fallback (when norm_ref is None) is approximate —
        # rows whose actual per-row scale < max will be under-dequantized.
        if norm_ref is not None:
            fp8_max = torch.finfo(x_fp8.dtype).max
            per_row = (
                norm_ref.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
                / fp8_max
            )
            return torchao_dequant_fp8(x_fp8, per_row, output_dtype=torch.bfloat16)
        return torchao_dequant_fp8(x_fp8, scale.float(), output_dtype=torch.bfloat16)
    elif scaling_type == "per_token":
        return torchao_dequant_fp8(x_fp8, scale.float(), output_dtype=torch.bfloat16)
    elif scaling_type in ("blockwise", "blockwise2d"):
        return torchao_dequant_fp8(x_fp8, scale, output_dtype=torch.bfloat16)
    elif scaling_type == "mxfp8":
        # torchao MX-format dequantization (CPU-based OCP spec reference).
        # Assumes Lumen's MXFP8 block layout matches torchao's OCP layout.
        x_deq = torchao_to_dtype(
            x_fp8.cpu().reshape(-1),
            scale.cpu().reshape(-1).to(torch.uint8),
            torch.float8_e4m3fn,
            block_size,
            torch.float32,
        )
        return x_deq.reshape(x_fp8.shape).cuda().to(torch.bfloat16)
    else:
        return x_fp8


def _blockwise_quant_ref(x, block_size, fp8_dtype, axis=1):
    """Pure PyTorch blockwise FP8 quantization reference (axis=1 only)."""
    M, N = x.shape
    fp8_max = torch.finfo(fp8_dtype).max
    x_f32 = x.float()
    x_blocked = x_f32.reshape(M, N // block_size, block_size)
    amax = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scales = amax / fp8_max
    x_scaled = (x_blocked / scales).clamp(-fp8_max, fp8_max)
    x_fp8 = x_scaled.reshape(M, N).to(fp8_dtype)
    scales = scales.squeeze(-1)
    return x_fp8, scales


def _quant_golden(x, scaling_type, block_size=128):
    """Reference quantization using torchao (golden for fused norm+quant output).

    For tensorwise / per_token: torchao _quantize_affine_float8.
    For blockwise: torchao fp8_blockwise_act_quant (PyTorch fallback if unavailable).
    For mxfp8: torchao to_mx with ScaleCalculationMode.EVEN.
    """
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    if scaling_type == "delayed":
        amax = x.abs().max().float().clamp(min=1e-12)
        scale = (amax / fp8_max).unsqueeze(0)
        x_fp8 = torchao_quant_fp8(x, scale, fp8_dtype)
        return x_fp8, scale

    elif scaling_type in ("dynamic", "per_token"):
        amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
        scale = amax / fp8_max
        x_fp8 = torchao_quant_fp8(x, scale, fp8_dtype)
        return x_fp8, scale.squeeze(-1)

    elif scaling_type in ("blockwise", "blockwise2d"):
        if fp8_blockwise_act_quant is not None:
            try:
                x_fp8, scale = fp8_blockwise_act_quant(x, block_size, fp8_dtype)
                return x_fp8, scale
            except (AssertionError, RuntimeError):
                pass
        x_fp8, scale = _blockwise_quant_ref(x, block_size, fp8_dtype)
        return x_fp8, scale

    elif scaling_type == "mxfp8":
        scale, data_lp = torchao_to_mx(
            x.cpu().float(),
            fp8_dtype,
            block_size,
            scaling_mode=ScaleCalculationMode.EVEN,
        )
        return data_lp, scale

    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type}")


# ===================================================================
# RMSNorm forward + backward
# ===================================================================


@pytest.mark.parametrize("config", NORM_SHAPES, ids=NORM_IDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_fwd_bwd(config, dtype):
    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(config.N, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)

    out_ref = rmsnorm_ref(x_ref, w_ref, eps=1e-6)
    loss_ref = out_ref.float().mean()
    loss_ref.backward()

    out = norm_ops.rmsnorm(x, weight, eps=1e-6)
    loss = out.float().mean()
    loss.backward()

    out_snr = compute_snr(out_ref, out)
    dx_snr = compute_snr(x_ref.grad, x.grad)
    dw_snr = compute_snr(w_ref.grad, weight.grad)

    min_snr = 30 if dtype == torch.float32 else 20
    assert out_snr > min_snr, f"RMSNorm output SNR: {out_snr:.1f} dB"
    assert dx_snr > min_snr - 5, f"RMSNorm dx SNR: {dx_snr:.1f} dB"
    assert dw_snr > min_snr - 5, f"RMSNorm dw SNR: {dw_snr:.1f} dB"


# Per-head QK norms: many rows, each too short to fill a wavefront. This is the
# shape that routes to Lumen's row-tiling kernels instead of AITER's.
NARROW_SHAPES = [
    NormConfig(524288, 128),
    NormConfig(131072, 128),
    NormConfig(65535, 64),
]


@pytest.mark.parametrize("config", NARROW_SHAPES, ids=[repr(c) for c in NARROW_SHAPES])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_narrow_rows(config, dtype):
    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(config.N, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)

    go = torch.randn_like(x)
    rmsnorm_ref(x_ref, w_ref, eps=1e-6).backward(go)
    norm_ops.rmsnorm(x, weight, eps=1e-6).backward(go)

    min_snr = 30 if dtype == torch.float32 else 20
    for label, ref, got in (
        (
            "output",
            rmsnorm_ref(x_ref, w_ref, eps=1e-6),
            norm_ops.rmsnorm(x, weight, eps=1e-6),
        ),
        ("dx", x_ref.grad, x.grad),
        ("dw", w_ref.grad, weight.grad),
    ):
        snr = compute_snr(ref, got)
        assert snr > min_snr - 5, f"narrow RMSNorm {label} SNR: {snr:.1f} dB"


def test_rmsnorm_narrow_rows_handles_noncontiguous_input():
    """Megatron hands per-head norms a strided view out of the fused QKV output."""
    s, b, heads, head_dim = 512, 2, 8, 128
    qkv = torch.randn(s, b, heads, head_dim * 3, device="cuda", dtype=torch.bfloat16)
    x = qkv[..., :head_dim]
    assert not x.reshape(-1, head_dim).is_contiguous()

    weight = torch.randn(
        head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_lumen = x.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)

    go = torch.randn_like(x)
    out = norm_ops.rmsnorm(x_lumen, weight, eps=1e-6)
    out.backward(go)
    rmsnorm_ref(x_ref, w_ref, eps=1e-6).backward(go)

    assert out.shape == x.shape
    assert compute_snr(x_ref.grad, x_lumen.grad) > 15


# ===================================================================
# RMSNorm with fused quantization — forward
# ===================================================================


@pytest.mark.parametrize("config", NORM_SHAPES[:5], ids=NORM_IDS[:5])
@pytest.mark.parametrize("scaling_type", QUANT_SCALING_TYPES)
def test_rmsnorm_with_quant(config, scaling_type):
    dtype = torch.bfloat16
    block_size = 32 if scaling_type == "mxfp8" else 128

    if scaling_type == "mxfp8" and (
        config.N % block_size != 0 or config.M % block_size != 0
    ):
        pytest.skip(f"M={config.M}/N={config.N} not aligned to block_size={block_size}")
    if scaling_type in ("blockwise", "blockwise2d") and config.N % block_size != 0:
        pytest.skip(f"N={config.N} not divisible by block_size={block_size}")

    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype)
    weight = torch.randn(config.N, device="cuda", dtype=dtype)

    out_ref = rmsnorm_ref(x, weight, eps=1e-6)

    out_lumen = norm_ops.rmsnorm(x, weight, eps=1e-6)
    norm_snr = compute_snr(out_ref, out_lumen)
    assert norm_snr > 20, f"RMSNorm Lumen vs ref SNR: {norm_snr:.1f} dB"

    delayed_scale = _delayed_scale(out_ref) if scaling_type == "delayed" else None
    x_fp8, scale = norm_ops.rmsnorm_with_quant(
        x,
        weight,
        eps=1e-6,
        scaling_type=scaling_type,
        scale=delayed_scale,
        block_size=block_size,
    )

    out_dequant = _dequant_output(
        x_fp8, scale, scaling_type, block_size, norm_ref=out_ref
    )
    snr = compute_snr(out_ref, out_dequant)
    assert snr > 8, f"RMSNorm+quant({scaling_type}) SNR: {snr:.1f} dB"

    golden_fp8, _ = _quant_golden(out_ref, scaling_type, block_size)
    lumen_bytes = x_fp8.reshape(-1).cpu().view(torch.uint8)
    golden_bytes = golden_fp8.reshape(-1).cpu().view(torch.uint8)
    fp8_match = (lumen_bytes == golden_bytes).float().mean().item()
    assert (
        fp8_match >= 0.75
    ), f"RMSNorm fused FP8 vs torchao golden byte match: {fp8_match:.2%}"


# ===================================================================
# RMSNorm with fused quantization — backward (FP8 training simulation)
# ===================================================================


@pytest.mark.parametrize("config", NORM_SHAPES[:5], ids=NORM_IDS[:5])
@pytest.mark.parametrize("scaling_type", QUANT_SCALING_TYPES)
def test_rmsnorm_quant_bwd(config, scaling_type):
    """Simulate FP8 training: norm -> quant -> dequant -> linear -> MSE loss -> backward.

    The reference path has no quantization, so gradients only approximately
    match due to quantization noise propagating through the matmul.  The SNR
    threshold is a regression guard, not a strict correctness check.
    """
    dtype = torch.bfloat16
    block_size = 32 if scaling_type == "mxfp8" else 128
    N_out = min(config.N, 256)

    if scaling_type == "mxfp8" and (
        config.N % block_size != 0 or config.M % block_size != 0
    ):
        pytest.skip(f"M={config.M}/N={config.N} not aligned to block_size={block_size}")
    if scaling_type in ("blockwise", "blockwise2d") and config.N % block_size != 0:
        pytest.skip(f"N={config.N} not divisible by block_size={block_size}")

    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype)
    weight = torch.randn(config.N, device="cuda", dtype=dtype)
    w_proj = torch.randn(N_out, config.N, device="cuda", dtype=dtype) * 0.02
    target = torch.randn(config.M, N_out, device="cuda", dtype=dtype)

    # Reference: norm → linear → MSE (no quantization)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    out_ref = rmsnorm_ref(x_ref, w_ref, eps=1e-6)
    logits_ref = out_ref @ w_proj.T
    loss_ref = F.mse_loss(logits_ref, target)
    loss_ref.backward()

    # Lumen: norm → STE(quant→dequant) → linear → MSE
    x_lumen = x.detach().clone().requires_grad_(True)
    w_lumen = weight.detach().clone().requires_grad_(True)
    out_norm = norm_ops.rmsnorm(x_lumen, w_lumen, eps=1e-6)

    norm_ref_for_dequant = rmsnorm_ref(x, weight, eps=1e-6)
    delayed_scale = (
        _delayed_scale(norm_ref_for_dequant) if scaling_type == "delayed" else None
    )

    with torch.no_grad():
        x_fp8, scale = norm_ops.rmsnorm_with_quant(
            x,
            weight,
            eps=1e-6,
            scaling_type=scaling_type,
            scale=delayed_scale,
            block_size=block_size,
        )
        out_dequant = _dequant_output(
            x_fp8,
            scale,
            scaling_type,
            block_size,
            norm_ref=norm_ref_for_dequant,
        ).to(dtype)

    out_ste = out_norm + (out_dequant - out_norm).detach()
    logits = out_ste @ w_proj.T
    loss = F.mse_loss(logits, target)
    loss.backward()

    dx_snr = compute_snr(x_ref.grad, x_lumen.grad)
    dw_snr = compute_snr(w_ref.grad, w_lumen.grad)
    assert dx_snr > 8, f"RMSNorm+quant({scaling_type}) bwd dx SNR: {dx_snr:.1f} dB"
    assert dw_snr > 8, f"RMSNorm+quant({scaling_type}) bwd dw SNR: {dw_snr:.1f} dB"


# ===================================================================
# LayerNorm forward + backward
# ===================================================================


@pytest.mark.parametrize("config", NORM_SHAPES, ids=NORM_IDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_bias", [False, True])
def test_layernorm_fwd_bwd(config, dtype, use_bias):
    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(config.N, device="cuda", dtype=dtype, requires_grad=True)
    bias = (
        torch.randn(config.N, device="cuda", dtype=dtype, requires_grad=True)
        if use_bias
        else None
    )

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    b_ref = bias.detach().clone().requires_grad_(True) if bias is not None else None

    out_ref = layernorm_ref(x_ref, w_ref, b_ref, eps=1e-5)
    loss_ref = out_ref.float().mean()
    loss_ref.backward()

    out = norm_ops.layernorm(x, weight, bias=bias, eps=1e-5)
    loss = out.float().mean()
    loss.backward()

    out_snr = compute_snr(out_ref, out)
    dx_snr = compute_snr(x_ref.grad, x.grad)
    dw_snr = compute_snr(w_ref.grad, weight.grad)

    min_snr = 30 if dtype == torch.float32 else 20
    assert out_snr > min_snr, f"LayerNorm output SNR: {out_snr:.1f} dB"
    assert dx_snr > min_snr - 5, f"LayerNorm dx SNR: {dx_snr:.1f} dB"
    assert dw_snr > min_snr - 5, f"LayerNorm dw SNR: {dw_snr:.1f} dB"

    if use_bias:
        db_snr = compute_snr(b_ref.grad, bias.grad)
        assert db_snr > min_snr - 5, f"LayerNorm dbias SNR: {db_snr:.1f} dB"


# ===================================================================
# LayerNorm with fused quantization — forward
# ===================================================================


@pytest.mark.parametrize("config", NORM_SHAPES[:5], ids=NORM_IDS[:5])
@pytest.mark.parametrize("scaling_type", QUANT_SCALING_TYPES)
def test_layernorm_with_quant(config, scaling_type):
    dtype = torch.bfloat16
    block_size = 32 if scaling_type == "mxfp8" else 128

    if scaling_type == "mxfp8" and (
        config.N % block_size != 0 or config.M % block_size != 0
    ):
        pytest.skip(f"M={config.M}/N={config.N} not aligned to block_size={block_size}")
    if scaling_type in ("blockwise", "blockwise2d") and config.N % block_size != 0:
        pytest.skip(f"N={config.N} not divisible by block_size={block_size}")

    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype)
    weight = torch.randn(config.N, device="cuda", dtype=dtype)
    bias = torch.randn(config.N, device="cuda", dtype=dtype)

    out_ref = layernorm_ref(x, weight, bias, eps=1e-5)

    out_lumen = norm_ops.layernorm(x, weight, bias=bias, eps=1e-5)
    norm_snr = compute_snr(out_ref, out_lumen)
    assert norm_snr > 20, f"LayerNorm Lumen vs ref SNR: {norm_snr:.1f} dB"

    delayed_scale = _delayed_scale(out_ref) if scaling_type == "delayed" else None
    x_fp8, scale = norm_ops.layernorm_with_quant(
        x,
        weight,
        bias,
        eps=1e-5,
        scaling_type=scaling_type,
        scale=delayed_scale,
        block_size=block_size,
    )

    out_dequant = _dequant_output(
        x_fp8, scale, scaling_type, block_size, norm_ref=out_ref
    )
    snr = compute_snr(out_ref, out_dequant)
    assert snr > 8, f"LayerNorm+quant({scaling_type}) SNR: {snr:.1f} dB"

    golden_fp8, _ = _quant_golden(out_ref, scaling_type, block_size)
    lumen_bytes = x_fp8.reshape(-1).cpu().view(torch.uint8)
    golden_bytes = golden_fp8.reshape(-1).cpu().view(torch.uint8)
    fp8_match = (lumen_bytes == golden_bytes).float().mean().item()
    assert (
        fp8_match >= 0.75
    ), f"LayerNorm fused FP8 vs torchao golden byte match: {fp8_match:.2%}"


# ===================================================================
# LayerNorm with fused quantization — backward (FP8 training simulation)
# ===================================================================


@pytest.mark.parametrize("config", NORM_SHAPES[:5], ids=NORM_IDS[:5])
@pytest.mark.parametrize("scaling_type", QUANT_SCALING_TYPES)
def test_layernorm_quant_bwd(config, scaling_type):
    """Simulate FP8 training: norm -> quant -> dequant -> linear -> MSE loss -> backward.

    The reference path has no quantization, so gradients only approximately
    match due to quantization noise propagating through the matmul.  The SNR
    threshold is a regression guard, not a strict correctness check.
    """
    dtype = torch.bfloat16
    block_size = 32 if scaling_type == "mxfp8" else 128
    N_out = min(config.N, 256)

    if scaling_type == "mxfp8" and (
        config.N % block_size != 0 or config.M % block_size != 0
    ):
        pytest.skip(f"M={config.M}/N={config.N} not aligned to block_size={block_size}")
    if scaling_type in ("blockwise", "blockwise2d") and config.N % block_size != 0:
        pytest.skip(f"N={config.N} not divisible by block_size={block_size}")

    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype)
    weight = torch.randn(config.N, device="cuda", dtype=dtype)
    bias = torch.randn(config.N, device="cuda", dtype=dtype)
    w_proj = torch.randn(N_out, config.N, device="cuda", dtype=dtype) * 0.02
    target = torch.randn(config.M, N_out, device="cuda", dtype=dtype)

    # Reference: norm → linear → MSE (no quantization)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    b_ref = bias.detach().clone().requires_grad_(True)
    out_ref = layernorm_ref(x_ref, w_ref, b_ref, eps=1e-5)
    logits_ref = out_ref @ w_proj.T
    loss_ref = F.mse_loss(logits_ref, target)
    loss_ref.backward()

    # Lumen: norm → STE(quant→dequant) → linear → MSE
    x_lumen = x.detach().clone().requires_grad_(True)
    w_lumen = weight.detach().clone().requires_grad_(True)
    b_lumen = bias.detach().clone().requires_grad_(True)
    out_norm = norm_ops.layernorm(x_lumen, w_lumen, bias=b_lumen, eps=1e-5)
    norm_ref_for_dequant = layernorm_ref(x, weight, bias, eps=1e-5)
    delayed_scale = (
        _delayed_scale(norm_ref_for_dequant) if scaling_type == "delayed" else None
    )

    with torch.no_grad():
        x_fp8, scale = norm_ops.layernorm_with_quant(
            x,
            weight,
            bias,
            eps=1e-5,
            scaling_type=scaling_type,
            scale=delayed_scale,
            block_size=block_size,
        )
        out_dequant = _dequant_output(
            x_fp8,
            scale,
            scaling_type,
            block_size,
            norm_ref=norm_ref_for_dequant,
        ).to(dtype)

    out_ste = out_norm + (out_dequant - out_norm).detach()
    logits = out_ste @ w_proj.T
    loss = F.mse_loss(logits, target)
    loss.backward()

    dx_snr = compute_snr(x_ref.grad, x_lumen.grad)
    dw_snr = compute_snr(w_ref.grad, w_lumen.grad)
    db_snr = compute_snr(b_ref.grad, b_lumen.grad)
    assert dx_snr > 8, f"LayerNorm+quant({scaling_type}) bwd dx SNR: {dx_snr:.1f} dB"
    assert dw_snr > 8, f"LayerNorm+quant({scaling_type}) bwd dw SNR: {dw_snr:.1f} dB"
    assert db_snr > 8, f"LayerNorm+quant({scaling_type}) bwd dbias SNR: {db_snr:.1f} dB"


# ===================================================================
# LumenRMSNorm / LumenLayerNorm nn.Module wrappers
# ===================================================================


@pytest.mark.parametrize("config", NORM_SHAPES[:3], ids=NORM_IDS[:3])
def test_lumen_rmsnorm_module(config):
    dtype = torch.bfloat16
    module = norm_ops.LumenRMSNorm(config.N, eps=1e-6).to("cuda", dtype)
    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    out = module(x)
    out_ref = rmsnorm_ref(x_ref, module.weight.detach(), eps=1e-6)
    assert out.shape == x.shape
    assert compute_snr(out_ref, out) > 20, "LumenRMSNorm module output mismatch"

    loss = out.float().mean()
    loss.backward()
    out_ref.float().mean().backward()
    assert x.grad is not None
    assert compute_snr(x_ref.grad, x.grad) > 15, "LumenRMSNorm module dx mismatch"


@pytest.mark.parametrize("config", NORM_SHAPES[:3], ids=NORM_IDS[:3])
def test_lumen_layernorm_module(config):
    dtype = torch.bfloat16
    module = norm_ops.LumenLayerNorm(config.N, eps=1e-5).to("cuda", dtype)
    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    out = module(x)
    out_ref = layernorm_ref(
        x_ref,
        module.weight.detach(),
        module.bias.detach(),
        eps=1e-5,
    )
    assert out.shape == x.shape
    assert compute_snr(out_ref, out) > 20, "LumenLayerNorm module output mismatch"

    loss = out.float().mean()
    loss.backward()
    out_ref.float().mean().backward()
    assert x.grad is not None
    assert compute_snr(x_ref.grad, x.grad) > 15, "LumenLayerNorm module dx mismatch"


# ===================================================================
# Torchao benchmark: RMSNorm + MXFP8 fused vs separate norm→torchao quant
# ===================================================================


@pytest.mark.parametrize(
    "config",
    [
        NormConfig(32, 4096),
        NormConfig(256, 4096),
    ],
    ids=["M32_N4096", "M256_N4096"],
)
def test_rmsnorm_mxfp8_vs_torchao(config):
    """
    Compare Lumen's fused rmsnorm+mxfp8 against:
      1) PyTorch rmsnorm reference
      2) Then torchao MXFP8 quantization on the result
    """
    block_size = 32
    if config.N % block_size != 0:
        pytest.skip(f"N={config.N} not divisible by block_size")

    dtype = torch.bfloat16
    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype)
    weight = torch.randn(config.N, device="cuda", dtype=dtype)

    # Step 1: reference norm
    normed_ref = rmsnorm_ref(x, weight, eps=1e-6).float()

    # Step 1b: Lumen pure norm vs reference
    normed_lumen = norm_ops.rmsnorm(x, weight, eps=1e-6).float()
    norm_snr = compute_snr(normed_ref, normed_lumen)
    assert norm_snr > 20, f"RMSNorm Lumen vs ref SNR: {norm_snr:.1f} dB"

    # Step 2: torchao MXFP8 on the normed result
    normed_cpu = normed_ref.cpu()
    scale_torchao, data_lp_torchao = torchao_to_mx(
        normed_cpu,
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    deq_torchao = torchao_to_dtype(
        data_lp_torchao,
        scale_torchao,
        torch.float8_e4m3fn,
        block_size,
        torch.float32,
    ).cuda()

    # Step 3: Lumen fused rmsnorm+mxfp8, dequantized with torchao
    result = norm_ops.rmsnorm_with_quant(
        x,
        weight,
        eps=1e-6,
        scaling_type="mxfp8",
        block_size=block_size,
    )
    x_fp8_lumen, scales_lumen = result
    deq_lumen = _dequant_output(x_fp8_lumen, scales_lumen, "mxfp8", block_size).float()

    snr_vs_torchao = compute_snr(deq_torchao, deq_lumen)
    snr_vs_ref = compute_snr(normed_ref.cuda(), deq_lumen)

    assert (
        snr_vs_torchao > 8
    ), f"Lumen fused rmsnorm+mxfp8 vs torchao SNR: {snr_vs_torchao:.1f} dB"
    assert (
        snr_vs_ref > 6
    ), f"Lumen fused rmsnorm+mxfp8 vs exact norm SNR: {snr_vs_ref:.1f} dB"

    # Step 4: FP8 data byte-level comparison against torchao golden
    fp8_match = (
        (
            x_fp8_lumen.reshape(-1).cpu().view(torch.uint8)
            == data_lp_torchao.reshape(-1).view(torch.uint8)
        )
        .float()
        .mean()
        .item()
    )
    scale_match = (
        (
            scales_lumen.reshape(-1).cpu().view(torch.uint8)
            == scale_torchao.reshape(-1).view(torch.uint8)
        )
        .float()
        .mean()
        .item()
    )
    assert (
        fp8_match >= 0.80
    ), f"Fused rmsnorm+mxfp8 FP8 data vs torchao byte match: {fp8_match:.2%}"
    assert (
        scale_match >= 0.80
    ), f"Fused rmsnorm+mxfp8 E8M0 scale vs torchao byte match: {scale_match:.2%}"


@pytest.mark.parametrize(
    "config",
    [
        NormConfig(32, 4096),
        NormConfig(256, 4096),
    ],
    ids=["M32_N4096", "M256_N4096"],
)
def test_layernorm_mxfp8_vs_torchao(config):
    """
    Compare Lumen's fused layernorm+mxfp8 against:
      1) PyTorch layernorm reference
      2) Then torchao MXFP8 quantization on the result
    """
    block_size = 32
    if config.N % block_size != 0:
        pytest.skip(f"N={config.N} not divisible by block_size")

    dtype = torch.bfloat16
    x = torch.randn(config.M, config.N, device="cuda", dtype=dtype)
    weight = torch.randn(config.N, device="cuda", dtype=dtype)
    bias = torch.randn(config.N, device="cuda", dtype=dtype)

    normed_ref = layernorm_ref(x, weight, bias, eps=1e-5).float()

    normed_lumen = norm_ops.layernorm(x, weight, bias=bias, eps=1e-5).float()
    norm_snr = compute_snr(normed_ref, normed_lumen)
    assert norm_snr > 20, f"LayerNorm Lumen vs ref SNR: {norm_snr:.1f} dB"

    normed_cpu = normed_ref.cpu()
    scale_torchao, data_lp_torchao = torchao_to_mx(
        normed_cpu,
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.EVEN,
    )
    deq_torchao = torchao_to_dtype(
        data_lp_torchao,
        scale_torchao,
        torch.float8_e4m3fn,
        block_size,
        torch.float32,
    ).cuda()

    result = norm_ops.layernorm_with_quant(
        x,
        weight,
        bias,
        eps=1e-5,
        scaling_type="mxfp8",
        block_size=block_size,
    )
    x_fp8_lumen, scales_lumen = result
    deq_lumen = _dequant_output(x_fp8_lumen, scales_lumen, "mxfp8", block_size).float()

    snr_vs_torchao = compute_snr(deq_torchao, deq_lumen)
    snr_vs_ref = compute_snr(normed_ref.cuda(), deq_lumen)
    assert (
        snr_vs_torchao > 8
    ), f"Lumen fused layernorm+mxfp8 vs torchao SNR: {snr_vs_torchao:.1f} dB"
    assert (
        snr_vs_ref > 6
    ), f"Lumen fused layernorm+mxfp8 vs exact norm SNR: {snr_vs_ref:.1f} dB"

    fp8_match = (
        (
            x_fp8_lumen.reshape(-1).cpu().view(torch.uint8)
            == data_lp_torchao.reshape(-1).view(torch.uint8)
        )
        .float()
        .mean()
        .item()
    )
    scale_match = (
        (
            scales_lumen.reshape(-1).cpu().view(torch.uint8)
            == scale_torchao.reshape(-1).view(torch.uint8)
        )
        .float()
        .mean()
        .item()
    )
    assert (
        fp8_match >= 0.80
    ), f"Fused layernorm+mxfp8 FP8 data vs torchao byte match: {fp8_match:.2%}"
    assert (
        scale_match >= 0.80
    ), f"Fused layernorm+mxfp8 E8M0 scale vs torchao byte match: {scale_match:.2%}"
