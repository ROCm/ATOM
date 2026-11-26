# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################
"""
Tests for atomrl.lumen.ops.gemm.grouped_gemm: grouped GEMM for MoE.

Covers:
  - BF16 grouped GEMM forward — all shapes, expert counts, N!=K
  - BF16 grouped GEMM forward with bias
  - BF16 grouped GEMM wgrad
  - Zero-group edge cases
  - FP8 scaling_type dispatch (blockwise, blockwise2d)
  - Invalid scaling_type error
"""

import pytest
import torch
from atomrl.lumen.ops.gemm.grouped_gemm import grouped_gemm, grouped_gemm_wgrad
from conftest import compute_snr, grouped_gemm_ref


def _make_group_sizes(num_experts, tokens, device):
    """Create group_sizes that sum to tokens, distributed across experts."""
    base = tokens // num_experts
    remainder = tokens % num_experts
    sizes = [base + (1 if i < remainder else 0) for i in range(num_experts)]
    return torch.tensor(sizes, dtype=torch.int32, device=device)


def _wgrad_ref(grad_output, input_tensor, group_sizes):
    """Per-expert weight gradient reference: wgrad[g] = grad_g^T @ input_g."""
    num_experts = len(group_sizes)
    N = grad_output.shape[-1]
    K = input_tensor.shape[-1]
    wgrad = torch.zeros(
        num_experts, N, K, device=grad_output.device, dtype=grad_output.dtype
    )
    offset = 0
    for g, size in enumerate(group_sizes.tolist()):
        size = int(size)
        if size == 0:
            continue
        wgrad[g] = (
            grad_output[offset : offset + size].T @ input_tensor[offset : offset + size]
        )
        offset += size
    return wgrad


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

EXPERT_COUNTS = [1, 4, 8]
TOKEN_COUNTS = [64, 256]
# N != K shapes to verify independent input/output dims
SHAPES_NK = [(256, 256), (256, 512), (512, 256)]
SHAPE_IDS = [f"N{n}_K{k}" for n, k in SHAPES_NK]


# ===================================================================
# BF16 grouped GEMM forward
# ===================================================================


@pytest.mark.parametrize("num_experts", EXPERT_COUNTS)
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("N,K", SHAPES_NK, ids=SHAPE_IDS)
def test_grouped_gemm_fwd(num_experts, tokens, N, K):
    """BF16 grouped GEMM forward: compare against grouped_gemm_ref."""
    dtype = torch.bfloat16
    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes)

    rhs_lumen = rhs_ref.transpose(1, 2)
    out_lumen = grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type="none")

    assert out_lumen.shape == (total_tokens, N)
    snr = compute_snr(out_ref, out_lumen)
    assert snr > 25, f"Grouped GEMM fwd SNR: {snr:.1f} dB (expected > 25)"


# ===================================================================
# BF16 grouped GEMM forward with bias
# ===================================================================


@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("tokens", [64, 256])
def test_grouped_gemm_bias(num_experts, tokens):
    """Grouped GEMM forward with per-expert bias."""
    dtype = torch.bfloat16
    N, K = 256, 512
    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02
    bias = torch.randn(num_experts, N, device=device, dtype=dtype) * 0.01

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes, bias=bias)

    rhs_lumen = rhs_ref.transpose(1, 2)
    out_lumen = grouped_gemm(
        lhs, rhs_lumen, group_sizes, scaling_type="none", bias=bias
    )

    assert out_lumen.shape == (total_tokens, N)
    snr = compute_snr(out_ref, out_lumen)
    assert snr > 25, f"Grouped GEMM bias SNR: {snr:.1f} dB"


# ===================================================================
# Zero-group edge cases
# ===================================================================


@pytest.mark.parametrize(
    "sizes",
    [
        [16, 0, 16, 0],
        [0, 0, 32, 0],
    ],
    ids=["mixed_zeros", "leading_zeros"],
)
def test_grouped_gemm_zero_groups(sizes):
    """Experts with zero tokens should be handled without error."""
    dtype = torch.bfloat16
    N, K = 256, 256
    device = "cuda"
    num_experts = len(sizes)
    group_sizes = torch.tensor(sizes, dtype=torch.int32, device=device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes)

    rhs_lumen = rhs_ref.transpose(1, 2)
    out_lumen = grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type="none")

    assert out_lumen.shape == (total_tokens, N)
    snr = compute_snr(out_ref, out_lumen)
    assert snr > 25, f"Grouped GEMM zero-groups SNR: {snr:.1f} dB"


# ===================================================================
# BF16 grouped GEMM wgrad
# ===================================================================


@pytest.mark.parametrize("num_experts", [1, 4, 8])
@pytest.mark.parametrize("tokens", [64, 256])
def test_grouped_gemm_wgrad(num_experts, tokens):
    """BF16 grouped GEMM wgrad: compare against per-expert grad^T @ input."""
    dtype = torch.bfloat16
    N, K = 256, 512
    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    grad_output = torch.randn(total_tokens, N, device=device, dtype=dtype) * 0.1
    input_tensor = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1

    wgrad_ref = _wgrad_ref(grad_output, input_tensor, group_sizes)
    wgrad_lumen = grouped_gemm_wgrad(
        grad_output, input_tensor, group_sizes, scaling_type="none"
    )

    assert wgrad_lumen.shape == (num_experts, N, K)
    snr = compute_snr(wgrad_ref, wgrad_lumen)
    assert snr > 20, f"Grouped GEMM wgrad SNR: {snr:.1f} dB (expected > 20)"


# ===================================================================
# FP8 scaling_type dispatch (blockwise, blockwise2d)
# ===================================================================


@pytest.mark.parametrize(
    "scaling_type", ["blockwise", "blockwise2d", "per_token", "mxfp8"]
)
@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("tokens", [64, 256])
def test_grouped_gemm_fp8_scaling(num_experts, tokens, scaling_type):
    """Grouped GEMM forward with blockwise/blockwise2d scaling dispatch."""
    dtype = torch.bfloat16
    N, K = 256, 256
    device = "cuda"
    group_sizes = _make_group_sizes(num_experts, tokens, device)
    total_tokens = int(group_sizes.sum().item())

    lhs = torch.randn(total_tokens, K, device=device, dtype=dtype) * 0.1
    rhs_ref = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02

    out_ref = grouped_gemm_ref(lhs, rhs_ref, group_sizes)

    rhs_lumen = rhs_ref.transpose(1, 2)
    out_lumen = grouped_gemm(lhs, rhs_lumen, group_sizes, scaling_type=scaling_type)

    assert out_lumen.shape == (total_tokens, N)
    snr = compute_snr(out_ref, out_lumen)
    assert snr > 8, f"Grouped GEMM {scaling_type} SNR: {snr:.1f} dB (expected > 8)"


@pytest.mark.parametrize("scaling_type", ["dynamic", "per_token", "mxfp8"])
def test_grouped_gemm_prequantized_scales_are_token_aligned(scaling_type):
    from atomrl.lumen.ops.quantize.linear import quantize_input
    from atomrl.lumen.quantize.config import _get_float8_e4m3

    device = "cuda"
    fp8_dtype = _get_float8_e4m3()
    group_sizes = torch.tensor([17, 13, 19, 15], device=device, dtype=torch.int32)
    total_tokens = int(group_sizes.sum().item())
    n = k = 128
    lhs = torch.randn(total_tokens, k, device=device, dtype=torch.bfloat16) * 0.1
    rhs = torch.randn(4, n, k, device=device, dtype=torch.bfloat16) * 0.02
    reference = grouped_gemm_ref(lhs, rhs, group_sizes)

    lhs_desc = quantize_input(lhs, scaling_type, fp8_dtype)
    weight_descs = [quantize_input(weight, scaling_type, fp8_dtype) for weight in rhs]
    grouped_weight = torch.stack(
        [descriptor.data.T.contiguous() for descriptor in weight_descs]
    )
    grouped_weight_scale = torch.stack(
        [descriptor.scale for descriptor in weight_descs]
    )

    result = grouped_gemm(
        lhs_desc.data,
        grouped_weight,
        group_sizes,
        scaling_type=scaling_type,
        x_scale=lhs_desc.scale,
        w_scale=grouped_weight_scale,
    )
    assert result.shape == reference.shape
    assert compute_snr(reference, result) > 8


def test_grouped_gemm_prequantized_mxfp8_slices_padded_width():
    from atomrl.lumen.ops.quantize.linear import quantize_input
    from atomrl.lumen.quantize.config import _get_float8_e4m3

    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith(("gfx950", "gfx1250")):
        pytest.skip(f"AITER native MXFP8 GEMM is unavailable on {arch}")

    fp8_dtype = _get_float8_e4m3()
    group_sizes = torch.tensor([17, 13, 19, 15], device="cuda", dtype=torch.int32)
    lhs = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(4, 50, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    reference = grouped_gemm_ref(lhs, weights, group_sizes)

    lhs_desc = quantize_input(lhs, "mxfp8", fp8_dtype)
    weight_descs = [quantize_input(weight, "mxfp8", fp8_dtype) for weight in weights]
    grouped_weight = torch.stack(
        [descriptor.data.T.contiguous() for descriptor in weight_descs]
    )
    grouped_weight_scale = torch.stack(
        [descriptor.scale for descriptor in weight_descs]
    )
    result = grouped_gemm(
        lhs_desc.data,
        grouped_weight,
        group_sizes,
        scaling_type="mxfp8",
        x_scale=lhs_desc.scale,
        w_scale=grouped_weight_scale,
        output_features=50,
    )

    assert result.shape == (64, 50)
    assert compute_snr(reference, result) > 8


# ===================================================================
# Invalid scaling_type
# ===================================================================


def test_grouped_gemm_invalid_scaling_type():
    """Unknown scaling_type should raise ValueError."""
    dtype = torch.bfloat16
    device = "cuda"
    lhs = torch.randn(32, 256, device=device, dtype=dtype)
    rhs = torch.randn(4, 256, 256, device=device, dtype=dtype)
    group_sizes = torch.tensor([8, 8, 8, 8], dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="Unknown scaling_type"):
        grouped_gemm(lhs, rhs, group_sizes, scaling_type="invalid")
