# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.core.grad_quant — gradient quantization round-trips.

Covers:
  - quantize_grad_tensor with grad_quant_type=None is identity
  - quantize_grad_tensor with grad_quant_type="fp8" produces a lossy
    round-trip whose SNR is bounded above zero + matches golden reference
  - GRAD_QUANT_TYPES tuple contains expected values
  - Invalid grad_quant_type raises ValueError
  - Benchmarks for gradient quantization throughput
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from atomrl.lumen.core.grad_quant import (
    GRAD_QUANT_TYPES,
    quantize_grad_tensor,
)
from conftest import compute_snr, fp8_quant_dequant_ref

# ===================================================================
# GRAD_QUANT_TYPES enumeration
# ===================================================================


def test_grad_quant_types_contains_expected():
    """GRAD_QUANT_TYPES should include None, fp8, mxfp8, mxfp4, fp4."""
    assert None in GRAD_QUANT_TYPES
    assert "fp8" in GRAD_QUANT_TYPES
    assert "mxfp8" in GRAD_QUANT_TYPES
    assert "mxfp4" in GRAD_QUANT_TYPES
    assert "fp4" in GRAD_QUANT_TYPES


def test_grad_quant_types_matches_the_scaling_manager():
    """This shim duplicates the tuple as a literal, so it can drift silently."""
    from atomrl.lumen.quantize.scaling_manager import GRAD_QUANT_TYPES as _SOURCE

    assert GRAD_QUANT_TYPES == _SOURCE


# ===================================================================
# None pass-through
# ===================================================================


def test_quantize_grad_none_is_identity():
    """grad_quant_type=None should return the tensor unchanged."""
    t = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    result = quantize_grad_tensor(t, None)
    assert result is t


# ===================================================================
# FP8 round-trip
# ===================================================================


def test_quantize_grad_fp8_round_trip():
    """FP8 quant-dequant round-trip should be lossy but high SNR."""
    torch.manual_seed(42)
    t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    result = quantize_grad_tensor(t, "fp8")
    assert result.shape == t.shape
    assert result.dtype == t.dtype
    snr = compute_snr(t, result)
    assert snr > 10, f"FP8 grad quant SNR too low: {snr:.1f} dB"


def test_quantize_grad_fp8_not_identity():
    """FP8 round-trip should actually modify values (not a no-op)."""
    torch.manual_seed(42)
    t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    result = quantize_grad_tensor(t, "fp8")
    assert not torch.equal(t, result), "FP8 round-trip should be lossy"


def test_quantize_grad_fp8_matches_golden():
    """FP8 grad quant should match pure-PyTorch golden reference."""
    torch.manual_seed(42)
    t = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    result = quantize_grad_tensor(t, "fp8")
    golden, _ = fp8_quant_dequant_ref(t)
    snr = compute_snr(golden, result)
    assert snr > 20, f"FP8 grad quant vs golden SNR: {snr:.1f} dB"


# ===================================================================
# MXFP8 round-trip
# ===================================================================


def test_quantize_grad_mxfp8_round_trip():
    """MXFP8 quant-dequant round-trip preserves shape and dtype."""
    torch.manual_seed(42)
    t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    result = quantize_grad_tensor(t, "mxfp8")
    assert result.shape == t.shape
    assert result.dtype == t.dtype
    snr = compute_snr(t, result)
    assert snr > 5, f"MXFP8 grad quant SNR too low: {snr:.1f} dB"


# ===================================================================
# MXFP4 round-trip
# ===================================================================


def test_quantize_grad_mxfp4_round_trip():
    """MXFP4 quant-dequant round-trip preserves shape and dtype.

    E2M1 keeps one mantissa bit, so the floor sits well below the MXFP8 one.
    Measured SNR is 15.8 dB; the floor is set to catch a broken block scale,
    which collapses it to near zero, not to pin a 4-bit format's exact error.
    """
    torch.manual_seed(42)
    t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    result = quantize_grad_tensor(t, "mxfp4")
    assert result.shape == t.shape
    assert result.dtype == t.dtype
    snr = compute_snr(t, result)
    assert snr > 10, f"MXFP4 grad quant SNR too low: {snr:.1f} dB"


def test_quantize_grad_mxfp4_is_lossier_than_mxfp8():
    """A silent fallback to another format is the failure this guards against."""
    torch.manual_seed(42)
    t = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    snr_mxfp4 = compute_snr(t, quantize_grad_tensor(t, "mxfp4"))
    snr_mxfp8 = compute_snr(t, quantize_grad_tensor(t, "mxfp8"))
    assert (
        snr_mxfp4 < snr_mxfp8
    ), f"MXFP4 {snr_mxfp4:.1f} dB should be below MXFP8 {snr_mxfp8:.1f} dB"


# ===================================================================
# Invalid type
# ===================================================================


def test_quantize_grad_invalid_type():
    """Invalid grad_quant_type should raise ValueError."""
    t = torch.randn(4, 8, device="cuda")
    with pytest.raises(ValueError, match="Unknown grad_quant_type"):
        quantize_grad_tensor(t, "invalid_type")


# ===================================================================
# FP4 not implemented
# ===================================================================


def test_quantize_grad_fp4_not_implemented():
    """FP4 should raise NotImplementedError."""
    t = torch.randn(4, 8, device="cuda")
    with pytest.raises(NotImplementedError):
        quantize_grad_tensor(t, "fp4")


# ===================================================================
# Benchmarks
# ===================================================================


class TestGradQuantBenchmark:
    @pytest.mark.parametrize("size", [(128, 256), (1024, 4096), (4096, 4096)])
    def test_fp8_grad_quant_throughput(self, size):
        M, N = size
        t = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        for _ in range(3):
            quantize_grad_tensor(t, "fp8")
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 20

        start.record()
        for _ in range(iters):
            quantize_grad_tensor(t, "fp8")
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        total_bytes = M * N * 2
        bw_gb_s = (total_bytes / (avg_ms / 1000.0)) / (1024**3)
        print(f"\n[GradQuant FP8] {M}x{N}: {avg_ms:.3f}ms, {bw_gb_s:.1f} GB/s")
