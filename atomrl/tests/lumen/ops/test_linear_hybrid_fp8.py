# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################
"""
Tests for hybrid FP8 (e4m3 forward / e5m2 backward) GEMM correctness.

Covers:
  - QuantConfig hybrid mode uses E4M3 in forward, E5M2 in backward
  - Forward SNR vs torch.mm reference
  - Backward dX SNR vs reference
  - Hybrid vs non-hybrid dtype verification

SNR thresholds (empirical):
  - Hybrid forward:  >= 12 dB (same as delayed/dynamic e4m3)
  - Hybrid backward: >=  6 dB (e5m2 has wider range but less precision)
"""

import pytest
import torch
from conftest import LinearConfig, compute_snr

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


LINEAR_SHAPES = [
    LinearConfig(32, 256, 512),
    LinearConfig(128, 4096, 4096),
]
LINEAR_IDS = [repr(c) for c in LINEAR_SHAPES]


@_CUDA
class TestHybridFP8Config:
    """Verify that hybrid mode configures the correct FP8 dtypes."""

    def test_hybrid_config_dtypes(self):
        from atomrl.lumen.quantize.config import QuantConfig, QuantFormat

        cfg = QuantConfig(format=QuantFormat.HYBRID)
        fwd_dtype = cfg.torch_dtype
        bwd_dtype = cfg.torch_dtype_bwd
        assert "e4m3" in str(
            fwd_dtype
        ), f"Forward dtype should be e4m3, got {fwd_dtype}"
        assert "e5m2" in str(
            bwd_dtype
        ), f"Backward dtype should be e5m2, got {bwd_dtype}"
        assert fwd_dtype != bwd_dtype

    def test_non_hybrid_uses_same_dtype(self):
        from atomrl.lumen.quantize.config import QuantConfig, QuantFormat

        cfg = QuantConfig(format=QuantFormat.FP8_E4M3)
        assert cfg.torch_dtype == cfg.torch_dtype_bwd

    def test_hybrid_fp8_max_values_differ(self):
        from atomrl.lumen.quantize.config import QuantConfig, QuantFormat

        cfg = QuantConfig(format=QuantFormat.HYBRID)
        fwd_max = torch.finfo(cfg.torch_dtype).max
        bwd_max = torch.finfo(cfg.torch_dtype_bwd).max
        assert bwd_max > fwd_max, "E5M2 should have larger max than E4M3"


@_CUDA
class TestHybridFP8Forward:
    """Hybrid FP8 forward pass uses E4M3 (same as non-hybrid delayed)."""

    @pytest.mark.parametrize("config", LINEAR_SHAPES, ids=LINEAR_IDS)
    def test_forward_snr(self, config):
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import QuantConfig, QuantFormat, ScalingType
        from atomrl.lumen.quantize.scaling_manager import ScalingManager

        x = torch.randn(config.M, config.K, device="cuda", dtype=torch.bfloat16) * 0.1
        w = torch.randn(config.N, config.K, device="cuda", dtype=torch.bfloat16) * 0.02
        ref = (x.float() @ w.float().T).to(torch.bfloat16)

        cfg = QuantConfig(format=QuantFormat.HYBRID, scaling=ScalingType.DELAYED)
        mgr = ScalingManager(cfg)
        mgr.quantize("input", x, backward=False)
        mgr.quantize("weight", w, backward=False)

        out = quantized_linear(x, w, scaling_type="delayed", scaling_manager=mgr)
        snr = compute_snr(ref, out)
        assert snr > 12, f"Hybrid FP8 forward SNR too low: {snr:.1f} dB"


@_CUDA
class TestHybridFP8Backward:
    """Hybrid FP8 backward pass uses E5M2 for gradient quantization."""

    @pytest.mark.parametrize("config", LINEAR_SHAPES, ids=LINEAR_IDS)
    def test_backward_dgrad_snr(self, config):
        """dX = grad_output @ W should match bf16 reference within SNR threshold."""
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import QuantConfig, QuantFormat, ScalingType
        from atomrl.lumen.quantize.scaling_manager import ScalingManager

        x = (
            torch.randn(config.M, config.K, device="cuda", dtype=torch.bfloat16) * 0.1
        ).requires_grad_(True)
        w = torch.randn(config.N, config.K, device="cuda", dtype=torch.bfloat16) * 0.02

        cfg = QuantConfig(format=QuantFormat.HYBRID, scaling=ScalingType.DELAYED)
        mgr = ScalingManager(cfg)
        mgr.quantize("input", x.detach(), backward=False)
        mgr.quantize("weight", w, backward=False)

        out = quantized_linear(x, w, scaling_type="delayed", scaling_manager=mgr)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)

        assert x.grad is not None, "Backward did not produce gradient"
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"

        ref_dgrad = (grad_output.float() @ w.float()).to(torch.bfloat16)
        snr = compute_snr(ref_dgrad, x.grad)
        assert snr > 6, f"Hybrid FP8 backward dgrad SNR too low: {snr:.1f} dB"

    @pytest.mark.parametrize("config", LINEAR_SHAPES[:1], ids=LINEAR_IDS[:1])
    def test_backward_wgrad_snr(self, config):
        """dW = grad_output^T @ X should be close to bf16 reference."""
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import QuantConfig, QuantFormat, ScalingType
        from atomrl.lumen.quantize.scaling_manager import ScalingManager

        x = torch.randn(config.M, config.K, device="cuda", dtype=torch.bfloat16) * 0.1
        w_param = torch.nn.Parameter(
            torch.randn(config.N, config.K, device="cuda", dtype=torch.bfloat16) * 0.02
        )

        cfg = QuantConfig(format=QuantFormat.HYBRID, scaling=ScalingType.DELAYED)
        mgr = ScalingManager(cfg)
        mgr.quantize("input", x, backward=False)
        mgr.quantize("weight", w_param.detach(), backward=False)

        out = quantized_linear(x, w_param, scaling_type="delayed", scaling_manager=mgr)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)

        assert w_param.grad is not None, "No weight gradient produced"
        ref_wgrad = (grad_output.float().t() @ x.float()).to(torch.bfloat16)
        snr = compute_snr(ref_wgrad, w_param.grad)
        assert snr > 6, f"Hybrid FP8 backward wgrad SNR too low: {snr:.1f} dB"

    def test_hybrid_uses_e5m2_in_backward(self):
        """Verify the ScalingManager returns E5M2 dtype for backward quantization."""
        from atomrl.lumen.quantize.config import QuantConfig, QuantFormat, ScalingType
        from atomrl.lumen.quantize.scaling_manager import ScalingManager

        cfg = QuantConfig(format=QuantFormat.HYBRID, scaling=ScalingType.DELAYED)
        mgr = ScalingManager(cfg)
        assert "e5m2" in str(
            mgr.fp8_dtype_bwd
        ), f"Hybrid backward should use E5M2, got {mgr.fp8_dtype_bwd}"
        assert "e4m3" in str(
            mgr.fp8_dtype
        ), f"Hybrid forward should use E4M3, got {mgr.fp8_dtype}"
