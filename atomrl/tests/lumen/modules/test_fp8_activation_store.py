# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Tests for FP8 activation storage in fused MLP modules.

Covers:
  - Forward shape verification (gated + ungated)
  - Backward gradient flow (gated + ungated)
  - Forward output matches non-FP8 path approximately
  - SNR comparison of backward gradients (FP8 store vs non-FP8 store)
  - Large tensor shapes (realistic MLP dimensions)
  - Multiple activation functions (swiglu, geglu, gelu, relu)
  - No bias mode
"""

import pytest
import torch

DEVICE = "cuda"
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _compute_snr(ref, test):
    ref_f, test_f = ref.float(), test.float()
    signal = ref_f.norm().pow(2)
    noise = (ref_f - test_f).norm().pow(2)
    if noise < 1e-12:
        return float("inf")
    return (10.0 * torch.log10(signal / (noise + 1e-12))).item()


def _copy_params(src, dst):
    """Copy all parameters from src to dst (matching attribute names)."""
    with torch.no_grad():
        for name, param in src.named_parameters():
            dst_param = getattr(dst, name, None)
            if dst_param is not None:
                dst_param.copy_(param)


# =========================================================================
# Gated MLP (SwiGLU)
# =========================================================================


@_CUDA
class TestFP8ActivationStoreGatedMLP:

    def test_forward_shape(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_backward_runs(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert mlp.w_gate.grad is not None

    def test_matches_non_fp8_store_approximately(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        torch.manual_seed(42)
        mlp_ref = LumenGatedMLP(64, 128, fp8_activation_store=False).to(DEVICE)
        mlp_fp8 = LumenGatedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        _copy_params(mlp_ref, mlp_fp8)

        x = torch.randn(2, 16, 64, device=DEVICE)
        out_ref = mlp_ref(x)
        out_fp8 = mlp_fp8(x)
        torch.testing.assert_close(out_ref, out_fp8, atol=1e-5, rtol=1e-5)

    def test_backward_snr_vs_non_fp8(self):
        """FP8 activation store backward gradients should be close to non-FP8."""
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        torch.manual_seed(42)
        mlp_ref = LumenGatedMLP(128, 256, fp8_activation_store=False).to(DEVICE)
        mlp_fp8 = LumenGatedMLP(128, 256, fp8_activation_store=True).to(DEVICE)
        _copy_params(mlp_ref, mlp_fp8)

        x_ref = (torch.randn(4, 32, 128, device=DEVICE) * 0.1).requires_grad_(True)
        x_fp8 = x_ref.detach().clone().requires_grad_(True)

        out_ref = mlp_ref(x_ref)
        out_ref.sum().backward()

        out_fp8 = mlp_fp8(x_fp8)
        out_fp8.sum().backward()

        snr_dx = _compute_snr(x_ref.grad, x_fp8.grad)
        snr_dw_gate = _compute_snr(mlp_ref.w_gate.grad, mlp_fp8.w_gate.grad)
        snr_dw_up = _compute_snr(mlp_ref.w_up.grad, mlp_fp8.w_up.grad)
        snr_dw_down = _compute_snr(mlp_ref.w_down.grad, mlp_fp8.w_down.grad)

        # FP8 quant→dequant adds noise; 10 dB is a reasonable floor
        assert snr_dx > 10, f"FP8 store dX SNR too low: {snr_dx:.1f} dB"
        assert snr_dw_gate > 10, f"FP8 store dW_gate SNR too low: {snr_dw_gate:.1f} dB"
        assert snr_dw_up > 10, f"FP8 store dW_up SNR too low: {snr_dw_up:.1f} dB"
        assert snr_dw_down > 10, f"FP8 store dW_down SNR too low: {snr_dw_down:.1f} dB"

    def test_large_tensor(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(1024, 2048, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 128, 1024, device=DEVICE, requires_grad=True)
        out = mlp(x)
        assert out.shape == (2, 128, 1024)
        out.sum().backward()
        assert x.grad is not None

    def test_no_bias(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(64, 128, bias=False, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None

    @pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
    def test_activations(self, activation):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        mlp = LumenGatedMLP(
            64, 128, activation=activation, fp8_activation_store=True
        ).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        out = mlp(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert x.grad is not None


# =========================================================================
# Ungated MLP
# =========================================================================


@_CUDA
class TestFP8ActivationStoreFusedMLP:

    def test_forward_shape(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_backward_runs(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None

    def test_matches_non_fp8_store_approximately(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        torch.manual_seed(42)
        mlp_ref = LumenFusedMLP(64, 128, fp8_activation_store=False).to(DEVICE)
        mlp_fp8 = LumenFusedMLP(64, 128, fp8_activation_store=True).to(DEVICE)
        _copy_params(mlp_ref, mlp_fp8)

        x = torch.randn(2, 16, 64, device=DEVICE)
        out_ref = mlp_ref(x)
        out_fp8 = mlp_fp8(x)
        torch.testing.assert_close(out_ref, out_fp8, atol=1e-5, rtol=1e-5)

    def test_backward_snr_vs_non_fp8(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        torch.manual_seed(42)
        mlp_ref = LumenFusedMLP(128, 256, fp8_activation_store=False).to(DEVICE)
        mlp_fp8 = LumenFusedMLP(128, 256, fp8_activation_store=True).to(DEVICE)
        _copy_params(mlp_ref, mlp_fp8)

        x_ref = (torch.randn(4, 32, 128, device=DEVICE) * 0.1).requires_grad_(True)
        x_fp8 = x_ref.detach().clone().requires_grad_(True)

        out_ref = mlp_ref(x_ref)
        out_ref.sum().backward()

        out_fp8 = mlp_fp8(x_fp8)
        out_fp8.sum().backward()

        snr_dx = _compute_snr(x_ref.grad, x_fp8.grad)
        snr_dw_up = _compute_snr(mlp_ref.w_up.grad, mlp_fp8.w_up.grad)
        snr_dw_down = _compute_snr(mlp_ref.w_down.grad, mlp_fp8.w_down.grad)

        assert snr_dx > 10, f"FP8 store dX SNR too low: {snr_dx:.1f} dB"
        assert snr_dw_up > 10, f"FP8 store dW_up SNR too low: {snr_dw_up:.1f} dB"
        assert snr_dw_down > 10, f"FP8 store dW_down SNR too low: {snr_dw_down:.1f} dB"

    def test_large_tensor(self):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(1024, 2048, fp8_activation_store=True).to(DEVICE)
        x = torch.randn(2, 128, 1024, device=DEVICE, requires_grad=True)
        out = mlp(x)
        assert out.shape == (2, 128, 1024)
        out.sum().backward()
        assert x.grad is not None

    @pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
    def test_activations(self, activation):
        from atomrl.lumen.modules.fused_mlp import LumenFusedMLP

        mlp = LumenFusedMLP(
            64, 128, activation=activation, fp8_activation_store=True
        ).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, requires_grad=True)
        out = mlp(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert x.grad is not None
