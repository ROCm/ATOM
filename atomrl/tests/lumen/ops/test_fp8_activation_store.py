# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch


class TestFP8ActivationStore:
    """Gradients with FP8 activation store match eager within tolerance."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bf16_path_gradient_accuracy(self):
        """Path 1 (BF16 / scaling_type='none'): max rel error < 1%."""
        from atomrl.lumen.ops.quantize.linear import quantized_linear

        M, K, N = 128, 256, 512
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        out_ref = quantized_linear(
            x, w, scaling_type="none", fp8_activation_store=False
        )
        out_ref.sum().backward()
        grad_ref = x.grad.clone()
        x.grad = None

        out_fp8 = quantized_linear(x, w, scaling_type="none", fp8_activation_store=True)
        out_fp8.sum().backward()
        grad_fp8 = x.grad.clone()

        rel_error = (grad_ref - grad_fp8).abs().max() / grad_ref.abs().max()
        assert rel_error < 0.01, f"Path 1 gradient rel error {rel_error:.4f} exceeds 1%"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weight_only_fp8_path_gradient_accuracy(self):
        """Path 2 (weight-only FP8 / quantize_activation=False): max rel error < 1%."""
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize import ScalingManager

        M, K, N = 128, 256, 512
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        sm = ScalingManager()

        out_ref = quantized_linear(
            x,
            w,
            scaling_manager=sm,
            quantize_activation=False,
            fp8_activation_store=False,
        )
        out_ref.sum().backward()
        grad_ref = x.grad.clone()
        x.grad = None

        out_fp8 = quantized_linear(
            x,
            w,
            scaling_manager=sm,
            quantize_activation=False,
            fp8_activation_store=True,
        )
        out_fp8.sum().backward()
        grad_fp8 = x.grad.clone()

        rel_error = (grad_ref - grad_fp8).abs().max() / grad_ref.abs().max()
        assert rel_error < 0.01, f"Path 2 gradient rel error {rel_error:.4f} exceeds 1%"
