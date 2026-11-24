# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for atomrl.lumen.ops.quantize.gemm_primitives — standalone BF16 dgrad/wgrad."""

import pytest
import torch
import torch.nn.functional as F

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@_requires_cuda
class TestComputeDgradBf16:
    def test_matches_f_linear_backward(self):
        """dgrad = grad_output @ weight should match F.linear backward."""
        from atomrl.lumen.ops.quantize.gemm_primitives import compute_dgrad_bf16

        M, N, K = 128, 256, 512
        grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        result = compute_dgrad_bf16(grad_output, weight)

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = F.linear(x, weight)
        y.backward(grad_output)
        expected = x.grad

        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("shape", [(1, 64, 128), (4096, 4096, 14336)])
    def test_various_shapes(self, shape):
        from atomrl.lumen.ops.quantize.gemm_primitives import compute_dgrad_bf16

        M, N, K = shape
        grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        result = compute_dgrad_bf16(grad_output, weight)
        assert result.shape == (M, K)


@_requires_cuda
class TestComputeWgradBf16:
    def test_matches_analytic(self):
        """wgrad = grad_output^T @ input."""
        from atomrl.lumen.ops.quantize.gemm_primitives import compute_wgrad_bf16

        M, N, K = 128, 256, 512
        grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        input_tensor = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        result = compute_wgrad_bf16(grad_output, input_tensor)
        expected = grad_output.t() @ input_tensor

        assert result.shape == (N, K)
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)


@_requires_cuda
class TestMakeWgradClosureBf16:
    def test_accumulates_into_main_grad(self):
        from atomrl.lumen.ops.quantize.gemm_primitives import make_wgrad_closure_bf16

        M, N, K = 128, 256, 512
        grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        input_tensor = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        )
        weight.main_grad = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)

        closure = make_wgrad_closure_bf16(grad_output, input_tensor, weight, True)
        closure()

        expected = grad_output.reshape(-1, N).t() @ input_tensor.reshape(-1, K)
        assert weight.main_grad.abs().sum() > 0
        torch.testing.assert_close(weight.main_grad, expected, atol=1e-1, rtol=1e-1)

    def test_accumulates_into_grad(self):
        from atomrl.lumen.ops.quantize.gemm_primitives import make_wgrad_closure_bf16

        M, N, K = 128, 256, 512
        grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        input_tensor = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        )

        closure = make_wgrad_closure_bf16(grad_output, input_tensor, weight, False)
        closure()

        expected = grad_output.reshape(-1, N).t() @ input_tensor.reshape(-1, K)
        assert weight.grad is not None
        assert weight.grad.shape == (N, K)
        torch.testing.assert_close(weight.grad, expected, atol=1e-1, rtol=1e-1)

    def test_grad_accumulation_across_calls(self):
        """Two closure calls should accumulate (not overwrite) into weight.grad."""
        from atomrl.lumen.ops.quantize.gemm_primitives import make_wgrad_closure_bf16

        M, N, K = 128, 256, 512
        grad_output_1 = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        input_tensor_1 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        grad_output_2 = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        input_tensor_2 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        )

        c1 = make_wgrad_closure_bf16(grad_output_1, input_tensor_1, weight, False)
        c2 = make_wgrad_closure_bf16(grad_output_2, input_tensor_2, weight, False)
        c1()
        c2()

        expected = grad_output_1.reshape(-1, N).t() @ input_tensor_1.reshape(
            -1, K
        ) + grad_output_2.reshape(-1, N).t() @ input_tensor_2.reshape(-1, K)
        torch.testing.assert_close(weight.grad, expected, atol=1e-1, rtol=1e-1)
