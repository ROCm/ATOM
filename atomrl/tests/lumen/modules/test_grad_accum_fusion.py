# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for gradient accumulation fusion in quantized linear.

Verifies that when gradient_accumulation_fusion is enabled and weight has
main_grad, gradients accumulate into main_grad instead of param.grad.
"""

import pytest
import torch
from atomrl.lumen.ops.quantize.linear import quantized_linear


def _simple_gemm(a, w, scale_a=None, scale_w=None, bias=None):
    """Simple matmul reference for mocking AITER GEMM. Y = A @ W^T."""
    out = a @ w.T
    if bias is not None:
        out = out + bias
    return out


@pytest.fixture
def mock_gemm():
    """Patch dispatch_gemm to use simple matmul for tests."""
    from unittest import mock

    from atomrl.lumen.ops.quantize import linear as linear_mod
    from atomrl.lumen.quantize.descriptor import FP8Descriptor

    def _mock_dispatch(
        a, w, scale_a=None, scale_w=None, scaling_type="none", bias=None
    ):
        if isinstance(a, FP8Descriptor):
            scale_a = a.scale
            a = a.data
        if isinstance(w, FP8Descriptor):
            scale_w = w.scale
            w = w.data
        if scaling_type == "none":
            return _simple_gemm(a, w, bias=bias)
        if scale_a is not None and isinstance(scale_a, torch.Tensor):
            a = a.to(torch.bfloat16) * scale_a
        if scale_w is not None and isinstance(scale_w, torch.Tensor):
            w = w.to(torch.bfloat16) * scale_w
        return _simple_gemm(a, w, bias=bias)

    with mock.patch.object(linear_mod, "dispatch_gemm", side_effect=_mock_dispatch):
        yield


@pytest.fixture
def mock_gemm_bf16():
    """Patch gemm_bf16 for scaling_type=none path."""
    from unittest import mock

    from atomrl.lumen.ops.quantize import linear as linear_mod

    def _mock_gemm_bf16(a, w, bias=None):
        return _simple_gemm(a, w, bias=bias)

    with mock.patch.object(linear_mod, "gemm_bf16", side_effect=_mock_gemm_bf16):
        yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGradAccumFusion:
    def test_grad_accumulates_into_main_grad(self, mock_gemm_bf16):
        """With fusion enabled and main_grad present, grad accumulates into main_grad, param.grad is None."""
        M, K, N = 4, 64, 128
        device = "cuda"
        dtype = torch.bfloat16

        x = torch.randn(M, K, device=device, dtype=dtype) * 0.1
        w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
        w.requires_grad_(True)

        # Attach main_grad (as used by Megatron main param / grad buffer)
        main_grad = torch.zeros_like(w, device=device, dtype=dtype)
        w.main_grad = main_grad

        out = quantized_linear(
            x,
            w,
            scaling_type="none",
            gradient_accumulation_fusion=True,
        )
        loss = out.float().sum()
        loss.backward()

        # With fusion: grad goes to main_grad, not param.grad
        assert hasattr(w, "main_grad")
        assert w.main_grad is not None
        assert w.grad is None
        assert w.main_grad.abs().sum() > 0, "main_grad should have been updated"

    def test_grad_normal_without_fusion(self, mock_gemm_bf16):
        """Without fusion, normal behavior: param.grad is set, main_grad unchanged."""
        M, K, N = 4, 64, 128
        device = "cuda"
        dtype = torch.bfloat16

        x = torch.randn(M, K, device=device, dtype=dtype) * 0.1
        w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
        w.requires_grad_(True)

        main_grad = torch.zeros_like(w, device=device, dtype=dtype)
        w.main_grad = main_grad
        main_grad_initial = w.main_grad.clone()

        out = quantized_linear(
            x,
            w,
            scaling_type="none",
            gradient_accumulation_fusion=False,
        )
        loss = out.float().sum()
        loss.backward()

        assert w.grad is not None
        assert w.grad.shape == w.shape
        assert w.grad.abs().sum() > 0
        # main_grad should be unchanged (we didn't add to it)
        torch.testing.assert_close(w.main_grad, main_grad_initial)

    def test_grad_accum_multiple_microbatches(self, mock_gemm_bf16):
        """Accumulate gradients over 3 microbatches; main_grad contains the sum."""
        M, K, N = 4, 64, 128
        device = "cuda"
        dtype = torch.bfloat16

        w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
        w.requires_grad_(True)
        main_grad = torch.zeros_like(w, device=device, dtype=dtype)
        w.main_grad = main_grad

        accumulated = torch.zeros_like(w, device=device, dtype=dtype)

        for _ in range(3):
            x = torch.randn(M, K, device=device, dtype=dtype) * 0.1
            out = quantized_linear(
                x,
                w,
                scaling_type="none",
                gradient_accumulation_fusion=True,
            )
            loss = out.float().sum()
            loss.backward()

            # Reference: grad_weight = grad_output.T @ input; for sum loss grad_output = 1
            with torch.no_grad():
                grad_out = torch.ones_like(out)
                grad_w_mb = grad_out.T @ x
                accumulated.add_(grad_w_mb)

        assert w.grad is None
        assert w.main_grad is not None
        torch.testing.assert_close(w.main_grad, accumulated, atol=1e-2, rtol=1e-2)

    def test_grad_accum_with_fp8(self, mock_gemm):
        """Same accumulation test with FP8 scaling type (dynamic).

        Mocks both quantize_input (pass-through) and dispatch_gemm (matmul)
        to avoid AITER dependency.
        """
        from unittest import mock

        from atomrl.lumen.ops.quantize import linear as linear_mod
        from atomrl.lumen.quantize.descriptor import FP8Descriptor

        def _mock_quantize(
            x,
            scaling_type,
            fp8_dtype,
            block_size=128,
            manager=None,
            tensor_id=None,
            backward=False,
            **_kwargs,
        ):
            scale = torch.ones(1, device=x.device, dtype=torch.float32)
            return FP8Descriptor(data=x, scale=scale, fp8_dtype=fp8_dtype)

        def _mock_mixed(a, w, scale_a, scale_w, w_transpose=None):
            rhs = w_transpose if w_transpose is not None else w.transpose(0, 1)
            return (a.float() @ rhs.float()).to(torch.bfloat16)

        M, K, N = 4, 128, 128
        device = "cuda"
        dtype = torch.bfloat16

        w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
        w.requires_grad_(True)
        main_grad = torch.zeros_like(w, device=device, dtype=dtype)
        w.main_grad = main_grad

        with (
            mock.patch.object(linear_mod, "quantize_input", side_effect=_mock_quantize),
            mock.patch.object(
                linear_mod, "gemm_per_tensor_mixed", side_effect=_mock_mixed
            ),
        ):
            for _ in range(2):
                x = torch.randn(M, K, device=device, dtype=dtype) * 0.1
                out = quantized_linear(
                    x,
                    w,
                    scaling_type="dynamic",
                    gradient_accumulation_fusion=True,
                )
                loss = out.float().sum()
                loss.backward()

        assert w.grad is None
        assert w.main_grad is not None
        assert w.main_grad.abs().sum() > 0
