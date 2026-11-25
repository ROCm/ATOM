# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.nn.functional as F
from atomrl.lumen.modules.quantize import LumenLinear


def compute_snr(ref, test):
    ref_f, test_f = ref.float(), test.float()
    noise = ref_f - test_f
    signal_power = (ref_f**2).mean()
    noise_power = (noise**2).mean()
    if noise_power == 0:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


class TestLumenLinearConstruction:
    def test_triton_backend_construction(self):
        linear = LumenLinear(64, 128, backend_type="triton")
        assert linear.in_features == 64
        assert linear.out_features == 128
        assert linear.backend_type == "triton"
        assert linear.block_size == 128
        assert linear.fp8_dtype is not None
        assert linear.scaling_manager is not None

    def test_bias_true(self):
        linear = LumenLinear(64, 128, bias=True, backend_type="triton")
        assert linear.bias is not None
        assert linear.bias.shape == (128,)
        assert linear.bias.eq(0).all().item()

    def test_bias_false(self):
        linear = LumenLinear(64, 128, bias=False, backend_type="triton")
        assert linear.bias is None

    def test_weight_shape(self):
        linear = LumenLinear(64, 128, backend_type="triton")
        assert linear.weight.shape == (128, 64)

    def test_extra_repr(self):
        linear = LumenLinear(64, 128, backend_type="triton")
        s = linear.extra_repr()
        assert "64" in s
        assert "128" in s
        assert "triton" in s
        assert "block_size" in s


class TestLumenLinearForward:
    def test_forward_shape(self):
        linear = LumenLinear(64, 128, backend_type="triton").cuda().to(torch.bfloat16)
        x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)
        out = linear(x)
        assert out.shape == (2, 32, 128)

    def test_forward_correctness_snr(self):
        linear = LumenLinear(64, 128, backend_type="triton").cuda().to(torch.bfloat16)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        linear.reset_parameters()
        x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        out = linear(x)
        ref = F.linear(x, linear.weight, linear.bias)
        snr = compute_snr(ref, out)
        assert snr > 10, f"Forward SNR {snr:.1f} dB too low"

    def test_forward_no_bias(self):
        linear = (
            LumenLinear(64, 128, bias=False, backend_type="triton")
            .cuda()
            .to(torch.bfloat16)
        )
        x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)
        out = linear(x)
        assert out.shape == (2, 32, 128)


class TestLumenLinearBackward:
    def test_backward_computes_gradients(self):
        linear = LumenLinear(64, 128, backend_type="triton").cuda().to(torch.bfloat16)
        x = torch.randn(
            2, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out = linear(x)
        out.float().mean().backward()
        assert linear.weight.grad is not None
        assert x.grad is not None

    def test_backward_grad_snr(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        w = (
            torch.randn(128, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        b = torch.zeros(128, device="cuda", dtype=torch.bfloat16).requires_grad_(True)

        linear = LumenLinear(64, 128, backend_type="triton").cuda().to(torch.bfloat16)
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)
        x_in = x.clone().requires_grad_(True)
        out = linear(x_in)
        out.float().mean().backward()
        dinput_lumen = x_in.grad
        dweight_lumen = linear.weight.grad
        assert dinput_lumen is not None, "Lumen backward did not populate x_in.grad"
        assert (
            dweight_lumen is not None
        ), "Lumen backward did not populate linear.weight.grad"

        x_ref = x.clone().requires_grad_(True)
        ref = F.linear(x_ref, w, b)
        ref.float().mean().backward()
        dinput_snr = compute_snr(x_ref.grad, dinput_lumen)
        dweight_snr = compute_snr(w.grad, dweight_lumen)
        assert dinput_snr > 8, f"dInput SNR {dinput_snr:.1f} dB too low"
        assert dweight_snr > 8, f"dWeight SNR {dweight_snr:.1f} dB too low"


class TestLumenLinearBenchmark:
    def test_forward_throughput(self):
        linear = (
            LumenLinear(4096, 4096, backend_type="triton").cuda().to(torch.bfloat16)
        )
        x = torch.randn(1, 2048, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            linear(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 10
        start.record()
        for _ in range(iters):
            linear(x)
        end.record()
        torch.cuda.synchronize()
        avg_ms = start.elapsed_time(end) / iters
        print(f"\n[LumenLinear] 2048x4096x4096: {avg_ms:.2f}ms")
