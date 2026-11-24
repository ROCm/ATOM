# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import torch
from torch import nn


class TestQuantizeParamToFP8:

    def test_basic_quantize_dequantize(self):
        from atomrl.lumen.quantize.fp8_params import (
            dequantize_param_from_fp8,
            quantize_param_to_fp8,
        )

        param = torch.randn(64, 128)
        fp8_param, scale = quantize_param_to_fp8(param)
        recovered = dequantize_param_from_fp8(fp8_param, scale)
        # SNR should be reasonable for FP8
        noise = (param - recovered).norm()
        signal = param.norm()
        snr = 20 * torch.log10(signal / noise.clamp(min=1e-12))
        assert snr > 8.0, f"SNR too low: {snr:.1f} dB"

    def test_scale_is_positive(self):
        from atomrl.lumen.quantize.fp8_params import quantize_param_to_fp8

        param = torch.randn(32, 32)
        _, scale = quantize_param_to_fp8(param)
        assert scale > 0

    def test_zero_param(self):
        from atomrl.lumen.quantize.fp8_params import quantize_param_to_fp8

        param = torch.zeros(16, 16)
        _fp8_param, scale = quantize_param_to_fp8(param)
        assert torch.isfinite(scale)


class TestFP8ParamManager:

    def test_quantize_linear_params(self):
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        mgr = FP8ParamManager()
        count = mgr.quantize_params(model)
        assert count == 2  # two linear layers

    def test_register_hooks(self):
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        mgr = FP8ParamManager()
        mgr.quantize_params(model)
        hook_count = mgr.register_dequant_hooks(model)
        assert hook_count == 2

    def test_remove_hooks(self):
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        model = nn.Sequential(nn.Linear(64, 128))
        mgr = FP8ParamManager()
        mgr.quantize_params(model)
        mgr.register_dequant_hooks(model)
        mgr.remove_hooks()
        assert len(mgr._hooks) == 0

    def test_memory_savings(self):
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        model = nn.Sequential(nn.Linear(64, 128))
        mgr = FP8ParamManager()
        mgr.quantize_params(model)
        saved = mgr.memory_savings_bytes(model)
        assert saved > 0

    def test_forward_uses_dequantized_weight(self):
        """Forward pass produces output when FP8 params are dequantized on-the-fly."""
        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        mgr = FP8ParamManager()
        mgr.quantize_params(model)
        mgr.register_dequant_hooks(model)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 64)
        assert torch.isfinite(out).all()
