# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Straight-through FP8 activation simulation for DeepSeek-V4."""

from importlib import import_module

import torch


def _load_qat_ops():
    act_quant = import_module("atomrl.lumen.kernels.dsv4.quant.act_quant").act_quant
    per_token_cast_back = import_module("tile_kernels.quant").per_token_cast_back
    return act_quant, per_token_cast_back


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 cast and dequantization."""
    act_quant, per_token_cast_back = _load_qat_ops()
    x_c = x.contiguous()
    y, scale = act_quant(x_c, block_size, "ue8m0")

    width = x_c.size(-1)
    y_flat = y.view(-1, width)
    scale_flat = scale.reshape(y_flat.size(0), width // block_size).contiguous()
    output_dtype = "bf16" if x.dtype == torch.bfloat16 else "fp32"
    out_flat = per_token_cast_back((y_flat, scale_flat), output_dtype, block_size)
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
