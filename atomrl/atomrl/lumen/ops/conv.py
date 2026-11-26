# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Autograd-safe Conv2d and Conv3d routing through public AITER APIs."""

from __future__ import annotations

import logging
import types

import torch
import torch.nn.functional as F
from aiter.ops.triton.conv.conv2d import conv2d
from aiter.ops.triton.conv.conv3d import conv3d
from torch import nn
from torch.nn.grad import conv2d_input, conv2d_weight, conv3d_input, conv3d_weight

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_CALLS = {"aiter": 0, "torch": 0}


class _AiterConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, spatial):
        ctx.save_for_backward(x, weight)
        ctx.settings = (stride, padding, dilation, bias is not None, spatial)
        op = conv3d if spatial == 3 else conv2d
        return op(
            x.contiguous(),
            weight.to(dtype=x.dtype),
            None if bias is None else bias.to(dtype=x.dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride, padding, dilation, has_bias, spatial = ctx.settings
        weight_for_grad = weight.to(dtype=grad_output.dtype)
        input_grad = conv3d_input if spatial == 3 else conv2d_input
        weight_grad = conv3d_weight if spatial == 3 else conv2d_weight
        reduce_dims = (0, 2, 3, 4) if spatial == 3 else (0, 2, 3)

        grad_x = (
            input_grad(
                x.shape,
                weight_for_grad,
                grad_output,
                stride,
                padding,
                dilation,
            )
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weight = (
            weight_grad(
                x,
                weight.shape,
                grad_output,
                stride,
                padding,
                dilation,
            )
            if ctx.needs_input_grad[1]
            else None
        )
        if grad_weight is not None:
            grad_weight = grad_weight.to(dtype=weight.dtype)
        grad_bias = (
            grad_output.sum(reduce_dims)
            if has_bias and ctx.needs_input_grad[2]
            else None
        )
        return grad_x, grad_weight, grad_bias, None, None, None, None


def conv(
    inputs,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Apply AITER convolution when supported, with a PyTorch fallback."""
    spatial = weight.dim() - 2
    if (
        spatial not in (2, 3)
        or not inputs.is_cuda
        or inputs.dtype not in _SUPPORTED_DTYPES
        or groups != 1
    ):
        _CALLS["torch"] += 1
        op = F.conv3d if spatial == 3 else F.conv2d
        return op(inputs, weight, bias, stride, padding, dilation, groups)
    try:
        output = _AiterConv.apply(
            inputs,
            weight,
            bias,
            tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * spatial,
            (
                tuple(padding)
                if isinstance(padding, (list, tuple))
                else (padding,) * spatial
            ),
            (
                tuple(dilation)
                if isinstance(dilation, (list, tuple))
                else (dilation,) * spatial
            ),
            spatial,
        )
        _CALLS["aiter"] += 1
        return output
    except (AssertionError, RuntimeError, ValueError) as error:
        logger.warning(
            "AITER conv%dd rejected weight %s; using PyTorch: %s",
            spatial,
            tuple(weight.shape),
            error,
        )
        _CALLS["torch"] += 1
        op = F.conv3d if spatial == 3 else F.conv2d
        return op(inputs, weight, bias, stride, padding, dilation, groups)


def conv_calls() -> dict[str, int]:
    """Return AITER/PyTorch routing counts."""
    return dict(_CALLS)


def _supported(module: nn.modules.conv._ConvNd) -> bool:
    return (
        module.groups == 1
        and module.padding_mode == "zeros"
        and not isinstance(module.padding, str)
    )


def _bind(module: nn.modules.conv._ConvNd) -> None:
    if getattr(module._conv_forward, "_lumen_conv", False):
        return
    original = module._conv_forward

    def _conv_forward(self, inputs, weight, bias):
        if self.padding_mode != "zeros" or isinstance(self.padding, str):
            return original(inputs, weight, bias)
        return conv(
            inputs,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    _conv_forward._lumen_conv = True
    module._conv_forward = types.MethodType(_conv_forward, module)


def patch_conv(model: nn.Module) -> tuple[int, int]:
    """Patch supported Conv2d/Conv3d modules and return their counts."""
    count_2d = 0
    count_3d = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Conv3d)) or not _supported(module):
            continue
        if isinstance(module, nn.Conv3d):
            _bind(module)
            count_3d += 1
        else:
            _bind(module)
            count_2d += 1
    return count_2d, count_3d


__all__ = ["conv", "conv_calls", "patch_conv"]
