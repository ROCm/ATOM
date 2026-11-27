# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from atomrl.lumen.config import LumenConfig
from torch import nn


def _relative_error(actual, expected):
    actual = actual.detach().float()
    expected = expected.detach().float()
    return float((actual - expected).norm() / expected.norm())


def test_cpu_convolution_keeps_pytorch_path(monkeypatch):
    def _unexpected_aiter(*_args, **_kwargs):
        raise AssertionError("CPU convolution must not call AITER")

    for name in ("aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.triton.conv"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, function_name in (("conv2d", "conv2d"), ("conv3d", "conv3d")):
        module_name = f"aiter.ops.triton.conv.{name}"
        module = types.ModuleType(module_name)
        setattr(module, function_name, _unexpected_aiter)
        monkeypatch.setitem(sys.modules, module_name, module)

    path = Path(__file__).parents[3] / "atomrl" / "lumen" / "ops" / "conv.py"
    spec = importlib.util.spec_from_file_location("_test_lumen_conv", path)
    assert spec is not None and spec.loader is not None
    conv_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv_module)

    model = nn.Conv2d(2, 3, 3, padding=1)
    inputs = torch.randn(1, 2, 8, 8)
    expected = model(inputs)
    assert conv_module.patch_conv(model) == (1, 0)

    torch.testing.assert_close(model(inputs), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "kernel,stride,padding",
    [(3, 1, 1), (1, 1, 0), (3, 2, 1), (4, 2, 1)],
)
def test_conv2d_forward_backward_matches_torch(kernel, stride, padding):
    torch.manual_seed(0)
    reference = (
        nn.Conv2d(64, 96, kernel, stride=stride, padding=padding).cuda().bfloat16()
    )
    patched = copy.deepcopy(reference)
    LumenConfig(scaling="none", aiter_conv=True).enable(patched)

    inputs = torch.randn(
        2,
        64,
        32,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    patched_inputs = inputs.detach().clone().requires_grad_()
    expected = reference(inputs)
    actual = patched(patched_inputs)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)

    assert _relative_error(actual, expected) < 1e-2
    assert _relative_error(patched_inputs.grad, inputs.grad) < 1e-2
    assert _relative_error(patched.weight.grad, reference.weight.grad) < 1e-2
    assert _relative_error(patched.bias.grad, reference.bias.grad) < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_unsupported_convolutions_keep_torch_forward():
    model = nn.Sequential(
        nn.Conv2d(8, 8, 3, padding=1, groups=8),
        nn.Conv2d(8, 8, 3, padding=1, padding_mode="reflect"),
        nn.Conv2d(8, 8, 3, padding="same"),
        nn.Conv2d(8, 8, 3, padding=1),
    ).cuda()

    LumenConfig(scaling="none", aiter_conv=True).enable(model)

    assert ["_conv_forward" in vars(module) for module in model] == [
        False,
        False,
        False,
        True,
    ]
