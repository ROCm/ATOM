# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import torch
from atomrl.lumen.ops import fused_swiglu as swiglu


def test_fused_swiglu_uses_aiter_main_forward(monkeypatch):
    x = torch.randn(2, 8, requires_grad=True)
    expected = torch.randn(2, 4)
    monkeypatch.setattr(swiglu, "_AITER_SWIGLU_FWD", lambda value: expected)

    result = swiglu.fused_swiglu(x)
    assert result is expected


def test_fused_swiglu_uses_aiter_main_backward(monkeypatch):
    x = torch.randn(2, 8, requires_grad=True)
    grad_output = torch.randn(2, 4)
    expected = torch.randn(2, 8)
    monkeypatch.setattr(swiglu, "_AITER_SWIGLU_BWD", lambda grad, value: expected)

    result = swiglu.fused_swiglu_backward(grad_output, x)
    assert result is expected
