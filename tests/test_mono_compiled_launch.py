# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The V4.1 router's FlyDSL launcher goes through its compiled function: a
launch after the first traces nothing. With FlyDSL 0.5.0 a direct call of a
``@flyc.jit`` launcher traces the whole program again on every call."""

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from flydsl.compiler.kernel_function import KernelFunction

from atom.model_ops.deepseek_v41.router import (
    EXPERTS,
    HIDDEN,
    router_logits,
    router_supported,
)


def test_the_v41_router_does_not_trace_again(monkeypatch):
    x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(EXPERTS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    if not router_supported(x, weight):
        pytest.skip("a gfx950 kernel")
    first = router_logits(x, weight)
    emitted = []
    emit = KernelFunction._emit_kernel
    monkeypatch.setattr(
        KernelFunction,
        "_emit_kernel",
        lambda *a, **k: emitted.append(1) or emit(*a, **k),
    )
    again = router_logits(x, weight)
    torch.cuda.synchronize()
    assert torch.equal(first, again)
    assert not emitted
