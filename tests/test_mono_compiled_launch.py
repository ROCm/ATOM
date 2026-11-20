# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""FlyDSL launchers called on the hot path go through their compiled function:
a launch after the first traces nothing. With FlyDSL 0.5.0 a direct call of a
``@flyc.jit`` launcher traces the whole program again on every call."""

import importlib

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

# (target, buffer_ops, ops, program, kernel) of each MonoKernel package
_MONO_KERNEL_PACKAGES = {
    "qwen38": (
        "atom.models.qwen38.mono.kernels.target",
        "atom.models.qwen38.mono.kernels.device.buffer_ops",
        "atom.models.qwen38.mono.kernels.device.ops",
        "atom.models.qwen38.mono.kernels.program",
        "atom.models.qwen38.mono.kernels.kernel",
    ),
    "deepseek_v41_hi": (
        "atom.models.deepseek_v41.mono.hi.target",
        "atom.models.deepseek_v41.mono.hi.device.gfx950.buffer_ops",
        "atom.models.deepseek_v41.mono.hi.device.gfx950.ops",
        "atom.models.deepseek_v41.mono.hi.plan.program",
        "atom.models.deepseek_v41.mono.hi.runtime.kernel",
    ),
}


def _count_emits(monkeypatch) -> list:
    """Every kernel FlyDSL traces from here on appends one entry."""
    emitted = []
    emit = KernelFunction._emit_kernel
    monkeypatch.setattr(
        KernelFunction,
        "_emit_kernel",
        lambda *a, **k: emitted.append(1) or emit(*a, **k),
    )
    return emitted


@pytest.mark.parametrize("package", sorted(_MONO_KERNEL_PACKAGES))
def test_a_mono_kernel_launch_does_not_trace_the_program_again(monkeypatch, package):
    """Later launches trace nothing and still pass their own ``layer``. The
    program's one task stores its launch's tag (``step * LAYER_SLOTS + layer + 1``)."""
    target, bo, ops, program, kernel = (
        importlib.import_module(name) for name in _MONO_KERNEL_PACKAGES[package]
    )
    if target.device_arch() != target.MI355X.arch:
        pytest.skip("an MI355X program")

    def store_tag(k, t):
        bo.buffer_store(k.tag, ops.rsrc(k.args["out"]), 0)

    stage = program.Stage("tag", 1, store_tag, args=("out",), base=0)
    mono = kernel.MonoKernel(
        program.Program("launch_tag", program.Scenario(1, 1), [stage])
    )
    bufs, out = mono.buffers(), torch.zeros(1, dtype=torch.int32, device="cuda")
    table = mono.table({"out": out})
    mono.launch(table, bufs)
    torch.cuda.synchronize()
    assert out.item() == 1
    emitted = _count_emits(monkeypatch)
    for layer in (1, 2):
        mono.launch(table, bufs, layer=layer)
        torch.cuda.synchronize()
        assert out.item() == layer + 1
    assert not emitted


def test_the_v41_router_does_not_trace_again(monkeypatch):
    x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(EXPERTS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    if not router_supported(x, weight):
        pytest.skip("a gfx950 kernel")
    first = router_logits(x, weight)
    emitted = _count_emits(monkeypatch)
    again = router_logits(x, weight)
    torch.cuda.synchronize()
    assert torch.equal(first, again)
    assert not emitted
