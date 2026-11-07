# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi hook allocates every program's scratch uncached (hi ``Kernel.buffers(uncached=)``, hi
F19: the served c1 median +1.44 %, bitwise)."""

import inspect
import re

import pytest

pytest.importorskip("triton")
hook = pytest.importorskip("atom.models.deepseek_v41.mono.hi_runner")


def test_programs_share_one_arena(monkeypatch):
    """``hi_draft._Arena``: one uncached, 2 MiB-aligned scratch arena (the largest program's), its dual-copy word
    calibrated once, for all of a step counter's programs; ``_buffers``: each program the arena's scratch and word, its
    own private buffer, the given step counter."""
    import torch

    from atom.models.deepseek_v41.mono import hi_draft

    pytest.importorskip("flydsl")
    pytest.importorskip("aiter")
    from atom.models.deepseek_v41.mono.hi.plan.program import DUAL_PAGE
    from atom.models.deepseek_v41.mono.hi.runtime import dual, uncached

    zeros = []
    monkeypatch.setattr(
        uncached,
        "zeros",
        lambda n, dev: zeros.append(n) or torch.zeros(n, dtype=torch.uint8),
    )
    words = []
    monkeypatch.setattr(
        dual, "scratch_word", lambda s, n, name="": words.append((n, name)) or 5
    )
    seen = []

    class Kernel:
        def __init__(self, name, nbytes, dual_bytes):
            self.program = type("P", (), dict(name=name, scratch_bytes=nbytes, dual_bytes=dual_bytes,
                                              dual_on=dual_bytes > 0))()  # fmt: skip

        def buffers(self, device="cuda", scratch=None, uncached=False, word=None):
            seen.append((self.program.name, scratch.data_ptr(), uncached, word))
            return type("B", (), dict(scratch=scratch, private=torch.zeros(4)))()

    kerns = {
        6: Kernel("a6", 1000, 512),
        12: Kernel("a12", 5000, 1024),
        18: Kernel("b18", 300, 0),
    }
    arena = hi_draft._Arena("target", kerns.values(), "cpu")
    assert (
        zeros == [5000 + DUAL_PAGE]
        and words == [(1024, "target")]
        and arena.scratch.data_ptr() % DUAL_PAGE == 0
    )
    step = torch.zeros(1, dtype=torch.int32)
    out = hi_draft._buffers(kerns, arena, "cpu", step)
    assert {p for _, p, _, _ in seen} == {arena.scratch.data_ptr()} and all(
        u and w == 5 for _, _, u, w in seen
    )
    assert (
        all(b.step is step for b in out.values())
        and out[6].private is not out[12].private
    )


def test_every_program_buffer_goes_through_the_helper():
    """Every scratch allocation in the hook (``hi_runner``, ``hi_draft``) goes through ``_buffers`` (a new program's
    buffers would otherwise fall back to cached scratch): no ``.buffers(device=`` call outside it; the layer programs,
    the heads, the Engram kv launch, the draft's CTX write and its stages each allocate through it.
    """
    from atom.models.deepseek_v41.mono import hi_draft

    src = inspect.getsource(hook) + inspect.getsource(hi_draft)
    helper = inspect.getsource(hi_draft._buffers)
    assert not re.findall(r"\.buffers\(device=[^)]*\)", src.replace(helper, ""))
    assert (
        len(re.findall(r"= _buffers\(\s*self\.", src)) == 5
        and len(re.findall(r"_Arena\(", src)) == 2
    )


def test_hi_kernel_buffers_take_uncached():
    """With FlyDSL installed: the hi runtime's ``MonoKernel.buffers`` takes ``uncached=``."""
    pytest.importorskip("flydsl")
    pytest.importorskip(
        "aiter"
    )  # (the hi device code re-exports atom.mono.device's primitives)
    from atom.models.deepseek_v41.mono.hi.runtime import kernel

    assert "uncached" in inspect.signature(kernel.MonoKernel.buffers).parameters
