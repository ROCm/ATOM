# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`prepare()` answers for a ROW COUNT, and two of its refusals are row-keyed.

A build is cached by kernel chunk, which is right -- the kernel binary is
compiled per chunk. But `rows` carries two verdicts the chunk cannot express:
a width already in `_refused`, and a chunked width that `_build` turns down
because its launches would share one mailbox epoch. Asking the chunk cache
first let both through.
"""

import pytest

pytest.importorskip("flydsl", reason="the GLM mono runner is FlyDSL source")
# `runner.py` also binds the TP group through `aiter.dist.parallel_state` at
# module scope, so FlyDSL alone is not enough to collect this file -- the same
# guard `test_mono_runner.py` and `test_fused_indexer_dispatch.py` carry.
pytest.importorskip("aiter", reason="the GLM mono runner imports aiter.dist")

from atom.models.glm52.mono.runner import Glm52MonoRunner


def _runner(query_length, ops, refused):
    runner = object.__new__(Glm52MonoRunner)
    runner._query_length = query_length
    runner._ops = {(0, chunk): object() for chunk in ops}
    runner._refused = set(refused)
    runner._built = []

    def _build(rows):
        runner._built.append(rows)
        return False

    runner._build = _build
    return runner


def test_a_refused_width_stays_refused_even_when_its_chunk_is_built():
    """The chunk cache cannot answer for a width `_build` already turned down."""
    runner = _runner(1, ops=[2], refused=[2])
    assert runner.prepare(2) is False
    assert runner._built == []  # answered from `_refused`, not rebuilt


def test_an_unrefused_width_whose_chunk_exists_is_served():
    runner = _runner(1, ops=[2], refused=[])
    assert runner.prepare(2) is True
    assert runner._built == []


def test_a_chunked_width_reaches_build_even_when_its_chunk_exists(monkeypatch):
    """`rows != chunk_samples` is the mailbox-epoch refusal. It lives in
    `_build`, so a chunked `rows` must get there rather than be answered True
    by a same-chunk op built for a different row count."""
    from atom.models.glm52.mono import runner as mod

    monkeypatch.setattr(mod, "glm5_kernel_samples", lambda rows, q: 5)
    runner = _runner(5, ops=[5], refused=[])
    assert runner.prepare(15) is False  # `_build` refuses it
    assert runner._built == [15]
