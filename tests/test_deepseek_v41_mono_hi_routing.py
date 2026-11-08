# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's hi routing (``hi_dispatch.HiDecodeModel``): the forward follows the builder's routing decision
(``AttentionMetaData.served_by_round``). A step routed to the hook runs on the hi path and never creates ATOM's K1 / K2
runner; another decode step creates it (once) and runs on it, and invalidates the hook's incremental import; a step
routed to the hook whose forward the hi layers cannot run (``inputs_embeds``) raises instead of leaving Engram staging
and the draft's context write on the hook's side while the forward takes ATOM's.
"""

from types import SimpleNamespace

import sys
import types

import pytest
import torch
from aiter_stub import _drop, atom_without_aiter


@pytest.fixture(scope="module")
def D():
    """``hi_dispatch`` with ATOM's own mono runners (FlyDSL kernels) stood in for (AITER and Triton stubbed where they
    are not installed); every module it loaded dropped after."""
    before = set(sys.modules)
    with pytest.MonkeyPatch.context() as mp:
        for mod, cls in (
            ("runner", "V41MonoDecodeRunner"),
            ("draft_runner", "DraftMonoRunner"),
        ):
            stub = types.ModuleType(f"atom.models.deepseek_v41.mono.{mod}")
            setattr(stub, cls, object)
            mp.setitem(sys.modules, stub.__name__, stub)
        with atom_without_aiter() as load:
            yield load("atom.models.deepseek_v41.mono.hi_dispatch")
        for name in set(sys.modules) - before:
            if name.startswith("atom.models.deepseek_v41.mono"):
                _drop(name)


class _Hook:
    def __init__(self):
        self.head_rows, self.invalidated = 0, []

    def invalidate(self, step, dummy=False):
        self.invalidated.append(step)


class _K12:
    def __init__(self, made):
        made.append(self)
        self.ran = []

    def prepare(self, rows):
        return True

    def forward(self, input_ids, positions):
        self.ran.append(input_ids.numel())
        return "k12"


def _model(D, monkeypatch, hook, served=()):
    """The wrapped model; ``served``: the widths the builder routes to the hook."""
    from atom.mono.runtime.route import LazyRunner
    from atom.utils.forward_context import RoundService

    m = D.HiDecodeModel.__new__(D.HiDecodeModel)
    torch.nn.Module.__init__(m)
    made = []
    m.model, m._dp_ranks, m._hi = (lambda ids, pos, inputs_embeds=None: "atom"), 0, hook
    m._mono = LazyRunner(lambda: _K12(made), "V4.1 mono decode")
    m._hi_forward = lambda ids, pos: "hi"
    md = SimpleNamespace(step=SimpleNamespace(), served_by_round=RoundService.NONE)
    fwd = SimpleNamespace(
        attn_metadata=md,
        context=SimpleNamespace(is_dummy_run=False),
        ubatch_slices=None,
    )
    monkeypatch.setattr(D, "get_forward_context", lambda: fwd)

    def routed(ids, emb=None):
        md.served_by_round = (
            RoundService.FORWARD if ids.numel() in served else RoundService.NONE
        )
        return m(ids, None, inputs_embeds=emb)

    monkeypatch.setattr(
        D, "step_supported", lambda ids, emb, dp: emb is None and ids.numel() <= 48
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return routed, made


def test_a_served_step_runs_on_hi_and_builds_no_k1_k2(D, monkeypatch):
    hook = _Hook()
    m, made = _model(D, monkeypatch, hook, served={6, 12})
    for rows in (6, 12, 6):
        assert m(torch.zeros(rows, dtype=torch.int64)) == "hi"
    assert made == [] and hook.invalidated == []


def test_another_decode_step_creates_k1_k2_once(D, monkeypatch):
    hook = _Hook()
    m, made = _model(D, monkeypatch, hook, served={6})
    assert m(torch.zeros(7, dtype=torch.int64)) == "k12"
    assert m(torch.zeros(6, dtype=torch.int64)) == "hi"
    assert m(torch.zeros(9, dtype=torch.int64)) == "k12"
    assert len(made) == 1 and made[0].ran == [7, 9]
    assert len(hook.invalidated) == 2


def test_a_served_step_the_hi_layers_cannot_run_raises(D, monkeypatch):
    """A step routed to the hook (Engram staging skipped its lookup, the draft's context write left to the hook) whose
    forward carries ``inputs_embeds`` raises: the forward never falls back to ATOM's path on such a step.
    """
    m, made = _model(D, monkeypatch, _Hook(), served={6})
    with pytest.raises(RuntimeError, match="routed to the hi path"):
        m(torch.zeros(6, dtype=torch.int64), emb=torch.zeros(6, 4))
    # the same width not routed to the hook: ATOM's path, embeddings and all
    m2, _ = _model(D, monkeypatch, _Hook(), served=())
    assert m2(torch.zeros(6, dtype=torch.int64), emb=torch.zeros(6, 4)) == "atom"
    assert made == []
