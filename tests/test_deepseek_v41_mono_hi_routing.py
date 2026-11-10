# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's hi routing (``hi_dispatch.HiDecodeModel``): the forward follows the builder's routing decision
(``AttentionMetaData.served_by_round``). A step routed to the hook runs on the hi path and never creates ATOM's K1 / K2
runner; another decode step creates it (once) and runs on it, and invalidates the hook's incremental import; a step
routed to the hook whose forward the hi layers cannot run (``inputs_embeds``) raises instead of leaving Engram staging
and the draft's context write on the hook's side while the forward takes ATOM's. A served forward expands ATOM's own
embedding rows into the hi layers' input; its head output is the hook's in-graph verify (``compute_head``).
"""

import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from aiter_stub import _drop, atom_without_aiter


@pytest.fixture(scope="module")
def D():
    """``hi_dispatch`` with ATOM's own mono runners (FlyDSL kernels) stood in for where they are not imported yet (AITER
    and Triton stubbed where they are not installed); every module it loaded dropped after. (A runner module already
    imported stays: an imported ``mono.dispatch`` holds its classes.)"""
    before = set(sys.modules)
    with pytest.MonkeyPatch.context() as mp:
        for mod, cls in (
            ("runner", "V41MonoDecodeRunner"),
            ("draft_runner", "DraftMonoRunner"),
        ):
            if f"atom.models.deepseek_v41.mono.{mod}" in sys.modules:
                continue
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
        self.verified, self.invalidated = 0, []

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


def test_the_head_output_is_the_hooks_in_graph_verify(D, monkeypatch):
    """``compute_head``: the hook's head output at the forward's rows (``HiLayers.head_output``); every forward
    starts with no verify of its own (``verified``), so a forward that served none gives None (the logits).
    """
    hook = _Hook()
    hook.head_output = lambda rows: (
        ("ids", "verdict", rows) if hook.verified == rows else None
    )
    m, _ = _model(D, monkeypatch, hook, served={6})
    hook.verified = 6
    m(torch.zeros(7, dtype=torch.int64))  # (ATOM's path: no verify of its own)
    assert hook.verified == 0
    model = D.HiDecodeModel.__new__(D.HiDecodeModel)
    torch.nn.Module.__init__(model)
    model._hi = hook
    assert model.compute_head(torch.zeros(6, 4)) is None
    hook.verified = 6
    assert model.compute_head(torch.zeros(6, 4)) == ("ids", "verdict", 6)


def test_a_served_forward_takes_atoms_embedding_rows(D, monkeypatch):
    """``_hi_forward``: ATOM's embedding lookup of the step's ids (its masked lookup and TP all-reduce) is the forward's
    boundary input and the begin-step's rows, which it expands into layer 0's streams; then every layer, the
    end-step, and ATOM's collapse back as the forward's output rows."""
    calls = []
    rows, hidden = 6, 4
    emb = torch.arange(rows * hidden, dtype=torch.float32).view(rows, hidden)

    class Model:
        topology: ClassVar[list] = [SimpleNamespace(layer_id=i) for i in range(3)]

        def embed(self, ids):
            calls.append(("embed", ids.numel()))
            return emb

        def begin_forward(self, h, engram):
            calls.append(
                ("begin_forward", tuple(h.shape), h.data_ptr() == emb.data_ptr())
            )

        def end_forward(self, h, engram):
            calls.append(("end_forward", tuple(h.shape)))

    out = torch.zeros(rows, hidden)

    class Hook:
        served_widths: ClassVar[set] = set()

        def begin_step(self, cache, step, n, h, staged=None, input_ids=None):
            calls.append(("begin_step", n, torch.equal(h, emb), input_ids.numel()))

        def layer(self, i, n):
            calls.append(("layer", i))

        def end_step(self, cache, step, n):
            calls.append(("end_step", n))

        def collapse_view(self, n):
            return out[:n]

    md = SimpleNamespace(
        step="step", cache=SimpleNamespace(packed=False), engram_embeddings=None
    )
    monkeypatch.setattr(
        D, "get_forward_context", lambda: SimpleNamespace(attn_metadata=md)
    )
    m = D.HiDecodeModel.__new__(D.HiDecodeModel)
    torch.nn.Module.__init__(m)
    m.model, m._hi = Model(), Hook()
    got = m._hi_forward(torch.zeros(rows, dtype=torch.int64), None)
    assert got.data_ptr() == out.data_ptr() and got.shape == (rows, hidden)
    assert calls == [
        ("embed", rows),
        ("begin_forward", (1, rows, hidden), True),
        ("begin_step", rows, True, rows),
        *[("layer", i) for i in range(3)],
        ("end_step", rows),
        ("end_forward", (1, rows, hidden)),
    ]
    assert m._hi.served_widths == {rows}
