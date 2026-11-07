# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's hi routing (``hi_dispatch.HiDecodeModel``): a decode step the hi hook serves runs on the hi path
and never creates ATOM's K1 / K2 runner; another decode step creates it (once) and runs on it; a lean-host step the
hook does not serve is refused; every step the hook does not serve invalidates its incremental import.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")


@pytest.fixture(scope="module")
def D():
    from atom.models.deepseek_v41.mono import hi_dispatch

    return hi_dispatch


class _Hook:
    def __init__(self, served, lean_host=False):
        self.served, self.lean_host = served, lean_host
        self.head_rows, self.invalidated = 0, []

    def serves_step(self, rows, step):
        return rows in self.served

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


def _model(D, monkeypatch, hook, lean_step=False):
    from atom.mono.runtime.route import LazyRunner

    m = D.HiDecodeModel.__new__(D.HiDecodeModel)
    torch.nn.Module.__init__(m)
    made = []
    m.model, m._dp_ranks, m._hi = (lambda ids, pos: "atom"), 0, hook
    m._mono = LazyRunner(lambda: _K12(made), "V4.1 mono decode")
    m._hi_forward = lambda ids, pos: "hi"
    step = SimpleNamespace(lean_host=lean_step)
    fwd = SimpleNamespace(
        attn_metadata=SimpleNamespace(step=step),
        context=SimpleNamespace(is_dummy_run=False),
    )
    monkeypatch.setattr(D, "get_forward_context", lambda: fwd)
    monkeypatch.setattr(D, "step_supported", lambda ids, emb, dp: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return m, made


def test_a_served_step_runs_on_hi_and_builds_no_k1_k2(D, monkeypatch):
    hook = _Hook({6, 12})
    m, made = _model(D, monkeypatch, hook)
    for rows in (6, 12, 6):
        assert m(torch.zeros(rows, dtype=torch.int64), None) == "hi"
    assert made == [] and hook.invalidated == []


def test_another_decode_step_creates_k1_k2_once(D, monkeypatch):
    hook = _Hook({6})
    m, made = _model(D, monkeypatch, hook)
    assert m(torch.zeros(7, dtype=torch.int64), None) == "k12"
    assert m(torch.zeros(6, dtype=torch.int64), None) == "hi"
    assert m(torch.zeros(9, dtype=torch.int64), None) == "k12"
    assert len(made) == 1 and made[0].ran == [7, 9]
    assert len(hook.invalidated) == 2


def test_an_unserved_lean_host_step_is_refused(D, monkeypatch):
    m, made = _model(D, monkeypatch, _Hook({6}, lean_host=True), lean_step=True)
    with pytest.raises(RuntimeError, match="lean-host step is not served"):
        m(torch.zeros(7, dtype=torch.int64), None)
    assert made == []
