# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dsa_indexer.dispatch``: which steps of a model whose DSA indexers are its
only mono part run them on mono (a decode, plain or a uniform verify), and the
wrapper serving exactly those inside the model's own forward."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from atom.mono.dsa_indexer import dispatch


def _context(prefill=False, qlen=1, requests=1):
    return SimpleNamespace(
        context=SimpleNamespace(is_prefill=prefill),
        attn_metadata=SimpleNamespace(
            max_seqlen_q=qlen, context_lens=torch.zeros(requests)
        ),
    )


def test_a_decode_step_is_served_at_its_uniform_query_length():
    assert dispatch.decode_qlen(_context(qlen=1, requests=8), 8) == 1
    assert dispatch.decode_qlen(_context(qlen=4, requests=2), 8) == 4


def test_any_other_step_is_not():
    assert dispatch.decode_qlen(_context(prefill=True), 1) is None
    # rows that are not every request at the step's query length
    assert dispatch.decode_qlen(_context(qlen=4, requests=2), 6) is None
    no_step = SimpleNamespace(context=None, attn_metadata=None)
    assert dispatch.decode_qlen(no_step, 1) is None


class _Runner:
    def __init__(self):
        self.served = []

    @contextmanager
    def serving(self, rows, qlen):
        self.served.append((rows, qlen))
        yield


def _wrapped(monkeypatch, ready, qlen):
    """The wrapper around a model that records which steps were being served
    when it ran, its runner deciding ``ready``."""
    calls = []
    runner = _Runner()
    wrapper = dispatch.IndexerMonoModel(
        lambda *args: calls.append(list(runner.served)) or "out",
        hook=None,
        metadata_builder=None,
    )
    wrapper._mono = SimpleNamespace(ready=lambda rows, q: ready, runner=runner)
    monkeypatch.setattr(dispatch, "decode_qlen", lambda context, rows: qlen)
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: None)
    return wrapper, runner, calls


def test_the_model_forward_runs_inside_the_served_step(monkeypatch):
    wrapper, runner, calls = _wrapped(monkeypatch, ready=True, qlen=4)
    assert wrapper(torch.zeros(8), torch.zeros(8)) == "out"
    assert runner.served == [(8, 4)] and calls == [[(8, 4)]]


def test_a_declined_step_runs_the_model_alone(monkeypatch):
    for ready, qlen in ((False, 4), (True, None)):
        wrapper, runner, calls = _wrapped(monkeypatch, ready, qlen)
        assert wrapper(torch.zeros(8), torch.zeros(8)) == "out"
        assert runner.served == [] and calls == [[]]
