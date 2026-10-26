# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: per-width build preparation."""

import pytest
import torch

# `mono.runner` binds the TP group through `aiter.dist.parallel_state` at
# module load.  Skip on the non-GPU unit gate; runs wherever AITER is present.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.models.glm52.mono.runner import Glm52MonoRunner


def _bare_runner():
    """A runner with only the fields ``prepare`` reads: no model, no GPU."""
    runner = object.__new__(Glm52MonoRunner)
    runner._ops = {}
    runner._refused = set()
    runner._query_length = 5
    return runner


def test_prepare_declines_during_graph_capture(monkeypatch):
    # Review Focus 4: allocation and the peer handshake cannot be captured.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _bare_runner().prepare(5) is False


def test_a_refused_width_stays_refused(monkeypatch):
    # Review Focus 5: a refused width must not rebuild on every step.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runner = _bare_runner()
    runner._refused.add(5)
    calls = []
    monkeypatch.setattr(
        Glm52MonoRunner, "_build", lambda self, rows: calls.append(rows)
    )
    assert runner.prepare(5) is False
    assert calls == []
