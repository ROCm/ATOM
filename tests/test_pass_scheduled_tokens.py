# SPDX-License-Identifier: MIT
"""A pass's real row count reaches only consumers of that pass's rows: the
MoE sort reads it as its row count, so a subset of the rows must not see it,
and a pass over other rows (a decoder over the step's tails) declares its own."""

import torch

from atom.utils import forward_context as fc


def _forward(monkeypatch, running_tokens, count):
    forward = fc.ForwardContext(
        context=fc.Context(
            positions=torch.zeros(running_tokens, dtype=torch.int64),
            running_tokens=running_tokens,
            scheduled_tokens_gpu=count,
        )
    )
    monkeypatch.setattr(fc, "get_forward_context", lambda: forward)
    return forward


def test_only_the_pass_rows_see_the_count(monkeypatch):
    step = torch.tensor([802], dtype=torch.int32)
    _forward(monkeypatch, 1024, step)
    assert fc.pass_scheduled_tokens_gpu(1024) is step
    # a subset (a PCP shard, a tail no pass declared) runs every row
    assert fc.pass_scheduled_tokens_gpu(128) is None


def test_a_decoder_pass_declares_its_rows_and_count(monkeypatch):
    step = torch.tensor([802], dtype=torch.int32)
    tail = torch.tensor([100], dtype=torch.int32)
    forward = _forward(monkeypatch, 1024, step)
    with forward.use_pass_rows(128, tail):
        assert fc.pass_scheduled_tokens_gpu(128) is tail
        assert fc.pass_scheduled_tokens_gpu(1024) is None
    assert fc.pass_scheduled_tokens_gpu(1024) is step


def test_no_context_no_count(monkeypatch):
    monkeypatch.setattr(fc, "get_forward_context", lambda: fc.ForwardContext())
    assert fc.pass_scheduled_tokens_gpu(4) is None
