# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DSpark proposer on the builder's routing decision (``AttentionMetaData.served_by_round``): a step the model's
decode round served wrote the draft's context KV and staged the anchors inside the target forward; every other step
(a 6-token prefill of the served shape included) is ATOM's write and staging. CPU, the proposer's own code.
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.utils.forward_context import RoundService


@pytest.fixture(scope="module")
def proposer_module():
    with atom_without_aiter() as load:
        yield load("atom.spec_decode.dspark_proposer")


def _context(monkeypatch, module, service):
    fwd = SimpleNamespace(
        attn_metadata=SimpleNamespace(served_by_round=service),
        context=SimpleNamespace(scheduled_bs=1),
    )
    monkeypatch.setattr(module, "get_forward_context", lambda: fwd)


def _proposer(module, writes, staged=True):
    p = module.DSparkProposer.__new__(module.DSparkProposer)
    p.model = SimpleNamespace(
        write_context_kv=lambda hidden, positions: writes.append(hidden.shape),
        vocab_size=100,
    )
    p.aux_for = lambda hidden: [hidden, hidden]
    asked = []
    p.runner = SimpleNamespace(
        decode_round=SimpleNamespace(
            anchors_staged=lambda running_bs: asked.append(running_bs) or staged
        )
    )
    bufs = {
        "anchor_ids": torch.full((4,), -1, dtype=torch.int64),
        "anchor_positions": torch.full((4,), -1, dtype=torch.int64),
    }
    p.block = SimpleNamespace(
        buffer=lambda name, bs: bufs[name][:bs],
        stage=lambda bs, inputs: {k: v.clone() for k, v in inputs.items()},
    )
    return p, asked


@pytest.mark.parametrize(
    "service, written",
    [
        (
            RoundService.NONE,
            True,
        ),  # ATOM's forward, a 6-token prefill of the served shape among them
        (RoundService.FORWARD, False),
        (RoundService.LEAN, False),
    ],
)
def test_the_context_write_follows_the_routing_decision(
    monkeypatch, proposer_module, service, written
):
    writes = []
    _context(monkeypatch, proposer_module, service)
    p, _ = _proposer(proposer_module, writes)
    p.compute_draft_kv(torch.arange(6), torch.zeros(6, 4), None)
    assert writes == ([torch.Size([6, 8])] if written else [])


def test_the_anchors_follow_the_routing_decision(monkeypatch, proposer_module):
    """A served step asks the decode round whether its forward staged the anchors (none written here when it did);
    any other step stages them as before, without asking."""
    ids = torch.tensor([7, 9])
    positions = torch.arange(12)
    rows = torch.tensor([5, 11])
    _context(monkeypatch, proposer_module, RoundService.LEAN)
    p, asked = _proposer(proposer_module, [])
    staged = p._stage_anchors(2, ids, positions, rows)
    assert asked == [2] and staged["anchor_ids"].tolist() == [-1, -1]
    _context(monkeypatch, proposer_module, RoundService.NONE)
    p, asked = _proposer(proposer_module, [])
    staged = p._stage_anchors(2, ids, positions, rows)
    assert asked == [] and staged["anchor_ids"].tolist() == [7, 9]
    assert staged["anchor_positions"].tolist() == [5, 11]
