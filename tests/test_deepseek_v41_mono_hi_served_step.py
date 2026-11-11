# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DSpark proposer on the builder's routing decision (``AttentionMetaData.served_by_round``): a step the model's
decode round served wrote the draft's context KV inside the target forward; every other step (a 6-token prefill of
the served shape included) is ATOM's write. The anchors are staged by the proposer on every step, from the next ids
the runner gathered from the TP-broadcast ids. CPU, the proposer's own code.
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


def _proposer(module, writes):
    p = module.DSparkProposer.__new__(module.DSparkProposer)
    p.model = SimpleNamespace(
        write_context_kv=lambda hidden, positions: writes.append(hidden.shape),
        vocab_size=100,
    )
    p.aux_for = lambda hidden: [hidden, hidden]
    bufs = {
        "anchor_ids": torch.full((4,), -1, dtype=torch.int64),
        "anchor_positions": torch.full((4,), -1, dtype=torch.int64),
    }
    p.block = SimpleNamespace(
        buffer=lambda name, bs: bufs[name][:bs],
        stage=lambda bs, inputs: {k: v.clone() for k, v in inputs.items()},
    )
    return p


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
    p = _proposer(proposer_module, writes)
    p.compute_draft_kv(torch.arange(6), torch.zeros(6, 4), None)
    assert writes == ([torch.Size([6, 8])] if written else [])


@pytest.mark.parametrize(
    "service", [RoundService.NONE, RoundService.FORWARD, RoundService.DEVICE]
)
def test_the_anchors_are_the_next_ids_on_every_step(
    monkeypatch, proposer_module, service
):
    """Served or not, the proposer stages the anchors from the next ids it is handed (the runner's gather from the
    TP-broadcast ids): the forward stages none."""
    ids = torch.tensor([7, 9])
    positions = torch.arange(12)
    rows = torch.tensor([5, 11])
    _context(monkeypatch, proposer_module, service)
    proposer = _proposer(proposer_module, [])
    staged = proposer._stage_anchors(proposer.block, 2, ids, positions, rows)
    assert staged["anchor_ids"].tolist() == [7, 9]
    assert staged["anchor_positions"].tolist() == [5, 11]
