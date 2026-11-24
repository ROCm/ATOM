# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A P/D producer's DSpark drafter only writes context K/V (#3038's gate).

On a TP1 PP stage the MiniMax-M3.1 draft holds all four KV heads, which its
block pass cannot decode, so a producer must never propose: it commits the
target step and returns no drafts. Its drafts were never forwarded anyway (a
producer's scheduler does not speculate). Under PP every stage warms up on
the rows the last stage's block drafter needs.
"""

from __future__ import annotations

import types

import pytest
import torch
from aiter_stub import atom_under_stubs


@pytest.fixture(scope="module")
def mods():
    with atom_under_stubs() as load:
        yield types.SimpleNamespace(
            runner=load("atom.model_engine.model_runner"),
            drafter=load("atom.spec_decode.drafter"),
        )


@pytest.mark.parametrize(
    "producer, duplicates, expected",
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_only_a_producer_whose_draft_kv_is_apart_from_propose_is_context_only(
    mods, producer, duplicates, expected
):
    # A DSpark draft writes its context K/V apart from propose; EAGLE's first
    # draft step is that write, so an EAGLE producer still proposes.
    cls = type(
        "D",
        (mods.drafter.Drafter,),
        {"_resolve_mtp_k": lambda self: 0, "propose": lambda self, **_: None},
    )
    drafter = object.__new__(cls)
    drafter.runner = types.SimpleNamespace(
        tokenID_processor=types.SimpleNamespace(is_pd_producer=producer)
    )
    cls.draft_kv_duplicates_propose = property(lambda self: duplicates)
    assert drafter.context_kv_only is expected


def runner_double(mods, *, context_only: bool):
    runner = object.__new__(mods.runner.ModelRunner)
    calls = []
    runner.drafter = types.SimpleNamespace(
        context_kv_only=context_only,
        prepare_inputs=lambda n: torch.arange(n),
    )
    runner.attn_metadata_builder = types.SimpleNamespace(
        commit_speculative_state=lambda md, rows: calls.append(("commit", rows))
    )
    runner.propose_draft_token_ids = lambda *a, **k: calls.append(("propose",)) or [7]
    runner.tokenID_processor = types.SimpleNamespace(
        input_ids=types.SimpleNamespace(gpu=torch.zeros(9))
    )
    return runner, calls


def test_a_context_only_producer_commits_the_step_and_never_proposes(mods):
    runner, calls = runner_double(mods, context_only=True)
    batch = types.SimpleNamespace(total_seqs_num=3)
    anchors = torch.tensor([2, 5, 8])
    ctx = types.SimpleNamespace(attn_metadata="md")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mods.runner, "get_forward_context", lambda: ctx)
        out = runner._draft_or_commit(batch, torch.zeros(9, 4), None, None, anchors)
    assert out is None
    assert [c[0] for c in calls] == ["commit"]
    assert torch.equal(calls[0][1], anchors)


def test_any_other_runner_proposes(mods):
    runner, calls = runner_double(mods, context_only=False)
    batch = types.SimpleNamespace(total_seqs_num=3)
    out = runner._draft_or_commit(batch, torch.zeros(9, 4), None, None, None)
    assert out == [7]
    assert calls == [("propose",)]
