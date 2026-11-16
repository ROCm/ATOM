# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The presence penalty without a GPU: request admission, the host bookkeeping
and staging of `presence_penalty.py`, and the rule at the token-constraint
boundary (the kernels' torch twins stand in for Triton)."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from presence_penalty_reference import (
    bit_rows,
    expected_logits,
    mark_emitted_twin,
    penalize_rows_twin,
)

from atom.model_engine import token_constraints as boundary
from atom.model_ops.presence_penalty import PresencePenalties
from atom.sampling_params import SamplingParams

VOCAB = 70  # three table words: ids 31, 32 and 63 sit on word edges


def _buffer(n=256):
    host = np.zeros(n, dtype=np.int32)
    return SimpleNamespace(np=host, gpu=torch.from_numpy(host))


def _batch(req_ids, penalties, final=None):
    return SimpleNamespace(
        total_seqs_num=len(req_ids),
        req_ids=list(req_ids),
        presence_penalties=np.asarray(penalties, dtype=np.float32),
        is_final_chunk=final,
        structured_requests=None,
        required_channel=None,
    )


# -- admission ---------------------------------------------------------------


@pytest.mark.parametrize("penalty", [-2.0, 0.0, 1.5, 2.0])
def test_openai_range_is_accepted(penalty):
    assert SamplingParams(presence_penalty=penalty).presence_penalty == penalty


@pytest.mark.parametrize("penalty", [-2.5, 2.01, float("nan")])
def test_out_of_range_penalty_is_refused(penalty):
    with pytest.raises(ValueError, match="presence_penalty"):
        SamplingParams(presence_penalty=penalty)


@pytest.mark.parametrize(
    "config, kv_transfer, match",
    [
        (SimpleNamespace(enable_rapidserve=True), None, "RapidServe"),
        (SimpleNamespace(), {"do_remote_prefill": True}, "P/D"),
        (
            SimpleNamespace(
                kv_transfer_config={"kv_connector": "moriio", "kv_role": "kv_consumer"}
            ),
            None,
            "P/D",
        ),
        (SimpleNamespace(pipeline_parallel_size=2), None, "pipeline"),
    ],
)
def test_engines_that_emit_tokens_elsewhere_refuse_a_penalty(
    config, kv_transfer, match
):
    params = SamplingParams(presence_penalty=1.0)
    with pytest.raises(ValueError, match=match):
        params.validate_presence_penalty(config, kv_transfer=kv_transfer)
    SamplingParams().validate_presence_penalty(config, kv_transfer=kv_transfer)


@pytest.mark.parametrize("role", ["offload", "kv_producer", "kv_consumer"])
def test_a_kv_offload_tier_keeps_the_penalty(role):
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": role}
    )
    SamplingParams(presence_penalty=1.5).validate_presence_penalty(config)


# -- staging -----------------------------------------------------------------


def test_one_row_per_request_outside_verification():
    table = PresencePenalties(VOCAB, 4, "cpu")
    rows, count = table.stage(_batch([7, 8, 9], [1.5, 0.0, -0.5]), None, _buffer())
    assert rows.slots.tolist() == [0, -1, 1]
    assert rows.penalties.tolist() == [1.5, 0.0, -0.5]
    assert rows.drafts_before.tolist() == [0, 0, 0]
    assert rows.emit_slots.tolist() == [0, -1, 1]
    assert rows.fresh.tolist() == [0, 1]
    assert count == 3 * 3 + 3 + 2


def test_verification_rows_count_the_drafts_before_them():
    table = PresencePenalties(VOCAB, 4, "cpu")
    batch = _batch([7, 8], [1.0, 2.0])
    rows, _ = table.stage(batch, np.array([4, 2]), _buffer())
    assert rows.slots.tolist() == [0] * 4 + [1] * 2
    assert rows.penalties.tolist() == [1.0] * 4 + [2.0] * 2
    assert rows.drafts_before.tolist() == [0, 1, 2, 3, 0, 1]
    assert rows.emit_slots.tolist() == [0, 1]


def test_a_middle_prefill_chunk_emits_nothing():
    table = PresencePenalties(VOCAB, 4, "cpu")
    rows, _ = table.stage(
        _batch([7, 8], [1.0, 1.0], final=[False, True]), None, _buffer()
    )
    assert rows.slots.tolist() == [0, 1]
    assert rows.emit_slots.tolist() == [-1, 1]


def test_rows_are_claimed_once_and_reused_after_release():
    table = PresencePenalties(VOCAB, 2, "cpu")
    table.stage(_batch([7, 8], [1.0, 1.0]), None, _buffer())
    again, _ = table.stage(_batch([8, 7], [1.0, 1.0]), None, _buffer())
    assert again.slots.tolist() == [1, 0]
    assert again.fresh.tolist() == []
    table.release([7, 7, 99])  # repeated and unknown releases are harmless
    newcomer, _ = table.stage(_batch([10], [1.0]), None, _buffer())
    assert newcomer.slots.tolist() == [0]
    assert newcomer.fresh.tolist() == [0]


def test_live_requests_keep_their_rows_and_the_table_grows():
    table = PresencePenalties(VOCAB, 2, "cpu")
    rows, _ = table.stage(_batch([7, 8], [1.0, 1.0]), None, _buffer())
    mark_emitted_twin(table.seen, rows, torch.tensor([[31], [63]]), VOCAB)
    # 7 and 8 are preempted, not released: a third request must not take
    # their rows.
    later, _ = table.stage(_batch([9], [1.0]), None, _buffer())
    assert later.slots.tolist() == [2]
    assert later.num_slots == 4
    seen = table.table(later.num_slots)
    assert seen.shape[0] == 4
    bits = bit_rows(seen)
    assert bits[0].nonzero()[0].tolist() == [31]
    assert bits[1].nonzero()[0].tolist() == [63]


# -- token-constraint boundary -----------------------------------------------


@pytest.fixture
def twins(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.triton_presence_penalty",
        SimpleNamespace(
            penalize_rows=penalize_rows_twin, mark_emitted=mark_emitted_twin
        ),
    )


def test_a_penalty_is_a_constraint_and_keeps_logprobs_raw():
    batch = _batch([7], [1.0])
    assert boundary.has_constraints(batch)
    assert boundary.needs_constraint_snapshot(batch, None)
    batch.presence_penalties = None
    assert not boundary.has_constraints(batch)
    assert not boundary.needs_constraint_snapshot(batch, None)


@pytest.mark.parametrize("penalty, released", [(1.5, True), (0.0, False)])
def test_a_finished_request_with_a_penalty_is_released(seq_factory, penalty, released):
    from aiter_stub import atom_without_aiter

    seq = seq_factory(
        [1, 2, 3, 4], sampling_params=SamplingParams(presence_penalty=penalty)
    )
    calls = []
    core = SimpleNamespace(
        runner_mgr=SimpleNamespace(call_func=lambda *args: calls.append(args))
    )
    with atom_without_aiter() as load:
        load("atom.model_engine.engine_core").EngineCore._release_request_resources(
            core, [seq]
        )
    assert (("release_constraints", [seq.id]) in calls) is released


def test_release_frees_the_rows(twins):
    penalties = PresencePenalties(VOCAB, 2, "cpu")
    penalties.stage(_batch([7], [1.0]), None, _buffer())
    runner = SimpleNamespace(structured_output=None, presence_penalties=penalties)
    boundary.release_token_constraints(runner, [7])
    rows, _ = penalties.stage(_batch([8], [1.0]), None, _buffer())
    assert rows.slots.tolist() == [0]
    assert rows.fresh.tolist() == [0]


def test_decode_then_verify_penalize_what_was_emitted(twins):
    """Request 0 (penalty 1.5) and request 1 (none) through a decode step,
    then a verify step whose second draft is rejected."""
    penalties = PresencePenalties(VOCAB, 4, "cpu")
    runner = SimpleNamespace(structured_output=None, presence_penalties=penalties)
    pens = [1.5, 0.0]
    generated = [[], []]
    torch.manual_seed(0)

    # Decode: one row each, nothing generated yet; the step emits 31 and 5.
    batch = _batch([7, 8], pens)
    batch.presence_rows, _ = penalties.stage(batch, None, _buffer())
    logits = torch.randn(2, VOCAB)
    raw = logits.clone()
    step = boundary.TokenConstraints.apply(runner, batch, logits, None)
    torch.testing.assert_close(logits, raw)  # nothing generated yet
    step.commit(torch.tensor([31, 5]), batch, deferred=False)
    generated = [[31], []]  # request 1 has no penalty, so nothing is kept

    # Verify: request 0 drafts [63, 31, 2] (row 2 already counts 31 from the
    # draft before it); request 1 drafts [6].
    batch = _batch([7, 8], pens)
    batch.presence_rows, _ = penalties.stage(batch, np.array([4, 2]), _buffer())
    input_ids = torch.tensor([31, 63, 31, 2, 5, 6])
    metadata = SimpleNamespace(
        input_ids=input_ids,
        num_spec_steps=3,
        bonus_logits_indices=torch.tensor([3, 5]),
        target_logits_indices=torch.tensor([0, 1, 2, 4]),
    )
    logits = torch.randn(6, VOCAB)
    expected = expected_logits(
        logits,
        generated,
        row_request=[0, 0, 0, 0, 1, 1],
        row_drafts=[[], [63], [63, 31], [63, 31, 2], [], [6]],
        penalties=pens,
    )
    step = boundary.TokenConstraints.apply(runner, batch, logits, metadata)
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(step.target, expected[[0, 1, 2, 4]])
    # Request 0 accepts 63 then rejects 31 (bonus 40); request 1 takes 6, 9.
    step.commit(torch.tensor([[63, 40, -1, -1], [6, 9, -1, -1]]), batch, deferred=True)
    bits = bit_rows(penalties.seen)
    slot0, slot1 = batch.presence_rows.slots[0], batch.presence_rows.slots[4]
    assert int(slot1) == -1
    assert bits[int(slot0)].nonzero()[0].tolist() == [31, 40, 63]
