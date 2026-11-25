# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

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


def _batch(req_ids, penalties, final=None, resumed=None):
    return SimpleNamespace(
        total_seqs_num=len(req_ids),
        req_ids=list(req_ids),
        presence_penalties=np.asarray(penalties, dtype=np.float32),
        presence_resumed=resumed or {},
        is_final_chunk=final,
        structured_requests=None,
        required_channel=None,
    )


def _config(**fields):
    return SimpleNamespace(
        **{"pipeline_parallel_size": 1, "enable_rapidserve": False, **fields}
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
        (_config(enable_rapidserve=True), None, "RapidServe"),
        (_config(), {"do_remote_prefill": True}, "P/D"),
        (
            _config(
                kv_transfer_config={"kv_connector": "moriio", "kv_role": "kv_consumer"}
            ),
            None,
            "P/D",
        ),
        (_config(pipeline_parallel_size=2), None, "pipeline"),
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
    config = _config(
        kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": role}
    )
    SamplingParams(presence_penalty=1.5).validate_presence_penalty(config)


# -- staging -----------------------------------------------------------------


def test_one_row_per_request_outside_verification():
    table = PresencePenalties(VOCAB, 2, "cpu")
    buffer = _buffer()
    rows, count = table.stage(_batch([7, 8, 9], [1.5, 0.0, -0.5]), None, buffer)
    assert rows.slots.tolist() == [0, -1, 1]
    assert rows.penalties.tolist() == [1.5, 0.0, -0.5]
    assert rows.drafts_before.tolist() == [0, 0, 0]
    assert rows.emit_slots.tolist() == [0, -1, 1]
    assert rows.fresh.tolist() == [0, 1]
    # Nothing resumed: no copy of its own, an empty view of the buffer.
    assert rows.resumed_ids.numel() == 0
    storage = buffer.gpu.untyped_storage().data_ptr()
    assert rows.resumed_ids.untyped_storage().data_ptr() == storage
    assert count == 3 * 3 + 3 + 2


def test_verification_rows_count_the_drafts_before_them():
    table = PresencePenalties(VOCAB, 2, "cpu")
    batch = _batch([7, 8], [1.0, 2.0])
    rows, _ = table.stage(batch, np.array([4, 2]), _buffer())
    assert rows.slots.tolist() == [0] * 4 + [1] * 2
    assert rows.penalties.tolist() == [1.0] * 4 + [2.0] * 2
    assert rows.drafts_before.tolist() == [0, 1, 2, 3, 0, 1]
    assert rows.emit_slots.tolist() == [0, 1]


def test_a_middle_prefill_chunk_claims_no_row():
    """Its token is not sampled, and its step may return before the
    boundary clears a claimed row."""
    table = PresencePenalties(VOCAB, 2, "cpu")
    rows, _ = table.stage(
        _batch([7, 8], [1.0, 1.0], final=[False, True]), None, _buffer()
    )
    assert rows.slots.tolist() == [-1, 0]
    assert rows.emit_slots.tolist() == [-1, 0]
    assert rows.fresh.tolist() == [0]


def test_rows_are_claimed_once_and_reused_after_release():
    table = PresencePenalties(VOCAB, 1, "cpu")
    table.stage(_batch([7, 8], [1.0, 1.0]), None, _buffer())
    again, _ = table.stage(_batch([8, 7], [1.0, 1.0]), None, _buffer())
    assert again.slots.tolist() == [1, 0]
    assert again.fresh.tolist() == []
    table.release([7, 7, 99])  # repeated and unknown releases are harmless
    newcomer, _ = table.stage(_batch([10], [1.0]), None, _buffer())
    assert newcomer.slots.tolist() == [0]
    assert newcomer.fresh.tolist() == [0]


def test_a_resumed_request_marks_its_earlier_output_again():
    table = PresencePenalties(VOCAB, 2, "cpu")
    batch = _batch([7, 8], [1.0, 1.0], resumed={8: np.array([31, 63], np.int32)})
    rows, _ = table.stage(batch, None, _buffer())
    assert rows.fresh.tolist() == [0, 1]
    assert rows.resumed_slots.tolist() == [1, 1]
    assert rows.resumed_ids.tolist() == [31, 63]


def test_the_table_has_two_rows_per_max_num_seqs():
    table = PresencePenalties(VOCAB, 1, "cpu")
    assert table.seen.shape == (2, 3)
    table.stage(_batch([7, 8], [1.0, 1.0]), None, _buffer())
    with pytest.raises(RuntimeError, match="table full"):
        table.stage(_batch([9], [1.0]), None, _buffer())


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
    penalties = PresencePenalties(VOCAB, 1, "cpu")
    penalties.stage(_batch([7], [1.0]), None, _buffer())
    runner = SimpleNamespace(structured_output=None, presence_penalties=penalties)
    boundary.release_token_constraints(runner, [7])
    rows, _ = penalties.stage(_batch([8], [1.0]), None, _buffer())
    assert rows.slots.tolist() == [0]
    assert rows.fresh.tolist() == [0]


def test_decode_then_verify_penalize_what_was_emitted(twins):
    """Request 0 (penalty 1.5) and request 1 (none) through a decode step,
    then a verify step whose second draft is rejected."""
    penalties = PresencePenalties(VOCAB, 2, "cpu")
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


def _emit(penalties, runner, batch, tokens, lengths=None):
    """One step through the boundary: stage, apply, commit `tokens`."""
    batch.presence_rows, _ = penalties.stage(batch, lengths, _buffer())
    logits = torch.zeros(len(batch.req_ids), VOCAB)
    step = boundary.TokenConstraints.apply(runner, batch, logits, None)
    step.commit(torch.tensor(tokens), batch, deferred=False)
    return logits


def test_a_reused_row_is_clear_after_a_step_of_middle_chunks(twins):
    """Request 7 emits 5 and is released. Request 8 then runs a step that is
    only a middle chunk (it returns before the boundary), then its final
    chunk: 5 is not 8's output."""
    penalties = PresencePenalties(VOCAB, 1, "cpu")
    runner = SimpleNamespace(structured_output=None, presence_penalties=penalties)
    _emit(penalties, runner, _batch([7], [1.0]), [5])
    penalties.release([7])
    penalties.stage(_batch([8], [1.0], final=[False]), None, _buffer())
    logits = _emit(penalties, runner, _batch([8], [1.0], final=[True]), [6])
    assert not logits.any()


def test_a_preempted_request_resumes_with_its_output(twins):
    """Request 7 emits 31 and 63 and is preempted; request 8 takes its row and
    emits 5; 7 resumes on another row with 31 and 63 penalized, not 5."""
    penalties = PresencePenalties(VOCAB, 1, "cpu")
    runner = SimpleNamespace(structured_output=None, presence_penalties=penalties)
    _emit(penalties, runner, _batch([7], [1.0]), [31])
    _emit(penalties, runner, _batch([7], [1.0]), [63])
    penalties.release([7])  # `prepare_sample`, from `presence_preempted`
    _emit(penalties, runner, _batch([8], [1.0]), [5])
    resumed = {7: np.array([31, 63], np.int32)}
    batch = _batch([8, 7], [1.0, 1.0], resumed=resumed)
    logits = _emit(penalties, runner, batch, [9, 9])
    assert logits[1].nonzero()[:, 0].tolist() == [31, 63]
    assert logits[0].nonzero()[:, 0].tolist() == [5]


def test_the_scheduler_hands_preempted_and_resumed_requests_to_the_runner():
    from conftest import MockConfig

    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
    from atom.model_engine.sequence import Sequence

    sched = Scheduler(MockConfig(num_kvcache_blocks=64))
    seq = Sequence(
        [10, 11, 12, 13],
        4,
        SamplingParams(max_tokens=40, ignore_eos=True, presence_penalty=1.0),
    )
    sched.add(seq)
    batch, active = sched.schedule()
    assert not batch.presence_resumed and not batch.presence_preempted
    out = ScheduledBatchOutput([seq.id], [(20,)], None, None, None, False)
    sched.postprocess(list(active.values()), out, batch=batch)
    sched.running.remove(seq)
    assert sched.preempt(seq)
    batch, _ = sched.schedule()
    assert batch.presence_preempted == [seq.id]
    assert batch.presence_resumed[seq.id].tolist() == list(seq.output_tokens)
    assert list(seq.output_tokens)


def test_a_batch_without_a_penalty_pickles_no_presence_fields(seq_factory):
    """Every batch is pickled to the TP workers each step: the default path
    carries nothing new."""
    import pickle

    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import SequenceType

    def batch(penalty):
        seq = seq_factory(
            [3, 4], sampling_params=SamplingParams(presence_penalty=penalty)
        )
        seq.type = SequenceType.DECODE
        return ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=[1],
            total_tokens_num=1,
            total_seqs_num=1,
        )

    plain = batch(0.0)
    assert not [name for name in vars(plain) if name.startswith("presence")]
    assert plain.presence_penalties is None
    assert not plain.presence_resumed and not plain.presence_preempted
    restored = pickle.loads(pickle.dumps(plain))
    assert restored.presence_penalties is None and not restored.presence_preempted
    penalized = batch(1.5)
    assert penalized.presence_penalties.tolist() == [1.5]
    assert [n for n in vars(penalized) if n.startswith("presence")] == [
        "presence_penalties"
    ]


def test_ids_preempted_in_a_pass_that_forwards_nothing_wait_for_a_forward(
    monkeypatch,
):
    """Under KV pressure a decode pass can preempt its only seq and return a
    batch without requests, which `EngineCore` does not forward. The id must
    stay for the next forwarded batch, whose `prepare_sample` releases it."""
    from conftest import MockConfig

    from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
    from atom.model_engine.sequence import Sequence

    sched = Scheduler(MockConfig(num_kvcache_blocks=64))
    seq = Sequence(
        [10, 11, 12, 13],
        4,
        SamplingParams(max_tokens=40, ignore_eos=True, presence_penalty=1.0),
    )
    sched.add(seq)
    batch, active = sched.schedule()
    out = ScheduledBatchOutput([seq.id], [(20,)], None, None, None, False)
    sched.postprocess(list(active.values()), out, batch=batch)

    monkeypatch.setattr(sched.block_manager, "_has_page_units", lambda *args: False)
    result = sched.schedule()
    monkeypatch.undo()
    assert seq.id in sched._presence_preempted
    if result is not None:
        assert not result[0].req_ids
        assert not result[0].presence_preempted

    batch, _ = sched.schedule()
    assert list(batch.req_ids) == [seq.id]
    assert batch.presence_preempted == [seq.id]
    assert not sched._presence_preempted
