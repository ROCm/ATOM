# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""When a step takes the previous forward's speculative status.

An output prefill carries no request over, so nothing before its launch reads
that status (only `postprocess` reports it): `ModelRunner.prepare_model`
defers its wait past the target's launch. A decode that carries requests over
shifts its attention metadata by the rejected counts and still waits before.

Pure Python against stand-ins: the real `prepare_model` deferral decision and
`tokenIDProcessor.prepare_input_ids`, every other producer stubbed."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def runner_module():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


# The previous output forward's (rejected, bonus) counts, one per request.
STATUS = (np.array([2, 1], np.int32), np.array([1, 0], np.int32))


def _batch(*, prefill, final=True):
    """Two requests: prefills on their last chunk (or a middle one), or decodes
    carried over from the previous forward."""
    return SimpleNamespace(
        total_seqs_num=2,
        total_tokens_num=12 if prefill else 4,
        has_prefill=prefill,
        num_spec_step=0 if prefill else 1,
        produces_output=lambda: final,
        num_rejected=np.zeros(2, np.int32),
        num_bonus=np.zeros(2, np.int32),
    )


def _runner(runner_module, monkeypatch, batch, *, builder_defers=False):
    """A runner whose `prepare_model` reaches the deferral decision and the real
    `prepare_input_ids`; `events` records the status wait against the host
    metadata build (`prepare_inputs`)."""
    events = []
    carried = SimpleNamespace(deferred_curr=np.arange(2), deferred_prev=np.arange(2))
    processor = object.__new__(runner_module.tokenIDProcessor)
    processor.runner = SimpleNamespace(h2d_groups=None)  # no publication registry
    processor.batch_info = SimpleNamespace(np=np.zeros(3, np.int32))
    processor.draft_tokens_per_seq = 0
    processor._stage_input_ids = lambda b, q, group: (
        "ids",
        None if b.has_prefill else carried,
    )
    processor.recv_spec_status = mock.Mock(
        side_effect=lambda: events.append("wait") or tuple(a.copy() for a in STATUS)
    )
    processor.prev_batch = SimpleNamespace(total_seqs_num=2)
    processor.prev_rejected_num = processor.prev_bonus_num = None
    processor.spec_status_owed = False

    mode = SimpleNamespace(
        sync=SimpleNamespace(any_rank_has_prefill=batch.has_prefill),
        is_prefill=batch.has_prefill,
        running_bs=2,
        running_tokens=batch.total_tokens_num,
        decode_ubatch_slices=None,
        max_seqlen_q=batch.num_spec_step + 1,
    )
    monkeypatch.setattr(
        runner_module, "ForwardMode", SimpleNamespace(decide=lambda **_: mode)
    )
    builder = SimpleNamespace(
        step_needs_eager=lambda b: False,
        query_lengths=lambda *a: None,
        prepare_query_metadata=lambda *a: {},
        consumes_device_verdict=mock.Mock(return_value=builder_defers),
    )
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_tbo=False,
        enable_tbo_decode=False,
    )
    runner.enforce_eager = True
    runner.capture_sizes_np = np.array([], np.int64)
    runner.running_tokens_per_dp_rank = np.zeros(1, np.int64)
    runner.tbo_decode_graphs = False
    runner.attn_metadata_builder = builder
    runner.h2d_groups = {"token_inputs": SimpleNamespace(set_counts=lambda c: None)}
    runner.tokenID_processor = processor
    runner._dspark_apply_q_bucket = lambda b: None
    runner._local_tbo_eligibility = lambda b: (False, False, 0, 0)
    runner._producer_graph_shapes = lambda: None
    runner._cut_ubatches = lambda m, b: m
    runner.prepare_sample = lambda b, group: (None, None, None, True, False)
    runner.prepare_inputs = lambda *a, **k: events.append("metadata")
    return runner, processor, builder, events


def _launch(runner, processor, events):
    """`forward` after `prepare_model`: the target's launch, then the wait the
    step deferred."""
    events.append("launch")
    if runner.spec_status_deferred:
        processor.take_deferred_spec_status()


def test_an_output_prefill_takes_the_status_after_its_launch(
    runner_module, monkeypatch
):
    batch = _batch(prefill=True)
    runner, processor, builder, events = _runner(runner_module, monkeypatch, batch)
    runner_module.ModelRunner.prepare_model(runner, batch)
    processor.recv_spec_status.assert_not_called()
    assert runner.spec_status_deferred and processor.spec_status_owed
    # The builder is still asked every step, a prefill's included.
    builder.consumes_device_verdict.assert_called_once_with(batch, 1)
    _launch(runner, processor, events)
    assert events == ["metadata", "launch", "wait"]
    # `postprocess` reports the previous forward's real counts.
    assert processor.prev_rejected_num.tolist() == [2, 1]
    assert processor.prev_bonus_num.tolist() == [1, 0]
    assert not processor.spec_status_owed


def test_a_middle_prefill_neither_waits_nor_owes(runner_module, monkeypatch):
    """Its `postprocess` is skipped, so the status stays queued for the next
    output-producing step."""
    batch = _batch(prefill=True, final=False)
    runner, processor, _, events = _runner(runner_module, monkeypatch, batch)
    runner_module.ModelRunner.prepare_model(runner, batch)
    _launch(runner, processor, events)
    processor.recv_spec_status.assert_not_called()
    assert not processor.spec_status_owed and processor.prev_rejected_num is None


def test_a_decode_still_waits_before_its_metadata(runner_module, monkeypatch):
    """A decode carries requests over and shifts their metadata by the
    rejected counts, so it takes the status before building it."""
    batch = _batch(prefill=False)
    runner, processor, _, events = _runner(runner_module, monkeypatch, batch)
    runner_module.ModelRunner.prepare_model(runner, batch)
    assert not runner.spec_status_deferred
    _launch(runner, processor, events)
    assert events == ["wait", "metadata", "launch"]
    assert processor.num_rejected.tolist() == [2, 1]


def test_a_decode_the_builder_advances_on_the_device_waits_after(
    runner_module, monkeypatch
):
    batch = _batch(prefill=False)
    runner, processor, _, events = _runner(
        runner_module, monkeypatch, batch, builder_defers=True
    )
    runner_module.ModelRunner.prepare_model(runner, batch)
    _launch(runner, processor, events)
    assert events == ["metadata", "launch", "wait"]
    processor.recv_spec_status.assert_called_once_with()
