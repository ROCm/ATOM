# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""When a step takes the previous forward's speculative status.

An output prefill carries no request over, so nothing before its launch reads
that status (only `postprocess` reports it): `ModelRunner.prepare_model`
defers its wait past the target's launch. A decode that carries requests over
shifts its attention metadata by the rejected counts and still waits before.
The builder is asked every step, a prefill's included: the V4.1 mono device
round (`DeviceRound.consumes`) learns from that call which forward ran last.

Pure Python against stand-ins: the real `prepare_model` deferral decision and
`tokenIDProcessor.prepare_input_ids`, every other producer stubbed; the device
round is the real one, its Triton kernel never launched."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def runner_module():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


@pytest.fixture(scope="module")
def device_round():
    """`mono/round/device_round.py` loaded by path, as the mono host tests do."""
    pytest.importorskip("triton")
    path = (
        Path(__file__).resolve().parents[1]
        / "atom/models/deepseek_v41/mono/round/device_round.py"
    )
    old = os.environ.get("TRITON_INTERPRET")
    os.environ["TRITON_INTERPRET"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("_spec_status_dr", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop("TRITON_INTERPRET")
        else:
            os.environ["TRITON_INTERPRET"] = old
    return module


# The previous output forward's (rejected, bonus) counts, one per request.
STATUS = (np.array([2, 1], np.int32), np.array([1, 0], np.int32))


def _batch(*, prefill, final=True, req_ids=(7, 8)):
    """Two requests: prefills on their last chunk (or a middle one), or decodes
    carried over from the previous forward."""
    return SimpleNamespace(
        total_seqs_num=2,
        total_seqs_num_prefill=2 if prefill else 0,
        req_ids=list(req_ids),
        is_dummy_run=False,
        is_mixed=False,
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
        settles_spec_status=lambda *a: False,
        take_spec_status=lambda processor: None,
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


def _round_step(runner_module, monkeypatch, rnd, batch):
    """`prepare_model` of one forward with the V4.1 mono builder's answer: the
    real device round behind `consumes_device_verdict`."""
    runner, _, builder, _ = _runner(runner_module, monkeypatch, batch)
    builder.consumes_device_verdict = rnd.consumes
    runner_module.ModelRunner.prepare_model(runner, batch)
    return runner


def _serve(device_round, rnd, requests=2):
    """The builder's layout and the commit of a served lean verify
    (`HiRound.laid_out` / `committed`): it leaves its verdict to the next
    step."""
    metadata = SimpleNamespace(
        served_by_round=device_round.RoundService.LEAN,
        step=SimpleNamespace(slots=np.zeros(requests), running_tokens=2 * requests),
    )
    rnd.laid_out(metadata)
    rnd.committed(metadata, np.zeros(requests, np.int32))


def _hook():
    """A hook serving two-request verifies of one draft (width 4)."""
    return SimpleNamespace(
        lean_host=True,
        runs_every_layer=True,
        served_widths={4},
        split={4: (2, 2)},
        bmax=2,
    )


def test_a_prefill_between_two_served_verifies_ends_the_device_round(
    runner_module, device_round, monkeypatch
):
    """A decode advances on the device from the previous forward's verdict
    only when that forward was a served verify of the same requests. A middle
    prefill chunk of other requests runs between them: the host takes that
    verdict after the prefill's launch, so the next verify waits for its own
    status instead of advancing a second time."""
    rnd = device_round.DeviceRound(_hook())
    runner = _round_step(runner_module, monkeypatch, rnd, _batch(prefill=False))
    assert not runner.spec_status_deferred  # no verdict on the device yet
    _serve(device_round, rnd)
    assert rnd.state.ready == (7, 8)
    middle = _batch(prefill=True, final=False, req_ids=(9, 10))
    _round_step(runner_module, monkeypatch, rnd, middle)
    assert rnd.state.ready is None
    runner = _round_step(runner_module, monkeypatch, rnd, _batch(prefill=False))
    assert not runner.spec_status_deferred
    assert rnd.state.advance_from is None


def test_the_first_verify_after_its_prefill_skips_the_wait(
    runner_module, device_round, monkeypatch
):
    """The prefill's status is known (nothing rejected), so the first served
    verify of the requests it prefilled waits after its launch
    (`device_round.first_step`)."""
    rnd = device_round.DeviceRound(_hook())
    _round_step(runner_module, monkeypatch, rnd, _batch(prefill=True))
    assert rnd.state.prefilled == (7, 8)
    runner = _round_step(runner_module, monkeypatch, rnd, _batch(prefill=False))
    assert runner.spec_status_deferred and rnd.state.first
