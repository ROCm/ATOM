# SPDX-License-Identifier: MIT

"""Streaming a P/D consumer's injected T0 ahead of its first decode
(``ATOM_PD_EARLY_T0``)."""

from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.sampling_params import SamplingParams

with stubbed_aiter():
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.pp_engine_core import PPEngineCoreProc

MTP_K = 3
T0 = 99


def _scheduler():
    return Scheduler(
        MockConfig(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=MTP_K, use_dspark=lambda: False
            ),
            num_kvcache_blocks=64,
        )
    )


def _first_decode(sched, seq_factory, max_tokens=8):
    seq = seq_factory(
        [1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )
    seq.kv_transfer_params = {"first_token_id": T0, "transfer_id": 5}
    sched._schedule_first_decode_after_remote_kv(seq)
    return seq


@pytest.mark.parametrize("flag", ["0", "1"])
def test_t0_is_queued_once_and_not_left_for_postprocess(monkeypatch, seq_factory, flag):
    monkeypatch.setenv("ATOM_PD_EARLY_T0", flag)
    sched = _scheduler()

    seq = _first_decode(sched, seq_factory)

    assert list(seq.token_ids)[4] == T0
    early = sched.take_early_outputs()
    if flag == "1":
        assert seq._injected_t0 is None
        [(seq_id, output)] = early
        assert seq_id == seq.id
        assert output.output_tokens == [T0]
        assert not output.finished and output.finish_reason is None
        # Postprocess no longer sees T0, so these are taken when it is queued.
        assert seq.first_token_time > 0.0
        assert sched.engine_stats.num_generation_tokens == 1
    else:
        assert seq._injected_t0 == T0
        assert early == []
        assert seq.first_token_time == 0.0
        assert sched.engine_stats.num_generation_tokens == 0
    assert sched.take_early_outputs() == []


def test_no_early_t0_without_room_for_it(monkeypatch, seq_factory):
    monkeypatch.setenv("ATOM_PD_EARLY_T0", "1")
    sched = _scheduler()

    seq = _first_decode(sched, seq_factory, max_tokens=0)

    assert sched.take_early_outputs() == []
    assert seq._injected_t0 == T0


@pytest.mark.parametrize(
    ("max_tokens", "expected_rest", "expected_count"),
    [(1, [], 1), (3, [10, 11], 3)],
)
def test_postprocess_counts_but_does_not_resend_an_early_t0(
    seq_factory, max_tokens, expected_rest, expected_count
):
    sched = _scheduler()
    seq = seq_factory(
        [1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )
    sched.add(seq)
    sched.schedule()
    # T0 is sequence state, as `_schedule_first_decode_after_remote_kv` leaves
    # it, but already streamed, so `_injected_t0` is clear.
    seq.append_token(T0)
    seq._injected_t0 = None
    seq.num_placeholder_tokens = MTP_K
    for token in (87, 88, 89, 90, 91, 92, 93):
        seq.append_token(token)
    stream_queue = mock.Mock()

    finished = sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(10, 11, 12, 13)],
            num_rejected=np.asarray([0]),
            num_bonus=np.asarray([0]),
            draft_token_ids=np.asarray([[20, 21, 22]]),
            is_deferred_out=True,
        ),
        stream_output_queue=stream_queue,
    )

    assert finished[0].leave_reason == "max_tokens"
    assert finished[0].num_completion_tokens == expected_count
    emitted = stream_queue.put_nowait.call_args.args[0][0][1]
    assert emitted.output_tokens == expected_rest
    assert emitted.finished is True


def test_engine_streams_early_t0_before_the_forward():
    order = []
    early = [(7, SimpleNamespace(output_tokens=[T0]))]
    step_outputs = [(7, SimpleNamespace(output_tokens=[10]))]
    stream_queue = mock.Mock()
    stream_queue.empty.side_effect = [False, True]
    stream_queue.get_nowait.return_value = step_outputs
    batch = SimpleNamespace(
        connector_meta_output=None, req_ids=[7], total_seqs_num_prefill=0
    )

    core = EngineCore.__new__(EngineCore)
    core.kv_transfer_enabled = False
    core.stream_output_queue = stream_queue
    core.output_queue = SimpleNamespace(
        put_nowait=lambda item: order.append(("out", item))
    )
    core.scheduler = SimpleNamespace(
        schedule=lambda: (batch, {7: object()}),
        take_rejected=list,
        take_early_outputs=lambda: list(early),
        compute_detailed_aggregates=lambda *_: None,
        metrics=SimpleNamespace(record_forward=lambda *_: None),
        prefill_delayer=None,
        postprocess=lambda *_a, **_k: [],
    )
    core.runner_mgr = SimpleNamespace(
        call_func=lambda name, *_a, **_k: order.append(("call", name))
    )
    core._poll_kv_transfer_progress = lambda: None

    assert core._process_engine_step_inner() is True

    assert order == [
        ("out", ("STREAM", early)),
        ("call", "forward"),
        ("out", ("STREAM", step_outputs)),
    ]


def test_pp_head_loop_streams_early_t0():
    order = []
    early = [(7, SimpleNamespace(output_tokens=[T0]))]
    pending = [list(early)]

    core = PPEngineCoreProc.__new__(PPEngineCoreProc)
    core.pp_size = 2
    core._in_flight = deque()
    core.output_queue = SimpleNamespace(
        put_nowait=lambda item: order.append(("out", item))
    )
    core.scheduler = SimpleNamespace(
        schedule=lambda: None,
        take_rejected=list,
        take_early_outputs=lambda: pending.pop() if pending else [],
    )
    core.runner_mgr = SimpleNamespace(
        call_func=lambda name, *_a, **_k: order.append(("call", name))
    )
    core._advance_idle_kv_transfer = lambda: None
    core._poll_kv_transfer_progress = lambda: None
    core._poll_dynamic_chunking_calibration = lambda: None

    core._pp_head_step()

    assert order == [("out", ("STREAM", early)), ("call", "flush_pp_send")]
