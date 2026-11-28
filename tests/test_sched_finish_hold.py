# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Flag ATOM_SCHED_FINISH_HOLD_LEFT: hold prefill while a short running
# request is about to reach max_tokens (atom/model_engine/prefill_finish_hold.py).
from collections import deque
from types import SimpleNamespace
from unittest import mock

from conftest import MockConfig

from atom.model_engine.prefill_finish_hold import FinishHold
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus, SequenceType
from atom.sampling_params import SamplingParams


def _seq(max_tokens, done, partial=False):
    return SimpleNamespace(
        max_tokens=max_tokens, num_completion_tokens=done, is_partial_prefill=partial
    )


def test_due_needs_short_request_near_its_end():
    fh = FinishHold(tokens_left=64, max_hold_s=0.5, max_tokens_cap=400)
    assert fh.due([_seq(300, 240)])
    assert fh.due([_seq(300, 236)])  # exactly 64 left
    assert not fh.due([_seq(300, 235)])  # 65 left
    # Long requests amortise a stall over many tokens: never held for.
    assert not fh.due([_seq(401, 400)])
    # A partial prefill is not a decode.
    assert not fh.due([_seq(100, 90, partial=True)])
    assert fh.due([_seq(5000, 10), _seq(100, 90)])
    assert not fh.due([])


def test_hold_is_bounded_and_resets():
    fh = FinishHold(tokens_left=64, max_hold_s=0.5, max_tokens_cap=400)
    assert fh.hold(10.0, True)
    assert fh.hold(10.4, True)
    # Bound reached: one tick released, then a new hold may start.
    assert not fh.hold(10.5, True)
    assert fh.releases == 1
    assert fh.hold(10.6, True)
    # Not due: released at once and the hold clock restarts.
    assert not fh.hold(10.7, False)
    assert fh.hold(20.0, True)
    assert fh.hold(20.49, True)


def _sched(**env):
    with mock.patch.dict("os.environ", {k: str(v) for k, v in env.items()}):
        return Scheduler(
            MockConfig(
                max_num_seqs=4,
                max_num_batched_tokens=512,
                max_model_len=4096,
                num_kvcache_blocks=400,
                enable_chunked_prefill=True,
            )
        )


def _decoding(seq_factory, max_tokens):
    seq = seq_factory(
        [1, 2, 3, 4], sampling_params=SamplingParams(max_tokens=max_tokens)
    )
    seq.status = SequenceStatus.RUNNING
    seq.type = SequenceType.DECODE
    return seq


def test_off_by_default():
    assert _sched()._finish_hold is None


def test_holds_prefill_while_short_decode_finishes(seq_factory):
    sched = _sched(
        ATOM_SCHED_FINISH_HOLD_LEFT=64,
        ATOM_SCHED_FINISH_HOLD_MAX_MS=500,
        ATOM_SCHED_FINISH_HOLD_MAX_TOKENS=400,
    )
    assert sched._finish_hold is not None
    sched.running = deque([_decoding(seq_factory, 50)])
    sched.add(seq_factory(list(range(64))))
    batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 0
    assert batch.total_seqs_num_decode == 1
    assert len(sched.waiting) == 1
    # Past the bound the prefill goes.
    with mock.patch(
        "atom.model_engine.scheduler.time.perf_counter",
        return_value=sched._finish_hold._since + 1.0,
    ):
        batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 1


def test_no_hold_for_long_or_distant_requests(seq_factory):
    env = {
        "ATOM_SCHED_FINISH_HOLD_LEFT": 64,
        "ATOM_SCHED_FINISH_HOLD_MAX_MS": 500,
        "ATOM_SCHED_FINISH_HOLD_MAX_TOKENS": 400,
    }
    for max_tokens in (300, 1000):  # 300 left > 64; 1000 > cap
        sched = _sched(**env)
        sched.running = deque([_decoding(seq_factory, max_tokens)])
        sched.add(seq_factory(list(range(64))))
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 1


def test_no_hold_without_decode(seq_factory):
    sched = _sched(ATOM_SCHED_FINISH_HOLD_LEFT=64)
    sched.add(seq_factory(list(range(64))))
    batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 1


def test_zero_max_hold_never_holds():
    fh = FinishHold(tokens_left=64, max_hold_s=0.0, max_tokens_cap=400)
    assert not fh.hold(10.0, True)
    assert not fh.hold(10.1, True)
    assert fh.holds == 0
