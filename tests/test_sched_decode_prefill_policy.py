# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Prefill policy while decodes run: ATOM_SCHED_DECODE_PREFILL_CAP and
# ATOM_SCHED_PREFILL_DUTY (atom/model_engine/scheduler.py, prefill_duty.py).
from collections import deque
from unittest import mock

from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus, SequenceType


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


def _decoding(seq_factory, tokens):
    seq = seq_factory(tokens)
    seq.status = SequenceStatus.RUNNING
    seq.type = SequenceType.DECODE
    return seq


def test_cap_applies_only_while_decoding(seq_factory):
    sched = _sched(ATOM_SCHED_DECODE_PREFILL_CAP=128)
    assert sched._decode_prefill_cap == 128
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    # No decode running: the full 512-token budget.
    assert batch.total_tokens_num_prefill == 400

    sched = _sched(ATOM_SCHED_DECODE_PREFILL_CAP=128)
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 128
    # The budget is restored after the tick.
    assert sched.max_num_batched_tokens == 512


def test_cap_off_by_default(seq_factory):
    sched = _sched()
    assert not sched._decode_prefill_cap
    assert sched._prefill_duty is None
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 400


def test_duty_hold_schedules_decode_only(seq_factory):
    sched = _sched(ATOM_SCHED_PREFILL_DUTY=0.5)
    assert sched._prefill_duty is not None
    dec = _decoding(seq_factory, [1, 2, 3, 4])
    sched.running = deque([dec])
    sched.add(seq_factory(list(range(64))))
    with mock.patch.object(sched._prefill_duty, "hold", return_value=True) as hold:
        batch, _ = sched.schedule()
    hold.assert_called_once()
    assert batch.total_seqs_num_prefill == 0
    assert batch.total_seqs_num_decode == 1
    assert len(sched.waiting) == 1
    with mock.patch.object(sched._prefill_duty, "hold", return_value=False):
        batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 1


def test_idle_cap_applies_only_without_decodes(seq_factory):
    # ATOM_SCHED_IDLE_PREFILL_CAP is the idle twin.
    sched = _sched(ATOM_SCHED_IDLE_PREFILL_CAP=256)
    assert sched._idle_prefill_cap == 256 and not sched._decode_prefill_cap
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 256
    assert sched.max_num_batched_tokens == 512

    sched = _sched(ATOM_SCHED_IDLE_PREFILL_CAP=256)
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    # decodes running and no decode cap: the full budget, as before
    assert batch.total_tokens_num_prefill == 400


def test_idle_and_decode_caps_together(seq_factory):
    env = {"ATOM_SCHED_IDLE_PREFILL_CAP": 256, "ATOM_SCHED_DECODE_PREFILL_CAP": 128}
    sched = _sched(**env)
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 256
    sched = _sched(**env)
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 128
    assert sched.max_num_batched_tokens == 512


def test_idle_cap_off_by_default_and_clamped(seq_factory):
    assert _sched()._idle_prefill_cap == 0
    assert _sched(ATOM_SCHED_IDLE_PREFILL_CAP=4096)._idle_prefill_cap == 512


def _sched_no_chunking(**env):
    with mock.patch.dict("os.environ", {k: str(v) for k, v in env.items()}):
        return Scheduler(
            MockConfig(
                max_num_seqs=4,
                max_num_batched_tokens=512,
                max_model_len=4096,
                num_kvcache_blocks=400,
                enable_chunked_prefill=False,
            )
        )


def test_caps_are_ignored_without_chunked_prefill(seq_factory):
    # A 400-token prompt cannot be split; under a 128-token cap it would have
    # been rejected as "input tokens > max_num_batched_tokens=128".
    sched = _sched_no_chunking(ATOM_SCHED_DECODE_PREFILL_CAP=128)
    assert sched._decode_prefill_cap == 0
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 400
    assert not sched._rejected


def test_capped_tick_judges_capacity_against_the_configured_budget(seq_factory):
    sched = _sched(ATOM_SCHED_DECODE_PREFILL_CAP=128)
    seq = seq_factory(list(range(400)))
    sched.enable_chunked_prefill = False  # atomic prompt (as multimodal is)
    assert sched._unschedulable_reason(seq) is None  # 400 <= 512
    sched.max_num_batched_tokens = 128
    sched._uncapped_max_num_batched_tokens = 512  # inside a capped tick
    assert sched._unschedulable_reason(seq) is None
    sched._uncapped_max_num_batched_tokens = None  # same budget, a real limit
    assert "max_num_batched_tokens=128" in sched._unschedulable_reason(seq)


def test_nonpositive_duty_time_constant_is_rejected_at_startup():
    import pytest

    for bad in ("0", "-1", "nan"):
        with pytest.raises(ValueError):
            _sched(ATOM_SCHED_PREFILL_DUTY=0.5, ATOM_SCHED_PREFILL_DUTY_TAU_S=bad)


def test_negative_caps_are_rejected_at_startup():
    import pytest

    for name in ("ATOM_SCHED_DECODE_PREFILL_CAP", "ATOM_SCHED_IDLE_PREFILL_CAP"):
        with pytest.raises(ValueError):
            _sched(**{name: -1})
        with pytest.raises(ValueError):
            _sched(**{name: "x"})


def test_decode_cap_is_floored_at_one_full_decode_batch():
    # (spec_k + 1) * max_num_seqs = 4 here: a smaller cap would starve decode.
    assert _sched(ATOM_SCHED_DECODE_PREFILL_CAP=1)._decode_prefill_cap == 4


def test_atomic_prompt_above_the_cap_is_admitted(seq_factory):
    # An atomic (never chunked) prompt larger than the idle cap used to be
    # requeued on every tick, with nothing running to change the cap.
    sched = _sched(ATOM_SCHED_IDLE_PREFILL_CAP=256)
    seq = seq_factory(list(range(400)))
    seq.multimodal_data = {}  # no embedding spans: atomic
    sched.add(seq)
    batch, _ = sched.schedule()
    assert batch is not None and batch.total_tokens_num_prefill == 400
    assert sched.max_num_batched_tokens == 512
    # A chunkable prompt is still capped.
    sched = _sched(ATOM_SCHED_IDLE_PREFILL_CAP=256)
    sched.add(seq_factory(list(range(400))))
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 256


def test_observe_is_fed_by_every_schedule_call(seq_factory):
    sched = _sched(ATOM_SCHED_PREFILL_DUTY=0.5, ATOM_SCHED_ITL_RISK_MS=5.0)
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    with (
        mock.patch.object(sched._prefill_duty, "observe") as duty_obs,
        mock.patch.object(sched._itl_risk, "observe") as risk_obs,
    ):
        sched.schedule()
        sched.add(seq_factory(list(range(64))))
        sched.schedule()
    assert [c.args[1] for c in duty_obs.call_args_list] == ["d", "p"]
    assert [c.args[1] for c in risk_obs.call_args_list] == ["d", "p"]


def test_local_holds_stay_off_when_the_group_prefills(seq_factory):
    # With token graphs, the group prefill decision is the delayer's reduce;
    # the per-rank holds must not veto it (they would turn this rank's decode
    # into an idle step).
    sched = _sched(ATOM_SCHED_PREFILL_DUTY=0.5, ATOM_SCHED_FINISH_HOLD_LEFT=8)
    sched.running = deque([_decoding(seq_factory, [1, 2, 3, 4])])
    sched.add(seq_factory(list(range(64))))
    with (
        mock.patch.object(sched, "_group_prefill_tick", return_value=True),
        mock.patch.object(sched._prefill_duty, "hold") as duty_hold,
        mock.patch.object(sched._finish_hold, "hold") as finish_hold,
    ):
        sched.schedule()
    duty_hold.assert_not_called()
    finish_hold.assert_not_called()


def test_disaggregated_prefill_scheduler_warns_about_ignored_flags(caplog):
    import logging

    from atom.model_engine.scheduler import PrefillScheduler

    cfg = MockConfig(
        max_num_seqs=4,
        max_num_batched_tokens=512,
        max_model_len=4096,
        num_kvcache_blocks=400,
        enable_chunked_prefill=True,
    )
    env = {"ATOM_SCHED_DECODE_PREFILL_CAP": "128", "ATOM_SCHED_PREFILL_DUTY": "0.5"}
    with mock.patch.dict("os.environ", env), caplog.at_level(logging.WARNING):
        PrefillScheduler(cfg)
    assert "ATOM_SCHED_DECODE_PREFILL_CAP" in caplog.text
    assert "ATOM_SCHED_PREFILL_DUTY" in caplog.text


def test_capped_stateful_prefill_budget_covers_the_replay(seq_factory):
    # A forward that finishes a stateful prompt starts `min_final_prefill_rows`
    # before its end (`replay_start`), so with a cap near that size the next
    # chunk ended at or below `resume` and `_partial_prefill_chunk` returned 0
    # on every capped tick. The tick's budget must cover replay + one block.
    sched = _sched(ATOM_SCHED_IDLE_PREFILL_CAP=128)
    seq = seq_factory(list(range(200)))
    seq.status = SequenceStatus.RUNNING
    seq.num_cached_tokens = 128
    seq.is_partial_prefill = True
    sched.running = deque([seq])
    sched._partial_prefill_count = 1
    seen = []
    with (
        mock.patch.object(sched.block_manager, "replay_start", return_value=0),
        mock.patch.object(
            sched,
            "_schedule",
            side_effect=lambda: seen.append(sched.max_num_batched_tokens),
        ),
    ):
        sched.schedule()
    assert seen == [128 + sched.block_manager.hash_block_size]
    assert sched.max_num_batched_tokens == 512
