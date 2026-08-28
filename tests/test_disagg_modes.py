# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for intra-GPU disagg constrained vs unconstrained modes.

Only the scheduler-level shm gating is exercised here; the IPC handshake
and CUDA stream pool are out of scope for the no-GPU test environment.
"""

import pytest
from conftest import MockConfig


@pytest.fixture
def prefill_scheduler_unconstrained():
    from atom.model_engine.scheduler import PrefillScheduler

    return PrefillScheduler(MockConfig(), disagg_cu_shm_name="")


@pytest.fixture
def decode_scheduler_unconstrained():
    from atom.model_engine.scheduler import DecodeScheduler

    return DecodeScheduler(MockConfig(), disagg_cu_shm_name="")


@pytest.fixture
def seq_factory():
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    def make(token_ids, block_size=4):
        return Sequence(token_ids, block_size, sampling_params=SamplingParams())

    return make


# ── Unconstrained: no shm handle attached ────────────────────────────────


def test_prefill_scheduler_skips_shm_when_name_empty(prefill_scheduler_unconstrained):
    assert prefill_scheduler_unconstrained._cu_shm is None


def test_decode_scheduler_skips_shm_when_name_empty(decode_scheduler_unconstrained):
    assert decode_scheduler_unconstrained._cu_shm is None


# ── Unconstrained: batches carry cu_stream_fraction=None ─────────────────


def test_unconstrained_prefill_batch_has_none_cu_fraction(
    prefill_scheduler_unconstrained, seq_factory
):
    """Without shm, PrefillScheduler must produce batches keyed by the
    plain (None) stream — never a fractional CU mask."""
    seq = seq_factory([10, 20, 30, 40])
    seq.block_table = [0, 1]
    seq.num_cached_tokens = 0
    prefill_scheduler_unconstrained.add(seq)

    batch, _ = prefill_scheduler_unconstrained.schedule()
    assert batch is not None
    assert batch.cu_stream_fraction is None


# ── engine-status line: the two P/D processes must be distinguishable ─────


def test_pd_schedulers_label_their_engine_lines(
    prefill_scheduler_unconstrained, decode_scheduler_unconstrained
):
    """Both P/D processes run as engine index 0 and usually log to the same
    place, so the line carries a label; the aggregated engine keeps none."""
    from atom.model_engine.scheduler import Scheduler

    assert prefill_scheduler_unconstrained.engine_stats.label == "Prefill "
    assert decode_scheduler_unconstrained.engine_stats.label == "Decode "
    assert Scheduler(MockConfig()).engine_stats.label == ""


def test_decode_request_counts_fold_in_the_pd_queues(
    decode_scheduler_unconstrained, seq_factory
):
    """`allocate_waiting()` drains `waiting` almost at once, so a decode
    engine at full load parks its requests in `prefill_waiting` /
    `prefill_done`. Counting only the base pair reports an idle engine."""
    sched = decode_scheduler_unconstrained
    assert sched.get_request_counts() == (0, 0)

    parked = seq_factory([1, 2, 3, 4])
    ready = seq_factory([5, 6, 7, 8])
    sched.prefill_waiting[parked.id] = parked
    sched.prefill_done.append(ready)

    running, waiting = sched.get_request_counts()
    assert waiting == 1, "prefill_waiting must count as waiting"
    assert running == 1, "prefill_done must count as running"


def test_decode_metrics_and_status_line_agree(
    decode_scheduler_unconstrained, seq_factory
):
    """The status line and `/metrics` must not describe the same engine
    differently. `collect_metrics` reads `get_request_counts`, so overriding
    anything else would leave the dashboard reporting an idle engine while the
    log says it is at full load."""
    sched = decode_scheduler_unconstrained
    for i in range(3):
        sched.prefill_waiting[i] = seq_factory([1, 2, 3, 4])
    sched.prefill_done.append(seq_factory([5, 6, 7, 8]))

    running, waiting = sched.get_request_counts()
    assert (running, waiting) == (1, 3)
    # Everything else that counts in-flight work derives from that one pair.
    assert sched.get_num_unfinished_requests() == 4
    assert sched.has_requests() is True
    assert sched.has_unfinished_requests() is True


def test_metrics_work_on_the_prefill_scheduler():
    """`PrefillEngineCore` swaps in a `PrefillScheduler` and rewires the
    utility handler to it, so `collect_metrics` has to survive a scheduler
    that is not a `Scheduler` subclass and owns no BlockManager — it used to
    raise AttributeError on `block_manager.kv` inside the busy loop."""
    import queue

    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.model_engine.scheduler import PrefillScheduler

    sched = PrefillScheduler(MockConfig(enable_log_stats=True))
    handler = EngineUtilityHandler(
        runner_mgr=None,
        output_queue=queue.Queue(),
        label="PrefillEngineCore",
        scheduler=sched,
    )

    metrics = handler.collect_metrics()
    assert metrics["enabled"] is True
    assert metrics["requests_running"] == 0
    # No KV pool on this side (decode owns the blocks), so the snapshot omits
    # those keys rather than reporting a pool of size zero. The aggregator
    # sums with `.get(key, 0)`, so the decode rank's real figures still land.
    assert not [k for k in metrics if k.startswith("kv_blocks")]

    handler.push_metrics()  # must not raise from inside the busy loop


def test_idle_heartbeat_is_available_on_every_scheduler(caplog):
    """The busy loop calls this on every idle pass, whichever scheduler the
    engine core is driving, so all three must accept it, close the window,
    and stay silent while nothing is running."""
    import logging
    import time

    from atom.model_engine.scheduler import (
        DecodeScheduler,
        PrefillScheduler,
        Scheduler,
    )

    cfg = MockConfig(enable_log_stats=True)
    for sched in (
        PrefillScheduler(cfg, disagg_cu_shm_name=""),
        DecodeScheduler(cfg, disagg_cu_shm_name=""),
        Scheduler(cfg),
    ):
        stats = sched.engine_stats
        before = stats._throughput_last_log_time
        sched.heartbeat_throughput(time.monotonic())
        assert stats._throughput_last_log_time == before, "window was not due"

        # Force the window open on an engine with nothing running or queued:
        # it must close (fresh start) without logging.
        stats._throughput_last_log_time -= 43.0
        with caplog.at_level(logging.INFO, logger="atom"):
            sched.heartbeat_throughput(time.monotonic())
        assert time.monotonic() - stats._throughput_last_log_time < 1.0
        assert "Engine" not in caplog.text
        caplog.clear()


def test_idle_heartbeat_is_inert_when_log_stats_off():
    import time

    from atom.model_engine.scheduler import Scheduler

    sched = Scheduler(MockConfig(enable_log_stats=False))
    sched.engine_stats._throughput_last_log_time -= 43.0
    before = sched.engine_stats._throughput_last_log_time
    sched.heartbeat_throughput(time.monotonic())
    assert sched.engine_stats._throughput_last_log_time == before


def test_decode_schedule_records_throughput_when_log_stats_on(seq_factory):
    """DecodeScheduler overrides schedule() without calling super(), so its
    throughput wiring is separate code — exercise it with the production
    default (log stats ON) on both the empty and the scheduled path."""
    from atom.model_engine.scheduler import DecodeScheduler

    sched = DecodeScheduler(MockConfig(enable_log_stats=True), disagg_cu_shm_name="")
    assert sched.engine_stats.throughput_enabled is True

    # Empty path: `running` is empty, so schedule() takes the early return.
    # Must still tick the cadence rather than raise.
    assert sched.schedule() is None

    # Scheduled path: a promoted seq goes through the full return.
    seq = seq_factory([1, 2, 3, 4])
    sched.prefill_done.append(seq)
    batch, seqs = sched.schedule()
    assert batch is not None
    assert seq.id in seqs
