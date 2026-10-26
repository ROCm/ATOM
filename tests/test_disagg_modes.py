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


def test_decode_handoff_builds_full_mtp_window(seq_factory):
    class SpecConfig:
        num_speculative_tokens = 3

        @staticmethod
        def use_dspark():
            return True

    from atom.model_engine.scheduler import DecodeScheduler

    scheduler = DecodeScheduler(
        MockConfig(speculative_config=SpecConfig()), disagg_cu_shm_name=""
    )
    prompt = [11, 22, 33, 44, 55]
    t0 = 14
    seq = seq_factory(prompt)
    scheduler.block_manager.allocate(seq)
    scheduler.prefill_waiting[seq.id] = seq

    scheduler.on_prefill_done(seq.id, len(prompt), t0)

    assert seq.is_first_decode is True
    assert list(seq.token_ids[-4:]) == [t0, 2, 2, 2]
    assert not set(seq.token_ids[-4:]) & set(prompt)

    batch, _ = scheduler.schedule()
    assert batch.is_first_decode_without_local_prefill == [True]
    assert seq.is_first_decode is False


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


# ── /metrics does not count a P/D request twice ────────────────────────────


def _aggregate(*snapshots):
    """Run the real aggregator over hand-built rank snapshots.

    `get_metrics_statistics` only reads `core_mgr.latest_metrics`, so a stub
    with that attribute exercises the actual aggregation code without an
    engine, GPUs, or a running EngineCore.
    """
    from types import SimpleNamespace

    from atom.model_engine.llm_engine import LLMEngine

    engine = SimpleNamespace(
        core_mgr=SimpleNamespace(
            latest_metrics=dict(enumerate(snapshots)),
            get_dp_router_statistics=dict,
        )
    )
    return LLMEngine.get_metrics_statistics(engine)


def _snapshot(role, running, waiting, **extra):
    return {
        "enabled": True,
        "role": role,
        "requests_running": running,
        "requests_waiting": waiting,
        **extra,
    }


def test_snapshots_carry_their_pd_role():
    """The aggregation below keys off `role`, so the snapshot has to carry it.

    Half the fix lives in `collect_metrics`; without this the aggregator could
    be perfectly correct and still see `role` missing from every rank, quietly
    falling back to the summing path that double-counts.
    """
    import queue

    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.model_engine.scheduler import (
        DecodeScheduler,
        PrefillScheduler,
        Scheduler,
    )

    def role_of(sched):
        handler = EngineUtilityHandler(
            runner_mgr=None,
            output_queue=queue.Queue(),
            label="test",
            scheduler=sched,
        )
        return handler.collect_metrics()["role"]

    assert role_of(PrefillScheduler(MockConfig())) == "prefill"
    assert role_of(DecodeScheduler(MockConfig(), disagg_cu_shm_name="")) == "decode"
    # The aggregated (non-P/D) engine is neither half of a pair.
    assert role_of(Scheduler(MockConfig())) == ""


def test_pd_queue_depths_are_not_double_counted():
    """One in-flight request is held by both P/D ranks at the same instant —
    the prefill rank's `running` and the decode rank's `prefill_waiting`. The
    decode side's queues already span the whole lifetime, so summing the pair
    reports more requests than exist."""
    metrics = _aggregate(
        _snapshot("prefill", running=6, waiting=2),
        _snapshot("decode", running=3, waiting=6),
    )
    assert metrics["requests_running"] == 3
    assert metrics["requests_waiting"] == 6


def test_cumulative_counters_still_sum_across_the_pd_pair():
    """Only queue depths are deduplicated. Lifetime counters are written on
    one side only, so dropping the prefill rank from *those* would lose real
    numbers rather than remove a duplicate."""
    metrics = _aggregate(
        _snapshot("prefill", running=6, waiting=2, requests_finished=0, preemptions=1),
        _snapshot("decode", running=3, waiting=6, requests_finished=40, preemptions=2),
    )
    assert metrics["requests_finished"] == 40
    assert metrics["preemptions"] == 3


def test_queue_depths_sum_when_no_rank_is_a_decode_rank():
    """Two DP ranks of an aggregated engine. Neither is half of a pair, so
    their queues are disjoint and must add up."""
    metrics = _aggregate(
        _snapshot("", running=4, waiting=1),
        _snapshot("", running=5, waiting=2),
    )
    assert metrics["requests_running"] == 9
    assert metrics["requests_waiting"] == 3


def test_prefill_only_window_still_reports_its_queues():
    """Between the prefill rank's first metrics push and the decode rank's,
    the prefill counts are all there is — and nothing is duplicating them yet,
    so reporting 0 here would blank the dashboard instead of deduplicating."""
    metrics = _aggregate(_snapshot("prefill", running=6, waiting=2))
    assert metrics["requests_running"] == 6
    assert metrics["requests_waiting"] == 2
