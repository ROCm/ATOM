# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/scheduler.py — public API only


import logging
import time
from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from conftest import MockConfig

from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    SaveOperationId,
)
from atom.kv_transfer.offload._offload_common import OffloadSchedulerMixin
from atom.model_engine.engine_stats import EngineStats
from atom.model_engine.scheduler import (
    ScheduledBatch,
    ScheduledBatchOutput,
    Scheduler,
)
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
from atom.sampling_params import SamplingParams


class _OffloadMixinStub(OffloadSchedulerMixin):
    """Concrete `OffloadSchedulerMixin` for scheduler tests.

    `OffloadSchedulerMixin` declares the six save/load lifecycle methods abstract
    so a missing forwarder is a construction-time TypeError. These test doubles
    exercise only the scheduler's deferred-free / preemption paths, so this base
    fills the contract with harmless defaults and each local `_Connector`
    overrides the methods it drives.
    """

    is_producer = False
    is_offload = True

    def save_finished(self, req_id) -> None: ...
    def abandon_save(self, req_id) -> None: ...
    def release_stalled_save(self, seq) -> None: ...
    def load_failed(self, req_id) -> bool:
        return False

    def load_finished(self, req_id) -> bool:
        return True

    def cancel_pending_load(self, seq) -> None: ...


# ── EngineStats: spec section ────────────────────────────────────────────────


class TestSpecSection:
    def test_no_division_by_zero_with_valid_mtp_k(self):
        """EngineStats spec section with mtp_k >= 1 must not raise on update."""
        stats = EngineStats(use_spec=True, mtp_k=1)
        # Should not raise ZeroDivisionError
        stats.update_spec(num_accepted_tokens=1)
        stats.update_spec(num_accepted_tokens=2)

    def test_update_accumulates_draft_tokens(self):
        stats = EngineStats(use_spec=True, mtp_k=2)
        stats.update_spec(num_accepted_tokens=1)
        assert stats.total_draft_tokens == 2

    def test_acceptance_rate_zero_when_no_updates(self):
        stats = EngineStats(use_spec=True, mtp_k=3)
        assert stats.acceptance_rate == 0.0


# ── EngineStats: cache section ───────────────────────────────────────────────


class TestCacheSection:
    def test_update_accumulates_tokens(self):
        stats = EngineStats(enable_prefix_caching=True)
        # update_cache(cached, full, compressed, wanted, reusable)
        stats.update_cache(4, 10, 8, 6, 9)
        assert stats.total_requests == 1
        assert stats.total_cached_tokens == 4
        assert stats.total_full_tokens == 10
        assert stats.total_compressed_tokens == 8
        assert stats.total_wanted_tokens == 6
        assert stats.total_reusable_tokens == 9

    def test_hit_rate_zero_when_no_updates(self):
        stats = EngineStats(enable_prefix_caching=True)
        assert stats.cache_hit_rate == 0.0

    def test_hit_rate_is_over_reusable_not_full(self):
        """`full` includes the trailing block no cache may serve, so the rate is
        denominated in `reusable` — 4/8, not 4/10."""
        stats = EngineStats(enable_prefix_caching=True)
        stats.update_cache(4, 10, 8, 6, 8)
        assert stats.cache_hit_rate == 0.5

    def test_recent_hit_rate_tracks_the_window_not_all_history(self):
        """The reviewer's scenario: a long run whose early traffic reused a lot
        and whose later traffic reuses nothing. The lifetime figure stays high
        and barely moves; the windowed one follows the current workload."""
        stats = EngineStats(enable_prefix_caching=True, cache_hit_rate_window=100)
        for _ in range(100):  # early: 80% reuse
            stats.update_cache(80, 100, 80, 80, 100)
        assert stats.recent_cache_hit_rate == pytest.approx(0.8)

        for _ in range(100):  # later: none at all, filling the window
            stats.update_cache(0, 100, 0, 0, 100)
        assert stats.recent_cache_hit_rate == pytest.approx(0.0)
        # Lifetime still reports the average of both halves, as it should.
        assert stats.cache_hit_rate == pytest.approx(0.4)

    def test_recent_hit_rate_is_none_before_any_observation(self):
        stats = EngineStats(enable_prefix_caching=True)
        assert stats.recent_cache_hit_rate is None

    def test_window_is_bounded(self):
        """The deque must not grow with the run — it is a fixed-size window."""
        stats = EngineStats(enable_prefix_caching=True, cache_hit_rate_window=10)
        for _ in range(500):
            stats.update_cache(1, 10, 1, 1, 10)
        assert len(stats._recent_hits) == 10
        assert stats._recent_reusable_tokens == 100  # 10 requests x 10 tokens
        assert stats.total_requests == 500, "lifetime counters keep counting"

    def test_update_is_noop_when_cache_disabled(self):
        """The cache section gates internally, so a disabled EngineStats
        ignores update_cache rather than the caller having to guard it."""
        stats = EngineStats(enable_prefix_caching=False)
        stats.update_cache(4, 10, 8, 6, 9)
        assert stats.total_requests == 0
        assert stats.total_cached_tokens == 0


# ── EngineStats: throughput section ──────────────────────────────────────────


class TestThroughputSection:
    def test_update_accumulates_tokens(self):
        stats = EngineStats(enable_log_stats=True)
        stats.update_throughput(num_prompt_tokens=10, num_generation_tokens=5)
        assert stats.num_prompt_tokens == 10
        assert stats.num_generation_tokens == 5

    def test_maybe_log_below_interval_keeps_counters(self):
        """Time-based pace: below the wall-clock interval nothing is logged or
        reset, so the accumulated counts survive to the next tick."""
        stats = EngineStats(enable_log_stats=True, throughput_log_interval_s=1e6)
        stats.update_throughput(num_prompt_tokens=10, num_generation_tokens=5)
        stats.maybe_log_throughput(num_running_reqs=1, num_waiting_reqs=0, kv_usage=0.0)
        assert stats.num_prompt_tokens == 10
        assert stats.num_generation_tokens == 5

    def test_update_is_noop_when_log_stats_disabled(self):
        stats = EngineStats(enable_log_stats=False)
        stats.update_throughput(num_prompt_tokens=10, num_generation_tokens=5)
        assert stats.num_prompt_tokens == 0
        assert stats.num_generation_tokens == 0

    def test_an_all_quiet_window_closes_without_logging(self, caplog):
        """A quiet engine must not fill the log with 0.0 lines — but the
        window still has to close, because a stale start is what made the
        first line after a lull divide its tokens by the whole lull."""
        stats = EngineStats(enable_log_stats=True)
        stats._throughput_last_log_time -= 43.0
        with caplog.at_level(logging.INFO, logger="atom"):
            stats.maybe_log_throughput(
                num_running_reqs=0, num_waiting_reqs=0, kv_usage=0.0
            )
        assert "Engine" not in caplog.text, "all-quiet window must stay silent"
        # Closed anyway: the start is fresh, so the next window measures its
        # own interval rather than the 43s that preceded it.
        assert time.monotonic() - stats._throughput_last_log_time < 1.0

    def test_a_burst_then_idle_is_still_reported(self, caplog):
        """Silence is gated on zero tokens *as well as* an empty engine, so a
        window still holding a finished burst's tokens prints even though
        nothing is running by the time it closes."""
        stats = EngineStats(enable_log_stats=True)
        stats.update_throughput(num_prompt_tokens=30000)
        stats._throughput_last_log_time -= 43.0
        with caplog.at_level(logging.INFO, logger="atom"):
            stats.maybe_log_throughput(
                num_running_reqs=0, num_waiting_reqs=0, kv_usage=0.0
            )
        assert "Engine 000" in caplog.text, "a burst must not be swallowed"
        assert stats.num_prompt_tokens == 0, "reported tokens must be cleared"

    def test_running_requests_keep_the_zero_line(self, caplog):
        """Zero tokens with requests in flight means the engine is stuck —
        exactly when the 0.0 line is worth printing. Only a window with
        nothing running *and* nothing queued is suppressed."""
        stats = EngineStats(enable_log_stats=True)
        stats._throughput_last_log_time -= 43.0
        with caplog.at_level(logging.INFO, logger="atom"):
            stats.maybe_log_throughput(
                num_running_reqs=8, num_waiting_reqs=0, kv_usage=0.5
            )
        assert "Avg prompt throughput: 0.0 tokens/s" in caplog.text
        assert "Running: 8 reqs" in caplog.text

    def test_idle_does_not_leave_the_window_start_stale(self, caplog):
        """The regression this fixes: after a lull, the next active window
        must measure its own interval, not lull-plus-interval.

        Both stretches are moved on the clock rather than slept through. A
        `sleep` here would put the assertion on a wall-time budget that a GC
        pause or a contended runner can blow, turning an unrelated hiccup into
        a red build pointing at a regression that never happened.
        """
        interval = 10.0
        stats = EngineStats(enable_log_stats=True, throughput_log_interval_s=interval)
        # 43s of idleness, closed silently by the heartbeat — that close is
        # what keeps the window start fresh.
        stats._throughput_last_log_time -= 43.0
        stats.maybe_log_throughput(num_running_reqs=0, num_waiting_reqs=0, kv_usage=0.0)
        # Work arrives, then exactly one interval goes by.
        stats.update_throughput(num_prompt_tokens=7700)
        stats._throughput_last_log_time -= interval
        caplog.clear()  # only the line under test, so the parse below is exact
        with caplog.at_level(logging.INFO, logger="atom"):
            stats.maybe_log_throughput(
                num_running_reqs=1, num_waiting_reqs=0, kv_usage=0.1
            )
        assert caplog.text.count("Avg prompt throughput: ") == 1
        rate = float(caplog.text.split("Avg prompt throughput: ")[1].split(" ")[0])
        # 7700 over the 10s window it belongs to (770/s). Without the silent
        # close the window would still span the lull as well — 53s, ~145/s.
        assert 700 < rate < 800, f"window does not match its own interval: {rate}"

    def test_non_positive_interval_is_refused(self):
        """A `ValueError`, not an `assert`: `python -O` strips asserts, and
        this check is all that stands between the interval and a
        ZeroDivisionError raised inside the scheduler loop."""
        for bad in (0, -1, -0.5):
            with pytest.raises(ValueError, match="throughput_log_interval_s"):
                EngineStats(enable_log_stats=True, throughput_log_interval_s=bad)

    def test_zero_elapsed_cannot_divide(self):
        """The division is guarded at its own site too, so reaching it with a
        window that has not advanced returns instead of raising."""
        stats = EngineStats(enable_log_stats=True, throughput_log_interval_s=10.0)
        stats.update_throughput(num_prompt_tokens=100)
        # Interval tampered with after construction, window not advanced.
        stats.throughput_log_interval_s = 0.0
        stats._throughput_last_log_time = time.monotonic() + 5.0
        stats.maybe_log_throughput(num_running_reqs=1, num_waiting_reqs=0, kv_usage=0.0)
        assert stats.num_prompt_tokens == 100, "nothing should have been reported"

    def test_interval_comes_from_config(self):
        """`--throughput-log-interval` has to reach EngineStats, not just sit
        on Config — the cadence was a hard-coded keyword default before."""
        sched = Scheduler(
            MockConfig(enable_log_stats=True, throughput_log_interval=2.5)
        )
        assert sched.engine_stats.throughput_log_interval_s == 2.5

    def test_hit_rate_window_comes_from_config(self):
        """Same wiring as the interval above: `--cache-hit-rate-window` has to
        reach EngineStats rather than stop at Config."""
        sched = Scheduler(MockConfig(cache_hit_rate_window=250))
        assert sched.engine_stats._recent_window == 250

    def test_non_positive_hit_rate_window_is_rejected(self):
        """0 would evict each request as it arrives and read `n/a` forever;
        a negative window never evicts, turning the "recent" rate into a
        lifetime one. Neither should fail silently — and a bare assert would,
        under `python -O`."""
        with pytest.raises(ValueError, match="cache_hit_rate_window"):
            EngineStats(enable_prefix_caching=True, cache_hit_rate_window=0)

    def test_window_expired_gates_the_heartbeat(self):
        stats = EngineStats(enable_log_stats=True, throughput_log_interval_s=10.0)
        assert stats.window_expired(time.monotonic()) is False
        assert stats.window_expired(time.monotonic() + 11.0) is True

    def test_window_expired_is_false_when_log_stats_disabled(self):
        stats = EngineStats(enable_log_stats=False)
        assert stats.window_expired(time.monotonic() + 1e6) is False

    def test_hit_rate_is_na_until_something_is_measured(self, caplog):
        """Prefix caching enabled but nothing observed yet — "0.0%" would be a
        claim about reuse made without ever having looked. This is the shape a
        P/D decode engine is in permanently (it never reaches `update_cache`)
        and the aggregated scheduler is in until its first prefill."""
        stats = EngineStats(
            enable_log_stats=True,
            enable_prefix_caching=True,
            throughput_log_interval_s=1e-6,
        )
        with caplog.at_level(logging.INFO, logger="atom"):
            stats.update_throughput(num_generation_tokens=100)
            stats.maybe_log_throughput(
                num_running_reqs=8, num_waiting_reqs=0, kv_usage=0.5
            )
        assert "Prefix cache hit rate: n/a" in caplog.text

    def test_hit_rate_appears_once_measured(self, caplog):
        stats = EngineStats(
            enable_log_stats=True,
            enable_prefix_caching=True,
            throughput_log_interval_s=1e-6,
        )
        stats.update_cache(4, 10, 8, 6, 8)  # cached=4 of reusable=8
        with caplog.at_level(logging.INFO, logger="atom"):
            stats.update_throughput(num_prompt_tokens=10)
            stats.maybe_log_throughput(
                num_running_reqs=1, num_waiting_reqs=0, kv_usage=0.1
            )
        assert "Prefix cache hit rate: 50.0%" in caplog.text

    def test_label_defaults_to_empty_so_the_line_is_unchanged(self):
        assert EngineStats(enable_log_stats=True).label == ""

    def test_absent_kv_pool_logs_na_not_zero(self, caplog):
        """`kv_usage=None` means "this scheduler owns no KV pool" (P/D prefill),
        which must not read as a real, empty pool."""
        stats = EngineStats(
            enable_log_stats=True, label="Prefill ", throughput_log_interval_s=1e-6
        )
        with caplog.at_level(logging.INFO, logger="atom"):
            stats.update_throughput(num_prompt_tokens=100)
            stats.maybe_log_throughput(
                num_running_reqs=2, num_waiting_reqs=1, kv_usage=None
            )
        line = caplog.text
        assert "Prefill Engine 000" in line
        assert "GPU KV cache usage: n/a" in line
        # Prefix caching is off here too, so that one is n/a as well.
        assert "Prefix cache hit rate: n/a" in line


# ── schedule() closes the throughput window ────────────────────────────────


class TestScheduleTicksTheWindow:
    """`schedule()` is the single tick for the scheduling side.

    These pin the property the tick was moved there for: no return path inside
    `_schedule` can stall the 10s cadence. Written to fail under the two
    mutations that previously stayed green — dropping the prompt-token
    argument, and making the tick a no-op.
    """

    def test_prompt_tokens_reach_the_window(self, seq_factory):
        """Catches a dropped `num_prompt_tokens`: the headline number of this
        feature would otherwise read 0.0 forever with the suite still green."""
        sched = Scheduler(MockConfig(enable_log_stats=True))
        sched.add(seq_factory([1, 2, 3, 4]))
        batch, _ = sched.schedule()
        assert batch.total_tokens_num_prefill == 4
        assert sched.engine_stats.num_prompt_tokens == 4

    def test_empty_return_path_still_closes_the_window(self):
        """`_schedule` bails out early with nothing running or waiting. The
        window must still close, or an idle stretch lands in the denominator
        of the next line."""
        sched = Scheduler(MockConfig(enable_log_stats=True))
        stats = sched.engine_stats
        stats._throughput_last_log_time -= 43.0
        assert sched.schedule() is None
        assert time.monotonic() - stats._throughput_last_log_time < 1.0

    def test_decode_override_inherits_the_tick(self, seq_factory):
        """DecodeScheduler overrides `_schedule`, not `schedule()`, so it gets
        the tick for free — the three hand-placed ones it used to carry could
        all be deleted with its own test still passing."""
        from atom.model_engine.scheduler import DecodeScheduler

        sched = DecodeScheduler(
            MockConfig(enable_log_stats=True), disagg_cu_shm_name=""
        )
        stats = sched.engine_stats
        stats._throughput_last_log_time -= 43.0
        assert sched.schedule() is None  # nothing running: early return
        assert time.monotonic() - stats._throughput_last_log_time < 1.0

    def test_prefill_scheduler_reports_its_prompt_tokens(self, seq_factory):
        from atom.model_engine.scheduler import PrefillScheduler

        sched = PrefillScheduler(MockConfig(enable_log_stats=True))
        seq = seq_factory([10, 20, 30, 40])
        seq.block_table = [0, 1]
        seq.num_cached_tokens = 0
        sched.add(seq)
        sched.schedule()
        assert sched.engine_stats.num_prompt_tokens == 4


# ── add / extend / query ───────────────────────────────────────────────────


class TestSchedulerAddQuery:
    def test_is_finished_when_empty(self, scheduler):
        assert scheduler.is_finished()

    def test_add_makes_not_finished(self, scheduler, seq_factory):
        scheduler.add(seq_factory([1, 2, 3]))
        assert not scheduler.is_finished()

    def test_deferred_offload_work_keeps_scheduler_alive(self, scheduler):
        scheduler.deferred_free_blocks[17] = SimpleNamespace(id=17)

        assert not scheduler.is_finished()

    def test_extend(self, scheduler, seq_factory):
        scheduler.extend([seq_factory([1]), seq_factory([2])])
        assert scheduler.get_num_unfinished_requests() == 2

    def test_has_unfinished_requests(self, scheduler, seq_factory):
        assert not scheduler.has_unfinished_requests()
        scheduler.add(seq_factory([1]))
        assert scheduler.has_unfinished_requests()

    def test_get_request_counts(self, scheduler, seq_factory):
        scheduler.add(seq_factory([1, 2, 3, 4]))
        assert scheduler.get_request_counts() == (0, 1)
        scheduler.schedule()
        assert scheduler.get_request_counts() == (1, 0)


# ── schedule() ─────────────────────────────────────────────────────────────


class TestSchedule:
    def test_non_offload_abort_keeps_existing_receive_cleanup(self):
        seq = SimpleNamespace(
            id=96,
            status=SequenceStatus.ABORTED,
            _counted_as_inflight_load=True,
        )
        sched = Scheduler.__new__(Scheduler)
        sched._rejected = []
        sched.deferred_free_blocks = {}
        sched.finished_recving_kv_req_ids = []
        sched.failed_recving_kv_req_ids = []
        sched._num_parked_remote_kv = 1
        sched.kv_connector = SimpleNamespace(is_offload=False)

        sched._reject_aborted_waiting(seq)

        assert sched.deferred_free_blocks == {}
        assert sched._num_parked_remote_kv == 0
        assert sched._rejected == [seq]

    @pytest.mark.parametrize("first_terminal", ["load", "save"])
    def test_aborted_load_and_save_both_finish_before_release(
        self,
        first_terminal,
    ):
        load = 97
        save = 97
        seq = SimpleNamespace(
            id=97,
            status=SequenceStatus.ABORTED,
            block_table=[1],
            has_per_req_cache=False,
            _counted_as_inflight_load=True,
        )
        events = []

        class _Connector(_OffloadMixinStub):
            is_offload = True
            is_producer = False

            def __init__(self):
                self.pending_save = True

            def load_finished(self, operation):
                events.append(("load_finished", operation))
                return operation == load

            def save_finished(self, operation):
                events.append(("save_finished", operation))
                if operation == save:
                    self.pending_save = False

            def should_defer_free(self, value):
                assert value is seq
                return self.pending_save

            def request_finished(self, value):
                events.append(("request_finished", value.id))

        sched = Scheduler.__new__(Scheduler)
        sched.waiting = deque()
        sched.running = deque()
        sched._rejected = []
        sched.deferred_free_blocks = {}
        sched.finished_recving_kv_req_ids = []
        sched.failed_recving_kv_req_ids = []
        sched._num_parked_remote_kv = 1
        sched.kv_connector = _Connector()
        sched.block_manager = SimpleNamespace(
            deallocate=lambda value: events.append(("deallocate", value.id))
        )
        sched._reject_aborted_waiting(seq)

        assert sched.deferred_free_blocks == {seq.id: seq}
        assert seq._awaiting_aborted_load_cleanup is True

        outputs = {
            "load": KVConnectorOutput(finished_loading={load}),
            "save": KVConnectorOutput(finished_saving={save}),
        }
        second_terminal = "save" if first_terminal == "load" else "load"

        sched._update_from_kv_xfer_finished(outputs[first_terminal])

        assert sched.deferred_free_blocks == {seq.id: seq}
        assert not any(event[0] == "deallocate" for event in events)

        sched._update_from_kv_xfer_finished(outputs[second_terminal])

        assert sched.deferred_free_blocks == {}
        assert sched._num_parked_remote_kv == 0
        assert events.count(("request_finished", seq.id)) == 1
        assert events.count(("deallocate", seq.id)) == 1

    def test_empty_returns_none(self, scheduler):
        assert scheduler.schedule() is None

    def test_prefill(self, scheduler, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        scheduler.add(seq)
        batch, _seqs = scheduler.schedule()
        assert batch.total_seqs_num_prefill == 1
        assert batch.total_tokens_num_prefill == 4
        assert seq.status == SequenceStatus.RUNNING
        assert seq.type == SequenceType.PREFILL

    def test_prefill_respects_max_num_seqs(self, seq_factory):
        sched = Scheduler(
            MockConfig(
                max_num_seqs=2, max_num_batched_tokens=1000, num_kvcache_blocks=100
            )
        )
        for _ in range(5):
            sched.add(seq_factory([1, 2, 3, 4]))
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 2

    def test_remote_kv_decode_promotion_respects_max_num_seqs(self, seq_factory):
        """A PD consumer ready for first-decode must NOT be promoted into a full
        running queue — the bug that let decode-side running climb to 2x
        max_num_seqs and thrash (preempt -> full recompute) at KV exhaustion."""
        sched = Scheduler(
            MockConfig(
                max_num_seqs=2, max_num_batched_tokens=1000, num_kvcache_blocks=100
            )
        )
        # Fill running to the cap with two decode seqs.
        r0, r1 = seq_factory([1, 2, 3, 4]), seq_factory([5, 6, 7, 8])
        for r in (r0, r1):
            r.status = SequenceStatus.RUNNING
            r.type = SequenceType.DECODE
        sched.running = deque([r0, r1])
        # A consumer whose remote KV has arrived and is ready for first decode.
        consumer = seq_factory([9, 10, 11, 12])
        consumer.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        sched.waiting = deque([consumer])

        promoted = []
        with (
            mock.patch.object(sched, "_resolve_waiting_remote_kv", return_value=True),
            mock.patch.object(
                sched,
                "_schedule_first_decode_after_remote_kv",
                side_effect=lambda s: promoted.append(s),
            ),
        ):
            sched.schedule()

        assert promoted == []  # capped out, not promoted
        assert consumer in sched.waiting  # requeued for a later tick
        assert len(sched.running) <= sched.max_num_seqs

    def test_prefill_respects_max_batched_tokens(self, seq_factory):
        # Budgets here are multiples of the 64-token chunk alignment: a leftover
        # under one aligned unit is deliberately not scheduled at all (see
        # Scheduler._align_truncated_chunk), so a 6-token budget would pack one
        # seq, not two, and prove nothing about the budget being respected.
        sched = Scheduler(
            MockConfig(
                max_num_batched_tokens=192,
                max_model_len=1024,
                num_kvcache_blocks=100,
                enable_chunked_prefill=True,
            )
        )
        sched.add(seq_factory(list(range(128))))
        sched.add(seq_factory(list(range(200, 328))))  # only 64 fit in budget
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 2
        assert batch.total_tokens_num_prefill == 192
        assert list(batch.num_scheduled_tokens) == [128, 64]

    def test_budget_sliver_is_left_for_the_next_step(self, seq_factory):
        """Chunk alignment must not manufacture its own tail.

        Flooring seq 2's chunk to the block grid frees the remainder, and
        handing that remainder to seq 3 splits seq 3's prefill for nothing — it
        lands in the same later step either way, one forward worse off and off
        the block grid. Production saw a 16384-token budget go out as
        `..., 640, 10`.
        """
        sched = Scheduler(
            MockConfig(
                max_num_batched_tokens=200,
                max_model_len=1024,
                num_kvcache_blocks=400,
                enable_chunked_prefill=True,
            )
        )
        for start in (0, 200, 400):
            sched.add(seq_factory(list(range(start, start + 128))))
        batch, _ = sched.schedule()
        # 200 - 128 = 72 for seq 2, floored to 64; the freed 8 stays unspent.
        assert list(batch.num_scheduled_tokens) == [128, 64]
        assert batch.total_tokens_num_prefill == 192

    def test_chunked_prefill_splits_prompt_across_steps(self, seq_factory):
        sched = Scheduler(
            MockConfig(
                max_num_batched_tokens=6,
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                enable_chunked_prefill=True,
            )
        )
        seq = seq_factory(list(range(10)))
        sched.add(seq)

        batch1, _ = sched.schedule()
        assert batch1.total_tokens_num_prefill == 6
        assert list(batch1.scheduled_tokens) == list(range(6))
        assert list(batch1.num_cached_tokens) == [0]

        sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[],
                token_ids=[],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch1,
        )
        assert seq.is_partial_prefill is True
        assert seq.num_cached_tokens == 6

        batch2, _ = sched.schedule()
        assert batch2.total_tokens_num_prefill == 4
        assert list(batch2.scheduled_tokens) == list(range(6, 10))
        assert list(batch2.num_cached_tokens) == [6]

    def test_multimodal_prefill_shortened_after_alloc_is_requeued_whole(
        self, seq_factory
    ):
        # A multimodal prompt must forward in one chunk (its vision embeddings
        # are scattered onto placeholder positions for the whole prompt). The
        # pre-allocation atomic guard enforces that, but a post-allocation
        # adjuster -- offload chunk deferral or state-checkpoint alignment --
        # can shorten the chunk *after* that guard passed. When it does, the
        # prompt must be requeued whole, not split into a partial chunk that
        # would scatter the embeddings against the wrong positions.
        sched = Scheduler(
            MockConfig(
                max_num_batched_tokens=64,
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                enable_chunked_prefill=True,
            )
        )
        # Whole 8-token prompt clears the pre-alloc guard (budget 64 >= 8);
        # only the post-alloc adjuster shortens it.
        sched._adjust_prefill_chunk_after_alloc = lambda seq, chunk: 4
        seq = seq_factory(list(range(8)), multimodal_data={"pixel_values": object()})
        sched.add(seq)

        batch, _ = sched.schedule()

        # Not split into a 4-token partial chunk -- deferred whole.
        assert batch.total_seqs_num_prefill == 0
        assert seq.is_partial_prefill is False
        assert seq in sched.waiting

    def test_text_prefill_shortened_after_alloc_still_splits(self, seq_factory):
        # Control for the guard above: a non-multimodal prompt shortened by the
        # same post-alloc adjuster is chunked as usual -- the requeue is
        # multimodal-only.
        sched = Scheduler(
            MockConfig(
                max_num_batched_tokens=64,
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                enable_chunked_prefill=True,
            )
        )
        sched._adjust_prefill_chunk_after_alloc = lambda seq, chunk: 4
        seq = seq_factory(list(range(8)))
        sched.add(seq)

        batch, _ = sched.schedule()

        assert batch.total_seqs_num_prefill == 1
        assert list(batch.num_scheduled_tokens) == [4]

    def test_multimodal_prefill_spanning_checkpoint_rung_admits_whole(
        self, seq_factory
    ):
        # The requeue test above forces the shortening through the *offload*
        # adjuster, whose real-world shortening goes away once the load lands --
        # so a single pass is enough to prove the requeue. The state-checkpoint
        # rung cut in `_finalize_prefill_chunk` is different: it is deterministic,
        # so a multimodal prompt spanning one interval is shortened the *same* way
        # every pass, and requeue-whole then loops forever (idle GPUs, head-of-
        # line blocking). A one-pass test cannot see that. Drive two passes with a
        # fixed rung cut in place and assert the prompt is admitted whole and
        # prefill actually completes -- the cut must be suppressed for multimodal.
        sched = Scheduler(
            MockConfig(
                max_num_batched_tokens=64,
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                enable_chunked_prefill=True,
            )
        )
        # A rung that always lands 4 tokens short of the chunk end -- the same
        # shortening on every pass, exactly as a real ladder cuts a prompt that
        # spans an interval. If the cut were honoured for multimodal, the atomic
        # re-assert would requeue whole and this seq would never make progress.
        sched.block_manager.checkpoint_cut = lambda seq, start, end: end - 4
        seq = seq_factory(list(range(8)), multimodal_data={"pixel_values": object()})
        sched.add(seq)

        # Pass 1: admitted whole, not shortened to a 4-token partial, not requeued.
        batch1, _ = sched.schedule()
        assert batch1.total_seqs_num_prefill == 1
        assert list(batch1.num_scheduled_tokens) == [8]
        assert seq.is_partial_prefill is False
        assert seq not in sched.waiting

        sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[],
                token_ids=[],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch1,
        )

        # Pass 2: prefill is done, so the seq is not bounced back to waiting to
        # be re-shortened -- the livelock would show here as the seq reappearing
        # in `waiting` with no forward progress.
        assert seq not in sched.waiting

    def test_prefill_respects_block_availability(self, seq_factory):
        sched = Scheduler(MockConfig(num_kvcache_blocks=1, kv_cache_block_size=4))
        sched.add(seq_factory([1, 2, 3, 4]))  # 1 block
        sched.add(seq_factory([5, 6, 7, 8, 9]))  # 2 blocks → no room
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 1

    def test_decode_after_prefill(self, scheduler, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        scheduler.add(seq)
        scheduler.schedule()  # prefill
        seq.num_cached_tokens = seq.num_prompt_tokens  # simulate forward pass
        seq.append_token(5)
        batch, _ = scheduler.schedule()  # decode
        assert batch.total_seqs_num_decode == 1

    def test_decode_preemption(self, seq_factory):
        sched = Scheduler(MockConfig(num_kvcache_blocks=2, kv_cache_block_size=4))
        s1 = seq_factory([1, 2, 3, 4])
        s2 = seq_factory([5, 6, 7, 8])
        sched.add(s1)
        sched.add(s2)
        sched.schedule()  # prefill both
        s1.num_cached_tokens = s1.num_prompt_tokens  # simulate forward pass
        s2.num_cached_tokens = s2.num_prompt_tokens
        s1.append_token(9)
        s2.append_token(10)
        sched.schedule()  # one preempted
        statuses = {s1.status, s2.status}
        assert SequenceStatus.RUNNING in statuses
        assert SequenceStatus.WAITING in statuses

    def test_decode_preemption_skips_save_pinned_victim(self, seq_factory):
        sched = Scheduler(MockConfig(num_kvcache_blocks=2, kv_cache_block_size=4))
        current = seq_factory([1, 2, 3, 4])
        pinned_victim = seq_factory([5, 6, 7, 8])
        sched.add(current)
        sched.add(pinned_victim)
        sched.schedule()
        current.num_cached_tokens = current.num_prompt_tokens
        pinned_victim.num_cached_tokens = pinned_victim.num_prompt_tokens
        current.append_token(9)
        pinned_victim.append_token(10)
        operation = SaveOperationId(pinned_victim.id, 50)

        class _Connector(_OffloadMixinStub):
            is_producer = False
            is_offload = True
            _do_load = False

            def __init__(self):
                self.pending = {operation}

            def should_defer_free(self, seq):
                return seq is pinned_victim and operation in self.pending

            def save_finished(self, value):
                self.pending.discard(value)

            def build_connector_meta(self):
                return None

        sched.kv_connector = _Connector()
        pinned_blocks = list(pinned_victim.block_table)

        batch, _ = sched.schedule()

        assert current.status == SequenceStatus.WAITING
        assert pinned_victim.status == SequenceStatus.RUNNING
        assert list(pinned_victim.block_table[:1]) == pinned_blocks
        assert list(batch.req_ids) == [pinned_victim.id]

    def test_decode_preemption_stalls_pinned_current_until_save_terminal(
        self, seq_factory
    ):
        sched = Scheduler(MockConfig(num_kvcache_blocks=1, kv_cache_block_size=4))
        pinned = seq_factory([1, 2, 3, 4])
        sched.add(pinned)
        sched.schedule()
        pinned.num_cached_tokens = pinned.num_prompt_tokens
        pinned.append_token(9)
        operation = SaveOperationId(pinned.id, 51)

        class _Connector(_OffloadMixinStub):
            is_producer = False
            is_offload = True
            _do_load = False

            def __init__(self):
                self.pending = {operation}

            def should_defer_free(self, seq):
                return seq is pinned and operation in self.pending

            def save_finished(self, value):
                self.pending.discard(value)

            def build_connector_meta(self):
                return None

        sched.kv_connector = _Connector()
        pinned_blocks = list(pinned.block_table)

        blocked, _ = sched.schedule()

        assert list(blocked.req_ids) == []
        assert pinned.status == SequenceStatus.RUNNING
        assert list(pinned.block_table) == pinned_blocks
        assert sched.preempt(pinned) is False
        assert list(pinned.block_table) == pinned_blocks

        sched._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_saving={operation})
        )
        sched.schedule()

        assert pinned.status == SequenceStatus.WAITING
        assert len(pinned.block_table) == 0
        assert sched.waiting.popleft() is pinned
        sched.kv_connector = None
        replacement = seq_factory([10, 11, 12, 13])
        sched.add(replacement)
        resumed, _ = sched.schedule()
        assert list(resumed.req_ids) == [replacement.id]

    def test_ready_remote_kv_waiter_is_promoted_ahead_of_fresh_head(self):
        sched = Scheduler.__new__(Scheduler)
        fresh = SimpleNamespace(id=1, status=SequenceStatus.WAITING)
        ready = SimpleNamespace(id=2, status=SequenceStatus.WAITING_FOR_REMOTE_KVS)
        blocked = SimpleNamespace(id=3, status=SequenceStatus.WAITING_FOR_REMOTE_KVS)
        sched.waiting = deque([fresh, ready, blocked])
        sched.finished_recving_kv_req_ids = ["2"]
        sched.failed_recving_kv_req_ids = []

        sched._promote_ready_remote_kv_requests()

        assert [seq.id for seq in sched.waiting] == [2, 1, 3]

    def test_offload_parked_count_released_only_after_resume(self, seq_factory):
        sched = Scheduler(
            MockConfig(
                max_num_seqs=2,
                max_num_batched_tokens=64,
                num_kvcache_blocks=100,
            )
        )
        sched.kv_connector = SimpleNamespace(
            is_offload=True,
            build_connector_meta=lambda: None,
        )

        seq = seq_factory(list(range(8)))
        seq.num_cached_tokens = 4
        seq.block_table = [0]
        seq.offload_loaded_tokens = 4
        sched._park_for_remote_load(seq, deque())
        sched._count_inflight_load(seq)
        sched.finished_recving_kv_req_ids.append(seq.id)

        assert sched._resolve_waiting_remote_kv(seq, deque()) is False
        assert sched._num_parked_remote_kv == 1
        sched.waiting.append(seq)

        batch, _ = sched.schedule()

        assert batch.total_seqs_num_prefill == 1
        assert seq in sched.running
        assert sched._num_parked_remote_kv == 0
        assert seq._counted_as_inflight_load is False

        sched.running.clear()
        failed = seq_factory(list(range(8)))
        failed.num_cached_tokens = 4
        failed.block_table = [0]
        sched._park_for_remote_load(failed, deque())
        sched.failed_recving_kv_req_ids.append(failed.id)
        sched.waiting.append(failed)

        sched.schedule()

        assert failed.offload_load_failed is True
        assert failed in sched.running
        assert sched._num_parked_remote_kv == 0

    def test_a_resumed_offload_prefill_reports_the_hit_the_load_gave_it(
        self, seq_factory
    ):
        """`cached_tokens` must count the tokens LMCache brought back.

        The load is the entire point of parking: `_mark_offload_load_ready`
        raises `num_cached_tokens` from the pre-park HBM-only hit to the
        post-load one. But the resume branch `continue`s before either
        `prefix_cache_hit_tokens` assignment, so the field keeps the pre-park
        value while `CacheStats` is fed the fresh `num_cached_tokens` in
        `_schedule_prefill_seq`. The two then disagree about the same request,
        and the one the user sees is the one that undercounts -- making the
        offload tier look like it did nothing.
        """
        sched = Scheduler(
            MockConfig(
                max_num_seqs=2,
                max_num_batched_tokens=64,
                num_kvcache_blocks=100,
            )
        )
        sched.kv_connector = SimpleNamespace(
            is_offload=True,
            build_connector_meta=lambda: None,
        )

        seq = seq_factory(list(range(8)))
        seq.num_cached_tokens = 2  # the HBM-only hit, before the load
        seq.prefix_cache_hit_tokens = 2
        seq.block_table = [0]
        seq.offload_loaded_tokens = 6  # LMCache returned four more
        sched._park_for_remote_load(seq, deque())
        sched._count_inflight_load(seq)
        sched.finished_recving_kv_req_ids.append(seq.id)
        assert sched._resolve_waiting_remote_kv(seq, deque()) is False
        sched.waiting.append(seq)

        sched.schedule()

        assert seq in sched.running, "precondition: the resume must be admitted"
        assert seq.num_cached_tokens == 6, "precondition: the load was applied"
        assert seq.prefix_cache_hit_tokens == 6

    def test_partial_prefill_ready_for_offload_load_moves_to_waiting(self):
        class _Connector:
            def should_park_partial_prefill_for_load(self, seq):
                return seq.id == 2

        sched = Scheduler.__new__(Scheduler)
        sched.kv_connector = _Connector()
        sched.waiting = deque()
        sched._partial_prefill_count = 1
        sched._num_parked_remote_kv = 0
        keep = SimpleNamespace(
            id=1,
            status=SequenceStatus.RUNNING,
            is_partial_prefill=False,
        )
        ready = SimpleNamespace(
            id=2,
            status=SequenceStatus.RUNNING,
            is_partial_prefill=True,
        )
        sched.running = deque([keep, ready])

        sched._park_ready_offload_partial_prefills()

        assert [seq.id for seq in sched.running] == [1]
        assert [seq.id for seq in sched.waiting] == [2]
        assert ready.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
        assert ready.is_partial_prefill is False
        assert not hasattr(ready, "_discard_next_deferred_output")
        assert sched._partial_prefill_count == 0
        assert sched._num_parked_remote_kv == 1
        assert ready._counted_as_inflight_load is True

    def test_offload_partial_handoff_keeps_resumed_deferred_output(self, seq_factory):
        sched = Scheduler(
            MockConfig(
                max_num_batched_tokens=64,
                num_kvcache_blocks=10,
                kv_cache_block_size=4,
                enable_chunked_prefill=True,
            )
        )
        seq = seq_factory(list(range(10)), sampling_params=SamplingParams(max_tokens=4))
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.PREFILL
        seq.num_cached_tokens = 8
        # The pre-handoff partial-prefill postprocess already appended the
        # deferred-output placeholder before the request was parked.
        seq.append_token(sched.eos_token_id)
        sched.running = deque([seq])

        sched.postprocess(
            [seq],
            ScheduledBatchOutput(
                req_ids=[seq.id],
                token_ids=[(999,)],
                num_rejected=[0],
                num_bonus=[0],
                draft_token_ids=None,
                is_deferred_out=True,
            ),
            batch=SimpleNamespace(req_ids=[seq.id], num_scheduled_tokens=[2]),
        )

        assert seq.num_cached_tokens == 10
        assert list(seq.output_tokens) == [999, sched.eos_token_id]


# ── _waiting_new_token_count (PrefillDelayer queue signal) ─────────────────


class TestWaitingNewTokenCount:
    """The coalescer fill signal must count only ADMITTABLE waiting seqs,
    mirroring `_can_admit_head_prefill`'s skip set — otherwise remote-KV /
    unschedulable tokens inflate the aggregate and reach the fill target early."""

    def _sched(self):
        return Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=1000,
                max_model_len=64,
                enable_chunked_prefill=True,
            )
        )

    def test_counts_normal_waiting_tokens(self, seq_factory):
        sched = self._sched()
        sched.waiting = deque(
            [seq_factory(list(range(8))), seq_factory(list(range(10)))]
        )
        assert sched._waiting_new_token_count() == 18

    def test_skips_remote_kv_seqs(self, seq_factory):
        sched = self._sched()
        normal = seq_factory(list(range(8)))
        remote = seq_factory(list(range(10)))
        remote.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        sched.waiting = deque([normal, remote])
        # Only the 8 admittable tokens count; the 10 remote-KV tokens are skipped.
        assert sched._waiting_new_token_count() == 8

    def test_skips_unschedulable_oversized_seq(self, seq_factory):
        # Prompt longer than max_model_len is permanently unschedulable → skipped.
        sched = self._sched()
        normal = seq_factory(list(range(8)))
        oversized = seq_factory(list(range(200)))  # > max_model_len=64
        sched.waiting = deque([normal, oversized])
        assert sched._waiting_new_token_count() == 8

    def test_saturates_at_cap(self, seq_factory):
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=16,
                max_model_len=64,
                enable_chunked_prefill=True,
            )
        )
        sched.waiting = deque([seq_factory(list(range(10))) for _ in range(5)])
        assert sched._waiting_new_token_count() == 16  # capped, scan short-circuits


class TestPartialPrefillRemainingTokens:
    """Remaining tokens of mid-chunked-prefill seqs, folded into the coalescer
    pending signal so a small partial tail chunk batches instead of firing
    its own tiny forward. `remaining = num_tokens - num_cached_tokens`."""

    def _sched(self):
        return Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=1000,
                max_model_len=64,
                enable_chunked_prefill=True,
            )
        )

    def test_zero_when_no_partials(self, seq_factory):
        sched = self._sched()
        sched.running = deque([seq_factory(list(range(8)))])  # not partial
        assert sched._partial_prefill_remaining_tokens() == 0

    def test_sums_partial_remaining(self, seq_factory):
        sched = self._sched()
        p1 = seq_factory(list(range(20)))
        p1.is_partial_prefill = True
        p1.num_cached_tokens = 8  # 12 remaining
        p2 = seq_factory(list(range(30)))
        p2.is_partial_prefill = True
        p2.num_cached_tokens = 25  # 5 remaining
        plain = seq_factory(list(range(10)))  # not partial → excluded
        sched.running = deque([p1, p2, plain])
        sched._partial_prefill_count = 2
        assert sched._partial_prefill_remaining_tokens() == 17

    def test_saturates_at_cap(self, seq_factory):
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=1000,
                kv_cache_block_size=4,
                max_num_batched_tokens=16,
                max_model_len=4096,
                enable_chunked_prefill=True,
            )
        )
        big = seq_factory(list(range(100)))
        big.is_partial_prefill = True
        sched.running = deque([big])
        sched._partial_prefill_count = 1
        assert sched._partial_prefill_remaining_tokens() == 16  # capped


class TestOldestWaitingPrefillAge:
    """TTFT SLA guard signal: age (ms) of the oldest ADMITTABLE waiting prefill,
    skipping the same non-admittable seqs as _can_admit_head_prefill."""

    def _sched(self):
        return Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=1000,
                max_model_len=64,
                enable_chunked_prefill=True,
            )
        )

    def test_zero_when_empty(self):
        sched = self._sched()
        sched.waiting = deque()
        assert sched._oldest_waiting_prefill_age_ms() == 0.0

    def test_uses_oldest_arrival(self, seq_factory):
        sched = self._sched()
        new = seq_factory(list(range(8)))
        old = seq_factory(list(range(8)))
        sched.waiting = deque([new, old])
        with mock.patch("atom.model_engine.scheduler.time.time", return_value=1000.0):
            new.arrive_time = 999.0  # 1s ago
            old.arrive_time = 997.5  # 2.5s ago → oldest
            assert sched._oldest_waiting_prefill_age_ms() == 2500.0

    def test_skips_remote_kv(self, seq_factory):
        sched = self._sched()
        admittable = seq_factory(list(range(8)))
        remote = seq_factory(list(range(8)))
        remote.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        sched.waiting = deque([admittable, remote])
        with mock.patch("atom.model_engine.scheduler.time.time", return_value=1000.0):
            admittable.arrive_time = 999.0  # 1s
            remote.arrive_time = 990.0  # 10s but skipped (remote-KV)
            assert sched._oldest_waiting_prefill_age_ms() == 1000.0


# ── long_prefill_token_threshold ──────────────────────────────────────────


class TestLongPrefillTokenThreshold:
    """Per-request cap on prefill tokens per step (vLLM parity)."""

    def test_disabled_by_default(self, seq_factory):
        """threshold=0 → no per-request cap, only max_num_batched_tokens applies."""
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=1000,
                enable_chunked_prefill=True,
            )
        )
        sched.add(seq_factory(list(range(20))))
        batch, _ = sched.schedule()
        assert list(batch.num_scheduled_tokens) == [20]

    def test_caps_single_long_request(self, seq_factory):
        """A 20-token prompt with threshold=8 → first step does 8 tokens."""
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=1000,
                long_prefill_token_threshold=8,
                enable_chunked_prefill=True,
            )
        )
        sched.add(seq_factory(list(range(20))))
        batch, _ = sched.schedule()
        assert list(batch.num_scheduled_tokens) == [8]

    def test_short_request_unaffected(self, seq_factory):
        """Prompt shorter than threshold → full prefill in one step."""
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=1000,
                long_prefill_token_threshold=16,
                enable_chunked_prefill=True,
            )
        )
        sched.add(seq_factory([1, 2, 3, 4, 5]))
        batch, _ = sched.schedule()
        assert list(batch.num_scheduled_tokens) == [5]

    def test_applied_per_request_not_batch(self, seq_factory):
        """Two long prompts each capped at 8 → batch carries 16 tokens."""
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=1000,
                long_prefill_token_threshold=8,
                enable_chunked_prefill=True,
            )
        )
        sched.add(seq_factory(list(range(20))))
        sched.add(seq_factory(list(range(20, 40))))
        batch, _ = sched.schedule()
        assert list(batch.num_scheduled_tokens) == [8, 8]
        assert batch.total_tokens_num_prefill == 16

    def test_min_with_budget_remaining(self, seq_factory):
        """budget < threshold → chunk is bounded by budget, not threshold."""
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=400,
                kv_cache_block_size=4,
                max_model_len=1024,
                max_num_batched_tokens=192,
                long_prefill_token_threshold=128,
                enable_chunked_prefill=True,
            )
        )
        sched.add(seq_factory(list(range(320))))  # capped at the threshold, 128
        sched.add(seq_factory(list(range(400, 720))))  # budget left = 64
        batch, _ = sched.schedule()
        assert list(batch.num_scheduled_tokens) == [128, 64]

    def test_ignored_when_chunked_prefill_disabled(self, seq_factory):
        """No chunked prefill → threshold is a no-op (full prompt or reject)."""
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=1000,
                long_prefill_token_threshold=8,
                enable_chunked_prefill=False,
            )
        )
        sched.add(seq_factory(list(range(20))))
        batch, _ = sched.schedule()
        # Full 20-token prompt scheduled in one shot, threshold ignored.
        assert list(batch.num_scheduled_tokens) == [20]

    def test_partial_prefill_resume_capped(self, seq_factory):
        """Phase-1 resume of a partial-prefill seq is also capped by threshold."""
        sched = Scheduler(
            MockConfig(
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=8,  # forces chunking on the 20-tok prompt
                long_prefill_token_threshold=8,
                enable_chunked_prefill=True,
            )
        )
        seq = seq_factory(list(range(20)))
        sched.add(seq)

        # Step 1: new request, capped at 8.
        batch1, _ = sched.schedule()
        assert list(batch1.num_scheduled_tokens) == [8]
        # Simulate postprocess marking it partial (would normally happen after
        # forward returns and num_cached_tokens < num_prompt_tokens).
        seq.num_cached_tokens = 8
        seq.is_partial_prefill = True
        sched._partial_prefill_count += 1

        # Step 2: partial-prefill resume, also capped at 8 (not 12 remaining).
        batch2, _ = sched.schedule()
        assert list(batch2.num_scheduled_tokens) == [8]


# ── prefix caching ────────────────────────────────────────────────────────


class TestPrefixCaching:
    """Verify that prefix cache hits correctly reduce scheduled token counts."""

    def _make_prefix_scheduler(self):
        return Scheduler(
            MockConfig(
                enable_prefix_caching=True,
                kv_cache_block_size=4,
                num_kvcache_blocks=20,
                max_num_seqs=4,
                max_num_batched_tokens=256,
            )
        )

    def test_generated_blocks_feed_the_next_turn(self, seq_factory):
        """Multi-turn reuse: turn 2's prompt is turn 1's prompt plus its answer.

        Exercises the postprocess call site, where the committed KV length is
        the only thing separating a finalized block from one the next step may
        still rewrite.
        """
        sched = Scheduler(
            MockConfig(
                enable_prefix_caching=True,
                kv_cache_block_size=4,
                num_kvcache_blocks=40,
                max_num_seqs=4,
                max_num_batched_tokens=256,
                max_model_len=64,
            )
        )
        prompt = [1, 3, 4, 5, 6, 7, 8, 9]  # 2 whole blocks
        seq1 = seq_factory(prompt, sampling_params=SamplingParams(max_tokens=64))
        sched.add(seq1)
        batch, _ = sched.schedule()  # prefill

        generated = list(range(100, 112))  # 3 more blocks
        for token in generated:
            sched.postprocess(
                list(sched.running),
                ScheduledBatchOutput(
                    req_ids=[seq1.id],
                    token_ids=[(token,)],
                    num_rejected=None,
                    num_bonus=None,
                    draft_token_ids=None,
                ),
                batch=batch,
            )
            batch, _ = sched.schedule()  # next decode step

        assert list(seq1.token_ids) == prompt + generated
        # 20 tokens on the seq, but token 20 was sampled this step and no
        # forward has written its KV — the block it closes stays unhashed until
        # the next step consumes it.
        assert seq1.num_hashed_tokens == 16

        sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[seq1.id],
                token_ids=[(112,)],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )
        assert seq1.num_hashed_tokens == 20

        followup = seq_factory(prompt + generated)
        sched.add(followup)
        batch2, _ = sched.schedule()
        # 20 tokens, 5 blocks; the last is never reused so 16 tokens are cached
        # and only the final block's 4 tokens get forwarded.
        assert batch2.total_tokens_num_prefill == 4

    def test_deferred_output_hashes_up_to_the_committed_length(self, seq_factory):
        """The same KV line, reached from the other side of the output lag.

        Deferred output patches sampled ids one step late and appends its
        placeholder after hashing, so the committed length it hands over
        already excludes the token still in flight. Subtracting one there —
        correct for undeferred output, see the test above — would leave every
        generated block a step behind for the whole run.
        """
        sched = Scheduler(
            MockConfig(
                enable_prefix_caching=True,
                kv_cache_block_size=4,
                num_kvcache_blocks=40,
                max_num_seqs=4,
                max_num_batched_tokens=256,
                max_model_len=64,
            )
        )
        prompt = [1, 3, 4, 5, 6, 7, 8, 9]  # 2 whole blocks
        seq1 = seq_factory(prompt, sampling_params=SamplingParams(max_tokens=64))
        sched.add(seq1)
        batch, _ = sched.schedule()  # prefill

        def step(token_ids):
            nonlocal batch
            sched.postprocess(
                list(sched.running),
                ScheduledBatchOutput(
                    req_ids=[seq1.id] if token_ids else [],
                    token_ids=[token_ids] if token_ids else [],
                    num_rejected=np.zeros(1, dtype=np.int32),
                    num_bonus=np.zeros(1, dtype=np.int32),
                    draft_token_ids=None,
                    is_deferred_out=True,
                ),
                batch=batch,
            )
            batch, _ = sched.schedule()

        # The prefill step returns nothing; its sampled token surfaces next.
        step(())
        for token in range(100, 108):
            step((token,))

        # Every id that surfaced was sampled by a forward that has since run
        # again, so all eight are backed by KV: 16 tokens, 4 whole blocks. A
        # blanket subtract-one would stop at 12 and stay a block behind.
        assert seq1.num_hashed_tokens == 16

    def test_prefix_cache_reduces_token_count(self, seq_factory):
        """After a first request populates the cache, a second request sharing
        the same prefix should only schedule the non-cached tokens."""
        sched = self._make_prefix_scheduler()

        # First request: [1,2,3,4, 5,6,7,8, 9] — 3 blocks, first 2 full
        seq1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 9])
        sched.add(seq1)
        batch1, _ = sched.schedule()
        assert batch1.total_tokens_num_prefill == 9  # no cache, all tokens

        # Complete seq1 so its blocks are freed (but hashes remain).
        # `batch=batch1` is required for postprocess to call hash_blocks().
        seq1.append_token(2)  # EOS
        sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[seq1.id],
                token_ids=[(2,)],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch1,
        )

        # Second request shares the same prefix, differs in last block
        # [1,2,3,4, 5,6,7,8, 10,11] — first 2 blocks (8 tokens) should be cached
        seq2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 10, 11])
        sched.add(seq2)
        batch2, _ = sched.schedule()

        # With the fix: only 2 new tokens (10, 11) should be scheduled
        # Without the fix: all 10 tokens would be scheduled (the bug)
        assert batch2.total_tokens_num_prefill == 2
        assert batch2.num_scheduled_tokens == [2]
        assert seq2.num_cached_tokens == 8

    def test_prefix_cache_scheduled_tokens_content(self, seq_factory):
        """Verify that scheduled_tokens only contains the non-cached suffix."""
        sched = self._make_prefix_scheduler()

        seq1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 9])
        sched.add(seq1)
        batch1, _ = sched.schedule()

        seq1.append_token(2)  # EOS
        sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[seq1.id],
                token_ids=[(2,)],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch1,
        )

        seq2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 10, 11])
        sched.add(seq2)
        batch2, _ = sched.schedule()

        # scheduled_tokens should be the last num_new_tokens of token_ids
        import numpy as np

        np.testing.assert_array_equal(batch2.scheduled_tokens, [10, 11])

    def test_no_prefix_cache_full_tokens_scheduled(self, seq_factory):
        """Without prefix caching, all tokens should be scheduled."""
        sched = Scheduler(
            MockConfig(
                enable_prefix_caching=False,
                kv_cache_block_size=4,
                num_kvcache_blocks=20,
            )
        )

        seq1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 9])
        sched.add(seq1)
        sched.schedule()

        seq1.append_token(2)  # EOS
        sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[seq1.id],
                token_ids=[(2,)],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
        )

        seq2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 10, 11])
        sched.add(seq2)
        batch2, _ = sched.schedule()

        # No prefix caching → all 10 tokens are scheduled
        assert batch2.total_tokens_num_prefill == 10
        assert seq2.num_cached_tokens == 0


# ── preempt ────────────────────────────────────────────────────────────────


class TestPreempt:
    def test_preempt(self, scheduler, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        scheduler.add(seq)
        scheduler.schedule()
        scheduler.preempt(seq)
        assert seq.status == SequenceStatus.WAITING
        assert len(seq.block_table) == 0

    def test_preempt_releases_stalled_save_before_freeing_blocks(
        self, scheduler, seq_factory
    ):
        """A stall-escaped save is preemptable, and the free runs no
        `request_finished`; the connector must be told to drop its save tracker
        at the free, and told BEFORE the blocks are deallocated so its save loop
        can never race in and read them."""
        seq = seq_factory([1, 2, 3, 4])
        scheduler.add(seq)
        scheduler.schedule()
        events: list[str] = []
        block_table_at_release: list = []

        class _Connector:
            def should_defer_free(self, s):
                return False  # stall-escaped -> preemptable

            def release_stalled_save(self, s):
                events.append("release")
                block_table_at_release[:] = list(s.block_table)

        scheduler.kv_connector = _Connector()
        original_deallocate = scheduler.block_manager.deallocate

        def _recording_deallocate(s):
            events.append("deallocate")
            return original_deallocate(s)

        scheduler.block_manager.deallocate = _recording_deallocate

        assert scheduler.preempt(seq) is True
        # Released, and released first -- the blocks were still held then.
        assert events == ["release", "deallocate"]
        assert block_table_at_release  # non-empty at the moment of release
        assert seq.status == SequenceStatus.WAITING
        assert len(seq.block_table) == 0


# ── postprocess ────────────────────────────────────────────────────────────


class _DSparkSpecConfig:
    """The two fields Scheduler reads off speculative_config, DSpark-shaped."""

    def __init__(self, num_speculative_tokens: int):
        self.num_speculative_tokens = num_speculative_tokens

    def use_dspark(self) -> bool:
        return True


class TestPostprocess:
    def _prefill(self, scheduler, seq):
        scheduler.add(seq)
        scheduler.schedule()
        return seq

    def _output(self, seq_id, tokens):
        return ScheduledBatchOutput(
            req_ids=[seq_id],
            token_ids=[tuple(tokens)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        )

    def test_appends_token(self, scheduler, seq_factory):
        seq = self._prefill(scheduler, seq_factory([1, 2, 3, 4]))
        finished = scheduler.postprocess(
            list(scheduler.running), self._output(seq.id, [10])
        )
        assert 10 in seq.token_ids
        assert finished == []

    def test_generation_counted_from_committed_not_scheduled(self, seq_factory):
        """Throughput's generation count comes from postprocess (the tokens
        actually committed), not from schedule()'s scheduled draft count —
        aligning with vLLM's `len(output.new_token_ids)`."""
        sched = Scheduler(MockConfig(enable_log_stats=True))
        sp = SamplingParams(ignore_eos=True, max_tokens=100)
        seq = seq_factory([1, 2, 3, 4], sampling_params=sp)
        sched.add(seq)
        sched.schedule()
        # schedule() does not feed generation tokens anymore.
        assert sched.engine_stats.num_generation_tokens == 0
        # Two committed tokens this step → generation count bumps by exactly 2.
        sched.postprocess(list(sched.running), self._output(seq.id, [10, 11]))
        assert sched.engine_stats.num_generation_tokens == 2

    def test_generation_excludes_tokens_dropped_past_eos(self, seq_factory):
        """Counted from what the client receives, not from the forward output.

        The sampler does not inspect EOS, so on a spec-decode step it can emit
        accepted drafts after it; postprocess trims them off `new_tokens`
        before they reach RequestOutput. Counting the untrimmed output made the
        status line claim tokens nobody was sent, and disagree with the
        `total_generation_tokens` the same call derives from the trimmed length.
        """
        sched = Scheduler(MockConfig(enable_log_stats=True))
        seq = seq_factory([1, 2, 3, 4])
        sched.add(seq)
        sched.schedule()
        # EOS (2) lands second, so the trailing 99 never reaches the client.
        finished = sched.postprocess(
            list(sched.running), self._output(seq.id, [10, 2, 99])
        )
        assert len(finished) == 1 and finished[0].leave_reason == "eos"
        assert (
            sched.engine_stats.num_generation_tokens == 2
        ), "the post-EOS token must not be counted"

    @staticmethod
    def _sched_with_kv_role(kv_config):
        """Scheduler carrying a kv_transfer_config, without standing up a real
        connector (Mooncake's needs a distributed config and open ports)."""
        cfg = MockConfig(
            speculative_config=_DSparkSpecConfig(3), kv_transfer_config=kv_config
        )
        with mock.patch(
            "atom.utils.forward_context.get_kvconnector", return_value=None
        ):
            return Scheduler(cfg)

    def test_a_pd_producer_does_not_speculate(self):
        """The producer hands off after T0 and never proposes, so speculation
        must be off: with it on, postprocess pads mtp_k placeholders that hold
        num_tokens short of max_tokens and the handoff never fires."""
        for kv_config in (
            {"kv_connector": "mooncake", "kv_role": "kv_producer"},
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                ],
            },
        ):
            sched = self._sched_with_kv_role(kv_config)
            assert sched.use_spec is False
            assert sched.mtp_k == 0

    def test_a_pd_consumer_still_speculates(self):
        sched = self._sched_with_kv_role(
            {"kv_connector": "mooncake", "kv_role": "kv_consumer"}
        )
        assert sched.use_spec is True
        assert sched.mtp_k == 3

    def test_absent_drafts_leave_spec_token_ids_empty(self, seq_factory):
        """A P/D producer runs with mtp_k set but never calls propose(): it
        disables deferred output so the consumer consumes T0 once, and the
        runner only drafts on the deferred path. postprocess used to index the
        resulting None and take the prefill engine down on the first request."""
        sched = Scheduler(MockConfig(speculative_config=_DSparkSpecConfig(3)))
        assert sched.mtp_k == 3
        seq = self._prefill(sched, seq_factory([1, 2, 3, 4]))

        # What the runner emits with propose() skipped: counts are still real
        # zero-filled arrays, only the drafts are absent.
        sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[seq.id],
                token_ids=[(10,)],
                num_rejected=np.zeros(1, dtype=np.int32),
                num_bonus=np.zeros(1, dtype=np.int32),
                draft_token_ids=None,
            ),
        )

        assert list(seq.spec_token_ids) == []

    def test_eos_finishes(self, scheduler, seq_factory):
        seq = self._prefill(scheduler, seq_factory([1, 2, 3, 4]))
        finished = scheduler.postprocess(
            list(scheduler.running), self._output(seq.id, [2])
        )
        assert len(finished) == 1
        assert finished[0].leave_reason == "eos"
        assert finished[0].status == SequenceStatus.FINISHED

    def test_ignore_eos(self, scheduler, seq_factory):
        sp = SamplingParams(ignore_eos=True, max_tokens=100)
        seq = self._prefill(scheduler, seq_factory([1, 2, 3, 4], sampling_params=sp))
        finished = scheduler.postprocess(
            list(scheduler.running), self._output(seq.id, [2])
        )
        assert finished == []

    def test_max_tokens(self, scheduler, seq_factory):
        sp = SamplingParams(max_tokens=2, ignore_eos=True)
        seq = self._prefill(scheduler, seq_factory([1, 2, 3, 4], sampling_params=sp))
        scheduler.postprocess(list(scheduler.running), self._output(seq.id, [10]))
        finished = scheduler.postprocess(
            list(scheduler.running), self._output(seq.id, [11])
        )
        assert len(finished) == 1
        assert finished[0].leave_reason == "max_tokens"

    def test_stop_token_ids(self, seq_factory):
        sched = Scheduler(MockConfig(stop_token_ids=[99]))
        seq = seq_factory([1, 2, 3, 4])
        sched.add(seq)
        sched.schedule()
        finished = sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[seq.id],
                token_ids=[(99,)],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
        )
        assert len(finished) == 1
        assert "stop_99" in finished[0].leave_reason

    def test_stop_token_sequences(self, scheduler, seq_factory):
        seq = self._prefill(
            scheduler, seq_factory([1, 2, 3, 4], stop_token_sequences=[[10, 11]])
        )
        scheduler.postprocess(list(scheduler.running), self._output(seq.id, [10]))
        finished = scheduler.postprocess(
            list(scheduler.running), self._output(seq.id, [11])
        )
        assert len(finished) == 1
        assert finished[0].leave_reason == "stop_sequence"

    def test_finished_removed_from_running(self, scheduler, seq_factory):
        seq = self._prefill(scheduler, seq_factory([1, 2, 3, 4]))
        scheduler.postprocess(list(scheduler.running), self._output(seq.id, [2]))
        assert scheduler.get_request_counts() == (0, 0)


# ── get_next_batch_info ────────────────────────────────────────────────────


class TestGetNextBatchInfo:
    def test_empty(self, scheduler):
        assert scheduler.get_next_batch_info() == (False, 0, 0)

    def test_waiting(self, scheduler, seq_factory):
        scheduler.add(seq_factory([1, 2, 3, 4]))
        is_prefill, n, num_reqs = scheduler.get_next_batch_info()
        assert is_prefill is True
        assert n == 4
        assert num_reqs == 1

    def test_running(self, scheduler, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        scheduler.add(seq)
        scheduler.schedule()
        seq.num_cached_tokens = seq.num_prompt_tokens  # simulate forward pass
        is_prefill, n, num_reqs = scheduler.get_next_batch_info()
        assert is_prefill is False
        assert n == 1
        assert num_reqs == 1


# ── ScheduledBatch: PD consumer first decode primed with T0 + drafts (MTP) ──


class TestScheduledBatchPDFirstDecodeMTP:

    def test_first_decode_slices_t0_then_drafts(self):
        mtp_k = 3
        prompt_tok, t0 = 6366, 14
        drafts = [101, 102, 103]  # mtp_k transferred drafts
        seq = Sequence([prompt_tok], block_size=16)  # 1-token prompt
        seq.append_token(t0)  # injected T0
        for d in drafts:  # primed drafts
            seq.append_token(d)
        seq.type = SequenceType.DECODE
        assert seq.num_tokens == 1 + 1 + mtp_k  # prompt + T0 + drafts

        batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=[mtp_k + 1],
            total_tokens_num=mtp_k + 1,
            total_tokens_num_decode=mtp_k + 1,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            num_spec_step=mtp_k,
        )

        assert list(batch.scheduled_tokens) == [t0, *drafts]

    def test_window_starts_at_t0_when_the_producer_sent_no_drafts(self):
        """The producer does not speculate, so the handoff carries T0 alone.
        The window must still open at T0: unpadded, the trailing mtp_k+1 slice
        reaches back into the prompt and verifies prompt tokens as drafts."""
        mtp_k = 3
        prompt = [11, 22, 33, 44, 55, 66]
        t0 = 14
        sched = Scheduler(MockConfig(speculative_config=_DSparkSpecConfig(mtp_k)))
        seq = Sequence(prompt, block_size=16)
        seq.kv_transfer_params = {"first_token_id": t0}

        sched._schedule_first_decode_after_remote_kv(seq)
        seq.type = SequenceType.DECODE

        batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=[mtp_k + 1],
            total_tokens_num=mtp_k + 1,
            total_tokens_num_decode=mtp_k + 1,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            num_spec_step=mtp_k,
        )

        assert list(batch.scheduled_tokens)[0] == t0
        assert not set(batch.scheduled_tokens) & set(prompt)

    def test_normal_decode_window_unchanged(self):
        """offset >= 0 path is byte-for-byte the trailing mtp_k+1 slice."""
        mtp_k = 3
        toks = list(range(100, 110))  # 10 tokens, ample context
        seq = Sequence(toks[:6], block_size=16)
        for t in toks[6:]:
            seq.append_token(t)
        seq.type = SequenceType.DECODE

        batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=[mtp_k + 1],
            total_tokens_num=mtp_k + 1,
            total_tokens_num_decode=mtp_k + 1,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            num_spec_step=mtp_k,
        )

        assert list(batch.scheduled_tokens) == toks[-(mtp_k + 1) :]


# ── detailed annotation aggregates ──────────────────────────────────────────


class TestComputeDetailedAggregates:
    """Unit tests for Scheduler.compute_detailed_aggregates (pure Python).

    The method only touches ``self.profile_active`` and the cached
    ``self._detailed_annotation_enabled`` flag, so a lightweight
    SimpleNamespace stands in for both the scheduler and the sequences — no
    GPU or full Scheduler construction required.
    """

    @staticmethod
    def _make_batch(num_scheduled_tokens):
        return SimpleNamespace(
            num_scheduled_tokens=num_scheduled_tokens,
            detailed_sqsq=None,
            detailed_sqsk=None,
            detailed_sk=None,
        )

    @staticmethod
    def _make_seqs():
        # Two prefill requests + one decode request.
        #   prefill A: N_Q=4, cached=2 -> N_KV=6  -> sqsq 16, sqsk 24, sk 6
        #   prefill B: N_Q=3, cached=0 -> N_KV=3  -> sqsq  9, sqsk  9, sk 3
        #   decode  C: N_Q=1,          -> N_KV=10 -> sqsq  1, sqsk 10, sk 10
        return {
            0: SimpleNamespace(
                type=SequenceType.PREFILL, num_tokens=6, num_cached_tokens=2
            ),
            1: SimpleNamespace(
                type=SequenceType.PREFILL, num_tokens=3, num_cached_tokens=0
            ),
            2: SimpleNamespace(
                type=SequenceType.DECODE, num_tokens=10, num_cached_tokens=9
            ),
        }

    def test_aggregates_when_enabled(self):
        fake_self = SimpleNamespace(
            profile_active=True, _detailed_annotation_enabled=True
        )
        batch = self._make_batch([4, 3, 1])

        Scheduler.compute_detailed_aggregates(fake_self, batch, self._make_seqs())

        assert batch.detailed_sqsq == 16 + 9 + 1
        assert batch.detailed_sqsk == 24 + 9 + 10
        assert batch.detailed_sk == 6 + 3 + 10

    def test_noop_when_flag_disabled(self):
        fake_self = SimpleNamespace(
            profile_active=True, _detailed_annotation_enabled=False
        )
        batch = self._make_batch([4, 3, 1])

        Scheduler.compute_detailed_aggregates(fake_self, batch, self._make_seqs())

        assert batch.detailed_sqsq is None
        assert batch.detailed_sqsk is None
        assert batch.detailed_sk is None

    def test_noop_when_profiling_inactive(self):
        fake_self = SimpleNamespace(
            profile_active=False, _detailed_annotation_enabled=True
        )
        batch = self._make_batch([4, 3, 1])

        Scheduler.compute_detailed_aggregates(fake_self, batch, self._make_seqs())

        assert batch.detailed_sqsq is None
        assert batch.detailed_sqsk is None
        assert batch.detailed_sk is None

    def test_no_int32_overflow_large_prefill(self):
        # Regression: num_scheduled_tokens is np.int32, so nq*nq must not
        # overflow for long prefills. np.int32(65536)**2 wraps to 0, which
        # would silently corrupt the estimate the feature exists to produce.
        fake_self = SimpleNamespace(
            profile_active=True, _detailed_annotation_enabled=True
        )
        nq = 65536
        batch = self._make_batch(np.asarray([nq], dtype=np.int32))
        seqs = {
            0: SimpleNamespace(
                type=SequenceType.PREFILL, num_tokens=nq, num_cached_tokens=0
            )
        }

        Scheduler.compute_detailed_aggregates(fake_self, batch, seqs)

        assert batch.detailed_sqsq == nq * nq  # 4294967296, not 0
        assert batch.detailed_sqsk == nq * nq
        assert batch.detailed_sk == nq
        assert isinstance(batch.detailed_sqsq, int)

    def test_decode_counts_scheduled_query_tokens(self):
        # MTP/spec-decode schedules mtp_k+1 query tokens; nq must reflect the
        # scheduled count rather than a hardcoded 1 (otherwise undercounted).
        fake_self = SimpleNamespace(
            profile_active=True, _detailed_annotation_enabled=True
        )
        batch = self._make_batch(np.asarray([3], dtype=np.int32))
        seqs = {
            0: SimpleNamespace(
                type=SequenceType.DECODE, num_tokens=100, num_cached_tokens=97
            )
        }

        Scheduler.compute_detailed_aggregates(fake_self, batch, seqs)

        assert batch.detailed_sqsq == 9  # 3^2
        assert batch.detailed_sqsk == 300  # 3 * 100
        assert batch.detailed_sk == 100


class TestStalledOffloadSaveReclaim:
    """`_reconcile_stalled_deferred_saves`: the way out for a save nobody answers.

    LMCache's pin monitor force-unpins a stalled transfer without emitting a
    completion, so `should_defer_free` stays True forever, `has_pending_kv_work()`
    never clears, and the engine busy-loops with every GPU idle. Reproduced on
    the k3-dev line as a hard hang under a tight pool.
    """

    @staticmethod
    def _sched(monkeypatch, deferred, connector=None):
        import atom.model_engine.scheduler as sched_mod

        s = object.__new__(sched_mod.Scheduler)
        s.deferred_free_blocks = {seq.id: seq for seq in deferred}
        s._abandoned_saves = 0
        s._next_save_reconcile_at = 0.0
        # The window is now sourced from the connector, not a scheduler constant:
        # the scheduler asks `kv_connector.save_abandon_timeout_s()` for it (the
        # value is LMCache knowledge). Tests choose their own via the stub.
        if connector is None:
            connector = SimpleNamespace(save_abandon_timeout_s=lambda: 100.0)
        s.kv_connector = connector
        freed: list[int] = []
        s.block_manager = SimpleNamespace(deallocate=lambda q: freed.append(q.id))
        return s, freed

    def test_a_save_past_the_window_gets_its_blocks_back(self, monkeypatch):
        import time as _time

        now = _time.monotonic()
        stale = SimpleNamespace(id=1, _deferred_save_at=now - 500.0)
        fresh = SimpleNamespace(id=2, _deferred_save_at=now)
        s, freed = self._sched(monkeypatch, [stale, fresh])

        assert s._reconcile_stalled_deferred_saves() == 1
        assert freed == [1], "only the stalled save is reclaimed"
        assert 2 in s.deferred_free_blocks, "a save still inside its window is kept"
        assert s._abandoned_saves == 1

    def test_reclaim_notifies_the_connector_to_drop_the_save(self, monkeypatch):
        """Freeing blocks is not enough: without `abandon_save` the connector's
        `_save_inflight` keeps the request, `has_pending_kv_work()` never clears,
        and the engine busy-loops (review finding #4)."""
        import time as _time

        now = _time.monotonic()
        stale = SimpleNamespace(id=1, _deferred_save_at=now - 500.0)
        abandoned: list = []
        connector = SimpleNamespace(
            save_abandon_timeout_s=lambda: 100.0,
            abandon_save=lambda sid: abandoned.append(sid),
        )
        s, freed = self._sched(monkeypatch, [stale], connector=connector)

        assert s._reconcile_stalled_deferred_saves() == 1
        assert freed == [1]
        # Notified with the string request id, matching the connector's sid keys.
        assert abandoned == ["1"]

    def test_it_self_throttles_so_a_1ms_poll_is_cheap(self, monkeypatch):
        import time as _time

        stale = SimpleNamespace(id=1, _deferred_save_at=_time.monotonic() - 500.0)
        s, freed = self._sched(monkeypatch, [stale])

        assert s._reconcile_stalled_deferred_saves() == 1
        # Second call inside the throttle interval must not rescan.
        s.deferred_free_blocks = {
            9: SimpleNamespace(id=9, _deferred_save_at=_time.monotonic() - 500.0)
        }
        assert s._reconcile_stalled_deferred_saves() == 0
        assert freed == [1]

    def test_a_non_positive_window_restores_wait_forever(self, monkeypatch):
        import time as _time

        import atom.model_engine.scheduler as sched_mod

        s = object.__new__(sched_mod.Scheduler)
        s.deferred_free_blocks = {
            1: SimpleNamespace(id=1, _deferred_save_at=_time.monotonic() - 1e6)
        }
        s._abandoned_saves = 0
        s._next_save_reconcile_at = 0.0
        # A connector reporting a non-positive window disables reclamation.
        s.kv_connector = SimpleNamespace(save_abandon_timeout_s=lambda: 0.0)
        s.block_manager = SimpleNamespace(deallocate=lambda q: pytest.fail("reclaimed"))

        assert s._reconcile_stalled_deferred_saves() == 0

    def test_the_window_sits_above_lmcaches_own_pin_timeout(self, monkeypatch):
        """Deriving it from LMCache's knob IS the safety argument.

        Two independent env vars would let ours be set below the timeout it has
        to exceed, and nothing would say so. The derivation now lives on the
        offload connector (`offload_save_abandon_timeout_s`), since the pin
        timeout is LMCache knowledge; the scheduler only asks for the result.
        """
        import atom.kv_transfer.offload._offload_common as offload_common

        monkeypatch.setattr(offload_common, "_save_abandon_timeout_s", None)
        monkeypatch.setenv("LMCACHE_EC_PIN_TIMEOUT_SEC", "900")
        assert offload_common.offload_save_abandon_timeout_s() == 930.0

        monkeypatch.setattr(offload_common, "_save_abandon_timeout_s", None)
        monkeypatch.delenv("LMCACHE_EC_PIN_TIMEOUT_SEC", raising=False)
        assert offload_common.offload_save_abandon_timeout_s() == 330.0

    def test_no_offload_connector_disables_reclamation(self):
        """`_save_abandon_timeout_s` returns 0 when nothing offloads.

        A non-offload connector (or none at all) has no save to reclaim, so the
        scheduler must read a non-positive window and skip the scan rather than
        raise reaching for a method that is not there.
        """
        import atom.model_engine.scheduler as sched_mod

        s = object.__new__(sched_mod.Scheduler)
        s.kv_connector = None
        assert s._save_abandon_timeout_s() == 0.0
        s.kv_connector = SimpleNamespace()  # a connector without the offload face
        assert s._save_abandon_timeout_s() == 0.0


class TestStateStorePendingCap:
    """`_state_store_pending_cap`: the state leg reads the KV leg's save bound.

    It must share the connector's real `max_pending_saves` so both legs pin the
    same slice of the pool -- and read it off the *public* accessor, never by
    reaching through the delegating shell's `_impl` (review finding §2b).
    """

    @staticmethod
    def _sched(connector):
        import atom.model_engine.scheduler as sched_mod

        s = object.__new__(sched_mod.Scheduler)
        s.kv_connector = connector
        return s

    def test_reads_the_public_bound_off_the_connector(self):
        s = self._sched(SimpleNamespace(max_pending_saves=5))
        assert s._state_store_pending_cap() == 5

    def test_falls_back_to_env_when_the_connector_does_not_bound(self, monkeypatch):
        """dense reports None (its save queue is unbounded); use the env reader."""
        import atom.model_engine.scheduler as sched_mod

        monkeypatch.setattr(sched_mod, "_MAX_PENDING_OFFLOAD", None)
        monkeypatch.setenv("OFFLOAD_MAX_PENDING_SAVES", "3")
        s = self._sched(SimpleNamespace(max_pending_saves=None))
        assert s._state_store_pending_cap() == 3

    def test_under_multi_reaches_the_bound_on_the_state_tier_sub(self):
        """The composite bounds nothing of its own; the sub carries the bound."""
        sub = SimpleNamespace(max_pending_saves=7)
        conn = SimpleNamespace(max_pending_saves=None, _state_tier_sub=lambda: sub)
        s = self._sched(conn)
        assert s._state_store_pending_cap() == 7


class TestTheTierSplitPartitionsServedReuse:
    """`[Cache Tiers]` exists to answer "what does the CPU tier buy", so its two
    halves have to be two halves of one thing.

    `cached` and `offload` are that: `cached` is what the HBM walk actually
    handed over, `offload` is what the tier added on top, and they sum to
    `num_cached`. `compressed` is NOT -- it is how far the walk reached before
    the state gates cut it, so it counts reuse nobody got.
    """

    @staticmethod
    def stats(**kw):
        from atom.model_engine.engine_stats import EngineStats

        s = EngineStats(enable_prefix_caching=True)
        s.update_cache(**kw)
        return s

    def test_the_two_halves_sum_to_the_end_to_end_rate(self):
        s = self.stats(
            num_cached_tokens=300,
            num_full_tokens=1200,
            num_compressed_tokens=800,
            num_wanted_tokens=300,
            num_reusable_tokens=1000,
            num_offload_tokens=700,
        )
        assert s.cache_hit_rate + s.lmcache_hit_rate == pytest.approx(1.0)

    def test_the_halves_never_exceed_the_denominator(self):
        """K3's ordinary anchor-only shape: the walk reaches 8 blocks, the only
        resumable rung is at 3, the joint boundary lands at 10. Pairing
        `compressed` against `offload` prints 80% + 70% here -- 150% of a
        denominator that is the ceiling."""
        s = self.stats(
            num_cached_tokens=300,  # the gate cut the walk from 800 to 300
            num_full_tokens=1200,
            num_compressed_tokens=800,
            num_wanted_tokens=300,
            num_reusable_tokens=1000,
            num_offload_tokens=700,
        )
        assert s.paged_hit_rate + s.lmcache_hit_rate > 1.0, (
            "precondition: this is the shape that makes the wrong pairing "
            "exceed 100%, so the assertion below is not vacuous"
        )
        assert s.cache_hit_rate + s.lmcache_hit_rate <= 1.0

    def test_the_line_reports_cached_not_compressed(self, caplog):
        """Reads the emitted text: swapping `hit_rate` back for
        `paged_hit_rate` has to be what fails here."""
        import logging

        s = self.stats(
            num_cached_tokens=300,
            num_full_tokens=1200,
            num_compressed_tokens=800,
            num_wanted_tokens=300,
            num_reusable_tokens=1000,
            num_offload_tokens=700,
        )
        with caplog.at_level(logging.INFO, logger="atom"):
            s._log_pools()
        line = next(
            r.getMessage() for r in caplog.records if "[Cache Tiers]" in r.getMessage()
        )
        assert "300/1000" in line, f"HBM half must be `cached`, got: {line}"
        assert "800/1000" not in line, f"`compressed` is reach, not served: {line}"
        assert "700/1000" in line

    def test_no_tier_attached_emits_no_tier_line(self, caplog):
        import logging

        s = self.stats(
            num_cached_tokens=300,
            num_full_tokens=1200,
            num_compressed_tokens=800,
            num_wanted_tokens=300,
            num_reusable_tokens=1000,
        )
        with caplog.at_level(logging.INFO, logger="atom"):
            s._log_pools()
        assert not any("[Cache Tiers]" in r.getMessage() for r in caplog.records)
