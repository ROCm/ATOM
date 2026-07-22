# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/scheduler.py — public API only


from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np

from atom.model_engine.scheduler import (
    ScheduledBatch,
    Scheduler,
    ScheduledBatchOutput,
    SpecStats,
)
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
from atom.sampling_params import SamplingParams
from conftest import MockConfig

# ── SpecStats ──────────────────────────────────────────────────────────────


class TestSpecStats:
    def test_no_division_by_zero_with_valid_mtp_k(self):
        """SpecStats with mtp_k >= 1 must not raise on update()."""
        stats = SpecStats(mtp_k=1)
        # Should not raise ZeroDivisionError
        stats.update(num_accepted_tokens=1)
        stats.update(num_accepted_tokens=2)

    def test_update_accumulates_draft_tokens(self):
        stats = SpecStats(mtp_k=2)
        stats.update(num_accepted_tokens=1)
        assert stats.total_draft_tokens == 2

    def test_acceptance_rate_zero_when_no_updates(self):
        stats = SpecStats(mtp_k=3)
        assert stats.acceptance_rate == 0.0


# ── add / extend / query ───────────────────────────────────────────────────


class TestSchedulerAddQuery:
    def test_is_finished_when_empty(self, scheduler):
        assert scheduler.is_finished()

    def test_add_makes_not_finished(self, scheduler, seq_factory):
        scheduler.add(seq_factory([1, 2, 3]))
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
    def test_empty_returns_none(self, scheduler):
        assert scheduler.schedule() is None

    def test_prefill(self, scheduler, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        scheduler.add(seq)
        batch, seqs = scheduler.schedule()
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

    def test_prefill_respects_max_batched_tokens(self, seq_factory):
        sched = Scheduler(
            MockConfig(
                max_num_batched_tokens=6,
                num_kvcache_blocks=100,
                enable_chunked_prefill=True,
            )
        )
        sched.add(seq_factory([1, 2, 3, 4]))  # 4 tokens
        sched.add(seq_factory([5, 6, 7, 8]))  # 4 tokens total, but only 2 fit in budget
        batch, _ = sched.schedule()
        # Chunked prefill: seq2 gets a 2-token chunk (budget 6-4=2)
        assert batch.total_seqs_num_prefill == 2
        assert batch.total_tokens_num_prefill == 6
        assert list(batch.num_scheduled_tokens) == [4, 2]

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

    def test_partial_prefill_ready_for_offload_load_moves_to_waiting(self):
        class _Connector:
            def should_park_partial_prefill_for_load(self, seq):
                return seq.id == 2

        sched = Scheduler.__new__(Scheduler)
        sched.kv_connector = _Connector()
        sched.waiting = deque()
        sched._partial_prefill_count = 1
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
        assert ready._discard_next_deferred_output is True
        assert sched._partial_prefill_count == 0

    def test_offload_partial_handoff_discards_stale_deferred_output(self, seq_factory):
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
        seq._discard_next_deferred_output = True
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
        assert seq._discard_next_deferred_output is False
        assert 999 not in seq.output_tokens
        assert seq.output_tokens == [sched.eos_token_id]


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
                num_kvcache_blocks=100,
                kv_cache_block_size=4,
                max_num_batched_tokens=10,
                long_prefill_token_threshold=8,
                enable_chunked_prefill=True,
            )
        )
        sched.add(seq_factory(list(range(20))))  # capped at 8
        sched.add(seq_factory(list(range(20, 40))))  # budget left = 2
        batch, _ = sched.schedule()
        assert list(batch.num_scheduled_tokens) == [8, 2]

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
        assert seq.block_table == []


# ── postprocess ────────────────────────────────────────────────────────────


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
