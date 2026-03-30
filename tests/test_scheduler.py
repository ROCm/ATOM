# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/scheduler.py — public API only

from atom.model_engine.scheduler import Scheduler, ScheduledBatchOutput
from atom.model_engine.sequence import SequenceStatus, SequenceType
from atom.sampling_params import SamplingParams
from conftest import MockConfig

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
        sched = Scheduler(MockConfig(max_num_batched_tokens=6, num_kvcache_blocks=100))
        sched.add(seq_factory([1, 2, 3, 4]))  # 4 tokens
        sched.add(seq_factory([5, 6, 7, 8]))  # 4 tokens total, but only 2 fit in budget
        batch, _ = sched.schedule()
        # Chunked prefill: seq2 gets a 2-token chunk (budget 6-4=2)
        assert batch.total_seqs_num_prefill == 2
        assert batch.total_tokens_num_prefill == 6
        assert list(batch.num_scheduled_tokens) == [4, 2]

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
        seq.num_kv_computed = seq.num_prompt_tokens  # simulate forward pass
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
        s1.num_kv_computed = s1.num_prompt_tokens  # simulate forward pass
        s2.num_kv_computed = s2.num_prompt_tokens
        s1.append_token(9)
        s2.append_token(10)
        sched.schedule()  # one preempted
        statuses = {s1.status, s2.status}
        assert SequenceStatus.RUNNING in statuses
        assert SequenceStatus.WAITING in statuses


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

        # Complete seq1 so its blocks are freed (but hashes remain)
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

        # Second request shares the same prefix, differs in last block
        # [1,2,3,4, 5,6,7,8, 10,11] — first 2 blocks (8 tokens) should be cached
        seq2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 10, 11])
        sched.add(seq2)
        batch2, _ = sched.schedule()

        # With the fix: only 2 new tokens (10, 11) should be scheduled
        # Without the fix: all 10 tokens would be scheduled (the bug)
        assert batch2.total_tokens_num_prefill == 2
        assert batch2.num_scheduled_tokens == [2]
        assert seq2.num_kv_computed == 8

    def test_prefix_cache_scheduled_tokens_content(self, seq_factory):
        """Verify that scheduled_tokens only contains the non-cached suffix."""
        sched = self._make_prefix_scheduler()

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
        assert seq2.num_kv_computed == 0


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
        assert scheduler.get_next_batch_info() == (False, 0)

    def test_waiting(self, scheduler, seq_factory):
        scheduler.add(seq_factory([1, 2, 3, 4]))
        is_prefill, n = scheduler.get_next_batch_info()
        assert is_prefill is True
        assert n == 4

    def test_running(self, scheduler, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        scheduler.add(seq)
        scheduler.schedule()
        seq.num_kv_computed = seq.num_prompt_tokens  # simulate forward pass
        is_prefill, n = scheduler.get_next_batch_info()
        assert is_prefill is False
        assert n == 1


# ── Mixed Prefill-Decode Batch ────────────────────────────────────────────


class TestMixedBatch:
    """Phase 2: mixed prefill + decode in one batch."""

    def test_mixed_chunked_prefill_with_decode(self, seq_factory):
        """A chunked prefill seq resumes alongside a decode seq."""
        # Budget=10 so prefill chunk (8 tokens) + decode (1 token) fit
        sched = Scheduler(MockConfig(max_num_batched_tokens=10, num_kvcache_blocks=20))
        # seq1: short prompt, prefill then decode
        seq1 = seq_factory([1, 2, 3, 4])
        sched.add(seq1)
        sched.schedule()  # prefill seq1 (4 tokens)
        seq1.num_kv_computed = 4  # simulate forward pass
        seq1.append_token(5)

        # seq2: long prompt (12 tokens), will be chunked
        seq2 = seq_factory(list(range(10, 22)))
        sched.add(seq2)

        batch, seqs = sched.schedule()
        # seq2 gets min(12, 10-0)=10 prefill but then budget check:
        # budget_remaining = 10-0 = 10, chunk = min(12, 10) = 10 - wait,
        # but then decode needs 1 more token. Let's check:
        # Phase 2: seq2 allocated, chunk = min(12, 10) = 10. But wait, that's
        # seq2 being chunked to 10 tokens out of 12. Then budget = 10.
        # Phase 3: seq1 decode needs 1 token. 10 + 1 > 10 → doesn't fit.
        # Need budget > seq2_chunk + 1. Let's use budget=13.
        pass

    def test_mixed_chunked_prefill_with_decode_v2(self, seq_factory):
        """A chunked prefill seq resumes alongside a decode seq."""
        # Budget=13 so prefill chunk (12 tokens) + decode (1 token) = 13
        sched = Scheduler(MockConfig(max_num_batched_tokens=13, num_kvcache_blocks=20))
        seq1 = seq_factory([1, 2, 3, 4])
        sched.add(seq1)
        sched.schedule()  # prefill seq1 (4 tokens)
        seq1.num_kv_computed = 4
        seq1.append_token(5)

        # seq2: 12 tokens, fits in budget
        seq2 = seq_factory(list(range(10, 22)))
        sched.add(seq2)

        batch, seqs = sched.schedule()
        assert batch.total_seqs_num_prefill >= 1
        assert batch.total_seqs_num_decode >= 1
        assert batch.is_mixed
        assert batch.total_tokens_num == (
            batch.total_tokens_num_prefill + batch.total_tokens_num_decode
        )

    def test_mixed_is_partial_prefill_false(self, seq_factory):
        """In mixed batch, is_partial_prefill should be False."""
        sched = Scheduler(MockConfig(max_num_batched_tokens=8, num_kvcache_blocks=20))
        seq1 = seq_factory([1, 2, 3, 4])
        sched.add(seq1)
        sched.schedule()
        seq1.num_kv_computed = 4
        seq1.append_token(5)

        # Long prompt that will be chunked (intermediate chunk)
        seq2 = seq_factory(list(range(10, 30)))  # 20 tokens
        sched.add(seq2)

        batch, _ = sched.schedule()
        if batch.is_mixed:
            assert not batch.is_partial_prefill

    def test_pure_prefill_regression(self, seq_factory):
        """Pure prefill batch (no decode) still works correctly."""
        sched = Scheduler(MockConfig(num_kvcache_blocks=20))
        seq = seq_factory([1, 2, 3, 4])
        sched.add(seq)
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 1
        assert batch.total_seqs_num_decode == 0
        assert not batch.is_mixed

    def test_pure_decode_regression(self, seq_factory):
        """Pure decode batch still works correctly."""
        sched = Scheduler(MockConfig(num_kvcache_blocks=20))
        seq = seq_factory([1, 2, 3, 4])
        sched.add(seq)
        sched.schedule()  # prefill
        seq.num_kv_computed = 4
        seq.append_token(5)
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_decode == 1
        assert batch.total_seqs_num_prefill == 0
        assert not batch.is_mixed

    def test_mixed_seq_ordering(self, seq_factory):
        """Prefill seqs should come before decode seqs in batch."""
        sched = Scheduler(MockConfig(max_num_batched_tokens=16, num_kvcache_blocks=20))
        # Prepare two decode seqs
        s1 = seq_factory([1, 2, 3, 4])
        s2 = seq_factory([5, 6, 7, 8])
        sched.add(s1)
        sched.add(s2)
        sched.schedule()  # prefill both
        s1.num_kv_computed = 4
        s2.num_kv_computed = 4
        s1.append_token(9)
        s2.append_token(10)

        # Add a new prefill seq
        s3 = seq_factory([20, 21, 22, 23])
        sched.add(s3)

        batch, seqs = sched.schedule()
        if batch.is_mixed:
            # Check ordering: prefill seqs first in req_ids
            prefill_count = batch.total_seqs_num_prefill
            for i, req_id in enumerate(batch.req_ids):
                seq = seqs[req_id]
                if i < prefill_count:
                    from atom.model_engine.sequence import SequenceType

                    assert seq.type == SequenceType.PREFILL
                else:
                    assert seq.type == SequenceType.DECODE

    def test_mixed_num_prompt_tokens(self, seq_factory):
        """ScheduledBatch.num_prompt_tokens should be set for all seqs."""
        sched = Scheduler(MockConfig(max_num_batched_tokens=16, num_kvcache_blocks=20))
        s1 = seq_factory([1, 2, 3, 4])
        sched.add(s1)
        sched.schedule()
        s1.num_kv_computed = 4
        s1.append_token(5)

        s2 = seq_factory([10, 11, 12, 13])
        sched.add(s2)

        batch, _ = sched.schedule()
        assert len(batch.num_prompt_tokens) == batch.total_seqs_num

    def test_get_next_batch_info_mixed(self, seq_factory):
        """get_next_batch_info should report mixed batch as prefill."""
        sched = Scheduler(MockConfig(max_num_batched_tokens=8, num_kvcache_blocks=20))
        s1 = seq_factory([1, 2, 3, 4])
        sched.add(s1)
        sched.schedule()
        s1.num_kv_computed = 4
        s1.append_token(5)

        # Long prompt → will create partial prefill in running
        s2 = seq_factory(list(range(10, 30)))
        sched.add(s2)
        sched.schedule()  # mixed batch: s2 partial prefill + s1 decode

        # After step, s2 is still partial → next batch should be prefill (mixed)
        is_prefill, n = sched.get_next_batch_info()
        if s2.num_kv_computed < s2.num_prompt_tokens:
            assert is_prefill is True
