# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for PrefillScheduler — no GPU required.

NOTE: All atom.* imports are deferred to inside fixtures/tests.
test_mxfp4_moe_has_bias.py purges sys.modules["atom.*"] at collection time;
top-level imports here would create a second copy of atom.model_engine.sequence
and break SequenceStatus enum identity checks in test_scheduler.py.
"""

import pytest
from conftest import MockConfig


@pytest.fixture
def prefill_scheduler():
    from atom.model_engine.scheduler import PrefillScheduler

    return PrefillScheduler(MockConfig())


@pytest.fixture
def seq_factory():
    from atom.sampling_params import SamplingParams
    from atom.model_engine.sequence import Sequence

    def make(token_ids, block_size=4):
        return Sequence(token_ids, block_size, sampling_params=SamplingParams())

    return make


# ── is_finished / has_requests ────────────────────────────────────────────


def test_is_finished_when_empty(prefill_scheduler):
    assert prefill_scheduler.is_finished()


def test_not_finished_after_add(prefill_scheduler, seq_factory):
    prefill_scheduler.add(seq_factory([1, 2, 3]))
    assert not prefill_scheduler.is_finished()


def test_has_requests_false_when_empty(prefill_scheduler):
    assert not prefill_scheduler.has_requests()


def test_has_requests_true_after_add(prefill_scheduler, seq_factory):
    prefill_scheduler.add(seq_factory([1, 2]))
    assert prefill_scheduler.has_requests()


# ── schedule() ────────────────────────────────────────────────────────────


def test_schedule_returns_none_when_no_block_table(prefill_scheduler, seq_factory):
    """Sequences without a block_table (not yet assigned by decode) are not scheduled."""
    seq = seq_factory([1, 2, 3, 4])
    assert seq.block_table == []
    prefill_scheduler.add(seq)

    result = prefill_scheduler.schedule()
    assert result == (None, {}), "Expected (None, {}) when no block_table assigned"
    assert seq in prefill_scheduler.waiting, "Seq should remain in waiting"


def test_schedule_runs_seq_once_block_table_populated(prefill_scheduler, seq_factory):
    """schedule() picks up a sequence as soon as its block_table is non-empty."""
    seq = seq_factory([10, 20, 30, 40])
    prefill_scheduler.add(seq)

    # Simulate DecodeEngineCore assigning blocks.
    seq.block_table = [0, 1]
    seq.num_cached_tokens = 0

    batch, seqs = prefill_scheduler.schedule()

    assert batch is not None
    assert batch.total_seqs_num_prefill == 1
    assert batch.total_tokens_num_prefill == 4
    assert seq.id in seqs
    from atom.model_engine.sequence import SequenceStatus, SequenceType

    assert seq.status == SequenceStatus.RUNNING
    assert seq.type == SequenceType.PREFILL
    assert seq not in prefill_scheduler.waiting
    assert seq in prefill_scheduler.running


def test_schedule_skips_seqs_without_block_table_but_runs_ready_ones(
    prefill_scheduler, seq_factory
):
    """Ready sequences are scheduled even when others still lack a block_table."""
    unready = seq_factory([1, 2])
    ready = seq_factory([3, 4, 5, 6])
    prefill_scheduler.extend([unready, ready])

    ready.block_table = [5]
    ready.num_cached_tokens = 0

    batch, seqs = prefill_scheduler.schedule()

    assert batch is not None
    assert ready.id in seqs
    assert unready.id not in seqs
    assert unready in prefill_scheduler.waiting


def test_schedule_respects_max_num_seqs(seq_factory):
    """max_num_seqs=1 limits the batch to one sequence even when more are ready."""
    from atom.model_engine.scheduler import PrefillScheduler

    cfg = MockConfig(max_num_seqs=1)
    sched = PrefillScheduler(cfg)

    s1 = seq_factory([1, 2])
    s2 = seq_factory([3, 4])
    s1.block_table = [0]
    s2.block_table = [1]
    s1.num_cached_tokens = 0
    s2.num_cached_tokens = 0
    sched.extend([s1, s2])

    batch, seqs = sched.schedule()
    assert batch.total_seqs_num_prefill == 1


def test_schedule_respects_max_num_batched_tokens(seq_factory):
    """max_num_batched_tokens limits total tokens in a single batch."""
    from atom.model_engine.scheduler import PrefillScheduler

    cfg = MockConfig(max_num_batched_tokens=4)
    sched = PrefillScheduler(cfg)

    # 4 tokens each; only the first should fit.
    s1 = seq_factory([1, 2, 3, 4])
    s2 = seq_factory([5, 6, 7, 8])
    s1.block_table = [0]
    s2.block_table = [1]
    s1.num_cached_tokens = 0
    s2.num_cached_tokens = 0
    sched.extend([s1, s2])

    batch, seqs = sched.schedule()
    assert batch.total_tokens_num_prefill == 4
    assert s1.id in seqs
    assert s2.id not in seqs


def test_schedule_accounts_for_cached_tokens(prefill_scheduler, seq_factory):
    """num_cached_tokens is subtracted from tokens to schedule."""
    seq = seq_factory([1, 2, 3, 4])  # 4 tokens
    seq.block_table = [0]
    seq.num_cached_tokens = 2  # 2 already cached by prefix caching
    prefill_scheduler.add(seq)

    batch, _ = prefill_scheduler.schedule()
    # Only 2 new tokens should be scheduled.
    assert batch.total_tokens_num_prefill == 2


# ── postprocess() ─────────────────────────────────────────────────────────


def test_postprocess_is_noop(prefill_scheduler):
    result = prefill_scheduler.postprocess([], None)
    assert result == []


def test_postprocess_returns_empty_regardless_of_input(prefill_scheduler, seq_factory):
    seq = seq_factory([1, 2])
    result = prefill_scheduler.postprocess([seq], object())
    assert result == []


# ── block_manager ─────────────────────────────────────────────────────────


def test_no_block_manager(prefill_scheduler):
    """PrefillScheduler must never create a BlockManager."""
    assert prefill_scheduler.block_manager is None
