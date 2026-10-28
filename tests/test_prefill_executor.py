# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for PrefillExecutor — no GPU required.

The rapidserve prefill process chooses nothing: decode owns the BlockManager,
so decode decides which sequences run, how wide each chunk is and which
checkpoint rung it lands on, and ships that decision as a `PrefillBatchMsg`.
This class resolves the sequence ids, applies the spans, and turns them into a
`ScheduledBatch`.

NOTE: All atom.* imports are deferred to inside fixtures/tests.
test_mxfp4_moe_has_bias.py purges sys.modules["atom.*"] at collection time;
top-level imports here would create a second copy of atom.model_engine.sequence
and break SequenceStatus enum identity checks in test_scheduler.py.
"""

import pytest
from conftest import MockConfig


@pytest.fixture
def executor():
    from atom.model_engine.scheduler import PrefillExecutor

    return PrefillExecutor(MockConfig())


@pytest.fixture
def seq_factory():
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    def make(token_ids, block_size=4):
        return Sequence(token_ids, block_size, sampling_params=SamplingParams())

    return make


def _msg(chunks, batch_id=0, cu_stream_fraction=None):
    from atom.model_engine.disagg_types import PrefillBatchMsg

    return PrefillBatchMsg(
        batch_id=batch_id,
        target_rank=0,
        cu_stream_fraction=cu_stream_fraction,
        chunks=chunks,
    )


def _chunk(seq, start=0, num_tokens=None, is_final=True, block_table=(0, 1), **kw):
    from atom.model_engine.disagg_types import PrefillChunk

    if num_tokens is None:
        num_tokens = seq.num_tokens - start
    return PrefillChunk(
        seq_id=seq.id,
        start=start,
        num_tokens=num_tokens,
        is_final_chunk=is_final,
        block_table=list(block_table) if block_table is not None else None,
        **kw,
    )


# ── is_finished / has_requests ────────────────────────────────────────────


def test_is_finished_when_empty(executor):
    assert executor.is_finished()


def test_still_idle_after_add(executor, seq_factory):
    """An ADD gives this process a Sequence, not work: decode decides when any
    of it runs. See TestIdleMeansNoBatch for why the distinction is expensive.
    """
    executor.add(seq_factory([1, 2, 3]))
    assert executor.has_requests()
    assert executor.is_finished()


def test_has_requests_false_when_empty(executor):
    assert not executor.has_requests()


def test_has_requests_true_after_add(executor, seq_factory):
    executor.add(seq_factory([1, 2]))
    assert executor.has_requests()


# ── liveness: a batch is work, a held sequence is not ────────────────────


class TestIdleMeansNoBatch:
    """`is_finished()` gates the busy loop, so it must mean "nothing to RUN".

    `DPEngineCoreProc.busy_loop` answers "unfinished, but the step ran nothing"
    with `_execute_dummy_batch()` -- a real forward, issued to keep the MoE
    collective matched with ranks that did run something. So a rank that claims
    work it cannot do pays a full forward for every tick it waits.

    Reporting the sequence queues here did exactly that: a chunked prompt is
    held for its whole life, so every rank claimed work between every pair of
    chunks. Measured on a 670k-token agentic prompt: 4.7 dummy forwards per
    real chunk, ~40% of prefill's GPU time, with the prefill backlog reading 0
    the entire time.
    """

    def test_holding_a_sequence_is_not_work(self, executor, seq_factory):
        seq = seq_factory(list(range(16)))
        executor.add(seq)
        assert executor.has_requests() is True
        assert executor.is_finished() is True, (
            "a sequence with no batch to run must report idle, or every rank "
            "burns a dummy forward per tick while waiting on decode"
        )

    def test_a_queued_batch_is_work(self, executor, seq_factory):
        seq = seq_factory(list(range(16)))
        executor.add(seq)
        executor.enqueue_batch(_msg([_chunk(seq, num_tokens=8, is_final=False)]))
        assert executor.is_finished() is False

    def test_a_mid_prompt_rank_is_idle_between_chunks(self, executor, seq_factory):
        """The state that used to cost a dummy forward on every tick."""
        seq = seq_factory(list(range(16)))
        executor.add(seq)
        msg = _msg([_chunk(seq, num_tokens=8, is_final=False)])
        executor.enqueue_batch(msg)
        _, seqs = executor.apply_batch(executor.peek_batch())
        executor.drop_batch(msg)
        executor.retire_batch(seqs)

        assert executor.has_requests() is True, "decode still owes it a chunk"
        assert executor.is_finished() is True, "but there is nothing to run"

    def test_the_queue_is_the_one_the_engine_drains(self, executor, seq_factory):
        """Socket thread fills it, engine thread drains it, `is_finished()`
        reads it. Three callers, one queue -- a second copy on the engine core
        would let them disagree about whether this rank has work."""
        seq = seq_factory([1, 2, 3, 4])
        executor.add(seq)
        msg = _msg([_chunk(seq)])
        executor.enqueue_batch(msg)
        assert executor.peek_batch() is msg
        assert executor.pending_batch_count == 1
        executor.drop_batch(msg)
        assert executor.peek_batch() is None
        assert executor.is_finished() is True


# ── apply_batch() ─────────────────────────────────────────────────────────


def test_a_batch_naming_an_unknown_sequence_is_refused(executor, seq_factory):
    """None means "not yet", never an error: the manager sends decode's copy of
    a request first, so decode can plan a batch before this process has drained
    its own input queue. The caller retries next tick."""
    stranger = seq_factory([1, 2, 3, 4])
    assert executor.apply_batch(_msg([_chunk(stranger)])) is None


def test_a_resolved_batch_becomes_a_scheduled_batch(executor, seq_factory):
    from atom.model_engine.sequence import SequenceStatus, SequenceType

    seq = seq_factory([10, 20, 30, 40])
    executor.add(seq)

    batch, seqs = executor.apply_batch(_msg([_chunk(seq)]))

    assert batch.total_seqs_num_prefill == 1
    assert batch.total_tokens_num_prefill == 4
    assert seq.id in seqs
    assert seq.status == SequenceStatus.RUNNING
    assert seq.type == SequenceType.PREFILL
    assert seq not in executor.waiting
    assert seq in executor.running


def test_the_span_decode_planned_is_adopted_verbatim(executor, seq_factory):
    """Decode owns the offset; this side only adopts it. Deriving it locally
    instead is what made prefill's copy authoritative, which is the thing the
    decode-driven design removes."""
    seq = seq_factory(list(range(16)))
    executor.add(seq)

    batch, _ = executor.apply_batch(
        _msg([_chunk(seq, start=4, num_tokens=8, is_final=False)])
    )

    assert seq.num_cached_tokens == 4
    assert batch.total_tokens_num_prefill == 8
    assert seq.is_partial_prefill is True


def test_the_batch_reports_per_chunk_finality(executor, seq_factory):
    """Left unset, `ScheduledBatch.is_final_chunk` stays None and
    `produces_output()` answers True unconditionally -- so a batch of nothing
    but middle chunks claims to produce output and the draft pass runs for
    sequences that are mid-prompt."""
    a, b = seq_factory(list(range(8))), seq_factory(list(range(8)))
    executor.extend([a, b])

    batch, _ = executor.apply_batch(
        _msg(
            [
                _chunk(a, num_tokens=4, is_final=False),
                _chunk(b, num_tokens=8, is_final=True),
            ]
        )
    )

    assert list(batch.is_final_chunk) == [False, True]
    assert batch.produces_output() is True


def test_the_block_table_is_only_overwritten_when_sent(executor, seq_factory):
    """Decode allocates the whole prompt at admission, so the table is stable
    after the first chunk and `None` means "unchanged"."""
    from atom.model_engine.sequence import new_block_table

    seq = seq_factory(list(range(16)))
    executor.add(seq)
    executor.apply_batch(_msg([_chunk(seq, num_tokens=8, is_final=False)]))
    assert list(seq.block_table) == [0, 1]

    seq.block_table = new_block_table([7, 8, 9])
    executor.apply_batch(
        _msg([_chunk(seq, start=8, num_tokens=8, block_table=None)], batch_id=1)
    )
    assert list(seq.block_table) == [7, 8, 9]


def test_the_cu_fraction_is_forwarded_unchanged(executor, seq_factory):
    seq = seq_factory([1, 2, 3, 4])
    executor.add(seq)

    batch, _ = executor.apply_batch(_msg([_chunk(seq)], cu_stream_fraction=0.5))
    assert batch.cu_stream_fraction == 0.5


def test_a_store_op_becomes_state_maintenance(executor, seq_factory):
    """`build()` drains these BEFORE the forward, which is why decode attaches
    an image to the chunk AFTER the one that produced it."""
    seq = seq_factory(list(range(16)))
    executor.add(seq)

    batch, _ = executor.apply_batch(
        _msg([_chunk(seq, start=8, num_tokens=8, store_op="OP")])
    )
    assert batch.state_maintenance_ops.checkpoint_stores == ("OP",)


def test_no_store_op_leaves_maintenance_unset(executor, seq_factory):
    seq = seq_factory([1, 2, 3, 4])
    executor.add(seq)

    batch, _ = executor.apply_batch(_msg([_chunk(seq)]))
    assert batch.state_maintenance_ops.empty


# ── retire_batch() ────────────────────────────────────────────────────────


def test_a_finished_sequence_is_released(executor, seq_factory):
    """Decode keeps it for the rest of its life; this side is done with it."""
    seq = seq_factory([1, 2, 3, 4])
    executor.add(seq)
    _, seqs = executor.apply_batch(_msg([_chunk(seq)]))

    executor.retire_batch(seqs)

    assert executor.is_finished()


def test_a_mid_prompt_sequence_goes_back_to_waiting(executor, seq_factory):
    """`is_finished()` reads off these queues, so dropping a sequence decode
    still owes chunks to would let the rank report idle -- and an idle rank
    stops stepping, so the next batch naming it would never be drained."""
    seq = seq_factory(list(range(16)))
    executor.add(seq)
    _, seqs = executor.apply_batch(_msg([_chunk(seq, num_tokens=8, is_final=False)]))

    executor.retire_batch(seqs)

    assert seq in executor.waiting
    assert seq not in executor.running
    assert executor.has_requests()


# ── drop() ────────────────────────────────────────────────────────────────


def test_drop_forgets_an_aborted_request(executor, seq_factory):
    """Without it the Sequence stays queued forever and this rank never reports
    idle, so the prefill group keeps stepping and runs a dummy forward per tick
    for work that no longer exists."""
    seq = seq_factory([1, 2, 3, 4])
    executor.add(seq)

    assert executor.drop(seq.id) is True
    assert executor.is_finished()
    assert executor.drop(seq.id) is False


# ── postprocess() ─────────────────────────────────────────────────────────


def test_postprocess_is_noop(executor):
    result = executor.postprocess([], None)
    assert result == []


def test_postprocess_returns_empty_regardless_of_input(executor, seq_factory):
    seq = seq_factory([1, 2])
    result = executor.postprocess([seq], object())
    assert result == []


# ── block_manager ─────────────────────────────────────────────────────────


def test_no_block_manager(executor):
    """PrefillExecutor must never create a BlockManager: it allocates nothing,
    and a pool here would be a second owner of decode's blocks."""
    assert executor.block_manager is None


# ── engine_stats (throughput) ─────────────────────────────────────────────


def test_throughput_section_follows_enable_log_stats():
    """Prefill side enables only the throughput section, gated on the config
    flag; spec and cache are always off here (no drafter, no BlockManager)."""
    from atom.model_engine.scheduler import PrefillExecutor

    on = PrefillExecutor(MockConfig(enable_log_stats=True)).engine_stats
    assert on.throughput_enabled is True
    assert on.spec_enabled is False
    assert on.cache_enabled is False

    off = PrefillExecutor(MockConfig(enable_log_stats=False)).engine_stats
    assert off.throughput_enabled is False


def test_applying_a_batch_records_prompt_throughput(seq_factory):
    """`Scheduler.schedule` ticks the throughput window at its single entry
    point; this class has no `schedule`, so the tick lives at the one place a
    batch becomes real.

    The default 10s wall-clock interval means maybe_log does not fire (or
    reset) here, so the accumulated prompt-token count survives to assert on.
    """
    from atom.model_engine.scheduler import PrefillExecutor

    sched = PrefillExecutor(MockConfig(enable_log_stats=True))
    seq = seq_factory([10, 20, 30, 40])
    sched.add(seq)

    sched.apply_batch(_msg([_chunk(seq)]))

    assert sched.engine_stats.num_prompt_tokens == 4
    # Prefill produces no sampled tokens.
    assert sched.engine_stats.num_generation_tokens == 0
