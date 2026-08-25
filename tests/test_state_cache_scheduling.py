# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Scheduler-side wiring of the recurrent-state checkpoint pool.

The pool's write path has no kernel component: instead of teaching the
conv/scan kernels to snapshot mid-scan states (which `fla`'s chunked KDA/GDN
kernels do not expose — they return only a final state), the scheduler shrinks
each prefill chunk so it ENDS on a `state_cache_block_size` boundary. The
forward's single leftover state then IS the checkpoint, and saving it is a
device-to-device copy.

That makes the scheduler responsible for two things these tests pin down:
  * every path that decides a prefill chunk length aligns it, so a long prompt
    checkpoints at every boundary it passes rather than only the first;
  * the per-step slot lists handed to the runner (`state_restore_slots`,
    `state_ckpt_write_slots`) stay index-aligned with `state_slots`,
    since both address the same working slots.

Pure Python — no GPU, no model. Drives the real Scheduler through schedule().
"""

import ast
from pathlib import Path

import pytest

from atom.model_engine.scheduler import ScheduledBatch, Scheduler
from conftest import MockConfig

REPO = Path(__file__).resolve().parents[1]
SCHEDULER_SRC = REPO / "atom" / "model_engine" / "scheduler.py"

BS = 4  # paged block size
M = 16  # checkpoint granularity: 4 paged blocks


def make_scheduler(m=M, num_state_cache_blocks=8, **overrides):
    kwargs = dict(
        kv_cache_block_size=BS,
        num_kvcache_blocks=256,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        state_cache_block_size=m,
        num_state_cache_blocks=num_state_cache_blocks,
    )
    kwargs.update(overrides)
    return Scheduler(MockConfig(**kwargs))


def tokens(n, base=0):
    return list(range(base, base + n))


def drain_prefill(sched, seq):
    """Run schedule() until `seq` has forwarded its whole prompt.

    Returns the per-step chunk sizes. Mirrors the engine loop closely enough
    for the pool: advance num_cached_tokens and publish the finalized blocks,
    as postprocess would after the forward.
    """
    chunks = []
    bm = sched.block_manager
    while seq.num_cached_tokens < seq.num_tokens:
        batch, _ = sched.schedule()
        assert batch is not None, "scheduler stalled mid-prefill"
        idx = batch.req_ids.index(seq.id)
        chunk = int(batch.num_scheduled_tokens[idx])
        chunks.append(chunk)
        bm.hash_blocks(seq, chunk)
        seq.num_cached_tokens += chunk
        seq.is_partial_prefill = seq.num_cached_tokens < seq.num_tokens
        if not seq.is_partial_prefill:
            sched._partial_prefill_count = 0
        else:
            sched._partial_prefill_count = 1
        bm.state_cache.end_step(seq)
        bm.state_cache.release_restore(seq)
    return chunks


# ── chunk alignment reaches every prefill path ─────────────────────────────


class TestEveryChunkPathIsAligned:
    def test_all_prefill_chunk_decisions_go_through_the_aligner(self):
        """Every `_assert_positive_prefill_chunk` call site must be preceded by
        `_align_prefill_chunk`.

        That assertion guards the three places a chunk length is finalized
        (fresh admission, offload resume, connector-adjusted). Missing the
        aligner on any of them would let a chunk end mid-span, and the request
        would silently stop checkpointing from there on — a cache miss no test
        of the aligned paths could see.
        """
        tree = ast.parse(SCHEDULER_SRC.read_text())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "schedule"
        )
        asserts, aligns = [], []
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_assert_positive_prefill_chunk":
                    asserts.append(node.lineno)
                elif node.func.attr == "_align_prefill_chunk":
                    aligns.append(node.lineno)
        assert asserts, "no chunk assertions found — did schedule() change?"
        for line in asserts:
            assert any(
                line - 6 <= a < line for a in aligns
            ), f"chunk finalized at scheduler.py:{line} without alignment"

    def test_partial_prefill_resume_is_aligned(self):
        """Phase 1 (resume) computes its own chunk and must align it too."""
        src = SCHEDULER_SRC.read_text()
        tree = ast.parse(src)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "schedule"
        )
        # The resume loop is the one reading `is_partial_prefill`.
        aligned = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_align_prefill_chunk"
            for n in ast.walk(fn)
        )
        assert aligned and "is_partial_prefill" in src


# ── observable scheduling behaviour ────────────────────────────────────────


class TestChunkingIsAligned:
    def test_long_prompt_is_split_at_every_boundary(self, seq_factory):
        sched = make_scheduler()
        seq = seq_factory(tokens(64))
        sched.add(seq)
        assert drain_prefill(sched, seq) == [16, 16, 16, 16]

    def test_trailing_partial_span_is_forwarded_in_one_go(self, seq_factory):
        """A prompt that does not end on a boundary spends its last step on the
        remainder, which simply checkpoints nothing."""
        sched = make_scheduler()
        seq = seq_factory(tokens(40))
        sched.add(seq)
        assert drain_prefill(sched, seq) == [16, 16, 8]

    def test_short_prompt_is_untouched(self, seq_factory):
        sched = make_scheduler()
        seq = seq_factory(tokens(12))  # never reaches a boundary
        sched.add(seq)
        assert drain_prefill(sched, seq) == [12]

    def test_disabled_pool_leaves_chunking_alone(self, seq_factory):
        """Non-recurrent models must see byte-identical scheduling."""
        sched = make_scheduler(num_state_cache_blocks=0)
        assert not sched.block_manager.state_cache_enabled
        seq = seq_factory(tokens(64))
        sched.add(seq)
        assert drain_prefill(sched, seq) == [64]

    def test_every_boundary_is_checkpointed(self, seq_factory):
        sched = make_scheduler()
        seq = seq_factory(tokens(64))
        sched.add(seq)
        drain_prefill(sched, seq)
        # 4 spans forwarded, 4 checkpoints published.
        assert len(sched.block_manager.state_cache.hash_to_block_id) == 4

    def test_a_second_identical_prompt_resumes_from_the_last_checkpoint(
        self, seq_factory
    ):
        sched = make_scheduler()
        warm = seq_factory(tokens(64))
        sched.add(warm)
        drain_prefill(sched, warm)

        later = seq_factory(tokens(64))
        hit = sched.block_manager.can_allocate(later)
        # Paged pool alone would offer 15 blocks; the state bounds it to 12
        # (= 3 spans), the last boundary whose checkpoint exists.
        assert hit == 12
        sched.block_manager.allocate(later, hit)
        assert later.state_restore_slot >= 0

    def test_a_resumed_request_keeps_checkpointing(self, seq_factory):
        """Resuming lands on a boundary, so the following chunks stay aligned."""
        sched = make_scheduler()
        warm = seq_factory(tokens(64))
        sched.add(warm)
        drain_prefill(sched, warm)

        later = seq_factory(tokens(64))
        sched.add(later)
        assert drain_prefill(sched, later) == [16]  # only the last span remains
        assert later.num_cached_tokens == 64


# ── batch slot plumbing ────────────────────────────────────────────────────


@pytest.fixture
def batch_seq(seq_factory):
    """A real Sequence with the checkpoint fields the batch reads pre-set."""

    def make(has_per_req_cache, group, restore=-1, writes=None):
        seq = seq_factory(tokens(4), has_per_req_cache=has_per_req_cache)
        seq.state_slot = group
        seq.state_restore_slot = restore
        seq.state_ckpt_writes = dict(writes or {})
        return seq

    return make


def make_batch(seqs):
    return ScheduledBatch(
        {s.id: s for s in seqs},
        [4] * len(seqs),
        total_tokens_num=4 * len(seqs),
        total_seqs_num=len(seqs),
        total_seqs_num_prefill=len(seqs),
        total_tokens_num_prefill=4 * len(seqs),
    )


class TestBatchSlots:
    def test_slot_lists_align_with_cache_groups(self, batch_seq):
        seqs = [
            batch_seq(True, 0, restore=5, writes={3: 7}),
            batch_seq(True, 1, restore=-1, writes={}),
            batch_seq(False, -1),  # no recurrent state: filtered from all three
            batch_seq(True, 2, restore=-1, writes={11: 2}),
        ]
        batch = make_batch(seqs)
        assert batch.state_slots_committed == [0, 1, 2]
        assert batch.state_restore_slots == [5, -1, -1]
        assert batch.state_ckpt_write_slots == [7, -1, 2]

    def test_unclaimed_group_is_excluded_from_every_list(self, batch_seq):
        """A seq whose working slot was not claimed this step has no slot to
        restore into or save out of, so it must not shift the others."""
        seqs = [
            batch_seq(True, -1, restore=9, writes={3: 9}),
            batch_seq(True, 0, restore=1),
        ]
        batch = make_batch(seqs)
        assert batch.state_slots_committed == [0]
        assert batch.state_restore_slots == [1]
        assert batch.state_ckpt_write_slots == [-1]

    def test_non_recurrent_batch_has_empty_slot_lists(self, batch_seq):
        batch = make_batch([batch_seq(False, -1), batch_seq(False, -1)])
        assert batch.state_restore_slots == []
        assert batch.state_ckpt_write_slots == []

    @pytest.mark.parametrize("attr", ["state_restore_slots", "state_ckpt_write_slots"])
    def test_slot_lists_match_cache_group_length(self, attr, batch_seq):
        seqs = [batch_seq(True, i, restore=i, writes={i: i}) for i in range(4)]
        seqs.insert(2, batch_seq(False, -1))
        batch = make_batch(seqs)
        assert len(getattr(batch, attr)) == len(batch.state_slots_committed)
