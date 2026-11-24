# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Exercise chunk settlement, batch snapshots and advancement through schedule()."""

from unittest.mock import patch

import pytest

from atom.model_engine.scheduler import Scheduler
from tests.test_scheduler import MockConfig


class TestMixedBatchBookkeeping:
    """A real mixed batch must get the same per-chunk bookkeeping as a
    prefill-only one. Mirrors TestMixedDecodeFirst's construction."""

    def _mixed_sched(self, **kw):
        cfg = {
            "enable_mixed_prefill_decode": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 8,
            "num_kvcache_blocks": 100,
            "kv_cache_block_size": 4,
            "max_num_seqs": 8,
        }
        cfg.update(kw)
        return Scheduler(MockConfig(**cfg))

    def _arm_mixed(self, sched, seq_factory):
        """Everything up to (but NOT including) the mixed schedule() call.

        Split out so a spy can cover only the step under test: the setup step
        below is itself a prefill schedule and calls the same bookkeeping.
        """
        d = seq_factory([1, 2, 3, 4])
        sched.add(d)
        sched.schedule()
        d.num_cached_tokens = d.num_prompt_tokens
        d.append_token(5)
        p = seq_factory([10, 11, 12, 13, 14, 15, 16, 17])
        sched.add(p)
        return p, d

    def _mixed_batch(self, sched, seq_factory):
        p, d = self._arm_mixed(sched, seq_factory)
        batch, _ = sched.schedule()
        assert batch.is_mixed, "harness failed to build a mixed batch"
        return batch, p, d

    def test_plan_midstep_runs_for_the_prefill_rows(self, seq_factory):
        """Its only call site lived inside the skipped fast path."""
        sched = self._mixed_sched()
        p, _d = self._arm_mixed(sched, seq_factory)
        with patch.object(
            sched.block_manager, "plan_midstep", wraps=sched.block_manager.plan_midstep
        ) as spy:
            batch, _ = sched.schedule()
        assert batch.is_mixed, "harness failed to build a mixed batch"
        assert spy.call_count == batch.total_seqs_num_prefill, (
            f"plan_midstep called {spy.call_count}x for "
            f"{batch.total_seqs_num_prefill} prefill row(s)"
        )
        # The prefill row, not the decode one.
        assert spy.call_args[0][0] is p

    def test_is_final_chunk_is_set_and_row_aligned(self, seq_factory):
        sched = self._mixed_sched()
        batch, _p, _d = self._mixed_batch(sched, seq_factory)
        assert batch.is_final_chunk is not None, (
            "is_final_chunk left None on a mixed batch; produces_output() "
            "reads None as True"
        )
        assert len(batch.is_final_chunk) == len(batch.req_ids), (
            "is_final_chunk must have one entry per ROW -- "
            "_record_kv_cache_ready zips it against req_ids with strict=True"
        )
        # This prompt is 8 tokens against a budget of 8 minus 1 reserved for
        # decode, so the prefill row is a MIDDLE chunk.
        assert batch.is_final_chunk[0] is False
        # Decode rows are not prompt chunks.
        assert batch.is_final_chunk[-1] is False

    def test_record_kv_cache_ready_can_zip_the_batch(self, seq_factory):
        """The strict=True zip in ModelRunner must not raise on this batch."""
        sched = self._mixed_sched()
        batch, _p, _d = self._mixed_batch(sched, seq_factory)
        # Exactly what _record_kv_cache_ready does.
        ready = [
            req_id
            for req_id, is_final in zip(
                batch.req_ids, batch.is_final_chunk, strict=True
            )
            if is_final
        ]
        assert ready == [], "a middle chunk must not be announced as ready"

    @pytest.mark.parametrize("mixed", [False, True])
    def test_a_pure_prefill_batch_is_settled_with_either_flag(self, seq_factory, mixed):
        """The gate is about the BATCH's shape, not the feature's state."""
        sched = self._mixed_sched(enable_mixed_prefill_decode=mixed)
        p = seq_factory([10, 11, 12, 13])
        sched.add(p)
        with patch.object(
            sched.block_manager, "plan_midstep", wraps=sched.block_manager.plan_midstep
        ) as spy:
            batch, _ = sched.schedule()
        assert batch.total_seqs_num_decode == 0, "expected a pure-prefill batch"
        assert spy.call_count == 1, (
            "a pure-prefill batch scheduled with the mixed flag ON still owes "
            "plan_midstep; gating it on the flag skipped it"
        )
        assert batch.is_final_chunk == [True]

    def test_batch_snapshots_midstep_destinations_before_advancing(
        self, monkeypatch, seq_factory
    ):
        sched = self._mixed_sched()
        p, d = self._arm_mixed(sched, seq_factory)
        sched.advance_on_schedule = True
        sched.drafter_needs_next_token = True
        decode_cached = d.num_cached_tokens
        planned = []

        def plan(seq, start, end):
            planned.append((seq.id, start, end))
            seq.has_per_req_cache = True
            seq.state_slots = [41]
            seq.midstep_reservations = [(42, end, 0)]

        monkeypatch.setattr(sched.block_manager, "plan_midstep", plan)
        batch, _ = sched.schedule()
        assert batch.is_mixed
        assert batch.req_ids == [p.id, d.id]
        assert planned == [(p.id, 0, 7)]
        assert batch.state_save_all == [[(42, 7)]]
        assert batch.num_cached_tokens == [0, decode_cached]
        assert batch.scheduled_tokens[:7].tolist() == list(range(10, 17))
        assert batch.is_final_chunk == [False, False]
        assert batch.next_token_ids == [17, -1]
        assert p.num_cached_tokens == 7 and p.is_partial_prefill
        assert d.num_cached_tokens == decode_cached and not d.is_partial_prefill
