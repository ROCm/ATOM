# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Every batch carrying prefill rows owes the same per-chunk bookkeeping.

The prefill-only fast path used to own four things outright: `is_final_chunk`
(the repo's only assignment), `plan_midstep` (the repo's only call),
`next_token_ids`, and `_advance_prefill_on_schedule` (also the only call). Its
gate asked whether mixed batching was ENABLED rather than whether THIS BATCH
had decode rows -- so turning the flag on skipped all four, on pure-prefill
batches as much as mixed ones.

None of it is loud. `is_final_chunk` left None reads as True in
`produces_output`, a missing midstep checkpoint only matters once something
preempts, and short prompts never chunk at all -- so a GSM8K run whose prompts
each fit one chunk scores exactly the same either way.

These are source-level checks: they need no aiter or GPU and so run in CI.
"""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from atom.model_engine.scheduler import Scheduler
from tests.test_scheduler import MockConfig

_ROOT = Path(__file__).resolve().parents[1]
_SCHED = _ROOT / "atom/model_engine/scheduler.py"


def _schedule_src() -> str:
    src = _SCHED.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_schedule":
            return "\n".join(src.splitlines()[node.lineno - 1 : node.end_lineno])
    pytest.fail("scheduler.py has no _schedule")


def _fallthrough_src() -> str:
    """The part of `_schedule` after the prefill-only early return."""
    body = _schedule_src()
    marker = "# --- Decode scheduling (also fall-through for mixed batches) ---"
    assert marker in body, "the fall-through marker moved; re-anchor this test"
    return body[body.index(marker) :]


def test_the_feature_flag_does_not_gate_the_bookkeeping():
    """Only the early RETURN may depend on `enable_mixed_prefill_decode`."""
    body = _schedule_src()
    gated = body.count("not self.enable_mixed_prefill_decode")
    assert gated == 1, (
        f"_schedule tests `not enable_mixed_prefill_decode` {gated} times; "
        "exactly one (the prefill-only early return) is expected -- gating "
        "anything else on the flag is what silently dropped plan_midstep, "
        "is_final_chunk and the schedule-time advance"
    )


def test_the_fallthrough_settles_prefill_chunks():
    """A mixed / flag-on-pure-prefill batch must do the same bookkeeping."""
    tail = _fallthrough_src()
    assert "_settle_prefill_chunks(" in tail, (
        "the fall-through path builds a batch with prefill rows without "
        "settling their chunks: no plan_midstep, no is_final_chunk"
    )
    assert (
        "_advance_prefill_on_schedule(" in tail
    ), "the fall-through path never advances chunked-prefill progress"


def test_settle_is_called_before_the_batch_is_built():
    """`plan_midstep` leaves state the batch snapshots."""
    tail = _fallthrough_src()
    assert tail.index("_settle_prefill_chunks(") < tail.index(
        "decode_batch = ScheduledBatch("
    ), "prefill chunks must be settled before the batch snapshots them"


def test_advance_runs_after_the_batch_is_built():
    """The batch must keep pre-advance offsets."""
    tail = _fallthrough_src()
    assert tail.index("decode_batch = ScheduledBatch(") < tail.index(
        "_advance_prefill_on_schedule("
    ), "advancing before the build would hand the batch post-advance offsets"


def test_row_aligned_lists_are_padded_for_decode_rows():
    """`is_final_chunk` is zipped against req_ids with strict=True.

    A mixed batch has more rows than prefill seqs, so a prefill-length list is
    a crash there -- and a trap anywhere else, since every other per-row list
    on a ScheduledBatch has one entry per row.
    """
    tail = _fallthrough_src()
    assert "[False] * num_seqs_decode" in tail, (
        "is_final_chunk is not padded out to one entry per row; "
        "_record_kv_cache_ready zips it against req_ids with strict=True"
    )
    assert (
        "[-1] * num_seqs_decode" in tail
    ), "next_token_ids is not padded out to one entry per row"


def test_the_batch_carries_both_lists():
    tail = _fallthrough_src()
    assert "is_final_chunk=is_final_chunk" in tail
    assert "next_token_ids=next_token_ids" in tail


def test_bookkeeping_helpers_take_prefill_rows_only():
    """Handing the whole batch in would advance decode rows' prompt cursors."""
    src = _SCHED.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for name in ("_settle_prefill_chunks", "_advance_prefill_on_schedule"):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                args = [a.arg for a in node.args.args]
                assert "prefill_seqs" in args, (
                    f"{name} takes {args}; it must take the prefill rows "
                    "explicitly, not a whole mixed batch's seq dict"
                )
                break
        else:
            pytest.fail(f"scheduler.py has no {name}")


# ── behavioural: the checks above are source-level and a clever edit can keep
# the substring while removing the call. These build a real mixed batch. ──────


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

    def test_a_flag_on_pure_prefill_batch_is_settled_too(self, seq_factory):
        """The gate is about the BATCH's shape, not the feature's state."""
        sched = self._mixed_sched()
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
