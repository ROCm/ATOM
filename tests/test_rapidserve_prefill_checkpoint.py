# SPDX-License-Identifier: MIT
"""Cutting a rapidserve prefill onto a checkpoint rung.

A state checkpoint is filed under a block hash, so it sits on a hash-block
boundary; and it holds the state as of a forward's LAST token, so a forward has
to end exactly there. The ordinary scheduler buys that by shortening the chunk
(`_finalize_prefill_chunk` -> `checkpoint_cut`). Rapidserve's prefill has no
BlockManager and ran every prompt in one forward, which lands off every rung --
so `checkpointers_at` kept nothing, `_gated_hit` zeroed every later hit, and
blocks were found while reuse was refused.

The split follows this pair's one contract, decode allocates and prefill
writes: decode computes the cut and reserves the PAGE units, prefill stops
there and scatters the image between its two chunks, where its own stream
orders the copy against both forwards for free.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEDULER = ROOT / "atom/model_engine/scheduler.py"
BLOCK_MANAGER = ROOT / "atom/model_engine/block_manager.py"
CHECKPOINT = ROOT / "atom/model_engine/page_unit_checkpoint.py"
CORE = ROOT / "atom/model_engine/engine_core.py"


def _method(path: pathlib.Path, owner: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = [
        fn
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == owner
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
    ]
    assert len(found) == 1, f"{owner}.{name} is defined {len(found)} times"
    return found[0]


def _src(path: pathlib.Path, owner: str, name: str) -> str:
    out = ast.get_source_segment(path.read_text(), _method(path, owner, name))
    assert out is not None
    return out


# ── The image is armed only once its bytes exist ─────────────────────────


class TestReservationIsNotPublishable:
    """The ordering bug this design walked into once.

    `begin_store` armed every record for `complete_inflight`, and
    `DecodeScheduler._schedule` calls that every pass -- so a record reserved
    at admission was published before the peer had written a byte, and a
    resumer would have restored whatever those units held before.
    """

    def test_a_peer_written_store_is_reserved_unarmed(self):
        assert "inflight=False" in _src(
            CHECKPOINT, "PagedStateCheckpointCoordinator", "reserve_store"
        )

    def test_arming_is_a_separate_step(self):
        """...and it exists, so the image can still be published eventually."""
        assert _method(CHECKPOINT, "PageUnitCheckpointStore", "mark_inflight")
        assert _method(
            CHECKPOINT, "PagedStateCheckpointCoordinator", "settle_prefill_store"
        )

    def test_arming_happens_per_completed_chunk(self):
        """A batch result is sent only after `prefill_forward` has returned,
        and that call ends in `stream.synchronize()` with the scatter enqueued
        ahead of the forward -- so by then the bytes are real.

        Per chunk, not per prompt: the image a chunk proves is the rung at its
        START, scattered at the head of the pass that produced it. Settling
        anything a forward had not ended on published resume points over
        unwritten units, which is the bug this ordering exists for."""
        src = _src(SCHEDULER, "DecodeScheduler", "_apply_chunk_result")
        assert "cp.settle_prefill_store(store_hash)" in src

    def test_the_ordinary_path_still_arms_immediately(self):
        """Every other caller issues its own scatter in the batch it drains
        into, so deferring there would strand the image unpublished."""
        fn = _method(CHECKPOINT, "PageUnitCheckpointStore", "begin_store")
        default = fn.args.defaults[-1]
        assert isinstance(default, ast.Constant) and default.value is True


# ── The reserved hash is the one the publish will produce ────────────────


class TestRungHash:
    """An image filed under a hash no lookup ever produces is invisible, and
    silently so: the units are spent, the record reaches READY, and every
    probe misses. So the reservation and the publish have to chain the same
    way -- same seed, same slices, same `compute_hash`.
    """

    @staticmethod
    def _bm():
        from aiter_stub import stubbed_aiter
        from conftest import MockConfig

        with stubbed_aiter():
            from atom.model_engine.block_manager import BlockManager

        return BlockManager(MockConfig(enable_prefix_caching=True))

    @staticmethod
    def _seq(ntok):
        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        return Sequence(list(range(ntok)), 4, sampling_params=SamplingParams())

    def test_it_matches_hash_blocks_arithmetic(self):
        """Recomputed the way `hash_blocks` does, independently of the method
        under test -- if the two ever diverge this fails rather than quietly
        filing under the wrong key."""
        bm = self._bm()
        hbs = bm.hash_block_size
        seq = self._seq(8 * hbs)
        blocks = 4

        want = bm._chain_parent_hash(seq, 0)
        for i in range(blocks):
            want = bm.compute_hash(bm._hash_block_tokens(seq, i), want)

        assert bm._chain_hash_to(seq, blocks) == want

    def test_it_does_not_read_the_midstep_chain(self):
        """`seq.block_hashes` is empty for a COPY backend -- `_extend_hash_chain`
        returns at its first line unless the state class reserves midstep. That
        emptiness is exactly the state this runs in, and it was the bug."""
        bm = self._bm()
        seq = self._seq(8 * bm.hash_block_size)
        seq.block_hashes = []
        assert bm._chain_hash_to(seq, 4) is not None

    def test_the_chain_depends_on_content(self):
        bm = self._bm()
        hbs = bm.hash_block_size
        a, b = self._seq(8 * hbs), self._seq(8 * hbs)
        b.token_ids[0] = 9999
        assert bm._chain_hash_to(a, 4) != bm._chain_hash_to(b, 4)

    def test_a_zero_length_chain_names_nothing(self):
        """cut below one block: there is no rung to file under."""
        assert self._bm()._chain_hash_to(self._seq(16), 0) is None


# ── The op rides the chunk AFTER the rung ────────────────────────────────


def _decode(**cfg):
    from conftest import MockConfig

    from atom.model_engine.scheduler import DecodeScheduler

    return DecodeScheduler(MockConfig(**cfg))


def _executor(**cfg):
    from conftest import MockConfig

    from atom.model_engine.scheduler import PrefillExecutor

    return PrefillExecutor(MockConfig(**cfg))


def _seq(ntok, block_size=4):
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    return Sequence(list(range(ntok)), block_size, sampling_params=SamplingParams())


def _complete(sched, msg, *, sampled=7):
    """Round-trip one dispatched batch back through the completion path."""
    from atom.model_engine.disagg_types import PrefillBatchDone, PrefillChunkResult

    sched.on_prefill_batch_done(
        PrefillBatchDone(
            batch_id=msg.batch_id,
            results=[
                PrefillChunkResult(
                    seq_id=c.seq_id,
                    prefill_pos=c.start + c.num_tokens,
                    sampled_token_id=sampled,
                    is_final=c.is_final_chunk,
                )
                for c in msg.chunks
            ],
        )
    )
    sched.drain_prefill_completions()


class TestTheOpRidesTheChunkAfterTheRung:
    """A checkpoint holds the state as of a forward's LAST token.

    So the chunk ENDING on a rung is what produces the image, and the scatter
    can only run once that forward has retired -- at the head of the chunk
    STARTING there, where `build()` drains state maintenance. Decode reserves
    when it plans the first chunk, carries the op on the Sequence, and attaches
    it to the second.

    This is the same two-clock problem the deleted `take_prefill_store_op`
    solved, with one difference that removes a whole class of bug: decode now
    reserves only the rung a chunk actually lands on, so there is no ladder of
    speculative rungs to run over, skip, or release.
    """

    def test_the_op_is_taken_from_the_sequence_and_cleared(self):
        src = _src(SCHEDULER, "DecodeScheduler", "_emit_chunk")
        take = src.index("store_op = seq.pending_checkpoint_op")
        clear = src.index("seq.pending_checkpoint_op = None")
        assert take < clear, "the carried op must be taken before it is cleared"

    def test_the_next_rung_is_reserved_after_the_op_is_taken(self):
        """Reserving first would overwrite the op this chunk is owed with the
        one the NEXT chunk is owed, and the image for this rung would never be
        scattered."""
        src = _src(SCHEDULER, "DecodeScheduler", "_emit_chunk")
        take = src.index("store_op = seq.pending_checkpoint_op")
        reserve = src.index("self._reserve_chunk_checkpoint(seq, end)")
        assert take < reserve

    def test_a_rung_at_the_prompt_end_is_never_reserved(self):
        """An image is scattered by the pass AFTER the chunk that ends on its
        rung, and a chunk ending at the prompt has no pass after it: the
        sequence leaves this scheduler.

        Nothing is lost by refusing it -- decode files that boundary itself out
        of the slot prefill leaves behind, which is exactly how the non-disagg
        path keeps a prompt-end image. Reserving it as well would be a second
        reservation for one hash, written by nobody.
        """
        src = _src(SCHEDULER, "DecodeScheduler", "_emit_chunk")
        assert "if not is_final:" in src
        guard = src.index("if not is_final:")
        reserve = src.index("self._reserve_chunk_checkpoint(seq, end)")
        assert guard < reserve

    def test_the_op_travels_on_the_chunk_it_scatters(self):
        src = _src(SCHEDULER, "DecodeScheduler", "_emit_chunk")
        assert "store_op=store_op," in src

    def test_the_executor_hands_it_to_state_maintenance(self):
        """`build()` drains these BEFORE the forward, which is the whole reason
        the op is attached to the chunk after the one that produced the
        state."""
        src = _src(SCHEDULER, "PrefillExecutor", "apply_batch")
        assert "stores.append(chunk.store_op)" in src
        assert "StateMaintenanceOps(checkpoint_stores=tuple(stores))" in src

    def test_the_old_speculative_ladder_is_gone(self):
        """`prefill_cuts` shipped every rung a prompt crossed so a
        BlockManager-less prefill could pick one. Decode picks it now, from the
        chunk it just sized, so the ladder has no reason to exist -- and a
        field nothing fills is a trap for the next reader."""
        from atom.model_engine.disagg_types import PrefillChunk
        from atom.model_engine.sequence import Sequence

        seq = _seq(8)
        for gone in (
            "prefill_cuts",
            "prefill_store_hashes",
            "prefill_landed_on_rung",
        ):
            assert not hasattr(seq, gone), gone
        for gone in ("prefill_cut_within", "take_prefill_store_op"):
            assert not hasattr(Sequence, gone), gone
        assert "prefill_cuts" not in PrefillChunk.__dataclass_fields__


class TestTheRungComesFromTheSharedHelper:
    """Decode calls `_finalize_prefill_chunk`, the same helper the non-disagg
    scheduler uses.

    That is the point of moving prefill scheduling here: the half that sizes
    the chunk is now the half holding the BlockManager, so it can ask where the
    rung is instead of being told. One implementation of the cut rule, not two
    that drift.
    """

    def test_decode_uses_the_baseline_helper(self):
        src = _src(SCHEDULER, "DecodeScheduler", "_plan_prefill_chunk")
        assert "self._finalize_prefill_chunk(seq, start, chunk)" in src

    def test_the_budget_is_applied_before_the_rung(self):
        """Taken the other way round, a rung further out than one batch allows
        produces a chunk nothing can schedule, and the caller's `break` parks
        it at the head of the queue forever: a 226k prompt against a 16k budget
        asks for its 221184-token rung, never fits, and every request behind it
        stops too."""
        src = _src(SCHEDULER, "DecodeScheduler", "_plan_prefill_chunk")
        budget = src.index("self._prefill_chunk_for_budget(")
        rung = src.index("self._finalize_prefill_chunk(")
        assert budget < rung

    def test_a_rung_off_the_hash_grid_is_refused(self):
        """A checkpoint is filed under a block hash, so a cut that is not on a
        hash-block boundary has nothing to file under. A chunk stopped by the
        budget alone generally lands off the grid, and that is the common
        case."""
        src = _src(SCHEDULER, "DecodeScheduler", "_reserve_chunk_checkpoint")
        assert "if cut % bm.hash_block_size:" in src

    def test_a_disabled_coordinator_reserves_nothing(self):
        """`is None` is not the same question as "is this doing anything". The
        coordinator is constructed whenever the model has a checkpoint spec and
        carries `enabled = enable_prefix_caching and num_state_slots > 0`, so
        with prefix caching off it exists, answers every call and stores
        nothing -- and reserving against it pins units no one can resume
        from."""
        src = _src(SCHEDULER, "DecodeScheduler", "_reserve_chunk_checkpoint")
        assert "if cp is None or not cp.enabled:" in src


class TestTheImageIsSettledOnlyWhenItsForwardRetired:
    """`PrefillBatchDone` is the evidence, and the only evidence.

    `prefill_forward` is synchronous and ends in `stream.synchronize()`, so a
    batch result means every scatter it carried has landed. Settling earlier
    would arm a record for `complete_inflight` and publish a resume point over
    units nobody had written.
    """

    def test_the_settle_is_driven_by_the_recorded_hash(self):
        src = _src(SCHEDULER, "DecodeScheduler", "_apply_chunk_result")
        assert "start, end, store_hash = self._inflight_spans[result.seq_id]" in src
        assert "cp.settle_prefill_store(store_hash)" in src

    def test_nothing_is_settled_for_a_chunk_that_carried_no_image(self):
        src = _src(SCHEDULER, "DecodeScheduler", "_apply_chunk_result")
        guard = src.index("if store_hash:")
        settle = src.index("cp.settle_prefill_store(store_hash)")
        assert guard < settle

    def test_an_abandoned_reservation_is_released(self):
        """A reservation is made one chunk ahead of the chunk that writes it,
        so a sequence that dies in between leaves units pinned under a hash
        nobody will ever publish."""
        src = _src(SCHEDULER, "DecodeScheduler", "_release_pending_checkpoint")
        assert "cp.unindex(seq.pending_checkpoint_hash)" in src
        assert "seq.pending_checkpoint_op = None" in src

    def test_the_release_runs_on_every_path_that_drops_a_request(self):
        """Abort while queued, abort on landing, and preemption. A preempted
        sequence is re-prefilled from scratch, so the chunk its reservation was
        waiting for is never going to run."""
        for method in (
            "build_prefill_batch",
            "_apply_chunk_result",
            "preempt",
        ):
            src = _src(SCHEDULER, "DecodeScheduler", method)
            assert "_release_pending_checkpoint(seq)" in src, method


# ── The batch round trip ────────────────────────────────────────────────


def test_the_scheduled_batch_is_not_asked_for_its_seqs():
    """`ScheduledBatch.__init__` takes a `seqs` mapping and keeps none of it --
    it derives `req_ids` and the per-seq arrays and drops the rest. Reading
    `scheduled_batch.seqs` therefore raises at the first prefill, which no
    source-text assertion can see: the string is there either way, and only
    the attribute is wrong.

    Checked structurally, and against the class, so a future field named
    `seqs` relaxes this test instead of leaving it lying.
    """
    stored = _scheduled_batch_attrs()
    if "seqs" in stored:
        pytest.skip("ScheduledBatch now stores `seqs`; the guard is moot")
    tree = ast.parse(CORE.read_text(), filename=str(CORE))
    bad = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "seqs"
        and isinstance(node.value, ast.Name)
        and node.value.id == "scheduled_batch"
    ]
    assert not bad, (
        f"engine_core.py:{bad} reads scheduled_batch.seqs, which does not "
        "exist; the mapping apply_batch() returns alongside the batch is the "
        "one to use"
    )


def _scheduled_batch_attrs() -> set[str]:
    """Attributes `ScheduledBatch.__init__` actually assigns to self."""
    cls = next(
        c
        for c in ast.walk(ast.parse(SCHEDULER.read_text()))
        if isinstance(c, ast.ClassDef) and c.name == "ScheduledBatch"
    )
    init = next(
        f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"
    )
    return {
        t.attr
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute)
        and isinstance(t.value, ast.Name)
        and t.value.id == "self"
    }


class TestChunkedPrefillRoundTrip:
    """Decode plans every chunk and learns of each one landing.

    The old split let prefill chunk on its own and report only the chunks that
    reached a rung; decode now dispatches each chunk and waits for it, so the
    offsets on both sides advance in lockstep by construction.
    """

    @staticmethod
    def _pair(prompt=40, budget=16):
        cfg = {
            "max_num_batched_tokens": budget,
            "enable_chunked_prefill": True,
        }
        decode, executor = _decode(**cfg), _executor(**cfg)
        seq = _seq(prompt)
        twin = _seq(prompt)
        twin.id = seq.id
        decode.waiting.append(seq)
        executor.add(twin)
        return decode, executor, seq

    def test_a_long_prompt_is_chunked_until_it_finishes(self):
        decode, executor, seq = self._pair(prompt=40, budget=16)
        spans = []
        while True:
            msg = decode.build_prefill_batch()
            if msg is None:
                break
            chunk = msg.chunks[0]
            spans.append((chunk.start, chunk.num_tokens, chunk.is_final_chunk))
            seqs = executor.apply_batch(msg)[1]
            executor.retire_batch(seqs)
            _complete(decode, msg)
        assert spans == [(0, 16, False), (16, 16, False), (32, 8, True)]
        assert seq.num_cached_tokens == 40

    def test_the_block_table_is_sent_once(self):
        """Decode allocates the whole prompt at admission, so the table is
        stable for the rest of the sequence's life. Re-sending it per chunk is
        most of a long prompt's message cost."""
        decode, executor, _ = self._pair()
        sent = []
        while True:
            msg = decode.build_prefill_batch()
            if msg is None:
                break
            sent.append(msg.chunks[0].block_table is not None)
            seqs = executor.apply_batch(msg)[1]
            executor.retire_batch(seqs)
            _complete(decode, msg)
        assert sent[0] is True
        assert not any(sent[1:]), "only the first chunk carries the table"

    def test_a_mid_prompt_request_waits_in_its_own_queue(self):
        """Not in `running`, which is the DECODE queue: a partial prefill there
        would be picked up by the decode batch builder."""
        decode, executor, seq = self._pair()
        msg = decode.build_prefill_batch()
        executor.retire_batch(executor.apply_batch(msg)[1])
        _complete(decode, msg)
        assert [s.id for s in decode.prefill_partial] == [seq.id]
        assert not decode.running
        assert not decode.prefill_done

    def test_partials_are_drawn_from_before_new_requests(self):
        """A prompt already holding blocks should finish before another is let
        in to hold more -- the baseline scheduler's phase order."""
        src = _src(SCHEDULER, "DecodeScheduler", "build_prefill_batch")
        assert src.index("while self.prefill_partial:") < src.index(
            "while self.waiting:"
        )

    def test_the_executor_keeps_a_mid_prompt_sequence_but_reports_idle(self):
        """It must still HOLD the sequence -- the next chunk names it, and the
        prompt tokens live only here -- while reporting that it has nothing to
        run.

        The two answers deliberately differ. A rank waiting on decode's next
        chunk is idle, and saying otherwise costs a dummy forward on every tick
        of the wait. Nothing is lost by going quiet: the batch queue is filled
        by the socket thread, so the next `_sync_dp_state` picks the work back
        up whether or not this rank was stepping.
        """
        decode, executor = self._pair()[:2]
        msg = decode.build_prefill_batch()
        executor.retire_batch(executor.apply_batch(msg)[1])
        assert executor.has_requests(), "the next chunk names this sequence"
        assert executor.is_finished(), "but there is nothing to run right now"

    def test_the_executor_releases_a_finished_sequence(self):
        decode, executor = self._pair(prompt=8, budget=16)[:2]
        msg = decode.build_prefill_batch()
        executor.retire_batch(executor.apply_batch(msg)[1])
        assert executor.is_finished(), "decode keeps it for the rest of its life"

    def test_a_batch_naming_an_unknown_sequence_is_retried_not_dropped(self):
        """The manager sends decode's ADD first, so decode can plan a batch
        before this process has drained its own input queue."""
        decode = self._pair()[0]
        msg = decode.build_prefill_batch()
        empty = _executor(max_num_batched_tokens=16, enable_chunked_prefill=True)
        assert empty.apply_batch(msg) is None


class TestOneBatchInFlight:
    def test_a_second_build_is_refused_while_one_is_outstanding(self):
        decode = _decode()
        decode.waiting.append(_seq(8))
        decode.waiting.append(_seq(8))
        assert decode.build_prefill_batch() is not None
        assert decode.build_prefill_batch() is None

    def test_the_slot_frees_when_the_batch_lands(self):
        decode = _decode()
        decode.waiting.append(_seq(8))
        msg = decode.build_prefill_batch()
        assert decode.prefill_batch_inflight is True
        _complete(decode, msg)
        assert decode.prefill_batch_inflight is False
        assert not decode.prefill_inflight
        assert not decode._inflight_spans

    def test_a_dispatched_chunk_with_no_result_is_not_lost(self):
        """It would otherwise be dropped when the in-flight map is cleared:
        never decoded, never freed, and invisible except as a request that
        stops."""
        from atom.model_engine.disagg_types import PrefillBatchDone

        decode = _decode()
        seq = _seq(8)
        decode.waiting.append(seq)
        msg = decode.build_prefill_batch()
        free_before = decode.block_manager.kv.num_free

        decode.on_prefill_batch_done(
            PrefillBatchDone(batch_id=msg.batch_id, results=[])
        )
        decode.drain_prefill_completions()

        assert not decode.prefill_done
        assert not decode.prefill_partial
        assert decode.block_manager.kv.num_free > free_before, "blocks leaked"

    def test_a_result_for_the_wrong_batch_is_refused(self):
        """Only one batch is ever outstanding, so this cannot happen without
        the two sides having diverged about what was sent."""
        from atom.model_engine.disagg_types import (
            PrefillBatchDone,
            PrefillChunkResult,
        )

        decode = _decode()
        seq = _seq(8)
        decode.waiting.append(seq)
        decode.build_prefill_batch()
        decode.on_prefill_batch_done(
            PrefillBatchDone(
                batch_id=999,
                results=[
                    PrefillChunkResult(seq_id=seq.id, prefill_pos=8, sampled_token_id=1)
                ],
            )
        )
        decode.drain_prefill_completions()
        assert decode.prefill_batch_inflight is True, "the real batch is still out"
        assert not decode.prefill_done
