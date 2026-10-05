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

    def test_arming_happens_on_the_final_chunk(self):
        """`PrefillDone` is sent only after the forward's event retires and the
        scatter was enqueued ahead of it, so by then the bytes are real."""
        src = _src(SCHEDULER, "DecodeScheduler", "on_prefill_done")
        assert "settle_prefill_store" in src

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


def test_the_store_is_taken_before_the_offset_advances():
    """`take_prefill_store_op` gates on `num_cached_tokens`, so whichever pass
    reads it decides which chunk carries the scatter. Taken after the advance
    it would attach to the chunk that produces the state, and `build()` runs
    maintenance BEFORE its forward -- scattering a slot not yet filled."""
    src = _src(SCHEDULER, "PrefillScheduler", "_schedule")
    assert "take_prefill_store_op()" in src
    assert "seq.num_cached_tokens +=" not in src, (
        "the offset must advance after the forward (the runner reads it as the "
        "chunk's start), not in _schedule"
    )


def test_the_offset_advances_after_the_forward():
    src = _src(CORE, "PrefillEngineCore", "_process_engine_step")
    assert "seq.num_cached_tokens += int(num_tokens)" in src


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
        "exist; the mapping schedule() returns alongside the batch is the "
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


class TestTakeGuard:
    @staticmethod
    def _seq(cached, cuts):
        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        seq = Sequence([1, 2, 3, 4], 4, sampling_params=SamplingParams())
        seq.num_cached_tokens = cached
        seq.prefill_cuts = list(cuts)
        return seq

    def test_withheld_before_the_rung(self):
        """The chunk that ends AT the rung must not carry it."""
        seq = self._seq(cached=0, cuts=[(1024, "A")])
        assert seq.take_prefill_store_op() is None

    def test_handed_out_once_past_it(self):
        seq = self._seq(cached=1024, cuts=[(1024, "A")])
        assert seq.take_prefill_store_op() == "A"

    def test_handed_out_only_once(self):
        seq = self._seq(cached=1024, cuts=[(1024, "A")])
        assert seq.take_prefill_store_op() == "A"
        assert seq.take_prefill_store_op() is None

    def test_an_unchunked_prompt_carries_nothing(self):
        """An empty ladder is every non-rapidserve request and any prompt with
        no rung; it must not read as 'past it'."""
        seq = self._seq(cached=0, cuts=[])
        assert seq.take_prefill_store_op() is None
        assert seq.prefill_cut_pos == 0

    def test_the_ladder_is_consumed_in_order(self):
        """Each rung is handed out by the pass after the chunk that ends on it,
        and `prefill_cut_pos` advances to the next -- which is what makes a
        prompt crossing many rungs checkpoint at all of them instead of one."""
        seq = self._seq(cached=0, cuts=[(1024, "A"), (2048, "B"), (3072, "C")])
        assert seq.prefill_cut_pos == 1024
        assert seq.take_prefill_store_op() is None

        seq.num_cached_tokens = 1024
        assert seq.take_prefill_store_op() == "A"
        assert seq.prefill_cut_pos == 2048, "the next rung becomes the next cut"
        assert seq.take_prefill_store_op() is None

        seq.num_cached_tokens = 2048
        assert seq.take_prefill_store_op() == "B"
        seq.num_cached_tokens = 3072
        assert seq.take_prefill_store_op() == "C"
        assert seq.prefill_cut_pos == 0, "ladder exhausted"

    def test_a_chunk_spanning_two_rungs_does_not_skip_one(self):
        """Budget and rungs interleave, so a pass can land past more than one.
        Each is still taken separately: one scatter per image, in order."""
        seq = self._seq(cached=4096, cuts=[(1024, "A"), (2048, "B")])
        assert seq.take_prefill_store_op() == "A"
        assert seq.take_prefill_store_op() == "B"
        assert seq.take_prefill_store_op() is None


# ── The ladder, not just its last rung ───────────────────────────────────


class TestLadderWalk:
    """Why the walk is per chunk and not per prompt.

    `checkpoint_cut` answers with the RIGHTMOST rung inside the window it is
    handed (`rung = min(end, limit)` floored to the interval, then
    `min(rung, demand, anchor)`). Ask it once about a whole prompt and a 226k
    prompt yields one checkpoint at 221184; the ordinary scheduler asks once
    per chunk and walks the whole ladder. One resume point against fourteen
    measured as 59% against 66% on agentic traffic.
    """

    def test_the_walk_steps_a_budget_sized_window(self):
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert "min(pos + budget, end)" in src, (
            "the window has to match the chunk prefill will actually run, or "
            "the rungs chosen are not the rungs reachable"
        )
        assert "while pos < end:" in src

    def test_every_rung_gets_its_own_reservation(self):
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert "seq.prefill_cuts.append((cut, op))" in src

    def test_an_unreservable_rung_does_not_stall_the_walk(self):
        """`pos` advances past a rung whether or not it was reserved. Left
        where it was, the same window would answer with the same rung forever
        and admission would hang."""
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        advance = src.index("pos = cut")
        assert src.index("if op is None:", advance) > advance, (
            "pos must advance before the unreserved-rung continue"
        )

    def test_a_window_with_no_rung_still_advances(self):
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert "pos = window" in src

    def test_the_ladder_is_capped(self):
        """Each reservation pins units from admission until the chunk that
        writes it, so an uncapped ladder on a long prompt holds many images for
        the whole prefill."""
        import ast as _ast

        cls = next(
            c
            for c in _ast.walk(_ast.parse(SCHEDULER.read_text()))
            if isinstance(c, _ast.ClassDef) and c.name == "DecodeScheduler"
        )
        caps = [
            n.value.value
            for n in cls.body
            if isinstance(n, _ast.Assign)
            and any(
                isinstance(t, _ast.Name) and t.id == "MAX_PREFILL_CHECKPOINTS"
                for t in n.targets
            )
            and isinstance(n.value, _ast.Constant)
        ]
        assert caps and caps[0] > 0

    def test_the_cap_keeps_the_tail(self):
        """The last rungs are the ones agentic traffic resumes at -- 93.5% of
        resumes land on a previous prompt end, 0.0% on the 8192 ladder -- so
        dropping from the front is what keeps the useful ones."""
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert "seq.prefill_cuts[dropped:]" in src

    def test_dropped_reservations_are_released(self):
        """`begin_store` took the units; discarding the op without unindexing
        leaks them for the pool's lifetime."""
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert "unindex(h)" in src


# ── A prompt bigger than the budget must still run ───────────────────────


class TestBudgetBound:
    """The deadlock this scheduler shipped with, and the one the cut added.

    `PrefillScheduler` had no chunked prefill: it took the whole remaining
    prompt and `break`-ed when that exceeded `max_num_batched_tokens`. A prompt
    longer than the budget could therefore only be queued, never run, and it
    parked at the head of the queue with every request behind it. Measured on
    an agentic workload: 46 requests of up to 226k tokens against a 16384
    budget, two forwards total, then nothing.

    Applying the rung before the budget made it worse in one direction -- a
    rung 221184 tokens out asks for a chunk no batch can hold even when the
    prompt itself would have fit.
    """

    @staticmethod
    def _order() -> tuple[int, int]:
        """Source offsets of the budget clamp and the cut, in `_schedule`."""
        src = _src(SCHEDULER, "PrefillScheduler", "_schedule")
        return src.index("num_new_tokens = budget"), src.index(
            "num_new_tokens = cut - seq.num_cached_tokens"
        )

    def test_the_budget_is_applied_before_the_rung(self):
        """Order is the fix. The rung is chosen from what this pass can
        afford, never the other way round."""
        budget_at, cut_at = self._order()
        assert budget_at < cut_at

    def test_a_chunk_never_exceeds_the_remaining_budget(self):
        src = _src(SCHEDULER, "PrefillScheduler", "_schedule")
        assert "budget = self.max_num_batched_tokens - num_batched_tokens" in src
        assert "if num_new_tokens > budget:" in src

    def test_an_oversized_prompt_is_partial_rather_than_refused(self):
        """It must leave `waiting` only when finished; a budget-clipped chunk
        is as partial as a rung-clipped one."""
        src = _src(SCHEDULER, "PrefillScheduler", "_schedule")
        clip = src.index("num_new_tokens = budget")
        assert "partial = True" in src[clip : clip + 120]

    def test_the_offset_advances_for_any_partial_chunk(self):
        """Gated on the cut, a budget-clipped chunk of a prompt with no rung
        never advanced -- so every pass re-ran the same first chunk and the
        request never finished."""
        src = _src(CORE, "PrefillEngineCore", "_process_engine_step")
        advance = src.index("seq.num_cached_tokens += int(num_tokens)")
        guard = src.rindex("if seq is not None", 0, advance)
        assert "prefill_cut_pos" not in src[guard:advance]


# ── A partial chunk is KV without a token ────────────────────────────────


def test_a_partial_chunk_stays_in_waiting():
    """It has more prompt to run, and the next pass has to find it again."""
    src = _src(SCHEDULER, "PrefillScheduler", "_schedule")
    assert "if not partial:" in src
    assert "self.waiting.remove(seq)" in src


def test_a_partial_chunk_does_not_promote_to_running():
    """Real KV, no sampled token: promoting it would have decode try to decode
    a request that is still mid-prompt."""
    src = _src(SCHEDULER, "DecodeScheduler", "on_prefill_done")
    assert "if not is_final:" in src
    prem = src.index("if not is_final:")
    assert prem < src.index("self.prefill_waiting.pop")


def test_a_partial_chunk_keeps_its_routing_entry():
    """`_seq_target_rank` is how a later chunk of the same prompt finds its
    decode peer; dropping it on chunk 1 orphans chunk 2."""
    src = _src(CORE, "PrefillEngineCore", "_process_engine_step")
    drop = src.index("self._seq_target_rank.pop(seq_id, None)")
    assert "if is_final:" in src[:drop]


# ── Publishing moved off the postprocess path ────────────────────────────


def test_chunks_publish_before_schedule():
    """Two reasons, both fatal otherwise: a partial chunk never reaches
    `postprocess` (decode runs no forward for it), and a checkpoint filed there
    would be drained a forward later, against a slot that has moved."""
    src = _src(CORE, "DecodeEngineCore", "_process_engine_step")
    pub = src.index("publish_landed_prefill_chunks()")
    assert pub < src.index("self.scheduler.schedule()")


def test_the_publish_uses_the_recorded_offset():
    src = _src(SCHEDULER, "DecodeScheduler", "_publish_prefill_hashes")
    assert "start_tokens=start" in src


@pytest.mark.parametrize("queue", ["prefill_partial"])
def test_landed_chunks_have_their_own_queue(queue):
    """Separate from `prefill_done`, which means 'decodable now'."""
    src = _src(SCHEDULER, "DecodeScheduler", "__init__")
    assert queue in src
