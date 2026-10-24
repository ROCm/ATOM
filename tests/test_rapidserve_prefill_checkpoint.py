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

    def test_arming_happens_per_reported_chunk(self):
        """`PrefillDone` is sent only after the forward's event retires and the
        scatter was enqueued ahead of it, so by then the bytes are real.

        Per chunk, not per prompt: the image a chunk proves is the rung at its
        START, scattered at the head of the pass that produced it. Settling the
        whole ladder on the final chunk instead armed rungs no forward had ever
        ended on -- see `test_a_rung_run_over_is_released_not_settled`."""
        src = _src(SCHEDULER, "DecodeScheduler", "_settle_reported_chunk")
        assert "cp.settle_prefill_store(h)" in src
        assert "if cut == start:" in src

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
        assert seq.prefill_cut_within(1 << 30) == 0

    def test_the_ladder_is_consumed_in_order(self):
        """Each rung is handed out by the pass after the chunk that ends on it,
        which is what makes a prompt crossing many rungs checkpoint at all of
        them instead of one. The cut position tracks the same ladder a pass
        ahead -- see `test_the_cut_skips_a_rung_already_reached`."""
        seq = self._seq(cached=0, cuts=[(1024, "A"), (2048, "B"), (3072, "C")])
        assert seq.prefill_cut_within(1024) == 1024
        assert seq.take_prefill_store_op() is None

        seq.num_cached_tokens = 1024
        assert seq.take_prefill_store_op() == "A"
        assert seq.prefill_cut_within(2048) == 2048, "the next rung is next"
        assert seq.take_prefill_store_op() is None

        seq.num_cached_tokens = 2048
        assert seq.take_prefill_store_op() == "B"
        seq.num_cached_tokens = 3072
        assert seq.take_prefill_store_op() == "C"
        assert seq.prefill_cut_within(1 << 30) == 0, "ladder exhausted"

    def test_a_rung_run_over_is_dropped_not_scattered(self):
        """A chunk takes the furthest rung it can reach, so nearer ones are
        passed over -- and the slot at the end of that chunk holds the state as
        of its END. Scattering a skipped rung's op would file that state under
        a hash meaning somewhere else, and a later request resuming from it
        would restore a position it never asked for. Decode releases the units
        (`on_prefill_done` unindexes them); this end only declines to write."""
        seq = self._seq(cached=4096, cuts=[(1024, "A"), (2048, "B")])
        assert seq.take_prefill_store_op() is None
        assert seq.prefill_cuts == [], "both dropped, neither written"

    def test_the_rung_landed_on_is_still_handed_out(self):
        """Dropping what is behind must not drop what is exactly here."""
        seq = self._seq(cached=2048, cuts=[(1024, "A"), (2048, "B"), (3072, "C")])
        assert seq.take_prefill_store_op() == "B", "A skipped, B landed on"
        assert seq.prefill_cuts == [(3072, "C")]

    def test_the_cut_skips_a_rung_already_reached(self):
        """Standing ON R, the next question is "where after R", not "R".

        The two uses of `prefill_cuts` run one pass apart.
        `take_prefill_store_op` has to hold R for the pass after the chunk that
        reached it, because that is the forward which scatters R's image. The
        cut cannot wait: the chunk being sized right now starts at R. Answering
        it with R made `num_cached_tokens < cut` false, so nothing cut and the
        chunk took the whole budget.
        """
        seq = self._seq(cached=1024, cuts=[(1024, "A"), (2048, "B")])
        assert seq.prefill_cut_within(4096) == 2048
        # The hold is still a hold -- A has not been handed out yet.
        assert seq.prefill_cuts[0] == (1024, "A")
        assert seq.take_prefill_store_op() == "A"

    def test_the_cut_takes_the_furthest_rung_that_fits(self):
        """Baseline parity: `_finalize_prefill_chunk` lets `checkpoint_cut`
        collapse the grid to its rightmost member inside the chunk it was
        already going to run. Taking the nearest instead would shorten every
        chunk to the ladder's spacing and buy nothing -- the rungs in between
        are run over either way."""
        seq = self._seq(cached=0, cuts=[(8192, "A"), (16384, "B"), (24576, "C")])
        assert seq.prefill_cut_within(16384) == 16384, "a full 16384 budget"
        assert seq.prefill_cut_within(20000) == 16384, "no rung past 16384 fits"
        assert seq.prefill_cut_within(8192) == 8192, "half a budget still lands"
        assert seq.prefill_cut_within(4096) == 0, "nothing reachable -> no cut"

    def test_the_cut_rule_this_replay_models_is_still_the_scheduler_s(self):
        """Pin the lines the replay below stands in for, so it cannot drift
        into testing a rule the scheduler stopped using."""
        src = _src(SCHEDULER, "PrefillScheduler", "_schedule")
        assert "reach = seq.num_cached_tokens + num_new_tokens" in src
        assert "cut = seq.prefill_cut_within(reach)" in src
        assert "if cut and cut < reach:" in src
        assert "seq.prefill_landed_on_rung = bool(cut) and (" in src

    def _replay(self, seq, prompt, budget):
        """Drive one prompt through the scheduler's cut rule and the hold.

        Returns `(chunks, reported, scattered)` -- every chunk's size, the end
        offset of the chunks prefill would send a PrefillDone for, and the ops
        it actually wrote.
        """
        chunks, reported, scattered = [], [], []
        while seq.num_cached_tokens < prompt:
            start = seq.num_cached_tokens
            chunk = min(prompt - start, budget)
            cut = seq.prefill_cut_within(start + chunk)
            if cut and cut < start + chunk:
                chunk = cut - start
            on_rung = bool(cut) and cut == start + chunk
            # The scheduler takes the op BEFORE the engine advances the offset,
            # which is what makes the hold one pass long.
            op = seq.take_prefill_store_op()
            if op is not None:
                scattered.append(op)
            seq.num_cached_tokens = start + chunk
            chunks.append(chunk)
            if on_rung or seq.num_cached_tokens >= prompt:
                reported.append(seq.num_cached_tokens)
        return chunks, reported, scattered

    def test_the_prompt_end_anchor_is_not_sailed_past(self):
        """The whole ladder has to be landed on, not just the rungs that happen
        to fall a budget apart.

        Observed before the fix, on prompt 37211 with rungs
        [16384, 32768, 37120]: chunks came out 4891 / 16384 / 4443. The middle
        one hit 32768 by arithmetic luck -- a full budget from 16384 -- and the
        last ran from 32768 clean over the prompt-end anchor at 37120, which is
        the rung a later turn of the same conversation resumes from. Each
        chunk's successor started exactly on a rung, which is the one state
        that used to read its cut as the position it was already at.
        """
        seq = self._seq(cached=11493, cuts=[(16384, "A"), (32768, "B"), (37120, "C")])
        seq.num_tokens = 37211
        chunks, reported, scattered = self._replay(seq, 37211, 16384)

        assert chunks == [4891, 16384, 4352, 91]
        for rung in (16384, 32768, 37120):
            assert rung in reported, f"no forward ended on rung {rung}"
        assert scattered == ["A", "B", "C"], "every rung's image was written"

    def test_an_interval_ladder_does_not_halve_every_chunk(self):
        """The whole point of pairing the finer ladder with the furthest-rung
        cut: at 8192 spacing and a 16384 budget, a chunk takes a full budget
        whenever a rung lands there. Taking the NEAREST rung instead would make
        every chunk 8192, for rungs that are run over either way.

        The ladder stops short of the prompt end: `_prepare_prefill_checkpoint`
        refuses a rung at `end`, which nothing could scatter.
        """
        seq = self._seq(cached=0, cuts=[(8192, "A"), (16384, "B"), (24576, "C")])
        seq.num_tokens = 32768
        chunks, reported, scattered = self._replay(seq, 32768, 16384)

        # 0 -> 16384 is a whole budget, skipping 8192. From there the furthest
        # rung that fits is 24576, so the tail splits -- the finer ladder buys
        # a resume point at 24576 that the 16384-spaced one had no candidate
        # for, and pays one extra forward for it.
        assert chunks == [16384, 8192, 8192]
        assert reported == [16384, 24576, 32768]
        # A was run over by chunk 1 and dropped by the pass after it; B and C
        # are rungs a forward ended on, so both images were written.
        assert scattered == ["B", "C"]

    def test_a_chunk_reaching_no_rung_is_not_reported(self):
        """A leftover-budget crumb writes real KV but produces no image, so
        there is nothing for decode to settle -- and `pending_hash_start`
        already spans the gap, so its blocks are published with the next
        rung's message."""
        seq = self._seq(cached=0, cuts=[(8192, "A")])
        seq.num_tokens = 8192
        chunks, reported, _ = self._replay(seq, 8192, 85)
        assert chunks[0] == 85
        assert reported == [8192], "only the rung, not the crumbs before it"


# ── Two fates for a rung, decided by one chunk's boundaries ──────────────


class TestSettleReportedChunk:
    """`_settle_reported_chunk` is the only thing that arms a record or gives
    its units back, so every rung's fate is decided here.

    Driven unbound against a stub coordinator: the method reads nothing off
    `self` but `block_manager.paged_state_checkpoints`, and building a real
    `DecodeScheduler` would drag in a block pool and a KV cache for a rule that
    is four lines of arithmetic.
    """

    class _Coordinator:
        def __init__(self):
            self.settled, self.released = [], []

        def settle_prefill_store(self, h):
            self.settled.append(h)

        def unindex(self, h):
            self.released.append(h)

    @staticmethod
    def _call(seq, start, end, cp):
        from types import SimpleNamespace

        from atom.model_engine.scheduler import DecodeScheduler

        stub = SimpleNamespace(
            block_manager=SimpleNamespace(paged_state_checkpoints=cp)
        )
        DecodeScheduler._settle_reported_chunk(stub, seq, start, end)

    @staticmethod
    def _seq(cuts, hashes):
        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        seq = Sequence([1, 2, 3, 4], 4, sampling_params=SamplingParams())
        seq.prefill_cuts = [(c, f"op{c}") for c in cuts]
        seq.prefill_store_hashes = list(hashes)
        return seq

    def test_the_rung_at_the_start_is_settled(self):
        """Its image was scattered at the head of the pass that produced this
        chunk, and PrefillDone waits for that pass's event -- so the bytes are
        on the device by the time this runs. That is the per-hash evidence
        `mark_inflight` asks for."""
        cp = self._Coordinator()
        seq = self._seq([8192, 16384], ["hA", "hB"])
        self._call(seq, start=8192, end=16384, cp=cp)
        assert cp.settled == ["hA"]
        assert cp.released == []

    def test_a_rung_run_over_has_its_units_released(self):
        """Nothing will ever be written there -- the slot holds the state as of
        `end`. Holding the reservation for the rest of the prefill buys nothing
        and the pool is the scarce thing."""
        cp = self._Coordinator()
        seq = self._seq([8192, 16384, 24576], ["hA", "hB", "hC"])
        self._call(seq, start=0, end=24576, cp=cp)
        assert cp.settled == [], "0 is not a rung on this ladder"
        assert cp.released == ["hA", "hB"], "both run over; 24576 is the end"

    def test_the_rung_at_the_end_is_left_alone(self):
        """It is the next chunk's start, and that chunk's message settles it.
        Settling it now would publish before its scatter has even been issued
        -- the exact thing the unarmed reservation exists to prevent."""
        cp = self._Coordinator()
        seq = self._seq([8192, 16384], ["hA", "hB"])
        self._call(seq, start=0, end=16384, cp=cp)
        assert cp.settled == []
        assert cp.released == ["hA"]
        assert "hB" not in cp.released

    def test_a_prompt_with_no_ladder_touches_nothing(self):
        cp = self._Coordinator()
        seq = self._seq([], [])
        self._call(seq, start=0, end=16384, cp=cp)
        assert cp.settled == [] and cp.released == []

    def test_no_coordinator_is_not_an_error(self):
        """Every prompt takes this path; a model without per-request state has
        no coordinator at all."""
        seq = self._seq([8192], ["hA"])
        self._call(seq, start=0, end=16384, cp=None)


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

    def test_the_walk_steps_one_checkpoint_interval(self):
        """The window width IS the ladder's spacing.

        `checkpoint_cut` answers with the rightmost grid point inside the
        window, so a window of `max_num_batched_tokens` produced a rung every
        16384 while the interval was 8192 -- half the resume points the grid
        defines. Stepping the interval enumerates all of them.
        """
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert "interval = bm.state_checkpoint_interval_tokens" in src
        assert "min(pos + step, end)" in src
        assert "while pos < end:" in src

    def test_the_walk_falls_back_when_the_grid_is_off(self):
        """`interval <= 0` places no grid rung at all (-1 keeps checkpoints
        without a grid, 0 disables them). Only a demand or the prompt-end
        anchor can cut then, and neither cares how wide the window is -- so
        step a batch rather than walking the prompt a token at a time."""
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert (
            "step = interval if interval > 0 else max(1, self.max_num_batched_tokens)"
            in src
        )

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

    def test_the_ladder_is_not_capped(self):
        """Every rung becomes a cut, as on the non-disagg path.

        The ladder used to be trimmed to `MAX_PREFILL_CHECKPOINTS` keeping the
        tail, because rapidserve reserves the WHOLE ladder at admission (prefill
        has no BlockManager and `BLOCK_ASSIGNMENT` is the only decode->prefill
        message), so each reservation pinned units for the entire prefill.
        Baseline has no such cap -- it reserves one image at a time, per chunk --
        so the two produced different cut positions on any prompt crossing more
        than eight rungs. Parity won that trade.
        """
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert "MAX_PREFILL_CHECKPOINTS" not in src
        assert "prefill_cuts[dropped:]" not in src, (
            "the trim is what made the ladder diverge from baseline's"
        )

    def test_unreservable_rungs_are_reported(self):
        """With no cap, a full state pool is the only thing left that can still
        shorten this ladder below baseline's -- and it does so silently:
        `reserve_prefill_checkpoint` returns `(0, None)`, the rung is skipped,
        and the prompt is never cut there. Counted and logged so a parity gap
        is visible instead of inferred."""
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        assert "unreserved += 1" in src
        assert "unreserved" in src and "logger.warning" in src

    def test_a_rung_at_the_prompt_end_is_not_reserved(self):
        """Prefill cannot write it, and decode already does.

        An image is scattered by the pass AFTER the chunk ending on its rung,
        and a rung at the prompt's last token has no such pass -- that chunk is
        the prompt's last and the sequence leaves this scheduler.

        Nothing is lost by refusing it. `Scheduler.hash_blocks` ends every
        chunk with `checkpointers_at(pos)`; at the prompt end
        `next_forward_tokens` defaults to the remainder (0) and the PAGE
        coordinator's `successor_room` is 0.0, so decode files the boundary
        from `_publish_prefill_hashes` and scatters it out of the slot prefill
        left behind -- the same way the non-disagg path keeps a prompt-end
        image. Reserving it here too would be a second reservation for one
        hash, written by nobody.

        `checkpoint_cut` admits `p == end` and the walk's `end` is the prompt,
        so this fires whenever the last rung and the prompt end coincide: a
        32768-token prompt with an 8192 interval lands on it exactly.
        """
        src = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        guard = src.index("if cut >= end:")
        assert guard < src.index("bm.reserve_prefill_checkpoint(seq, cut)"), (
            "the rung has to be refused BEFORE it reserves units, or they stay "
            "pinned for a whole prefill against an image nothing will write"
        )

    def test_the_prompt_end_is_still_reachable_through_the_ordinary_path(self):
        """The guard above is only safe because this is true: decode's own
        publish runs the same `checkpointers_at` call baseline does, with the
        defaults that admit a zero-remainder position."""
        src = _src(SCHEDULER, "DecodeScheduler", "_publish_prefill_hashes")
        assert "self.block_manager.hash_blocks(seq, num_new, start_tokens=start)" in src
        hb = _src(BLOCK_MANAGER, "BlockManager", "hash_blocks")
        assert (
            "kept = self.checkpointers_at(seq, pos, next_forward_tokens, aimed)" in hb
        )
        ca = _src(BLOCK_MANAGER, "BlockManager", "checkpointers_at")
        assert "next_forward_tokens = seq.num_prompt_tokens - pos" in ca
        assert "next_forward_tokens >= c.successor_room" in ca

    def test_a_rung_run_over_is_released_not_settled(self):
        """The other half of the furthest-rung cut.

        A chunk takes the last rung it can reach, so nearer ones are passed
        over routinely. Prefill declines to scatter them
        (`take_prefill_store_op` drops what is behind), but only decode can
        give the units back -- and settling them instead, which the wholesale
        loop this replaced did, publishes a resume point over bytes nobody
        wrote.
        """
        src = _src(SCHEDULER, "DecodeScheduler", "_settle_reported_chunk")
        assert "elif start < cut < end:" in src
        assert "cp.unindex(h)" in src
        # And the wholesale settle is gone, not merely supplemented.
        done = _src(SCHEDULER, "DecodeScheduler", "on_prefill_done")
        assert "for h in seq.prefill_store_hashes:" not in done

    def test_the_settle_pairs_cuts_with_hashes_positionally(self):
        """`prefill_cuts` and `prefill_store_hashes` are appended together in
        `_prepare_prefill_checkpoint`; a `zip` is only correct while that
        holds."""
        walk = _src(SCHEDULER, "DecodeScheduler", "_prepare_prefill_checkpoint")
        append_cut = walk.index("seq.prefill_cuts.append((cut, op))")
        append_hash = walk.index("seq.prefill_store_hashes.append(h)")
        assert append_cut < append_hash
        src = _src(SCHEDULER, "DecodeScheduler", "_settle_reported_chunk")
        assert "zip(seq.prefill_cuts, seq.prefill_store_hashes)" in src


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

    def test_the_offset_advances_even_for_an_unreported_chunk(self):
        """Gated on the cut, a budget-clipped chunk of a prompt with no rung
        never advanced -- so every pass re-ran the same first chunk and the
        request never finished.

        The rung now gates the MESSAGE instead, which puts the same trap one
        line further down: a chunk prefill stays silent about still has to move
        prefill's own offset, or a sequence living on leftover-budget crumbs
        never reaches a rung to be silent about.
        """
        src = _src(CORE, "PrefillEngineCore", "_process_engine_step")
        advance = src.index("seq.num_cached_tokens += int(num_tokens)")
        gate = src.index("if not (is_final or seq.prefill_landed_on_rung):")
        assert advance < gate


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
