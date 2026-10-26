# SPDX-License-Identifier: MIT
"""SWA bounded replay on the engine side: a `replay` state transfer keeps no
checkpoint and accepts any hit, and a prefill resuming somewhere starts early
enough to rebuild what its forward reads below that point
(`BlockManager.replay_start`, `Scheduler._resume_prefill`)."""

import math
from types import SimpleNamespace

import pytest

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Replay, Sequence
from atom.model_engine.state_pool import StateSlotPool
from atom.model_engine.state_runtime import StateRuntime, StateTransfer


def test_a_replay_transfer_round_trips_and_keeps_nothing():
    transfer = StateTransfer.replay()
    assert transfer.replays and not transfer.copies and not transfer.forks
    assert transfer.successor_room == float("inf")
    runtime = StateRuntime(transfer=transfer)
    assert StateRuntime.from_wire(runtime.to_wire()) == runtime
    with pytest.raises(ValueError):
        StateTransfer("replay", paged_layout_id="layout")


def test_a_replay_pool_resumes_any_hit():
    pool = StateSlotPool(4, transfer=StateTransfer.replay(), hash_block_size=128)
    seq = SimpleNamespace(has_per_req_cache=True, num_tokens=5000)
    assert pool.resumable_hit(seq, 7, [0] * 7) == 7
    assert pool.resumable_hit(seq, 1, [0]) == 1
    assert pool.resumable_hit(seq, 0, []) == 0


def _manager(transfer, min_final_rows=0, replay_tokens=0):
    manager = SimpleNamespace(
        state=SimpleNamespace(transfer=transfer),
        hash_block_size=128,
        min_final_prefill_rows=min_final_rows,
        state_replay_tokens=replay_tokens,
    )
    manager.replay_start = lambda seq, start, hit: BlockManager.replay_start(
        manager, seq, start, hit=hit
    )
    return manager


def _seq(tokens, cached=0):
    return SimpleNamespace(
        id=0, has_per_req_cache=True, num_tokens=tokens, num_cached_tokens=cached
    )


def test_a_replay_transfer_hit_rebuilds_its_state_from_one_block_below():
    manager = _manager(StateTransfer.replay())
    assert manager.replay_start(_seq(5000), 1792, hit=True) == 1664
    # a chunk boundary carries its state along: nothing to rebuild
    assert manager.replay_start(_seq(5000), 1792, hit=False) == 1792
    assert manager.replay_start(_seq(5000), 0, hit=True) == 0


def test_a_short_last_forward_backs_off_one_block_to_its_window():
    manager = _manager(StateTransfer.copy("layout"), min_final_rows=128)
    # 1000 rows left, or exactly a window: nothing to recompute
    assert manager.replay_start(_seq(17384), 16384, hit=False) == 16384
    assert manager.replay_start(_seq(16512), 16384, hit=False) == 16384
    # 100 rows left: one block back, 228 rows
    assert manager.replay_start(_seq(16484), 16384, hit=False) == 16256
    assert manager.replay_start(_seq(16385), 16384, hit=True) == 16256
    # never below the prompt's start
    assert manager.replay_start(_seq(200), 128, hit=False) == 0
    # a replay transfer's hit backs off its one block however short the suffix
    replay = _manager(StateTransfer.replay(), min_final_rows=128)
    assert replay.replay_start(_seq(16385), 16384, hit=True) == 16256
    assert replay.replay_start(_seq(20000), 16384, hit=True) == 16256


def test_a_dsv4_bounded_replay_hit_rebuilds_its_state_from_its_field_below():
    # `can_allocate` admitted the hit at 9984 with no checkpoint behind it
    manager = _manager(StateTransfer.copy("layout"), replay_tokens=7875)
    seq = _seq(20000)
    seq.replays_hit = True
    assert manager.replay_start(seq, 9984, hit=True) == 2109
    assert manager.replay_start(seq, 4096, hit=True) == 0
    # a chunk boundary carries its state along: nothing to rebuild
    assert manager.replay_start(seq, 9984, hit=False) == 9984
    # a hit restored from its checkpoint
    seq.replays_hit = False
    assert manager.replay_start(seq, 9984, hit=True) == 9984


def test_a_resume_off_the_block_grid_backs_off_onto_it():
    """A chunk cut off the hash-block grid (a long-prefill threshold that is no
    multiple of it) still backs off: to the grid, at least its window."""
    manager = _manager(StateTransfer.copy("layout"), min_final_rows=128)
    assert manager.replay_start(_seq(20000), 16300, hit=False) == 16300
    # 100 rows left: back to the grid below 16272
    assert manager.replay_start(_seq(16400), 16300, hit=False) == 16256
    assert manager.replay_start(_seq(2050), 2000, hit=False) == 1920


def test_resuming_records_where_the_forward_resumed():
    scheduler = SimpleNamespace(
        block_manager=SimpleNamespace(
            state=SimpleNamespace(transfer=StateTransfer.replay())
        ),
        replay_history_tokens=3,
    )

    def seq(tokens, cached):
        return SimpleNamespace(
            num_cached_tokens=cached, token_ids=list(range(tokens)), replay=None
        )

    s = seq(16584, 16384)
    Scheduler._resume_prefill(scheduler, s, 16256, hit=True)
    assert s.num_cached_tokens == 16256
    assert s.replay == Replay(16384, True, (16253, 16254, 16255), 16256)
    # a chunk boundary keeps its state: the rows below are not the state
    s = seq(16584, 16384)
    Scheduler._resume_prefill(scheduler, s, 16256, hit=False)
    assert s.replay.until == 16384 and not s.replay.rebuilds
    s = seq(20000, 16384)
    Scheduler._resume_prefill(scheduler, s, 16384, hit=True)
    assert (s.num_cached_tokens, s.replay) == (16384, None)


def test_a_backed_off_chunk_has_to_pass_where_it_resumed():
    """The long-prefill cap counts from where the prefill resumed, the rows a
    back-off recomputes riding on top; a chunk that still cannot pass that
    point (a full budget) waits, since one ending there only recomputes cached
    rows and hands the next pass the same back-off, forever."""
    scheduler = SimpleNamespace(
        long_prefill_token_threshold=128,
        max_num_batched_tokens=16384,
        block_manager=SimpleNamespace(block_size=128),
        _tick_cost_limit=math.inf,  # no DP prefill cost balancing
        _dynamic_chunk_limit=lambda history_len, kind: None,
    )
    scheduler._chunk_alignment = lambda: Scheduler._chunk_alignment(scheduler)
    scheduler._chunked_prefill_size = lambda *a, **kw: Scheduler._chunked_prefill_size(
        scheduler, *a, **kw
    )
    scheduler._prefill_room = lambda *a: Scheduler._prefill_room(scheduler, *a)
    seq = SimpleNamespace(num_tokens=16484)
    chunk = Scheduler._partial_prefill_chunk
    # backed off one block: 128 recomputed rows plus the 100 left
    assert chunk(scheduler, seq, 16256, 16384, 0) == 228
    # a budget with room for the recomputed rows alone
    assert chunk(scheduler, seq, 16256, 16384, 16384 - 128) == 0
    # no back-off: the cap as before
    assert chunk(scheduler, SimpleNamespace(num_tokens=20000), 16384, 16384, 0) == 128


def test_a_backed_off_forward_places_checkpoints_from_where_it_resumed():
    """A forward recomputing rows below where it resumed must not be cut at or
    below that point: the state is there already, and a cut there hands the
    next pass the same resume, which backs off again, forever."""
    calls = []

    def checkpoint_cut(seq, start, end, record=True):
        calls.append((start, end))
        return 0

    scheduler = SimpleNamespace(
        block_manager=SimpleNamespace(
            checkpoint_cut=checkpoint_cut,
            # not a midstep-readable class: an uncut chunk keeps its length
            align_midstep_chunk=lambda seq, start, chunk: chunk,
        ),
        _requires_atomic_prefill=lambda seq: False,
    )
    seq = SimpleNamespace(replay_until=6784, replays_hit=False, state_fork_src=-1)
    assert Scheduler._finalize_prefill_chunk(scheduler, seq, 6528, 272) == 272
    assert Scheduler._preview_prefill_chunk(scheduler, seq, 6528, 6784, 272) == 272
    assert calls == [(6784, 6800), (6784, 6800)]


def test_a_batch_carries_only_the_forwards_starting_below_their_resume():
    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import SequenceType

    def prefill(cached, replay=None):
        seq = Sequence(list(range(1, 301)), block_size=128)
        seq.type = SequenceType.PREFILL
        seq.num_cached_tokens, seq.replay = cached, replay
        return seq

    replay = Replay(256, True, (126, 127, 128), 128)
    # a rebuilt hit, a cold prefill, and one whose resume was already passed
    seqs = [prefill(128, replay), prefill(0), prefill(256, replay._replace())]
    batch = ScheduledBatch(
        seqs={s.id: s for s in seqs},
        num_scheduled_tokens=[172, 300, 44],
        total_tokens_num=516,
        total_seqs_num=3,
    )
    assert batch.replaying == {0: replay}
