# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The in-process LMCache tier's chunks in the engine's KV events (`medium=CPU`):
announced once every rank stored them, retracted when a lookup asked after the
announcement finds them gone, or past the cap."""

from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.distributed.kv_events import MEDIUM_CPU, BlockRemoved, BlockStored
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.tier_events import TierEvents
from atom.model_engine.block_manager import BlockManager

# Two 4-token hash blocks per chunk.
CHUNK = 8


def _bm(enable=True, block_size=4):
    cfg = MockConfig(
        enable_prefix_caching=True,
        kv_cache_block_size=block_size,
        num_kvcache_blocks=8,
    )
    cfg.kv_events_config = SimpleNamespace(enable=enable)
    return BlockManager(cfg)


def _chunk_hashes(bm, seq, chunks):
    """The prefix hash of each chunk's last hash block."""
    chain = bm.prefix_hash_chain(seq, [], chunks * CHUNK // bm.hash_block_size)
    return chain[CHUNK // bm.hash_block_size - 1 :: CHUNK // bm.hash_block_size]


def test_a_stored_save_announces_its_whole_chunks(seq_factory):
    bm = _bm()
    seq = seq_factory(list(range(20)))
    TierEvents(bm, CHUNK).stored(seq, 0, 20)
    (event,) = bm.take_events()
    assert isinstance(event, BlockStored)
    assert event.medium == MEDIUM_CPU
    assert event.block_size == CHUNK
    # The partial third chunk is not one the tier serves.
    assert event.block_hashes == _chunk_hashes(bm, seq, 2)
    assert event.token_ids == list(range(16))
    assert event.parent_block_hash is None
    assert event.token_offset == 0


@pytest.mark.parametrize("start", [8, 4], ids=["aligned", "inside-a-chunk"])
def test_a_later_save_continues_the_chain(seq_factory, start):
    bm = _bm()
    seq = seq_factory(list(range(24)))
    TierEvents(bm, CHUNK).stored(seq, start, 24)
    (event,) = bm.take_events()
    first, second, third = _chunk_hashes(bm, seq, 3)
    assert event.block_hashes == [second, third]
    assert event.parent_block_hash == first
    assert event.token_ids == list(range(8, 24))
    assert event.token_offset == 8


def test_a_chunk_hash_names_its_whole_prefix(seq_factory):
    bm = _bm()
    a = seq_factory(list(range(16)))
    b = seq_factory(list(range(8)) + [99] * 8)
    assert _chunk_hashes(bm, a, 2)[0] == _chunk_hashes(bm, b, 2)[0]
    assert _chunk_hashes(bm, a, 2)[1] != _chunk_hashes(bm, b, 2)[1]


def test_a_save_short_of_a_chunk_announces_nothing(seq_factory):
    bm = _bm()
    TierEvents(bm, CHUNK).stored(seq_factory(list(range(20))), 9, 15)
    assert bm.take_events() == []


def test_a_lookup_that_stops_short_retracts_what_was_announced_before(seq_factory):
    bm = _bm()
    tier = TierEvents(bm, CHUNK)
    seq = seq_factory(list(range(24)))
    tier.stored(seq, 0, 24)
    bm.take_events()
    tier.looked_up(seq, 24, 8, asked=float("inf"))
    (event,) = bm.take_events()
    assert isinstance(event, BlockRemoved)
    assert event.medium == MEDIUM_CPU
    assert event.block_hashes == _chunk_hashes(bm, seq, 3)[1:]
    # Retracted once.
    tier.looked_up(seq, 24, 8, asked=float("inf"))
    assert bm.take_events() == []


def test_a_lookup_asked_before_the_announcement_retracts_nothing(seq_factory):
    bm = _bm()
    tier = TierEvents(bm, CHUNK)
    seq = seq_factory(list(range(16)))
    tier.stored(seq, 0, 16)
    bm.take_events()
    tier.looked_up(seq, 16, 0, asked=float("-inf"))
    assert bm.take_events() == []


def test_a_lookup_retracts_nothing_past_its_extent(seq_factory):
    bm = _bm()
    tier = TierEvents(bm, CHUNK)
    seq = seq_factory(list(range(16)))
    tier.stored(seq, 0, 16)
    bm.take_events()
    tier.looked_up(seq, 8, 8, asked=float("inf"))
    assert bm.take_events() == []


def test_past_the_cap_the_oldest_chunks_are_retracted(seq_factory):
    bm = _bm()
    tier = TierEvents(bm, CHUNK, max_chunks=2)
    seq = seq_factory(list(range(24)))
    tier.stored(seq, 0, 16)
    tier.stored(seq, 16, 24)
    stored, removed = bm.take_events()[-2:]
    assert isinstance(stored, BlockStored) and isinstance(removed, BlockRemoved)
    assert removed.block_hashes == _chunk_hashes(bm, seq, 1)


def test_the_tier_publishes_only_with_events_and_whole_hash_blocks():
    assert TierEvents.for_tier(None, CHUNK) is None
    assert TierEvents.for_tier(_bm(enable=False), CHUNK) is None
    assert TierEvents.for_tier(_bm(block_size=16), CHUNK) is None
    assert isinstance(TierEvents.for_tier(_bm(), CHUNK), TierEvents)


# ── through the scheduler ─────────────────────────────────────────────────


class _Tier:
    """A lookup client whose tier holds the first `held` tokens of a prompt."""

    def __init__(self):
        self.held = 0
        self.started = None

    def lookup(self, token_ids, lookup_id=None, **_):
        return min(self.held, len(token_ids))

    def query_started(self, lookup_id):
        return self.started

    def close(self):
        pass


def _scheduler(bm, publish=True):
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "offload"},
        kv_cache_block_size=4,
        decode_context_parallel_size=1,
        offload_min_load_tokens=0,
    )
    tier = _Tier()
    sched = ChunkedOffloadSchedulerBase(config, chunk_size=CHUNK, lookup_client=tier)
    sched.bind_block_manager(bm)
    if publish:
        sched.publish_tier_events()
    return sched, tier


def _emit_save(sched, seq):
    """Emit `seq`'s save of its computed prompt chunks; its operation."""
    seq.num_cached_tokens = seq.num_prompt_tokens
    sched._save_tracker[str(seq.id)] = [seq, 0]
    sched.build_connector_meta()
    return sched._save_inflight[str(seq.id)]


def test_the_scheduler_announces_a_save_once_every_rank_stored_it(seq_factory):
    bm = _bm()
    sched, _ = _scheduler(bm)
    seq = seq_factory(list(range(20)))
    operation = _emit_save(sched, seq)
    assert bm.take_events() == []
    sched.save_finished(operation)
    (event,) = bm.take_events()
    assert event.medium == MEDIUM_CPU
    assert event.block_hashes == _chunk_hashes(bm, seq, 2)
    assert sched._tier_saves == {}


def test_the_scheduler_announces_no_save_it_dropped(seq_factory):
    bm = _bm()
    sched, _ = _scheduler(bm)
    operation = _emit_save(sched, seq_factory(list(range(16))))
    sched._cancel_save_statistics(operation)
    assert sched._tier_saves == {}
    sched.save_finished(operation)
    assert bm.take_events() == []


def test_a_scheduler_lookup_retracts_chunks_the_tier_lost(seq_factory):
    bm = _bm()
    sched, tier = _scheduler(bm)
    sched.save_finished(_emit_save(sched, seq_factory(list(range(16)))))
    bm.take_events()
    again = seq_factory(list(range(16)) + [7, 7])
    tier.held = 8
    assert sched._fresh_tier_lookup(again, str(again.id)) == 8
    (event,) = bm.take_events()
    assert isinstance(event, BlockRemoved)
    assert event.block_hashes == _chunk_hashes(bm, again, 2)[1:]


def test_an_answer_older_than_the_announcement_retracts_nothing(seq_factory):
    bm = _bm()
    sched, tier = _scheduler(bm)
    sched.save_finished(_emit_save(sched, seq_factory(list(range(16)))))
    bm.take_events()
    again = seq_factory(list(range(16)))
    # The asynchronous lookup behind this answer was sent before the save.
    tier.started = float("-inf")
    sched._fresh_tier_lookup(again, str(again.id))
    assert bm.take_events() == []


def test_a_scheduler_without_tier_events_publishes_none(seq_factory):
    bm = _bm()
    sched, _ = _scheduler(bm, publish=False)
    sched.save_finished(_emit_save(sched, seq_factory(list(range(16)))))
    sched._fresh_tier_lookup(seq_factory(list(range(16))), "x")
    assert bm.take_events() == []
