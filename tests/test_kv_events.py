# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for atom/distributed/kv_events.py and the BlockManager hooks.

Covers:
  * Event schema + msgspec round-trip
  * BlockManager emits BlockStored on first store / suppresses on cache hit
  * BlockManager emits BlockRemoved on lazy eviction
  * `take_events()` drain semantics
  * `clear_cache()` emits AllBlocksCleared
  * `record_remote_store()` emits BlockStored(medium=REMOTE)
  * ZmqEventPublisher PUB→SUB round-trip
  * NullEventPublisher is a no-op
"""

from __future__ import annotations

import time

import msgspec
import pytest

from atom.distributed.kv_events import (
    MEDIUM_GPU,
    MEDIUM_REMOTE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockTransferred,
    EventBatch,
    NullEventPublisher,
    ZmqEventPublisher,
    make_publisher,
)
from atom.model_engine.block_manager import BlockManager
from conftest import MockConfig


# ── helpers ───────────────────────────────────────────────────────────────


def _bm_with_events(**overrides) -> BlockManager:
    """BlockManager wired up with KV events enabled."""

    class _KVEventsConfig:
        enable = True

    kwargs = dict(
        enable_prefix_caching=True,
        kv_cache_block_size=4,
        num_kvcache_blocks=8,
    )
    kwargs.update(overrides)
    cfg = MockConfig(**kwargs)
    cfg.kv_events_config = _KVEventsConfig()
    return BlockManager(cfg)


# ── schema / msgspec round-trip ───────────────────────────────────────────


class TestEventSchema:
    def test_block_stored_roundtrip(self):
        evt = BlockStored(
            block_hashes=[111, 222],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            block_size=4,
            medium=MEDIUM_GPU,
        )
        enc = msgspec.msgpack.Encoder().encode(evt)
        dec = msgspec.msgpack.Decoder(BlockStored).decode(enc)
        assert dec.block_hashes == [111, 222]
        assert dec.parent_block_hash is None
        assert dec.medium == MEDIUM_GPU
        assert dec.block_size == 4

    def test_block_removed_roundtrip(self):
        evt = BlockRemoved(block_hashes=[111], medium=MEDIUM_GPU)
        enc = msgspec.msgpack.Encoder().encode(evt)
        dec = msgspec.msgpack.Decoder(BlockRemoved).decode(enc)
        assert dec.block_hashes == [111]

    def test_all_blocks_cleared_roundtrip(self):
        evt = AllBlocksCleared()
        enc = msgspec.msgpack.Encoder().encode(evt)
        dec = msgspec.msgpack.Decoder(AllBlocksCleared).decode(enc)
        assert dec.medium is None

    def test_block_transferred_roundtrip(self):
        evt = BlockTransferred(
            block_hashes=[1, 2, 3],
            from_medium=MEDIUM_GPU,
            to_medium="CPU",
        )
        enc = msgspec.msgpack.Encoder().encode(evt)
        dec = msgspec.msgpack.Decoder(BlockTransferred).decode(enc)
        assert dec.from_medium == MEDIUM_GPU
        assert dec.to_medium == "CPU"

    def test_event_batch_tagged_union(self):
        batch = EventBatch(
            ts=time.time(),
            events=[
                BlockStored(
                    block_hashes=[1],
                    parent_block_hash=None,
                    token_ids=[1, 2, 3, 4],
                    block_size=4,
                ),
                BlockRemoved(block_hashes=[2]),
                AllBlocksCleared(),
            ],
            data_parallel_rank=0,
        )
        enc = msgspec.msgpack.Encoder().encode(batch)
        dec = msgspec.msgpack.Decoder(EventBatch).decode(enc)
        assert len(dec.events) == 3
        assert isinstance(dec.events[0], BlockStored)
        assert isinstance(dec.events[1], BlockRemoved)
        assert isinstance(dec.events[2], AllBlocksCleared)


# ── BlockManager hooks ─────────────────────────────────────────────────────


def _admit(bm: BlockManager, seq):
    """allocate(seq, num_cached) + hash_blocks() — mirrors the scheduler."""
    n = bm.can_allocate(seq)
    if n < 0:
        raise AssertionError("no admission for seq")
    bm.allocate(seq, n)
    num_new_tokens = (seq.num_blocks - n) * bm.block_size
    bm.hash_blocks(seq, num_new_tokens)


class TestBlockManagerHooks:
    def test_disabled_no_overhead(self, block_manager_prefix, seq_factory):
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(block_manager_prefix, seq)
        assert block_manager_prefix.take_events() == []

    def test_block_stored_on_first_allocate(self, seq_factory):
        bm = _bm_with_events()
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(bm, seq)
        events = bm.take_events()
        stored = [e for e in events if isinstance(e, BlockStored)]
        assert len(stored) == 1
        assert stored[0].block_size == 4
        assert stored[0].medium == MEDIUM_GPU

    def test_drain_is_destructive(self, seq_factory):
        bm = _bm_with_events()
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(bm, seq)
        first = bm.take_events()
        second = bm.take_events()
        assert first
        assert second == []

    def test_cache_hit_emits_no_new_store(self, seq_factory):
        bm = _bm_with_events()
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(bm, s1)
        first = bm.take_events()
        first_stored = [e for e in first if isinstance(e, BlockStored)]
        assert len(first_stored) == 1
        first_hashes = first_stored[0].block_hashes

        s2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(bm, s2)
        events = bm.take_events()
        stored = [e for e in events if isinstance(e, BlockStored)]
        assert len(stored) == 1
        assert stored[0].parent_block_hash == first_hashes[0]

    def test_eviction_emits_block_removed(self, seq_factory):
        # Pool with a single block so the free FIFO has no choice but to
        # recycle the block that still carries s1's stale hash → eviction.
        bm = _bm_with_events(num_kvcache_blocks=1, kv_cache_block_size=4)
        s1 = seq_factory([1, 2, 3, 4])
        _admit(bm, s1)
        bm.deallocate(s1)
        bm.take_events()

        s2 = seq_factory([9, 9, 9, 9])
        _admit(bm, s2)
        events = bm.take_events()
        removed = [e for e in events if isinstance(e, BlockRemoved)]
        assert removed, f"expected BlockRemoved on eviction, got: {events}"
        assert removed[0].medium == MEDIUM_GPU

    def test_cache_hit_reuse_does_not_emit_block_removed(self, seq_factory):
        bm = _bm_with_events(num_kvcache_blocks=8, kv_cache_block_size=4)
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(bm, s1)
        bm.deallocate(s1)
        bm.take_events()

        s2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(bm, s2)
        events = bm.take_events()
        removed = [e for e in events if isinstance(e, BlockRemoved)]
        assert removed == [], f"cache hit must not emit BlockRemoved, got: {events}"

    def test_clear_cache_emits_all_cleared(self, seq_factory):
        bm = _bm_with_events()
        s1 = seq_factory([1, 2, 3, 4])
        _admit(bm, s1)
        bm.deallocate(s1)
        bm.take_events()

        bm.clear_cache()
        events = bm.take_events()
        cleared = [e for e in events if isinstance(e, AllBlocksCleared)]
        assert len(cleared) == 1

    def test_clear_cache_drops_hash_index(self, seq_factory):
        bm = _bm_with_events()
        s1 = seq_factory([1, 2, 3, 4])
        _admit(bm, s1)
        bm.deallocate(s1)
        assert bm.hash_to_block_id, "preconditions: hash should be cached"
        bm.clear_cache()
        assert bm.hash_to_block_id == {}

    def test_record_remote_store(self, seq_factory):
        bm = _bm_with_events()
        bm.record_remote_store(
            block_hashes=[42, 43],
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            parent_block_hash=None,
        )
        events = bm.take_events()
        assert len(events) == 1
        assert isinstance(events[0], BlockStored)
        assert events[0].medium == MEDIUM_REMOTE
        assert events[0].block_hashes == [42, 43]

    def test_record_remote_store_no_op_when_disabled(self, block_manager):
        # block_manager fixture has events disabled
        block_manager.record_remote_store(block_hashes=[1], token_ids=[0])
        assert block_manager.take_events() == []


# ── Publisher ──────────────────────────────────────────────────────────────


class TestPublisher:
    def test_null_publisher_is_no_op(self):
        pub = NullEventPublisher()
        pub.publish([BlockRemoved(block_hashes=[1])])
        pub.shutdown()

    def test_make_publisher_disabled_returns_null(self):
        pub = make_publisher(enabled=False, publisher_kind="zmq", endpoint="tcp://*:0")
        assert isinstance(pub, NullEventPublisher)

    def test_make_publisher_null_kind_returns_null(self):
        pub = make_publisher(enabled=True, publisher_kind="null", endpoint="")
        assert isinstance(pub, NullEventPublisher)

    def test_make_publisher_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            make_publisher(enabled=True, publisher_kind="kafka", endpoint="")

    def test_zmq_publisher_roundtrip(self):
        # Skip cleanly when pyzmq isn't installed (zmq is an optional dep of
        # the publisher, not of the engine).
        zmq = pytest.importorskip("zmq")

        # inproc:// avoids TCP port collisions in CI; it shares the
        # process-wide zmq.Context.instance() the publisher binds to.
        endpoint = "inproc://test-kv-events-roundtrip"
        pub = ZmqEventPublisher(endpoint=endpoint, buffer_steps=16)
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        try:
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(endpoint)
            decoder = msgspec.msgpack.Decoder(EventBatch)
            payload: bytes | None = None
            for _ in range(10):
                pub.publish([BlockRemoved(block_hashes=[7])])
                if sub.poll(timeout=200):
                    payload = sub.recv()
                    break
            assert payload is not None, "SUB did not receive any batch"
            batch = decoder.decode(payload)
            assert len(batch.events) == 1
            assert isinstance(batch.events[0], BlockRemoved)
        finally:
            sub.close(linger=0)
            pub.shutdown()
