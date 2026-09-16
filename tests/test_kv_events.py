# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for atom/distributed/kv_events.py and the BlockManager hooks.

Covers:
  * Event schema + msgspec round-trip
  * BlockManager emits BlockStored only for newly finalized blocks (cache-hit reuse skips)
  * BlockManager emits BlockRemoved on lazy eviction
  * `take_events()` drain semantics
  * `clear_cache()` emits AllBlocksCleared
  * `record_remote_store()` emits BlockStored(medium=REMOTE)
  * ZmqEventPublisher PUB→SUB round-trip, `[topic, seq, payload]` framing,
    per-DP-rank endpoint offset
  * NullEventPublisher is a no-op
"""

from __future__ import annotations

import time

import msgspec
import pytest
from conftest import MockConfig

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
    offset_endpoint_port,
)
from atom.model_engine.block_manager import BlockManager

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


def _dcp_bm_with_events(monkeypatch) -> BlockManager:
    bm = _bm_with_events(decode_context_parallel_size=2)
    monkeypatch.setattr(
        bm,
        "num_pool_blocks",
        lambda seq_len: (seq_len + bm.hash_block_size - 1) // bm.hash_block_size,
    )
    return bm


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

    def test_cache_hit_emits_only_new_blocks(self, seq_factory):
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
        assert bm.kv.num_indexed, "preconditions: hash should be cached"
        bm.clear_cache()
        assert bm.kv.num_indexed == 0

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


class TestDCPBlockStoredGranularity:
    @staticmethod
    def _only_stored_event(bm):
        stored = [event for event in bm.take_events() if isinstance(event, BlockStored)]
        assert len(stored) == 1
        return stored[0]

    @staticmethod
    def _assert_hash_block_aligned(event, expected_hashes):
        assert event.block_size == 8
        assert len(event.block_hashes) == expected_hashes
        assert len(event.token_ids) == expected_hashes * event.block_size

    def test_hash_blocks_reports_hash_block_size(self, seq_factory, monkeypatch):
        bm = _dcp_bm_with_events(monkeypatch)
        seq = seq_factory(list(range(16)))
        bm.allocate(seq)

        bm.hash_blocks(seq, seq.num_prompt_tokens)

        event = self._only_stored_event(bm)
        self._assert_hash_block_aligned(event, expected_hashes=2)
        assert event.token_ids == list(range(16))

    def test_publish_loaded_prefix_reports_hash_block_size(
        self, seq_factory, monkeypatch
    ):
        bm = _dcp_bm_with_events(monkeypatch)
        seq = seq_factory(list(range(8)))
        bm.allocate(seq)

        assert bm.publish_loaded_prefix(seq, start_token=0, end_token=8) == 8

        event = self._only_stored_event(bm)
        self._assert_hash_block_aligned(event, expected_hashes=1)
        assert event.token_ids == list(range(8))

    def test_record_remote_store_reports_hash_block_size(self, monkeypatch):
        bm = _dcp_bm_with_events(monkeypatch)

        bm.record_remote_store(
            block_hashes=[42, 43],
            token_ids=list(range(16)),
        )

        event = self._only_stored_event(bm)
        self._assert_hash_block_aligned(event, expected_hashes=2)
        assert event.medium == MEDIUM_REMOTE


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
            topic, seq_bytes, payload = _first_frames(
                pub, sub, BlockRemoved(block_hashes=[7])
            )
            assert topic == b""
            assert len(seq_bytes) == 8
            batch = msgspec.msgpack.Decoder(EventBatch).decode(payload)
            assert len(batch.events) == 1
            assert isinstance(batch.events[0], BlockRemoved)
        finally:
            sub.close(linger=0)
            pub.shutdown()

    def test_wire_frames_are_topic_seq_payload(self):
        # vLLM layout: [topic, uint64 big-endian seq, msgpack EventBatch], with
        # seq advancing by one per published batch.
        zmq = pytest.importorskip("zmq")
        endpoint = "inproc://test-kv-events-frames"
        topic = b"kv@10.0.0.1@model"
        pub = ZmqEventPublisher(
            endpoint=endpoint, topic=topic.decode(), buffer_steps=16
        )
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        try:
            sub.setsockopt(zmq.SUBSCRIBE, b"kv@")
            sub.connect(endpoint)
            decoder = msgspec.msgpack.Decoder(EventBatch)
            first = _first_frames(pub, sub, BlockRemoved(block_hashes=[0]))
            assert first[0] == topic
            first_seq = int.from_bytes(first[1], "big")
            for expected in range(first_seq + 1, first_seq + 4):
                pub.publish([BlockRemoved(block_hashes=[expected])])
                assert sub.poll(timeout=2000), f"batch {expected} not received"
                frames = sub.recv_multipart()
                assert len(frames) == 3
                assert frames[0] == topic
                assert len(frames[1]) == 8
                assert int.from_bytes(frames[1], "big") == expected
                batch = decoder.decode(frames[2])
                assert batch.events[0].block_hashes == [expected]
        finally:
            sub.close(linger=0)
            pub.shutdown()

    def test_dropped_batches_consume_seq(self):
        # Stopped sender + buffer_steps=1: publishes 0 and 1 are dropped on
        # overflow, and the surviving queued batch carries seq 2.
        pytest.importorskip("zmq")
        pub = ZmqEventPublisher(
            endpoint="inproc://test-kv-events-seq-gap", buffer_steps=1
        )
        pub._queue.put_nowait(None)
        pub._sender.join(timeout=2.0)
        try:
            for i in range(3):
                pub.publish([BlockRemoved(block_hashes=[i])])
            seq, _ = pub._queue.get_nowait()
            assert seq == 2
            assert pub.stats["dropped"] == 2
        finally:
            pub._socket.close(linger=0)

    def test_publish_drops_oldest_on_overflow(self):
        # buffer_steps=1 + stopped sender => every publish past the first must
        # drop the oldest queued item and tick stats["dropped"].
        pytest.importorskip("zmq")
        pub = ZmqEventPublisher(endpoint="inproc://test-kv-events-drop", buffer_steps=1)
        # Stop the sender so the queue stays at capacity.
        pub._queue.put_nowait(None)
        pub._sender.join(timeout=2.0)
        try:
            for i in range(5):
                pub.publish([BlockRemoved(block_hashes=[i])])
            assert pub.stats["dropped"] >= 4
        finally:
            try:
                pub._socket.close(linger=0)
            except Exception:
                pass

    def test_publish_counts_encode_errors_without_raising(self):
        pytest.importorskip("zmq")
        pub = ZmqEventPublisher(
            endpoint="inproc://test-kv-events-encode-error", buffer_steps=4
        )

        class _BadEncoder:
            def encode(self, _):
                raise RuntimeError("boom")

        pub._encoder = _BadEncoder()
        try:
            pub.publish([BlockRemoved(block_hashes=[1])])
            pub.publish([BlockRemoved(block_hashes=[2])])
            assert pub.stats["encode_errors"] == 2
            assert pub.stats["sent"] == 0
        finally:
            pub.shutdown()


# ── Endpoint offset ────────────────────────────────────────────────────────


class TestOffsetEndpointPort:
    @pytest.mark.parametrize(
        ("endpoint", "rank", "expected"),
        [
            ("tcp://*:5557", 0, "tcp://*:5557"),
            ("tcp://*:5557", 3, "tcp://*:5560"),
            ("tcp://0.0.0.0:5557", 1, "tcp://0.0.0.0:5558"),
            ("tcp://[::1]:5557", 2, "tcp://[::1]:5559"),
            ("inproc://kv", 2, "inproc://kv_dp2"),
            ("ipc:///tmp/kv.sock", 1, "ipc:///tmp/kv.sock_dp1"),
        ],
    )
    def test_offsets(self, endpoint, rank, expected):
        assert offset_endpoint_port(endpoint, rank) == expected

    def test_tcp_without_numeric_port_rejected(self):
        with pytest.raises(ValueError):
            offset_endpoint_port("tcp://host", 1)

    def test_publisher_binds_rank_offset_endpoint(self):
        # Two ranks on one base endpoint bind distinct sockets, and the rank-1
        # stream is reachable at the offset address.
        zmq = pytest.importorskip("zmq")
        base = "inproc://test-kv-events-dp"
        rank0 = ZmqEventPublisher(endpoint=base, buffer_steps=4, data_parallel_rank=0)
        rank1 = ZmqEventPublisher(endpoint=base, buffer_steps=4, data_parallel_rank=1)
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        try:
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(offset_endpoint_port(base, 1))
            _, _, payload = _first_frames(rank1, sub, BlockRemoved(block_hashes=[1]))
            batch = msgspec.msgpack.Decoder(EventBatch).decode(payload)
            assert batch.data_parallel_rank == 1
        finally:
            sub.close(linger=0)
            rank0.shutdown()
            rank1.shutdown()


def _first_frames(pub: ZmqEventPublisher, sub, event) -> list[bytes]:
    """Publish `event` until the SUB (a slow joiner) sees a batch; return its
    frames."""
    for _ in range(20):
        pub.publish([event])
        if sub.poll(timeout=200):
            return sub.recv_multipart()
    raise AssertionError("SUB did not receive any batch")
