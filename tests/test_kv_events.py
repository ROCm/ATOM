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
  * ZmqEventPublisher PUB→SUB round-trip
  * NullEventPublisher is a no-op
"""

from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace

import msgspec
import pytest
from conftest import MockConfig

from atom.config import KVEventsConfig
from atom.distributed.kv_events import (
    MEDIUM_GPU,
    MEDIUM_REMOTE,
    REPLAY_CHUNK,
    REPLAY_DONE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockTransferred,
    EventBatch,
    NullEventPublisher,
    ZmqEventPublisher,
    make_publisher,
    offset_endpoint,
)
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler

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

    def test_block_stored_token_offset_roundtrip(self):
        # token_offset records the sequence position the first block of the run
        # covers, so consumers can map each block to [offset + i*block_size, ...).
        evt = BlockStored(
            block_hashes=[111, 222],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            block_size=4,
            token_offset=16,
        )
        enc = msgspec.msgpack.Encoder().encode(evt)
        dec = msgspec.msgpack.Decoder(BlockStored).decode(enc)
        assert dec.token_offset == 16

    def test_block_stored_token_offset_defaults_none(self):
        evt = BlockStored(
            block_hashes=[1],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4],
            block_size=4,
        )
        dec = msgspec.msgpack.Decoder(BlockStored).decode(
            msgspec.msgpack.Encoder().encode(evt)
        )
        assert dec.token_offset is None

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

    def test_block_stored_first_run_offset_is_zero(self, seq_factory):
        bm = _bm_with_events()
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(bm, seq)
        stored = [e for e in bm.take_events() if isinstance(e, BlockStored)]
        assert len(stored) == 1
        assert stored[0].token_offset == 0

    def test_block_stored_offset_after_cached_prefix(self, seq_factory):
        # s2 reuses s1's first two blocks (tokens 1-8) and adds a third
        # block (tokens 9-12). The new run starts at block index 2, so its
        # token_offset must be 2 * block_size == 8.
        bm = _bm_with_events()
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        _admit(bm, s1)
        bm.take_events()

        s2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        _admit(bm, s2)
        stored = [e for e in bm.take_events() if isinstance(e, BlockStored)]
        assert len(stored) == 1
        assert stored[0].token_offset == 8

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

    def test_publish_loaded_prefix_offset_after_cached_prefix(self, seq_factory):
        # Blocks 0-1 hashed by the prefill path, blocks 2-3 restored by the
        # offload loader: one BlockStored per loaded block, positioned at
        # tokens 8 and 12, the first chained off block 1's hash.
        bm = _bm_with_events()
        seq = seq_factory(list(range(16)))
        bm.allocate(seq)
        bm.hash_blocks(seq, 8)
        local = [e for e in bm.take_events() if isinstance(e, BlockStored)]
        assert len(local) == 1 and local[0].token_offset == 0

        assert bm.publish_loaded_prefix(seq, start_token=8, end_token=16) == 8

        loaded = [e for e in bm.take_events() if isinstance(e, BlockStored)]
        assert [e.token_offset for e in loaded] == [8, 12]
        assert all(e.block_size == 4 for e in loaded)
        assert loaded[0].parent_block_hash == local[0].block_hashes[-1]
        assert loaded[1].parent_block_hash == loaded[0].block_hashes[0]
        assert loaded[0].token_ids == list(range(8, 12))
        assert loaded[1].token_ids == list(range(12, 16))

    def test_record_remote_store_carries_token_offset(self, seq_factory):
        bm = _bm_with_events()
        bm.record_remote_store(
            block_hashes=[42, 43],
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            parent_block_hash=None,
            token_offset=16,
        )
        events = bm.take_events()
        assert len(events) == 1
        assert events[0].token_offset == 16

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

    def test_publish_loaded_prefix_offset_after_cached_prefix_dcp(
        self, seq_factory, monkeypatch
    ):
        # Block-table entry 0 was hashed locally (8 tokens at hbs=8); the
        # offload load fills entry 1. Its event must start at token 8, in
        # hash-block units, and chain off the locally hashed block.
        bm = _dcp_bm_with_events(monkeypatch)
        seq = seq_factory(list(range(16)))
        bm.allocate(seq)
        bm.hash_blocks(seq, 8)
        local = self._only_stored_event(bm)

        assert bm.publish_loaded_prefix(seq, start_token=8, end_token=16) == 8

        event = self._only_stored_event(bm)
        self._assert_hash_block_aligned(event, expected_hashes=1)
        assert event.token_offset == 8
        assert event.parent_block_hash == local.block_hashes[-1]
        assert event.token_ids == list(range(8, 16))

    def test_record_remote_store_reports_hash_block_size(self, monkeypatch):
        bm = _dcp_bm_with_events(monkeypatch)

        bm.record_remote_store(
            block_hashes=[42, 43],
            token_ids=list(range(16)),
        )

        event = self._only_stored_event(bm)
        self._assert_hash_block_aligned(event, expected_hashes=2)
        assert event.medium == MEDIUM_REMOTE


# ── Scheduler: BlockStored(REMOTE) on remote-KV completion ─────────────────


class TestSchedulerRemoteStore:
    """`Scheduler._update_waiting_for_remote_kv` walks the block table and
    emits one BlockStored(REMOTE) for the blocks that arrived from the
    producer. The stub block manager records the call so the test pins the
    offset/parent derivation without a real pool."""

    @staticmethod
    def _scheduler(hash_block_size: int, blocks: dict[int, tuple[int, list[int]]]):
        calls: list[dict] = []
        table = {
            bid: SimpleNamespace(hash=h, token_ids=toks)
            for bid, (h, toks) in blocks.items()
        }
        bm = SimpleNamespace(
            kv_events_enabled=True,
            enable_prefix_caching=False,  # isolate the event path
            hash_block_size=hash_block_size,
            kv=SimpleNamespace(block=lambda bid: table[bid]),
            record_remote_store=lambda **kw: calls.append(kw),
        )
        sched = Scheduler.__new__(Scheduler)
        sched.block_manager = bm
        sched.kv_connector = SimpleNamespace(is_offload=False)
        sched.finished_recving_kv_req_ids = []
        return sched, calls

    def test_offset_and_parent_follow_first_remote_block(self):
        # Blocks 0-1 were a local prefix hit (8 cached tokens at hbs=4), blocks
        # 2-3 arrived remotely. The event must start at token 8 and chain off
        # the last cached hash.
        sched, calls = self._scheduler(
            hash_block_size=4,
            blocks={
                10: (100, [0, 1, 2, 3]),
                11: (101, [4, 5, 6, 7]),
                12: (102, [8, 9, 10, 11]),
                13: (103, [12, 13, 14, 15]),
            },
        )
        sched.finished_recving_kv_req_ids = ["7"]
        seq = SimpleNamespace(id=7, block_table=[10, 11, 12, 13], num_cached_tokens=8)

        assert sched._update_waiting_for_remote_kv(seq) is True
        assert calls == [
            {
                "block_hashes": [102, 103],
                "token_ids": list(range(8, 16)),
                "parent_block_hash": 101,
                "token_offset": 8,
            }
        ]

    def test_unhashed_block_pushes_offset_past_cached_prefix(self):
        # An unhashed (partial) block right after the cached prefix is skipped;
        # the offset must come from the first remote block's real index (3),
        # not from num_cached_blocks (2).
        sched, calls = self._scheduler(
            hash_block_size=4,
            blocks={
                10: (100, [0, 1, 2, 3]),
                11: (101, [4, 5, 6, 7]),
                12: (-1, []),
                13: (103, [12, 13, 14, 15]),
            },
        )
        sched.finished_recving_kv_req_ids = [8]
        seq = SimpleNamespace(id=8, block_table=[10, 11, 12, 13], num_cached_tokens=8)

        assert sched._update_waiting_for_remote_kv(seq) is True
        assert len(calls) == 1
        assert calls[0]["block_hashes"] == [103]
        assert calls[0]["parent_block_hash"] == 101
        assert calls[0]["token_offset"] == 12

    def test_offset_uses_hash_block_size_under_dcp(self):
        # DCP world size 2, block_size 4 -> each block-table entry spans 8
        # global tokens. One cached entry (8 tokens) then one remote entry: the
        # remote event starts at token 8, not at 1 * block_size == 4.
        sched, calls = self._scheduler(
            hash_block_size=8,
            blocks={
                20: (200, list(range(8))),
                21: (201, list(range(8, 16))),
            },
        )
        sched.finished_recving_kv_req_ids = ["9"]
        seq = SimpleNamespace(id=9, block_table=[20, 21], num_cached_tokens=8)

        assert sched._update_waiting_for_remote_kv(seq) is True
        assert calls == [
            {
                "block_hashes": [201],
                "token_ids": list(range(8, 16)),
                "parent_block_hash": 200,
                "token_offset": 8,
            }
        ]

    def test_no_event_when_everything_was_cached(self):
        sched, calls = self._scheduler(
            hash_block_size=4, blocks={10: (100, [0, 1, 2, 3])}
        )
        sched.finished_recving_kv_req_ids = ["1"]
        seq = SimpleNamespace(id=1, block_table=[10], num_cached_tokens=4)

        assert sched._update_waiting_for_remote_kv(seq) is True
        assert calls == []


# ── Config ─────────────────────────────────────────────────────────────────


class TestKVEventsConfig:
    def test_new_fields_do_not_shift_positional_callers(self):
        # replay_* were appended after the pre-existing fields so a positional
        # KVEventsConfig(enable, publisher, endpoint, topic, hwm, buffer_steps)
        # still binds the same way.
        cfg = KVEventsConfig(True, "zmq", "tcp://127.0.0.1:5557", "t", 128, 64)
        assert cfg.hwm == 128
        assert cfg.buffer_steps == 64
        assert cfg.replay_endpoint == ""
        assert cfg.replay_buffer_steps == 10_000


class TestPPPublisherOwnership:
    @staticmethod
    def _scheduler(pp_rank: int) -> Scheduler:
        class _KVEventsConfig:
            enable = True
            publisher = "zmq"
            endpoint = f"inproc://test-kv-pp-{pp_rank}"
            topic = ""
            hwm = 0
            buffer_steps = 8
            replay_endpoint = ""
            replay_buffer_steps = 8

        cfg = MockConfig(enable_prefix_caching=True)
        cfg.kv_events_config = _KVEventsConfig()
        cfg.parallel_config = SimpleNamespace(
            data_parallel_rank=0, pipeline_parallel_rank=pp_rank
        )
        return Scheduler(cfg)

    def test_head_stage_publishes(self):
        pytest.importorskip("zmq")
        sched = self._scheduler(pp_rank=0)
        try:
            assert isinstance(sched.kv_event_publisher, ZmqEventPublisher)
        finally:
            sched.shutdown_kv_events()

    def test_downstream_stage_gets_null_publisher(self):
        # Downstream PP stages never schedule, so they must not bind the
        # (per-DP-rank) endpoints the head stage on the same host owns.
        sched = self._scheduler(pp_rank=1)
        assert isinstance(sched.kv_event_publisher, NullEventPublisher)
        sched.shutdown_kv_events()


# ── Per-DP-rank endpoints ──────────────────────────────────────────────────


class TestEndpointOffset:
    @pytest.mark.parametrize(
        ("endpoint", "rank", "expected"),
        [
            ("tcp://127.0.0.1:5557", None, "tcp://127.0.0.1:5557"),
            ("tcp://127.0.0.1:5557", 0, "tcp://127.0.0.1:5557"),
            ("tcp://127.0.0.1:5557", 3, "tcp://127.0.0.1:5560"),
            ("tcp://*:5557", 1, "tcp://*:5558"),
            ("tcp://[::1]:5557", 2, "tcp://[::1]:5559"),
            ("ipc:///tmp/kv-events.sock", 1, "ipc:///tmp/kv-events.sock_dp1"),
            ("inproc://kv", 2, "inproc://kv_dp2"),
        ],
    )
    def test_offset_endpoint(self, endpoint, rank, expected):
        assert offset_endpoint(endpoint, rank) == expected

    def test_tcp_without_numeric_port_is_rejected_for_nonzero_rank(self):
        with pytest.raises(ValueError, match="numeric port"):
            offset_endpoint("tcp://127.0.0.1:*", 1)

    def test_bind_collision_explains_endpoint_rules(self):
        pytest.importorskip("zmq")
        pub = ZmqEventPublisher(endpoint="inproc://test-kv-collide", buffer_steps=4)
        try:
            with pytest.raises(RuntimeError, match="data_parallel_size apart"):
                ZmqEventPublisher(endpoint="inproc://test-kv-collide", buffer_steps=4)
        finally:
            pub.shutdown()

    def test_dp_ranks_bind_distinct_pub_and_replay_sockets(self):
        # Two ranks configured with the *same* endpoints (as every DP Scheduler
        # is) must both start; before the offset rank 1 died on address-in-use.
        pytest.importorskip("zmq")
        pub_ep = "inproc://test-kv-dp-pub"
        replay_ep = "inproc://test-kv-dp-router"
        pubs = []
        try:
            for rank in (0, 1):
                pubs.append(
                    ZmqEventPublisher(
                        endpoint=pub_ep,
                        replay_endpoint=replay_ep,
                        buffer_steps=4,
                        data_parallel_rank=rank,
                    )
                )
            assert pubs[0].endpoint == pub_ep
            assert pubs[0].replay_endpoint == replay_ep
            assert pubs[1].endpoint == pub_ep + "_dp1"
            assert pubs[1].replay_endpoint == replay_ep + "_dp1"
        finally:
            for pub in pubs:
                pub.shutdown()


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


class TestReplayEndpointWiring:
    def test_make_publisher_forwards_replay_endpoint(self):
        pytest.importorskip("zmq")
        pub = make_publisher(
            enabled=True,
            publisher_kind="zmq",
            endpoint="inproc://mp-replay-pub",
            replay_endpoint="inproc://mp-replay-router",
        )
        try:
            assert pub._replay is not None
            assert pub._replay_buffer is not None
        finally:
            pub.shutdown()

    def test_make_publisher_no_replay_by_default(self):
        pytest.importorskip("zmq")
        pub = make_publisher(
            enabled=True, publisher_kind="zmq", endpoint="inproc://mp-noreplay"
        )
        try:
            assert pub._replay is None
        finally:
            pub.shutdown()

    def test_env_replay_endpoint_default_and_override(self, monkeypatch):
        from atom.utils import envs

        monkeypatch.delenv("ATOM_KV_EVENTS_REPLAY_ENDPOINT", raising=False)
        assert envs.ATOM_KV_EVENTS_REPLAY_ENDPOINT == ""
        monkeypatch.setenv("ATOM_KV_EVENTS_REPLAY_ENDPOINT", "tcp://127.0.0.1:5558")
        assert envs.ATOM_KV_EVENTS_REPLAY_ENDPOINT == "tcp://127.0.0.1:5558"

    def test_replay_buffer_steps_is_independent_knob(self):
        pytest.importorskip("zmq")
        # replay buffer size is decoupled from the send-queue depth.
        pub = ZmqEventPublisher(
            endpoint="inproc://rb-knob-pub",
            replay_endpoint="inproc://rb-knob-router",
            buffer_steps=64,
            replay_buffer_steps=3,
        )
        try:
            assert pub._replay_buffer.maxlen == 3
        finally:
            pub.shutdown()

    def test_env_replay_buffer_steps(self, monkeypatch):
        from atom.utils import envs

        monkeypatch.delenv("ATOM_KV_EVENTS_REPLAY_BUFFER_STEPS", raising=False)
        assert envs.ATOM_KV_EVENTS_REPLAY_BUFFER_STEPS == 10000
        monkeypatch.setenv("ATOM_KV_EVENTS_REPLAY_BUFFER_STEPS", "7")
        assert envs.ATOM_KV_EVENTS_REPLAY_BUFFER_STEPS == 7

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
            frames: list[bytes] | None = None
            for _ in range(10):
                pub.publish([BlockRemoved(block_hashes=[7])])
                if sub.poll(timeout=200):
                    frames = sub.recv_multipart()
                    break
            assert frames is not None, "SUB did not receive any batch"
            # Wire layout is [topic, seq, payload]; topic is empty by default.
            assert len(frames) == 3
            topic, seq_bytes, payload = frames
            assert topic == b""
            # 8-byte big-endian seq frame (don't assume which batch arrived
            # first — the warm-up loop above may drop early ones on connect).
            assert len(seq_bytes) == 8
            batch = decoder.decode(payload)
            assert len(batch.events) == 1
            assert isinstance(batch.events[0], BlockRemoved)
        finally:
            sub.close(linger=0)
            pub.shutdown()

    def test_zmq_publisher_seq_is_monotonic(self):
        zmq = pytest.importorskip("zmq")
        endpoint = "inproc://test-kv-events-seq"
        pub = ZmqEventPublisher(endpoint=endpoint, buffer_steps=64)
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        try:
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(endpoint)
            # Warm up past the ZMQ "slow joiner": publish until the SUB actually
            # receives one (subscription now live), then drain extras.
            warmed = False
            for _ in range(20):
                pub.publish([BlockRemoved(block_hashes=[0])])
                if sub.poll(timeout=200):
                    sub.recv_multipart()
                    warmed = True
                    break
            assert warmed, "SUB never received a warm-up batch"
            while sub.poll(timeout=100):
                sub.recv_multipart()
            # The next batches must arrive with strictly consecutive seqs.
            for i in range(5):
                pub.publish([BlockRemoved(block_hashes=[i])])
            seqs: list[int] = []
            polls = 50
            while len(seqs) < 5 and polls > 0:
                if sub.poll(timeout=200):
                    _, seq_bytes, _ = sub.recv_multipart()
                    seqs.append(int.from_bytes(seq_bytes, "big"))
                polls -= 1
            assert len(seqs) == 5, seqs
            assert all(seqs[i + 1] == seqs[i] + 1 for i in range(4)), seqs
        finally:
            sub.close(linger=0)
            pub.shutdown()

    def test_replay_recovers_missed_batches(self):
        # No SUB here: the replay buffer is populated by the sender regardless
        # of delivery, so we wait on stats["sent"] and avoid slow-joiner flake.
        zmq = pytest.importorskip("zmq")
        pub_ep = "inproc://test-kv-replay-pub"
        replay_ep = "inproc://test-kv-replay-router"
        pub = ZmqEventPublisher(
            endpoint=pub_ep, replay_endpoint=replay_ep, buffer_steps=64
        )
        ctx = zmq.Context.instance()
        try:
            for i in range(3):
                pub.publish([BlockRemoved(block_hashes=[i])])
            polls = 100
            while pub.stats["sent"] < 3 and polls > 0:
                time.sleep(0.02)
                polls -= 1
            assert pub.stats["sent"] == 3

            dealer = ctx.socket(zmq.DEALER)
            dealer.connect(replay_ep)
            dealer.send(b"\x00" * 8)  # start_seq = 0
            replayed_seqs: list[int] = []
            while len(replayed_seqs) < 3 and dealer.poll(timeout=1000):
                seq_bytes, payload = dealer.recv_multipart()
                replayed_seqs.append(int.from_bytes(seq_bytes, "big"))
                msgspec.msgpack.Decoder(EventBatch).decode(payload)
            dealer.close(linger=0)
            assert replayed_seqs == [0, 1, 2]
        finally:
            pub.shutdown()

    def test_replay_only_returns_from_start_seq(self):
        zmq = pytest.importorskip("zmq")
        pub_ep = "inproc://test-kv-replay-pub2"
        replay_ep = "inproc://test-kv-replay-router2"
        pub = ZmqEventPublisher(
            endpoint=pub_ep, replay_endpoint=replay_ep, buffer_steps=64
        )
        ctx = zmq.Context.instance()
        try:
            for i in range(4):
                pub.publish([BlockRemoved(block_hashes=[i])])
            polls = 100
            while pub.stats["sent"] < 4 and polls > 0:
                time.sleep(0.02)
                polls -= 1
            assert pub.stats["sent"] == 4

            dealer = ctx.socket(zmq.DEALER)
            dealer.connect(replay_ep)
            dealer.send((2).to_bytes(8, "big"))  # start_seq = 2
            replayed_seqs: list[int] = []
            while len(replayed_seqs) < 2 and dealer.poll(timeout=1000):
                seq_bytes, _ = dealer.recv_multipart()
                replayed_seqs.append(int.from_bytes(seq_bytes, "big"))
            dealer.close(linger=0)
            assert replayed_seqs == [2, 3]
        finally:
            pub.shutdown()

    def test_replay_sends_terminal_frame(self):
        # After the matched batches, a REPLAY_DONE terminal frame carries the
        # [oldest, latest] window so the consumer knows replay is complete.
        zmq = pytest.importorskip("zmq")
        pub = ZmqEventPublisher(
            endpoint="inproc://term-pub",
            replay_endpoint="inproc://term-router",
            buffer_steps=64,
        )
        ctx = zmq.Context.instance()
        try:
            for i in range(3):
                pub.publish([BlockRemoved(block_hashes=[i])])
            polls = 100
            while pub.stats["sent"] < 3 and polls > 0:
                time.sleep(0.02)
                polls -= 1
            assert pub.stats["sent"] == 3
            dealer = ctx.socket(zmq.DEALER)
            dealer.connect("inproc://term-router")
            dealer.send(b"\x00" * 8)
            data: list[int] = []
            window = None
            while dealer.poll(timeout=1000):
                seq_bytes, payload = dealer.recv_multipart()
                if seq_bytes == REPLAY_DONE:
                    window = msgspec.msgpack.decode(payload)
                    break
                data.append(int.from_bytes(seq_bytes, "big"))
            dealer.close(linger=0)
            assert data == [0, 1, 2]
            assert window == [0, 2]  # [oldest, latest]
        finally:
            pub.shutdown()

    def test_replay_zero_match_still_terminates(self):
        # A start_seq past the newest batch yields no data frames but still a
        # terminal frame, so the consumer never hangs waiting.
        zmq = pytest.importorskip("zmq")
        pub = ZmqEventPublisher(
            endpoint="inproc://term-pub2",
            replay_endpoint="inproc://term-router2",
            buffer_steps=64,
        )
        ctx = zmq.Context.instance()
        try:
            for i in range(2):
                pub.publish([BlockRemoved(block_hashes=[i])])
            polls = 100
            while pub.stats["sent"] < 2 and polls > 0:
                time.sleep(0.02)
                polls -= 1
            assert pub.stats["sent"] == 2
            dealer = ctx.socket(zmq.DEALER)
            dealer.connect("inproc://term-router2")
            dealer.send((99).to_bytes(8, "big"))  # start_seq beyond latest
            data: list[int] = []
            got_terminal = False
            while dealer.poll(timeout=1000):
                seq_bytes, _ = dealer.recv_multipart()
                if seq_bytes == REPLAY_DONE:
                    got_terminal = True
                    break
                data.append(int.from_bytes(seq_bytes, "big"))
            dealer.close(linger=0)
            assert data == []
            assert got_terminal
        finally:
            pub.shutdown()

    def test_malformed_replay_request_is_rejected(self):
        # start_seq must be exactly 8 bytes: b"" would parse as 0 and trigger
        # a full replay, so short/long frames are dropped and only the
        # well-formed request that follows is answered.
        zmq = pytest.importorskip("zmq")
        pub = ZmqEventPublisher(
            endpoint="inproc://test-kv-replay-malformed-pub",
            replay_endpoint="inproc://test-kv-replay-malformed-router",
            buffer_steps=64,
        )
        ctx = zmq.Context.instance()
        try:
            for i in range(3):
                pub.publish([BlockRemoved(block_hashes=[i])])
            polls = 200
            while pub.stats["sent"] < 3 and polls > 0:
                time.sleep(0.02)
                polls -= 1
            dealer = ctx.socket(zmq.DEALER)
            dealer.connect("inproc://test-kv-replay-malformed-router")
            dealer.send(b"")  # empty start_seq
            dealer.send(b"\x00" * 3)  # too short
            dealer.send(b"\x00" * 9)  # too long
            dealer.send((1).to_bytes(8, "big"))  # the only valid one
            got: list[int] = []
            dones = 0
            while dealer.poll(timeout=500):
                seq_bytes, _ = dealer.recv_multipart()
                if seq_bytes == REPLAY_DONE:
                    dones += 1
                else:
                    got.append(int.from_bytes(seq_bytes, "big"))
            dealer.close(linger=0)
            assert got == [1, 2]
            assert dones == 1
            assert pub.stats["replayed"] == 2
        finally:
            pub.shutdown()

    def test_large_replay_is_chunked_and_complete(self):
        # A window several chunks long is delivered in order and finished with
        # REPLAY_DONE; the chunking is invisible to the consumer.
        zmq = pytest.importorskip("zmq")
        n = REPLAY_CHUNK * 3 + 5
        pub = ZmqEventPublisher(
            endpoint="inproc://test-kv-replay-chunk-pub",
            replay_endpoint="inproc://test-kv-replay-chunk-router",
            buffer_steps=n,
            replay_buffer_steps=n,
        )
        ctx = zmq.Context.instance()
        try:
            for i in range(n):
                pub.publish([BlockRemoved(block_hashes=[i])])
            polls = 500
            while pub.stats["sent"] < n and polls > 0:
                time.sleep(0.02)
                polls -= 1
            assert pub.stats["sent"] == n
            dealer = ctx.socket(zmq.DEALER)
            dealer.connect("inproc://test-kv-replay-chunk-router")
            dealer.send((10).to_bytes(8, "big"))
            got: list[int] = []
            window = None
            while dealer.poll(timeout=1000):
                seq_bytes, payload = dealer.recv_multipart()
                if seq_bytes == REPLAY_DONE:
                    window = msgspec.msgpack.decode(payload)
                    break
                got.append(int.from_bytes(seq_bytes, "big"))
            dealer.close(linger=0)
            assert got == list(range(10, n))
            assert window == [0, n - 1]
            assert pub.stats["replayed"] == n - 10
        finally:
            pub.shutdown()

    def test_replay_yields_to_live_queue_between_chunks(self):
        # Drive the sender-loop pieces by hand: one _advance_replay call sends
        # exactly one chunk and leaves the request pending, which is the seam
        # where _run drains the live queue before the next chunk.
        zmq = pytest.importorskip("zmq")
        pub = ZmqEventPublisher(
            endpoint="inproc://test-kv-replay-yield-pub",
            replay_endpoint="inproc://test-kv-replay-yield-router",
            buffer_steps=64,
            replay_buffer_steps=64,
        )
        ctx = zmq.Context.instance()
        dealer = ctx.socket(zmq.DEALER)
        try:
            for i in range(20):
                pub.publish([BlockRemoved(block_hashes=[i])])
            polls = 200
            while pub.stats["sent"] < 20 and polls > 0:
                time.sleep(0.02)
                polls -= 1
            pub._queue.put_nowait(None)  # stop the sender thread
            pub._sender.join(timeout=2.0)
            pub._replay_chunk = 7

            dealer.connect("inproc://test-kv-replay-yield-router")
            dealer.send(b"\x00" * 8)
            assert pub._replay.poll(timeout=1000)
            pub._accept_replay_request()
            assert pub._pending_replay is not None

            pub._advance_replay()
            assert pub.stats["replayed"] == 7
            assert pub._pending_replay[1] == 7  # resumes at seq 7
            pub._advance_replay()
            assert pub.stats["replayed"] == 14
            pub._advance_replay()  # 6 left, then the terminal frame
            assert pub.stats["replayed"] == 20
            assert pub._pending_replay is None

            got: list[int] = []
            done = False
            while dealer.poll(timeout=1000):
                seq_bytes, _ = dealer.recv_multipart()
                if seq_bytes == REPLAY_DONE:
                    done = True
                    break
                got.append(int.from_bytes(seq_bytes, "big"))
            assert got == list(range(20))
            assert done
        finally:
            dealer.close(linger=0)
            with contextlib.suppress(Exception):
                pub._socket.close(linger=0)
            with contextlib.suppress(Exception):
                pub._replay.close(linger=0)

    def test_slow_replay_client_does_not_stall_live_publishing(self):
        # A DEALER that requests a large window and never reads fills its pipe.
        # The replay must be abandoned (non-blocking send -> EAGAIN under
        # ROUTER_MANDATORY) and the sender thread must keep serving PUB.
        zmq = pytest.importorskip("zmq")
        pub_ep = "inproc://test-kv-replay-slow-pub"
        replay_ep = "inproc://test-kv-replay-slow-router"
        # An inproc pipe holds ROUTER SNDHWM + DEALER RCVHWM complete messages,
        # and libzmq snapshots the binder's HWM at bind time, so the publisher
        # side stays at its default (1000) and the dealer side is pinned to 1:
        # a replay of more than 1001 batches must hit the wall.
        n = 1200
        pub = ZmqEventPublisher(
            endpoint=pub_ep,
            replay_endpoint=replay_ep,
            buffer_steps=n,
            replay_buffer_steps=n,
        )
        ctx = zmq.Context.instance()
        dealer = ctx.socket(zmq.DEALER)
        dealer.set_hwm(1)
        try:
            for i in range(n):
                pub.publish([BlockRemoved(block_hashes=[i])])
            polls = 500
            while pub.stats["sent"] < n and polls > 0:
                time.sleep(0.02)
                polls -= 1
            assert pub.stats["sent"] == n

            dealer.connect(replay_ep)
            dealer.send(b"\x00" * 8)  # start_seq = 0: replay everything, never read
            polls = 500
            while pub.stats["replay_aborted"] == 0 and polls > 0:
                time.sleep(0.02)
                polls -= 1
            assert pub.stats["replay_aborted"] == 1
            assert pub.stats["replayed"] < n
            assert pub._sender.is_alive()

            # Live publication keeps flowing after the abandoned replay.
            pub.publish([BlockRemoved(block_hashes=[n])])
            polls = 200
            while pub.stats["sent"] < n + 1 and polls > 0:
                time.sleep(0.02)
                polls -= 1
            assert pub.stats["sent"] == n + 1
        finally:
            dealer.close(linger=0)
            pub.shutdown()

    def test_replay_buffer_steps_must_be_positive(self):
        pytest.importorskip("zmq")
        with pytest.raises(ValueError):
            ZmqEventPublisher(
                endpoint="inproc://rb-zero-pub",
                replay_endpoint="inproc://rb-zero-router",
                replay_buffer_steps=0,
            )

    def test_seq_never_equals_replay_done_sentinel(self):
        # The reserved terminal value (2**64-1) must never be produced as a
        # data seq, even when the raw counter lands exactly on it.
        pytest.importorskip("zmq")
        pub = ZmqEventPublisher(endpoint="inproc://sentinel-pub", buffer_steps=4)
        pub._queue.put_nowait(None)  # stop sender so we can inspect the queue
        pub._sender.join(timeout=2.0)
        try:
            reserved = int.from_bytes(REPLAY_DONE, "big")  # 2**64 - 1
            pub._seq_gen = iter([reserved, reserved + 1, 5])
            pub.publish([BlockRemoved(block_hashes=[1])])
            item = next(it for it in list(pub._queue.queue) if it is not None)
            assert item[0] != reserved
            assert item[0] == 0  # reserved % (2**64 - 1)
        finally:
            with contextlib.suppress(Exception):
                pub._socket.close(linger=0)

    def test_encode_failure_consumes_seq_number(self):
        # seq is assigned before encoding, so a batch lost to an encode error is
        # a visible gap like an overflow drop, not a silent renumbering.
        pytest.importorskip("zmq")
        pub = ZmqEventPublisher(endpoint="inproc://test-kv-seq-encode", buffer_steps=4)
        pub._queue.put_nowait(None)  # stop sender so we can inspect the queue
        pub._sender.join(timeout=2.0)

        class _BadEncoder:
            def encode(self, _):
                raise RuntimeError("boom")

        try:
            good_encoder = pub._encoder
            pub._encoder = _BadEncoder()
            pub.publish([BlockRemoved(block_hashes=[1])])
            pub.publish([BlockRemoved(block_hashes=[2])])
            pub._encoder = good_encoder
            pub.publish([BlockRemoved(block_hashes=[3])])
            assert pub.stats["encode_errors"] == 2
            item = next(it for it in list(pub._queue.queue) if it is not None)
            assert item[0] == 2  # seqs 0 and 1 were consumed by the failures
        finally:
            with contextlib.suppress(Exception):
                pub._socket.close(linger=0)

    def test_dropped_batches_consume_seq_numbers(self):
        # seq must be assigned at enqueue time so that a batch dropped on queue
        # overflow still consumes a sequence number -> the drop shows up as a
        # gap for subscribers (rather than silently vanishing).
        pytest.importorskip("zmq")
        pub = ZmqEventPublisher(endpoint="inproc://test-kv-seq-drop", buffer_steps=1)
        pub._queue.put_nowait(None)  # stop the sender so the queue stays full
        pub._sender.join(timeout=2.0)
        try:
            for i in range(5):
                pub.publish([BlockRemoved(block_hashes=[i])])
            # buffer_steps=1 => 4 dropped, 1 remains; the survivor is the LAST
            # published batch (seq=4), proving seqs 0..3 were assigned to the
            # dropped batches and are now gaps.
            remaining = [it for it in list(pub._queue.queue) if it is not None]
            assert len(remaining) == 1
            item = remaining[0]
            assert (
                isinstance(item, tuple) and item[0] == 4
            ), f"expected (seq=4, payload); got {item!r}"
            assert pub.stats["dropped"] >= 4
        finally:
            with contextlib.suppress(Exception):
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
            with contextlib.suppress(Exception):
                pub._socket.close(linger=0)

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
