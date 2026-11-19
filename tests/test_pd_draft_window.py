# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A P/D producer sends a windowed draft's window, and only that.

Both ends keep the draft's KV in a pool of its own ids, aligned with their
block tables and -1 outside the window. What crosses is what the consumer's
window needs past its own cached prefix; a block it needs that the producer
does not hold fails the request, and a pair that keeps the draft two different
ways is refused on first contact, before any write request goes out.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

from atom.kv_transfer.disaggregation.types import (
    DRAFT_LAYOUT_FULL,
    DRAFT_LAYOUT_WINDOW,
    ConnectorMetadata,
    KVTransferRegion,
    KVTransferTensors,
    ReqMeta,
)

UNIT = 256


def _regions(base: int, roles=("draft.k.layer_0", "draft.v.layer_0")):
    return [
        KVTransferRegion(
            base_addr=base + i * (1 << 20),
            total_bytes=64 * UNIT,
            unit_bytes=UNIT,
            semantic_role=role,
        )
        for i, role in enumerate(roles)
    ]


def _producer(regions, *, layout=DRAFT_LAYOUT_WINDOW):
    writes = []

    def rdma_write(target, src, dst, sizes, req_id, label, *, engine, deadline):
        writes.append((list(src), list(dst), list(sizes), label))
        return True

    return (
        SimpleNamespace(
            _draft_layout=layout,
            _draft_regions=regions,
            _rdma_write=rdma_write,
        ),
        writes,
    )


def _request(dst_ids, src_ids, consumer_regions):
    return {
        "dst_draft_block_ids": dst_ids,
        "src_draft_block_ids": src_ids,
        "consumer_draft_regions": [asdict(r) for r in consumer_regions],
    }


def _meta(**kw):
    return ReqMeta(
        local_block_ids=[10, 11, 12, 13],
        remote_block_ids=[20, 21, 22, 23],
        remote_host="p",
        remote_port=1,
        remote_handshake_port=2,
        remote_engine_id="p",
        tp_size=1,
        remote_dp_size=1,
        **kw,
    )


def test_only_the_consumer_window_crosses():
    src_regions, dst_regions = _regions(0x1000_0000), _regions(0x9000_0000)
    producer, writes = _producer(src_regions)
    # Blocks 0-1 are below the consumer's window; block 1 is still held by the
    # producer, whose prompt window started a block earlier.
    request = _request([-1, -1, 5, 6], [-1, 3, 4, 8], dst_regions)
    assert mc.MooncakeConnector._execute_draft_window_transfer(
        producer, request, "peer", "req"
    )
    ((src, dst, sizes, label),) = writes
    assert label == "draft"
    assert sizes == [UNIT] * 4
    want_src, want_dst = [], []
    for s_region, d_region in zip(src_regions, dst_regions):
        for s, d in ((4, 5), (8, 6)):
            want_src.append(s_region.base_addr + s * UNIT)
            want_dst.append(d_region.base_addr + d * UNIT)
    assert (src, dst) == (want_src, want_dst)


def test_a_needed_block_the_producer_lacks_fails_the_request():
    producer, writes = _producer(_regions(0x1000_0000))
    request = _request([-1, 5], [-1, -1], _regions(0x9000_0000))
    with pytest.raises(RuntimeError, match="does not hold"):
        mc.MooncakeConnector._execute_draft_window_transfer(
            producer, request, "peer", "req"
        )
    assert writes == []


@pytest.mark.parametrize(
    "dst_ids, src_ids",
    [([5], [64]), ([64], [4]), ([1 << 40], [4])],
    ids=["src-past-the-pool", "dst-past-the-pool", "dst-far-past"],
)
def test_ids_outside_the_pools_fail_the_request(dst_ids, src_ids):
    # The ids come over the wire; 64 is one block past a 64-block region.
    producer, writes = _producer(_regions(0x1000_0000))
    request = _request(dst_ids, src_ids, _regions(0x9000_0000))
    with pytest.raises(RuntimeError, match="outside the pools"):
        mc.MooncakeConnector._execute_draft_window_transfer(
            producer, request, "peer", "req"
        )
    assert writes == []


def test_regions_that_do_not_pair_fail_the_request():
    producer, _ = _producer(_regions(0x1000_0000))
    request = _request(
        [5],
        [4],
        _regions(0x9000_0000, roles=("draft.v.layer_0", "draft.k.layer_0")),
    )
    with pytest.raises(RuntimeError, match="meets"):
        mc.MooncakeConnector._execute_draft_window_transfer(
            producer, request, "peer", "req"
        )


def test_a_draft_riding_the_target_blocks_sends_nothing_here():
    producer, writes = _producer([], layout=DRAFT_LAYOUT_FULL)
    assert mc.MooncakeConnector._execute_draft_window_transfer(
        producer, {}, "peer", "req"
    )
    assert writes == []


def test_the_consumer_asks_for_its_window_past_its_cached_prefix():
    """Both tables are cut at the same `off` as the block ids, so they stay
    aligned with them; a consumer that keeps no window sends no draft fields."""
    meta = _meta(
        local_draft_block_ids=[-1, -1, 5, 6],
        remote_draft_block_ids=[-1, 3, 4, 8],
    )
    regions = _regions(0x9000_0000)
    consumer = SimpleNamespace(
        _draft_layout=DRAFT_LAYOUT_WINDOW, _draft_regions=tuple(regions)
    )
    fields = mc.MooncakeConnector._draft_window_fields(consumer, meta, 1)
    assert fields == {
        "dst_draft_block_ids": [-1, 5, 6],
        "src_draft_block_ids": [3, 4, 8],
        "consumer_draft_regions": [asdict(r) for r in regions],
    }
    consumer._draft_layout = DRAFT_LAYOUT_FULL
    assert mc.MooncakeConnector._draft_window_fields(consumer, meta, 1) == {}


def test_a_windowed_consumer_fails_a_chunked_receive():
    conn = object.__new__(mc.MooncakeConnector)
    conn.__dict__.update(
        is_producer=False,
        tp_size=1,
        pp_size=1,
        _draft_layout=DRAFT_LAYOUT_WINDOW,
        _completion_lock=threading.Lock(),
        failed_recving=set(),
    )
    metadata = ConnectorMetadata()
    metadata.reqs_to_recv["req"] = _meta(chunked_transfer=True)
    conn.start_load_kv(metadata)
    assert conn.failed_recving == {"req"}


def _consumer(layout, producer_layout):
    """A consumer whose write-request worker is driven by hand, and the socket
    it asks a producer keeping its draft as `producer_layout` on."""
    conn = object.__new__(mc.MooncakeConnector)
    conn.__dict__.update(
        is_producer=False,
        _cuda_device=0,
        _write_queue=[],
        _write_cv=threading.Condition(),
        _write_closed=False,
        _write_thread=object(),
        _write_request_error=None,
        _verified_producers=set(),
        _has_slot_regions=False,
        _staging_slots=mc.SlotPool(),
        _draft_layout=layout,
        _decoder=mc.msgspec.msgpack.Decoder(mc.MooncakeAgentMetadata),
    )
    reply = mc.msgspec.msgpack.encode(
        mc.MooncakeAgentMetadata(
            engine_id="p", rpc_port=1, draft_layout=producer_layout
        )
    )
    sock = MagicMock()
    sock.poll.return_value = 1
    sock.recv_multipart.return_value = [b"", reply]
    conn.zmq_context = MagicMock()
    conn.zmq_context.socket.return_value = sock
    sent = []
    conn._send_on_socket = lambda addr, parts: sent.append(parts)
    return conn, sock, sent


@pytest.mark.parametrize(
    "consumer_layout, producer_layout",
    [
        (DRAFT_LAYOUT_WINDOW, DRAFT_LAYOUT_FULL),
        (DRAFT_LAYOUT_WINDOW, None),
        (DRAFT_LAYOUT_FULL, DRAFT_LAYOUT_WINDOW),
    ],
)
def test_a_pair_that_keeps_the_draft_two_ways_is_refused_before_any_request(
    monkeypatch, consumer_layout, producer_layout
):
    monkeypatch.setattr(mc.torch.cuda, "synchronize", lambda *a: None)
    conn, sock, sent = _consumer(consumer_layout, producer_layout)
    conn._enqueue_write_requests([("tcp://p:1", [b"r0"])])
    assert not conn._post_queued_write_requests()
    assert sent == []
    assert "keeps its draft KV as" in str(conn._write_request_error)
    sock.send_multipart.assert_called_once_with([mc.MSG_GET_META])


@pytest.mark.parametrize(
    "layout", [DRAFT_LAYOUT_WINDOW, DRAFT_LAYOUT_FULL], ids=["window", "full"]
)
def test_a_pair_that_keeps_the_draft_one_way_posts_and_asks_once(monkeypatch, layout):
    monkeypatch.setattr(mc.torch.cuda, "synchronize", lambda *a: None)
    conn, sock, sent = _consumer(layout, layout)
    for i in range(2):
        conn._enqueue_write_requests([("tcp://p:1", [b"r%d" % i])])
        assert conn._post_queued_write_requests()
    assert sent == [[b"r0"], [b"r1"]]
    sock.send_multipart.assert_called_once_with([mc.MSG_GET_META])


def test_a_consumer_without_a_draft_pool_asks_nothing(monkeypatch):
    monkeypatch.setattr(mc.torch.cuda, "synchronize", lambda *a: None)
    conn, sock, sent = _consumer(None, DRAFT_LAYOUT_WINDOW)
    conn._enqueue_write_requests([("tcp://p:1", [b"r0"])])
    assert conn._post_queued_write_requests()
    assert sent == [[b"r0"]]
    sock.send_multipart.assert_not_called()


def test_merging_a_draft_takes_its_layout_and_windowed_regions():
    """The runner merges the draft's transfer description into the target's;
    a windowed draft's regions arrive in `draft_pages`, not in `pages`."""
    target = KVTransferTensors()
    page = SimpleNamespace(region=_regions(0)[0])
    target.merge_pages(
        KVTransferTensors(draft_layout=DRAFT_LAYOUT_WINDOW, draft_pages=[page])
    )
    assert target.draft_layout == DRAFT_LAYOUT_WINDOW
    assert target.draft_pages == [page] and target.pages == []
