# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A P/D producer sends a windowed draft's window, and only that.

Both ends keep the draft's KV in a pool of its own ids, aligned with their
block tables and -1 outside the window. What crosses is what the consumer's
window needs past its own cached prefix; a block it needs that the producer
does not hold, or a pair that keeps the draft two different ways, fails the
request instead of leaving a row unwritten or writing the wrong one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import KVTransferRegion

UNIT = 256


def _mooncake():
    return pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )


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


def _producer(regions, *, layout="window", max_entries=1024):
    writes = []

    def rdma_write(target, src, dst, sizes, req_id, label, *, engine, deadline):
        writes.append((list(src), list(dst), list(sizes), label))
        return True

    return (
        SimpleNamespace(
            _draft_layout=layout,
            _draft_regions=regions,
            _MAX_RDMA_ENTRIES_PER_BATCH=max_entries,
            _rdma_write=rdma_write,
        ),
        writes,
    )


def _request(dst_ids, src_ids, consumer_regions):
    from dataclasses import asdict

    return {
        "dst_draft_block_ids": dst_ids,
        "src_draft_block_ids": src_ids,
        "consumer_draft_regions": [asdict(r) for r in consumer_regions],
    }


def test_only_the_consumer_window_crosses():
    mc = _mooncake()
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
    mc = _mooncake()
    producer, writes = _producer(_regions(0x1000_0000))
    request = _request([-1, 5], [-1, -1], _regions(0x9000_0000))
    with pytest.raises(RuntimeError, match="does not hold"):
        mc.MooncakeConnector._execute_draft_window_transfer(
            producer, request, "peer", "req"
        )
    assert writes == []


def test_regions_that_do_not_pair_fail_the_request():
    mc = _mooncake()
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
    mc = _mooncake()
    producer, writes = _producer([], layout="full")
    assert mc.MooncakeConnector._execute_draft_window_transfer(
        producer, {}, "peer", "req"
    )
    assert writes == []


@pytest.mark.parametrize(
    "producer_layout, consumer_layout",
    [("window", "full"), ("window", None), ("full", "window"), (None, "window")],
)
def test_a_pair_that_keeps_the_draft_two_ways_fails_before_any_byte(
    producer_layout, consumer_layout
):
    mc = _mooncake()
    results = []
    producer = SimpleNamespace(
        tp_size=1,
        _draft_layout=producer_layout,
        _notify_transfer_result=lambda data, success: results.append(success),
    )
    request = {
        "request_id": "req",
        "consumer_host": "peer",
        "consumer_rpc_port": 1,
        "dst_block_ids": [1, 2],
        "consumer_draft_layout": consumer_layout,
    }
    mc.MooncakeConnector._execute_transfer(producer, request)
    assert results == [False]
