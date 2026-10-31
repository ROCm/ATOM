# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Staged MLA pushes on the per-request-state path to a DCP consumer.

Each case runs one request from an unsharded producer through the staged path
and through the per-token in-place path into two copies of the consumer's
memory, and requires the two to match byte for byte -- including everything
neither should have written.
"""

import threading
from collections import deque
from unittest.mock import MagicMock

import pytest
from aiter_stub import stubbed_aiter

torch = pytest.importorskip("torch")

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

from atom.kv_transfer.disaggregation.kv_staging import (
    MLAShardStaging,
    mla_staging_for,
)
from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.sharded_transfer import (
    build_dcp_shard_plan,
)
from atom.kv_transfer.disaggregation.types import (
    INDEX_CACHE_ROLE,
    MLA_KV_ROLE,
)

_BLOCK = 8
_TOKEN_BYTES = 16
_BPB = _BLOCK * _TOKEN_BYTES
_SRC_BLOCKS = 12
_DST_BLOCKS = 10
_REGIONS = 2

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _TensorNic:
    """Copies bytes between registered tensors, addressed by device pointer."""

    def __init__(self, *tensors):
        self._tensors = [t.view(torch.uint8).view(-1) for t in tensors]
        self.calls = []

    def _resolve(self, addr, n):
        for t in self._tensors:
            off = addr - t.data_ptr()
            if 0 <= off and off + n <= t.numel():
                return t[off : off + n]
        raise AssertionError(f"address {addr:#x}+{n} is outside every tensor")

    def write(
        self, _target, src, dst, sizes, _req_id, label, *, engine=None, deadline=None
    ):
        for s, d, n in zip(src, dst, sizes):
            self._resolve(d, n).copy_(self._resolve(s, n))
        self.calls.append((label, len(sizes)))
        return True

    def descriptors(self, label):
        return sum(n for lab, n in self.calls if lab == label)


def _producer(device):
    g = torch.Generator().manual_seed(0)
    return [
        torch.randint(0, 256, (_SRC_BLOCKS, _BPB), generator=g, dtype=torch.uint8).to(
            device
        )
        for _ in range(_REGIONS)
    ]


def _consumer(device):
    return [
        torch.full((_DST_BLOCKS, _BPB), 0xAB, dtype=torch.uint8, device=device)
        for _ in range(_REGIONS)
    ]


def _connector(src, nic, staging):
    connector = object.__new__(mc.MooncakeConnector)
    connector.dcp_size = 1
    connector.block_size = _BLOCK
    connector.pp_size = 2
    connector._num_local_layers = _REGIONS
    connector._start_layer = 0
    connector.kv_caches_base_addr = [t.data_ptr() for t in src]
    connector._per_block_bytes_list = [_BPB] * _REGIONS
    connector._block_regions = [(t.data_ptr(), _BPB) for t in src]
    connector._block_region_roles = [MLA_KV_ROLE] * _REGIONS
    connector._block_region_consumer_indices = list(range(_REGIONS))
    connector._fp4_index_layout = False
    connector._swa_block_regions = []
    connector._slot_regions = []
    connector._kv_staging = staging
    connector._rdma_write = nic.write
    return connector


def _staging(src, chunk_pages):
    pages = [page_region(t, semantic_role=MLA_KV_ROLE) for t in src]
    return MLAShardStaging.build(
        pages,
        block_size=_BLOCK,
        pool_size=2,
        chunk_pages=chunk_pages,
        device=src[0].device,
    )


def _request(dst, *, dcp_size, dcp_rank, interleave):
    return {
        "consumer_base_addrs": [t.data_ptr() for t in dst],
        "consumer_block_base_addrs": [t.data_ptr() for t in dst],
        "consumer_num_layers": _REGIONS,
        "consumer_region_roles": [MLA_KV_ROLE] * _REGIONS,
        "consumer_block_bpb": [_BPB] * _REGIONS,
        "consumer_slot_base_addrs": [],
        "consumer_slot_bps": [],
        "dst_slot_index": -1,
        "consumer_dcp_size": dcp_size,
        "consumer_dcp_rank": dcp_rank,
        "consumer_dcp_interleave": interleave,
    }


def _ready_event(device):
    """The event the runner records after a batch's KV writes. CPU staging has
    no side stream, so it never waits on one."""
    if device == "cpu":
        return object()
    event = torch.cuda.Event()
    event.record()
    return event


def _push(connector, request, src_ids, dst_ids, ready_event):
    return mc.MooncakeConnector._execute_block_slot_transfer(
        connector,
        request,
        "consumer:1",
        src_ids,
        dst_ids,
        {"slot_index": -1},
        "req-1",
        ready_event,
    )


# Five source blocks over four ranks: rank 3 at interleave 1 gets 10 tokens,
# so its second page is partial. Destination pages 4 and 5 are adjacent.
_SRC_IDS = [5, 2, 9, 0, 7]
_DST_IDS = [4, 5]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("interleave", [1, 2])
@pytest.mark.parametrize("chunk_pages", [1, 4])
@pytest.mark.parametrize("dcp_rank", [0, 3])
def test_staged_push_lands_the_same_bytes_as_token_runs(
    device, interleave, chunk_pages, dcp_rank
):
    src = _producer(device)
    staged_dst, direct_dst = _consumer(device), _consumer(device)
    nic = _TensorNic(*src, *staged_dst, *direct_dst)
    staging = _staging(src, chunk_pages)
    nic._tensors.append(staging._pool.view(-1))

    for dst, kv_staging in ((staged_dst, staging), (direct_dst, None)):
        request = _request(dst, dcp_size=4, dcp_rank=dcp_rank, interleave=interleave)
        connector = _connector(src, nic, kv_staging)
        assert _push(
            connector, request, list(_SRC_IDS), list(_DST_IDS), _ready_event(device)
        )

    for got, want in zip(staged_dst, direct_dst):
        assert torch.equal(got, want)
    # The direct path really moved something, so equality is not vacuous.
    assert not all(bool((t == 0xAB).all()) for t in direct_dst)
    # One descriptor per region per chunk once adjacent pages merge, against
    # one per token run in place.
    pages = len(_DST_IDS)
    assert nic.descriptors("staged-kv") == _REGIONS * (pages if chunk_pages == 1 else 1)
    assert nic.descriptors("block") == _REGIONS * 10 // interleave


@pytest.mark.parametrize("device", _DEVICES)
def test_without_a_ready_event_the_push_keeps_token_runs(device):
    """Only the ready event orders the gather's side stream behind the batch's
    KV writes, so without one nothing is staged."""
    src = _producer(device)
    dst = _consumer(device)
    nic = _TensorNic(*src, *dst)
    staging = _staging(src, chunk_pages=4)
    nic._tensors.append(staging._pool.view(-1))
    request = _request(dst, dcp_size=4, dcp_rank=3, interleave=1)

    assert _push(
        _connector(src, nic, staging), request, list(_SRC_IDS), list(_DST_IDS), None
    )

    assert nic.descriptors("staged-kv") == 0
    assert nic.descriptors("block") == _REGIONS * 10
    assert not all(bool((t == 0xAB).all()) for t in dst)


@pytest.mark.parametrize("device", _DEVICES)
def test_staged_pages_follow_the_shard_plan(device):
    src = _producer(device)
    staging = _staging(src, chunk_pages=4)
    plan = build_dcp_shard_plan(
        _SRC_IDS, block_size=_BLOCK, dcp_size=4, dcp_rank=1, dst_pages=2
    )
    shard = staging.prepare(plan)
    assert shard.tokens == 10
    assert shard.page_tokens(0, 2).tolist() == [8, 2]

    addr, page_bytes = staging.gather(1, shard, 0, 2, slot=0)
    assert addr == staging._pool[0].data_ptr()
    assert page_bytes == _BPB
    rows = src[1].view(-1, _TOKEN_BYTES)
    # Rank 1 of 4 at interleave 1 owns global tokens 1, 5, 9, ...
    want = torch.stack(
        [rows[_SRC_IDS[g // _BLOCK] * _BLOCK + g % _BLOCK] for g in range(1, 4 * 10, 4)]
    )
    got = staging._pool[0, : 10 * _TOKEN_BYTES].view(10, _TOKEN_BYTES)
    assert torch.equal(got, want)


def test_staging_covers_only_token_contiguous_mla_pages():
    src = _producer("cpu")
    index = torch.zeros((_SRC_BLOCKS, 64), dtype=torch.uint8)
    pages = [
        page_region(src[0], semantic_role=MLA_KV_ROLE),
        page_region(index, semantic_role=INDEX_CACHE_ROLE),
        page_region(src[1], semantic_role=MLA_KV_ROLE),
    ]
    staging = MLAShardStaging.build(
        pages, block_size=_BLOCK, pool_size=1, chunk_pages=2, device=src[0].device
    )
    assert [staging.stages(i) for i in range(3)] == [True, False, True]
    assert staging.slot_bytes >= 2 * _BPB
    assert (
        MLAShardStaging.build(
            pages[1:2],
            block_size=_BLOCK,
            pool_size=1,
            chunk_pages=2,
            device=src[0].device,
        )
        is None
    )


@pytest.mark.parametrize("has_kv_staging", [True, False])
def test_mla_staging_alone_records_kv_cache_ready(monkeypatch, has_kv_staging):
    connector = object.__new__(mc.MooncakeConnector)
    connector.is_producer = True
    connector._index_staging_pool_size = 0
    connector._mla_staging = None
    connector._kv_staging = object() if has_kv_staging else None
    connector._cuda_device = 0
    connector._kv_cache_ready_events = {}
    connector._kv_cache_ready_batches = deque()
    connector._kv_ready_done_event = None
    connector._kv_ready_expired = {}
    connector._completed_prefills_lock = threading.Lock()
    ready_event = MagicMock()
    monkeypatch.setattr(mc.torch.cuda, "Event", MagicMock(return_value=ready_event))
    monkeypatch.setattr(mc.torch.cuda, "current_stream", MagicMock())

    connector.record_kv_cache_ready([11])

    want = {11: ready_event} if has_kv_staging else {}
    assert connector._kv_cache_ready_events == want


def test_a_page_wide_interleave_keeps_whole_page_writes():
    staging = _staging(_producer("cpu"), chunk_pages=4)
    whole = build_dcp_shard_plan(
        _SRC_IDS, block_size=_BLOCK, dcp_size=4, dcp_rank=0, interleave_size=_BLOCK
    )
    split = build_dcp_shard_plan(
        _SRC_IDS, block_size=_BLOCK, dcp_size=4, dcp_rank=0, interleave_size=4
    )
    assert mla_staging_for(staging, whole) is None
    assert mla_staging_for(staging, split) is staging
    assert mla_staging_for(staging, None) is None
    assert mla_staging_for(None, split) is None
