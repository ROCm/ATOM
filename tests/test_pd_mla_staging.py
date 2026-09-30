# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Staged MLA sends to a DCP consumer write exactly the per-token bytes.

The producer's per-token path (one RDMA descriptor per 576-byte MLA token) is
the reference. The staged path gathers a DCP rank's tokens into destination
page order and sends one descriptor per run of adjacent destination pages.
Both drive `_execute_block_transfer` on CPU tensors with a memmove stand-in
for the NIC, so every destination byte is compared, not just the plan.
"""

import ctypes
import threading
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from atom.kv_transfer.disaggregation.sharded_transfer import (
    build_dcp_shard_plan,
    pack_staging_slots,
)
from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE

BLOCK_SIZE = 16
TOKEN_BYTES = 576
PAGE_BYTES = BLOCK_SIZE * TOKEN_BYTES


def _mooncake():
    return pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )


def _expand(starts, lengths):
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lengths)])


# ---------------------------------------------------------------------------
# pure planning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("num_src", [1, 5, 13, 16])
def test_staged_page_runs_cover_exactly_the_token_runs(dcp_size, interleave, num_src):
    rng = np.random.default_rng(num_src)
    src_ids = rng.permutation(64)[:num_src]
    dst_pages = -(-num_src // dcp_size)
    dst_ids = rng.permutation(40)[:dst_pages]
    for rank in range(dcp_size):
        plan = build_dcp_shard_plan(
            src_ids,
            block_size=BLOCK_SIZE,
            dcp_size=dcp_size,
            dcp_rank=rank,
            interleave_size=interleave,
        )
        run_src, run_dst, run_len = plan.token_runs(dst_ids)
        # Token-unit descriptors; staging holds page p at token p*BLOCK_SIZE.
        st_src, st_dst, st_len = plan.staged_page_runs(dst_ids, 0, 0, 1, 1 << 40)
        expected_dst = _expand(run_dst, run_len)
        np.testing.assert_array_equal(_expand(st_dst, st_len), expected_dst)
        # Staged token k of the plan reads the source token the old run did.
        staged_rows = _expand(st_src, st_len)
        src_of_row = plan.source_token_per_dst_token()[staged_rows]
        np.testing.assert_array_equal(src_of_row, _expand(run_src, run_len))


def test_staged_page_runs_stop_at_destination_mr_boundaries():
    """Reviewer case: DCP4 rank 1, contiguous dst pages across a 2 GiB MR."""
    mc = _mooncake()
    conn = object.__new__(mc.MooncakeConnector)
    pages_per_mr = conn._rdma_chunk_units(PAGE_BYTES)
    assert pages_per_mr == 233009
    mr_bytes = pages_per_mr * PAGE_BYTES
    src_ids = np.arange(64)
    dst_ids = np.arange(233001, 233017)
    plan = build_dcp_shard_plan(src_ids, block_size=BLOCK_SIZE, dcp_size=4, dcp_rank=1)
    _, dst, length = plan.staged_page_runs(dst_ids, 0, 0, TOKEN_BYTES, pages_per_mr)
    assert len(dst) == 2  # split exactly once, at page 233009
    assert dst[1] == mr_bytes
    for d, n in zip(dst, length):
        assert d // mr_bytes == (d + n - 1) // mr_bytes
    # Same bytes as without the boundary, just one more descriptor.
    _, merged_dst, merged_len = plan.staged_page_runs(
        dst_ids, 0, 0, TOKEN_BYTES, 1 << 40
    )
    np.testing.assert_array_equal(_expand(dst, length), _expand(merged_dst, merged_len))
    assert len(merged_dst) == 1


@pytest.mark.parametrize("pages_per_mr", [1, 2, 3, 5])
@pytest.mark.parametrize("scattered", [True, False])
def test_staged_mla_descriptors_stay_inside_one_destination_mr(
    monkeypatch, pages_per_mr, scattered
):
    # The consumer's MR chunk is pages_per_mr whole pages (aligned down).
    _transfer_both(
        monkeypatch,
        dcp_size=4,
        dcp_rank=1,
        num_src_total=37,
        scattered=scattered,
        slot_pages=16,
        mr_chunk_bytes=pages_per_mr * PAGE_BYTES + PAGE_BYTES // 2,
    )


def test_staged_index_descriptors_stay_inside_one_destination_mr(monkeypatch):
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    conn = object.__new__(mc.MooncakeConnector)
    conn._MAX_RDMA_CHUNK_BYTES = 2 * 64 + 10  # two 64-byte pages per MR
    conn._acquire_index_staging_slot = lambda: 0
    conn._release_index_staging_slot = lambda _idx: None
    conn._send_worker_stream = _Stream
    conn._gather_sharded_index = lambda *_args: (10000, 5)
    conn._nic = _Nic()
    writes = []
    conn._rdma_write_with_retry = lambda _t, src, dst, sizes, *_a, **_k: (
        writes.append((src, dst, sizes)) or True
    )
    assert conn._execute_staged_index_layer_chunk(
        "consumer:1", 0, 20000, 64, [3, 4, 5, 6, 7], "req", object()
    )
    # Pages 3..7 are contiguous; MR chunks start at pages 4 and 6.
    assert writes == [
        (
            [10000, 10064, 10192],
            [20192, 20256, 20384],
            [64, 128, 128],
        )
    ]


def test_pack_staging_slots_fills_and_splits_regions():
    slots = pack_staging_slots([10, 10, 20], dst_pages=5, slot_bytes=60)
    assert slots == [
        [(0, 0, 5, 0), (1, 0, 1, 50)],
        [(1, 1, 5, 0), (2, 0, 1, 40)],
        [(2, 1, 4, 0)],
        [(2, 4, 5, 0)],
    ]
    covered = {}
    for slot in slots:
        used = 0
        for region, start, stop, offset in slot:
            assert offset == used
            used += (stop - start) * [10, 10, 20][region]
            covered.setdefault(region, []).extend(range(start, stop))
        assert used <= 60
    assert covered == {r: list(range(5)) for r in range(3)}
    # Items from several regions share one slot when they fit.
    assert pack_staging_slots([10, 10], 2, 40) == [[(0, 0, 2, 0), (1, 0, 2, 20)]]
    with pytest.raises(ValueError, match="cannot hold"):
        pack_staging_slots([100], 1, 50)


# ---------------------------------------------------------------------------
# connector: old vs new destination bytes
# ---------------------------------------------------------------------------


class _Nic:
    """RDMA stand-in: memmove every descriptor and count them."""

    def __init__(self):
        self.descriptors = 0
        self.labels = []
        self.dst_spans = []

    def write(self, _target, src, dst, sizes, _req, label, *, engine=None):
        assert len(src) == len(dst) == len(sizes)
        for s, d, n in zip(src, dst, sizes):
            ctypes.memmove(d, s, n)
            self.dst_spans.append((d, n))
        self.descriptors += len(src)
        self.labels.append(label)
        return True


class _Stream:
    def __init__(self):
        self.waited = []
        self.synchronized = 0

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        self.synchronized += 1


def _source_regions(num_regions, num_blocks):
    """MLA regions whose every token is distinct: (region, block, token) tag."""
    gen = torch.Generator().manual_seed(num_regions * 1000 + num_blocks)
    regions = []
    for r in range(num_regions):
        t = torch.randint(0, 256, (num_blocks, PAGE_BYTES), generator=gen).to(
            torch.uint8
        )
        tag = t.view(num_blocks * BLOCK_SIZE, TOKEN_BYTES)[:, :4].view(torch.int32)
        tag[:, 0] = torch.arange(num_blocks * BLOCK_SIZE, dtype=torch.int32) + (r << 20)
        regions.append(t)
    return regions


def _producer(mc, regions, *, pool_size=0, slot_pages=0):
    conn = object.__new__(mc.MooncakeConnector)
    conn.dcp_size = 1
    conn.block_size = BLOCK_SIZE
    conn.kv_caches_base_addr = [t.data_ptr() for t in regions]
    conn._per_block_bytes_list = [PAGE_BYTES] * len(regions)
    conn._block_region_roles = [MLA_KV_ROLE] * len(regions)
    conn._block_region_consumer_indices = None
    conn._consumer_region_map = lambda *_a, **_k: list(range(len(regions)))
    conn._prepare_sharded_index = None
    conn._gather_sharded_index = None
    conn._index_staging_chunk_pages = 0
    conn._nic = _Nic()
    conn._rdma_write_with_retry = conn._nic.write
    if pool_size:
        conn._mla_staging = torch.empty(
            (pool_size, slot_pages * PAGE_BYTES), dtype=torch.uint8
        )
        conn._mla_staging_pool_size = pool_size
        conn._mla_staging_free = list(range(pool_size))
        conn._mla_staging_cv = threading.Condition()
        conn._stream = _Stream()
        conn._send_worker_stream = lambda: conn._stream
        conn._mla_token_views = {
            i: t.view(-1, TOKEN_BYTES) for i, t in enumerate(regions)
        }
    return conn


def _request(num_regions, dcp_size, dcp_rank, interleave, dst_bufs):
    return {
        "consumer_base_addrs": [t.data_ptr() for t in dst_bufs],
        "consumer_num_layers": num_regions,
        "consumer_region_roles": [MLA_KV_ROLE] * num_regions,
        "consumer_block_bpb": [PAGE_BYTES] * num_regions,
        "consumer_dcp_size": dcp_size,
        "consumer_dcp_rank": dcp_rank,
        "consumer_dcp_interleave": interleave,
    }


def _run(mc, conn, request, src_ids, dst_ids, event):
    return mc.MooncakeConnector._execute_block_transfer(
        conn, request, "consumer:1", list(src_ids), list(dst_ids), "req", event
    )


def _transfer_both(
    monkeypatch,
    *,
    dcp_size,
    dcp_rank,
    interleave=1,
    num_regions=3,
    num_src_total=13,
    num_computed_blocks=0,
    scattered=True,
    slot_pages=5,
    pool_size=2,
    mr_chunk_bytes=None,
):
    """Run old and new paths into identically pre-filled destinations."""
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    src_blocks = 40
    regions = _source_regions(num_regions, src_blocks)
    rng = np.random.default_rng(dcp_size * 100 + dcp_rank + num_computed_blocks)
    full_src = rng.permutation(src_blocks)[:num_src_total]
    # The incremental slice _execute_transfer applies at skip == dcp_size.
    src_ids = full_src[num_computed_blocks * dcp_size :]
    dst_pages = -(-len(src_ids) // dcp_size)
    num_dst = 48
    dst_ids = (
        rng.permutation(num_dst)[:dst_pages]
        if scattered
        else np.arange(7, 7 + dst_pages)
    )
    fill = torch.Generator().manual_seed(7)
    dst_old = [
        torch.randint(0, 256, (num_dst, PAGE_BYTES), generator=fill).to(torch.uint8)
        for _ in range(num_regions)
    ]
    dst_new = [t.clone() for t in dst_old]
    before = [t.clone() for t in dst_old]

    old = _producer(mc, regions)
    new = _producer(mc, regions, pool_size=pool_size, slot_pages=slot_pages)
    if mr_chunk_bytes is not None:
        new._MAX_RDMA_CHUNK_BYTES = mr_chunk_bytes
    event = object()
    assert _run(
        mc,
        old,
        _request(num_regions, dcp_size, dcp_rank, interleave, dst_old),
        src_ids,
        dst_ids,
        event,
    )
    assert _run(
        mc,
        new,
        _request(num_regions, dcp_size, dcp_rank, interleave, dst_new),
        src_ids,
        dst_ids,
        event,
    )
    for a, b, orig in zip(dst_old, dst_new, before):
        assert torch.equal(a, b)
        assert not torch.equal(a, orig)
    if mr_chunk_bytes is not None:
        # Destination regions register as MRs of whole pages from their base.
        mr_bytes = new._rdma_chunk_units(PAGE_BYTES) * PAGE_BYTES
        bases = [t.data_ptr() for t in dst_new]
        for addr, size in new._nic.dst_spans:
            base = max(b for b in bases if b <= addr)
            first, last = addr - base, addr - base + size - 1
            assert first // mr_bytes == last // mr_bytes, (first, size)
    return old, new, src_ids, dst_ids


@pytest.mark.parametrize(
    ("dcp_size", "dcp_rank"), [(2, 0), (2, 1), (4, 0), (4, 1), (4, 3)]
)
@pytest.mark.parametrize("num_src_total", [1, 7, 13, 32])
@pytest.mark.parametrize("scattered", [True, False])
def test_staged_mla_matches_per_token_bytes(
    monkeypatch, dcp_size, dcp_rank, num_src_total, scattered
):
    old, new, src_ids, dst_ids = _transfer_both(
        monkeypatch,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        num_src_total=num_src_total,
        scattered=scattered,
    )
    # Every staged descriptor carries at least one page's worth of tokens
    # except a trailing partial page, so descriptors never exceed pages.
    assert new._nic.descriptors <= 3 * len(dst_ids)
    assert set(new._nic.labels) == {"staged-mla"}
    assert new._stream.waited  # ordered after the ready event
    assert sorted(new._mla_staging_free) == [0, 1]  # every slot returned
    # The per-token path sends one descriptor per owned token per region.
    plan = build_dcp_shard_plan(
        src_ids, block_size=BLOCK_SIZE, dcp_size=dcp_size, dcp_rank=dcp_rank
    )
    assert old._nic.descriptors == 3 * int(plan.valid.sum())


@pytest.mark.parametrize("num_computed_blocks", [1, 2, 3])
@pytest.mark.parametrize("dcp_size", [2, 4])
def test_staged_mla_matches_per_token_bytes_incremental(
    monkeypatch, num_computed_blocks, dcp_size
):
    for rank in range(dcp_size):
        _transfer_both(
            monkeypatch,
            dcp_size=dcp_size,
            dcp_rank=rank,
            num_src_total=29,
            num_computed_blocks=num_computed_blocks,
        )


@pytest.mark.parametrize("interleave", [2, 4])
def test_staged_mla_matches_per_token_bytes_interleaved(monkeypatch, interleave):
    for rank in range(4):
        _transfer_both(monkeypatch, dcp_size=4, dcp_rank=rank, interleave=interleave)


@pytest.mark.parametrize("slot_pages", [1, 2, 9, 64])
def test_staged_mla_slot_sizes(monkeypatch, slot_pages):
    _transfer_both(
        monkeypatch, dcp_size=4, dcp_rank=2, num_src_total=37, slot_pages=slot_pages
    )


def test_staged_mla_cuts_descriptors_at_least_16x(monkeypatch):
    """GLM-5.2-like stage: 20 MLA regions, 64K-token request, DCP4."""
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    num_regions, src_blocks, dcp_size = 20, 4096, 4
    # One shared source keeps host memory small; only counts matter here.
    regions = [torch.zeros(src_blocks, PAGE_BYTES, dtype=torch.uint8)] * num_regions
    dst_pages = src_blocks // dcp_size
    dst = [torch.zeros(dst_pages * 2, PAGE_BYTES, dtype=torch.uint8)]
    request = _request(num_regions, dcp_size, 1, 1, dst * num_regions)
    rng = np.random.default_rng(0)
    src_ids = rng.permutation(src_blocks)
    counts = {}
    for name, dst_ids in {
        "scattered": rng.permutation(dst_pages * 2)[:dst_pages],
        "contiguous": np.arange(dst_pages),
    }.items():
        old = _producer(mc, regions)
        new = _producer(mc, regions, pool_size=1, slot_pages=910)  # ~8 MiB slot
        assert _run(mc, old, request, src_ids, dst_ids, object())
        assert _run(mc, new, request, src_ids, dst_ids, object())
        counts[name] = (old._nic.descriptors, new._nic.descriptors)
    old_scattered, new_scattered = counts["scattered"]
    assert old_scattered == num_regions * dst_pages * BLOCK_SIZE
    assert old_scattered >= 16 * new_scattered
    old_contig, new_contig = counts["contiguous"]
    # One descriptor per (region, slot) piece when destination pages are
    # adjacent: 20 regions x 1024 pages over 910-page slots.
    slots = pack_staging_slots([PAGE_BYTES] * num_regions, dst_pages, 910 * PAGE_BYTES)
    assert new_contig == sum(len(s) for s in slots)
    assert old_contig >= 1000 * new_contig


def test_staged_mla_workers_wait_for_a_shared_slot():
    """Fewer slots than workers: a worker blocks until a slot is released."""
    mc = _mooncake()
    conn = _producer(mc, _source_regions(1, 1), pool_size=1, slot_pages=1)
    assert conn._acquire_mla_staging_slot() == 0
    got = []
    waiter = threading.Thread(
        target=lambda: got.append(conn._acquire_mla_staging_slot())
    )
    waiter.start()
    waiter.join(0.2)
    assert waiter.is_alive() and got == []
    conn._release_mla_staging_slot(0)
    waiter.join(5)
    assert got == [0]


def test_send_workers_gather_on_their_own_streams(monkeypatch):
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "Stream", lambda device=None: _Stream())
    conn = object.__new__(mc.MooncakeConnector)
    conn._cuda_device = 0
    conn._send_worker_streams = threading.local()
    mine = conn._send_worker_stream()
    assert conn._send_worker_stream() is mine
    other = []
    worker = threading.Thread(target=lambda: other.append(conn._send_worker_stream()))
    worker.start()
    worker.join()
    assert other[0] is not mine


def test_staged_mla_needs_the_ready_event(monkeypatch):
    mc = _mooncake()
    regions = _source_regions(1, 8)
    conn = _producer(mc, regions, pool_size=1, slot_pages=4)
    dst = [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8)]
    assert _run(mc, conn, _request(1, 2, 0, 1, dst), range(4), [3, 5], None)
    # No ready event: the request keeps the per-token path.
    assert conn._nic.labels == ["block"]
    assert conn._stream.waited == []


def test_staged_mla_leaves_index_regions_to_index_staging(monkeypatch):
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    regions = _source_regions(1, 8)
    conn = _producer(mc, regions, pool_size=1, slot_pages=4)
    index = torch.zeros(8, BLOCK_SIZE * 144, dtype=torch.uint8)
    conn.kv_caches_base_addr.append(index.data_ptr())
    conn._per_block_bytes_list.append(BLOCK_SIZE * 144)
    conn._block_region_roles.append(INDEX_CACHE_ROLE)
    conn._consumer_region_map = lambda *_a, **_k: [0, 1]
    conn._index_staging_chunk_pages = 256
    conn._prepare_sharded_index = lambda plan: plan
    conn._gather_sharded_index = object()
    staged_index = []
    conn._execute_staged_index_layer_chunk = lambda *a, **k: (
        staged_index.append(a[1]) or True
    )
    dst = [
        torch.zeros(8, PAGE_BYTES, dtype=torch.uint8),
        torch.zeros(8, BLOCK_SIZE * 144, dtype=torch.uint8),
    ]
    request = {
        "consumer_base_addrs": [t.data_ptr() for t in dst],
        "consumer_num_layers": 1,
        "consumer_region_roles": [MLA_KV_ROLE, INDEX_CACHE_ROLE],
        "consumer_block_bpb": [PAGE_BYTES, BLOCK_SIZE * 144],
        "consumer_dcp_size": 2,
        "consumer_dcp_rank": 1,
        "consumer_dcp_interleave": 1,
    }
    event = object()
    assert _run(mc, conn, request, range(4), [3, 5], event)
    assert conn._nic.labels == ["staged-mla"]
    assert staged_index == [1]
    # Both staged paths order this worker's stream after the ready event.
    assert conn._stream.waited == [event, event]


def test_mla_staging_is_disabled_by_env(monkeypatch):
    mc = _mooncake()
    monkeypatch.setenv("ATOM_PD_MLA_STAGING", "0")
    conn = object.__new__(mc.MooncakeConnector)
    conn.is_producer = True
    conn.dcp_size = 1
    conn._has_slot_regions = False
    assert conn._build_mla_staging(object()) is None
    assert conn._mla_staging_pool_size == 0


class _DeviceNic:
    """RDMA stand-in for device memory: resolve addresses to tensor bytes."""

    def __init__(self, tensors):
        self.spans = [(t.data_ptr(), t.view(torch.uint8).view(-1)) for t in tensors]
        self.labels = []

    def _at(self, addr, n):
        for base, flat in self.spans:
            if base <= addr and addr + n <= base + flat.numel():
                return flat[addr - base : addr - base + n]
        raise AssertionError(f"address {addr:#x}+{n} outside every buffer")

    def write(self, _target, src, dst, sizes, _req, label, *, engine=None):
        for s, d, n in zip(src, dst, sizes):
            self._at(d, n).copy_(self._at(s, n))
        self.labels.append(label)
        return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dcp_rank", [0, 3])
def test_staged_mla_gathers_on_the_gpu(dcp_rank):
    mc = _mooncake()
    device = torch.device("cuda", torch.cuda.current_device())
    num_regions, src_blocks, dcp_size = 2, 24, 4
    regions = [t.to(device) for t in _source_regions(num_regions, src_blocks)]
    src_ids = list(range(src_blocks - 1, 0, -2))  # 12 blocks, reversed
    dst_ids = [5, 6, 1]
    dst_old = [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8, device=device)]
    dst_old.append(torch.zeros_like(dst_old[0]))
    dst_new = [torch.zeros_like(t) for t in dst_old]
    old = _producer(mc, regions)
    new = _producer(mc, regions, pool_size=1, slot_pages=2)
    new._mla_staging = new._mla_staging.to(device)
    new._cuda_device = device.index
    new._send_worker_streams = threading.local()
    del new._send_worker_stream  # the real per-worker stream
    old._rdma_write_with_retry = _DeviceNic(regions + dst_old).write
    nic = _DeviceNic(regions + dst_new + [new._mla_staging])
    new._rdma_write_with_retry = nic.write
    event = torch.cuda.Event()
    event.record()
    for conn, dst in ((old, dst_old), (new, dst_new)):
        assert _run(
            mc,
            conn,
            _request(num_regions, dcp_size, dcp_rank, 1, dst),
            src_ids,
            dst_ids,
            event,
        )
    torch.cuda.synchronize()
    assert set(nic.labels) == {"staged-mla"}
    for a, b in zip(dst_old, dst_new):
        assert torch.equal(a, b)
        assert a.any()
