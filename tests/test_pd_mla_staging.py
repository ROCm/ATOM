# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Staged MLA sends to a DCP consumer write exactly the per-token bytes.

The producer's per-token path (one RDMA descriptor per 576-byte MLA token) is
the reference. The staged path gathers a DCP rank's tokens into destination
page order and sends one descriptor per run of adjacent destination pages.
Both drive `_execute_block_transfer` on CPU tensors with a memmove stand-in
for the NIC, so every destination byte is compared, not just the plan.
"""

import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from kv_staging_helpers import (
    BLOCK_SIZE,
    PAGE_BYTES,
    TOKEN_BYTES,
    Nic,
    Stream,
    make_producer,
    mla_producer_config,
    mooncake,
    run_transfer,
    source_regions,
    write_request,
)

from atom.kv_transfer.disaggregation.sharded_transfer import (
    build_dcp_shard_plan,
    pack_slots,
)
from atom.kv_transfer.disaggregation.transfer_plan import (
    RegionLayout,
    TransferTopology,
    build_transfer_plan,
)
from atom.kv_transfer.disaggregation.types import INDEX_CACHE_ROLE, MLA_KV_ROLE


def _expand(starts, lengths):
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lengths)])


# ---------------------------------------------------------------------------
# pure planning
# ---------------------------------------------------------------------------


def _token_plan(src_ids, dst_ids, *, dcp_size, dcp_rank, interleave=1, token_bytes=4):
    page_bytes = BLOCK_SIZE * token_bytes
    return build_transfer_plan(
        TransferTopology(
            target_dcp=dcp_size, target_dcp_rank=dcp_rank, interleave=interleave
        ),
        [RegionLayout(0, 0, 0, page_bytes, page_bytes)],
        src_ids,
        dst_ids,
        BLOCK_SIZE,
    )


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("num_src", [1, 5, 13, 16])
def test_direct_descriptors_cover_exactly_the_token_runs(dcp_size, interleave, num_src):
    rng = np.random.default_rng(num_src)
    src_ids = rng.permutation(64)[:num_src]
    dst_pages = -(-num_src // dcp_size)
    dst_ids = rng.permutation(40)[:dst_pages]
    for rank in range(dcp_size):
        sharded = build_dcp_shard_plan(
            src_ids,
            block_size=BLOCK_SIZE,
            dcp_size=dcp_size,
            dcp_rank=rank,
            interleave_size=interleave,
        )
        run_src, run_dst, run_len = sharded.token_runs(dst_ids)
        plan = _token_plan(
            src_ids, dst_ids, dcp_size=dcp_size, dcp_rank=rank, interleave=interleave
        )
        # One slot holding every row from staging 0; 4-byte tokens to units.
        (items,) = plan.pack(1 << 40)
        st_src, st_dst, st_len = (
            np.array(part) // 4
            for part in plan.direct_descriptors(items, 0, lambda _: 1 << 40)
        )
        np.testing.assert_array_equal(
            _expand(st_dst, st_len), _expand(run_dst, run_len)
        )
        # Staged row k carries the source token the per-token run did.
        staged_rows = _expand(st_src, st_len)
        np.testing.assert_array_equal(
            plan.source_units[staged_rows], _expand(run_src, run_len)
        )


def test_direct_descriptors_stop_at_destination_mr_boundaries():
    """Reviewer case: DCP4 rank 1, contiguous dst pages across a 2 GiB MR."""
    mc = mooncake()
    conn = object.__new__(mc.MooncakeConnector)
    pages_per_mr = conn._rdma_chunk_units(PAGE_BYTES)
    assert pages_per_mr == 233009
    mr_bytes = pages_per_mr * PAGE_BYTES
    dst_ids = np.arange(233001, 233017)
    plan = _token_plan(
        np.arange(64), dst_ids, dcp_size=4, dcp_rank=1, token_bytes=TOKEN_BYTES
    )
    (items,) = plan.pack(1 << 40)
    _, dst, length = plan.direct_descriptors(items, 0, lambda _: pages_per_mr)
    assert len(dst) == 2  # split exactly once, at page 233009
    assert dst[1] == mr_bytes
    for d, n in zip(dst, length):
        assert d // mr_bytes == (d + n - 1) // mr_bytes
    # Same bytes as without the boundary, just one more descriptor.
    _, merged_dst, merged_len = plan.direct_descriptors(items, 0, lambda _: 1 << 40)
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
    mc = mooncake(monkeypatch)
    conn = object.__new__(mc.MooncakeConnector)
    conn._MAX_RDMA_CHUNK_BYTES = 2 * 64 + 10  # two 64-byte pages per MR
    conn._index_staging_slots = mc.SlotPool(1)
    conn._send_worker_stream = Stream
    conn._gather_sharded_index = lambda *_args: (10000, 5)
    conn._nic = Nic()
    writes = []
    conn._rdma_write = lambda _t, src, dst, sizes, *_a, **_k: (
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


def test_pack_slots_fills_and_splits_regions():
    slots = pack_slots([10, 10, 20], num_units=5, slot_bytes=60, align=1)
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
    assert pack_slots([10, 10], 2, 40, 1) == [[(0, 0, 2, 0), (1, 0, 2, 20)]]
    with pytest.raises(ValueError, match="cannot hold"):
        pack_slots([100], 1, 50, 1)


# ---------------------------------------------------------------------------
# connector: old vs new destination bytes
# ---------------------------------------------------------------------------


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
    mc = mooncake(monkeypatch)
    src_blocks = 40
    regions = source_regions(num_regions, src_blocks)
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

    old = make_producer(mc, regions)
    new = make_producer(
        mc, regions, slots=pool_size, slot_bytes=slot_pages * PAGE_BYTES
    )
    if mr_chunk_bytes is not None:
        new._MAX_RDMA_CHUNK_BYTES = mr_chunk_bytes
    event = object()
    assert run_transfer(
        mc,
        old,
        write_request(num_regions, dcp_size, dcp_rank, dst_old, interleave=interleave),
        src_ids,
        dst_ids,
        event,
    )
    assert run_transfer(
        mc,
        new,
        write_request(num_regions, dcp_size, dcp_rank, dst_new, interleave=interleave),
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
    assert set(new._nic.labels) == {"staged-kv"}
    assert new._stream.waited  # ordered after the ready event
    assert sorted(new._kv_staging_slots.free) == [0, 1]  # every slot returned
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
    mc = mooncake(monkeypatch)
    num_regions, src_blocks, dcp_size = 20, 4096, 4
    # One shared source keeps host memory small; only counts matter here.
    regions = [torch.zeros(src_blocks, PAGE_BYTES, dtype=torch.uint8)] * num_regions
    dst_pages = src_blocks // dcp_size
    dst = [torch.zeros(dst_pages * 2, PAGE_BYTES, dtype=torch.uint8)]
    request = write_request(num_regions, dcp_size, 1, dst * num_regions)
    rng = np.random.default_rng(0)
    src_ids = rng.permutation(src_blocks)
    counts = {}
    for name, dst_ids in {
        "scattered": rng.permutation(dst_pages * 2)[:dst_pages],
        "contiguous": np.arange(dst_pages),
    }.items():
        old = make_producer(mc, regions)
        new = make_producer(
            mc, regions, slots=1, slot_bytes=910 * PAGE_BYTES  # ~8 MiB slot
        )
        assert run_transfer(mc, old, request, src_ids, dst_ids, object())
        assert run_transfer(mc, new, request, src_ids, dst_ids, object())
        counts[name] = (old._nic.descriptors, new._nic.descriptors)
    old_scattered, new_scattered = counts["scattered"]
    assert old_scattered == num_regions * dst_pages * BLOCK_SIZE
    assert old_scattered >= 16 * new_scattered
    old_contig, new_contig = counts["contiguous"]
    # One descriptor per (region, slot) piece when destination pages are
    # adjacent: 20 regions x 1024 pages over 910-page slots.
    slots = pack_slots([PAGE_BYTES] * num_regions, dst_pages, 910 * PAGE_BYTES, 1)
    assert new_contig == sum(len(s) for s in slots)
    assert old_contig >= 1000 * new_contig


def test_staged_mla_workers_wait_for_a_shared_slot():
    """Fewer slots than workers: a worker blocks until a slot is released."""
    mc = mooncake()
    conn = make_producer(mc, source_regions(1, 1), slots=1, slot_bytes=PAGE_BYTES)
    pool = conn._kv_staging_slots
    assert pool.acquire() == 0
    got = []
    waiter = threading.Thread(target=lambda: got.append(pool.acquire()))
    waiter.start()
    waiter.join(0.2)
    assert waiter.is_alive() and got == []
    pool.release(0)
    waiter.join(5)
    assert got == [0]


@pytest.mark.parametrize("landing", [False, True], ids=["staged", "landed"])
def test_a_failed_gather_ends_before_its_slot_is_released(monkeypatch, landing):
    """A staging slot goes back only once no gather queued into it can run.

    Another send worker may take the slot as soon as it is released, so a
    gather that raises still has its stream synchronized first.
    """
    mc = mooncake(monkeypatch)
    conn = make_producer(mc, source_regions(2, 8), slots=1, slot_bytes=4 * PAGE_BYTES)
    dst = [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8) for _ in range(2)]
    request = write_request(2, 2, 0, dst)
    if landing:
        monkeypatch.setattr(mc, "LANDING_MIN_SLOTS", 1)
        offer = {"epoch": 1, "base": 0, "slot_bytes": 4 * PAGE_BYTES, "slots": [0]}
        conn._landing_credits.sync("consumer:1", offer)
        request["mla_landing"] = offer
    events = []
    conn._stream.synchronize = lambda: events.append("sync")
    release = conn._kv_staging_slots.release

    def release_slot(idx):
        events.append("release")
        release(idx)

    conn._kv_staging_slots.release = release_slot
    index_select = torch.index_select
    calls = []

    def second_gather_fails(*args, **kwargs):
        # Both regions share the slot: the first gather is already queued.
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("gather failed")
        return index_select(*args, **kwargs)

    monkeypatch.setattr(mc.torch, "index_select", second_gather_fails)
    with pytest.raises(RuntimeError, match="gather failed"):
        run_transfer(mc, conn, request, range(4), [3, 5], object())
    assert events[:2] == ["sync", "release"]
    assert events[-1] == "sync"  # the shared index upload lifetime also ends
    assert conn._kv_staging_slots.free == [0]
    assert conn._nic.labels == []  # gather failed before any RDMA submission
    if landing:
        # Never announced, so the credit went back too.
        assert conn._landing_credits.acquire("consumer:1", 1, 0) == 0


def test_a_failed_index_gather_ends_before_its_slot_is_released(monkeypatch):
    mc = mooncake(monkeypatch)
    conn = object.__new__(mc.MooncakeConnector)
    events = []
    stream = Stream()
    stream.synchronize = lambda: events.append("sync")
    conn._send_worker_stream = lambda: stream
    conn._index_staging_slots = mc.SlotPool(1)
    conn._index_staging_slots.release = lambda _idx: events.append("release")

    def gather_fails(*_args):
        raise RuntimeError("gather failed")

    conn._gather_sharded_index = gather_fails
    with pytest.raises(RuntimeError, match="gather failed"):
        conn._execute_staged_index_layer_chunk(
            "consumer:1", 0, 20000, 64, [3], "req", object()
        )
    assert events == ["sync", "release"]


def test_staged_mla_needs_the_ready_event(monkeypatch):
    mc = mooncake()
    regions = source_regions(1, 8)
    conn = make_producer(mc, regions, slots=1, slot_bytes=4 * PAGE_BYTES)
    dst = [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8)]
    assert run_transfer(mc, conn, write_request(1, 2, 0, dst), range(4), [3, 5], None)
    # No ready event: the request keeps the per-token path.
    assert conn._nic.labels == ["block"]
    assert conn._stream.waited == []


def test_staged_mla_builds_no_per_token_descriptors(monkeypatch):
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    conn = _producer(mc, _source_regions(3, 8), pool_size=1, slot_pages=4)

    def per_token(region_idx, *_args):
        raise AssertionError(f"staged region {region_idx} built token descriptors")

    conn._dcp_token_descriptors = per_token
    dst = [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8) for _ in range(3)]
    assert _run(mc, conn, _request(3, 2, 1, 1, dst), range(8), [3, 5, 6, 7], object())
    assert set(conn._nic.labels) == {"staged-kv"}


def test_staged_mla_still_refuses_a_non_mla_region(monkeypatch):
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    conn = _producer(mc, _source_regions(1, 8), pool_size=1, slot_pages=4)
    conn._block_region_roles = ["state"]
    dst = [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8)]
    request = _request(1, 2, 0, 1, dst)
    request["consumer_region_roles"] = ["state"]
    with pytest.raises(RuntimeError, match="relayout refuses region 0"):
        _run(mc, conn, request, range(4), [3, 5], object())
    assert conn._nic.labels == []


def test_a_landing_offer_for_another_unit_layout_fails_before_any_rdma():
    mc = mooncake()
    regions = source_regions(1, 8)
    conn = make_producer(mc, regions, slots=1, slot_bytes=4 * PAGE_BYTES)
    request = write_request(1, 2, 1, [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8)])
    request["mla_landing"] = {
        "epoch": 1,
        "base": 0,
        "slot_bytes": 4 * PAGE_BYTES,
        "slots": [0],
        "layout": "mha_pages_v1",
    }
    with pytest.raises(ValueError, match="wire layout mismatch"):
        run_transfer(mc, conn, request, range(4), [3, 5], object())
    assert conn._nic.labels == []
    assert conn._kv_staging_slots.free == [0]


def _per_token_reference(mc, regions):
    expected = [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8) for _ in regions]
    old = make_producer(mc, regions)
    assert run_transfer(
        mc, old, write_request(2, 2, 1, expected), range(4), [3, 5], None
    )
    return expected


def test_a_failed_write_returns_its_slot_and_the_next_transfer_stages(
    monkeypatch, caplog
):
    """The NIC only reads a staging slot, so a failed write gives it back at
    once; its late bytes can land only where the failed request was going."""
    mc = mooncake(monkeypatch)
    regions = source_regions(2, 8)
    expected = _per_token_reference(mc, regions)
    new = make_producer(mc, regions, slots=1, slot_bytes=4 * PAGE_BYTES)
    new._rdma_write = lambda *_a, **_k: False
    failed = [torch.zeros_like(t) for t in expected]
    with caplog.at_level(logging.ERROR, logger="atom"):
        assert not run_transfer(
            mc, new, write_request(2, 2, 1, failed), range(4), [3, 5], object()
        )
    assert any("staged-kv transfer failed for req" in r.message for r in caplog.records)
    assert new._kv_staging_slots.free == [0]
    new._rdma_write = new._nic.write
    actual = [torch.zeros_like(t) for t in expected]
    assert run_transfer(
        mc, new, write_request(2, 2, 1, actual), range(4), [3, 5], object()
    )
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    assert new._nic.labels == ["staged-kv"]


def test_staged_chunks_write_the_bytes_of_the_whole_transfer(monkeypatch):
    """A chunked transfer stages each chunk's destination pages on their own."""
    mc = mooncake(monkeypatch)
    regions = source_regions(2, 8)
    expected = _per_token_reference(mc, regions)
    new = make_producer(mc, regions, slots=1, slot_bytes=4 * PAGE_BYTES)
    actual = [torch.zeros_like(t) for t in expected]
    # DCP2: destination page k takes source blocks 2k and 2k + 1.
    assert run_transfer(
        mc, new, write_request(2, 2, 1, actual), range(2), [3], object()
    )
    assert run_transfer(
        mc, new, write_request(2, 2, 1, actual), range(2, 4), [5], object()
    )
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    assert new._nic.labels == ["staged-kv", "staged-kv"]


def test_staged_mla_leaves_index_regions_to_index_staging(monkeypatch):
    mc = mooncake(monkeypatch)
    regions = source_regions(1, 8)
    conn = make_producer(mc, regions, slots=1, slot_bytes=4 * PAGE_BYTES)
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
    assert run_transfer(mc, conn, request, range(4), [3, 5], event)
    assert conn._nic.labels == ["staged-kv"]
    assert staged_index == [1]
    # Both staged paths order this worker's stream after the ready event.
    assert conn._stream.waited == [event, event]


def _build_kv_staging(mc, region, view, *, workers, reserve):
    """``register_kv_caches``'s KV staging pool for one PAGE region."""
    conn = object.__new__(mc.MooncakeConnector)
    conn._has_slot_regions = False
    conn._page_staging_enabled = False
    conn._num_send_workers = workers
    conn._kv_staging_reserve = reserve
    tt = SimpleNamespace(block_regions=[region], block_tensor_views=[view])
    return conn, conn._build_kv_staging(tt)


def test_staging_gathers_bytes_whatever_the_kv_dtype(monkeypatch):
    # page_region publishes every PAGE view as a uint8 alias, so a bf16 KV
    # cache's rows are still one token's bytes; a view of another dtype would
    # be reshaped wrong, so it keeps the per-token path instead.
    from atom.kv_transfer.disaggregation.page_region import page_region

    mc = mooncake()
    monkeypatch.delenv("ATOM_PD_MLA_STAGING", raising=False)
    kv = torch.zeros((4, 16, 576), dtype=torch.bfloat16)
    published = page_region(kv, semantic_role=MLA_KV_ROLE)
    assert published.view.dtype == torch.uint8
    assert published.region.unit_bytes == 16 * 576 * 2

    class _Bf16View:
        is_cuda = True
        dtype = torch.bfloat16

        def view(self, *_shape):
            raise AssertionError("a non-byte view must not be reshaped")

    conn, staging = _build_kv_staging(
        mc, published.region, _Bf16View(), workers=16, reserve=16 * (8 << 20)
    )
    assert staging is None
    assert conn._kv_staging is None


@pytest.mark.parametrize("enabled", [True, False])
def test_kv_staging_follows_the_env_through_its_reserve(monkeypatch, enabled):
    from atom.kv_transfer.disaggregation.staging_config import staging_reserve_bytes
    from atom.kv_transfer.disaggregation.types import KVTransferRegion

    mc = mooncake()
    monkeypatch.setenv("ATOM_PD_MLA_STAGING", "1" if enabled else "0")

    class _DeviceView:
        # What a GPU PAGE view reports, backed by host memory here.
        is_cuda = True
        dtype = torch.uint8
        device = torch.device("cpu")

        def is_contiguous(self):
            return True

    region = KVTransferRegion(0, 8 * PAGE_BYTES, PAGE_BYTES, semantic_role=MLA_KV_ROLE)
    # The reserve is what __init__ sets from the connector's config.
    conn, staging = _build_kv_staging(
        mc,
        region,
        _DeviceView(),
        workers=1,
        reserve=staging_reserve_bytes(mla_producer_config(1)),
    )
    assert (staging is not None) is enabled
    assert (conn._kv_staging is not None) is enabled


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

    def write(
        self, _target, src, dst, sizes, _req, label, *, engine=None, deadline=None
    ):
        for s, d, n in zip(src, dst, sizes):
            self._at(d, n).copy_(self._at(s, n))
        if torch.cuda.is_available():
            # Synchronous like the real write: the caller reuses the staging
            # slot as soon as this returns, from another stream.
            torch.cuda.current_stream().synchronize()
        self.labels.append(label)
        return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dcp_rank", [0, 3])
def test_staged_mla_gathers_on_the_gpu(dcp_rank):
    mc = mooncake()
    device = torch.device("cuda", torch.cuda.current_device())
    num_regions, src_blocks, dcp_size = 2, 24, 4
    regions = [t.to(device) for t in source_regions(num_regions, src_blocks)]
    src_ids = list(range(src_blocks - 1, 0, -2))  # 12 blocks, reversed
    dst_ids = [5, 6, 1]
    dst_old = [torch.zeros(8, PAGE_BYTES, dtype=torch.uint8, device=device)]
    dst_old.append(torch.zeros_like(dst_old[0]))
    dst_new = [torch.zeros_like(t) for t in dst_old]
    old = make_producer(mc, regions)
    new = make_producer(mc, regions, slots=1, slot_bytes=2 * PAGE_BYTES)
    new._kv_staging = new._kv_staging.to(device)
    new._cuda_device = device.index
    new._send_worker_streams = threading.local()
    del new._send_worker_stream  # the real per-worker stream
    old._rdma_write = _DeviceNic(regions + dst_old).write
    nic = _DeviceNic(regions + dst_new + [new._kv_staging])
    new._rdma_write = nic.write
    event = torch.cuda.Event()
    event.record()
    for conn, dst in ((old, dst_old), (new, dst_new)):
        assert run_transfer(
            mc,
            conn,
            write_request(num_regions, dcp_size, dcp_rank, dst),
            src_ids,
            dst_ids,
            event,
        )
    torch.cuda.synchronize()
    assert set(nic.labels) == {"staged-kv"}
    for a, b in zip(dst_old, dst_new):
        assert torch.equal(a, b)
        assert a.any()
