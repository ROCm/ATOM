# SPDX-License-Identifier: MIT
"""Ownership is selected by topology, with physical layouts as separate inputs."""

import ctypes

import numpy as np
import pytest
import torch

from atom.kv_transfer.disaggregation.gather import (
    GatherLayout,
    _src_offsets,
    gather_units,
)
from atom.kv_transfer.disaggregation.sharded_transfer import (
    coalesce_contiguous,
    consumer_head_slices,
    kv_heads_of_rank,
)
from atom.kv_transfer.disaggregation.transfer_plan import (
    PAGE_UNITS,
    TOKEN_UNITS,
    PlannedRegion,
    RegionLayout,
    TransferTopology,
    build_transfer_plan,
    page_transfer_plan,
)


def reference_gather_units(source, indices, out, plan):
    """``gather_units`` on host tensors: its non-whole-unit gathers are GPU kernels."""
    src = source.reshape(-1, plan.source_bytes)
    dst = out.view(indices.numel(), plan.unit_bytes)
    at = 0
    for first, size in plan.segments:
        dst[:, at : at + size] = src[indices, first : first + size]
        at += size


def execute_plan(plan, source, destination):
    """Exercise both layout primitives and real descriptors against a byte oracle."""
    for items in plan.pack(512):
        staging = torch.empty(512, dtype=torch.uint8)
        for pos, start, stop, offset in items:
            layout = plan.regions[pos].gather
            gather = gather_units if layout.whole_unit else reference_gather_units
            gather(
                source,
                torch.from_numpy(plan.source_units[start:stop]),
                staging[offset : offset + (stop - start) * layout.unit_bytes],
                layout,
            )
        src, dst, sizes = plan.direct_descriptors(
            items, staging.data_ptr(), lambda _: 2
        )
        for s, d, n in zip(src, dst, sizes):
            assert (
                destination.data_ptr()
                <= d
                < d + n
                <= destination.data_ptr() + destination.numel()
            )
            ctypes.memmove(d, s, n)


@pytest.mark.parametrize("rank", range(4))
def test_dcp_rank_owns_its_interleaved_token_rows(rank):
    source = (
        torch.arange(4 * 8 * 4 * 4, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(4, 8, 4, 4)
    )
    ids = [3, 0, 2]
    destination = torch.zeros(5, 8, 4, 4, dtype=torch.uint8)
    dst_ids = [3]
    topology = TransferTopology(target_dcp=4, target_dcp_rank=rank, interleave=2)
    layout = RegionLayout(0, 7, destination.data_ptr(), 128, 128)
    ordered = source[ids].reshape(-1, 4, 4)
    expected = ordered[(torch.arange(24) // 2) % 4 == rank]
    plan = build_transfer_plan(topology, [layout], ids, dst_ids, 8)
    assert plan.wire_layout == TOKEN_UNITS
    execute_plan(plan, source, destination)
    actual = destination[dst_ids].reshape(-1)
    want = expected.reshape(-1)
    assert torch.equal(actual[: want.numel()], want)
    assert not actual[want.numel() :].any()  # partial DCP PAGE tail stays untouched
    assert not destination[list(set(range(5)) - set(dst_ids))].any()
    assert plan.regions[0].target_region == 7  # explicit PP region mapping


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("heads", [4, 2])
def test_tp_rank_owns_its_heads_of_each_page(rank, heads):
    # TP4 splits four KV heads or replicates two. No attention-family input.
    source = (
        torch.arange(4 * 8 * heads * 4, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(4, 8, heads, 4)
    )
    ids = [3, 0, 2]
    destination = torch.zeros(5, 8, 1, 4, dtype=torch.uint8)
    dst_ids = [4, 1, 3]
    cuts = consumer_head_slices(
        32 * heads, 32, (heads, (4,) * 8), (1, (4,) * 8), rank, 4, "page"
    )
    gather = GatherLayout.from_slices(32 * heads, 32, cuts)
    head = rank * heads // 4
    # Equal runs are segments too: one per run, `heads` 4 B heads apart.
    assert gather.segments == tuple((r * heads * 4 + head * 4, 4) for r in range(8))
    plan = page_transfer_plan(
        [PlannedRegion(0, 7, destination.data_ptr(), gather)], ids, dst_ids
    )
    assert plan.wire_layout == PAGE_UNITS
    execute_plan(plan, source, destination)
    assert torch.equal(destination[dst_ids], source[ids, :, head : head + 1, :])
    assert not destination[list(set(range(5)) - set(dst_ids))].any()
    assert plan.regions[0].target_region == 7  # explicit PP region mapping


@pytest.mark.parametrize("rank", range(4))
def test_named_run_widths_gather_into_one_target_page(rank):
    # NVFP4-like page: runs of 8 B and 4 B per head (data, scales) for K then V.
    widths = (8, 4, 8, 4)
    heads = 4
    source = (
        torch.arange(5 * 24 * heads, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(5, 24 * heads)
    )
    ids = [3, 0, 2]
    destination = torch.zeros(5, 24, dtype=torch.uint8)
    dst_ids = [4, 1, 3]
    cuts = consumer_head_slices(
        24 * heads, 24, (heads, widths), (1, widths), rank, 4, "page"
    )
    gather = GatherLayout.from_slices(24 * heads, 24, cuts)
    assert len(gather.segments) == 4
    plan = page_transfer_plan(
        [PlannedRegion(0, 7, destination.data_ptr(), gather)], ids, dst_ids
    )
    execute_plan(plan, source, destination)
    want = []
    before = 0
    for w in widths:
        want.append(
            source[ids, before * heads + rank * w : before * heads + (rank + 1) * w]
        )
        before += w
    assert torch.equal(destination[dst_ids], torch.cat(want, dim=1))
    assert not destination[list(set(range(5)) - set(dst_ids))].any()


def test_a_contiguous_cut_gathers_the_whole_unit():
    # One KV head replicated to every TP4 rank: each rank's cut is the whole page.
    cuts = consumer_head_slices(
        24, 24, (1, (8, 4, 8, 4)), (1, (8, 4, 8, 4)), 2, 4, "kv"
    )
    assert len(cuts) == 4
    layout = GatherLayout.from_slices(24, 24, cuts)
    assert layout.segments == ((0, 24),) and layout.whole_unit
    source = torch.arange(3 * 24, dtype=torch.uint8)
    out = torch.empty(2 * 24, dtype=torch.uint8)
    gather_units(source, torch.tensor([2, 0]), out, layout)  # index_select, no Triton
    assert torch.equal(out, source.view(3, 24)[[2, 0]].reshape(-1))
    # Two of four heads in one run: the rank's heads merge into one segment.
    cuts = consumer_head_slices(64, 32, (4, (16,)), (2, (16,)), 1, 2, "kv")
    assert GatherLayout.from_slices(64, 32, cuts).segments == ((32, 32),)


@pytest.mark.parametrize("slices", [[], np.empty((0, 3), dtype=np.int64)])
def test_an_empty_cut_is_refused(slices):
    with pytest.raises(ValueError, match="0 gather slices"):
        GatherLayout.from_slices(24, 24, slices)


# TP1 pages as `(each run's width per head, heads)`, cut for every TP2 / TP4 rank.
CUTS = {
    "eight-equal-runs": ((128,) * 8, 4),
    "two-named-runs": ((64, 16), 4),
    "nvfp4": ((8192, 1024, 8192, 1024), 4),
    "five-named-runs": ((8, 4, 8, 4, 4), 4),
    "one-run": ((2048,), 4),
    "nvfp4-two-heads": ((8192, 1024, 8192, 1024), 2),
    "kv-data-plane": ((512,) * 16, 4),
    "many-short-runs": ((4,) * 96, 4),
    "one-long-run": ((40004,), 2),
}


def _cut_layouts(name):
    widths, heads = CUTS[name]
    for tp in (2, 4):
        for rank in range(tp):
            count = kv_heads_of_rank(rank, tp, heads)[1]
            src, dst = heads * sum(widths), count * sum(widths)
            cuts = consumer_head_slices(
                src, dst, (heads, widths), (count, widths), rank, tp, name
            )
            yield GatherLayout.from_slices(src, dst, cuts)


@pytest.mark.parametrize("name", CUTS)
def test_offset_table_lays_the_segments_end_to_end(name):
    for layout in _cut_layouts(name):
        src = _src_offsets(layout).astype(np.int64)
        assert src.size == layout.unit_bytes // 4  # one per destination word
        assert src.min() >= 0 and src.max() < layout.source_bytes // 4
        dst = np.arange(src.size)
        src, _, words = coalesce_contiguous(src, dst, np.ones_like(src))
        assert tuple(zip((4 * src).tolist(), (4 * words).tolist())) == layout.segments


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton gather kernel")
@pytest.mark.parametrize("name", CUTS)
def test_gather_kernel_matches_the_host_reference(name):
    # Every TP2 and TP4 rank's cut: up to 96 segments, or one spanning 40 blocks.
    for layout in _cut_layouts(name):
        generator = torch.Generator().manual_seed(layout.source_bytes)
        source = torch.randint(
            0, 256, (6 * layout.source_bytes,), dtype=torch.uint8, generator=generator
        )
        indices = torch.tensor([4, 0, 5, 2])
        want = torch.empty(indices.numel() * layout.unit_bytes, dtype=torch.uint8)
        reference_gather_units(source, indices, want, layout)
        out = torch.full_like(want, 0xA5, device="cuda")
        gather_units(source.cuda(), indices.cuda(), out, layout)
        assert torch.equal(out.cpu(), want), layout


def test_tp_replicated_units_can_also_be_partitioned_by_dcp():
    # A replicated latent/auxiliary plane has no per-head TP axis, so DCP
    # alone selects its tokens: the topology names no TP size.
    source = torch.arange(64, dtype=torch.uint8).view(2, 8, 4)
    destination = torch.full((3, 8, 4), 255, dtype=torch.uint8)
    layout = RegionLayout(0, 0, destination.data_ptr(), 32, 32)
    plan = build_transfer_plan(
        TransferTopology(target_dcp=2, target_dcp_rank=1),
        [layout],
        [1, 0],
        [2],
        8,
    )
    execute_plan(plan, source, destination)
    assert plan.wire_layout == TOKEN_UNITS
    assert torch.equal(destination[2], source[[1, 0]].reshape(16, 4)[1::2])
    assert (destination[:2] == 255).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU buffer allocation")
@pytest.mark.parametrize("page_units", [False, True])
def test_shared_buffer_builders_respect_reserved_memory(page_units):
    from types import SimpleNamespace

    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
        MooncakeConnector,
    )
    from atom.kv_transfer.disaggregation.staging_config import LandingSpec
    from atom.kv_transfer.disaggregation.types import MLA_KV_ROLE, KVTransferRegion

    source = torch.empty((4, 128), dtype=torch.uint8, device="cuda")
    role = "target.kv" if page_units else MLA_KV_ROLE
    c = object.__new__(MooncakeConnector)
    c._kv_staging_reserve = 8 << 20
    c._num_send_workers = 4  # Must share the single reserved source slot.
    c._page_staging_enabled = page_units
    c._has_slot_regions = False
    region = KVTransferRegion(
        source.data_ptr(), source.numel(), 128, semantic_role=role
    )
    staging = c._build_kv_staging(
        SimpleNamespace(block_regions=[region], block_tensor_views=[source])
    )
    assert staging.total_bytes == c._kv_staging_reserve
    assert c._kv_staging_slots.size == 1
    assert staging.base_addr == c._kv_staging.data_ptr()
    c._landing_spec = LandingSpec(
        1 if page_units else 8,
        16,
        PAGE_UNITS if page_units else TOKEN_UNITS,
        page_units,
    )
    c._block_region_roles = [role]
    c._per_block_bytes_list = [128]
    c.kv_caches_base_addr = [source.data_ptr()]
    c._cuda_device = torch.cuda.current_device()
    c.local_ip, c.rpc_port = "test", 1
    c._send_on_socket = lambda *args: None
    c._complete_recv = lambda *args: None
    receiver = c._build_landing()
    assert receiver.units_per_page == c._landing_spec.units_per_page
    assert receiver.region().total_bytes == 256 << 20
    assert receiver._table_dev.numel() * 8 == 16 * 24
