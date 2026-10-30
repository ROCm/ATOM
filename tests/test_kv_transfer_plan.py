# SPDX-License-Identifier: MIT
"""Ownership is selected by topology, with physical layouts as separate inputs."""

import ctypes
from dataclasses import replace

import pytest
import torch

from atom.kv_transfer.disaggregation.gather import gather_units
from atom.kv_transfer.disaggregation.transfer_plan import (
    PAGE_UNITS,
    TOKEN_UNITS,
    RegionLayout,
    TransferTopology,
    build_transfer_plan,
)


def execute_plan(plan, source, destination):
    """Exercise both layout primitives and real descriptors against a byte oracle."""
    for items in plan.pack(512):
        staging = torch.empty(512, dtype=torch.uint8)
        for pos, start, stop, offset in items:
            layout = plan.regions[pos].gather
            gather_units(
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
@pytest.mark.parametrize("ownership", ["heads", "tokens"])
def test_same_multihead_layout_changes_ownership_with_peer_topology(rank, ownership):
    # Token-major, four heads, four bytes/head. No attention-family input.
    source = (
        torch.arange(4 * 8 * 4 * 4, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(4, 8, 4, 4)
    )
    ids = [3, 0, 2]
    if ownership == "heads":
        destination = torch.zeros(5, 8, 1, 4, dtype=torch.uint8)
        dst_ids = [4, 1, 3]
        topology = TransferTopology(target_tp=4, target_tp_rank=rank)
        layout = RegionLayout(
            0, 7, destination.data_ptr(), 128, 32, True, (8, 4), (8, 1)
        )
        expected = source[ids, :, rank : rank + 1, :]
    else:
        destination = torch.zeros(5, 8, 4, 4, dtype=torch.uint8)
        dst_ids = [3]
        topology = TransferTopology(target_dcp=4, target_dcp_rank=rank, interleave=2)
        layout = RegionLayout(
            0, 7, destination.data_ptr(), 128, 128, True, (8, 4), (8, 4)
        )
        ordered = source[ids].reshape(-1, 4, 4)
        expected = ordered[(torch.arange(24) // 2) % 4 == rank]
    plan = build_transfer_plan(topology, [layout], ids, dst_ids, 8)
    execute_plan(plan, source, destination)
    actual = destination[dst_ids].reshape(-1)
    want = expected.reshape(-1)
    assert torch.equal(actual[: want.numel()], want)
    assert not actual[want.numel() :].any()  # partial DCP PAGE tail stays untouched
    assert not destination[list(set(range(5)) - set(dst_ids))].any()
    assert plan.regions[0].target_region == 7  # explicit PP region mapping


@pytest.mark.parametrize("rank", range(4))
def test_tp_replicated_units_can_also_be_partitioned_by_dcp(rank):
    # A replicated latent/auxiliary plane has no per-head TP axis. DCP still
    # selects tokens even with a different target TP size.
    source = torch.arange(64, dtype=torch.uint8).view(2, 8, 4)
    destination = torch.full((3, 8, 4), 255, dtype=torch.uint8)
    layout = RegionLayout(0, 0, destination.data_ptr(), 32, 32, token_contiguous=True)
    plan = build_transfer_plan(
        TransferTopology(
            target_tp=4, target_tp_rank=rank, target_dcp=2, target_dcp_rank=1
        ),
        [layout],
        [1, 0],
        [2],
        8,
    )
    execute_plan(plan, source, destination)
    assert plan.wire_layout == TOKEN_UNITS
    assert torch.equal(destination[2], source[[1, 0]].reshape(16, 4)[1::2])
    assert (destination[:2] == 255).all()


@pytest.mark.parametrize("rank", range(4))
def test_more_tp_ranks_than_kv_heads_replicate_the_correct_head(rank):
    source = torch.arange(64, dtype=torch.uint8).view(2, 4, 2, 4)
    destination = torch.zeros(2, 4, 1, 4, dtype=torch.uint8)
    layout = RegionLayout(0, 0, destination.data_ptr(), 32, 16, True, (4, 2), (4, 1))
    plan = build_transfer_plan(
        TransferTopology(target_tp=4, target_tp_rank=rank), [layout], [1], [0], 4
    )
    execute_plan(plan, source, destination)
    assert plan.wire_layout == PAGE_UNITS
    assert torch.equal(destination[0, :, 0], source[1, :, rank // 2])
    assert not destination[1].any()


def test_unknown_token_layout_and_composed_head_partition_are_rejected():
    layout = RegionLayout(0, 0, 0, 128, 128, False, (8, 4), (8, 4))
    dcp = TransferTopology(target_dcp=4)
    with pytest.raises(ValueError, match="token-contiguous"):
        build_transfer_plan(dcp, [layout], [0], [0], 8)
    with pytest.raises(ValueError, match="Combined token/head"):
        build_transfer_plan(
            replace(dcp, target_tp=4),
            [replace(layout, token_contiguous=True)],
            [0],
            [0],
            8,
        )
    with pytest.raises(ValueError, match="non-DCP producer"):
        build_transfer_plan(replace(dcp, source_dcp=4), [layout], [0], [0], 8)


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
