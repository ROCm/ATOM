# SPDX-License-Identifier: MIT
"""Ownership is selected by topology, with physical layouts as separate inputs."""

import ctypes
from dataclasses import replace

import pytest
import torch

from atom.kv_transfer.disaggregation.gather import GatherLayout, gather_units
from atom.kv_transfer.disaggregation.transfer_plan import (
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
    layout = RegionLayout(0, 7, destination.data_ptr(), 128, 128, True)
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


@pytest.mark.parametrize("source_tp", [1, 2, 4])
@pytest.mark.parametrize("rank", range(4))
def test_tp_replicated_units_can_also_be_partitioned_by_dcp(source_tp, rank):
    # A replicated latent/auxiliary plane has no per-head TP axis. DCP still
    # selects tokens whatever the producer and target TP sizes are.
    source = torch.arange(64, dtype=torch.uint8).view(2, 8, 4)
    destination = torch.full((3, 8, 4), 255, dtype=torch.uint8)
    layout = RegionLayout(0, 0, destination.data_ptr(), 32, 32, token_contiguous=True)
    plan = build_transfer_plan(
        TransferTopology(
            source_tp=source_tp,
            target_tp=4,
            target_tp_rank=rank,
            target_dcp=2,
            target_dcp_rank=1,
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


def test_unknown_token_layout_and_unsupported_topologies_are_rejected():
    layout = RegionLayout(0, 0, 0, 128, 128, False)
    dcp = TransferTopology(target_dcp=4)
    with pytest.raises(ValueError, match="token-contiguous"):
        build_transfer_plan(dcp, [layout], [0], [0], 8)
    with pytest.raises(ValueError, match="equal source and target"):
        build_transfer_plan(
            dcp,
            [replace(layout, token_contiguous=True, target_page_bytes=64)],
            [0],
            [0],
            8,
        )
    with pytest.raises(ValueError, match="non-DCP producer"):
        build_transfer_plan(replace(dcp, source_dcp=4), [layout], [0], [0], 8)
    with pytest.raises(ValueError, match="PAGE adapter"):
        build_transfer_plan(TransferTopology(target_tp=4), [layout], [0], [0], 8)


def test_gather_refuses_strided_slices():
    source = torch.zeros(4, 32, dtype=torch.uint8)
    with pytest.raises(ValueError, match="layout adapter"):
        gather_units(
            source,
            torch.tensor([0, 1]),
            torch.empty(2 * 8, dtype=torch.uint8),
            GatherLayout(32, 8, 0, 16, 4),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU buffer allocation")
def test_shared_buffer_builders_respect_reserved_memory():
    from types import SimpleNamespace

    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
        MooncakeConnector,
    )
    from atom.kv_transfer.disaggregation.staging_config import LandingSpec
    from atom.kv_transfer.disaggregation.types import MLA_KV_ROLE, KVTransferRegion

    source = torch.empty((4, 128), dtype=torch.uint8, device="cuda")
    c = object.__new__(MooncakeConnector)
    c._kv_staging_reserve = 8 << 20
    c._num_send_workers = 4  # Must share the single reserved source slot.
    c._has_slot_regions = False
    region = KVTransferRegion(
        source.data_ptr(), source.numel(), 128, semantic_role=MLA_KV_ROLE
    )
    staging = c._build_kv_staging(
        SimpleNamespace(block_regions=[region], block_tensor_views=[source])
    )
    assert staging.total_bytes == c._kv_staging_reserve
    assert c._kv_staging_slots.size == 1
    assert staging.base_addr == c._kv_staging.data_ptr()
    c._landing_spec = LandingSpec(8, 16, TOKEN_UNITS)
    c._block_region_roles = [MLA_KV_ROLE]
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
