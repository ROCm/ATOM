# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU contracts for the native FABRIC allocation and PAGE layout adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.pool.umbp import pool as umbp_pool
from atom.kv_transfer.pool.umbp.pool import UmbpPoolMemory, pool_id_for
from atom.kv_transfer.pool.umbp.settings import UmbpPoolSettings

ADDRESS = "unix:///unused-umbp-l1.sock"


def transfer(**extra):
    return {
        "kv_connector": "umbp_pool",
        "kv_connector_extra_config": {"address": ADDRESS, **extra},
    }


def page_region(arena, offset, unit_bytes, num_blocks=8, reverse_indexed=False):
    return KVTransferRegion(
        base_addr=arena.data_ptr() + offset,
        total_bytes=num_blocks * unit_bytes,
        unit_bytes=unit_bytes,
        reverse_indexed=reverse_indexed,
    )


@pytest.fixture
def fabric_owner(monkeypatch):
    owners = []

    def make(ratio=1.0):
        client = Mock()
        native = Mock(spec=["allocate", "bind_layout", "close"])
        constructor = Mock(return_value=native)
        monkeypatch.setattr(
            umbp_pool, "_load_native_l1_helpers", lambda: (constructor, Mock())
        )
        memory = UmbpPoolMemory(
            client, "tray-a:atom:dp2:tp1", "compatible-kv", ratio, transport="fabric"
        )
        constructor.assert_called_once_with(
            client, "tray-a:atom:dp2:tp1", "compatible-kv"
        )
        owners.append(memory)
        return memory, native, client

    yield make
    for memory in owners:
        memory.close()


def test_allocate_returns_native_arena_without_another_allocation(
    fabric_owner, monkeypatch
):
    memory, native, client = fabric_owner()
    arena = torch.empty(1024, dtype=torch.uint8, device="cpu")
    native.allocate.return_value = arena
    fallback = Mock(side_effect=AssertionError("unexpected adapter allocation"))
    monkeypatch.setattr(torch, "zeros", fallback)

    assert memory.allocate(1024, arena.device) is arena
    native.allocate.assert_called_once_with(1024, arena.device)
    fallback.assert_not_called()
    client.attach.assert_not_called()

    with pytest.raises(NotImplementedError, match="second paged pool"):
        memory.allocate(1024, arena.device)
    assert native.allocate.call_count == 1


@pytest.mark.parametrize(
    "ratio,num_blocks,slots,layout",
    [
        (1.0, 8, 8, [(256, 8), (32, 16)]),
        (0.5, 8, 4, [(288, 8), (96, 16)]),
        (0.5, 7, 4, [(280, 8), (80, 16)]),
    ],
)
def test_bind_layout_registers_the_shared_tail_of_each_region(
    fabric_owner, ratio, num_blocks, slots, layout
):
    memory, native, client = fabric_owner(ratio)
    arena = torch.empty(1024, dtype=torch.uint8, device="cpu")
    native.allocate.return_value = arena
    allocated = memory.allocate(1024, arena.device)
    # Different block sizes and non-address order expose per-region offset
    # mistakes. The private prefix of each region must remain unregistered.
    regions = [
        page_region(allocated, 256, 8, num_blocks),
        page_region(allocated, 32, 16, num_blocks),
    ]

    memory.bind_layout(allocated, regions, num_blocks)

    native.bind_layout.assert_called_once_with(arena, slots, layout)
    assert native.bind_layout.call_args.args[0] is arena
    client.attach.assert_not_called()
    client.bind_layout.assert_not_called()


@pytest.mark.parametrize(
    "region_specs,match",
    [
        ([(0, 16, False), (96, 8, False)], "overlap"),
        ([(-8, 8, False)], "leave the pool"),
        ([(1000, 8, False)], "leave the pool"),
        ([(0, 8, True)], "reverse-indexed"),
    ],
    ids=["overlap", "before-arena", "past-arena", "reverse-indexed"],
)
def test_invalid_page_layout_is_rejected_before_native_registration(
    fabric_owner, region_specs, match
):
    memory, native, client = fabric_owner()
    arena = torch.empty(1024, dtype=torch.uint8, device="cpu")
    native.allocate.return_value = arena
    memory.allocate(1024, arena.device)
    regions = [
        page_region(arena, offset, unit, reverse_indexed=reverse)
        for offset, unit, reverse in region_specs
    ]

    with pytest.raises(NotImplementedError, match=match):
        memory.bind_layout(arena, regions, 8)

    native.bind_layout.assert_not_called()
    client.attach.assert_not_called()


@pytest.mark.parametrize("operation", ["allocate", "bind_layout"])
def test_native_errors_propagate_without_ipc_fallback(fabric_owner, operation):
    memory, native, client = fabric_owner()
    arena = torch.empty(1024, dtype=torch.uint8, device="cpu")
    native.allocate.return_value = arena
    getattr(native, operation).side_effect = RuntimeError("native FABRIC failed")

    with pytest.raises(RuntimeError, match="native FABRIC failed"):
        allocated = memory.allocate(1024, arena.device)
        if operation == "bind_layout":
            memory.bind_layout(allocated, [page_region(allocated, 0, 8)], 8)

    client.attach.assert_not_called()
    client.bind_layout.assert_not_called()


def test_default_settings_keep_the_ipc_pool_name():
    settings = UmbpPoolSettings.from_transfer_config(transfer())
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=2))

    assert settings.transport == "ipc"
    assert settings.node_id == ""
    assert settings.group == ""
    assert settings.share_ratio == 1.0
    assert pool_id_for(config, settings, rank=1) == "atom:dp2:tp1"


@pytest.mark.parametrize(
    "node_id,group",
    [("", "compatible-kv"), ("tray-a", ""), (" ", "compatible-kv"), ("tray-a", " ")],
)
def test_fabric_settings_require_explicit_node_and_compatibility_group(node_id, group):
    with pytest.raises(ValueError, match="node_id.*group"):
        UmbpPoolSettings.from_transfer_config(
            transfer(transport="fabric", node_id=node_id, group=group)
        )


@pytest.mark.parametrize("transport", ["auto", "rdma", ""])
def test_unknown_transport_is_rejected(transport):
    with pytest.raises(ValueError, match="transport"):
        UmbpPoolSettings.from_transfer_config(transfer(transport=transport))


def test_fabric_settings_normalize_transport_and_identifiers():
    settings = UmbpPoolSettings.from_transfer_config(
        transfer(transport=" FABRIC ", node_id=" tray-a ", group=" compatible-kv ")
    )
    assert (settings.transport, settings.node_id, settings.group) == (
        "fabric",
        "tray-a",
        "compatible-kv",
    )


def test_fabric_pool_ids_distinguish_nodes_namespaces_and_dp_tp_ranks():
    pool_ids = set()
    for node in ("tray-a", "tray-b"):
        for namespace in ("atom", "other-engine"):
            settings = UmbpPoolSettings.from_transfer_config(
                transfer(
                    transport="fabric",
                    node_id=node,
                    group="compatible-kv",
                    namespace=namespace,
                )
            )
            for dp in (0, 1):
                config = SimpleNamespace(
                    parallel_config=SimpleNamespace(data_parallel_rank=dp)
                )
                for tp in (0, 1):
                    pool_id = pool_id_for(config, settings, rank=tp)
                    assert pool_id == f"{node}:{namespace}:dp{dp}:tp{tp}"
                    pool_ids.add(pool_id)
    assert len(pool_ids) == 16


def test_ipc_pool_id_keeps_its_existing_namespace_even_with_node_id():
    settings = UmbpPoolSettings.from_transfer_config(
        transfer(transport="ipc", node_id="tray-a", namespace="other-engine")
    )
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=3))
    assert pool_id_for(config, settings, rank=2) == "other-engine:dp3:tp2"
