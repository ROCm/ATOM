# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How `KVConnectorFactory` reads a transfer topology.

`leaf_connectors` is the one walk over `multi`; the region-map predicates the
attention backends gate on are answered from it and from the registration.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory


def _config(kv_transfer_config, *, hf_config=None):
    # `hf_config` reaches `select_offload_layout`; None is the dense layout.
    return SimpleNamespace(kv_transfer_config=kv_transfer_config, hf_config=hf_config)


def _names(config):
    leaves = KVConnectorFactory.leaf_connectors(config)
    return None if leaves is None else [name for name, _ in leaves]


@pytest.mark.parametrize("kv_transfer_config", [None, {}])
def test_no_transfer_config_runs_no_transport(kv_transfer_config):
    config = _config(kv_transfer_config)

    assert KVConnectorFactory.connector_name(kv_transfer_config) is None
    assert _names(config) == []
    assert not KVConnectorFactory.topology_reads_block_regions(config)
    assert not KVConnectorFactory.topology_region_readers_copy_whole_blocks(config)


def test_names_resolve_through_aliases():
    assert KVConnectorFactory.connector_name(
        {"kv_connector": "LMCacheMPConnector"}
    ) == ("lmcache_mp")
    assert _names(_config({"kv_connector": "LMCacheConnectorV1"})) == [
        "lmcache_offload"
    ]


def test_multi_expands_into_its_subs_each_seeing_its_own_config():
    mp = {"kv_connector": "lmcache_mp", "kv_role": "offload"}
    pd = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    config = _config({"kv_connector": "multi", "connectors": [mp, pd]})

    leaves = KVConnectorFactory.leaf_connectors(config)

    assert [name for name, _ in leaves] == ["lmcache_mp", "mooncake"]
    assert [leaf.kv_transfer_config for _, leaf in leaves] == [mp, pd]
    # Model fields still come from the real config; the original is untouched.
    assert all(leaf.hf_config is config.hf_config for _, leaf in leaves)
    assert config.kv_transfer_config["kv_connector"] == "multi"


@pytest.mark.parametrize(
    "connectors",
    [
        pytest.param([], id="empty"),
        pytest.param(["lmcache_mp"], id="not-a-dict"),
    ],
)
def test_unparsable_multi_is_not_read_as_no_transport(connectors):
    """`_build_subconnectors` raises on these; the predicates must not serve them."""
    config = _config({"kv_connector": "multi", "connectors": connectors})

    assert KVConnectorFactory.leaf_connectors(config) is None
    assert KVConnectorFactory.topology_reads_block_regions(config)
    assert not KVConnectorFactory.topology_region_readers_copy_whole_blocks(config)


@pytest.mark.parametrize(
    ("kv_transfer_config", "reads", "whole_blocks"),
    [
        pytest.param({"kv_connector": "lmcache_mp"}, True, True, id="lmcache_mp"),
        pytest.param({"kv_connector": "mooncake"}, True, False, id="mooncake"),
        pytest.param({"kv_connector": "lmcache_offload"}, False, False, id="dense"),
        pytest.param(
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_offload"},
                    {"kv_connector": "lmcache_mp"},
                ],
            },
            True,
            True,
            id="multi-dense-and-mp",
        ),
        pytest.param(
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_mp"},
                    {"kv_connector": "moriio"},
                ],
            },
            True,
            False,
            id="multi-mp-and-pd",
        ),
    ],
)
def test_region_map_predicates(kv_transfer_config, reads, whole_blocks):
    config = _config(kv_transfer_config)

    assert KVConnectorFactory.topology_reads_block_regions(config) is reads
    assert (
        KVConnectorFactory.topology_region_readers_copy_whole_blocks(config)
        is whole_blocks
    )


def test_offload_layout_that_reads_regions_is_not_a_whole_block_copier():
    """Same connector name as dense offload, opposite answer: the layout decides."""
    config = _config(
        {"kv_connector": "lmcache_offload"},
        hf_config=SimpleNamespace(compress_ratios=[1, 2], architectures=[]),
    )

    assert KVConnectorFactory.topology_reads_block_regions(config)
    assert not KVConnectorFactory.topology_region_readers_copy_whole_blocks(config)
