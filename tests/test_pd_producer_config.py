# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.pd_producer import (
    index_staging_pool_size,
    mooncake_pd_producer_configured,
    pd_producer_configured,
)


@pytest.mark.parametrize(
    "kv_transfer_config, is_pd, is_mooncake_staging",
    [
        (None, False, False),
        ({}, False, False),
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, True, True),
        ({"kv_connector": "moriio", "kv_role": "kv_producer"}, True, False),
        ({"kv_connector": "mooncake"}, True, True),
        ({"kv_connector": "mooncake", "kv_role": "kv_consumer"}, False, False),
        ({"kv_connector": "lmcache_offload"}, False, False),
        ({"kv_connector": "lmcache_offload", "kv_role": "offload"}, False, False),
        ({"kv_connector": "lmcache_offload", "kv_role": "kv_producer"}, False, False),
        ({"kv_connector": "multi", "connectors": []}, False, False),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            },
            True,
            True,
        ),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "mooncake", "kv_role": "kv_consumer"},
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            },
            False,
            False,
        ),
    ],
)
def test_pd_producer_classification(kv_transfer_config, is_pd, is_mooncake_staging):
    config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    assert pd_producer_configured(config) is is_pd
    assert mooncake_pd_producer_configured(config) is is_mooncake_staging


@pytest.mark.parametrize(
    "kv_transfer_config, expected",
    [
        (None, 0),
        ({"kv_connector": "moriio", "kv_role": "kv_producer"}, 0),
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, 16),
        (
            {
                "kv_connector": "mooncake",
                "kv_role": "kv_producer",
                "num_worker_threads": 32,
            },
            32,
        ),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {
                        "kv_connector": "mooncake",
                        "kv_role": "kv_producer",
                        "num_worker_threads": 24,
                    },
                    {
                        "kv_connector": "mooncake",
                        "kv_role": "kv_consumer",
                        "num_worker_threads": 64,
                    },
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            },
            24,
        ),
    ],
)
def test_index_staging_pool_matches_mooncake_worker_count(kv_transfer_config, expected):
    config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    assert index_staging_pool_size(config) == expected


@pytest.mark.parametrize("worker_count", [0, -1, True, "32"])
def test_index_staging_pool_rejects_invalid_worker_count(worker_count):
    config = SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "mooncake",
            "kv_role": "kv_producer",
            "num_worker_threads": worker_count,
        }
    )
    with pytest.raises(ValueError, match="positive integer"):
        index_staging_pool_size(config)


def test_index_staging_pool_rejects_two_mooncake_producers():
    config = SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "multi",
            "connectors": [
                {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                {"kv_connector": "mooncake", "kv_role": "kv_producer"},
            ],
        }
    )
    with pytest.raises(ValueError, match="multiple Mooncake"):
        index_staging_pool_size(config)
