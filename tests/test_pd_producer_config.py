# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.pd_producer import (
    index_staging_pool_size,
    index_staging_shape,
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


@pytest.mark.parametrize("workers", [16, 64])
@pytest.mark.parametrize("token_bytes", [576, 1152])
def test_index_staging_default_cap_bounds_mla_allocation(workers, token_bytes):
    config = SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "mooncake",
            "num_worker_threads": workers,
        }
    )
    page_bytes = 64 * token_bytes
    slots, pages = index_staging_shape(config, page_bytes)
    assert slots == workers
    assert 0 < pages <= 256
    assert slots * pages * page_bytes <= 256 * 1024**2
    if pages < 256:
        assert slots * (pages + 1) * page_bytes > 256 * 1024**2


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("requested,cap,expected", [(7, 10000, 7), (256, 1000, 5)])
def test_index_staging_shape_honors_producer_options(multi, requested, cap, expected):
    connector = {
        "kv_connector": "mooncake",
        "num_worker_threads": 2,
        "index_staging_chunk_pages": requested,
        "index_staging_max_bytes": cap,
    }
    transfer = (
        {
            "kv_connector": "multi",
            "connectors": [
                {"kv_connector": "lmcache_offload"},
                connector,
            ],
        }
        if multi
        else connector
    )
    assert index_staging_shape(SimpleNamespace(kv_transfer_config=transfer), 100) == (
        2,
        expected,
    )


@pytest.mark.parametrize(
    "option", ["index_staging_chunk_pages", "index_staging_max_bytes"]
)
@pytest.mark.parametrize("value", [0, -1, True, "256", 1.5])
def test_index_staging_shape_rejects_invalid_options(option, value):
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector": "mooncake", option: value}
    )
    with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
        index_staging_shape(config, 100)


def test_index_staging_shape_rejects_cap_smaller_than_one_page_per_worker():
    config = SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "mooncake",
            "num_worker_threads": 2,
            "index_staging_max_bytes": 199,
        }
    )
    with pytest.raises(ValueError, match="cannot hold one page per worker"):
        index_staging_shape(config, 100)


def test_index_staging_shape_without_producer_needs_no_memory():
    assert index_staging_shape(SimpleNamespace(kv_transfer_config=None), 100) == (0, 0)
