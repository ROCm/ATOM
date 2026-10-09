# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation import pd_producer
from atom.kv_transfer.disaggregation.pd_producer import (
    index_staging_pool_size,
    mla_staging_reserve_bytes,
    mla_staging_slot_count,
    mooncake_pd_producer_configured,
    pd_producer_configured,
    send_worker_count,
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


@pytest.mark.parametrize("worker_count", [None, 0, True, "32"])
def test_send_worker_count_rejects_invalid_values(worker_count):
    with pytest.raises(ValueError, match="positive integer"):
        send_worker_count({"num_worker_threads": worker_count})


def test_send_worker_count_defaults_to_sixteen():
    assert send_worker_count({"kv_connector": "mooncake"}) == 16


def test_mla_staging_slots_are_capped_by_pool_bytes(monkeypatch):
    slot = 8 << 20
    assert mla_staging_slot_count(16, slot) == 16
    assert mla_staging_slot_count(128, slot) == 32  # 256 MiB cap
    monkeypatch.setattr(pd_producer, "MLA_STAGING_POOL_BYTES", 1 << 20)
    assert mla_staging_slot_count(128, slot) == 1  # never below one slot


def _mla_producer_config(workers, **overrides):
    config = {
        "kv_transfer_config": {
            "kv_connector": "mooncake",
            "kv_role": "kv_producer",
            "num_worker_threads": workers,
        },
        "hf_config": SimpleNamespace(kv_lora_rank=512),
        "decode_context_parallel_size": 1,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def test_mla_staging_reserve_covers_the_connector_pool(monkeypatch):
    monkeypatch.delenv("ATOM_PD_MLA_STAGING", raising=False)
    assert mla_staging_reserve_bytes(_mla_producer_config(16)) == 128 << 20
    assert mla_staging_reserve_bytes(_mla_producer_config(128)) == 256 << 20
    # The connector rounds a slot down to whole pages, so the reserve bounds
    # what it allocates for any worker count.
    page = 16 * 576
    slot = (8 << 20) // page * page
    for workers in (1, 16, 31, 32, 33, 128):
        allocated = mla_staging_slot_count(workers, slot) * slot
        assert allocated <= mla_staging_reserve_bytes(_mla_producer_config(workers))


@pytest.mark.parametrize(
    "overrides, env",
    [
        ({"hf_config": SimpleNamespace()}, {}),
        ({"decode_context_parallel_size": 4}, {}),
        (
            {
                "kv_transfer_config": {
                    "kv_connector": "mooncake",
                    "kv_role": "kv_consumer",
                }
            },
            {},
        ),
        ({}, {"ATOM_PD_MLA_STAGING": "0"}),
    ],
)
def test_mla_staging_reserve_is_zero_without_a_pool(monkeypatch, overrides, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert mla_staging_reserve_bytes(_mla_producer_config(16, **overrides)) == 0
