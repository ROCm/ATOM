# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What MiniMax-M3 index sharding (`dcp_config.index_shard`) refuses.

The flag changes the KV pool's layout, so `Config.__post_init__` raises with
`index_shard_unsupported_reason`'s answer instead of falling back. These pin
the predicate: the indexer-CP verdict it builds on, the FP4 index, one
scheduler block a sparse block, and the KV transports that read a PAGE
region's shard.
"""

import pytest

from atom.config import DCPConfig, index_shard_unsupported_reason

FABRIC = {
    "kv_connector": "moriio",
    "moriio_backend": "fabric",
    "kv_role": "kv_producer",
}
OFFLOAD = {"kv_connector": "lmcache_offload", "kv_role": "offload"}


def reason(**overrides):
    args = {
        "indexer_cp": True,
        "index_cache_dtype": "fp4",
        "kv_cache_block_size": 128,
        "sparse_block_size": 128,
        "kv_transfer_config": None,
    }
    args.update(overrides)
    return index_shard_unsupported_reason(**args)


def test_the_deployment_it_is_for_passes():
    assert reason() is None
    assert reason(kv_transfer_config=FABRIC) is None
    assert reason(kv_transfer_config=OFFLOAD) is None
    multi = {"kv_connector": "multi", "connectors": [FABRIC, OFFLOAD]}
    assert reason(kv_transfer_config=multi) is None


def test_it_needs_the_indexer_cp_decode():
    assert "ATOM_M3_INDEXER_CP=1" in reason(indexer_cp=False)


@pytest.mark.parametrize("dtype", ["fp8", "bf16", None])
def test_it_needs_the_fp4_index(dtype):
    assert "--index-cache-dtype fp4" in reason(index_cache_dtype=dtype)


def test_one_scheduler_block_is_one_sparse_block():
    assert "--block-size 128" in reason(kv_cache_block_size=256)


@pytest.mark.parametrize(
    "cfg",
    [
        {"kv_connector": "mooncake", "kv_role": "kv_producer"},
        {"kv_connector": "lmcache_mp", "kv_role": "offload"},
        {"kv_connector": "mooncake_store", "kv_role": "offload"},
        {
            "kv_connector": "multi",
            "connectors": [{"kv_connector": "mooncake"}, OFFLOAD],
        },
    ],
)
def test_transports_that_move_whole_regions_are_refused(cfg):
    assert reason(kv_transfer_config=cfg) is not None


def test_the_dcp_config_key_is_accepted():
    assert DCPConfig.from_dict({"index_shard": True}).index_shard is True
    assert DCPConfig().index_shard is False
