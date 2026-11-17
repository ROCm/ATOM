# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The `umbp_l1` connector provides the KV pool; the engine asks connectors
for it before falling back to its own (atom/kv_transfer/l1)."""

import types

import pytest
import test_umbp_l1_pool as umbp_tests
from conftest import MockConfig
from test_umbp_l1_pool import make_pool

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.l1.umbp_l1 import pool as umbp_pool
from atom.kv_transfer.l1.umbp_l1.connector import UmbpL1ConnectorScheduler
from atom.kv_transfer.l1.umbp_l1.pool import UmbpBlockPool
from atom.kv_transfer.l1.umbp_l1.settings import UmbpL1Settings
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.kv_block_pool import LocalPoolMemory, make_pool_memory

ADDRESS = "unix:///tmp/umbp-l1/server.sock"
# The L1 service fixture, the Python model and then the real one.
l1 = umbp_tests.l1


def transfer(**extra):
    return {
        "kv_connector": "umbp_l1",
        "kv_connector_extra_config": {"address": ADDRESS, **extra},
    }


def config_with(kv_transfer_config):
    return types.SimpleNamespace(kv_transfer_config=kv_transfer_config)


class TestSettings:
    def test_defaults(self):
        s = UmbpL1Settings.from_transfer_config(transfer())
        assert (s.address, s.namespace, s.group, s.fetch, s.share_ratio) == (
            ADDRESS,
            "atom",
            "",
            True,
            1.0,
        )
        assert s.session_migrate_margin is None

    def test_the_address_is_required(self):
        with pytest.raises(ValueError, match="address"):
            UmbpL1Settings.from_transfer_config({"kv_connector": "umbp_l1"})

    @pytest.mark.parametrize("ratio", [0, -0.1, 1.5])
    def test_the_share_ratio_is_a_share(self, ratio):
        with pytest.raises(ValueError, match="share_ratio"):
            UmbpL1Settings.from_transfer_config(transfer(share_ratio=ratio))

    def test_fetch_accepts_text(self):
        assert not UmbpL1Settings.from_transfer_config(transfer(fetch="false")).fetch


class TestRegistry:
    def test_registered_under_its_name(self):
        assert KVConnectorFactory.canonical_name("umbp_l1") == "umbp_l1"

    def test_routing_learns_that_ranks_load_from_each_other(self):
        sharing = KVConnectorFactory.pool_sharing(
            config_with(transfer(session_migrate_margin=0.1))
        )
        assert sharing == {"loads": True, "session_migrate_margin": 0.1}

    def test_no_fetching_means_no_loads_and_no_moves(self):
        sharing = KVConnectorFactory.pool_sharing(
            config_with(transfer(fetch=False, session_migrate_margin=0.1))
        )
        assert sharing == {"loads": False, "session_migrate_margin": None}

    def test_found_inside_multi(self):
        cfg = {
            "kv_connector": "multi",
            "connectors": [
                {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                transfer(),
            ],
        }
        assert KVConnectorFactory.pool_sharing(config_with(cfg))["loads"]

    def test_other_connectors_share_no_pool(self):
        cfg = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
        assert KVConnectorFactory.pool_sharing(config_with(cfg)) is None


class TestProvidesThePool:
    @pytest.fixture
    def shared_pools(self, l1, monkeypatch):
        made = []

        def make(config, settings, num_blocks, **kwargs):
            made.append(num_blocks)
            return make_pool(l1, f"conn{len(made)}", num_blocks, **kwargs)

        monkeypatch.setattr(umbp_pool, "make_umbp_l1_pool", make)
        return made

    def test_whole_pool_by_default(self, shared_pools):
        # The same pool at every ratio: at 1 its private part is empty.
        conn = UmbpL1ConnectorScheduler(config_with(transfer()))
        pool = conn.make_block_pool(None, 8)
        assert isinstance(pool, UmbpBlockPool)
        assert pool.offset == 0 and pool.private.num_blocks == 0
        assert shared_pools == [8]

    def test_a_share_ratio_splits_the_pool(self, shared_pools):
        conn = UmbpL1ConnectorScheduler(config_with(transfer(share_ratio=0.25)))
        pool = conn.make_block_pool(None, 8)
        assert isinstance(pool, UmbpBlockPool)
        assert pool.offset == 6 and shared_pools == [2]

    def test_block_manager_takes_the_connectors_pool(self, shared_pools):
        cfg = MockConfig(
            num_kvcache_blocks=16, kv_cache_block_size=4, enable_prefix_caching=True
        )
        conn = UmbpL1ConnectorScheduler(config_with(transfer(share_ratio=0.5)))
        bm = BlockManager(cfg, kv_connector=conn)
        assert isinstance(bm.kv, UmbpBlockPool)

    def test_without_a_providing_connector_the_pool_is_atoms(self):
        cfg = MockConfig(
            num_kvcache_blocks=16, kv_cache_block_size=4, enable_prefix_caching=True
        )
        assert isinstance(BlockManager(cfg, kv_connector=object()).kv, BlockPool)
        assert isinstance(BlockManager(cfg).kv, BlockPool)

    def test_runner_memory_falls_back_to_its_own(self):
        assert isinstance(make_pool_memory(None, 0, kv_connector=None), LocalPoolMemory)

        class Provider:
            def kv_pool_memory(self, config, rank):
                return "theirs"

        assert make_pool_memory(None, 0, kv_connector=Provider()) == "theirs"
