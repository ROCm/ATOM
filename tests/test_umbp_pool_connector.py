# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The `umbp_pool` connector provides the KV pool; the engine asks connectors
for it before falling back to its own (atom/kv_transfer/pool)."""

import ctypes
import os
import random
import sys
import threading
import time
import types

import pytest
import test_umbp_l1_pool as umbp_tests
from conftest import MockConfig, _MockHFConfig
from test_umbp_l1_pool import make_pool

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.pool.umbp import pool as umbp_pool
from atom.kv_transfer.pool.umbp.connector import UmbpPoolConnectorScheduler
from atom.kv_transfer.pool.umbp.pool import UmbpBlockPool
from atom.kv_transfer.pool.umbp.settings import UmbpPoolSettings
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.kv_block_pool import LocalPoolMemory, make_pool_memory

ADDRESS = "unix:///tmp/umbp-l1/server.sock"
# The L1 service fixture, the Python model and then the real one.
l1 = umbp_tests.l1


def transfer(**extra):
    return {
        "kv_connector": "umbp_pool",
        "kv_connector_extra_config": {"address": ADDRESS, **extra},
    }


def config_with(kv_transfer_config, architectures=("DeepseekV41ForCausalLM",), tp=1):
    """A config the connector accepts unless told otherwise: DeepSeek-V4.1
    under data-parallel attention."""
    return MockConfig(
        kv_transfer_config=kv_transfer_config,
        hf_config=_MockHFConfig(list(architectures)),
        tensor_parallel_size=tp,
    )


class TestSettings:
    def test_defaults(self):
        s = UmbpPoolSettings.from_transfer_config(transfer())
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
            UmbpPoolSettings.from_transfer_config({"kv_connector": "umbp_pool"})

    @pytest.mark.parametrize("ratio", [0, -0.1, 1.5])
    def test_the_share_ratio_is_a_share(self, ratio):
        with pytest.raises(ValueError, match="share_ratio"):
            UmbpPoolSettings.from_transfer_config(transfer(share_ratio=ratio))

    def test_fetch_accepts_text(self):
        assert not UmbpPoolSettings.from_transfer_config(transfer(fetch="false")).fetch


class TestRegistry:
    def test_registered_under_its_name(self):
        assert KVConnectorFactory.canonical_name("umbp_pool") == "umbp_pool"

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
        conn = UmbpPoolConnectorScheduler(config_with(transfer()))
        pool = conn.make_block_pool(None, 8)
        assert isinstance(pool, UmbpBlockPool)
        assert pool.offset == 0 and pool.private.num_blocks == 0
        assert shared_pools == [8]

    def test_a_share_ratio_splits_the_pool(self, shared_pools):
        conn = UmbpPoolConnectorScheduler(config_with(transfer(share_ratio=0.25)))
        pool = conn.make_block_pool(None, 8)
        assert isinstance(pool, UmbpBlockPool)
        assert pool.offset == 6 and shared_pools == [2]

    def test_block_manager_takes_the_connectors_pool(self, shared_pools):
        cfg = MockConfig(
            num_kvcache_blocks=16, kv_cache_block_size=4, enable_prefix_caching=True
        )
        conn = UmbpPoolConnectorScheduler(config_with(transfer(share_ratio=0.5)))
        bm = BlockManager(cfg, kv_connector=conn)
        assert isinstance(bm.kv, UmbpBlockPool)

    def test_without_a_providing_connector_the_pool_is_atoms(self):
        cfg = MockConfig(
            num_kvcache_blocks=16, kv_cache_block_size=4, enable_prefix_caching=True
        )
        # A connector that moves KV but provides no pool (the base class's answer).
        mover = types.SimpleNamespace(make_block_pool=lambda *a, **k: None)
        assert isinstance(BlockManager(cfg, kv_connector=mover).kv, BlockPool)
        assert isinstance(BlockManager(cfg).kv, BlockPool)

    def test_runner_memory_falls_back_to_its_own(self):
        assert isinstance(make_pool_memory(None, 0, kv_connector=None), LocalPoolMemory)

        class Provider:
            def kv_pool_memory(self, config, rank):
                return "theirs"

        assert make_pool_memory(None, 0, kv_connector=Provider()) == "theirs"


class TestRefusedAtStartup:
    def config(self, architectures, tp=1):
        return config_with(transfer(), architectures, tp)

    def test_a_model_without_the_pool_layout(self):
        with pytest.raises(ValueError, match="supports"):
            UmbpPoolConnectorScheduler(self.config(["LlamaForCausalLM"]))

    def test_tensor_parallel_attention_is_one_pool_of_parts(self):
        UmbpPoolConnectorScheduler(self.config(["DeepseekV41ForCausalLM"], tp=2))

    def test_minimax_m3_with_its_draft_at_tp4(self):
        UmbpPoolConnectorScheduler(
            self.config(["MiniMaxM3SparseForConditionalGeneration"], tp=4)
        )

    @pytest.mark.parametrize(
        "field, value, message",
        [
            ("is_multinode_tp", True, "one node"),
            ("pcp_group_size", 2, "prefill context parallel"),
            ("pipeline_parallel_size", 2, "pipeline parallel"),
        ],
    )
    def test_layouts_one_server_and_one_scheduler_cannot_cover(
        self, field, value, message
    ):
        config = self.config(["DeepseekV41ForCausalLM"], tp=4)
        setattr(config, field, value)
        with pytest.raises(ValueError, match=message):
            UmbpPoolConnectorScheduler(config)

    def test_deepseek_v41_under_dp_attention(self):
        UmbpPoolConnectorScheduler(self.config(["DeepseekV41ForCausalLM"]))


class TestTensorParallelRank:
    """A rank with tensor-parallel attention: one pool, a part per runner."""

    BLOCKS = 4
    UNITS = (16, 8)  # block bytes on runner 0 and on runner 1

    @pytest.fixture
    def service(self):
        mori = umbp_tests.mori_umbp_l1
        if mori is None or "num_parts" not in (mori.L1Client.attach.__doc__ or ""):
            pytest.skip("mori_umbp_l1 is not built, or has no pool parts")
        path = f"/tmp/atom_l1_parts_{os.getpid()}_{random.randrange(1 << 30)}.sock"
        server = mori.HostTestServer(f"unix://{path}")
        assert server.ok()
        yield mori.L1Client(f"unix://{path}")
        del server
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture(autouse=True)
    def host_handles(self, monkeypatch):
        """The host test server adopts memory by address, not by HIP handle."""
        from atom.mono import runtime

        stub = types.SimpleNamespace(
            get_allocation_base=lambda address: address,
            get_ipc_handle=lambda address: address.to_bytes(8, "little"),
        )
        monkeypatch.setattr(runtime, "hip_ipc", stub, raising=False)
        monkeypatch.setitem(sys.modules, "atom.mono.runtime.hip_ipc", stub)

    def config(self, dp_rank):
        return types.SimpleNamespace(
            tensor_parallel_size=2,
            parallel_config=types.SimpleNamespace(data_parallel_rank=dp_rank),
            model="m",
        )

    def runner(self, client, dp_rank, rank, device):
        """What `ModelRunner` does with its paged pool on one TP rank."""
        settings = UmbpPoolSettings(address=ADDRESS)
        memory = umbp_pool.UmbpPoolMemory(
            client,
            umbp_pool.pool_id_for(self.config(dp_rank), settings),
            "g",
            1.0,
            part=rank,
            num_parts=umbp_pool.pool_parts(self.config(dp_rank)),
        )
        unit = self.UNITS[rank]
        raw = (ctypes.c_uint8 * (self.BLOCKS * unit))()
        buf = types.SimpleNamespace(
            data_ptr=lambda: ctypes.addressof(raw),
            numel=lambda: len(raw),
            device=types.SimpleNamespace(index=device),
        )
        region = types.SimpleNamespace(
            reverse_indexed=False, base_addr=ctypes.addressof(raw), unit_bytes=unit
        )
        memory.bind_layout(buf, [region], self.BLOCKS)
        return memory, raw

    def scheduler(self, client, dp_rank):
        settings = UmbpPoolSettings(address=ADDRESS)
        return umbp_pool.UmbpL1Pool(
            client,
            umbp_tests.OPS,
            umbp_pool.pool_id_for(self.config(dp_rank), settings),
            self.BLOCKS,
            # One block a grant: a larger grant would hold the whole pool.
            grant_batch=1,
            parts=2,
        )

    def test_a_block_moves_between_ranks_shard_by_shard(self, service):
        runners = {
            (dp, tp): self.runner(service, dp, tp, 4 * dp + tp)
            for dp in (0, 1)
            for tp in (0, 1)
        }
        a, b = self.scheduler(service, 0), self.scheduler(service, 1)
        b.allocate(b.pop())  # so the block lands at another id on rank 1
        block = a.pop()
        a.allocate(block)
        for tp in (0, 1):
            unit = self.UNITS[tp]
            raw = runners[0, tp][1]
            raw[block * unit : (block + 1) * unit] = bytes([7 + tp]) * unit
        a.publish(block, 5, umbp_tests.toks(5))
        a.free(block)
        a.flush()
        stamp = time.monotonic_ns()
        # Runner 0's device has landed the forward; runner 1's has not.
        service.commit(a.pool_id, stamp, part=0)
        assert b.fetch([5], lambda i: umbp_tests.toks(5)) == 0
        service.commit(a.pool_id, stamp, part=1)
        assert b.fetch([5], lambda i: umbp_tests.toks(5)) == 1
        landed = b.lookup(5)
        assert landed not in (-1, block)
        for tp in (0, 1):
            unit = self.UNITS[tp]
            raw = runners[1, tp][1]
            assert (
                bytes(raw[landed * unit : (landed + 1) * unit])
                == bytes([7 + tp]) * unit
            )
        stats = {s["pool_id"]: s for s in service.stats()}
        assert stats[a.pool_id]["parts"] == 2
        assert stats[b.pool_id]["fetched_in_bytes"] == sum(self.UNITS)

    def test_the_scheduler_waits_for_the_runners_it_was_not_told_about(self, service):
        self.runner(service, 0, 0, 0)
        late = threading.Timer(0.3, self.runner, (service, 0, 1, 1))
        late.start()
        try:
            assert self.scheduler(service, 0).num_blocks == self.BLOCKS
        finally:
            late.join()

    def test_a_runner_outside_the_group_has_no_part(self):
        with pytest.raises(ValueError, match="no part"):
            umbp_pool.make_umbp_pool_memory(
                self.config(0), UmbpPoolSettings(address=ADDRESS), rank=2
            )
