# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""umbp_pool beside the P/D connector of MiniMax-M3.1's 1P1D.

MoRIIO fabric allocates the paged pool itself, in fabric-exportable memory a
peer maps (`KVConnectorBase.allocate_kv_pool`). UMBP manages a pool it adopts
through a HIP IPC handle, so beside MoRIIO it has to adopt that allocation
(`KVPoolMemory.adopts_exportable`, opted into with `adopt_exportable_pool`)
rather than bring its own. The composite closes the pool's provider last, and
a lazy offload tier, which needs ATOM's pool to choose evictions, is refused
beside it.
"""

import types

import pytest
import test_moriio_fabric
import test_umbp_l1_pool as umbp_tests
import test_umbp_store_lazy as lazy_tests
import torch
from conftest import MockConfig
from test_multi_connector import FakeSchedSub, FakeWorkerSub, _sched, _worker

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    validate_multi_config,
)
from atom.kv_transfer.pool.umbp import pool as umbp_pool
from atom.kv_transfer.pool.umbp.settings import UmbpPoolSettings

stack = test_moriio_fabric.stack
l1 = umbp_tests.l1

ADDRESS = "unix:///tmp/umbp-l1/server.sock"
PRODUCER = {
    "kv_connector": "moriio",
    "kv_role": "kv_producer",
    "moriio_backend": "fabric",
    "moriio_mode": "write",
}
CONSUMER = {**PRODUCER, "kv_role": "kv_consumer"}


def pool_sub(**extra):
    return {
        "kv_connector": "umbp_pool",
        "kv_connector_extra_config": {"address": ADDRESS, **extra},
    }


class TestConfigurations:
    def test_p_runs_the_producer_the_pool_and_the_offload_tier(self):
        store = {
            "kv_connector": "umbp_store",
            "kv_role": "offload",
            "kv_connector_extra_config": {"umbp_store.address": ADDRESS},
        }
        subs = [PRODUCER, pool_sub(), store]
        assert validate_multi_config({"kv_connector": "multi", "connectors": subs}) == (
            subs
        )

    def test_d_runs_the_consumer_and_the_pool(self):
        subs = [CONSUMER, pool_sub()]
        assert validate_multi_config({"kv_connector": "multi", "connectors": subs}) == (
            subs
        )


class TestAdoptionSetting:
    def test_off_unless_asked(self):
        settings = UmbpPoolSettings.from_transfer_config(pool_sub())
        assert settings.adopt_exportable_pool is False

    @pytest.mark.parametrize("value", [True, "1", "true", "on"])
    def test_on(self, value):
        settings = UmbpPoolSettings.from_transfer_config(
            pool_sub(adopt_exportable_pool=value)
        )
        assert settings.adopt_exportable_pool is True

    @pytest.mark.parametrize("value", [False, "0", "false", "off"])
    def test_off(self, value):
        settings = UmbpPoolSettings.from_transfer_config(
            pool_sub(adopt_exportable_pool=value)
        )
        assert settings.adopt_exportable_pool is False

    def test_fetch_reads_as_before(self):
        assert UmbpPoolSettings.from_transfer_config(pool_sub(fetch="")).fetch
        assert not UmbpPoolSettings.from_transfer_config(pool_sub(fetch="no")).fetch


class _Provider:
    """A pool provider that shares blocks across processes, as UMBP's does."""

    shares_blocks = True
    counted_by_torch = True

    def __init__(self, adopts):
        self.adopts_exportable = adopts
        self.allocated = []

    def allocate(self, nbytes, device):
        self.allocated.append(nbytes)
        return torch.zeros(nbytes, dtype=torch.uint8, device=device)


class TestTheRunnerAdoptsTheExportablePool:
    BLOCK_BYTES = 512

    def runner(self, stack, provider):
        runner = object.__new__(stack.runner.ModelRunner)
        runner.device = torch.device("cpu")
        runner.attn_metadata_builder = types.SimpleNamespace(
            invalidate_pool_rows=lambda: None,
            paged_pool_bytes=lambda blocks: blocks * self.BLOCK_BYTES,
            sub_pool_specs=lambda: [stack.sub_pool.page_pool(self.BLOCK_BYTES)],
        )
        runner.config = types.SimpleNamespace(
            kv_transfer_config={
                "kv_connector": "multi",
                "connectors": [PRODUCER, pool_sub()],
            }
        )
        runner._kv_pool_memory_provider = provider
        return runner

    @pytest.fixture
    def exportable(self, stack, monkeypatch):
        """MoRIIO fabric's allocation: a buffer outside the torch allocator."""
        made = []

        def allocate(cls, config, nbytes, device):
            made.append(torch.zeros(nbytes, dtype=torch.uint8))
            return made[-1]

        monkeypatch.setattr(
            stack.factory.KVConnectorFactory, "allocate_kv_pool", classmethod(allocate)
        )
        return made

    def test_a_provider_that_cannot_adopt_it_is_refused(self, stack, exportable):
        runner = self.runner(stack, _Provider(adopts=False))
        with pytest.raises(NotImplementedError, match="adopt_exportable_pool"):
            runner._carve_paged_pool(4)

    def test_a_provider_that_can_gets_that_allocation(self, stack, exportable):
        provider = _Provider(adopts=True)
        runner = self.runner(stack, provider)
        buf, regions, _ = runner._carve_paged_pool(4)
        assert buf is exportable[0]  # MoRIIO's, which UMBP then adopts
        assert provider.allocated == []
        assert runner._kv_pool_outside_torch_bytes == 4 * self.BLOCK_BYTES
        assert regions[0].numel() == 4 * self.BLOCK_BYTES


class TestNoHandleNoAdoption:
    def test_the_error_says_who_allocated_the_pool(self, monkeypatch):
        from atom.mono import runtime

        def no_handle(address):
            raise RuntimeError("hipIpcGetMemHandle failed: invalid argument")

        stub = types.SimpleNamespace(
            get_allocation_base=lambda address: address, get_ipc_handle=no_handle
        )
        monkeypatch.setattr(runtime, "hip_ipc", stub, raising=False)
        monkeypatch.setitem(
            __import__("sys").modules, "atom.mono.runtime.hip_ipc", stub
        )
        memory = umbp_pool.UmbpPoolMemory(
            object(), "atom:dp0", "g", 1.0, adopts_exportable=True
        )
        raw = torch.zeros(64, dtype=torch.uint8)
        region = types.SimpleNamespace(
            reverse_indexed=False, base_addr=raw.data_ptr(), unit_bytes=16
        )
        with pytest.raises(RuntimeError, match="cannot adopt it") as err:
            memory.bind_layout(raw, [region], 4)
        assert "adopt_exportable_pool" in str(err.value)
        assert "invalid argument" in str(err.value)


class _Closing(FakeWorkerSub):
    def __init__(self, name, order, memory=None):
        super().__init__()
        self.name, self._order, self._memory = name, order, memory

    def kv_pool_memory(self, config, rank):
        return self._memory

    def close(self):
        self._order.append(self.name)


class TestCloseOrder:
    def test_the_pool_provider_closes_last(self):
        order = []
        worker = _worker(
            [
                _Closing("moriio", order),
                _Closing("umbp_pool", order, memory="pool"),
                _Closing("umbp_store", order),
            ]
        )
        assert worker.kv_pool_memory(None, 0) == "pool"
        worker.close()
        assert order == ["moriio", "umbp_store", "umbp_pool"]

    def test_without_a_provider_the_order_is_as_listed(self):
        order = []
        worker = _worker([_Closing("a", order), _Closing("b", order)])
        assert worker.kv_pool_memory(None, 0) is None
        worker.close()
        assert order == ["a", "b"]


class TestLazyTierBesideThePool:
    def test_a_lazy_tier_needs_atoms_pool(self):
        sched = lazy_tests.make_umbp_store_scheduler(lazy_tests._config())
        umbp_like = types.SimpleNamespace()  # no write-back hooks
        with pytest.raises(ValueError, match="chooses its own evictions"):
            sched.bind_block_manager(
                types.SimpleNamespace(kv=umbp_like, enable_prefix_caching=True)
            )


class _Receiving:
    """A scheduler-side connector whose P/D receive writes whole block tables,
    or skips what this engine holds (incremental push)."""

    def __init__(self, whole):
        self.whole = whole

    def receive_rewrites_prefix(self, seq):
        return self.whole

    def make_block_pool(self, *args, **kwargs):
        return None


class TestReceiveGuard:
    """D beside a pool other ranks read: a request its receive writes whole
    takes no prefix hit and starts no copy (ROCm/ATOM#3919's rule, for
    MoRIIO)."""

    BLOCKS_HIT = 5  # 20 tokens in blocks of 4

    @pytest.fixture
    def engine(self, l1, monkeypatch):
        from atom.model_engine import block_manager as bm_module

        cfg = MockConfig(
            num_kvcache_blocks=32, kv_cache_block_size=4, enable_prefix_caching=True
        )
        made = iter(range(8))

        def make(config, num_blocks, *, on_evict=None, on_remap=None, **_):
            return umbp_tests.make_pool(
                l1,
                f"rank{next(made)}",
                num_blocks,
                on_evict=on_evict,
                on_remap=on_remap,
            )

        monkeypatch.setattr(bm_module, "make_block_pool", make)
        return lambda whole: bm_module.BlockManager(cfg, kv_connector=_Receiving(whole))

    @staticmethod
    def compute(bm, seq_factory, tokens):
        seq = seq_factory(tokens)
        bm.allocate(seq, 0)
        bm.hash_blocks(seq, len(tokens))
        if bm.kv.read_by_other_pools:  # UMBP's: the device reports its writes
            umbp_tests.settle(bm.kv)
        return seq

    def test_received_whole_it_takes_no_hit_even_from_its_own_pool(
        self, engine, seq_factory
    ):
        rank = engine(True)
        tokens = list(range(20))
        self.compute(rank, seq_factory, tokens)
        calls = []
        umbp_tests.count_fetches(rank, calls)
        received = seq_factory(tokens + [100, 101])
        assert not rank.shared_prefix_pending(received)
        assert rank.can_allocate(received) == 0
        assert calls == []

    def test_received_whole_it_copies_nothing_from_another_rank(
        self, engine, seq_factory
    ):
        rank0, rank1 = engine(False), engine(True)
        tokens = list(range(20))
        self.compute(rank0, seq_factory, tokens)
        calls = []
        umbp_tests.count_fetches(rank1, calls)
        received = seq_factory(tokens + [100, 101])
        assert not rank1.shared_prefix_pending(received)
        assert rank1.can_allocate(received) == 0
        assert calls == []

    def test_with_incremental_push_it_hits_what_another_rank_holds(
        self, engine, seq_factory
    ):
        rank0, rank1 = engine(False), engine(False)
        tokens = list(range(20))
        self.compute(rank0, seq_factory, tokens)
        received = seq_factory(tokens + [100, 101])
        if rank1.kv.async_fetch:
            assert rank1.shared_prefix_pending(received)
            umbp_tests.land(rank1, received)
        assert rank1.can_allocate(received) == self.BLOCKS_HIT

    def test_a_request_holding_blocks_starts_no_copy(self, engine, seq_factory):
        rank0, rank1 = engine(False), engine(False)
        tokens = list(range(20))
        self.compute(rank0, seq_factory, tokens)
        calls = []
        umbp_tests.count_fetches(rank1, calls)
        parked = seq_factory(tokens + [100, 101])
        parked.block_table = [0]  # holds its blocks while its receive lands
        rank1._fetch_shared_prefix(parked)
        assert calls == []
        assert parked not in rank1._shared_prefix_asked

    def test_atoms_own_pool_keeps_its_hits(self, seq_factory):
        from atom.model_engine.block_manager import BlockManager
        from atom.model_engine.block_pool import BlockPool

        cfg = MockConfig(
            num_kvcache_blocks=32, kv_cache_block_size=4, enable_prefix_caching=True
        )
        rank = BlockManager(cfg, kv_connector=_Receiving(True))
        assert isinstance(rank.kv, BlockPool) and not rank.kv.read_by_other_pools
        tokens = list(range(20))
        self.compute(rank, seq_factory, tokens)
        assert rank.can_allocate(seq_factory(tokens + [100, 101])) == self.BLOCKS_HIT


class TestWhoReceivesWhole:
    def test_moriio_without_incremental_push(self, stack, monkeypatch):
        sched = object.__new__(stack.conn.MoRIIOConnectorScheduler)
        sched.is_producer = False
        remote = types.SimpleNamespace(
            kv_transfer_params={"do_remote_prefill": True}, has_per_req_cache=False
        )
        local = types.SimpleNamespace(kv_transfer_params=None, has_per_req_cache=False)
        monkeypatch.setenv("ATOM_MORIIO_INCREMENTAL_PUSH", "0")
        assert sched.receive_rewrites_prefix(remote)
        assert not sched.receive_rewrites_prefix(local)
        monkeypatch.setenv("ATOM_MORIIO_INCREMENTAL_PUSH", "1")
        assert not sched.receive_rewrites_prefix(remote)
        # Per-request state is pushed whole whatever the flag says.
        stateful = types.SimpleNamespace(
            kv_transfer_params={"do_remote_prefill": True}, has_per_req_cache=True
        )
        assert sched.receive_rewrites_prefix(stateful)
        sched.is_producer = True
        assert not sched.receive_rewrites_prefix(remote)

    def test_multi_asks_every_sub(self):
        seq = types.SimpleNamespace()
        assert _sched([FakeSchedSub(), _Receiving(True)]).receive_rewrites_prefix(seq)
        assert not _sched([FakeSchedSub(), _Receiving(False)]).receive_rewrites_prefix(
            seq
        )
