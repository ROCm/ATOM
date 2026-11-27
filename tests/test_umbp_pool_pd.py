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
import test_umbp_store_lazy as lazy_tests
import torch
from test_multi_connector import FakeWorkerSub, _worker

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    validate_multi_config,
)
from atom.kv_transfer.pool.umbp import pool as umbp_pool
from atom.kv_transfer.pool.umbp.settings import UmbpPoolSettings

stack = test_moriio_fabric.stack

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
