# SPDX-License-Identifier: MIT
"""Real lookup construction boundaries preserve passive and unavailable clients."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from test_lmcache_mp import _config

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
from atom.kv_transfer.offload.lookup_client import LookupClient, as_lookup_client
from atom.model_engine.dp_routing.topology import shared_prefix_routing_enabled


def test_owned_and_unavailable_lookup_identity():
    owned = LookupClient()
    assert as_lookup_client(owned) is owned
    assert as_lookup_client(None) is None


@pytest.mark.parametrize("available", [False, True])
def test_dense_factory_passive_lookup_contract(monkeypatch, available):
    # Exercise the external LMCache factory -> Dense -> shared base constructor.
    external = SimpleNamespace(lookup=lambda *_args: 512)
    module = ModuleType("lmcache.v1.lookup_client.factory")

    def create(*_args):
        if not available:
            raise ConnectionError("lookup service unavailable")
        return external

    module.LookupClientFactory = SimpleNamespace(create_lookup_client=create)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        offcfg, "build_lmcache_config", lambda _kvc: SimpleNamespace(chunk_size=512)
    )
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_metadata",
        lambda *_args: SimpleNamespace(engine_id="external"),
    )
    monkeypatch.setattr(offcfg, "lmcache_replica_world_size", lambda _cfg: 1)
    config = _config()
    config.kv_transfer_config = {
        "kv_connector": "lmcache_offload",
        "kv_role": "offload",
    }
    scheduler = DenseOffloadScheduler(config)
    seq = SimpleNamespace(id="passive")
    assert not scheduler.lookup_ready(seq)
    scheduler._reconcile_lookups_each_step()
    if available:
        client = scheduler._lookup_client
        assert client.lookup() == 512
        assert client.pending_ids() == ()
        assert client.query_started(seq.id) is None
        client.close()
    else:
        assert scheduler._lookup_client is None


def test_external_async_contract_derives_answer_and_delegates_lifecycle():
    calls = []
    client = as_lookup_client(
        SimpleNamespace(
            pending_ids=lambda: ("done", "active"),
            is_pending=lambda sid: sid == "active",
            pump=lambda: calls.append("pump"),
            close=lambda: calls.append("close"),
        )
    )
    assert client.answered("done")
    assert not client.answered("active")
    assert not client.answered("absent")
    client.pump()
    client.close()
    assert calls == ["pump", "close"]


def test_shared_backend_and_alignment_are_registration_capabilities(monkeypatch):
    name = "review_shared_probe"
    monkeypatch.setitem(KVConnectorFactory._registry, name, {})
    monkeypatch.setitem(KVConnectorFactory._offload, name, True)
    monkeypatch.setitem(KVConnectorFactory._shared_tier, name, True)
    monkeypatch.setitem(KVConnectorFactory._lowers_hbm_hit_to_chunk, name, True)
    assert name in KVConnectorFactory.shared_offload_connectors()
    assert KVConnectorFactory.lowers_hbm_hit_to_chunk(name)
    monkeypatch.setitem(KVConnectorFactory._shared_tier, name, False)
    assert name not in KVConnectorFactory.shared_offload_connectors()


@pytest.mark.parametrize("policy", ["prefix_spread", "share_cache"])
@pytest.mark.parametrize("stage_kind", ["pipeline", "rapidserve"])
def test_shared_prefix_topology_excludes_stages_of_one_replica(policy, stage_kind):
    config = SimpleNamespace(
        dp_load_balance=policy,
        data_parallel_size=2,
        pipeline_parallel_size=2 if stage_kind == "pipeline" else 1,
        enable_rapidserve=stage_kind == "rapidserve",
    )
    assert not shared_prefix_routing_enabled(config)


@pytest.mark.parametrize("policy", ["prefix_spread", "share_cache"])
def test_shared_prefix_topology_reads_folded_parallel_rank_count(policy):
    config = SimpleNamespace(
        dp_load_balance=policy,
        data_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=2),
        tensor_parallel_size=1,
        enable_dp_attention=True,
    )
    assert shared_prefix_routing_enabled(config)
    assert not shared_prefix_routing_enabled(config, routable_engines=1)
