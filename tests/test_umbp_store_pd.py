# SPDX-License-Identifier: MIT
"""port note: umbp_store on the M3.1 PP4 x TP1 fabric prefill (CPU only).

- `multi[moriio fabric producer, umbp_store]` is the prefill's topology: the
  factory sees both leaves, the M3 PAGE map is read, and the fabric hook still
  allocates the paged pool under `multi`.
- Each PP stage's real publication (the M3 builder's NVFP4 K/V + FP4 index,
  the DSpark draft merged on the last stage, `merge_pages`) is a complete PAGE
  map `pure_page_codec` takes, and umbp_store registers on it with the stage's
  own key prefix `w{pp}of4`.
"""

from __future__ import annotations

import json
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from kv_store_test_support import engine_config, use_one_rank
from test_moriio_pp_fabric import PARTITION, rank_pools, stage_layers, stack  # noqa: F401
from test_umbp_store_offload import ADDRESS, FakeServer

from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
from atom.kv_transfer.offload.kv_store import pool as pool_mod
from atom.kv_transfer.offload.page_codec import pure_page_codec
from atom.kv_transfer.offload.umbp_store import client as umbp_client
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store.worker import UMBPStoreOffloadConnector

BLOCKS = 8
CHUNK = 256  # umbp_store's default: two 128-token blocks

P_MULTI_UMBP = {
    "kv_connector": "multi",
    "connectors": [
        {
            "kv_connector": "moriio",
            "kv_role": "kv_producer",
            "moriio_backend": "fabric",
            "moriio_mode": "write",
            "handshake_port": 6301,
        },
        {
            "kv_connector": "umbp_store",
            "kv_role": "offload",
            "kv_connector_extra_config": {
                "umbp_store.address": ADDRESS,
                "umbp_store.chunk_tokens": CHUNK,
                "umbp_store.save_pool_mib": 4,
                "umbp_store.load_pool_mib": 4,
            },
        },
    ],
}


@pytest.fixture(autouse=True)
def _process_state(monkeypatch):
    monkeypatch.setattr(taint, "_TAINTED", {})
    monkeypatch.setattr(umbp_client, "_RETAINED_CLIENTS", [])
    monkeypatch.setattr(pool_mod, "_UNSETTLED_ALLOCATIONS", [])
    monkeypatch.setattr(
        DSV4PageSlotCodec, "has_fused_chunk_major_staging", property(lambda _: True)
    )
    with mock.patch("torch.cuda.current_device", lambda: 0):
        yield


@pytest.fixture
def server(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(umbp_client, "_new_umbp_client", server.connect)
    return server


def test_the_prefill_topology_is_moriio_then_umbp_store(stack):
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        validate_multi_config,
    )

    kv = json.loads(json.dumps(P_MULTI_UMBP))
    assert len(validate_multi_config(kv)) == 2
    cfg = SimpleNamespace(
        kv_transfer_config=kv,
        hf_config=SimpleNamespace(model_type="minimax_m3", architectures=[]),
    )
    leaves = KVConnectorFactory.leaf_connectors(cfg)
    assert [name for name, _ in leaves] == ["moriio", "umbp_store"]
    assert KVConnectorFactory.topology_reads_block_regions(cfg)
    # MoRIIO parses one region per role, so the builder keeps its P/D map.
    assert not KVConnectorFactory.topology_region_readers_copy_whole_blocks(cfg)


@pytest.mark.parametrize("stage", range(len(PARTITION)))
def test_every_pp4_stage_publication_is_one_complete_page_codec(stack, stage):
    last = stage == len(PARTITION) - 1
    side = rank_pools(stack, stage_layers(stage), 1, last, BLOCKS, "kv_producer")
    tt = side.tt
    codec = pure_page_codec(tt)
    regions = list(tt.block_regions)
    assert codec.bytes_per_block == sum(r.unit_bytes for r in regions)
    assert all(r.shard_world == 1 for r in regions)  # no index shard at TP1
    # Every region lies in the stage's one paged pool (the fabric hook's).
    base, end = side.pool.data_ptr(), side.pool.data_ptr() + side.pool.numel()
    assert all(base <= r.base_addr and r.base_addr + r.total_bytes <= end for r in regions)
    roles = sorted({str(getattr(r, "semantic_role", "")) for r in regions})
    print(
        f"stage {stage}: layers={len(stage_layers(stage))} draft={last} "
        f"regions={len(regions)} bytes/block={codec.bytes_per_block} roles={roles}"
    )


@pytest.mark.parametrize("stage", [0, len(PARTITION) - 1])
def test_umbp_store_registers_on_a_stage_with_its_own_key_prefix(
    stack, server, monkeypatch, stage
):
    use_one_rank(monkeypatch)
    last = stage == len(PARTITION) - 1
    side = rank_pools(stack, stage_layers(stage), 1, last, BLOCKS, "kv_producer")
    extra = P_MULTI_UMBP["connectors"][1]["kv_connector_extra_config"]
    config = engine_config(
        "umbp_store",
        dict(extra),
        pp=len(PARTITION),
        layers=60,
        kv_cache_block_size=128,
        index_cache_dtype="fp4",
        parallel_config=types.SimpleNamespace(pipeline_parallel_rank=stage),
    )
    config.hf_config.model_type = "minimax_m3"
    worker = UMBPStoreOffloadConnector(config)
    try:
        worker.register_kv_caches({}, side.tt, num_blocks=BLOCKS)
        assert worker._rank == stage and worker._world == len(PARTITION)
        assert worker._chunk_bytes == worker._codec.bytes_per_block * CHUNK // 128
        [probe] = [k for k in server.at() if "/probe/" in k]
        assert f"/probe/w{stage}/" in probe
    finally:
        worker.close()


def test_an_index_sharded_deployment_is_refused():
    from atom.config import DCPConfig
    from atom.kv_transfer.offload.umbp_store.config import (
        check_engine_compatibility,
        parse_umbp_store_config,
    )

    extra = P_MULTI_UMBP["connectors"][1]["kv_connector_extra_config"]
    cfg = parse_umbp_store_config(extra)
    config = engine_config(
        "umbp_store",
        dict(extra),
        kv_cache_block_size=128,
        dcp_config=DCPConfig(indexer_dcp_only=True, index_shard=True),
    )
    config.hf_config.model_type = "minimax_m3"
    with pytest.raises(ValueError, match="ATOM_M3_INDEX_SHARD"):
        check_engine_compatibility(cfg, config)
