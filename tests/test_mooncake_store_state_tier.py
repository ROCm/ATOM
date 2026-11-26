# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 on the Mooncake Store (`mooncake_store.state_tier`), on CPU: the fakes
of `test_mooncake_store_offload`, and the state gather as a CPU copy."""

# Pytest discovers these imported fixtures by their original names.
# ruff: noqa: F811

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from test_mooncake_store_offload import (  # noqa: F401  # fixtures
    BLOCK,
    CHUNK,
    MASTER,
    METADATA,
    _completed_match,
    _config,
    _extra,
    _seq,
    aiter_arch,
    cluster,
)

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import StateStoreOperationId
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import StateOffloadFace
from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import TouchResult
from atom.kv_transfer.offload.hybrid.kimi_k3.state_tier import StateOffloadTier
from atom.kv_transfer.offload.kv_store import pool as pool_mod
from atom.kv_transfer.offload.kv_store.scheduler import prompt_chunk_hashes
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store import keys
from atom.kv_transfer.offload.mooncake_store import state_tier as store_k3
from atom.kv_transfer.offload.mooncake_store import taint as taint_mod
from atom.kv_transfer.offload.mooncake_store.config import (
    check_engine_compatibility,
    parse_mooncake_store_config,
)
from atom.kv_transfer.offload.mooncake_store.layouts import (
    KIMI_K3_CODEC_ID,
    mooncake_codec_id,
)
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    MooncakeStoreOffloadScheduler,
)
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)
from atom.model_engine import state_offload
from atom.model_engine.prefix_hash import chained_block_hashes

UNIT = 24  # bytes of one PAGE unit
UNITS = 8
LAYOUT_ID = "kda:test-layout"


def _k3_config(role="offload", **fields):
    return _config(
        role,
        hf_config=SimpleNamespace(num_hidden_layers=4, model_type="kimi_linear"),
        **fields,
    )


class CpuStaged:
    """`StagedTransfer` on CPU: the same gather and scatter, as tensor copies."""

    def __init__(self, device, staging_buffer_bytes, **_):
        self.device = device

    def pack(self, segments, dst):
        flat = torch.cat([s.reshape(-1).view(torch.uint8) for s in segments])
        dst.tensor[: flat.numel()].copy_(flat)

    def unpack(self, src, segments):
        offset = 0
        for s in segments:
            n = s.numel() * s.element_size()
            s.reshape(-1).view(torch.uint8).copy_(src.tensor[offset : offset + n])
            offset += n


class Backend:
    """PAGE units holding a checkpoint image, and the Active Slots it loads into."""

    def __init__(self) -> None:
        self.units = torch.randint(0, 256, (UNITS, UNIT), dtype=torch.uint8)
        self.slots = torch.zeros((4, 2, UNIT), dtype=torch.uint8)
        spec = SimpleNamespace(layout_id=LAYOUT_ID, image_bytes=2 * UNIT)
        self.model_runner = SimpleNamespace(
            state_runtime=SimpleNamespace(checkpoint_spec=spec)
        )

    def page_unit_views(self, unit_ids):
        return [self.units[u] for u in unit_ids]

    def state_entry_views(self, slot):
        return [self.slots[slot, 0], self.slots[slot, 1]]


@pytest.fixture
def k3_worker(cluster, monkeypatch):
    """A K3 Store worker whose state tier is built over `Backend`."""
    monkeypatch.setattr(store_k3, "StagedTransfer", CpuStaged)
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(
            get_tp_group=lambda: SimpleNamespace(world_size=1, rank_in_group=0)
        ),
    )
    built = []

    def build(namespace="atomkv1-k3"):
        worker = MooncakeStoreOffloadConnector(_k3_config())
        client = store_client.MooncakeStoreClient(
            local_hostname="10.0.0.2",
            metadata_server=METADATA,
            master_server_addr=MASTER,
            protocol="tcp",
            rdma_devices="",
        )
        worker._client = client
        worker._taint_marker = taint_mod.TaintMarker(client)
        worker._namespace = namespace
        worker._codec = SimpleNamespace(device=torch.device("cpu"))
        backend = Backend()
        worker._build_state_tier(SimpleNamespace(state_backend=backend))
        built.append(worker)
        return worker, backend, cluster.stores[-1]

    yield build
    for worker in built:
        worker.close()


# --- resolution --------------------------------------------------------------


def test_kimi_k3_is_a_store_layout_with_its_own_namespace():
    cfg = parse_mooncake_store_config(_extra())
    check_engine_compatibility(cfg, _k3_config())
    assert mooncake_codec_id(_k3_config()) == KIMI_K3_CODEC_ID


@pytest.mark.parametrize("role", ["worker", "scheduler"])
def test_the_store_classes_resolve_to_k3_for_a_k3_model(cluster, role):
    from atom.kv_transfer.offload.mooncake_store.state_tier import (
        MooncakeStoreStateConnector,
        MooncakeStoreStateScheduler,
    )

    want = {
        "worker": MooncakeStoreStateConnector,
        "scheduler": MooncakeStoreStateScheduler,
    }[role]
    connector = KVConnectorFactory.create_connector(_k3_config(), role=role)
    assert type(connector) is want
    other = KVConnectorFactory.create_connector(_config(), role=role)
    assert not isinstance(other, want)
    for c in (connector, other):
        c.close()


def test_the_k3_scheduler_hosts_the_state_face_and_both_save_gates(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_k3_config())
    try:
        assert isinstance(scheduler, StateOffloadFace)
        assert scheduler.has_state_tier
        assert not scheduler.supports_early_load
        assert scheduler._max_pending_saves >= 1
        assert scheduler._may_emit_save()
        # The Store's back-off after a failed save still holds saves.
        scheduler._saves_paused_until = float("inf")
        assert not scheduler._may_emit_save()
        scheduler._saves_paused_until = 0.0
        scheduler._save_inflight = dict.fromkeys(
            map(str, range(scheduler._max_pending_saves))
        )
        assert not scheduler._may_emit_save()
    finally:
        scheduler.close()


def test_state_stores_loads_and_touches_ride_the_store_metadata(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_k3_config())
    try:
        assert scheduler.enqueue_state_stores([("op", [1, 2])])
        assert scheduler.enqueue_state_loads([("7", 11, 0)])
        assert scheduler.has_pending_work()
        meta = scheduler.build_connector_meta()
        assert meta.state_stores == [("op", [1, 2])]
        assert meta.state_loads == [("7", 11, 0)]
        assert scheduler.build_connector_meta().state_stores == []
    finally:
        scheduler.close()


# --- keys --------------------------------------------------------------------


def test_state_keys_are_per_worker_and_groups_per_checkpoint():
    k0, g0 = store_k3.state_key_prefixes("ns", 0, 8, LAYOUT_ID)
    k5, g5 = store_k3.state_key_prefixes("ns", 5, 8, LAYOUT_ID)
    other, other_group = store_k3.state_key_prefixes("ns", 0, 8, LAYOUT_ID + "x")
    assert k0 != k5 and g0 == g5
    assert other != k0 and other_group != g0
    # Never a KV chunk key (a 32-hex digest) nor a chunk group.
    assert k0.startswith(keys.rank_key_prefix("ns", 0, 8) + "state-")
    assert not g0.startswith("ns/group/")


# --- images found by key ------------------------------------------------------

REBUILD = 12  # tokens a forward rebuilds the state over


def _bind_window_state(scheduler, *, rebuild=REBUILD, can_load=True):
    """Bind `scheduler` to an engine whose state a forward rebuilds."""
    index = state_offload.StateOffloadIndex(can_load=can_load)
    coordinator = SimpleNamespace(
        rebuild_tokens=rebuild,
        hash_block_size=BLOCK,
        store=SimpleNamespace(spec=SimpleNamespace(layout_id=LAYOUT_ID)),
    )
    scheduler.bind_block_manager(
        SimpleNamespace(paged_state_checkpoints=coordinator, state_offload=index)
    )
    return index


def test_the_probe_asks_every_holder_for_each_boundary_a_rebuild_misses():
    probe = store_k3.WindowImageProbe("ns", 4, range(2, 4), LAYOUT_ID, BLOCK, REBUILD)
    seq = _seq(1, 40, num_tokens=40, cache_seed=-1)
    hashes, asked = probe.keys(seq)
    # A rebuild covers boundaries up to 28 tokens; a hit stops a block short.
    chain = chained_block_hashes(seq.token_ids, BLOCK, 0, 10, -1)
    assert hashes == chain[7:9]
    prefixes = [store_k3.state_key_prefixes("ns", r, 4, LAYOUT_ID)[0] for r in (2, 3)]
    assert asked == [f"{p}/{h:016x}" for h in hashes for p in prefixes]
    assert probe.present(hashes, [1, 1, 1, 0]) == hashes[:1]
    assert probe.present(hashes, [1, None, 1, 1]) == hashes[1:]
    # 20 tokens past the prompt: a rebuild covers every prompt boundary.
    assert probe.keys(_seq(1, 40, num_tokens=60, cache_seed=-1)) == ([], [])


def test_a_window_image_is_found_in_the_kv_lookups_rpc(cluster, k3_worker):
    # Any hit is worth a load: a 40-token prompt is looked up.
    scheduler = MooncakeStoreOffloadScheduler(_k3_config(offload_min_load_tokens=0))
    try:
        index = _bind_window_state(scheduler)
        worker, _backend, _store = k3_worker(namespace=scheduler._namespace)
        seq = _seq(1, 40, num_tokens=40, cache_seed=-1)
        kv = keys.chunk_keys(
            scheduler._namespace, 0, 1, prompt_chunk_hashes(seq, CHUNK), 0, 5
        )
        cluster.objects.setdefault(MASTER, {}).update(dict.fromkeys(kv, b"kv"))
        chain = chained_block_hashes(seq.token_ids, BLOCK, 0, 9, -1)
        codec = worker._state_tier.codec
        assert codec.put(chain[8], [0, 1])  # the image at 36 tokens
        _completed_match(scheduler, seq)
        assert index.could_serve(chain[8]) and not index.could_serve(chain[7])
        [rpc] = [
            call[1]
            for store in cluster.stores
            for call in store.calls
            if call[0] == "exists" and len(call[1]) > 1
        ]
        image_keys = [codec.key(chain[7]), codec.key(chain[8])]
        assert rpc == [keys.STORE_TAINT_KEY, *kv, *image_keys]
    finally:
        scheduler.close()


@pytest.mark.parametrize("rebuild, can_load", [(0, True), (REBUILD, False)])
def test_k3_and_an_engine_that_cannot_load_ask_for_no_image(cluster, rebuild, can_load):
    scheduler = MooncakeStoreOffloadScheduler(_k3_config())
    try:
        _bind_window_state(scheduler, rebuild=rebuild, can_load=can_load)
        assert scheduler._lookup_client.backend.image_probe is None
    finally:
        scheduler.close()


def test_under_pp_the_last_stages_ranks_hold_the_window(cluster):
    scheduler = MooncakeStoreOffloadScheduler(_k3_config(pp=2, tp=2))
    try:
        _bind_window_state(scheduler)
        probe = scheduler._lookup_client.backend.image_probe
        namespace = scheduler._namespace
        assert probe._prefixes == [
            store_k3.state_key_prefixes(namespace, rank, 4, LAYOUT_ID)[0]
            for rank in (2, 3)
        ]
    finally:
        scheduler.close()


# --- the codec over the Store --------------------------------------------------


def test_a_checkpoint_round_trips_through_the_store(k3_worker):
    worker, backend, store = k3_worker()
    codec = worker._state_tier.codec
    assert isinstance(codec, store_k3.StoreStateCodec)
    assert codec.entry_bytes == 2 * UNIT
    released = []
    assert codec.put(0xABC, [5, 2], on_source_released=lambda: released.append(1))
    assert released == [1]
    image = bytes(backend.units[5].tolist() + backend.units[2].tolist())
    assert store.objects[codec.key(0xABC)] == image
    assert store.groups[codec.key(0xABC)] == codec._group(0xABC)
    assert codec.touch(0xABC) is TouchResult.PRESENT
    assert codec.touch(0xABD) is TouchResult.GONE

    assert codec.get(0xABC, 3)
    assert torch.equal(backend.slots[3, 0], backend.units[5])
    assert torch.equal(backend.slots[3, 1], backend.units[2])
    assert not codec.get(0xABD, 2)  # a miss
    assert codec._free.qsize() == store_k3.STATE_LANES

    # Stored once: a second store finds it and only releases its source.
    puts = sum(1 for call in store.calls if call[0] == "put")
    assert codec.put(0xABC, [5, 2], on_source_released=lambda: released.append(2))
    assert released == [1, 2]
    assert sum(1 for call in store.calls if call[0] == "put") == puts


def test_the_tier_unregisters_its_lanes_at_close(k3_worker):
    worker, _backend, store = k3_worker()
    lanes = worker._state_tier.codec._lanes.data_ptr()
    assert store.registered.get(lanes) == store_k3.STATE_LANES * 2 * UNIT
    worker.close()
    assert ("unregister", lanes) in store.calls


def test_a_failed_store_returns_its_lane(k3_worker):
    worker, _backend, store = k3_worker()
    codec = worker._state_tier.codec
    store.put_codes[codec.key(1)] = store_client.TRANSFER_FAIL
    assert not codec.put(1, [0, 1])
    assert codec._free.qsize() == store_k3.STATE_LANES
    assert worker._store_distrusted is None


def test_a_given_up_store_retires_its_lane_and_taints_the_store(k3_worker, monkeypatch):
    worker, _backend, store = k3_worker()
    codec = worker._state_tier.codec
    store.put_codes[codec.key(1)] = store_client.TRANSFER_FAIL
    monkeypatch.setattr(
        store_client.CallClock, "seconds", lambda _self: store_client.BATCH_WAIT_S
    )
    assert not codec.put(1, [0, 1])
    assert codec._free.qsize() == store_k3.STATE_LANES - 1
    assert worker._store_distrusted is not None
    assert keys.STORE_TAINT_KEY in store.objects
    # Loads stop; the retired lane outlives the tier.
    gets = sum(1 for call in store.calls if call[0] == "get")
    assert not codec.get(1, 0)
    assert sum(1 for call in store.calls if call[0] == "get") == gets
    kept = list(pool_mod._UNSETTLED_ALLOCATIONS)
    worker.close()
    assert len(pool_mod._UNSETTLED_ALLOCATIONS) == len(kept) + 1


def test_a_raising_put_retires_its_lane_and_taints_the_store(k3_worker):
    worker, _backend, store = k3_worker()
    codec = worker._state_tier.codec
    store.raise_on = "put"
    with pytest.raises(RuntimeError, match="put blew up"):
        codec.put(1, [0, 1])
    assert codec._free.qsize() == store_k3.STATE_LANES - 1
    assert worker._store_distrusted is not None


def test_a_raising_get_retires_only_its_lane(k3_worker):
    worker, _backend, store = k3_worker()
    codec = worker._state_tier.codec
    assert codec.put(1, [0, 1])
    store.raise_on = "get"
    with pytest.raises(RuntimeError, match="get blew up"):
        codec.get(1, 0)
    assert codec._free.qsize() == store_k3.STATE_LANES - 1
    assert worker._store_distrusted is None


def test_a_load_from_a_marked_store_is_refused_before_and_after_its_get(
    k3_worker, cluster
):
    worker, _backend, store = k3_worker()
    codec = worker._state_tier.codec
    assert codec.put(1, [0, 1])
    cluster.objects[MASTER][keys.STORE_TAINT_KEY] = b""
    assert not codec.get(1, 0)
    assert not any(call[0] == "get" for call in store.calls)
    del cluster.objects[MASTER][keys.STORE_TAINT_KEY]
    worker._store_distrusted = None

    # Marked while the get ran: what it read may be another put's late write.
    real_get = store.batch_get_into

    def get_then_mark(*args):
        codes = real_get(*args)
        cluster.objects[MASTER][keys.STORE_TAINT_KEY] = b""
        return codes

    store.batch_get_into = get_then_mark
    assert not codec.get(1, 0)
    assert codec._free.qsize() == store_k3.STATE_LANES


def test_a_transfer_waits_for_a_lane_then_fails(k3_worker, monkeypatch):
    worker, _backend, _store = k3_worker()
    codec = worker._state_tier.codec
    monkeypatch.setattr(store_k3, "LANE_WAIT_S", 0.01)
    for _ in range(store_k3.STATE_LANES):
        codec._take_lane()
    with pytest.raises(RuntimeError, match="no state lane came free"):
        codec.put(1, [0, 1])


def test_the_tier_drives_the_codec_and_reports(k3_worker):
    worker, backend, _store = k3_worker()
    tier = worker._state_tier
    assert isinstance(tier, StateOffloadTier)
    op = StateStoreOperationId(77, 0)
    tier.submit_store(op, [3, 4], SimpleNamespace(synchronize=lambda: None))
    tier.drain()
    assert tier.take_store_reports() == ({op}, set())
    tier.submit_load("r1", 77, 1)
    tier.drain()
    assert tier.get_finished() == ({"r1"}, set())
    assert torch.equal(backend.slots[1, 0], backend.units[3])


# --- the engine's view ---------------------------------------------------------


def _store_kvc(**overrides):
    kvc = {
        "kv_connector": "mooncake_store",
        "kv_role": "offload",
        "kv_connector_extra_config": _extra(chunk_tokens=512),
    }
    kvc.update(overrides)
    return kvc


def _multi_kvc():
    return {
        "kv_connector": "multi",
        "connectors": [
            {"kv_connector": "mooncake", "kv_role": "kv_producer"},
            _store_kvc(),
        ],
    }


@pytest.mark.parametrize("kvc", [_store_kvc(), _multi_kvc()], ids=["alone", "multi"])
def test_the_engine_sees_a_store_hosted_state_tier(kvc):
    config = _k3_config(kv_transfer_config=kvc)
    capability = state_offload.state_tier_capability(config)
    assert capability.hosts_state_tier, capability.reason
    assert capability.can_store_state and capability.can_load_state
    # The joint boundary is on the Store's chunk grid.
    assert state_offload.state_tier_chunk_tokens(config) == 512
    assert state_offload.state_tier_verified_for_s(config) == 5.0


def test_the_engine_index_takes_no_cpu_budget_cap_on_the_store():
    spec = SimpleNamespace(image_bytes=1 << 20, pipeline_op_bytes=0)
    store = _k3_config(kv_transfer_config=_store_kvc())
    assert state_offload.state_tier_index_cap(spec, store) is None
    lmcache = _k3_config(
        kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": "offload"}
    )
    cap = state_offload.state_tier_index_cap(spec, lmcache)
    assert cap == state_offload.state_tier_cpu_entries(spec, lmcache)
    assert state_offload.state_tier_index_cap(None, lmcache) is None


def test_the_store_hosts_no_state_tier_for_another_layout(monkeypatch):
    monkeypatch.setattr(offcfg, "select_offload_layout", lambda _config: "qwen")
    config = _k3_config(kv_transfer_config=_store_kvc())
    capability = state_offload.state_tier_capability(config)
    assert not capability.hosts_state_tier
    assert "Mooncake Store keeps no 'qwen' state" in capability.reason


def test_two_state_tier_hosts_under_multi_are_refused():
    kvc = {
        "kv_connector": "multi",
        "connectors": [
            {"kv_connector": "lmcache_offload", "kv_role": "offload"},
            _store_kvc(),
        ],
    }
    with pytest.raises(ValueError, match="lists 2 offload connectors"):
        state_offload.state_tier_capability(_k3_config(kv_transfer_config=kvc))


def test_a_stage_holding_no_state_answers_every_image_whole():
    codec = store_k3.NullStateCodec()
    released = []
    assert codec.put(1, [0, 1], on_source_released=lambda: released.append(1))
    assert released == [1]
    assert codec.get(1, 0)
    assert codec.touch(1) is TouchResult.PRESENT
