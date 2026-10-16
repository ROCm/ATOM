# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 on the native Mooncake Store offload, on CPU.

The Store is the fake of ``test_mooncake_store_offload``; the state images
are packed by a byte-copying stand-in for the Triton packer.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from test_mooncake_store_offload import (  # noqa: F401  # fixtures
    CHUNK,
    MASTER,
    METADATA,
    _config,
    _extra,
    aiter_arch,
    cluster,
)

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    SaveOperationId,
    SaveSourceGroupId,
    StateStoreOperationId,
)
from atom.kv_transfer.offload.chunked_scheduler import DENSE_PAGE_SOURCE_SAFE_CHANNEL
from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
    STATE_INDEX_CHANNEL,
    KimiK3OffloadScheduler,
)
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store import kimi_k3 as k3_store
from atom.kv_transfer.offload.mooncake_store.config import (
    parse_mooncake_store_config,
)
from atom.kv_transfer.offload.mooncake_store.kimi_k3 import (
    KimiK3MooncakeStoreConnector,
    KimiK3MooncakeStoreScheduler,
    StoreStateCodec,
    state_key_prefix,
)
from atom.kv_transfer.offload.mooncake_store.pool import (
    TransferSlotPool,
    pool_allocation_bytes,
)
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    MooncakeStoreOffloadScheduler,
)
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)
from atom.model_engine.state_offload import (
    state_tier_capability,
    state_tier_chunk_tokens,
)

NAMESPACE = "atomkv1-k3"
ENTRY = 64  # bytes per state image: two 32-byte segments
LAYOUT = "kimi_k3:tp=1:spec=4"


def _k3_config(**fields):
    config = _config(**fields)
    config.hf_config = SimpleNamespace(num_hidden_layers=4, model_type="kimi_linear")
    return config


class FakeBackend:
    """PAGE units and Active Slots of 64-byte images, in two segments each."""

    def __init__(self) -> None:
        self.units: dict[int, list[torch.Tensor]] = {}
        self.slots: dict[int, list[torch.Tensor]] = {}

    def image(self, unit: int) -> bytes:
        return bytes((unit * 7 + i) % 256 for i in range(ENTRY))

    def page_unit_views(self, unit_ids):
        [unit] = unit_ids
        raw = torch.tensor(list(self.image(unit)), dtype=torch.uint8)
        return [raw[: ENTRY // 2], raw[ENTRY // 2 :]]

    def state_entry_views(self, slot):
        views = self.slots.setdefault(
            slot, [torch.zeros(ENTRY // 2, dtype=torch.uint8) for _ in range(2)]
        )
        return views

    def slot_bytes(self, slot) -> bytes:
        return bytes(torch.cat(self.slots[slot]).tolist())


class FakeStaged:
    """`StagedTransfer`'s pack and unpack as plain copies into a slot."""

    def __init__(self) -> None:
        self.packs = 0

    def pack(self, segments, dst) -> None:
        self.packs += 1
        dst.tensor.copy_(torch.cat(list(segments)))

    def unpack(self, src, segments) -> None:
        flat = src.tensor
        offset = 0
        for segment in segments:
            n = segment.numel()
            segment.copy_(flat[offset : offset + n])
            offset += n


@pytest.fixture
def make_codec(cluster):  # noqa: F811  # the imported fixture
    built = []

    def build(*, max_entries=4, refusal=None, distrusted=None):
        client = store_client.MooncakeStoreClient(
            local_hostname="10.0.0.2",
            metadata_server=METADATA,
            master_server_addr=MASTER,
            protocol="tcp",
            rdma_devices="",
        )
        pool = TransferSlotPool(
            device="cpu",
            chunk_bytes=ENTRY,
            save_bytes=2 * 4096,
            load_bytes=2 * 4096,
            client=client,
        )
        backend, staged = FakeBackend(), FakeStaged()
        codec = StoreStateCodec(
            backend,
            staged,
            ENTRY,
            client=client,
            pool=pool,
            namespace=NAMESPACE,
            rank=1,
            world=8,
            layout_id=LAYOUT,
            max_cache_bytes=max_entries * ENTRY,
            load_refusal=lambda: refusal,
            distrust=(
                distrusted.append if distrusted is not None else lambda why: None
            ),
            settle=MooncakeStoreOffloadConnector._settle,
        )
        built.append(codec)
        return codec, backend, staged, cluster.store_of(MASTER)

    yield build
    for codec in built:
        codec.close()
        codec._client.close()


# --- state codec --------------------------------------------------------------


def test_an_image_round_trips_hard_pinned_under_the_state_prefix(make_codec):
    codec, backend, _staged, store = make_codec()
    released = []
    assert codec.put(11, [5], on_source_released=lambda: released.append(1))
    assert released == [1]
    key = codec.key(11)
    assert key.startswith(state_key_prefix(NAMESPACE, 1, 8))
    assert key.startswith(f"{NAMESPACE}/state/w1of8/")
    assert store.hard_pinned == {key}
    assert store.objects[key] == backend.image(5)
    assert codec.get(11, 3)
    assert backend.slot_bytes(3) == backend.image(5)
    # The slots went back.
    assert codec._pool.usable("save") == codec._pool.capacity("save")
    assert len(codec._pool._free["save"]) == codec._pool.capacity("save")
    assert len(codec._pool._free["load"]) == codec._pool.capacity("load")


def test_the_key_names_the_layout_and_the_full_unsigned_hash(make_codec):
    codec, *_ = make_codec()
    assert codec.key(2**64 - 1) != codec.key(2**63)
    other = StoreStateCodec.__new__(StoreStateCodec)
    other._prefix, other._layout_id = codec._prefix, b"kimi_k3:tp=1:spec=3"
    assert StoreStateCodec.key(other, 11) != codec.key(11)


def test_the_oldest_image_is_removed_past_the_budget(make_codec):
    codec, _backend, _staged, store = make_codec(max_entries=2)
    for h in (1, 2):
        assert codec.put(h, [h])
    assert codec.get(1, 0)  # 1 is now the newer of the two
    assert codec.put(3, [3])
    assert codec.key(2) not in store.objects
    assert ("remove", codec.key(2), False) in store.calls
    assert set(store.objects) == {codec.key(1), codec.key(3)}
    assert not codec.get(2, 0)


def test_a_held_image_is_not_packed_or_put_again(make_codec):
    codec, _backend, staged, store = make_codec()
    assert codec.put(7, [7])
    puts = [call for call in store.calls if call[0] == "put"]
    released = []
    assert codec.put(7, [7], on_source_released=lambda: released.append(1))
    assert released == [1]
    assert staged.packs == 1
    assert [call for call in store.calls if call[0] == "put"] == puts


def test_a_refused_removal_is_retried_then_forced(make_codec, monkeypatch):
    codec, _backend, _staged, store = make_codec(max_entries=1)
    assert codec.put(1, [1])
    old = codec.key(1)
    store.remove_codes[old] = store_client.OBJECT_HAS_LEASE
    clock = [1000.0]
    monkeypatch.setattr(k3_store.time, "monotonic", lambda: clock[0])
    assert codec.put(2, [2])
    assert old in store.objects and old in codec._pending_removals
    # Retried, unforced, before the next store; still leased.
    clock[0] += 1
    assert codec.put(2, [2])
    assert store.calls.count(("remove", old, False)) == 2
    # Past the force window it is removed whatever leases it.
    clock[0] += k3_store._FORCE_REMOVE_AFTER_S
    assert codec.put(2, [2])
    assert ("remove", old, True) in store.calls
    assert old not in store.objects and not codec._pending_removals


def test_an_image_whose_removal_is_pending_is_held_again(make_codec):
    codec, _backend, staged, store = make_codec(max_entries=1)
    assert codec.put(1, [1])
    store.remove_codes[codec.key(1)] = store_client.OBJECT_HAS_LEASE
    assert codec.put(2, [2])
    packs = staged.packs
    # Stored again before its removal went through: the Store still has it,
    # and it must not be removed under the index that learns it is back.
    assert codec.put(1, [1])
    assert staged.packs == packs
    assert codec.key(1) not in codec._pending_removals
    assert codec.get(1, 0)


def test_a_store_past_its_source_read_deadline_reads_nothing(make_codec):
    codec, _backend, staged, store = make_codec()
    codec.note_store_deadline(9, 0.0)
    released = []
    assert not codec.put(9, [9], on_source_released=lambda: released.append(1))
    # The tier releases the units of a store that never read them.
    assert released == [] and staged.packs == 0
    assert codec.key(9) not in store.objects
    # The deadline belonged to that store only.
    assert codec.put(9, [9])


def test_a_failed_put_stores_nothing_and_frees_its_slot(make_codec):
    codec, _backend, _staged, store = make_codec()
    store.put_codes[codec.key(4)] = store_client.NO_AVAILABLE_HANDLE
    assert not codec.put(4, [4])
    assert codec.key(4) not in codec._cache_keys
    assert len(codec._pool._free["save"]) == codec._pool.capacity("save")


def test_an_unsettled_put_distrusts_the_store(make_codec, slow_calls):
    distrusted = []
    codec, _backend, _staged, store = make_codec(distrusted=distrusted)
    store.put_codes[codec.key(4)] = store_client.TRANSFER_FAIL
    assert not codec.put(4, [4])
    assert codec._pool.quarantined("save") == 1
    assert len(distrusted) == 1


@pytest.fixture
def slow_calls(monkeypatch):
    monkeypatch.setattr(
        store_client.CallClock, "seconds", lambda _self: store_client.BATCH_WAIT_S
    )


def test_a_refused_or_missing_load_unpacks_nothing(make_codec):
    codec, backend, _staged, store = make_codec(refusal="tainted")
    assert codec.put(5, [5])
    assert not codec.get(5, 2)
    assert 2 not in backend.slots
    assert not any(call[0] == "get" for call in store.calls)
    codec2, backend2, *_ = make_codec()
    assert not codec2.get(6, 2)
    assert 2 not in backend2.slots


def test_an_object_of_another_size_is_never_unpacked(make_codec):
    codec, backend, _staged, store = make_codec()
    store.objects[codec.key(8)] = bytes(ENTRY // 2)
    assert not codec.get(8, 1)
    assert 1 not in backend.slots


# --- selection and engine gating --------------------------------------------------


@pytest.mark.parametrize(
    "config, scheduler, worker",
    [
        (_config, MooncakeStoreOffloadScheduler, MooncakeStoreOffloadConnector),
        (_k3_config, KimiK3MooncakeStoreScheduler, KimiK3MooncakeStoreConnector),
    ],
)
@pytest.mark.usefixtures("cluster")
def test_the_registration_picks_the_classes_by_layout(config, scheduler, worker):
    built = KVConnectorFactory.create_connector(config(), role="scheduler")
    assert type(built) is scheduler
    connector = KVConnectorFactory.create_connector(config(), role="worker")
    try:
        assert type(connector) is worker
    finally:
        connector.close()


def test_the_k3_worker_reserves_the_state_pool_too():
    extra = _extra(state_pool_mib=64)
    cfg = parse_mooncake_store_config(extra)
    config = _k3_config(extra=extra)
    dense = MooncakeStoreOffloadConnector.kv_budget_reserve_bytes(config)
    assert KVConnectorFactory.kv_budget_reserve_bytes(config) == (
        dense
        + pool_allocation_bytes(cfg.state_pool_bytes // 2, cfg.state_pool_bytes // 2)
    )


@pytest.mark.usefixtures("cluster")
def test_the_state_pool_must_fit_an_image_each_way():
    config = _k3_config(extra=_extra(state_pool_mib=2))
    worker = KimiK3MooncakeStoreConnector(config)
    try:
        worker._pool = SimpleNamespace(device=torch.device("cpu"))
        worker._client = object()
        worker._gpu_connector = object()
        with pytest.raises(
            ValueError, match="state_pool_mib=2 .* raise it to at least 3$"
        ):
            worker._make_state_codec(None, (1 << 20) + 1, LAYOUT, 0, 1)
    finally:
        worker._pool = worker._client = worker._gpu_connector = None
        worker.close()


def test_the_engine_hosts_the_state_tier_on_the_store():
    extra = _extra(chunk_tokens=4 * CHUNK)
    offload = {
        "kv_connector": "mooncake_store",
        "kv_role": "offload",
        "kv_connector_extra_config": extra,
    }
    config = _k3_config(extra=extra)
    config.kv_transfer_config = {
        "kv_connector": "multi",
        "connectors": [
            {"kv_connector": "mooncake", "kv_role": "kv_producer"},
            offload,
        ],
    }
    assert state_tier_capability(config).hosts_state_tier
    assert state_tier_chunk_tokens(config) == 4 * CHUNK
    dense = _config(extra=extra)
    dense.kv_transfer_config = json.loads(json.dumps(config.kv_transfer_config))
    assert not state_tier_capability(dense).hosts_state_tier


# --- scheduler and worker behaviour ------------------------------------------------


@pytest.mark.usefixtures("cluster")
def test_the_k3_scheduler_hosts_the_tier_and_routes_its_channels():
    scheduler = KVConnectorFactory.create_connector(_k3_config(), role="scheduler")
    assert scheduler.has_state_tier
    assert not getattr(
        KVConnectorFactory.create_connector(_config(), role="scheduler"),
        "has_state_tier",
        False,
    )
    # The K3 mixin and the Store's base share the chunked base's save path,
    # with early release off as on the LMCache path.
    assert not scheduler._early_release
    assert KimiK3OffloadScheduler._supports_early_block_release is False
    op = StateStoreOperationId(123, 0)
    assert scheduler.connector_completion(
        ConnectorCompletion(STATE_INDEX_CHANNEL, op, True)
    )
    assert scheduler.take_state_reports() == ({op}, set())
    # A KV channel reaches the chunked base, which does not take a malformed id.
    assert (
        scheduler.connector_completion(
            ConnectorCompletion(DENSE_PAGE_SOURCE_SAFE_CHANNEL, "not-a-group", True)
        )
        is False
    )
    # The stall latch and the Store's retry pause both hold saves back.
    assert scheduler._may_emit_save()
    scheduler._save_stalled = True
    assert not scheduler._may_emit_save()
    scheduler._save_stalled = False
    scheduler._saves_paused_until = float("inf")
    assert not scheduler._may_emit_save()
    assert scheduler.save_abandon_timeout_s() == 300.0


@pytest.mark.usefixtures("cluster")
def test_the_k3_worker_publishes_no_source_safe_groups():
    worker = KimiK3MooncakeStoreConnector(_k3_config())
    try:
        assert not worker._early_release
        operation = SaveOperationId("1", 0)
        worker._source_group_safe(SaveSourceGroupId(operation, ((0, CHUNK),)))
        assert not worker._connector_completions
        assert worker._save_abandon_timeout_s() == 300.0
    finally:
        worker.close()


@pytest.mark.usefixtures("cluster")
def test_the_k3_worker_queues_each_state_store_deadline(monkeypatch):
    worker = KimiK3MooncakeStoreConnector(_k3_config())
    noted = []
    worker._state_codec = SimpleNamespace(
        note_store_deadline=lambda h, deadline: noted.append((h, deadline))
    )
    submitted = []
    worker._state_tier = SimpleNamespace(
        submit_store=lambda op, units: submitted.append(op)
    )
    monkeypatch.setattr(k3_store.time, "monotonic", lambda: 50.0)
    op = StateStoreOperationId(77, 2)
    try:
        worker._start_state_stores(SimpleNamespace(state_stores=[(op, [3])]))
        assert noted == [(77, 50.0 + worker._source_read_window_s)]
        assert submitted == [op]
    finally:
        worker._state_codec = worker._state_tier = None
        worker.close()
