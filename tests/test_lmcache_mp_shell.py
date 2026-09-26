# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public contracts for the capability-selected LMCache MP connector."""

from types import SimpleNamespace

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload.mp import (
    backend,
    native_state_scheduler,
    native_state_worker,
)
from atom.kv_transfer.offload.mp.connector import (
    LMCacheMPConnector,
    LMCacheMPConnectorScheduler,
)


def _config():
    return SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "lmcache_mp",
            "kv_role": "offload",
        }
    )


def test_public_connector_is_a_layout_neutral_shell():
    assert LMCacheMPConnector is not backend.LMCacheMPConnector
    assert LMCacheMPConnectorScheduler is not backend.LMCacheMPConnectorScheduler


def test_factory_registration_resolves_public_connectors(monkeypatch):
    entry = KVConnectorFactory._registry["lmcache_mp"]
    assert entry == {
        "worker_module": "atom.kv_transfer.offload.mp.connector",
        "worker_class": "LMCacheMPConnector",
        "scheduler_module": "atom.kv_transfer.offload.mp.connector",
        "scheduler_class": "LMCacheMPConnectorScheduler",
    }

    monkeypatch.setattr(
        LMCacheMPConnector,
        "__init__",
        lambda self, config: setattr(self, "config", config),
    )
    monkeypatch.setattr(
        LMCacheMPConnectorScheduler,
        "__init__",
        lambda self, config: setattr(self, "config", config),
    )
    config = _config()

    worker = KVConnectorFactory.create_connector(config, role="worker")
    scheduler = KVConnectorFactory.create_connector(config, role="scheduler")

    assert isinstance(worker, LMCacheMPConnector)
    assert isinstance(scheduler, LMCacheMPConnectorScheduler)
    assert worker.config is config
    assert scheduler.config is config
    assert KVConnectorFactory.canonical_name("LMCacheMPConnector") == "lmcache_mp"


def test_worker_selects_native_state_from_published_contract(monkeypatch):
    selected = []

    class NativeWorker:
        def __init__(self, config):
            selected.append(("native", config))

        def register_kv_caches(self, caches, tensors, num_blocks):
            self.registered = (caches, tensors, num_blocks)

    class PageWorker(NativeWorker):
        def __init__(self, config):
            selected.append(("page", config))

    monkeypatch.setattr(
        native_state_worker, "NativeStateLMCacheMPConnector", NativeWorker
    )
    monkeypatch.setattr(backend, "LMCacheMPConnector", PageWorker)
    config = _config()
    worker = LMCacheMPConnector(config)
    tensors = SimpleNamespace(
        paged_state_checkpoint_spec=object(),
        execute_paged_state_copies=lambda *_: None,
    )
    worker.register_kv_caches({}, tensors, 7)
    assert selected == [("native", config)]
    assert worker._impl.registered == ({}, tensors, 7)


def test_scheduler_selects_native_state_from_block_manager(monkeypatch):
    selected = []

    class NativeScheduler:
        def __init__(self, config):
            selected.append(("native", config))

        def bind_block_manager(self, manager):
            self.manager = manager

    class PageScheduler:
        def __init__(self, config):
            selected.append(("page", config))

    monkeypatch.setattr(
        native_state_scheduler,
        "NativeStateLMCacheMPConnectorScheduler",
        NativeScheduler,
    )
    monkeypatch.setattr(backend, "LMCacheMPConnectorScheduler", PageScheduler)
    config = _config()
    scheduler = LMCacheMPConnectorScheduler(config)
    manager = SimpleNamespace(paged_state_checkpoints=object())
    scheduler.bind_block_manager(manager)
    assert selected == [("native", config)]
    assert scheduler._impl.manager is manager


def test_scheduler_shell_forwards_block_lifecycle_hooks():
    calls = []
    scheduler = LMCacheMPConnectorScheduler(_config())
    scheduler._impl = SimpleNamespace(
        should_defer_free=lambda seq: seq == "held",
        send_finished=lambda req_id: calls.append(("sent", req_id)),
        source_blocks_released=lambda seq: calls.append(("released", seq)),
    )

    assert scheduler.should_defer_free("held") is True
    assert scheduler.should_defer_free("free") is False
    scheduler.send_finished("request-1")
    scheduler.source_blocks_released("request-2")
    assert calls == [("sent", "request-1"), ("released", "request-2")]


def test_plain_page_layout_uses_generic_implementations(monkeypatch):
    selected = []

    class PageWorker:
        def __init__(self, config):
            selected.append("worker")

        def register_kv_caches(self, *_):
            pass

    class PageScheduler:
        def __init__(self, config):
            selected.append("scheduler")

    monkeypatch.setattr(backend, "LMCacheMPConnector", PageWorker)
    monkeypatch.setattr(backend, "LMCacheMPConnectorScheduler", PageScheduler)
    LMCacheMPConnector(_config()).register_kv_caches({}, SimpleNamespace(), 7)
    LMCacheMPConnectorScheduler(_config()).bind_block_manager(
        SimpleNamespace(paged_state_checkpoints=None)
    )
    assert selected == ["worker", "scheduler"]


def test_worker_rejects_paged_state_without_native_contract(monkeypatch):
    """A backend whose state is PAGE-copied but that publishes no native contract.

    `BlockManager` builds `paged_state_checkpoints` from the same
    `state_transfer().copies`, so the scheduler shell selects the native-state
    implementation. A PAGE-only worker would then store under a namespace the
    native scheduler never looks up: offload silently never hits.
    """
    import pytest

    selected = []

    class PageWorker:
        def __init__(self, config):
            selected.append("page")

        def register_kv_caches(self, *_):
            pass

    monkeypatch.setattr(backend, "LMCacheMPConnector", PageWorker)
    builder = SimpleNamespace(
        state_transfer=lambda: SimpleNamespace(copies=True),
    )
    tensors = SimpleNamespace(state_backend=builder)

    with pytest.raises(NotImplementedError, match="paged_state_checkpoint_spec"):
        LMCacheMPConnector(_config()).register_kv_caches({}, tensors, 7)
    assert selected == []


def test_worker_keeps_page_only_path_for_forked_state(monkeypatch):
    selected = []

    class PageWorker:
        def __init__(self, config):
            selected.append("page")

        def register_kv_caches(self, *_):
            pass

    monkeypatch.setattr(backend, "LMCacheMPConnector", PageWorker)
    builder = SimpleNamespace(
        state_transfer=lambda: SimpleNamespace(copies=False),
    )
    LMCacheMPConnector(_config()).register_kv_caches(
        {}, SimpleNamespace(state_backend=builder), 7
    )
    assert selected == ["page"]
