# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public contracts for the layout-neutral LMCache MP connector."""

from types import SimpleNamespace

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload.mp import backend
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


def test_public_connector_exports_generic_implementation():
    assert LMCacheMPConnector is backend.LMCacheMPConnector
    assert LMCacheMPConnectorScheduler is backend.LMCacheMPConnectorScheduler


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
