# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The public LMCache MP connector selects a model-specific implementation."""

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload.mp import connector as mp_shell
from atom.kv_transfer.offload.mp.connector import (
    LMCacheMPConnector,
    LMCacheMPConnectorScheduler,
    select_model,
)
from atom.kv_transfer.offload.mp.glm52 import GLM52MPConnectorPlugin
from atom.kv_transfer.offload.mp.registry import resolve_plugin


def _config(model_type: str = "glm_moe_dsa"):
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            text_config=SimpleNamespace(model_type=model_type),
        ),
        kv_transfer_config={
            "kv_connector": "lmcache_mp",
            "kv_role": "offload",
        },
    )


def test_model_selection_is_config_only():
    assert select_model(_config()) == "glm52"
    assert resolve_plugin(_config()) is GLM52MPConnectorPlugin

    with pytest.raises(NotImplementedError, match="no connector"):
        select_model(_config("unsupported_model"))


def test_worker_shell_delegates_to_selected_model(monkeypatch):
    calls = []
    finished = object()
    impl = SimpleNamespace(
        register_kv_caches=lambda *args: calls.append(("register", args)),
        start_load_kv=lambda metadata: calls.append(("start", metadata)),
        get_finished=lambda: finished,
        get_finished_recv_blocks=lambda: [4, 5],
    )
    monkeypatch.setattr(mp_shell, "_build_worker", lambda _config: impl)
    shell = LMCacheMPConnector(_config())
    caches = {"layer.0": object()}
    metadata = object()

    shell.register_kv_caches(caches, "transfer", 8)
    shell.start_load_kv(metadata)

    assert shell.get_finished() is finished
    assert shell.get_finished_recv_blocks() == [4, 5]
    assert calls == [
        ("register", (caches, "transfer", 8)),
        ("start", metadata),
    ]


def test_scheduler_shell_forwards_model_hooks():
    calls = []
    output = object()
    statistics = {"load_requests": 1}
    shell = LMCacheMPConnectorScheduler.__new__(LMCacheMPConnectorScheduler)
    shell._impl = SimpleNamespace(
        process_completions=lambda value: calls.append(("complete", value)) or value,
        load_failed=lambda req: calls.append(("failed", req)) or False,
        get_statistics=lambda: statistics,
    )

    assert shell.process_completions(output) is output
    assert shell.load_failed("request") is False
    assert shell.get_statistics() is statistics
    assert calls == [
        ("complete", output),
        ("failed", "request"),
    ]


def test_factory_registration_resolves_public_shells(monkeypatch):
    entry = KVConnectorFactory._registry["lmcache_mp"]
    assert entry == {
        "worker_module": "atom.kv_transfer.offload.mp.connector",
        "worker_class": "LMCacheMPConnector",
        "scheduler_module": "atom.kv_transfer.offload.mp.connector",
        "scheduler_class": "LMCacheMPConnectorScheduler",
    }
    monkeypatch.setattr(
        mp_shell,
        "_build_worker",
        lambda _config: SimpleNamespace(),
    )
    monkeypatch.setattr(
        mp_shell,
        "_build_scheduler",
        lambda _config: SimpleNamespace(),
    )
    config = _config()

    worker = KVConnectorFactory.create_connector(config, role="worker")
    scheduler = KVConnectorFactory.create_connector(config, role="scheduler")

    assert isinstance(worker, LMCacheMPConnector)
    assert isinstance(scheduler, LMCacheMPConnectorScheduler)
    assert KVConnectorFactory.canonical_name("LMCacheMPConnector") == "lmcache_mp"
