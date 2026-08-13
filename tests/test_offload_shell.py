# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The public connector name selects and delegates to one layout family."""

from __future__ import annotations

from types import SimpleNamespace

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.connector import (
    LMCacheOffloadConnector,
    LMCacheOffloadConnectorScheduler,
    select_variant,
)
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.hybrid.connector import (
    HybridOffloadConnector,
    HybridOffloadScheduler,
)


def _config(*, compress_ratios=None, layout=None):
    kv_transfer_config = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
    if layout is not None:
        kv_transfer_config["offload_layout"] = layout
    return SimpleNamespace(
        model="model",
        model_tag="model",
        kv_cache_dtype="fp8",
        kv_cache_block_size=4,
        decode_context_parallel_size=1,
        state_checkpoint_interval_tokens=8,
        tensor_parallel_size=1,
        hf_config=SimpleNamespace(
            compress_ratios=compress_ratios,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=16,
            head_dim=8,
        ),
        kv_transfer_config=kv_transfer_config,
    )


def test_variant_selection_is_config_only():
    assert select_variant(_config(compress_ratios=[4, 128])) == "hybrid"
    assert select_variant(_config(compress_ratios=None)) == "dense"
    assert select_variant(_config(compress_ratios=[4], layout="dense")) == "dense"
    assert select_variant(_config(compress_ratios=None, layout="page_slot")) == "hybrid"


def test_worker_shell_selects_family():
    hybrid = LMCacheOffloadConnector(_config(compress_ratios=[4]))
    dense = LMCacheOffloadConnector(_config(compress_ratios=None))

    assert isinstance(hybrid._impl, HybridOffloadConnector)
    assert isinstance(dense._impl, DenseOffloadConnector)


def test_scheduler_shell_selects_family(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )

    hybrid = LMCacheOffloadConnectorScheduler(_config(compress_ratios=[4]))
    dense = LMCacheOffloadConnectorScheduler(_config(compress_ratios=None))

    assert isinstance(hybrid._impl, HybridOffloadScheduler)
    assert isinstance(dense._impl, DenseOffloadScheduler)
    assert hybrid.is_offload and not hybrid.is_producer
    assert hybrid.recap_prefill_after_finalize is True
    assert dense.recap_prefill_after_finalize is False


def test_factory_registration_resolves_public_thin_shells(monkeypatch):
    entry = KVConnectorFactory._registry["lmcache_offload"]
    assert entry == {
        "worker_module": "atom.kv_transfer.offload.connector",
        "worker_class": "LMCacheOffloadConnector",
        "scheduler_module": "atom.kv_transfer.offload.connector",
        "scheduler_class": "LMCacheOffloadConnectorScheduler",
    }
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )
    config = _config(compress_ratios=None)

    worker = KVConnectorFactory.create_connector(config, role="worker")
    scheduler = KVConnectorFactory.create_connector(config, role="scheduler")

    assert isinstance(worker, LMCacheOffloadConnector)
    assert isinstance(scheduler, LMCacheOffloadConnectorScheduler)


def test_scheduler_shell_forwards_extended_hybrid_hooks():
    calls = []
    shell = LMCacheOffloadConnectorScheduler.__new__(LMCacheOffloadConnectorScheduler)
    shell._impl = SimpleNamespace(
        connector_meta_dispatched=lambda meta: calls.append(("dispatch", meta)),
        sidecar_save_finished=lambda req: calls.append(("sidecar-ok", req)),
        sidecar_save_failed=lambda req: calls.append(("sidecar-fail", req)),
        load_finished=lambda req: calls.append(("load-ok", req)) or False,
    )

    shell.connector_meta_dispatched("meta")
    shell.sidecar_save_finished("a")
    shell.sidecar_save_failed("b")

    assert shell.load_finished("c") is False
    assert calls == [
        ("dispatch", "meta"),
        ("sidecar-ok", "a"),
        ("sidecar-fail", "b"),
        ("load-ok", "c"),
    ]
