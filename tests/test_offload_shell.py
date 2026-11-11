# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The public connector name selects and delegates to one layout family."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import ConnectorCompletion, StateTouchId
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
from atom.kv_transfer.offload.hybrid.dsv4.connector import (
    DSV4OffloadConnector,
    DSV4OffloadScheduler,
)
from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
    STATE_TOUCH_CHANNEL,
    KimiK3OffloadScheduler,
)
from atom.model_engine.state_offload import StateOffloadIndex


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


def test_shells_refuse_deepseek_v41(monkeypatch):
    """V4.1 is MP-only: the dense codec would restore KV over a window ring
    it does not carry."""
    from atom.kv_transfer.offload import connector as shell

    monkeypatch.setattr(shell, "select_offload_layout", lambda _config: "dsv41")
    with pytest.raises(ValueError, match="lmcache_mp"):
        LMCacheOffloadConnector(_config())
    with pytest.raises(ValueError, match="lmcache_mp"):
        LMCacheOffloadConnectorScheduler(_config())


def test_worker_shell_selects_family():
    hybrid = LMCacheOffloadConnector(_config(compress_ratios=[4]))
    dense = LMCacheOffloadConnector(_config(compress_ratios=None))

    assert isinstance(hybrid._impl, DSV4OffloadConnector)
    assert isinstance(dense._impl, DenseOffloadConnector)


def test_scheduler_shell_selects_family(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())

    hybrid = LMCacheOffloadConnectorScheduler(_config(compress_ratios=[4]))
    dense = LMCacheOffloadConnectorScheduler(_config(compress_ratios=None))

    assert isinstance(hybrid._impl, DSV4OffloadScheduler)
    assert isinstance(dense._impl, DenseOffloadScheduler)
    assert hybrid.is_offload and not hybrid.is_producer


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
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    config = _config(compress_ratios=None)

    worker = KVConnectorFactory.create_connector(config, role="worker")
    scheduler = KVConnectorFactory.create_connector(config, role="scheduler")

    assert isinstance(worker, LMCacheOffloadConnector)
    assert isinstance(scheduler, LMCacheOffloadConnectorScheduler)


def test_scheduler_shell_forwards_completion_processing():
    calls = []
    output = object()
    shell = LMCacheOffloadConnectorScheduler.__new__(LMCacheOffloadConnectorScheduler)
    shell._impl = SimpleNamespace(
        process_completions=lambda value: calls.append(("completions", value)) or value,
        load_finished=lambda req: calls.append(("load-ok", req)) or False,
    )

    assert shell.process_completions(output) is output
    assert shell.load_finished("c") is False
    assert calls == [
        ("completions", output),
        ("load-ok", "c"),
    ]


def test_scheduler_shell_forwards_statistics():
    expected = {
        "load_requests": 2,
        "loaded_tokens": 512,
        "load_failures": 1,
        "save_requests": 3,
        "saved_tokens": 768,
        "loads_pending": 0,
        "saves_pending": 1,
    }
    shell = LMCacheOffloadConnectorScheduler.__new__(LMCacheOffloadConnectorScheduler)
    shell._impl = SimpleNamespace(get_statistics=lambda: expected)

    assert shell.get_statistics() is expected


@pytest.mark.parametrize("composite", [False, True], ids=["shell", "multi"])
def test_state_touches_round_trip_through_the_scheduler_shells(
    composite, offload_clock
):
    impl = KimiK3OffloadScheduler.__new__(KimiK3OffloadScheduler)
    impl._pending_state_touches = []
    impl._touch_generation = 0
    impl._state_touch_verdicts = {}
    connector = LMCacheOffloadConnectorScheduler.__new__(
        LMCacheOffloadConnectorScheduler
    )
    connector._impl = impl
    if composite:
        shell = connector
        connector = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
        connector._connectors = [SimpleNamespace(), shell]
    idx = StateOffloadIndex(verified_for_s=5.0)
    idx.note_stored(7)
    offload_clock[0] += 1
    idx.want_touch(7)
    idx.publish_touches(connector)
    assert idx.refresh_hits, "the touch reached the tier"
    impl.connector_completion(
        ConnectorCompletion(STATE_TOUCH_CHANNEL, StateTouchId(7, 0), False)
    )
    idx.collect_touch_verdicts(connector)
    assert not idx.could_serve(7), "the verdict came back"
