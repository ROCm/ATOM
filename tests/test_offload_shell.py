# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The single ``lmcache_offload`` shell picks + delegates to a family impl."""

from __future__ import annotations

from types import SimpleNamespace

from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.hybrid.connector import (
    HybridOffloadConnector,
    HybridOffloadScheduler,
)
from atom.kv_transfer.offload.connector import (
    LMCacheOffloadConnector,
    LMCacheOffloadConnectorScheduler,
)


def _cfg(compress_ratios=None):
    hf = SimpleNamespace(
        head_dim=128, sliding_window=128, num_hidden_layers=4
    )
    if compress_ratios is not None:
        hf.compress_ratios = compress_ratios
    return SimpleNamespace(
        model="m",
        kv_cache_dtype="auto",
        kv_cache_block_size=128,
        tensor_parallel_size=1,
        hf_config=hf,
        kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": "offload"},
    )


def test_scheduler_shell_selects_hybrid():
    sch = LMCacheOffloadConnectorScheduler(_cfg(compress_ratios=[4, 4, 0]))
    assert isinstance(sch._impl, HybridOffloadScheduler)
    assert sch.is_offload and not sch.is_producer
    # dense-only hook defaults to no-op on the hybrid impl.
    assert sch.adjust_prefill_chunk_after_alloc(SimpleNamespace(), 17) == 17
    assert sch.should_park_partial_prefill_for_load(SimpleNamespace()) is False


def test_scheduler_shell_selects_dense():
    sch = LMCacheOffloadConnectorScheduler(_cfg(compress_ratios=None))
    assert isinstance(sch._impl, DenseOffloadScheduler)
    assert sch.is_offload and not sch.is_producer


def test_scheduler_shell_override():
    # "dense" is the current name; "chunked"/"chunked_mla" stay accepted as legacy.
    for layout in ("dense", "chunked"):
        sch = LMCacheOffloadConnectorScheduler(
            SimpleNamespace(
                **{
                    **vars(_cfg(compress_ratios=[4])),
                    "kv_transfer_config": {
                        "kv_connector": "lmcache_offload",
                        "offload_layout": layout,
                    },
                }
            )
        )
        assert isinstance(sch._impl, DenseOffloadScheduler)


def test_worker_shell_selects_family():
    w_h = LMCacheOffloadConnector(_cfg(compress_ratios=[4, 4]))
    assert isinstance(w_h._impl, HybridOffloadConnector)
    w_c = LMCacheOffloadConnector(_cfg(compress_ratios=None))
    assert isinstance(w_c._impl, DenseOffloadConnector)
    assert not w_h.is_producer
