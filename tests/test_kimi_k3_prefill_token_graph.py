# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 prefill token graphs: which steps record."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="the Kimi builder imports the AITER runtime")


def test_kimi_records_prefill_token_graphs_only_without_dp():
    """Under DP every rank pads to a prefill bucket some rank chose, so a
    decoding or idle rank would replay a recording whose regions serve a
    prefill alone."""
    from atom.model_ops.attentions.kimi_mla_gdn_attn import _KimiMLAGDNCommon

    def supports(runner):
        builder = SimpleNamespace(model_runner=runner)
        return _KimiMLAGDNCommon.supports_token_graph.fget(builder)

    def runner(dp):
        parallel = SimpleNamespace(data_parallel_size=dp)
        return SimpleNamespace(config=SimpleNamespace(parallel_config=parallel))

    assert supports(runner(1))
    assert not supports(runner(2))
    assert not supports(runner(8))
    assert not supports(SimpleNamespace())
