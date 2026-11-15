# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 prefill token graphs: which steps record, and which rows a padded
prefill's regions run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.utils import forward_context
from atom.utils.region_graph import on_scheduled_rows, padded_prefill_rows


def _step(scheduled, is_prefill=True, padded=True):
    """A step's context; ``padded`` when `ForwardMode.decide` raised it to a
    recorded token-graph bucket."""
    mode = SimpleNamespace(prefill_token_graph=padded)
    return SimpleNamespace(
        is_prefill=is_prefill, scheduled_tokens=scheduled, forward_mode=mode
    )


@pytest.fixture
def rows(monkeypatch):
    def rows(height, context):
        ctx = SimpleNamespace(context=context)
        monkeypatch.setattr(forward_context, "_forward_context", ctx)
        return padded_prefill_rows(height)

    return rows


def test_a_prefill_padded_to_a_bucket_runs_its_scheduled_rows(rows):
    assert rows(32, _step(20)) == 20
    assert rows(20, _step(20)) is None
    assert rows(32, _step(20, is_prefill=False)) is None
    assert rows(32, None) is None


def test_rows_padded_without_a_token_graph_stay_with_the_region(rows):
    """A TBO micro-batch pads past its scheduled rows with no step decision of
    its own, and a step that settled on no bucket pads nothing a replay
    recorded: neither is cut."""
    tbo = SimpleNamespace(is_prefill=True, scheduled_tokens=20, forward_mode=None)
    assert rows(32, tbo) is None
    assert rows(32, _step(20, padded=False)) is None


def test_a_cut_region_sees_its_rows_and_hands_back_the_bucket():
    seen = []

    def region(x, weight, name):
        seen.append((x.shape[0], weight.shape[0], name))
        return x * 2

    x = torch.arange(1.0, 9.0).view(8, 1)
    weight = torch.ones(3)
    out = on_scheduled_rows(region, 5, x, weight, "layer")
    # Only a bucket-high argument is cut.
    assert seen == [(5, 3, "layer")]
    assert out.shape == x.shape
    torch.testing.assert_close(out[:5], x[:5] * 2)
    assert not out[5:].any()
    torch.testing.assert_close(on_scheduled_rows(region, None, x, weight, "l"), x * 2)


@pytest.fixture
def padded_layer(monkeypatch):
    """A prefill of 5 rows padded to an 8-row bucket, and a layer registered
    as "layer" that records the rows each of its calls is handed."""
    pytest.importorskip("aiter", reason="the attention and KDA ops are AITER ops")
    from atom.model_ops import base_attention
    from atom.models.kimi_k3 import model as kimi

    seen = {}

    def attend(query, k_nope, k_rope, positions, q_scale):
        seen["mla"] = (query.shape[0], k_nope.shape[0], k_rope.shape[0])
        seen["positions"] = positions.shape[-1]
        return query[:, :4] + 1

    def mix(hidden_states, hidden_states_scale):
        seen["kda"] = hidden_states.shape[0]
        return hidden_states + 1

    layer = SimpleNamespace(impl=SimpleNamespace(forward=attend), _forward_impl=mix)
    static = {"layer": layer}
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=static)
    )
    for module in (base_attention, kimi):
        monkeypatch.setattr(module, "get_current_atom_config", lambda: config)
    ctx = SimpleNamespace(context=_step(5), main_stream=None, in_hipgraph=False)
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    return seen, base_attention, kimi


def _bucket_of_scheduled_rows(out, width):
    assert out.shape == (8, width)
    assert out[:5].eq(2).all() and not out[5:].any()


def test_mla_attention_runs_a_padded_prefills_scheduled_rows(padded_layer):
    seen, base_attention, _ = padded_layer
    x = torch.ones(8, 6)
    out = base_attention.unified_attention_with_output_base(
        x, None, x, x, torch.arange(8), "layer", True, x
    )
    assert seen == {"mla": (5, 5, 5), "positions": 5}
    _bucket_of_scheduled_rows(out, 4)


def test_kda_mixer_runs_a_padded_prefills_scheduled_rows(padded_layer):
    seen, _, kimi = padded_layer
    out = kimi.kda_attention_with_output(torch.ones(8, 6), None, "layer")
    assert seen == {"kda": 5}
    _bucket_of_scheduled_rows(out, 6)


def test_kimi_records_prefill_token_graphs_without_dp_and_a_quantized_query():
    """Under DP every rank pads to a prefill bucket some rank chose, so a
    decoding or idle rank would replay a recording whose regions serve a
    prefill alone. A quantized MLA query's scale does not cut by rows."""
    pytest.importorskip("aiter", reason="the Kimi builder imports the AITER runtime")
    from atom.model_ops.attentions.kimi_mla_gdn_attn import _KimiMLAGDNCommon

    def supports(dp=1, quantized_q=None):
        parallel = SimpleNamespace(data_parallel_size=dp)
        runner = SimpleNamespace(config=SimpleNamespace(parallel_config=parallel))
        if quantized_q is not None:
            attn = torch.nn.Module()
            attn.fuse_qknorm_quant = quantized_q
            runner.model = torch.nn.Sequential(torch.nn.Sequential(attn))
        builder = SimpleNamespace(model_runner=runner)
        return _KimiMLAGDNCommon.supports_token_graph.fget(builder)

    assert supports()
    assert supports(quantized_q=False)
    assert not supports(quantized_q=True)
    assert not supports(dp=2)
    assert not supports(dp=8, quantized_q=False)
    assert not _KimiMLAGDNCommon.supports_token_graph.fget(
        SimpleNamespace(model_runner=SimpleNamespace())
    )
