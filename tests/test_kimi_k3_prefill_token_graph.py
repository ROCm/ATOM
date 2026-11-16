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


def _slots():
    """A 5-row prefill's KV slots, padded to an 8-row bucket as the builders
    pad them."""
    return SimpleNamespace(slot_mapping=torch.tensor([7, 8, 9, 10, 11, -1, -1, -1]))


def _slots_seen():
    return forward_context.get_forward_context().attn_metadata.slot_mapping.shape[0]


def test_a_cut_region_sees_its_rows_and_hands_back_the_bucket(monkeypatch):
    seen = []

    def region(x, weight, name):
        seen.append((x.shape[0], weight.shape[0], name, _slots_seen()))
        return x * 2

    ctx = SimpleNamespace(attn_metadata=_slots())
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    x = torch.arange(1.0, 9.0).view(8, 1)
    weight = torch.ones(3)
    out = on_scheduled_rows(region, 5, x, weight, "layer")
    # Only a bucket-high argument is cut; the drafter reads the slots at the
    # bucket after the forward.
    assert seen == [(5, 3, "layer", 5)]
    assert _slots_seen() == 8
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
        seen["slots"] = _slots_seen()
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
    ctx = SimpleNamespace(
        context=_step(5), attn_metadata=_slots(), main_stream=None, in_hipgraph=False
    )
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
    # Under DCP the MLA cache writer keeps a padded slot's query row, so a
    # bucket of slots would write past the cut query.
    assert seen == {"mla": (5, 5, 5), "positions": 5, "slots": 5}
    assert _slots_seen() == 8
    _bucket_of_scheduled_rows(out, 4)


def test_the_dsa_indexer_writes_a_padded_prefills_scheduled_slots(
    padded_layer, monkeypatch
):
    """OPUS's cache write aborts the process on more slots than K rows."""
    from atom.model_ops.indexer import indexer_forward

    class Written(Exception):
        pass

    def write(q, q_out, weights, weights_out, k, kv_cache, slot_mapping, *_, **__):
        raise Written(q.shape[0], k.shape[0], slot_mapping.shape[0])

    indexer = SimpleNamespace(_indexer_fp4=True, _index_cache_layout="mfma", rope_dim=0)
    config = SimpleNamespace(
        kv_cache_block_size=64,
        dcp_config=SimpleNamespace(interleave_size=1),
        compilation_config=SimpleNamespace(static_forward_context={"layer": indexer}),
    )
    monkeypatch.setattr(indexer_forward, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(indexer_forward, "indexer_qk_rope_quant_and_cache", write)
    monkeypatch.setattr(indexer_forward, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(indexer_forward, "pcp_is_enabled", lambda: False)
    forward_context.get_forward_context().context.is_dummy_run = False

    hidden, q, k, weights = (
        torch.ones(8, *s) for s in ((64,), (32, 128), (128,), (32,))
    )
    none = torch.empty(0)
    with pytest.raises(Written) as written:
        indexer_forward.sparse_attn_indexer(
            hidden, "layer.k_cache", none, q, k, weights, 128, "ue8m0", 64, 128,
            1024, 1024, none, none, none, none, none, 1e-6,
            torch.zeros(8, dtype=torch.int64), none, none, 1.0, False, True, False,
            kv_cache_scale=none,
        )  # fmt: skip
    assert written.value.args == (5, 5, 5)
    assert _slots_seen() == 8


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
