# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The indexer region reads live rows and slots on every token-graph replay."""

import ast
import functools
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _definitions(path, names, namespace):
    # Execute the production bodies without importing the GPU-only kernels.
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    tree.body = [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *body]
    exec(  # noqa: S102 - the repository's production definitions, not external input
        compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace
    )


@pytest.fixture
def indexer():
    slots = torch.tensor([7, 2, 9, 3, 11, -1, -1, -1])
    context = {
        "__name__": __name__,
        "torch": torch,
        "dataclass": dataclass,
        "field": field,
        "contextmanager": contextmanager,
        "_forward_context_local": threading.local(),
    }
    _definitions(
        "atom/utils/forward_context.py",
        {"ForwardContext", "get_forward_context", "side_stream"},
        context,
    )
    ctx = context["ForwardContext"](
        context=SimpleNamespace(
            is_prefill=True,
            is_dummy_run=False,
            scheduled_tokens=5,
            forward_mode=SimpleNamespace(prefill_token_graph=True),
        ),
        attn_metadata=SimpleNamespace(slot_mapping=slots, run_sparse=False),
    )
    context["_forward_context"] = ctx
    graph = {
        "torch": torch,
        "functools": functools,
        "_local": threading.local(),
        "forward_context": SimpleNamespace(side_stream=context["side_stream"]),
    }
    _definitions(
        "atom/utils/region_graph.py", {"_recorder", "eager_region", "_EagerStep"}, graph
    )
    owner = SimpleNamespace(_indexer_fp4=True, _index_cache_layout="mfma", rope_dim=0)
    config = SimpleNamespace(
        kv_cache_block_size=64,
        dcp_config=SimpleNamespace(interleave_size=1),
        compilation_config=SimpleNamespace(static_forward_context={"layer": owner}),
    )
    writes = []

    def write(q, q_out, weights, weights_out, k, cache, slots, *args, **kwargs):
        positions = args[2]
        assert slots.numel() <= k.shape[0], "cache writer received more slots than keys"
        assert q.shape[0] == k.shape[0] == weights.shape[0]
        assert positions is None or positions.shape[0] == k.shape[0]
        live = slots >= 0
        cache[slots[live]] = k[: slots.numel()][live, 0]
        q_out.copy_(q[..., : q_out.shape[-1]].to(q_out.dtype))
        weights_out.copy_(weights)
        writes.append((q.shape[0], slots.clone()))

    namespace = {
        "torch": torch,
        "eager_region": graph["eager_region"],
        "get_forward_context": context["get_forward_context"],
        "get_current_atom_config": lambda: config,
        "get_dcp_world_size": lambda: 1,
        "pcp_is_enabled": lambda: False,
        "MFMA": "mfma",
        "FP4_QUANT_BLOCK_SIZE": 32,
        "fp4_q_scale_shape": lambda rows, heads, dim, **_: (rows, heads, dim // 32),
        "indexer_qk_rope_quant_and_cache": write,
    }
    _definitions(
        "atom/model_ops/indexer/indexer_forward.py", {"sparse_attn_indexer"}, namespace
    )
    empty = torch.empty(0)
    values = torch.arange(1, 9, dtype=torch.float32)
    args = {
        "hidden_states": values[:, None].expand(-1, 64),
        "k_cache_prefix": "layer.k_cache",
        "kv_cache": torch.full((16,), -101.0),
        "q_input": values[:, None, None].expand(-1, 2, 128),
        "k": values[:, None].expand(-1, 128),
        "weights": values[:, None].expand(-1, 2),
        "quant_block_size": 128,
        "scale_fmt": "ue8m0",
        "topk_tokens": 4,
        "head_dim": 128,
        "max_model_len": 1024,
        "total_seq_lens": 1024,
        "sparse_kv_indices_buffer": empty,
        "dcp_sparse_kv_indptr_buffer": empty,
        "dcp_owned_counts_buffer": empty,
        "k_norm_weight": empty,
        "k_norm_bias": empty,
        "k_norm_eps": 1e-6,
        "positions": torch.arange(8),
        "cos_cache": empty,
        "sin_cache": empty,
        "weights_scale": 1.0,
        "is_neox_style": False,
        "use_qk_rope_cache_fusion": True,
        "stable_topk": False,
        "kv_cache_scale": empty,
    }
    return SimpleNamespace(
        call=namespace["sparse_attn_indexer"],
        graph=graph,
        ctx=ctx,
        args=args,
        writes=writes,
        namespace=namespace,
    )


def test_indexer_cuts_only_live_token_graph_rows_and_keeps_metadata(indexer):
    slots = indexer.ctx.attn_metadata.slot_mapping
    indexer.call(**indexer.args)
    assert indexer.writes[0][0] == 5
    torch.testing.assert_close(indexer.writes[0][1], slots[:5])
    assert indexer.ctx.attn_metadata.slot_mapping is slots
    assert slots.shape == (8,)
    torch.testing.assert_close(
        indexer.args["kv_cache"][slots[:5]], torch.arange(1.0, 6.0)
    )
    assert indexer.args["kv_cache"][0].item() == -101


@pytest.mark.parametrize("case", ["eager", "decode", "tbo", "full"])
def test_other_padding_keeps_existing_indexer_rows(indexer, case):
    context = indexer.ctx.context
    if case == "eager":
        context.forward_mode.prefill_token_graph = False
    elif case == "decode":
        context.is_prefill = False
    elif case == "tbo":
        context.forward_mode = None
    else:
        context.scheduled_tokens = 8
    writer = indexer.namespace["indexer_qk_rope_quant_and_cache"]

    def stop_after_write(*args, **kwargs):
        writer(*args, **kwargs)
        raise RuntimeError("cache written")

    indexer.namespace["indexer_qk_rope_quant_and_cache"] = stop_after_write
    with pytest.raises(RuntimeError, match="cache written"):
        indexer.call(**indexer.args)
    assert indexer.writes[0][0] == 8
    assert indexer.writes[0][1].shape == (8,)


def test_dummy_capture_touches_neither_cache_nor_slot_metadata(indexer):
    indexer.ctx.context.is_dummy_run = True
    indexer.ctx.attn_metadata = None
    before = indexer.args["kv_cache"].clone()
    indexer.call(**indexer.args)
    assert not indexer.writes
    torch.testing.assert_close(indexer.args["kv_cache"], before)


def test_dummy_warmup_uses_production_stream_context(indexer):
    context = {
        "contextmanager": contextmanager,
        "_forward_context": indexer.ctx,
        "_forward_context_local": threading.local(),
    }
    _definitions(
        "atom/utils/forward_context.py", {"get_forward_context", "side_stream"}, context
    )
    indexer.graph["forward_context"] = SimpleNamespace(
        side_stream=context["side_stream"]
    )
    indexer.ctx.in_hipgraph = True
    indexer.ctx.context.is_dummy_run = True
    before = indexer.args["kv_cache"].clone()
    indexer.call(**indexer.args)
    assert not indexer.writes
    torch.testing.assert_close(indexer.args["kv_cache"], before)


def test_recorded_indexer_reads_the_live_step_after_dummy_capture(indexer):
    steps = []

    class Recorder:
        def eager(self, fn, args, kwargs, side, live):
            fn(*args, **kwargs)
            steps.append(indexer.graph["_EagerStep"](fn, kwargs, live, side, None))

    indexer.ctx.context.is_dummy_run = True
    local = indexer.graph["_local"]
    local.recorder = Recorder()
    try:
        indexer.call(**indexer.args)
    finally:
        local.recorder = None
    assert len(steps) == 1
    assert not indexer.writes
    indexer.ctx.context.is_dummy_run = False
    # The second replay replaces metadata and shrinks the live prefix. Stale
    # captured slots or rows would overwrite the protected cache entries.
    for rows, live_slots in ((5, [1, 2, 3, 4, 5]), (2, [12, 13])):
        indexer.ctx.context.scheduled_tokens = rows
        slots = torch.tensor(live_slots + [-1] * (8 - rows))
        indexer.ctx.attn_metadata = SimpleNamespace(
            slot_mapping=slots, run_sparse=False
        )
        indexer.args["kv_cache"].fill_(-101)
        steps[0]()
        expected = torch.full((16,), -101.0)
        expected[live_slots] = torch.arange(1.0, rows + 1)
        torch.testing.assert_close(indexer.args["kv_cache"], expected)
        assert indexer.ctx.attn_metadata.slot_mapping is slots
        assert indexer.writes[-1][0] == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph replay")
@torch.inference_mode()
def test_real_capture_replays_indexer_cache_writes(indexer, monkeypatch):
    from atom.model_ops.indexer import indexer_forward
    from atom.utils import forward_context, region_graph

    for name, value in indexer.namespace.items():
        if name not in {"sparse_attn_indexer", "eager_region", "__builtins__"}:
            monkeypatch.setattr(indexer_forward, name, value, raising=False)
    indexer.ctx = forward_context.ForwardContext(
        context=indexer.ctx.context, attn_metadata=indexer.ctx.attn_metadata
    )
    monkeypatch.setattr(forward_context, "_forward_context", indexer.ctx)
    monkeypatch.setattr(
        forward_context._forward_context_local, "ctx", None, raising=False
    )
    monkeypatch.setattr(
        indexer_forward, "get_forward_context", forward_context.get_forward_context
    )
    indexer.args = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in indexer.args.items()
    }

    # The stand-in writer performs tensor work on the device; the production
    # region and recorder determine which work runs again on a live replay.
    def forward():
        indexer_forward.sparse_attn_indexer(**indexer.args)
        return indexer.args["kv_cache"] * 2

    indexer.ctx.context.is_dummy_run = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        indexer.ctx.main_stream = stream
        indexer.ctx.in_hipgraph = True
        indexer.ctx.tr_graph = "record"
        forward()
        segments, _ = region_graph._record_segments(forward)
    torch.cuda.current_stream().wait_stream(stream)
    indexer.ctx.main_stream = torch.cuda.current_stream()
    indexer.ctx.in_hipgraph = False
    indexer.ctx.tr_graph = "replay"
    indexer.ctx.context.is_dummy_run = False
    for rows, live_slots in ((5, [1, 2, 3, 4, 5]), (2, [12, 13])):
        indexer.ctx.context.scheduled_tokens = rows
        indexer.ctx.attn_metadata.slot_mapping = torch.tensor(
            live_slots + [-1] * (8 - rows), device="cuda"
        )
        indexer.args["kv_cache"].fill_(-101)
        actual = segments.replay().clone()
        indexer.args["kv_cache"].fill_(-101)
        expected = forward()
        torch.testing.assert_close(actual, expected)
        assert indexer.writes[-1][0] == rows


def test_writer_failure_does_not_replace_metadata_slots(indexer):
    slots = indexer.ctx.attn_metadata.slot_mapping

    def fail(*args, **kwargs):
        raise RuntimeError("writer failed")

    indexer.namespace["indexer_qk_rope_quant_and_cache"] = fail
    with pytest.raises(RuntimeError, match="writer failed"):
        indexer.call(**indexer.args)
    assert indexer.ctx.attn_metadata.slot_mapping is slots
