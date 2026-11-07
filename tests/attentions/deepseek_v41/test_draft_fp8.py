# SPDX-License-Identifier: MIT
"""True OPUS A8W8 draft attention and raw MXFP8 window contracts."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("A8W8 draft kernels require a GPU", allow_module_level=True)
pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.deepseek_v41.draft_block import DraftStep, rotate_rows
from atom.model_ops.deepseek_v41.dspark import (
    draft_attention_csr,
    draft_attention_quantized,
    draft_step_indices,
)
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from tests.attentions.deepseek_v41.test_a8w4_attention import mxfp8
from tests.attentions.deepseek_v41.test_draft_fp4 import oracle


def make_cache(target_dtype="fp4", window=128):
    geometry = V41PoolGeometry(
        3,
        ((0, 2),),
        32,
        window,
        512,
        128,
        kv_dtype=target_dtype,
        speculative_tokens=5,
    )
    cache = PagedAttentionCache(geometry, 32, 4, "cuda")
    for layer in range(3):
        data, scale = cache.ring_view(layer)
        values = quantize_fp8(
            torch.randn(data.shape[0], 512, device="cuda", dtype=torch.bfloat16)
        )
        for dst, src in zip((data, scale), values):
            dst.copy_(src.view(torch.uint8))
    return cache


def test_runtime_setting_reaches_draft_constructor(monkeypatch):
    from torch import nn

    from atom.config import DSparkConfig
    from atom.models.deepseek_v41 import dspark

    args = SimpleNamespace(
        num_nextn_predict_layers=1,
        dspark_block_size=5,
        sliding_window=128,
        vocab_size=128,
        dspark_n_routed_experts=2,
        dspark_num_experts_per_tok=1,
        num_hidden_layers=0,
        max_position_embeddings=512,
        qk_rope_head_dim=64,
        rope_theta=10000,
    )
    config = SimpleNamespace(hf_config=args, dspark=DSparkConfig(draft_kv_dtype="fp8"))
    monkeypatch.setattr(dspark, "make_v4_quant_config", lambda *a, **kw: None)
    monkeypatch.setattr(
        dspark, "build_attention_topology", lambda *a: [SimpleNamespace(layer_id=40)]
    )
    monkeypatch.setattr(dspark, "DraftBlock", lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(dspark, "RotaryEmbedding", lambda *a, **kw: nn.Identity())
    draft = dspark.DeepseekV41DSpark(config)
    assert draft.draft_kv_dtype == "fp8"


@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("width", [1, 5])
@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_a8w8_draft_matches_dense_quantized_oracle(heads, width, target_dtype):
    torch.manual_seed(6801)
    cache = make_cache(target_dtype)
    slots = torch.tensor([3, 0, 2], device="cuda", dtype=torch.int32)
    context = tuple(
        p[0]
        for p in cache.read_window_planes(
            torch.tensor([2], device="cuda"), slots, fp4=False
        )
    )
    anchors = torch.tensor([-1, 3, 268], device="cuda")
    positions, indices, _ = draft_step_indices(anchors, 133, 128, width)
    query = torch.randn(3, width, heads, 512, device="cuda", dtype=torch.bfloat16)
    current = quantize_fp8(
        torch.randn(3, width, 512, device="cuda", dtype=torch.bfloat16)
    )
    q8 = quantize_fp8(query)
    sink = torch.linspace(-1, 1, heads, device="cuda")
    step = DraftStep(positions, indices, csr=draft_attention_csr(indices, 133))
    original = [p.clone() for p in cache.window_sources]
    actual = draft_attention_quantized(
        query, context, current, sink, step, 512**-0.5, query_fp8=q8, fp4=False
    )
    expected = oracle(mxfp8(q8), mxfp8(context), mxfp8(current), anchors, 128, sink)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.006
    assert all(torch.equal(a, b) for a, b in zip(original, cache.window_sources))


def test_a8w8_gather_keeps_exact_bytes_for_all_stages_and_reordered_slots():
    cache = make_cache()
    for layer_ids, slot_ids in [([2, 0, 1], [3, 1]), ([1, 2], [0, 2, 3])]:
        layers = torch.tensor(layer_ids, device="cuda")
        slots = torch.tensor(slot_ids, device="cuda", dtype=torch.int32)
        raw = cache.read_window_planes(layers, slots, fp4=False)
        assert raw[0].shape == (len(layer_ids), len(slot_ids), 133, 512)
        assert raw[1].shape == (len(layer_ids), len(slot_ids), 133, 16)
        assert (
            raw[0].untyped_storage().data_ptr() == raw[1].untyped_storage().data_ptr()
        )
        for i, layer in enumerate(layer_ids):
            for p, expected in zip(raw, cache.ring_view(layer)):
                assert torch.equal(p[i], expected.view(4, 133, -1)[slots.long()])


def test_a8w8_real_draft_forward_uses_raw_history_and_fp8_query_after_rope():
    from atom.models.deepseek_v41.dspark import DraftAttention

    torch.manual_seed(6802)
    cache = make_cache()
    history = tuple(
        p[0]
        for p in cache.read_window_planes(
            torch.tensor([2], device="cuda"),
            torch.tensor([0, 3], device="cuda"),
            fp4=False,
        )
    )
    rope = RotaryEmbedding(64, 512, base=10000).cuda()
    q = torch.randn(2, 5, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    anchors = torch.tensor([3, 260], device="cuda")
    positions, indices, _ = draft_step_indices(anchors, 133, 128, 5)
    sink = torch.linspace(-1, 1, 64, device="cuda")
    instance = SimpleNamespace(
        heads=64,
        head_dim=512,
        spec=SimpleNamespace(layer_id=2),
        project_qkv=lambda *a: (None, kv),
        qk_norm=lambda *a: (None, None, kv),
        wq_b=lambda *a, **kw: q.clone().flatten(-2),
        attn_sink=sink,
        softmax_scale=512**-0.5,
        _project_out=lambda out, *a: out,
    )
    actual = DraftAttention.forward(
        instance, None, None, {2: history}, DraftStep(positions, indices), rope
    )
    q8 = quantize_fp8(rotate_rows(rope, q.clone(), positions))
    k8 = quantize_fp8(rotate_rows(rope, kv.clone(), positions))
    expected = oracle(mxfp8(q8), mxfp8(history), mxfp8(k8), anchors, 128, sink)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.03)


def test_native_a8w8_draft_graph_follows_live_requests_and_anchors(monkeypatch):
    from atom.spec_decode.draft_graph import DraftGraph, StagedInput

    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "1")
    cache = make_cache()
    layers = torch.tensor([2], device="cuda")
    q = torch.randn(4, 5, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4, 5, 512, device="cuda", dtype=torch.bfloat16)
    q8, k8 = quantize_fp8(q), quantize_fp8(kv)
    sink = torch.linspace(-1, 1, 64, device="cuda")

    def forward(bs, slots, anchors):
        history = tuple(
            p[0] for p in cache.read_window_planes(layers, slots, fp4=False)
        )
        positions, indices, _ = draft_step_indices(anchors, 133, 128, 5)
        step = DraftStep(positions, indices, csr=draft_attention_csr(indices, 133))
        return draft_attention_quantized(
            q, history, k8, sink, step, 512**-0.5, query_fp8=q8, fp4=False
        )

    graph = DraftGraph(
        forward=forward,
        capture_epilogue=True,
        inputs={
            "slots": StagedInput(),
            "anchors": StagedInput(dtype=torch.int64),
        },
    ).bind(SimpleNamespace(max_num_seqs=4), "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.warmup(4)
    torch.cuda.current_stream().wait_stream(stream)
    for slots, anchors in [
        ([3, 0, 2], [268, 3, -1]),
        ([1], [0]),
        ([2, 3, 0, 1], [130, -1, 400, 1]),
    ]:
        live = {
            "slots": torch.tensor(slots, device="cuda", dtype=torch.int32),
            "anchors": torch.tensor(anchors, device="cuda", dtype=torch.int64),
        }
        staged = graph.stage(4, live)
        actual = graph.run(4, **staged)
        history = tuple(
            p[0] for p in cache.read_window_planes(layers, staged["slots"], fp4=False)
        )
        expected = oracle(
            mxfp8(q8), mxfp8(history), mxfp8(k8), staged["anchors"], 128, sink
        )
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
    assert graph.is_captured(4)
