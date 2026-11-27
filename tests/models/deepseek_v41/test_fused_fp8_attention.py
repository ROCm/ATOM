# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Fused FP8 attention through ATOM's cache and output projections."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

if os.environ.get("ATOM_REQUIRE_FUSED_ATTENTION_TESTS") == "1":
    __import__("aiter")
else:
    pytest.importorskip("aiter")

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.fp8_attention import (
    fused_attention_available,
)
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.linear import LinearBase
from atom.models.deepseek_v41.attention import Attention
from atom.models.deepseek_v41.bounded_replay import QueryRows
from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
from atom.utils import forward_context
from atom.utils.region_graph import ReqGraph, TokenGraph
from tests.attentions.deepseek_v41.helpers import (
    PagedRequest,
    begin_step,
    metadata_buffers,
)


def window_bytes(cache):
    return torch.cat(list(cache.window_sources))


def restore_windows(cache, snapshot):
    offset = 0
    for source in cache.window_sources:
        source.copy_(snapshot[offset : offset + source.numel()])
        offset += source.numel()


@pytest.fixture
def fused_factory(single_rank):
    available = torch.cuda.is_available() and fused_attention_available()
    if os.environ.get("ATOM_REQUIRE_FUSED_ATTENTION_TESTS") == "1":
        assert available, "Required fused V4.1 attention is unavailable"
    elif not available:
        pytest.skip("Fused V4.1 attention requires gfx950 and the AITER fused op")

    def build(
        heads=64,
        ratio=1,
        max_tokens=32,
        slots=2,
        topk=512,
        hidden_size=256,
        q_lora_rank=256,
        o_lora_rank=128,
        norm_eps=1e-6,
        prefix_context=None,
        speculative_tokens=0,
    ):
        torch.manual_seed(6821 + heads + ratio)
        config = SimpleNamespace(
            fp4ptpc=False,
            head_dim=512,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            num_attention_heads=heads,
            o_groups=heads // 8,
            o_lora_rank=o_lora_rank,
            rms_norm_eps=norm_eps,
        )
        spec = LayerAttentionSpec(
            1,
            ratio,
            AttentionMode.REUSE if ratio else AttentionMode.WINDOW,
            0 if ratio else None,
            0 if ratio else None,
        )
        model = Attention(config, spec).cuda().eval()
        assert model.fused_attn
        with torch.no_grad():
            model.q_norm.weight.fill_(1)
            model.kv_norm.weight.fill_(1)
            model.attn_sink.normal_(std=0.2)
            for module in model.modules():
                if isinstance(module, LinearBase):
                    module.weight.copy_(
                        (torch.randn(module.weight.shape, device="cuda") * 16).to(
                            module.weight.dtype
                        )
                    )
                    module.weight_scale.copy_(
                        torch.exp2(
                            torch.randint(
                                -10, -7, module.weight_scale.shape, device="cuda"
                            ).float()
                        ).to(module.weight_scale.dtype)
                    )
                    if module.bias is not None:
                        module.bias.zero_()
                    module.process_weights_after_loading()
        context = max(640, topk + 128, prefix_context or 0)
        pages = max(32, (context + max_tokens + 127) // 128)
        geo = V41PoolGeometry(
            2,
            ((0, 1),),
            128,
            128,
            512,
            32,
            kv_dtype="fp8",
            layer_ratios=(0, 1),
            index_topk=topk,
            speculative_tokens=speculative_tokens,
        )
        cache = PagedAttentionCache(geo, pages, slots, "cuda", max_tokens=max_tokens)
        for values, scales in (
            cache.main_view(0),
            cache.ring_view(0),
            cache.ring_view(1),
        ):
            source = torch.randn(values.shape, device="cuda").bfloat16()
            quantized = quantize_fp8(source)
            values.copy_(quantized[0].view(torch.uint8))
            scales.copy_(quantized[1].view(torch.uint8))
        rope = RotaryEmbedding(64, context + max_tokens + 256, base=10000).cuda()
        return SimpleNamespace(
            model=model,
            cache=cache,
            rope=rope,
            topk=topk,
            context=context,
            initial=window_bytes(cache),
            config=config,
            selection=torch.arange(topk, device="cuda", dtype=torch.int32)
            .expand(1, max_tokens, -1)
            .contiguous(),
            slot_ids=torch.zeros(slots, device="cuda", dtype=torch.int32),
        )

    return build


def make_step(
    case,
    lengths,
    *,
    position=None,
    prefill=True,
    tentative=False,
    running_tokens=None,
    buffers=None,
    bounded=False,
    slot_order=None,
):
    position = case.context if position is None else position
    offset = 0
    requests = []
    pages = tuple(range(case.cache.num_pages))
    for request, length in enumerate(lengths):
        slot = request if slot_order is None else slot_order[request]
        requests.append(
            PagedRequest(request + 1, position, offset, length, slot, pages)
        )
        offset += length
    width = offset if running_tokens is None else running_tokens
    if buffers is None:
        buffers = metadata_buffers(
            case.cache.num_slots,
            width,
            case.cache.num_pages,
            "cuda",
            case.cache.geometry,
        )
    case.slot_ids.copy_(
        torch.tensor(
            [request.slot for request in requests]
            + [0] * (case.cache.num_slots - len(requests)),
            device="cuda",
            dtype=torch.int32,
        )
    )
    step = begin_step(
        case.cache,
        requests,
        is_prefill=prefill,
        tentative=tentative,
        running_bs=case.cache.num_slots,
        running_tokens=width,
        buffers=buffers,
        state_slot_out=case.slot_ids,
    )
    if case.model.spec.ratio:
        step.selected[0] = case.selection[:, : step.running_tokens]
    queries = None
    if bounded:
        rows = torch.tensor(
            [sum(lengths[: i + 1]) - 1 for i in range(len(lengths))],
            device="cuda",
            dtype=torch.int64,
        )
        ids = torch.full_like(step.batch_ids, -1)
        ids[rows] = step.batch_ids[rows]
        queries = QueryRows(
            rows,
            step.positions[rows],
            replace(step, batch_ids=ids, group_indices={}),
            torch.cat((rows, rows.new_tensor([step.running_tokens]))),
        )
    return step, queries


def error_metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    assert actual.shape == expected.shape and torch.isfinite(actual).all()
    delta = actual - expected
    row_relative = delta.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-20)
    return {
        "relative": (delta.norm() / expected.norm().clamp_min(1e-20)).item(),
        "max_abs": delta.abs().max().item(),
        "max_row_relative": row_relative.max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
    }


def assert_output(actual, expected):
    metrics = error_metrics(actual, expected)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert metrics["relative"] < 0.03, metrics
    assert metrics["cosine"] > 0.999, metrics
    assert metrics["max_row_relative"] < 0.05, metrics
    return metrics


@pytest.mark.parametrize("supported", [False, True])
def test_fused_support_query_is_owned_by_aiter(single_rank, monkeypatch, supported):
    from atom.models.deepseek_v41 import attention

    calls = []

    def supports(*args, **kwargs):
        calls.append((args, kwargs))
        return supported

    monkeypatch.setattr(attention, "fused_attention_available", supports)
    monkeypatch.setattr("atom.model_ops.linear.native_group32_runs_bf16", lambda: False)
    config = SimpleNamespace(
        fp4ptpc=False,
        head_dim=512,
        hidden_size=256,
        q_lora_rank=1280,
        num_attention_heads=64,
        o_groups=8,
        o_lora_rank=128,
        rms_norm_eps=1e-6,
    )
    spec = LayerAttentionSpec(1, 0, AttentionMode.WINDOW, None, None)
    with torch.device("meta"):
        model = Attention(config, spec)
        activation = torch.empty(1, 2, 256, dtype=torch.float16)
        rope = SimpleNamespace(
            cos_cache=torch.empty(16, 48, dtype=torch.float32),
            sin_cache=torch.empty(16, 48, dtype=torch.float16),
            rope_dim=96,
        )
    assert model.fused_attn is supported
    assert calls == [((64, 512), {})]
    assert all(p.device.type == "meta" for p in model.parameters())
    # A changed AITER answer must control dispatch, including future support
    # for RoPE cache dtypes/rotary widths ATOM does not currently use.
    model.fused_attn = True
    cache = SimpleNamespace(kv_fp8=True, packed=False)
    assert model._use_fused_attention(activation, cache, rope) is supported
    assert calls[-1] == (
        (64, 512),
        {
            "q_dtype": torch.bfloat16,
            "cos_dtype": torch.float32,
            "sin_dtype": torch.float16,
            "rope_dim": 96,
            "device": activation.device,
        },
    )
    count = len(calls)
    for _ in range(40):
        assert model._use_fused_attention(activation, cache, rope) is supported
    assert len(calls) == count, "Both supported and unsupported answers are cached"
    for kv_fp8, packed in ((False, False), (True, True)):
        cache.kv_fp8, cache.packed = kv_fp8, packed
        count = len(calls)
        assert not model._use_fused_attention(activation, cache, rope)
        assert len(calls) == count
    cache.kv_fp8, cache.packed = True, False
    model.fused_attn = False
    assert not model._use_fused_attention(activation, cache, rope)
    assert len(calls) == count


@pytest.mark.parametrize("corruption", ["zero", "sign", "nan"])
def test_output_check_rejects_one_bad_row(corruption):
    expected = torch.ones(1, 4096, 256, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[:, 37] = {"zero": 0, "sign": -1, "nan": float("nan")}[corruption]
    if corruption != "nan":
        metrics = error_metrics(actual, expected)
        assert metrics["cosine"] > 0.999
        if corruption == "zero":
            assert metrics["relative"] < 0.03
    with pytest.raises(AssertionError):
        assert_output(actual, expected)


@pytest.mark.parametrize(
    "fp4ptpc,wq_b_bf16,wo_a_bf16,eligible",
    [
        (False, False, False, True),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
    ],
)
def test_fused_attention_requires_mxfp8_projections(
    single_rank, monkeypatch, fp4ptpc, wq_b_bf16, wo_a_bf16, eligible
):
    from atom.model_ops import linear
    from atom.models.deepseek_v41 import attention

    # Exercise the real constructors and their quantization selection. These
    # are the native linears in construction order: wqkv_a, wq_b, wo_a, wo_b.
    bf16_answers = iter((False, wq_b_bf16, wo_a_bf16, False))
    monkeypatch.setattr(
        linear,
        "native_group32_runs_bf16",
        (lambda: wo_a_bf16) if fp4ptpc else (lambda: next(bf16_answers)),
    )
    supports = []
    monkeypatch.setattr(
        attention,
        "fused_attention_available",
        lambda *args: supports.append(args) or True,
    )
    config = SimpleNamespace(
        fp4ptpc=fp4ptpc,
        head_dim=512,
        hidden_size=256,
        q_lora_rank=1280,
        num_attention_heads=64,
        o_groups=8,
        o_lora_rank=128,
        rms_norm_eps=1e-6,
    )
    spec = LayerAttentionSpec(1, 0, AttentionMode.WINDOW, None, None)
    with torch.device("meta"):
        model = Attention(config, spec)
    assert model.fused_attn is eligible
    assert model.wo_a.runs_bf16 is wo_a_bf16
    norm_type = (
        attention.DualRMSNormPerToken
        if fp4ptpc
        else attention.DualRMSNormBF16 if wq_b_bf16 else attention.DualRMSNormMXFP8
    )
    assert isinstance(model.qk_norm, norm_type)
    assert supports == ([(64, 512)] if eligible else [])


@torch.inference_mode()
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("tokens", [128, 4096])
def test_fused_full_prefill(fused_factory, monkeypatch, heads, tokens, record_property):
    from atom.models.deepseek_v41 import attention

    original = attention.fused_sparse_attn_v41_paged_2src
    calls = []

    def tracked(query, *args, **kwargs):
        calls.append(query.shape[0])
        return original(query, *args, **kwargs)

    monkeypatch.setattr(attention, "fused_sparse_attn_v41_paged_2src", tracked)
    case = fused_factory(heads, max_tokens=tokens, q_lora_rank=1280)
    step, _ = make_step(case, (tokens,), prefill=True)
    hidden = torch.randn(1, tokens, 256, device="cuda", dtype=torch.bfloat16)
    outputs, rings = [], []
    for fused in (False, True):
        case.model.fused_attn = fused
        restore_windows(case.cache, case.initial)
        outputs.append(case.model(hidden, None, case.cache, step, case.rope))
        rings.append(window_bytes(case.cache))
    assert calls == [tokens], "The full forward must invoke fused attention"
    record_property("fused", assert_output(outputs[1], outputs[0]))
    assert torch.equal(rings[1], rings[0])
    assert not torch.equal(rings[0], case.initial)


@torch.inference_mode()
@pytest.mark.parametrize("ratio", [0, 1])
def test_fused_verify_is_causal(fused_factory, monkeypatch, ratio, record_property):
    from atom.models.deepseek_v41 import attention

    original = attention.fused_sparse_attn_v41_paged_2src
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(attention, "fused_sparse_attn_v41_paged_2src", tracked)
    case = fused_factory(64, ratio=ratio, q_lora_rank=1280, speculative_tokens=4)
    # Both requests cross the ring boundary, and their slots are permuted.
    ring = case.cache.geometry.ring_slots
    position = case.context + (-case.context % ring) + ring - 2
    step, _ = make_step(
        case,
        (5, 3),
        prefill=False,
        tentative=True,
        position=position,
        slot_order=(1, 0),
    )
    assert step.tentative and step.decode
    hidden = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    changed = hidden.clone()
    changed[:, 4] = torch.randn_like(changed[:, 4]) * 4
    outputs, rings = {}, {}
    for fused in (False, True):
        case.model.fused_attn = fused
        for label, value in (("original", hidden), ("changed", changed)):
            restore_windows(case.cache, case.initial)
            outputs[fused, label] = case.model(value, None, case.cache, step, case.rope)
            rings[fused, label] = window_bytes(case.cache)
        before, after = outputs[fused, "original"], outputs[fused, "changed"]
        # No indices from the implementation enter this oracle: rows 0..3
        # precede the mutation; rows 5..7 belong to a different request.
        unaffected = [0, 1, 2, 3, 5, 6, 7]
        torch.testing.assert_close(
            before[:, unaffected], after[:, unaffected], rtol=0, atol=0
        )
        assert not torch.equal(before[:, 4], after[:, 4])
    assert len(calls) == 2
    for label in ("original", "changed"):
        record_property(
            label, assert_output(outputs[True, label], outputs[False, label])
        )
        assert torch.equal(rings[True, label], rings[False, label])


@torch.inference_mode()
@pytest.mark.parametrize("stage", ["decode", "prefill", "long_prefill"])
@pytest.mark.parametrize("q_lora_rank", [32, 64, 96, 1280])
def test_norm_rope_quant_window_matches_two_launches(fused_factory, stage, q_lora_rank):
    # At 800 tokens, long_prefill exercises full-group splits for 64/96 and
    # V4.1's 1280; 32 and the shorter decode/prefill cases stay unsplit.
    lengths = {"decode": (1, 1), "prefill": (3, 5), "long_prefill": (800,)}[stage]
    case = fused_factory(
        64, slots=2, max_tokens=sum(lengths), q_lora_rank=q_lora_rank, hidden_size=256
    )
    model, cache, rope = case.model, case.cache, case.rope
    step, _ = make_step(case, lengths, prefill=stage != "decode")
    latent = torch.randn(
        1, step.running_tokens, q_lora_rank + 512, device="cuda", dtype=torch.bfloat16
    )
    q, kv = latent[..., :q_lora_rank], latent[..., q_lora_rank:]
    layer = model.spec.layer_id

    restore_windows(cache, case.initial)
    qr, qr_scale, normed = model.qk_norm(q, kv)
    rows, window = cache.rope_quant_window(layer, None, normed, rope, step)
    ring = window_bytes(cache)
    if rows is not None:
        rows = tuple(part.clone() for part in rows)

    restore_windows(cache, case.initial)
    actual = cache.norm_rope_quant_window(layer, model.qk_norm, q, kv, rope, step)
    assert torch.equal(actual[0].view(torch.uint8), qr.view(torch.uint8))
    assert torch.equal(actual[1].view(torch.uint8), qr_scale.view(torch.uint8))
    assert torch.equal(window_bytes(cache), ring)
    if stage == "decode":
        assert not torch.equal(ring, case.initial)
        assert actual[2:] == (None, None) == (rows, window)
    else:
        assert all(torch.equal(a, b) for a, b in zip(actual[2], rows))


@torch.inference_mode()
@pytest.mark.parametrize("heads", [16, 32, 64, 128])
@pytest.mark.parametrize("stage", ["decode", "prefill", "continued", "bounded"])
def test_fused_forward_matches_unfused(fused_factory, heads, stage, record_property):
    case = fused_factory(heads)
    lengths = (1, 1) if stage == "decode" else (3, 5)
    step, queries = make_step(
        case,
        lengths,
        prefill=stage != "decode",
        position=0 if stage == "prefill" else None,
        bounded=stage == "bounded",
    )
    hidden = torch.randn(
        1, step.running_tokens, 256, device="cuda", dtype=torch.bfloat16
    )
    outputs, caches = {}, {}
    for mode in ("unfused", "fused"):
        case.model.fused_attn = mode == "fused"
        restore_windows(case.cache, case.initial)
        outputs[mode] = case.model(hidden, None, case.cache, step, case.rope, queries)
        caches[mode] = window_bytes(case.cache)
    assert outputs["unfused"].abs().max() > 0
    assert not torch.equal(caches["unfused"], case.initial)
    metrics = assert_output(outputs["fused"], outputs["unfused"])
    record_property("fused", metrics)
    assert torch.equal(caches["fused"], caches["unfused"])


@torch.inference_mode()
@pytest.mark.parametrize("stage", ["decode", "continued", "bounded"])
def test_window_layer_fused_forward(fused_factory, stage, record_property):
    case = fused_factory(64, ratio=0)
    lengths = (1, 1) if stage == "decode" else (3, 5)
    step, queries = make_step(
        case, lengths, prefill=stage != "decode", bounded=stage == "bounded"
    )
    assert not step.selected
    hidden = torch.randn(
        1, step.running_tokens, 256, device="cuda", dtype=torch.bfloat16
    )
    reference, ring = None, None
    for mode in ("unfused", "fused"):
        case.model.fused_attn = mode == "fused"
        restore_windows(case.cache, case.initial)
        output = case.model(hidden, None, case.cache, step, case.rope, queries)
        if mode == "unfused":
            reference, ring = output, window_bytes(case.cache)
        else:
            record_property(mode, assert_output(output, reference))
            assert torch.equal(window_bytes(case.cache), ring)


@torch.inference_mode()
@pytest.mark.parametrize("heads,prefill", [(64, False), (128, True)])
def test_fused_forward_with_2048_prefix_rows(
    fused_factory, heads, prefill, record_property
):
    case = fused_factory(heads, topk=2048)
    step, _ = make_step(case, (3, 5) if prefill else (1, 1), prefill=prefill)
    hidden = torch.randn(
        1, step.running_tokens, 256, device="cuda", dtype=torch.bfloat16
    )
    reference, ring = None, None
    for mode in ("unfused", "fused"):
        case.model.fused_attn = mode == "fused"
        restore_windows(case.cache, case.initial)
        output = case.model(hidden, None, case.cache, step, case.rope)
        if mode == "unfused":
            reference, ring = output, window_bytes(case.cache)
        else:
            record_property(mode, assert_output(output, reference))
            assert torch.equal(window_bytes(case.cache), ring)


@torch.inference_mode()
@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("prefill", [False, True])
def test_fused_forward_t32_small_heads(
    fused_factory, monkeypatch, heads, prefill, record_property
):
    from atom.models.deepseek_v41 import attention

    original = attention.fused_sparse_attn_v41_paged_2src
    calls = []

    def tracked(*args, **kwargs):
        calls.append(isinstance(args[0], tuple))
        return original(*args, **kwargs)

    monkeypatch.setattr(attention, "fused_sparse_attn_v41_paged_2src", tracked)
    case = fused_factory(heads, slots=2 if prefill else 32, topk=2048)
    step, _ = make_step(case, (13, 19) if prefill else (1,) * 32, prefill=prefill)
    hidden = torch.randn(1, 32, 256, device="cuda", dtype=torch.bfloat16)
    reference, ring = None, None
    for mode in ("unfused", "fused"):
        case.model.fused_attn = mode == "fused"
        restore_windows(case.cache, case.initial)
        output = case.model(hidden, None, case.cache, step, case.rope)
        if mode == "unfused":
            reference, ring = output, window_bytes(case.cache)
        else:
            record_property(mode, assert_output(output, reference))
            assert torch.equal(window_bytes(case.cache), ring)
    assert calls == [False]


@torch.inference_mode()
@pytest.mark.parametrize("kind", ["token", "request"])
@pytest.mark.parametrize("idle", [False, True])
def test_fused_forward_graph_replay(fused_factory, monkeypatch, kind, idle):
    case = fused_factory(64, slots=8)
    model, cache, rope = case.model, case.cache, case.rope
    assert model.fused_attn
    buffers = metadata_buffers(8, 8, cache.num_pages, "cuda", cache.geometry)
    lengths = () if idle else (3, 5)
    step, _ = make_step(case, lengths, buffers=buffers, running_tokens=8)
    held_steps = [step]
    ctx = SimpleNamespace(
        in_hipgraph=False,
        main_stream=torch.cuda.current_stream(),
        context=SimpleNamespace(is_prefill=True),
        attn_metadata=SimpleNamespace(cache=cache, step=step, queries=None),
    )
    monkeypatch.setattr(forward_context, "_forward_context", ctx)

    def forward(hidden):
        live = ctx.attn_metadata.step
        live.begin_forward()
        if model.spec.ratio:
            live.selected[0] = case.selection[:, : live.running_tokens]
        return model(hidden, None, cache, live, rope)

    hidden = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    graph = TokenGraph() if kind == "token" else ReqGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ctx.main_stream = stream
        for _ in range(3):
            restore_windows(cache, case.initial)
            forward(hidden)
        restore_windows(cache, case.initial)
        ctx.in_hipgraph = True
        graph.record(forward, (hidden,), {}, key="attention", phase="prefill")
        ctx.in_hipgraph = False
    torch.cuda.current_stream().wait_stream(stream)
    ctx.main_stream = torch.cuda.current_stream()
    try:
        phases = [(True, lengths), (True, lengths)]
        if kind == "token":
            phases += [
                (True, (8,)),
                (True, (3, 5)),
                (True, (4, 4)),
                (False, (1,) * 8),
                (True, ()),
            ]
        previous = None
        for i, (prefill, lengths) in enumerate(phases):
            step, _ = make_step(
                case,
                lengths,
                position=case.context + i * 16,
                prefill=prefill,
                running_tokens=8,
                buffers=buffers,
                slot_order=(1, 0) if len(lengths) == 2 and i % 2 else None,
            )
            held_steps.append(step)
            assert step.slots.data_ptr() == case.slot_ids.data_ptr()
            ctx.attn_metadata.step = step
            ctx.context.is_prefill = prefill
            fresh = torch.randn_like(hidden)
            restore_windows(cache, case.initial)
            actual = graph.replay(
                (fresh,), {}, key="attention", phase="prefill"
            ).clone()
            actual_cache = window_bytes(cache)
            restore_windows(cache, case.initial)
            expected = forward(fresh)
            if lengths:
                assert_output(actual, expected)
                if previous is not None:
                    assert not torch.equal(actual, previous)
                previous = actual
            else:
                assert torch.equal(actual, torch.zeros_like(actual))
                assert torch.equal(actual_cache, case.initial)
            assert torch.equal(actual_cache, window_bytes(cache))
    finally:
        graph.release_graphs()
        held_steps.clear()


@torch.inference_mode()
def test_fused_dummy_initializes_both_output_planes(fused_factory):
    case = fused_factory(64)
    step, _ = make_step(case, (), running_tokens=8)
    query = torch.full(
        (1, 8, 64, 512), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    out = torch.full((8, 8, 4096), 255, device="cuda", dtype=torch.uint8)
    scale = torch.full((8, 8, 128), 255, device="cuda", dtype=torch.uint8)
    result = case.model._attend(
        query,
        case.cache,
        step,
        None,
        rope=case.rope,
        fused=True,
        out=out.view(torch.float8_e4m3fn),
        out_scale=scale.view(torch.float8_e8m0fnu),
    )
    assert result is None
    assert out.count_nonzero() == 0 and torch.all(scale == 127)
    assert torch.equal(window_bytes(case.cache), case.initial)


@pytest.mark.parametrize("heads", [16, 32, 64, 128])
def test_fused_attention_conditions(fused_factory, heads):
    case = fused_factory(heads)
    model, cache, rope = case.model, case.cache, case.rope
    for rows in (1, 127, 128):
        activation = torch.empty(
            1, rows, case.config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        assert model._use_fused_attention(activation, cache, rope)
    # wq_b produces BF16 Q; the normed input only supplies the device here.
    assert model._use_fused_attention(activation.half(), cache, rope)
    for field in ("kv_fp8", "packed"):
        original = getattr(cache, field)
        setattr(cache, field, field == "packed")
        assert not model._use_fused_attention(activation, cache, rope)
        setattr(cache, field, original)
    model.fused_attn = False
    assert not model._use_fused_attention(activation, cache, rope)
    groups = SimpleNamespace(**{**vars(case.config), "o_groups": heads // 4})
    assert not Attention(groups, model.spec).fused_attn


@torch.inference_mode()
@pytest.mark.parametrize("stage", ["decode", "prefill", "continued", "bounded"])
def test_vllm_fused_region_matches_native(fused_factory, monkeypatch, stage):
    from atom.plugin.vllm.v41_core_overrides import _attend_block, _forward

    case = fused_factory(64)
    lengths = (1, 1) if stage == "decode" else (3, 5)
    step, queries = make_step(
        case,
        lengths,
        prefill=stage != "decode",
        position=0 if stage == "prefill" else None,
        bounded=stage == "bounded",
    )
    hidden = torch.randn(
        1, step.running_tokens, 256, device="cuda", dtype=torch.bfloat16
    )
    restore_windows(case.cache, case.initial)
    expected = case.model(hidden, None, case.cache, step, case.rope, queries)
    expected_ring = window_bytes(case.cache)
    monkeypatch.setattr(
        Attention, "_atom_v41_attend_block", _attend_block, raising=False
    )
    restore_windows(case.cache, case.initial)
    actual = _forward(case.model, hidden, None, case.cache, step, case.rope, queries)
    assert_output(actual, expected)
    assert torch.equal(window_bytes(case.cache), expected_ring)
