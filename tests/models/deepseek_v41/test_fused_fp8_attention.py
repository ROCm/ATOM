# SPDX-License-Identifier: MIT
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
from atom.model_ops.attentions.deepseek_v41.packed_rows import pack_rows
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
    ):
        torch.manual_seed(6821 + heads + ratio)
        config = SimpleNamespace(
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
        )
        cache = PagedAttentionCache(geo, pages, slots, "cuda", max_tokens=max_tokens)
        for field in (cache.pages.view("main_0"), cache.state.view("window")):
            source = torch.randn(*field.shape[:-1], 512, device="cuda").bfloat16()
            field.copy_(pack_rows(*quantize_fp8(source)))
        rope = RotaryEmbedding(64, context + max_tokens + 256, base=10000).cuda()
        return SimpleNamespace(
            model=model,
            cache=cache,
            rope=rope,
            topk=topk,
            context=context,
            initial=cache.backing.clone(),
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
    return {
        "relative": (delta.norm() / expected.norm().clamp_min(1e-20)).item(),
        "max_abs": delta.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
    }


def assert_output(actual, expected):
    metrics = error_metrics(actual, expected)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert metrics["relative"] < 0.03, metrics
    assert metrics["cosine"] > 0.999, metrics
    return metrics


@torch.inference_mode()
@pytest.mark.parametrize("stage", ["decode", "prefill", "long_prefill"])
@pytest.mark.parametrize("q_lora_rank", [32, 64, 96, 1280])
def test_norm_rope_quant_window_matches_two_launches(fused_factory, stage, q_lora_rank):
    # 32 must stay unsplit; 64/96 and V4.1's 1280 exercise full-group splits.
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

    cache.backing.copy_(case.initial)
    qr, qr_scale, normed = model.qk_norm(q, kv)
    rows, window = cache.rope_quant_window(layer, None, normed, rope, step)
    ring = cache.backing.clone()

    cache.backing.copy_(case.initial)
    actual = cache.norm_rope_quant_window(layer, model.qk_norm, q, kv, rope, step)
    assert torch.equal(actual[0].view(torch.uint8), qr.view(torch.uint8))
    assert torch.equal(actual[1].view(torch.uint8), qr_scale.view(torch.uint8))
    assert torch.equal(cache.backing, ring)
    if stage == "decode":
        assert not torch.equal(ring, case.initial)
        assert actual[2:] == (None, None) == (rows, window)
    else:
        assert torch.equal(actual[2], rows)


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
        case.cache.backing.copy_(case.initial)
        outputs[mode] = case.model(hidden, None, case.cache, step, case.rope, queries)
        caches[mode] = case.cache.backing.clone()
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
        case.cache.backing.copy_(case.initial)
        output = case.model(hidden, None, case.cache, step, case.rope, queries)
        if mode == "unfused":
            reference, ring = output, case.cache.backing.clone()
        else:
            record_property(mode, assert_output(output, reference))
            assert torch.equal(case.cache.backing, ring)


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
        case.cache.backing.copy_(case.initial)
        output = case.model(hidden, None, case.cache, step, case.rope)
        if mode == "unfused":
            reference, ring = output, case.cache.backing.clone()
        else:
            record_property(mode, assert_output(output, reference))
            assert torch.equal(case.cache.backing, ring)


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
        case.cache.backing.copy_(case.initial)
        output = case.model(hidden, None, case.cache, step, case.rope)
        if mode == "unfused":
            reference, ring = output, case.cache.backing.clone()
        else:
            record_property(mode, assert_output(output, reference))
            assert torch.equal(case.cache.backing, ring)
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
            cache.backing.copy_(case.initial)
            forward(hidden)
        cache.backing.copy_(case.initial)
        ctx.in_hipgraph = True
        graph.record(forward, (hidden,), {}, key="attention", phase="prefill")
        ctx.in_hipgraph = False
    torch.cuda.current_stream().wait_stream(stream)
    ctx.main_stream = torch.cuda.current_stream()
    try:
        phases = [(True, lengths), (True, lengths)]
        if kind == "token":
            phases += [(True, (3, 5)), (True, (4, 4)), (True, ())]
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
            cache.backing.copy_(case.initial)
            actual = graph.replay(
                (fresh,), {}, key="attention", phase="prefill"
            ).clone()
            actual_cache = cache.backing.clone()
            cache.backing.copy_(case.initial)
            expected = forward(fresh)
            if lengths:
                assert_output(actual, expected)
                if previous is not None:
                    assert not torch.equal(actual, previous)
                previous = actual
            else:
                assert torch.equal(actual, torch.zeros_like(actual))
                assert torch.equal(actual_cache, case.initial)
            assert torch.equal(actual_cache, cache.backing)
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
        None,
        None,
        None,
        case.cache,
        step,
        None,
        rope=case.rope,
        out=out.view(torch.float8_e4m3fn),
        out_scale=scale.view(torch.float8_e8m0fnu),
    )
    assert result is None
    assert out.count_nonzero() == 0 and torch.all(scale == 127)
    assert torch.equal(case.cache.backing, case.initial)


@pytest.mark.parametrize("heads", [16, 32, 64, 128])
def test_fused_attention_conditions(fused_factory, heads):
    case = fused_factory(heads)
    model, cache, rope = case.model, case.cache, case.rope
    for rows in (1, 127, 128):
        query = torch.empty(1, rows, heads, 512, device="cuda", dtype=torch.bfloat16)
        assert model._use_fused_attention(query, cache, rope)
    assert not model._use_fused_attention(query.half(), cache, rope)
    for field in ("kv_fp8", "packed"):
        original = getattr(cache, field)
        setattr(cache, field, field == "packed")
        assert not model._use_fused_attention(query, cache, rope)
        setattr(cache, field, original)
    model.fused_attn = False
    assert not model._use_fused_attention(query, cache, rope)
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
    case.cache.backing.copy_(case.initial)
    expected = case.model(hidden, None, case.cache, step, case.rope, queries)
    expected_ring = case.cache.backing.clone()
    monkeypatch.setattr(
        Attention, "_atom_v41_attend_block", _attend_block, raising=False
    )
    case.cache.backing.copy_(case.initial)
    actual = _forward(case.model, hidden, None, case.cache, step, case.rope, queries)
    assert_output(actual, expected)
    assert torch.equal(case.cache.backing, expected_ring)
