# SPDX-License-Identifier: MIT
"""Direct A8W4 attention through ATOM's quantized producers and CSR builder."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="the V4.1 cache and CSR kernels need Triton")
pytest.importorskip("aiter", reason="V4.1 device attention needs AITER")

from atom.model_ops.attentions.deepseek_v41.a8w4_attention import (
    supports_a8w4_attention,
)
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.fp8_attention import (
    sparse_attn_v41_paged_2src,
)
from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
    MAIN_FP4,
    V41PoolGeometry,
)
from atom.model_ops.blockscale import quantize_fp4, quantize_fp8
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding


def mxfp8(planes):
    data, scale = planes
    values = data.view(torch.float8_e4m3fn).float()
    return values * torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(
        32, -1
    )


def nvfp4(planes):
    data, scale = planes
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    values = levels[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return values * scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)


def make_case(heads, decode, ratio=2, *, cache_dtype="fp4"):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    query = torch.empty(1, 4, heads, 512, device="cuda", dtype=torch.bfloat16)
    if cache_dtype == "fp4" and not supports_a8w4_attention(query, 512**-0.5):
        pytest.skip("This AITER/device combination has no A8W4 backend")
    torch.manual_seed(7407 + heads + ratio)
    geo = V41PoolGeometry(
        1,
        ((0, 2),),
        32,
        8,
        512,
        128,
        kv_dtype=cache_dtype,
        speculative_tokens=3,
        layer_ratios=(0, 2),
        index_topk=4,
    )
    cache = PagedAttentionCache(geo, 7, 4, "cuda", max_tokens=4)
    data, scale = cache.main_view(0)
    raw = torch.randn(data.shape[0], 512, device="cuda", dtype=torch.bfloat16)
    packed, scales = (
        quantize_fp4(raw, **MAIN_FP4) if cache_dtype == "fp4" else quantize_fp8(raw)
    )
    data.copy_(packed.view(torch.uint8))
    scale.copy_(scales.view(torch.uint8))
    data, scale = cache.ring_view(0)
    raw = torch.randn(data.shape, device="cuda", dtype=torch.bfloat16)
    values, scales = quantize_fp8(raw)
    data.copy_(values.view(torch.uint8))
    scale.copy_(scales.view(torch.uint8))
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
    step = BatchStep(
        (RequestSpan(0, 10, 0, 2, 2), RequestSpan(1, 31, 2, 2, 0)),
        tensor([10, 11, 31, 32]),
        tensor([0, 2, 4]),
        tensor([2, 0]),
        tensor([0, 0, 1, 1]),
        tensor([[2, 1, 0], [4, 3, 5]]),
        scheduled_tokens=4,
        max_q_len=2,
        is_prefill=not decode,
        tentative=decode,
        swa_replay_start=tensor([0, 0]),
        swa_cached_until=tensor([0, 0]),
    )
    cache._reserve_prefill_window(step)
    spec = SimpleNamespace(
        layer_id=0,
        kv_owner=0 if ratio else None,
        ratio=ratio,
        topk_owner=0,
        index_group_start=0,
        index_group_size=1,
    )
    return SimpleNamespace(
        cache=cache,
        step=step,
        spec=spec,
        query=torch.randn(1, 4, heads, 512, device="cuda", dtype=torch.bfloat16),
        kv=torch.randn(1, 4, 512, device="cuda", dtype=torch.bfloat16),
        selected=tensor([[0, 1, 2, 3]] * 4),
        rope=RotaryEmbedding(64, 128, base=10000).cuda(),
        sink=torch.linspace(-1, 1, heads, device="cuda"),
    )


def run_case(case, splits=None):
    cache, step = case.cache, case.step
    step.begin_forward()
    step.selected[0] = case.selected
    q8 = cache.query_fp8(case.query, a8w4=True)
    source, pending = cache.rope_quant_window(
        0, case.query, case.kv, case.rope, step, q8, a8w4=True
    )
    pi, pp, wi, wp = cache.attention_indices(case.spec, step, a8w4=True)
    if step.decode:
        cache.write_window(0, pending, step)
        source = cache.ring_view(0)
    query = tuple(value.flatten(0, 1) for value in q8)
    main = cache.main_view(case.spec.kv_owner)
    result = torch.empty_like(case.query.flatten(0, 1))
    sparse_attn_v41_paged_2src(
        query,
        main,
        pi,
        pp,
        source,
        wi,
        wp,
        case.sink,
        512**-0.5,
        out=result,
        num_kv_splits=splits,
        pool_scale_dtype=torch.float8_e4m3fn,
    )
    if not step.decode:
        cache.write_window(0, pending, step)
    return result, (query, main, source, (pi, pp, wi, wp))


def reference(case, inputs):
    # This oracle checks attention arithmetic over the supplied CSR rows.
    # Independent causal row membership is checked by test_contiguous_cache.py::
    # test_csr_resolves_to_causal_main_and_window_rows.
    query, main, source, (pi, pp, wi, wp) = inputs
    q, csa, swa = (
        mxfp8(query),
        nvfp4(main) if case.cache.packed else mxfp8(main),
        mxfp8(source),
    )
    pp, wp = pp.tolist(), wp.tolist()
    out = torch.zeros_like(q)
    for row in range(q.shape[0]):
        keys = torch.cat(
            (
                csa[pi[pp[row] : pp[row + 1]].long()],
                swa[wi[wp[row] : wp[row + 1]].long()],
            )
        )
        if keys.shape[0]:
            logits = q[row] @ keys.T * (512**-0.5)
            weights = torch.cat((logits, case.sink[:, None]), -1).softmax(-1)[:, :-1]
            out[row] = weights @ keys
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("heads", [2, 16, 32, 64])
@pytest.mark.parametrize(
    "decode,ratio,splits",
    [
        (False, 2, None),
        (True, 2, None),
        (False, 2, 1),
        (True, 2, 4),
        (False, 0, 4),
        (True, 0, 1),
    ],
)
def test_a8w4_matches_dequantized_attention(heads, decode, ratio, splits):
    case = make_case(heads, decode, ratio)
    output, inputs = run_case(case, splits)
    expected = reference(case, inputs)
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.02)


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("cache_dtype", ["fp4", "fp8"])
@torch.inference_mode()
def test_gfx950_preserves_requested_cache_and_attention(
    heads, decode, cache_dtype, monkeypatch
):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.config import Config
    from atom.models.deepseek_v41 import attention as model_attention
    from atom.models.deepseek_v41.attention import Attention
    from atom.utils import forward_context

    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
        pytest.skip("This regression exercises gfx950 OPUS attention")
    monkeypatch.setenv("ATOM_DSV41_A8W4_ATTN", "auto")
    cfg = SimpleNamespace(
        kv_cache_dtype=cache_dtype,
        hf_config=SimpleNamespace(
            model_type="deepseek_v41_text", num_attention_heads=heads
        ),
        tensor_parallel_size=4,
        enable_dp_attention=True,
    )
    Config._check_kv_cache_dtype(cfg)
    assert cfg.kv_cache_dtype == cache_dtype
    case = make_case(heads, decode, cache_dtype=cfg.kv_cache_dtype)
    cache, step = case.cache, case.step
    packed = cache_dtype == "fp4"
    assert cache.packed == packed
    assert cache.kv_fp8 == (not packed)
    owner = SimpleNamespace(
        spec=case.spec,
        attn_sink=case.sink,
        softmax_scale=512**-0.5,
    )
    owner._use_a8w4_attention = lambda query, cache: Attention._use_a8w4_attention(
        owner, query, cache
    )
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(
            in_hipgraph=False,
            main_stream=torch.cuda.current_stream(),
            attn_metadata=SimpleNamespace(cache=cache, step=step, queries=None),
        ),
    )
    calls = []
    name = "sparse_attn_v41_paged_2src"
    attention = getattr(model_attention, name)

    def attend(*args, **kwargs):
        assert kwargs.get("pool_scale_dtype", torch.float8_e8m0fnu) == (
            torch.float8_e4m3fn if packed else torch.float8_e8m0fnu
        )
        calls.append(True)
        return attention(*args, **kwargs)

    monkeypatch.setattr(model_attention, name, attend)
    for name in ("packed_prefill", "packed_decode"):
        monkeypatch.setattr(
            model_attention,
            name,
            lambda *a, **kw: pytest.fail("attention does not match requested KV dtype"),
        )

    def run():
        step.begin_forward()
        step.selected[0] = case.selected
        prepared = Attention._attention_inputs(
            owner,
            case.query,
            case.kv,
            cache,
            step,
            case.rope,
            None,
            a8w4=owner._use_a8w4_attention(case.query, cache),
        )
        assert prepared[0] is packed
        assert prepared[1] is not None
        output = Attention._attend(owner, case.query, cache, step, None, prepared)
        source = cache.ring_view(0) if decode else prepared[2]
        inputs = (
            tuple(part.flatten(0, 1) for part in prepared[1]),
            cache.main_view(0),
            source,
            cache.attention_indices(case.spec, step, a8w4=packed),
        )
        return output, inputs

    output, inputs = run()
    torch.testing.assert_close(output, reference(case, inputs), rtol=0.03, atol=0.02)
    assert calls
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, inputs = run()
    for active in (2, 1, 2):
        case.query.normal_()
        case.kv.normal_()
        case.step.batch_ids.copy_(
            torch.tensor([0, 0, 1, 1] if active == 2 else [0, 0, -1, -1], device="cuda")
        )
        case.step.cu_seqlens_q[-1] = active * 2
        case.step.slots[-1] = 0 if active == 2 else -1
        graph.replay()
        torch.testing.assert_close(
            output, reference(case, inputs), rtol=0.03, atol=0.02
        )


@pytest.mark.parametrize(
    "cache_dtype,a8w4", [("bf16", False), ("fp8", False), ("fp4", False), ("fp4", True)]
)
@pytest.mark.parametrize("capture_empty", [False, True])
@torch.inference_mode()
@pytest.mark.parametrize("side_stream", [False, True])
def test_native_token_graph_replays_different_request_counts(
    cache_dtype, capture_empty, side_stream, a8w4, monkeypatch
):
    from types import MethodType

    from atom.models.deepseek_v41.attention import Attention, _live_step
    from atom.utils import forward_context
    from atom.utils.region_graph import TokenGraph, eager_region

    if not torch.cuda.is_available():
        pytest.skip("Native TokenGraph requires a GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        pytest.skip("This regression exercises gfx950 OPUS attention")
    monkeypatch.setenv("ATOM_DSV41_A8W4_ATTN", "auto" if a8w4 else "off")
    torch.manual_seed(2989)
    width, ring, heads = 256, 128, 16
    geo = V41PoolGeometry(
        1,
        ((0, 2),),
        256,
        ring,
        512,
        128,
        kv_dtype=cache_dtype,
        layer_ratios=(0,),
        index_topk=4,
    )
    cache = PagedAttentionCache(geo, 4, 3, "cuda", max_tokens=width)
    if cache_dtype == "bf16":
        cache.pool.normal_()
        history = (cache.pool.clone(),)
    else:
        data, scales = cache.ring_view(0)
        values, steps = quantize_fp8(
            torch.randn(data.shape, device="cuda", dtype=torch.bfloat16)
        )
        data.copy_(values.view(torch.uint8))
        scales.copy_(steps.view(torch.uint8))
        history = tuple(p.clone() for p in cache.ring_view(0))
    rope = RotaryEmbedding(64, 512, base=10000).cuda()
    positions = torch.empty(width, dtype=torch.int32, device="cuda")
    batch_ids = torch.empty_like(positions)
    cu = torch.empty(3, dtype=torch.int32, device="cuda")
    slots = torch.empty(2, dtype=torch.int32, device="cuda")
    tables = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    floor = torch.zeros(2, dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(in_hipgraph=False, main_stream=torch.cuda.current_stream())
    monkeypatch.setattr(forward_context, "_forward_context", ctx)

    def prepare(requests):
        length = width // requests
        positions.copy_(
            torch.arange(ring, ring + length, device="cuda").repeat(requests)
        )
        batch_ids.copy_(torch.arange(requests, device="cuda").repeat_interleave(length))
        cu.copy_(torch.tensor([0, length, width], device="cuda"))
        slots.copy_(torch.tensor([2, 0 if requests == 2 else -1], device="cuda"))
        step = BatchStep(
            tuple(
                RequestSpan(b, ring, b * length, length, 2 if b == 0 else 0)
                for b in range(requests)
            ),
            positions,
            cu[: requests + 1],
            slots[:requests],
            batch_ids,
            tables[:requests],
            scheduled_tokens=width,
            max_q_len=length,
            is_prefill=True,
            swa_replay_start=floor[:requests],
            swa_cached_until=floor[:requests],
        )
        cache._reserve_prefill_window(step)
        step.begin_forward()
        ctx.attn_metadata = SimpleNamespace(cache=cache, step=step, queries=None)
        planes = (cache.pool,) if cache_dtype == "bf16" else cache.ring_view(0)
        for dst, saved in zip(planes, history):
            dst.copy_(saved)
        return step

    owner = SimpleNamespace(
        heads=heads,
        head_dim=512,
        spec=SimpleNamespace(
            layer_id=0,
            kv_owner=None,
            ratio=0,
            topk_owner=0,
            index_group_start=0,
            index_group_size=1,
        ),
        attn_sink=torch.zeros(heads, device="cuda"),
        softmax_scale=512**-0.5,
        index_stream=None,
        compress_stream=None,
        _fork_compress=lambda *a, **k: False,
        project_qkv=lambda q, kv: (q, kv),
        qk_norm=lambda q, kv: (q, None, kv),
        _fork_select=lambda *a, **k: None,
        wq_b=lambda q, **kw: q.clone().flatten(-2),
        _project_out=lambda output, *args: output.clone(),
    )
    if side_stream:
        owner.index_stream = torch.cuda.Stream()
        marker = torch.zeros(1, device="cuda")

        @eager_region(stream=lambda self: self.index_stream, live=_live_step)
        def select(self, hidden, qr, qr_scale, cache, step, rope, *, compressed):
            marker.add_(1)

        owner._fork_select = MethodType(select, owner)
    owner._use_a8w4_attention = MethodType(Attention._use_a8w4_attention, owner)
    for name in (
        "_prepare_attention",
        "_prepare_attention_live",
        "_attention_inputs",
        "_attend",
    ):
        setattr(owner, name, MethodType(getattr(Attention, name), owner))

    preparation_calls = []
    prepare_window = cache.rope_quant_window

    def track_preparation(*args, **kwargs):
        preparation_calls.append(torch.cuda.is_current_stream_capturing())
        result = prepare_window(*args, **kwargs)
        if cache_dtype == "fp8" or (cache_dtype == "fp4" and a8w4):
            source = result[0]
            reserved = ctx.attn_metadata.step.prefill_window_staging
            assert source[0].untyped_storage().data_ptr() == reserved.data_ptr()
            assert source[1].untyped_storage().data_ptr() == reserved.data_ptr()
        return result

    monkeypatch.setattr(cache, "rope_quant_window", track_preparation)

    def forward(q, kv):
        return Attention.forward(owner, q, kv, cache, ctx.attn_metadata.step, rope)

    query = torch.randn(1, width, heads, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, width, 512, device="cuda", dtype=torch.bfloat16)
    for requests in (2, 1):
        prepare(requests)
        forward(query, kv)
    prepare(2)
    if capture_empty:
        ctx.attn_metadata.step.requests = ()
    preparation_calls.clear()
    graph = TokenGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            ctx.main_stream = stream
            ctx.in_hipgraph = True
            graph.record(forward, (query, kv))
            ctx.in_hipgraph = False
        torch.cuda.current_stream().wait_stream(stream)
        ctx.main_stream = torch.cuda.current_stream()
        live_preparation = cache_dtype == "fp8" or (cache_dtype == "fp4" and a8w4)
        expected_capture = (
            [] if capture_empty and live_preparation else [not live_preparation]
        )
        assert preparation_calls == expected_capture
        for requests in (2, 1, 2):
            fresh_q, fresh_kv = torch.randn_like(query), torch.randn_like(kv)
            prepare(requests)
            preparation_calls.clear()
            got = graph.replay((fresh_q, fresh_kv)).clone()
            assert preparation_calls == ([False] if live_preparation else [])
            prepare(requests)
            expected = forward(fresh_q, fresh_kv)
            torch.testing.assert_close(
                got,
                expected,
                rtol=0.03,
                atol=0.02,
                msg=lambda message, requests=requests: f"{requests} requests: {message}",
            )
        assert len(graph.keys()) == 1
    finally:
        ctx.in_hipgraph = False
        torch.cuda.synchronize()
        graph.release_graphs()
