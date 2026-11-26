# SPDX-License-Identifier: MIT
"""Direct A8W4 attention through ATOM's quantized producers and CSR builder."""

from types import SimpleNamespace

import numpy as np
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
    "cache_dtype,a8w4,capture_attention,ratio,group_size,live_fusion",
    [
        ("bf16", False, False, 0, 1, False),
        ("fp8", False, False, 0, 1, False),
        ("fp4", False, False, 0, 1, False),
        ("fp4", True, False, 0, 1, False),
        ("fp8", False, True, 0, 1, False),
        ("fp4", True, True, 0, 1, False),
        ("fp8", False, True, 2, 1, False),
        ("fp4", True, True, 2, 1, False),
        ("fp8", False, True, 2, 2, False),
        ("fp4", True, True, 2, 2, False),
        ("fp8", False, False, 0, 1, True),
        ("fp4", True, False, 0, 1, True),
        ("fp8", False, False, 2, 2, True),
        ("fp4", True, False, 2, 2, True),
    ],
)
@pytest.mark.parametrize("capture_batch", ["ordinary", "resumed", "empty"])
@torch.inference_mode()
@pytest.mark.parametrize("side_stream", [False, True])
def test_native_token_graph_replays_different_request_counts(
    cache_dtype,
    capture_batch,
    side_stream,
    a8w4,
    capture_attention,
    ratio,
    group_size,
    live_fusion,
    monkeypatch,
):
    from types import MethodType

    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
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
    width, ring, heads = 256, 128, 64 if live_fusion else 16
    geo = V41PoolGeometry(
        group_size,
        ((0, 2),),
        256,
        ring,
        512,
        128,
        kv_dtype=cache_dtype,
        layer_ratios=(ratio,),
        index_topk=4,
    )
    cache = PagedAttentionCache(geo, 4, 5, "cuda", max_tokens=width)

    def ring_planes():
        return (
            (cache.pool,)
            if cache_dtype == "bf16"
            else tuple(p for layer in range(group_size) for p in cache.ring_view(layer))
        )

    if cache_dtype == "bf16":
        cache.pool.normal_()
    else:
        for layer in range(group_size):
            data, scales = cache.ring_view(layer)
            values, steps = quantize_fp8(
                torch.randn(data.shape, device="cuda", dtype=torch.bfloat16)
            )
            data.copy_(values.view(torch.uint8))
            scales.copy_(steps.view(torch.uint8))
    history = tuple(p.clone() for p in ring_planes())
    if ratio:
        data, scales = cache.main_view(0)
        raw = torch.randn(data.shape[0], 512, device="cuda", dtype=torch.bfloat16)
        packed, steps = quantize_fp4(raw, **MAIN_FP4) if a8w4 else quantize_fp8(raw)
        data.copy_(packed.view(torch.uint8))
        scales.copy_(steps.view(torch.uint8))
    rope = RotaryEmbedding(64, 512, base=10000).cuda()
    positions = torch.empty(width, dtype=torch.int32, device="cuda")
    batch_ids = torch.empty_like(positions)
    cu = torch.empty(5, dtype=torch.int32, device="cuda")
    slots = torch.empty(4, dtype=torch.int32, device="cuda")
    tables = torch.zeros(4, 4, dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(in_hipgraph=False, main_stream=torch.cuda.current_stream())
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    replay_builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    replay_builder.model_runner = SimpleNamespace(
        forward_vars=replay_builder._prefill_replay_buffers(4, "cuda")
    )
    boundary_pointers = None

    def prepare(
        requests,
        *,
        graph_layout=capture_attention,
        padded=False,
        reverse=False,
        cached_until=0,
        live_tokens=width,
        fuse=live_fusion,
        is_prefill=True,
    ):
        nonlocal boundary_pointers
        length = (live_tokens - (2 if padded else 0)) // requests if requests else 0
        live = length * requests
        order = (0, 3, 1, 4) if reverse else (4, 1, 3, 0)
        positions.zero_()
        batch_ids.fill_(-1)
        if requests:
            positions[:live].copy_(
                torch.arange(ring, ring + length, device="cuda").repeat(requests)
            )
            batch_ids[:live].copy_(
                torch.arange(requests, device="cuda").repeat_interleave(length)
            )
        cu.copy_(
            torch.tensor([min(b, requests) * length for b in range(5)], device="cuda")
        )
        slots.copy_(
            torch.tensor(list(order[:requests]) + [-1] * (4 - requests), device="cuda")
        )
        batches = 4 if graph_layout else requests
        cached = np.broadcast_to(np.asarray(cached_until, dtype=np.int32), (requests,))
        step = BatchStep(
            tuple(
                RequestSpan(b, ring, b * length, length, order[b])
                for b in range(requests)
            ),
            positions,
            cu[: batches + 1],
            slots[:batches],
            batch_ids,
            tables[:batches],
            scheduled_tokens=live,
            max_q_len=length,
            is_prefill=is_prefill,
            prefill_attention_graph=graph_layout,
            prefill_window_fusion=fuse and is_prefill,
            request_floors=np.where(cached > 0, ring // 2, 0).astype(np.int32),
            request_cached_until=cached,
        )
        # Production graph publication precedes the fused producer. The same
        # pointers must survive ordinary/resumed/empty batch transitions.
        if graph_layout:
            step.swa_replay_start, step.swa_cached_until = (
                replay_builder._stage_prefill_replay_layout(
                    step.request_floors, step.request_cached_until, batches
                )
            )
            pointers = tuple(
                p.data_ptr() for p in (step.swa_replay_start, step.swa_cached_until)
            )
            if boundary_pointers is None:
                boundary_pointers = pointers
            assert pointers == boundary_pointers
            assert step.swa_cached_until.numel() == batches
            assert step.swa_cached_until[requests:].count_nonzero() == 0
        else:
            cache.publish_replay_layout(step, batches)
        if cached.any() and not graph_layout:
            assert step.swa_cached_until.numel() == requests
            assert step.swa_replay_start.numel() == requests
        cache._reserve_prefill_window(step)
        step.begin_forward()
        ctx.attn_metadata = SimpleNamespace(cache=cache, step=step, queries=None)
        for dst, saved in zip(ring_planes(), history):
            dst.copy_(saved)
        return step

    def select_rows(step):
        if ratio and step.requests:
            # Fresh storage on every forward, like a live scorer. The captured
            # MLA must read the newly built CSR, not capture's scorer tensor.
            step.selected[0] = (
                (
                    torch.arange(4, device="cuda", dtype=torch.int32)[None, :]
                    + (step.slots[0] * 4)
                )
                .expand(width, 4)
                .contiguous()
            )

    selection_calls = []
    owner = SimpleNamespace(
        heads=heads,
        head_dim=512,
        spec=SimpleNamespace(
            layer_id=0,
            kv_owner=0 if ratio else None,
            ratio=ratio,
            topk_owner=0,
            index_group_start=0,
            index_group_size=group_size,
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
    if ratio or side_stream:
        owner.index_stream = torch.cuda.Stream() if side_stream else None
        marker = torch.zeros(1, device="cuda") if side_stream else None

        # Real Attention._select remains eager on the main stream too. An
        # inline scorer only fills capture's step.selected, leaving replay's
        # new BatchStep empty before _prefill_indices_live reads it.
        @eager_region(stream=lambda self: self.index_stream, live=_live_step)
        def select(self, hidden, qr, qr_scale, cache, step, rope, *, compressed):
            selection_calls.append(step)
            if marker is not None:
                marker.add_(1)
            select_rows(step)

        owner._fork_select = MethodType(select, owner)
    owner._use_a8w4_attention = MethodType(Attention._use_a8w4_attention, owner)
    for name in (
        "_prepare_attention",
        "_prepare_attention_live",
        "_attention_inputs",
        "_prefill_indices_live",
        "_attend_prefill_graph",
        "_attend",
    ):
        setattr(owner, name, MethodType(getattr(Attention, name), owner))

    owners = [owner]
    for layer_id in range(1, group_size):
        layer = SimpleNamespace(**vars(owner))
        layer.spec = SimpleNamespace(**{**vars(owner.spec), "layer_id": layer_id})
        # Each captured callback must retain its own layer owner. The later
        # layers share the first layer's selection/CSR, but have separate rings.
        for name, value in vars(owner).items():
            if isinstance(value, MethodType):
                setattr(layer, name, MethodType(value.__func__, layer))
        layer._fork_select = lambda *a, **k: None
        layer.index_stream = None
        owners.append(layer)

    preparation_calls = []
    prepare_window = cache.rope_quant_window

    def track_preparation(*args, **kwargs):
        preparation_calls.append(torch.cuda.is_current_stream_capturing())
        result = prepare_window(*args, **kwargs)
        if not ctx.attn_metadata.step.decode and (
            cache_dtype == "fp8" or (cache_dtype == "fp4" and a8w4)
        ):
            source = result[0]
            reserved = ctx.attn_metadata.step.prefill_window_staging
            assert source[0].untyped_storage().data_ptr() == reserved.data_ptr()
            assert source[1].untyped_storage().data_ptr() == reserved.data_ptr()
        return result

    monkeypatch.setattr(cache, "rope_quant_window", track_preparation)
    from atom.model_ops.attentions.deepseek_v41 import cache as cache_module

    fused_calls = []
    fused_prepare = cache_module.prefill_rope_window

    def track_fused(*args, **kwargs):
        fused_calls.append(torch.cuda.is_current_stream_capturing())
        return fused_prepare(*args, **kwargs)

    monkeypatch.setattr(cache_module, "prefill_rope_window", track_fused)
    commit_calls = []
    commit_window = cache.write_window

    def track_commit(layer, window_kv, step):
        if live_fusion and step.prefill_window_fusion:
            assert window_kv is None, "Fused producer must not write SWA twice"
        commit_calls.append(step)
        return commit_window(layer, window_kv, step)

    monkeypatch.setattr(cache, "write_window", track_commit)
    graph_commit_calls = []
    graph_commit = cache.write_prefill_window_graph

    def track_graph_commit(layer, window_kv, step, cached_until):
        graph_commit_calls.append(
            (layer, torch.cuda.is_current_stream_capturing(), cached_until.data_ptr())
        )
        return graph_commit(layer, window_kv, step, cached_until)

    monkeypatch.setattr(cache, "write_prefill_window_graph", track_graph_commit)
    index_calls = []
    index_group = cache._index_group

    def track_indices(spec, step, *args, **kwargs):
        index_calls.append((spec.layer_id, step))
        return index_group(spec, step, *args, **kwargs)

    monkeypatch.setattr(cache, "_index_group", track_indices)
    attention_calls = []
    from atom.models.deepseek_v41 import attention as attention_module

    attend = attention_module.sparse_attn_v41_paged_2src

    def track_attention(*args, **kwargs):
        attention_calls.append(torch.cuda.is_current_stream_capturing())
        return attend(*args, **kwargs)

    monkeypatch.setattr(attention_module, "sparse_attn_v41_paged_2src", track_attention)

    def forward(q, kv):
        for layer in owners:
            q = Attention.forward(layer, q, kv, cache, ctx.attn_metadata.step, rope)
        return q

    query = torch.randn(1, width, heads, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, width, 512, device="cuda", dtype=torch.bfloat16)
    for requests in (4, 1):
        prepare(requests)
        forward(query, kv)
    prepare(
        0 if capture_batch == "empty" else 4,
        cached_until=ring + 17 if capture_batch == "resumed" else 0,
    )
    preparation_calls.clear()
    attention_calls.clear()
    graph_commit_calls.clear()
    fused_calls.clear()
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
        live_preparation = (
            cache_dtype == "fp8" or (cache_dtype == "fp4" and a8w4)
        ) and not capture_attention
        expected_capture = (
            []
            if capture_batch == "empty" and live_preparation
            else [not live_preparation]
        )
        assert preparation_calls == expected_capture * group_size
        if capture_attention:
            assert attention_calls == [True] * group_size
            assert fused_calls == [True] * group_size
            assert graph_commit_calls == []
        else:
            assert fused_calls == (
                [False] * group_size if live_fusion and capture_batch != "empty" else []
            )
        # Capture can write only 64 tokens/request; this same bucket must
        # later commit all 128 ring rows of a single 256-token request.
        replay_cases = (
            (4, False, False, 0, width),
            (1, False, True, 0, width),
            (0, False, False, 0, width),
            (4, True, True, 0, width),
            (2, True, False, ring + 17, width),
            (3, True, True, (ring + 17, 0, ring + 33), width),
            (2, True, False, 0, width),
            (1, False, True, ring + width, width),
            (1, False, False, 0, 16),
            (1, False, True, 0, width),
        )
        for requests, padded, reverse, cached_until, live_tokens, is_prefill in (
            [(*case, True) for case in replay_cases]
            # A DPA prefill graph also serves a decoding peer, an idle peer,
            # then prefill again. The live callback must follow the new step.
            + (
                [
                    (4, False, True, 0, 4, False),
                    (0, False, False, 0, 4, False),
                    (4, False, False, 0, 4, True),
                    (1, False, False, 0, width, True),
                ]
                if live_fusion
                else []
            )
        ):
            fresh_q, fresh_kv = torch.randn_like(query), torch.randn_like(kv)
            step = prepare(
                requests,
                padded=padded,
                reverse=reverse,
                cached_until=cached_until,
                live_tokens=live_tokens,
                is_prefill=is_prefill,
            )
            live = step.scheduled_tokens
            preparation_calls.clear()
            attention_calls.clear()
            selection_calls.clear()
            commit_calls.clear()
            graph_commit_calls.clear()
            index_calls.clear()
            fused_calls.clear()
            got = graph.replay((fresh_q, fresh_kv)).clone()
            if live_fusion:
                assert fused_calls == (
                    [False] * group_size if requests and is_prefill else []
                )
                assert attention_calls == ([False] * group_size if requests else [])
            assert len(commit_calls) == group_size * int(
                bool(requests) and not capture_attention
            )
            assert all(s is step for s in commit_calls)
            assert graph_commit_calls == []
            assert len(index_calls) == int(bool(requests))
            assert all(layer_id == 0 and s is step for layer_id, s in index_calls)
            if ratio or side_stream:
                assert len(selection_calls) == 1 and selection_calls[0] is step
            assert (
                preparation_calls
                == ([False] if live_preparation and requests else []) * group_size
            )
            if capture_attention:
                assert attention_calls == []
            planes = ring_planes()
            committed = tuple(p.clone() for p in planes)
            if not requests or np.all(step.request_cached_until >= ring + width):
                for actual, saved in zip(committed, history):
                    torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            prepare(
                requests,
                graph_layout=False,
                fuse=False,
                padded=padded,
                reverse=reverse,
                cached_until=cached_until,
                live_tokens=live_tokens,
                is_prefill=is_prefill,
            )
            expected = forward(fresh_q, fresh_kv)
            assert torch.isfinite(got).all()
            torch.testing.assert_close(
                got[:, :live],
                expected[:, :live],
                rtol=0.03,
                atol=0.02,
                msg=lambda message, requests=requests: (
                    f"{requests} requests: {message}"
                ),
            )
            for actual, plane in zip(committed, planes):
                torch.testing.assert_close(actual, plane, rtol=0, atol=0)
        assert len(graph.keys()) == 1
    finally:
        ctx.in_hipgraph = False
        torch.cuda.synchronize()
        graph.release_graphs()
