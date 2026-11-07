# SPDX-License-Identifier: MIT
"""NVFP4 draft state and bidirectional A8W4 attention against decoded oracles."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("draft FP4 kernels require a GPU", allow_module_level=True)
pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.attentions.deepseek_v41.a8w4_attention import select_a8w4_backend
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
    MAIN_FP4,
    V41PoolGeometry,
)
from atom.model_ops.blockscale import quantize_fp4, quantize_fp8
from atom.model_ops.deepseek_v41.draft_block import DraftStep
from atom.model_ops.deepseek_v41.dspark import (
    draft_attention_csr,
    draft_attention_quantized,
    draft_step_indices,
)
from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step
from tests.attentions.deepseek_v41.test_a8w4_attention import mxfp8, nvfp4


def geometry(window=128):
    return V41PoolGeometry(
        3,
        ((0, 2),),
        32,
        window,
        512,
        128,
        kv_dtype="fp4",
        speculative_tokens=5,
        fp4_window_layers=(1, 2),
    )


def make_cache(window=128):
    cache = PagedAttentionCache(geometry(window), 32, 4, "cuda")
    for layer in range(3):
        data, scale = cache.ring_view(layer)
        raw = torch.randn(data.shape[0], 512, device="cuda", dtype=torch.bfloat16)
        values = quantize_fp4(raw, **MAIN_FP4) if layer else quantize_fp8(raw)
        for dst, src in zip((data, scale), values):
            dst.copy_(src.view(torch.uint8))
    return cache


def oracle(query, context, current, anchors, window, sink):
    # Independent visibility formula: historical positions within the window,
    # followed by ALL current block rows, including later draft positions.
    ring = context.shape[1]
    slots = torch.arange(ring, device=query.device)
    positions = anchors[:, None] - (anchors[:, None] - slots) % ring
    valid = (positions >= 0) & (positions > anchors[:, None] - window)
    keys = torch.cat((context.float(), current.float()), 1)
    valid = torch.cat(
        (valid, torch.ones(current.shape[:2], dtype=torch.bool, device=query.device)), 1
    )
    scores = torch.einsum("bthd,bkd->bhtk", query.float(), keys) * 512**-0.5
    scores.masked_fill_(~valid[:, None, None, :], -torch.inf)
    scores = torch.cat(
        (scores, sink[None, :, None, None].expand(*scores.shape[:-1], 1)), -1
    )
    return torch.einsum("bhtk,bkd->bthd", scores.softmax(-1)[..., :-1], keys).bfloat16()


def csr_oracle(indices, ring):
    """Independent host enumeration, including request-relative source offsets."""
    prefix, pp, extend, ep = [], [0], [], [0]
    width = indices.shape[1]
    for batch, queries in enumerate(indices.cpu().tolist()):
        for query in queries:
            prefix.extend(batch * ring + i for i in query if 0 <= i < ring)
            extend.extend(batch * width + i - ring for i in query if i >= ring)
            pp.append(len(prefix))
            ep.append(len(extend))
    return prefix, pp, extend, ep


def assert_csr_equal(actual, expected):
    prefix, pp, extend, ep = [p.cpu().tolist() for p in actual]
    assert pp == expected[1]
    assert ep == expected[3]
    assert prefix[: pp[-1]] == expected[0]
    assert extend[: ep[-1]] == expected[2]


@pytest.mark.parametrize("batch", [0, 1, 8, 17, 32, 65])
@pytest.mark.parametrize("width", [1, 5, 6])
@pytest.mark.parametrize("window", [8, 128])
def test_fused_csr_matches_host_enumeration(batch, width, window):
    anchors = torch.tensor(
        ([-1, 0, 3, window - 1, window + 4, 2 * window + 7] * (batch + 1))[:batch],
        device="cuda",
    )
    _, indices, _ = draft_step_indices(anchors, window + 5, window, width)
    expected = csr_oracle(indices, window + 5)
    assert_csr_equal(draft_attention_csr(indices, window + 5), expected)


def test_fused_csr_preserves_shared_sparse_masks_and_noncontiguous_input():
    # The offline draft_step path can contain arbitrary historical holes.
    ids = torch.tensor(
        [[7, -1, 0, 9, -1, 3, 8, 10, -1, -1, -1]] * 19, device="cuda", dtype=torch.int32
    )
    indices = ids[:, None, :].expand(-1, 3, -1)
    assert not indices.is_contiguous()
    assert_csr_equal(draft_attention_csr(indices, 8), csr_oracle(indices, 8))


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("supplied_step", [False, True])
def test_stages_share_one_live_csr_per_draft_round(monkeypatch, fp4, supplied_step):
    from torch import nn

    from atom.models.deepseek_v41 import dspark

    seen, builds = [], []

    class Stage(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.attn = SimpleNamespace(spec=SimpleNamespace(layer_id=layer_id))
            self.norm = nn.Identity()

        def forward(self, state, context_kv, step, rope):
            seen.append(step)
            return state

    def build(indices, ring):
        assert fp4, "BF16 drafting must not construct CSR"
        result = draft_attention_csr(indices, ring)
        builds.append(result)
        return result

    monkeypatch.setattr(dspark, "draft_attention_csr", build)
    draft = dspark.DeepseekV41DSpark.__new__(dspark.DeepseekV41DSpark)
    nn.Module.__init__(draft)
    draft.embed = nn.Embedding(8, 32, device="cuda", dtype=torch.bfloat16)
    draft.head = nn.Identity()
    draft.block_size, draft.window_size = 5, 128
    draft.config = SimpleNamespace(hc_mult=4)
    draft.rope = None
    draft._noise_embedding = lambda: draft.embed.weight[0]
    draft.mtp = nn.ModuleList([Stage(i) for i in (40, 41, 42)])
    raw = torch.zeros(3, 133, 512, device="cuda", dtype=torch.bfloat16)
    context = {i: quantize_fp4(raw, **MAIN_FP4) if fp4 else raw for i in (40, 41, 42)}
    previous = None
    for anchors in ([-1, 3, 260], [400, -1, 2]):
        anchors = torch.tensor(anchors, device="cuda")
        positions, indices, context_positions = draft_step_indices(anchors, 133, 128, 5)
        seen.clear()
        before = len(builds)
        draft.draft_hidden(
            torch.zeros(3, device="cuda", dtype=torch.long),
            anchors,
            context,
            context_positions,
            # Even if a caller reuses a step with old CSR, this invocation
            # must build from its live visibility rather than keep old rows.
            step=(
                DraftStep(positions, indices, True, previous) if supplied_step else None
            ),
        )
        assert len(seen) == 3 and all(step is seen[0] for step in seen)
        assert seen[0].decode == supplied_step
        assert len(builds) == before + int(fp4)
        if fp4:
            assert all(step.csr is builds[-1] for step in seen)
            assert_csr_equal(seen[0].csr, csr_oracle(indices, 133))
            assert seen[0].csr is not previous
            previous = seen[0].csr
        else:
            assert all(step.csr is None for step in seen)


def test_shared_csr_native_draft_graph_updates_live_request_masks(monkeypatch):
    from atom.spec_decode.draft_graph import DraftGraph, StagedInput

    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "1")
    captured = []

    def forward(bs, anchors):
        _, indices, _ = draft_step_indices(anchors, 133, 128, 5)
        csr = draft_attention_csr(indices, 133)
        captured.append(csr)
        return csr

    graph = DraftGraph(
        forward=forward,
        capture_epilogue=True,
        inputs={"anchors": StagedInput(dtype=torch.int64)},
    ).bind(SimpleNamespace(max_num_seqs=4), "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.warmup(4)
    torch.cuda.current_stream().wait_stream(stream)
    build_count = len(captured)
    for anchors in ([260, 400, 130, 2], [-1], [0, 260, -1]):
        live = torch.tensor(anchors, device="cuda", dtype=torch.int64)
        staged = graph.stage(4, {"anchors": live})
        actual = graph.run(4, **staged)
        _, indices, _ = draft_step_indices(staged["anchors"], 133, 128, 5)
        assert_csr_equal(actual, csr_oracle(indices, 133))
        # DraftGraph clones returned tensors to keep them alive outside its
        # pool. Its captured intermediates stay fixed and update in place.
        assert len(captured) == build_count
        assert_csr_equal(captured[-1], csr_oracle(indices, 133))
    assert graph.is_captured(4)


def test_shared_csr_attention_matches_original_order_without_rebuilding(monkeypatch):
    from dataclasses import replace

    from atom.model_ops.deepseek_v41 import dspark

    backend = select_a8w4_backend(64, 512, 512**-0.5)
    if backend is None:
        pytest.skip("No A8W4 backend on this device")
    torch.manual_seed(61106)
    query = torch.randn(3, 5, 64, 512, device="cuda", dtype=torch.bfloat16)
    history = quantize_fp4(
        torch.randn(3, 133, 512, device="cuda", dtype=torch.bfloat16), **MAIN_FP4
    )
    current = quantize_fp8(torch.randn(3, 5, 512, device="cuda", dtype=torch.bfloat16))
    q8 = quantize_fp8(query)
    sink = torch.linspace(-1, 1, 64, device="cuda")
    anchors = torch.tensor([-1, 3, 268], device="cuda")
    positions, indices, _ = draft_step_indices(anchors, 133, 128, 5)
    reference_csr = tuple(
        torch.tensor(p, device="cuda", dtype=torch.int32)
        for p in csr_oracle(indices, 133)
    )
    step = DraftStep(positions, indices, csr=reference_csr)
    reference = draft_attention_quantized(
        query,
        history,
        current,
        sink,
        step,
        512**-0.5,
        backend=backend,
        query_fp8=q8,
        fp4=True,
    )
    shared = draft_attention_csr(indices, 133)

    def unexpected(*args):
        pytest.fail("An attention stage rebuilt the shared CSR")

    monkeypatch.setattr(dspark, "draft_attention_csr", unexpected)
    actual = draft_attention_quantized(
        query,
        history,
        current,
        sink,
        replace(step, csr=shared),
        512**-0.5,
        backend=backend,
        query_fp8=q8,
        fp4=True,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("width", [1, 5])
@pytest.mark.parametrize("window", [8, 128])
@pytest.mark.parametrize("fallback", [False, True])
def test_draft_fp4_matches_dense_bidirectional_attention(
    heads, width, window, fallback
):
    backend = None if fallback else select_a8w4_backend(heads, 512, 512**-0.5)
    if backend is None and not fallback:
        pytest.skip("No A8W4 backend on this device")
    torch.manual_seed(5901)
    cache = make_cache(window)
    slots = torch.tensor([3, 0, 2], device="cuda", dtype=torch.int32)
    layers = torch.tensor([2, 1], device="cuda")
    gathered = cache.read_window_planes(layers, slots, fp4=True)
    context = tuple(p[0] for p in gathered)
    anchors = torch.tensor([-1, 3, 2 * window + 5], device="cuda")
    positions, indices, _ = draft_step_indices(anchors, window + 5, window, width)
    query = torch.randn(3, width, heads, 512, device="cuda", dtype=torch.bfloat16)
    current = quantize_fp8(
        torch.randn(3, width, 512, device="cuda", dtype=torch.bfloat16)
    )
    q8 = quantize_fp8(query)
    sink = torch.linspace(-1, 1, heads, device="cuda")
    before = [p.clone() for p in cache.window_sources]
    out = draft_attention_quantized(
        query,
        context,
        current,
        sink,
        DraftStep(positions, indices),
        512**-0.5,
        backend=backend,
        query_fp8=q8,
        fp4=True,
    )
    expected = oracle(
        query if fallback else mxfp8(q8),
        nvfp4(context),
        mxfp8(current),
        anchors,
        window,
        sink,
    )
    torch.testing.assert_close(out, expected, rtol=0.03, atol=0.02)
    for original, actual in zip(before, cache.window_sources):
        assert torch.equal(original, actual)


def test_fp4_gather_reset_and_graph_follow_live_layer_and_slot_ids():
    cache = make_cache()
    layers = torch.tensor([1, 2], device="cuda")
    slots = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    cache.read_window_planes(layers, slots, fp4=True)
    cache.read_windows(layers, slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        raw = cache.read_window_planes(layers, slots, fp4=True)
        decoded = cache.read_windows(layers, slots)
    for layer_order, slot_order in (([1, 2], [3, 0]), ([2, 1], [1, 3])):
        layers.copy_(torch.tensor(layer_order, device="cuda"))
        slots.copy_(torch.tensor(slot_order, device="cuda", dtype=torch.int32))
        graph.replay()
        for i, layer in enumerate(layer_order):
            expected = tuple(
                p.view(4, 133, -1)[slots.long()] for p in cache.ring_view(layer)
            )
            for a, b in zip(raw, expected):
                assert torch.equal(a[i], b)
            torch.testing.assert_close(
                decoded[i], nvfp4(expected).bfloat16(), rtol=0, atol=0
            )
    before = [p.clone() for p in cache.window_sources]
    cache._reset_state_slots(torch.tensor([1, 3], device="cuda", dtype=torch.int64))
    for layer in range(3):
        for plane in cache.ring_view(layer):
            per_slot = plane.view(4, 133, -1)
            assert not per_slot[[1, 3]].count_nonzero()
        from atom.model_ops.attentions.deepseek_v41.planes import split_source

        for old, new in zip(
            split_source(before[layer], 512, layer != 0), cache.ring_view(layer)
        ):
            assert torch.equal(
                old.view(4, 133, -1)[[0, 2]], new.view(4, 133, -1)[[0, 2]]
            )


def test_a8w4_draft_graph_replays_changed_anchors_and_reordered_requests():
    backend = select_a8w4_backend(64, 512, 512**-0.5)
    if backend is None:
        pytest.skip("No A8W4 backend on this device")
    cache = make_cache()
    slots = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    layers = torch.tensor([2], device="cuda")
    anchors = torch.tensor([127, 268], device="cuda")
    query = torch.randn(2, 5, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(-1, 1, 64, device="cuda")

    def run():
        context = tuple(p[0] for p in cache.read_window_planes(layers, slots, fp4=True))
        pos, indices, _ = draft_step_indices(anchors, 133, 128, 5)
        q8, k8 = quantize_fp8(query), quantize_fp8(kv)
        out = draft_attention_quantized(
            query,
            context,
            k8,
            sink,
            DraftStep(pos, indices),
            512**-0.5,
            backend=backend,
            query_fp8=q8,
            fp4=True,
        )
        return out, context, q8, k8

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, context, q8, k8 = run()
    for new_anchors, new_slots in (
        ([0, 2], [0, 3]),
        ([268, 128], [2, 1]),
        ([-1, 132], [3, 0]),
    ):
        anchors.copy_(torch.tensor(new_anchors, device="cuda"))
        slots.copy_(torch.tensor(new_slots, device="cuda", dtype=torch.int32))
        query.normal_()
        kv.normal_()
        graph.replay()
        expected = oracle(mxfp8(q8), nvfp4(context), mxfp8(k8), anchors, 128, sink)
        torch.testing.assert_close(out, expected, rtol=0.03, atol=0.02)


@pytest.mark.parametrize("accepted", range(1, 7))
@pytest.mark.parametrize("position", [127, 128, 265])
def test_rejected_draft_rows_are_invisible_next_round(single_rank, accepted, position):
    cache = PagedAttentionCache(geometry(), 32, 4, "cuda")
    torch.manual_seed(51)
    raw = torch.randn(position + 6, 512, device="cuda", dtype=torch.bfloat16)
    encoded = quantize_fp4(raw, **MAIN_FP4)
    for dst, src in zip(cache.ring_view(2), encoded):
        kept = torch.arange(max(0, position - 133), position, device="cuda")
        dst[3 * 133 + kept % 133] = src.view(torch.uint8)[kept]
    cache.cursor[3, 0] = position
    span = PagedRequest(19, position, 0, 6, 3, tuple(range(16)))
    step = begin_step(cache, (span,), tentative=True)
    cache.prepare_state(step)
    cache.pending.stage_history(span, [1, 2, 3, 4, 5, 6])
    cache.write_window(2, tuple(p[position:] for p in encoded), step)
    cache.commit_tentative(step, torch.tensor([accepted - 1], device="cuda"))
    anchor = position + accepted - 1
    context = cache.read_window_planes(
        torch.tensor([2], device="cuda"), torch.tensor([3], device="cuda"), fp4=True
    )
    _, indices, _ = draft_step_indices(
        torch.tensor([anchor], device="cuda"), 133, 128, 5
    )
    ids = indices[0, 0, :133]
    ids = ids[ids >= 0].long()
    logical = anchor - (anchor - ids) % 133
    assert logical.min() == max(0, anchor - 127)
    assert logical.max() == anchor
    for actual, reference in zip(context, encoded):
        assert torch.equal(actual[0, 0, ids], reference.view(torch.uint8)[logical])


def test_real_draft_forward_produces_quantized_queries_after_rope(monkeypatch):
    from atom.model_ops.deepseek_v41.draft_block import rotate_rows
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
    from atom.models.deepseek_v41.dspark import DraftAttention

    backend = select_a8w4_backend(64, 512, 512**-0.5)
    if backend is None:
        pytest.skip("No A8W4 backend on this device")
    rope = RotaryEmbedding(64, 512, base=10000).cuda()
    query = torch.randn(2, 5, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    cache = make_cache()
    history = tuple(
        p[0]
        for p in cache.read_window_planes(
            torch.tensor([2], device="cuda"),
            torch.tensor([0, 3], device="cuda"),
            fp4=True,
        )
    )
    anchors = torch.tensor([3, 260], device="cuda")
    positions, indices, _ = draft_step_indices(anchors, 133, 128, 5)
    sink = torch.linspace(-1, 1, 64, device="cuda")
    instance = SimpleNamespace(
        heads=64,
        head_dim=512,
        spec=SimpleNamespace(layer_id=2),
        project_qkv=lambda *args: (None, kv),
        qk_norm=lambda *args: (None, None, kv),
        wq_b=lambda *args, **kw: query.clone().flatten(-2),
        attn_sink=sink,
        softmax_scale=512**-0.5,
        _project_out=lambda out, *args: out,
    )
    step = DraftStep(positions, indices)
    out = DraftAttention.forward(instance, None, None, {2: history}, step, rope)
    q8 = quantize_fp8(rotate_rows(rope, query.clone(), positions))
    k8 = quantize_fp8(rotate_rows(rope, kv.clone(), positions))
    expected = oracle(mxfp8(q8), nvfp4(history), mxfp8(k8), anchors, 128, sink)
    torch.testing.assert_close(out, expected, rtol=0.04, atol=0.03)


@pytest.mark.parametrize("fused", [False, True])
def test_context_writer_quantizes_directly_to_fp4_without_fp8_roundtrip(
    monkeypatch, fused, single_rank
):
    from torch import nn

    from atom.model_ops.deepseek_v41.draft_block import rotate_rows
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
    from atom.model_ops.layernorm import RMSNorm, rmsnorm2d_fwd_
    from atom.models.deepseek_v41 import dspark

    torch.manual_seed(632)
    cache = PagedAttentionCache(geometry(8), 32, 4, "cuda")
    step = begin_step(cache, (PagedRequest(3, 0, 0, 17, 2, (0, 1)),))
    draft = dspark.DeepseekV41DSpark.__new__(dspark.DeepseekV41DSpark)
    nn.Module.__init__(draft)
    draft.rope = RotaryEmbedding(64, 64, base=10000).cuda()
    draft.project_context = lambda rows: rows
    projected = torch.randn(1, 17, 2, 512, device="cuda", dtype=torch.bfloat16)
    norms = [RMSNorm(512, 1e-6).cuda() for _ in range(2)]
    for norm in norms:
        norm.weight.data.uniform_(0.5, 1.5)
    layers = []
    for i, norm in enumerate(norms):

        def project(hidden, positions, rope, packed, fp4, i=i, norm=norm):
            normalized = rmsnorm2d_fwd_(
                projected[..., i, :], norm.weight, norm.eps, 512
            )
            return dspark.DraftAttention.context_keys(
                None, normalized, positions, rope, packed=packed, fp4=fp4
            )

        layers.append(
            SimpleNamespace(
                attn=SimpleNamespace(
                    spec=SimpleNamespace(layer_id=i + 1),
                    kv_norm=norm,
                    project_context=project,
                )
            )
        )
    draft.mtp = layers
    draft._context_kv_fusion = lambda: (None, None, 32, 512, False) if fused else None
    monkeypatch.setattr(
        dspark, "native_quant_linear", lambda *args, **kw: projected.flatten(-2)
    )
    monkeypatch.setattr(
        "atom.utils.forward_context.get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False),
            attn_metadata=SimpleNamespace(cache=cache, step=step),
        ),
    )
    positions = torch.arange(17, device="cuda")
    draft.write_context_kv(torch.empty(17, 512, device="cuda"), positions)
    kept = positions[-13:]
    for i, norm in enumerate(norms):
        normalized = rmsnorm2d_fwd_(projected[..., i, :], norm.weight, norm.eps, 512)
        rotated = rotate_rows(draft.rope, normalized, positions[None])
        expected = quantize_fp4(rotated, **MAIN_FP4)
        double_quant = quantize_fp4(quantize_fp8(rotated, dequantize=True), **MAIN_FP4)
        assert any(
            not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(expected, double_quant)
        )
        for actual, wanted in zip(cache.ring_view(i + 1), expected):
            assert torch.equal(
                actual[2 * 13 + kept % 13], wanted.view(torch.uint8)[0, kept]
            )
        for plane in cache.ring_view(i + 1):
            assert not plane.view(4, 13, -1)[[0, 1, 3]].count_nonzero()
    assert not cache.window_sources[0].count_nonzero()
