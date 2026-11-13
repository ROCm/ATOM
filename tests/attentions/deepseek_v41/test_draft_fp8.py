# SPDX-License-Identifier: MIT
"""Quantized draft attention, cache state, and replay against independent oracles."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("quantized draft kernels require a GPU", allow_module_level=True)
pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.deepseek_v41.draft_block import DraftStep
from atom.model_ops.deepseek_v41.dspark import (
    draft_attention_csr,
    draft_attention_fp8,
    draft_step_indices,
)
from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step
from tests.attentions.deepseek_v41.test_a8w4_attention import mxfp8


def geometry(window=128, *, target_dtype="fp4"):
    return V41PoolGeometry(
        3,
        ((0, 2),),
        32,
        window,
        512,
        128,
        kv_dtype=target_dtype,
        speculative_tokens=5,
    )


def make_cache(window=128, *, target_dtype="fp4"):
    cache = PagedAttentionCache(
        geometry(window, target_dtype=target_dtype), 32, 4, "cuda"
    )
    for layer in range(3):
        data, scale = cache.ring_view(layer)
        raw = torch.randn(data.shape[0], 512, device="cuda", dtype=torch.bfloat16)
        values = quantize_fp8(raw)
        for dst, src in zip((data, scale), values):
            dst.copy_(src.view(torch.uint8))
    return cache


@pytest.mark.parametrize("dtype", ["auto", "fp8"])
def test_runtime_setting_reaches_draft_constructor(monkeypatch, dtype):
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
    config = SimpleNamespace(hf_config=args, dspark=DSparkConfig(draft_kv_dtype=dtype))
    monkeypatch.setattr(dspark, "make_v4_quant_config", lambda *a, **kw: None)
    monkeypatch.setattr(
        dspark, "build_attention_topology", lambda *a: [SimpleNamespace(layer_id=40)]
    )
    monkeypatch.setattr(dspark, "DraftBlock", lambda *a, **kw: nn.Identity())
    # This test isolates config propagation with weightless draft stages.
    # The real norm/weight binding is covered by test_dspark_context_norm_quant.
    monkeypatch.setattr(
        dspark.DeepseekV41DSpark, "_build_main_norm_quant", lambda self: None
    )
    monkeypatch.setattr(dspark, "RotaryEmbedding", lambda *a, **kw: nn.Identity())
    assert dspark.DeepseekV41DSpark(config).draft_kv_dtype == dtype


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


@pytest.mark.parametrize(
    "batch,width,window",
    [
        (0, 1, 8),
        (0, 5, 128),
        (1, 1, 128),
        (1, 6, 8),
        (8, 5, 128),
        (17, 6, 8),
        (32, 1, 8),
        (32, 5, 128),
        (65, 5, 8),
        (65, 6, 128),
    ],
)
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


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("supplied_step", [False, True])
def test_stages_share_one_live_csr_per_draft_round(monkeypatch, dtype, supplied_step):
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
        assert dtype != "bf16", "BF16 drafting must not construct CSR"
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
    history = quantize_fp8(raw) if dtype == "fp8" else raw
    context = dict.fromkeys((40, 41, 42), history)
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
        assert len(builds) == before + int(dtype != "bf16")
        if dtype != "bf16":
            assert all(step.csr is builds[-1] for step in seen)
            assert_csr_equal(seen[0].csr, csr_oracle(indices, 133))
            assert seen[0].csr is not previous
            previous = seen[0].csr
        else:
            assert all(step.csr is None for step in seen)


@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("width", [1, 5])
@pytest.mark.parametrize("window", [8, 128])
@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_draft_matches_dense_bidirectional_attention(
    monkeypatch, heads, width, window, target_dtype
):
    from atom.model_ops.deepseek_v41 import dspark

    torch.manual_seed(5901)
    cache = make_cache(window, target_dtype=target_dtype)
    slots = torch.tensor([3, 0, 2], device="cuda", dtype=torch.int32)
    layers = torch.tensor([2, 1], device="cuda")
    gathered = cache.read_window_planes(layers, slots)
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
    step = DraftStep(positions, indices, csr=draft_attention_csr(indices, window + 5))

    def unexpected(*args):
        pytest.fail("An attention stage rebuilt the shared CSR")

    monkeypatch.setattr(dspark, "draft_attention_csr", unexpected)
    out = draft_attention_fp8(q8, context, current, sink, step, 512**-0.5)
    expected = oracle(
        mxfp8(q8),
        mxfp8(context),
        mxfp8(current),
        anchors,
        window,
        sink,
    )
    torch.testing.assert_close(out, expected, rtol=0.03, atol=0.02)
    relative_l2 = (out.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.006
    for original, actual in zip(before, cache.window_sources):
        assert torch.equal(original, actual)


@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_gather_reset_and_graph_follow_live_layer_and_slot_ids(target_dtype):
    from atom.model_ops.attentions.deepseek_v41.planes import split_source

    cache = make_cache(target_dtype=target_dtype)
    layers = torch.tensor([1, 2, 1], device="cuda")
    slots = torch.tensor([3, 0, 2], device="cuda", dtype=torch.int32)
    cache.read_window_planes(layers, slots)
    cache.read_windows(layers, slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        raw = cache.read_window_planes(layers, slots)
        decoded = cache.read_windows(layers, slots)
    for layer_order, slot_order in (
        ([1, 2, 1], [3, 0, 2]),
        ([2, 0, 1], [1, 3, 0]),
    ):
        layers.copy_(torch.tensor(layer_order, device="cuda"))
        slots.copy_(torch.tensor(slot_order, device="cuda", dtype=torch.int32))
        graph.replay()
        assert raw[0].shape == (3, 3, 133, 512)
        assert raw[1].shape == (3, 3, 133, 16)
        assert (
            raw[0].untyped_storage().data_ptr() == raw[1].untyped_storage().data_ptr()
        )
        for i, layer in enumerate(layer_order):
            expected = tuple(
                p.view(4, 133, -1)[slots.long()] for p in cache.ring_view(layer)
            )
            for a, b in zip(raw, expected):
                assert torch.equal(a[i], b)
            torch.testing.assert_close(
                decoded[i],
                mxfp8(expected).bfloat16(),
                rtol=0,
                atol=0,
            )
    before = [p.clone() for p in cache.window_sources]
    cache._reset_state_slots(torch.tensor([1, 3], device="cuda", dtype=torch.int64))
    for layer in range(3):
        for plane in cache.ring_view(layer):
            per_slot = plane.view(4, 133, -1)
            assert not per_slot[[1, 3]].count_nonzero()
        for old, new in zip(split_source(before[layer], 512), cache.ring_view(layer)):
            assert torch.equal(
                old.view(4, 133, -1)[[0, 2]], new.view(4, 133, -1)[[0, 2]]
            )


@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_native_draft_graph_replays_live_requests_queries_and_csr(
    monkeypatch, target_dtype
):
    from atom.spec_decode.draft_graph import DraftGraph, StagedInput

    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "1")
    cache = make_cache(target_dtype=target_dtype)
    layers = torch.tensor([2], device="cuda")
    sink = torch.linspace(-1, 1, 64, device="cuda")
    captured = []

    def forward(bs, slots, anchors, query, kv):
        context = tuple(p[0] for p in cache.read_window_planes(layers, slots))
        pos, indices, _ = draft_step_indices(anchors, 133, 128, 5)
        csr = draft_attention_csr(indices, 133)
        captured.append(csr)
        q8, k8 = quantize_fp8(query), quantize_fp8(kv)
        out = draft_attention_fp8(
            q8, context, k8, sink, DraftStep(pos, indices, csr=csr), 512**-0.5
        )
        return out

    graph = DraftGraph(
        forward=forward,
        capture_epilogue=True,
        inputs={
            "slots": StagedInput(),
            "anchors": StagedInput(dtype=torch.int64),
            "query": StagedInput((5, 64, 512), torch.bfloat16),
            "kv": StagedInput((5, 512), torch.bfloat16),
        },
    ).bind(SimpleNamespace(max_num_seqs=4), "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.warmup(4)
    torch.cuda.current_stream().wait_stream(stream)
    build_count = len(captured)
    for slot_ids, anchors in [
        ([3, 0, 2], [268, 3, -1]),
        ([1], [0]),
        ([2, 3, 0, 1], [130, -1, 400, 1]),
    ]:
        # DraftGraph stages exact bucket shapes; unused requests are sentinels.
        live = len(slot_ids)
        slot_ids = slot_ids + [0] * (4 - live)
        anchors = anchors + [-1] * (4 - live)
        staged = graph.stage(
            4,
            {
                "slots": torch.tensor(slot_ids, device="cuda", dtype=torch.int32),
                "anchors": torch.tensor(anchors, device="cuda", dtype=torch.int64),
                "query": torch.randn(
                    len(slot_ids), 5, 64, 512, device="cuda", dtype=torch.bfloat16
                ),
                "kv": torch.randn(
                    len(slot_ids), 5, 512, device="cuda", dtype=torch.bfloat16
                ),
            },
        )
        out = graph.run(4, **staged)
        context = tuple(p[0] for p in cache.read_window_planes(layers, staged["slots"]))
        q8, k8 = quantize_fp8(staged["query"]), quantize_fp8(staged["kv"])
        expected = oracle(
            mxfp8(q8),
            mxfp8(context),
            mxfp8(k8),
            staged["anchors"],
            128,
            sink,
        )
        torch.testing.assert_close(out, expected, rtol=0.03, atol=0.02)
        _, indices, _ = draft_step_indices(staged["anchors"], 133, 128, 5)
        assert len(captured) == build_count
        assert_csr_equal(captured[-1], csr_oracle(indices, 133))
    assert graph.is_captured(4)


@pytest.mark.parametrize("accepted", range(1, 7))
@pytest.mark.parametrize("position", [127, 128, 265])
def test_rejected_draft_rows_are_invisible_next_round(single_rank, accepted, position):
    cache = PagedAttentionCache(geometry(), 32, 4, "cuda")
    torch.manual_seed(51)
    raw = torch.randn(position + 6, 512, device="cuda", dtype=torch.bfloat16)
    encoded = quantize_fp8(raw)
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
        torch.tensor([2], device="cuda"), torch.tensor([3], device="cuda")
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


@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_real_draft_forward_produces_quantized_queries_after_rope(target_dtype):
    from atom.model_ops.deepseek_v41.draft_block import rotate_rows
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
    from atom.models.deepseek_v41.dspark import DraftAttention

    rope = RotaryEmbedding(64, 512, base=10000).cuda()
    query = torch.randn(2, 5, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    cache = make_cache(target_dtype=target_dtype)
    history = tuple(
        p[0]
        for p in cache.read_window_planes(
            torch.tensor([2], device="cuda"),
            torch.tensor([0, 3], device="cuda"),
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
    expected = oracle(
        mxfp8(q8),
        mxfp8(history),
        mxfp8(k8),
        anchors,
        128,
        sink,
    )
    torch.testing.assert_close(out, expected, rtol=0.04, atol=0.03)
