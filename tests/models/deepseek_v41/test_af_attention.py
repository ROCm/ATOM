# SPDX-License-Identifier: MIT
"""AF replay must refresh request data and update KV exactly once."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("aiter")

from atom.config import CUDAGraphMode
from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
from atom.models.deepseek_v41.model import Block
from atom.utils.cuda_graph import CudagraphCaptureRunner, StableOutputs

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")


def _forward(
    graphs,
    attention,
    hidden,
    cache,
    step,
    rope,
    queries=None,
    *,
    zero_copy_hidden=False,
):
    """Use the same decorated model entry as the runtime splitting op."""
    if not getattr(step, "_test_prepared", False):
        # Isolated cache helpers allocate fresh request tensors. Emulate the
        # serving builder's fixed addresses once per view, before any layer.
        from dataclasses import replace

        if not hasattr(cache, "test_buffers"):
            cache.test_buffers = StableOutputs()

        def keep(name, value):
            return value if value is None else cache.test_buffers.deliver(name, value)

        for name in (
            "positions",
            "cu_seqlens_q",
            "slots",
            "batch_ids",
            "block_tables",
            "swa_replay_start",
            "swa_cached_until",
        ):
            setattr(step, name, keep(name, getattr(step, name)))
        for name in ("visible", "planned"):
            setattr(
                step,
                name,
                {k: keep((name, k), v) for k, v in getattr(step, name).items()},
            )
        step.plans = {
            r: replace(
                p,
                **{
                    n: keep((r, n), getattr(p, n))
                    for n in (
                        "compress_plan_gpu",
                        "write_plan_gpu",
                        "key_rope_positions_gpu",
                    )
                },
            )
            for r, p in step.plans.items()
        }

        def indptrs(scope, values):
            return {
                r: tuple(
                    keep((scope, r, i), t) if isinstance(t, torch.Tensor) else t
                    for i, t in enumerate(parts)
                )
                for r, parts in values.items()
            }

        step.indptrs = indptrs("main_ptrs", step.indptrs)
        if queries is not None:
            queries = replace(
                queries,
                index=replace(
                    step,
                    batch_ids=keep("query_ids", queries.index.batch_ids),
                    indptrs=indptrs("query_ptrs", queries.index.indptrs),
                ),
                **{
                    n: keep(("query", n), getattr(queries, n))
                    for n in ("rows", "positions", "ptr_rows")
                },
            )
        if hasattr(cache, "prepare_attention"):
            cache.candidate_count = getattr(
                getattr(attention, "indexer", None), "topk_blocks", 0
            )
            cache.prepare_attention(step, [attention.spec])
        step._test_prepared = True
    if hasattr(cache, "prepare_attention"):
        cache.prepare_attention(step, [attention.spec])
        backend_key = cache.graph_layout(step, queries)
    else:
        backend_key = (step.width, step.cu_seqlens_q.numel(), step.max_q_len)
    metadata = SimpleNamespace(
        cache=cache, step=step, queries=queries, dummy=False, cudagraph_key=backend_key
    )
    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        in_hipgraph=False,
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=metadata,
    )
    block = SimpleNamespace(attn=attention, layer_name=f"test.af.{id(graphs)}")
    return Block.attention_forward(
        block,
        hidden,
        cache,
        step,
        rope,
        queries,
        piecewise=True,
        capture=True,
        forward_context=context,
        runner=graphs,
        zero_copy_inputs=("normed",) if zero_copy_hidden else (),
    )


@cuda
def test_refreshes_metadata_and_applies_state_effect_once_per_call():
    device = "cuda"
    cache = SimpleNamespace(pool=torch.zeros(1, device=device))
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    class Core:
        spec = LayerAttentionSpec(0, 0, AttentionMode.WINDOW)

        def __call__(self, hidden, scale, cache, step, rope, queries=None):
            cache.pool.add_(1)
            return hidden + step.positions[None, :, None] + step.slots[0]

    graphs = CudagraphCaptureRunner(token_dim=1)
    for i, (lengths, position, slot) in enumerate(
        [
            ([4], 0, 0),
            ([4], 10, 2),
            ([2, 2], 20, 1),
            ([4], 30, 3),
            ([1, 1, 1, 1], 40, 0),
        ],
        1,
    ):
        cu = np.concatenate(([0], np.cumsum(lengths)))
        step = BatchStep(
            tuple(
                RequestSpan(j, position, int(cu[j]), n, slot)
                for j, n in enumerate(lengths)
            ),
            torch.arange(position, position + 4, device=device, dtype=torch.int32),
            torch.tensor(cu, device=device, dtype=torch.int32),
            torch.full((len(lengths),), slot, device=device, dtype=torch.int32),
            torch.repeat_interleave(
                torch.arange(len(lengths), device=device),
                torch.tensor(lengths, device=device),
            ),
            torch.zeros(len(lengths), 1, device=device, dtype=torch.int32),
            scheduled=4,
            max_q_len=max(lengths),
        )
        hidden = torch.full((1, 4, 2), float(i), device=device)
        step.indptrs = {0: (step.cu_seqlens_q, None, 0)}
        actual = _forward(graphs, Core(), hidden, cache, step, None)
        expected = hidden + step.positions[None, :, None] + slot
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert cache.pool.item() == i
    assert len(graphs._graphs) == 3
    assert graphs.release_graphs() == 3


@cuda
@pytest.mark.parametrize("subset", [False, True])
def test_real_window_attention_matches_eager_outputs_and_kv_across_steps(subset):
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
    from atom.models.deepseek_v41.attention import Attention
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
    from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step

    torch.manual_seed(2107)
    geo = V41PoolGeometry(
        2, ((0, 1),), 32, 8, 512, 32, layer_ratios=(0, 1), index_topk=4
    )
    caches = [PagedAttentionCache(geo, 8, 4, "cuda") for _ in range(2)]
    for cache in caches:
        cache.backing.zero_()
    attn = Attention.__new__(Attention)
    torch.nn.Module.__init__(attn)
    attn.spec = LayerAttentionSpec(1, 0, AttentionMode.WINDOW)
    attn.compressor = attn.indexer = None
    attn.compress_stream = attn.index_stream = None
    attn.heads, attn.head_dim = 2, 512
    attn.softmax_scale = 512**-0.5
    attn.attn_sink = torch.zeros(2, device="cuda")
    attn.project_qkv = lambda x, scale=None: (x, x)
    attn.qk_norm = lambda q, k: (q, torch.ones_like(q[..., :2]), k)
    attn.wq_b = lambda x, x_scale=None: x.repeat(1, 1, 2)
    attn._project_out = lambda out, rope, pos: out.mean(dim=-2)
    rope = RotaryEmbedding(64, 128, base=10000).to("cuda")
    graphs = CudagraphCaptureRunner(token_dim=1)
    layouts = [
        [(0, 0, 4)],
        [(0, 4, 4)],
        [(0, 8, 4)],
        [(1, 0, 4)],
        [(1, 4, 2), (2, 0, 2)],
        [(0, 12, 4)],
        [(0, 16, 1)],
        [(0, 17, 1), (1, 6, 1)],
        [(0, 18, 1)],
    ]
    for layout in layouts:
        requests = []
        offset = 0
        for slot, start, n in layout:
            requests.append(PagedRequest(slot, start, offset, n, slot, (slot,)))
            offset += n
        steps = [begin_step(c, requests) for c in caches]
        hidden = torch.randn(1, offset, 512, device="cuda", dtype=torch.bfloat16)
        queries = [None, None]
        if subset:
            from dataclasses import replace

            from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs
            from atom.models.deepseek_v41.bounded_replay import QueryRows

            for i, (cache, step) in enumerate(zip(caches, steps)):
                rows = step.cu_seqlens_q[1:].long() - 1
                bids = torch.full_like(step.batch_ids, -1)
                bids[rows] = step.batch_ids[rows]
                index = replace(step, batch_ids=bids)
                index.indptrs = fill_step_indptrs(index, geo, cache.indptr_buffers)
                queries[i] = QueryRows(
                    rows,
                    step.positions[rows],
                    index,
                    torch.cat((rows, rows.new_tensor([step.width]))),
                )
        expected = attn(hidden, None, caches[0], steps[0], rope, queries=queries[0])
        actual = _forward(
            graphs, attn, hidden, caches[1], steps[1], rope, queries=queries[1]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(caches[1].backing, caches[0].backing, rtol=0, atol=0)
    graphs.release_graphs()


@cuda
@pytest.mark.parametrize("packed_scores", [False, True])
def test_indexed_graph_matches_eager_compression_selection_and_reindex(
    single_rank, packed_scores
):
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
    from atom.models.deepseek_v41.attention import Attention, Indexer
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
    from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step

    torch.manual_seed(2108)
    geo = V41PoolGeometry(
        2,
        ((0, 2),),
        128,
        8,
        512,
        128,
        layer_ratios=(0, 2),
        index_topk=64,
        index_block_rows=8,
        index_fp4=True,
    )
    caches = [PagedAttentionCache(geo, 8, 4, "cuda") for _ in range(2)]
    for cache in caches:
        cache.backing.zero_()
    if packed_scores:
        from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace

        for cache in caches:
            cache.workspace = ScoreWorkspace(geo, 130, 2, "cuda", pages=8)
            # Force several host-planned bands on the long prefill.
            cache.workspace.capacity = 4096
            cache.workspace._logits = torch.empty(4096, device="cuda")
            cache.workspace._maxima = torch.empty(512, device="cuda")
    rope = RotaryEmbedding(64, 512, base=10000).to("cuda")
    specs = [
        LayerAttentionSpec(0, 2, AttentionMode.FULL, 0, 0, 0),
        LayerAttentionSpec(1, 2, AttentionMode.REINDEX, 0, 1, 0),
    ]
    layers = []
    for spec in specs:
        attn = Attention.__new__(Attention)
        torch.nn.Module.__init__(attn)
        attn.spec = spec
        attn.compress_stream = attn.index_stream = None
        attn.heads, attn.head_dim = 2, 512
        attn.softmax_scale = 512**-0.5
        attn.attn_sink = torch.zeros(2, device="cuda")
        attn.project_qkv = lambda x, scale=None: (x, x)
        attn.qk_norm = lambda q, k: (q, torch.ones_like(q[..., :2]), k)
        attn.wq_b = lambda x, x_scale=None: x.repeat(1, 1, 2)
        attn._project_out = lambda out, rope, pos: out.mean(dim=-2)
        # Projection stubs isolate graph/state correctness; compression,
        # FP4 index writes, scoring, top-k, CSR and attention are real kernels.
        indexer = Indexer.__new__(Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.spec = spec
        indexer.heads, indexer.head_dim = 32, 128
        indexer.topk, indexer.block_size, indexer.topk_blocks = 64, 8, 16
        indexer.wq_b = lambda x, x_scale=None: x.repeat(1, 1, 8)
        indexer.weights_proj = lambda x: x[..., :32] + 0
        indexer.wk = lambda x: x[..., :128] + 0
        indexer.k_norm = SimpleNamespace(
            weight=torch.ones(128, device="cuda", dtype=torch.bfloat16), eps=1e-6
        )
        attn.indexer = indexer
        attn.compressor = None
        if spec.layer_id == 0:
            attn.compressor = SimpleNamespace(
                ratio=2,
                ape=torch.zeros(2, 512, device="cuda"),
                norm=SimpleNamespace(
                    weight=torch.ones(512, device="cuda", dtype=torch.bfloat16),
                    eps=1e-6,
                ),
                project=lambda x: (x.float(), -x.float()),
            )
        layers.append(attn)
    graphs = [CudagraphCaptureRunner(token_dim=1) for _ in layers]
    layouts = [
        [(0, 0, 130)],
        [(0, 130, 6)],
        [(0, 136, 6)],
        [(1, 0, 130)],
        [(1, 130, 3), (0, 142, 3)],
        [(0, 145, 6)],
        [(0, 151, 1)],
        [(0, 152, 1)],
        [(0, 153, 6)],
    ]
    # The late FULL layer receives no plans: compression already ran on
    # the entire early-stage input. Exercise both its miss and replay.
    layouts += [[(0, 153, 6)], [(0, 153, 6)]]
    for iteration, layout in enumerate(layouts):
        requests, offset = [], 0
        for slot, start, count in layout:
            requests.append(
                PagedRequest(slot, start, offset, count, slot, (slot * 2, slot * 2 + 1))
            )
            offset += count
        steps = [begin_step(cache, requests) for cache in caches]
        for cache, step in zip(caches, steps):
            step.positions = step.positions.long()
            if iteration >= len(layouts) - 2:
                step.plans = {}
            if packed_scores:
                visible = step.visible[2].cpu().numpy()
                offsets = np.empty_like(visible)
                blocks = np.empty_like(visible)
                bands = cache.workspace.pack(visible, offsets, blocks)
                step.planned = {
                    "v41_score_offsets_2": torch.from_numpy(offsets).cuda(),
                    "v41_score_block_offsets_2": torch.from_numpy(blocks).cuda(),
                }
                step.planned_host = {"v41_score_bands_2": bands}
        hidden = torch.randn(1, offset, 512, device="cuda", dtype=torch.bfloat16)
        for attn, graph in zip(layers, graphs):
            expected = attn(hidden, None, caches[0], steps[0], rope)
            actual = _forward(graph, attn, hidden, caches[1], steps[1], rope)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            layer = attn.spec.layer_id
            torch.testing.assert_close(
                steps[1].selected[layer], steps[0].selected[layer], rtol=0, atol=0
            )
            if layer in steps[0].candidates:
                torch.testing.assert_close(
                    steps[1].candidates[layer],
                    steps[0].candidates[layer],
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                caches[1].backing, caches[0].backing, rtol=0, atol=0
            )
    assert all(len(graph._graphs) < len(layouts) for graph in graphs)
    for graph in graphs:
        graph.release_graphs()


@cuda
@pytest.mark.parametrize("subset", [False, True])
def test_indexed_replay_publishes_current_outputs_and_updates_state_once(subset):
    from dataclasses import replace

    from atom.models.deepseek_v41.bounded_replay import QueryRows
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    cache = SimpleNamespace(pool=torch.zeros(1, device="cuda"))

    class Core:
        spec = LayerAttentionSpec(2, 2, AttentionMode.REINDEX, 0, 0, 0)

        def __call__(self, hidden, scale, cache, step, rope, queries=None):
            if queries is not None:
                assert queries.index.selected is step.selected
                assert queries.index.candidates is step.candidates
                assert queries.index.group_indices is step.group_indices
            cache.pool.add_(1)
            selected = hidden + step.candidates[0] + step.visible[2][None, :, None]
            step.selected[2].copy_(selected)
            step.candidates[2].copy_(selected + 1)
            step.group_indices[2][0].copy_(selected + 2)
            step.window_rows.copy_(selected + 3)
            return selected + step.positions[None, :, None] + step.slots[0]

    graph = CudagraphCaptureRunner(token_dim=1)
    graph.max_entries = 2
    for i, count in enumerate([4, 4, 6, 4, 8, 6], 1):
        step = BatchStep(
            (RequestSpan(i, i * 10, 0, count, i % 2),),
            torch.arange(count, device="cuda") + i * 10,
            torch.tensor([0, count], device="cuda", dtype=torch.int32),
            torch.tensor([i % 2], device="cuda", dtype=torch.int32),
            torch.zeros(count, device="cuda", dtype=torch.int32),
            torch.tensor([[i]], device="cuda", dtype=torch.int32),
            scheduled=count,
            max_q_len=count,
            visible={2: torch.full((count,), i, device="cuda", dtype=torch.int32)},
            candidates={0: torch.full((1, count, 2), i * 2, device="cuda")},
            indptrs={2: (torch.arange(count + 1, device="cuda"), None, 2)},
        )
        hidden = torch.full((1, count, 2), i * 3.0, device="cuda")
        upstream = hidden + 100
        step.selected[0] = upstream
        queries = None
        if subset:
            rows = torch.arange(count, device="cuda")
            queries = QueryRows(
                rows,
                step.positions,
                replace(step, batch_ids=step.batch_ids.clone()),
                torch.arange(count + 1, device="cuda"),
            )
        if not hasattr(cache, "results"):
            cache.results = StableOutputs()

        def result(name, count=count, hidden=hidden):
            return cache.results.empty(
                (count, name), hidden.shape, dtype=hidden.dtype, device=hidden.device
            )

        step.candidates[0] = cache.results.deliver(
            (count, "source"), step.candidates[0]
        )
        step.selected[2] = result("selected")
        step.candidates[2] = result("candidates")
        step.group_indices[2] = (result("group"),)
        step.window_rows = result("window")
        actual = _forward(graph, Core(), hidden, cache, step, None, queries)
        assert step.selected[0] is upstream
        selected = hidden + i * 3
        torch.testing.assert_close(
            actual, selected + step.positions[None, :, None] + i % 2, rtol=0, atol=0
        )
        torch.testing.assert_close(step.selected[2], selected, rtol=0, atol=0)
        torch.testing.assert_close(step.candidates[2], selected + 1, rtol=0, atol=0)
        torch.testing.assert_close(
            step.group_indices[2][0], selected + 2, rtol=0, atol=0
        )
        torch.testing.assert_close(step.window_rows, selected + 3, rtol=0, atol=0)
        assert cache.pool.item() == i
    assert graph.release_graphs() == 2


@cuda
def test_direct_producer_input_and_consumer_output_with_reused_step():
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    class Core:
        spec = LayerAttentionSpec(0, 0, AttentionMode.WINDOW)

        def __call__(self, hidden, scale, cache, step, rope, queries=None):
            cache.pool.add_(1)
            return hidden + step.positions[None, :, None] + step.slots[0]

    step = BatchStep(
        (RequestSpan(0, 0, 0, 4, 0),),
        torch.zeros(4, device="cuda", dtype=torch.int64),
        torch.tensor([0, 4], device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.zeros(4, device="cuda", dtype=torch.int32),
        torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        scheduled=4,
        max_q_len=4,
    )
    step.indptrs = {0: (step.cu_seqlens_q, None, 0)}
    cache = SimpleNamespace(pool=torch.zeros(1, device="cuda"))
    source = torch.zeros(1, 4, 2, device="cuda")
    hidden = torch.empty_like(source)
    consumer_output = torch.empty_like(source)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        producer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(producer):
            hidden.copy_(source * 2)
    torch.cuda.current_stream().wait_stream(stream)
    graphs = CudagraphCaptureRunner(token_dim=1)
    for i in range(1, 5):
        # Reuse the exact BatchStep AND device allocations with changed values.
        step.begin_forward()
        step.positions.fill_(i * 10)
        step.slots.fill_(i % 2)
        source.fill_(i)
        producer.replay()
        out = _forward(
            graphs,
            Core(),
            hidden,
            cache,
            step,
            None,
            zero_copy_hidden=True,
        )
        if i == 1:
            destination = out
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                consumer = torch.cuda.CUDAGraph()
                with torch.cuda.graph(consumer):
                    consumer_output.copy_(destination + 100)
            torch.cuda.current_stream().wait_stream(stream)
        assert out.data_ptr() == destination.data_ptr()
        consumer.replay()
        torch.testing.assert_close(consumer_output, source * 2 + i * 10 + i % 2 + 100)
        assert cache.pool.item() == i
    assert len(graphs._graphs) == 1
    # A changed producer address is a distinct graph, never a stale replay.
    moved = hidden + 7
    step.begin_forward()
    out = _forward(
        graphs,
        Core(),
        moved,
        cache,
        step,
        None,
        zero_copy_hidden=True,
    )
    torch.testing.assert_close(
        out, moved + step.positions[None, :, None] + step.slots[0]
    )
    assert cache.pool.item() == 5
    assert len(graphs._graphs) == 2
    graphs.release_graphs()
