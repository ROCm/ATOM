# SPDX-License-Identifier: MIT
"""AF replay must refresh request data and update KV exactly once."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("aiter")

from atom.config import CUDAGraphMode
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
        cache.candidate_count = getattr(
            getattr(attention, "indexer", None), "topk_blocks", 0
        )
        step.layer_specs = cache.topology or (attention.spec,)
        cache.update_step(step, fixed_results=True)
        step._test_prepared = True
    backend_key = cache.graph_layout(step, queries)
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
        zero_copy_inputs=(),
    )


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
        3,
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
        LayerAttentionSpec(1, 2, AttentionMode.REINDEX, 0, 1, 0, 1, 2),
        LayerAttentionSpec(2, 2, AttentionMode.REUSE, 0, 1, 0, 1, 2),
    ]
    for cache in caches:
        cache.topology = tuple(specs)
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
    # A producer may exhaust its graph budget while consumers still capture/replay.
    # Its eager fallback must keep writing their prebound candidate buffer.
    graphs[0].max_entries = 1
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
            if layer == 1:
                # Only this selection crosses a region boundary (to REUSE).
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
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_backend_shared_results_are_bounded_by_capacity(decode, packed):
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
    from tests.attentions.deepseek_v41.helpers import (
        PagedRequest,
        begin_step,
        metadata_buffers,
    )

    geo = V41PoolGeometry(
        4,
        ((0, 2),),
        128,
        8,
        512,
        128,
        layer_ratios=(0, 2),
        index_topk=4,
        index_block_rows=8,
        index_fp4=True,
        packed=packed,
    )
    specs = (
        LayerAttentionSpec(0, 2, AttentionMode.FULL, 0, 0, 0),
        LayerAttentionSpec(1, 2, AttentionMode.REINDEX, 0, 1, 0, 1, 2),
        LayerAttentionSpec(2, 2, AttentionMode.REUSE, 0, 1, 0, 1, 2),
        LayerAttentionSpec(3, 2, AttentionMode.REINDEX, 0, 3, 0),
    )
    cache = PagedAttentionCache(
        geo, 4, 16, "cuda", max_tokens=16, topology=specs, candidate_count=2
    )
    buffers = metadata_buffers(16, 16, 4, "cuda", geo)
    footprints = []
    pointers = []
    for width in (16, 1, 3, 5, 9, 2, 15, 4):
        requests = (
            [PagedRequest(i, 64, i, 1, i, (0, 1)) for i in range(width)]
            if decode
            else [PagedRequest(0, 64, 0, width, 0, (0, 1))]
        )
        step = begin_step(
            cache,
            requests,
            buffers=buffers,
            running_bs=len(requests),
            running_tokens=width,
            is_prefill=not decode,
        )
        cache.update_step(step, fixed_results=True)
        blocks = cache.candidate_blocks(step, 0, 2, layer=3)
        assert blocks.table.shape == (width, 2)
        assert blocks.context.shape == (width,)
        pointers.append(
            (
                step.selected[1].data_ptr(),
                blocks.table.data_ptr(),
                blocks.context.data_ptr(),
            )
        )
        if decode:
            cache.attention_indices(specs[2], step)
            assert all(
                values[0].is_contiguous() for values in step.group_indices.values()
            )
            assert step.group_indices[1][0].shape[0] == (2 if packed else 1)
        footprints.append(
            sum(
                t.untyped_storage().nbytes()
                for t in cache.result_buffers._slots.values()
            )
        )
    assert len(set(pointers)) == 1
    assert len(set(footprints)) == 1
