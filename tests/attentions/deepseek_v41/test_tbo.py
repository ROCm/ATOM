# SPDX-License-Identifier: MIT
"""Prefill microbatches preserve histories, cache plans and cross-layer state."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.attentions.deepseek_v41.backend import DeepseekV41MetadataBuilder
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.utils.forward_context import Context, ForwardContext, _forward_context_local
from atom.utils.tbo.ubatch_splitting import UBatchSlice, _split_prefill_token_midpoint
from atom.utils.tbo.ubatch_wrapper import UBatchWrapper
from tests.attentions.deepseek_v41.helpers import metadata_buffers


def attach_ubatch_buffers(builder, max_tokens):
    """What `DeepseekV41MetadataBuilder.__init__` sets up under prefill TBO,
    for a builder assembled without it: the `ub{i}_` step buffers in the
    runner's forward_vars and each microbatch's indptrs."""
    builder.max_num_batched_tokens = max_tokens
    builder.model_runner.forward_vars.update(builder._ubatch_step_buffers())
    builder._ubatch_indptrs = [
        {
            ratio: tuple(
                torch.empty(max_tokens + 1, dtype=torch.int32, device=builder.device)
                for _ in range(2)
            )
            for ratio in builder.geometry.layer_ratios
        }
        for _ in range(builder._NUM_TBO_UBATCHES)
    ]


def make_parent(device, lengths=(10, 4), starts=(3, 8)):
    geo = V41PoolGeometry(
        4,
        ((0, 2), (2, 1)),
        32,
        4,
        512,
        32,
        layer_ratios=(0, 1, 2),
        index_topk=4,
    )
    cache = PagedAttentionCache(geo, 8, 4, device, max_tokens=32)
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.device, builder.geometry, builder.block_size = device, geo, 32
    builder.cache = cache
    builder.model_runner = SimpleNamespace(
        forward_vars=metadata_buffers(4, 32, 4, device, geo)
    )
    attach_ubatch_buffers(builder, 32)
    batch = SimpleNamespace(
        is_dummy_run=False,
        state_slots_committed=[3, 1][: len(lengths)],
        req_ids=tuple(range(len(lengths))),
        num_scheduled_tokens=lengths,
        context_lens=tuple(a + b for a, b in zip(starts, lengths)),
        block_tables=((0, 1, 2, 3), (4, 5, 6, 7))[: len(lengths)],
        total_tokens_num=sum(lengths),
        total_seqs_num=len(lengths),
    )
    # The runner publishes the query prefix before the public builder entry.
    cu = builder.model_runner.forward_vars["cu_seqlens_q"]
    cu.np[0] = 0
    for i, length in enumerate(lengths):
        cu.np[i + 1] = cu.np[i] + length
    cu.copy_to_gpu(len(lengths) + 1)
    parent, _ = builder.prepare_prefill(batch, len(lengths))
    parent.engram_embeddings = {
        layer: torch.arange(sum(lengths) * 4, device=device).view(1, -1, 4) + layer
        for layer in (1, 3)
    }
    return builder, parent


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cut", [3, 7, 10, 11])
def test_prefill_slices_keep_absolute_positions_and_independent_plans(device, cut):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    builder, parent = make_parent(device)
    steps = []
    before = {
        name: value.gpu.clone()
        for name, value in builder.model_runner.forward_vars.items()
    }
    for i, (start, end, rs) in enumerate(
        (
            (0, cut, slice(0, 1 if cut <= 10 else 2)),
            (cut, 14, slice(0 if cut < 10 else 1, 2)),
        )
    ):
        part = UBatchSlice(rs, slice(start, end))
        child = builder.build_ubatch_prefill_metadata(
            parent, part, rs.stop - rs.start, i
        )
        steps.append(child.step)
        assert child.cache is parent.cache
        assert child.step.swa_replay_start is parent.step.swa_replay_start
        assert child.step.width == end - start
        torch.testing.assert_close(
            child.step.positions, parent.step.positions[start:end]
        )
        assert child.step.slots.tolist() == parent.step.slots[rs].tolist()
        torch.testing.assert_close(
            child.step.block_tables, parent.step.block_tables[rs]
        )
        assert child.step.cu_seqlens_q.tolist() == (
            [s.offset for s in child.step.requests] + [end - start]
        )
        for layer, rows in parent.engram_embeddings.items():
            torch.testing.assert_close(
                child.engram_embeddings[layer], rows[:, start:end]
            )
            assert (
                child.engram_embeddings[layer].untyped_storage().data_ptr()
                == rows.untyped_storage().data_ptr()
            )
        for ratio, plan in child.step.plans.items():
            expected = [
                [span.offset + j, bid, span.position + j]
                for bid, span in enumerate(child.step.requests)
                for j in range(span.length)
                if (span.position + j + 1) % ratio == 0
            ]
            assert plan.compress_plan_gpu[: plan.num_compress, :3].tolist() == expected
            torch.testing.assert_close(
                child.step.visible[ratio], ((child.step.positions + 1) // ratio).int()
            )
    # The parent's step buffers are untouched; the microbatches write only
    # their own `ub{i}_` copies.
    for name, value in builder.model_runner.forward_vars.items():
        if not name.startswith(("ub0_", "ub1_")):
            torch.testing.assert_close(value.gpu, before[name], rtol=0, atol=0)
    for ratio in steps[0].plans:
        assert (
            steps[0].plans[ratio].compress_plan_gpu.data_ptr()
            != steps[1].plans[ratio].compress_plan_gpu.data_ptr()
        )
    for ratio in steps[0].indptrs:
        assert (
            steps[0].indptrs[ratio][0].data_ptr()
            != steps[1].indptrs[ratio][0].data_ptr()
        )
        assert (
            steps[0].indptrs[ratio][0].data_ptr()
            != parent.step.indptrs[ratio][0].data_ptr()
        )
    for field in ("selected", "candidates", "tiles"):
        getattr(steps[0], field)[2] = torch.tensor([17])
        steps[1].begin_forward()
        assert getattr(steps[0], field)[2].item() == 17
    first = steps[1].requests[0]
    plan = steps[1].plans[2]
    if plan.num_compress and first.length >= 2:
        assert plan.compress_plan_gpu[0, 3].item() == first.position % 2
    assert parent.cache.pending is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefill_children_publish_independent_packed_score_plans(device):
    """Main's packed scorer must see each child's absolute rows and bands."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    from dataclasses import replace

    from atom.model_ops.attentions.deepseek_v41.score_planner import (
        ScorePlanner,
        score_layout,
    )
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace

    builder, parent = make_parent(device)
    workspace = ScoreWorkspace(
        replace(builder.geometry, index_fp4=True, index_dim=128, index_block_rows=8),
        32,
        4,
        device,
        pages=8,
    )
    builder.add_step_planner(ScorePlanner(workspace, (1, 2), 32))
    attach_ubatch_buffers(builder, 32)
    parent.cache.workspace = workspace
    parts = (
        UBatchSlice(slice(0, 1), slice(0, 7)),
        UBatchSlice(slice(0, 2), slice(7, 14)),
    )
    first = builder.build_ubatch_prefill_metadata(parent, parts[0], 1, 0)
    snapshots = {name: rows.clone() for name, rows in first.step.planned.items()}
    second = builder.build_ubatch_prefill_metadata(parent, parts[1], 2, 1)
    for name, expected in snapshots.items():
        torch.testing.assert_close(first.step.planned[name], expected)
        assert (
            first.step.planned[name].data_ptr() != second.step.planned[name].data_ptr()
        )
    for child, part in zip((first, second), parts):
        for ratio in (1, 2):
            layout = score_layout(child.step, ratio)
            expected = ((parent.step.positions[part.token_slice] + 1) // ratio).int()
            torch.testing.assert_close(layout.visible, expected)
            spans = (expected + 63) // 64 * 64
            offsets = torch.cumsum(spans, dim=0) - spans
            torch.testing.assert_close(layout.offsets, offsets.int())
            torch.testing.assert_close(
                layout.block_offsets, (offsets // workspace.block_rows).int()
            )
            assert layout.bands == ((0, 7),)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_kv_release_keeps_persistent_ubatch_buffers_for_rebind(device):
    """KV views can die while fixed-address execution metadata is reused."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    import gc
    import weakref

    builder, parent = make_parent(device)
    buffers = builder._ubatch_buffers(0)
    indptrs = builder._ubatch_indptrs[0]
    pointers = {name: buf.gpu.data_ptr() for name, buf in buffers.items()}
    cache_ref = weakref.ref(builder.cache)
    builder.copies = object()
    del parent
    builder.release_kv_pools()
    gc.collect()
    assert cache_ref() is None
    assert builder.cache is builder.copies is None
    assert builder._ubatch_indptrs[0] is indptrs
    assert {
        name: buf.gpu.data_ptr() for name, buf in builder._ubatch_buffers(0).items()
    } == pointers

    builder.cache = PagedAttentionCache(builder.geometry, 8, 4, device, max_tokens=32)
    batch = SimpleNamespace(
        is_dummy_run=False,
        state_slots_committed=[3, 1],
        req_ids=(0, 1),
        num_scheduled_tokens=(10, 4),
        context_lens=(13, 12),
        block_tables=((0, 1, 2, 3), (4, 5, 6, 7)),
        total_tokens_num=14,
        total_seqs_num=2,
    )
    parent, _ = builder.prepare_prefill(batch, 2)
    part = UBatchSlice(slice(0, 2), slice(7, 14))
    child = builder.build_ubatch_prefill_metadata(parent, part, 2, 0)
    torch.testing.assert_close(child.step.positions, parent.step.positions[7:14])
    assert child.cache is builder.cache
    assert {
        name: buf.gpu.data_ptr() for name, buf in builder._ubatch_buffers(0).items()
    } == pointers


def test_parent_engram_join_runs_when_a_microbatch_fails():
    from atom.model_ops.engram.device.staging import EngramStagedRows

    events = []

    class Rows(EngramStagedRows):
        def __init__(self):
            dict.__init__(self)

        def stage(self, *, tbo=False):
            assert tbo
            events.append("stage")

        def join(self):
            events.append("join")

    parent = SimpleNamespace(engram_embeddings=Rows())
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    with (
        pytest.raises(RuntimeError, match="failed child"),
        builder.ubatch_forward(parent),
    ):
        raise RuntimeError("failed child")
    assert events == ["stage", "join"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_narrowed_tile_rows_stay_compact_and_within_their_microbatch():
    """A child table packs rows at its own width, not the allocation's.

    `tile_slice` hands out a region whose rows are as wide as the workspace
    was built for, but a request whose live `block_tables` is narrower fills
    only part of each. The kernel is handed `out.stride(0)`, so it writes at
    the same compact pitch the view reads at -- the unused tail of a row is
    simply never addressed, and the rows stay inside the slice they were cut
    from.
    """
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace
    from atom.model_ops.deepseek_v41.unit_table import unit_table
    from tests.models.deepseek_v41.reference_unit_table import unit_table_reference

    units, alloc_columns, max_tokens = 2, 8, 8
    geometry = SimpleNamespace(
        index_fp4=False,
        index_block_rows=8,
        owners=((0, 1),),
        index_blocks_per_page=lambda ratio: units,
        rows_per_page=lambda ratio: 16,
    )
    workspace = ScoreWorkspace(geometry, max_tokens, alloc_columns, "cuda")

    torch.manual_seed(1409)
    live_columns, tokens = 3, 4
    assert live_columns < alloc_columns  # armed: the row has an unused tail
    batch_ids = torch.arange(tokens, dtype=torch.int32, device="cuda")
    first_pages = torch.randint(
        0, 999, (tokens, live_columns), dtype=torch.int32, device="cuda"
    )
    second_pages = first_pages + 500

    halves = [slice(0, tokens), slice(tokens, 2 * tokens)]
    tables = [
        unit_table(
            pages,
            batch_ids,
            units,
            workspace=workspace.tile_slice(half),
            ratio=1,
        )
        for pages, half in zip((first_pages, second_pages), halves)
    ]

    # Each child names its own requests' pages, so neither wrote over the
    # other's rows, and each matches the oracle element for element.
    for table, pages in zip(tables, (first_pages, second_pages)):
        assert table.stride(0) == live_columns * units
        assert torch.equal(table, unit_table_reference(pages, batch_ids, units))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_prefill_microbatches_keep_workspace_tiles_across_layer_interleaving():
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace

    builder, parent = make_parent("cuda")
    workspace = ScoreWorkspace(builder.geometry, 32, 4, "cuda")
    parent.cache.workspace = workspace
    parts = _split_prefill_token_midpoint(2, [10, 4], 2, None)
    children = [
        builder.build_ubatch_prefill_metadata(
            parent, part, part.request_slice.stop - part.request_slice.start, i
        )
        for i, part in enumerate(parts)
    ]
    for ratio in (1, 2):
        first = parent.cache.unit_tiles(children[0].step, ratio)
        expected = first.clone()
        second = parent.cache.unit_tiles(children[1].step, ratio)
        # The first worker revisits this ratio after its partner's attention.
        # Its memoized table must still name its own requests' pages.
        again = parent.cache.unit_tiles(children[0].step, ratio)
        assert again is first
        torch.testing.assert_close(again, expected, rtol=0, atol=0)
        assert first.data_ptr() != second.data_ptr()
        for table in (first, second):
            assert (
                table.untyped_storage().data_ptr()
                == workspace._tiles[ratio].untyped_storage().data_ptr()
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_real_tbo_workers_share_one_uva_prefetch_and_keep_cross_layer_state(
    monkeypatch,
):
    from atom.model_ops.engram.device.hashing import engram_row_indices_reference
    from atom.utils.forward_context import get_forward_context
    from atom.utils.tbo.ubatching import (
        tbo_switch_to_compute_sync,
        tbo_yield_and_switch_from_compute_to_comm,
    )
    from tests.model_ops.engram.test_hash_bounds import build, tiny_config
    from tests.model_ops.engram.test_overlap import (
        fill_step,
        make_batch,
        make_staging,
        make_step,
    )

    builder, parent = make_parent("cuda", lengths=(9, 5))
    mapping = build(tiny_config())
    staging, _backing = make_staging(mapping)
    engram_step = make_step(staging.uva.hash_tables, 32, verify=False)
    calls = []
    start = staging.start
    join = staging.join
    monkeypatch.setattr(
        staging, "start", lambda width: (calls.append("stage"), start(width))[-1]
    )
    monkeypatch.setattr(staging, "join", lambda: (calls.append("join"), join())[-1])

    class Model(torch.nn.Module):
        def forward(self, ids, positions, inputs_embeds=None):
            ctx = get_forward_context()
            step = ctx.attn_metadata.step
            step.begin_forward()
            step.selected[0] = ids.clone()
            # Alternate workers twice while the earlier layer's state lives.
            for _ in range(2):
                tbo_yield_and_switch_from_compute_to_comm()
                copy = ids.clone()
                tbo_switch_to_compute_sync()
            rows = ctx.attn_metadata.engram_embeddings.get(mapping.config.layer_ids[-1])
            return rows[0].float() + (copy + step.selected[0]).float()[:, None]

    wrapper = UBatchWrapper(Model(), builder)
    ids = torch.arange(14, device="cuda", dtype=torch.int32)
    slices = _split_prefill_token_midpoint(2, [9, 5], 2, None)
    for seed in (7, 31):
        _, tokens, histories, masks = make_batch(mapping, [9, 5], seed)
        fill_step(engram_step, tokens, histories, masks, starts=(3, 8))
        # Staging snapshots before advancing the real cursor. Both workers
        # must still read the original history through their parent lookup.
        rows = staging.prepare(engram_step, 14)
        layer = mapping.config.layer_ids[-1]
        indices = engram_row_indices_reference(mapping, layer, tokens, histories, masks)
        expected = (
            staging.host.prefetcher._tables[layer]
            ._tensor[torch.as_tensor(indices)]
            .reshape(14, -1)
            .to(device="cuda", dtype=torch.float32)
        )
        expected = expected + ids.float()[:, None] * 2
        calls.clear()
        parent.engram_embeddings = rows
        ctx = ForwardContext(
            attn_metadata=parent,
            no_compile_layers={},
            kv_cache_data={},
            context=Context(
                positions=parent.step.positions,
                is_prefill=True,
                scheduled_bs=2,
                running_bs=2,
                scheduled_tokens=14,
                running_tokens=14,
            ),
            ubatch_slices=slices,
        )
        monkeypatch.setattr(_forward_context_local, "ctx", ctx, raising=False)
        result = wrapper(ids, parent.step.positions)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert calls == ["stage", "join"]
        assert _forward_context_local.ctx is ctx
    for table in staging.host.prefetcher._tables.values():
        table.disable_uva()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("cut", [1, 7, 10, 11, 13])
def test_split_attention_reads_preceding_microbatch_window(cut, monkeypatch):
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

    torch.manual_seed(37)
    builder, parent = make_parent("cuda")
    cache = parent.cache
    query = torch.randn(14, 8, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(1, 14, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(8, device="cuda")
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    spec = LayerAttentionSpec(layer_id=0, ratio=0, mode=AttentionMode.WINDOW)
    cache.state.view("window").normal_()
    before = cache.backing.clone()

    def attend(step, start, end):
        q = query[start:end].clone()
        values = kv[:, start:end]
        prefix, pptr, extend, eptr = cache.attention_indices(spec, step)
        assert step.is_prefill and not step.decode
        output = sparse_attn_v4_paged_2src(
            q,
            cache.pool,
            prefix,
            pptr,
            values[0],
            extend,
            eptr,
            sink,
            512**-0.5,
            out=q,
        )
        cache.write_window(0, values, step)
        return output

    expected = attend(parent.step, 0, 14)
    final_window = cache.state.view("window").clone()
    cache.backing.copy_(before)
    parts = [
        UBatchSlice(slice(0, 1 if cut <= 10 else 2), slice(0, cut)),
        UBatchSlice(slice(0 if cut < 10 else 1, 2), slice(cut, 14)),
    ]
    from atom.utils.forward_context import get_forward_context
    from atom.utils.tbo.ubatching import (
        tbo_current_ubatch_id,
        tbo_switch_to_compute_sync,
        tbo_yield_and_switch_from_compute_to_comm,
    )

    order = []

    class Model(torch.nn.Module):
        def forward(self, ids, positions, inputs_embeds=None):
            index = tbo_current_ubatch_id()
            step = get_forward_context().attn_metadata.step
            part = parts[index]
            assert step.is_prefill and not step.decode
            order.append((index, "attention"))
            output = attend(step, part.token_slice.start, part.token_slice.stop)
            order.append((index, "write_window"))
            # Match the production boundary: attention writes its window before
            # FFN yields to the other worker. GPU launches remain asynchronous.
            tbo_yield_and_switch_from_compute_to_comm()
            torch.cuda._sleep(2_000_000)
            tbo_switch_to_compute_sync()
            return output

    ctx = ForwardContext(
        attn_metadata=parent,
        no_compile_layers={},
        kv_cache_data={},
        context=Context(
            positions=parent.step.positions,
            is_prefill=True,
            scheduled_bs=2,
            running_bs=2,
            scheduled_tokens=14,
            running_tokens=14,
        ),
        ubatch_slices=parts,
    )
    monkeypatch.setattr(_forward_context_local, "ctx", ctx, raising=False)
    actual = UBatchWrapper(Model(), builder)(
        torch.arange(14, device="cuda", dtype=torch.int32), parent.step.positions
    )
    assert order == [
        (0, "attention"),
        (0, "write_window"),
        (1, "attention"),
        (1, "write_window"),
    ]
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(cache.state.view("window"), final_window, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("layers", [1, 8])
@pytest.mark.parametrize("stress_lifetime", [False, True])
@pytest.mark.parametrize("level", [0, 3])
@pytest.mark.parametrize("dispatch", ["fallback", "complete"])
def test_dp_moe_original_schedule_preserves_outputs(
    monkeypatch, tmp_path, unified, layers, stress_lifetime, level, dispatch
):
    """Exercise the real MoE scheduler with delayed, local collective stand-ins.

    Keep the original MoE yield-before-communication schedule.
    Exercise shared FusedMoE allocation ownership with delayed compute
    and enough partner allocations to reuse an unprotected comm output.
    The synthetic complete=True return checks downstream consumers and flag
    preservation only. It does not exercise comm-fused backend internals;
    that backend's factory excludes TBO.
    """
    from atom.config import CompilationConfig, CUDAGraphMode
    from atom.model_ops import moe
    from atom.models.deepseek_v41.model import Block
    from atom.models.deepseek_v41.moe import MoE
    from atom.models.deepseek_v41.runtime import RuntimeBlock
    from atom.utils.decorators import support_torch_compile
    from atom.utils.forward_context import get_forward_context
    from atom.utils.tbo.ubatching import tbo_active, tbo_current_ubatch_id

    builder, parent = make_parent("cuda")
    calls = []
    width = 5120 if stress_lifetime else 1
    config = SimpleNamespace(
        enable_dp_attention=True,
        compilation_config=CompilationConfig(
            level=level,
            cudagraph_mode=CUDAGraphMode.FULL,
            cache_dir=str(tmp_path),
            splitting_ops=[],
            compile_sizes=[],
        ),
    )
    monkeypatch.setattr(moe, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(moe, "get_dp_group", lambda: None)

    def gather(hidden, router, eager, ctx, group):
        assert eager == (not unified)
        calls.append((tbo_current_ubatch_id(), "gather"))
        torch.cuda._sleep(2_000_000)
        return hidden.clone(), router.clone(), len(hidden), [len(hidden)]

    def scatter(hidden, *args):
        calls.append((tbo_current_ubatch_id(), "scatter"))
        if stress_lifetime:
            scratch = [torch.empty_like(hidden) for _ in range(32)]
            for value in scratch:
                value.fill_(12345)
        torch.cuda._sleep(2_000_000)
        return hidden.clone()

    def experts(**kwargs):
        calls.append((tbo_current_ubatch_id(), "experts"))
        return kwargs["x"] * 2 + kwargs["router_logits"]

    monkeypatch.setattr(moe, "dp_gather_hidden_and_router", gather)
    monkeypatch.setattr(moe, "reduce_scatterv", scatter)
    monkeypatch.setattr(moe, "reduce_scatter_with_unpadding", scatter)
    layer = SimpleNamespace(
        dp_size=2,
        moe_parallel_config=SimpleNamespace(
            use_all2all_kernels=False, dp_logical_ratio=1
        ),
        quant_method=SimpleNamespace(apply=experts),
        reduce_results=False,
        top_k=1,
        renormalize=False,
        use_grouped_topk=False,
        global_num_experts=1,
        expert_map=None,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
        activation=None,
        apply_router_weight_on_input=False,
        prefix="test",
    )

    def run_experts(hidden, router):
        # Compile on the main thread before exercising the TBO workers, as
        # ModelRunner warmup does. The custom op keeps scheduling at runtime.
        if not tbo_active():
            return hidden * 2 + router
        context = get_forward_context().context
        context.running_tokens_are_unified = unified
        iteration = getattr(context, "test_moe_iteration", 0)
        if iteration == 0:
            calls.append((tbo_current_ubatch_id(), "prepare"))
        context.test_moe_iteration = iteration + 1
        result = moe.FusedMoE.forward_impl_graph(layer, hidden, router)
        if stress_lifetime and iteration + 1 == layers:
            torch.cuda._sleep(100_000_000)
        return result

    layer.forward_impl = run_experts
    config.compilation_config.static_forward_context["test"] = layer

    class RoutedExperts(torch.nn.Module):
        def forward(self, hidden, router):
            return torch.ops.aiter.moe_forward(hidden, router, "test")

        def forward_maybe_comm_fused(self, hidden, router, shared, **kwargs):
            if dispatch == "complete":
                # Synthetic completed output; bypass Module.__call__ and hooks.
                return torch.ops.aiter.moe_forward(hidden, router, "test"), True
            return self(hidden, router), False

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # This synthetic gate exercises the ordinary GEMM fallback.
            self.register_buffer("weight", torch.empty(0, 0, device="cuda"))

        def forward(self, hidden):
            return hidden + 3

    def block_init(self):
        torch.nn.Module.__init__(self)
        self.ffn = MoE.__new__(MoE)
        torch.nn.Module.__init__(self.ffn)
        self.ffn.experts = RoutedExperts()
        self.ffn.gate = Gate()

    monkeypatch.setattr(Block, "__init__", block_init)

    @support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
    class Model(torch.nn.Module):
        def __init__(self, atom_config):
            super().__init__()
            # Keep the real V4.1 routed-expert dispatch boundary.
            self.block = RuntimeBlock()

        def forward(self, input_ids, positions, inputs_embeds=None):
            result = input_ids.float()[:, None].expand(-1, width).contiguous()
            for _ in range(layers):
                result, complete = self.block.ffn.routed_expert_forward(result)
                assert complete == (dispatch == "complete")
            return result + torch.ones_like(result)

    ids = torch.arange(14, device="cuda", dtype=torch.int32)
    ctx = ForwardContext(
        attn_metadata=parent,
        no_compile_layers={},
        kv_cache_data={},
        context=Context(
            positions=parent.step.positions,
            is_prefill=True,
            scheduled_bs=2,
            running_bs=2,
            scheduled_tokens=14,
            running_tokens=14,
        ),
        ubatch_slices=_split_prefill_token_midpoint(2, [10, 4], 2, None),
    )
    monkeypatch.setattr(_forward_context_local, "ctx", ctx, raising=False)
    model = Model(config)
    with torch.inference_mode():
        model(ids, parent.step.positions)
    if level == 3:
        assert len(model.compiled_codes) == 1
    wrapper = UBatchWrapper(model, builder)
    for _ in range(3):
        calls.clear()
        actual = wrapper(ids, parent.step.positions)
        scale = 3**layers
        torch.testing.assert_close(
            actual,
            (ids.float()[:, None] * scale + 3 * (scale - 1) // 2 + 1).expand(-1, width),
        )
        expected = [(0, "prepare"), (1, "prepare")]
        for _ in range(layers):
            expected.extend(
                [
                    (0, "gather"),
                    (0, "experts"),
                    (1, "gather"),
                    (1, "experts"),
                    (0, "scatter"),
                    (1, "scatter"),
                ]
            )
        assert calls == expected


@pytest.mark.parametrize("parent_unified", [False, True])
def test_prefill_microbatch_uses_existing_dp_padding(monkeypatch, parent_unified):
    from atom.model_ops import moe

    builder, parent = make_parent("cpu")
    ctx = ForwardContext(
        attn_metadata=parent,
        no_compile_layers={},
        kv_cache_data={},
        context=Context(
            positions=parent.step.positions,
            is_prefill=True,
            scheduled_bs=2,
            running_bs=2,
            scheduled_tokens=14,
            running_tokens=14,
            running_tokens_are_unified=parent_unified,
        ),
        ubatch_slices=_split_prefill_token_midpoint(2, [10, 4], 2, None),
        ub_max_tokens_across_dp=(11, 12),
    )
    wrapper = UBatchWrapper(torch.nn.Identity(), builder, dp_gather_scatter=True)
    counts = wrapper._compute_ub_running_tokens(ctx, 2, 2, 4, torch.device("cpu"))
    assert counts == [11, 12]
    # The collective itself is exercised in the multi-GPU integration run.
    # Here retain its padded height so real unpadding must drop the tail.
    monkeypatch.setattr(
        moe, "get_dp_group", lambda: SimpleNamespace(reduce_scatter_tensor=lambda x: x)
    )
    for i, part in enumerate(ctx.ubatch_slices):
        child = wrapper._make_ubatch_context(
            ctx,
            part,
            part.request_slice.stop - part.request_slice.start,
            i,
            ub_running_tokens=counts[i],
        )
        assert child.context.running_tokens_are_unified
        assert child.context.running_tokens == counts[i]
        assert child.attn_metadata.step.width == child.context.scheduled_tokens == 7
        monkeypatch.setattr(_forward_context_local, "ctx", child, raising=False)
        hidden = torch.arange(7, dtype=torch.float32)[:, None]
        padded, local_tokens = moe.pad_for_all_gather(hidden)
        assert padded.shape == (counts[i], 1)
        torch.testing.assert_close(padded[:7], hidden)
        padded[7:] = float("nan")
        torch.testing.assert_close(
            moe.reduce_scatter_with_unpadding(padded, local_tokens), hidden
        )


@pytest.mark.parametrize("dp_attention", [False, True])
def test_forward_does_not_repeat_request_admission(monkeypatch, dp_attention):
    from atom.models.deepseek_v41.runtime import v41_begin_forward

    metadata = SimpleNamespace(
        image_mask=torch.ones(1, 1, dtype=torch.bool),
        step=SimpleNamespace(begin_forward=lambda: None, requests=()),
    )
    context = SimpleNamespace(
        attn_metadata=metadata,
        dp_metadata=object() if dp_attention else None,
    )
    monkeypatch.setattr(_forward_context_local, "ctx", context, raising=False)
    hidden = torch.empty(1, 1, 1)
    # Admission rejects unsupported media before any distributed forward.
    # Repeating that check here could strand peers in their collectives.
    v41_begin_forward(hidden)


@pytest.mark.parametrize("ubatch_idx", [0, 1])
def test_tbo_children_do_not_replay_late_layer_tails(monkeypatch, ubatch_idx):
    import threading

    from atom.models.deepseek_v41.bounded_replay import replay_rows
    from atom.utils.tbo import ubatching

    builder, metadata = make_parent("cpu")
    slices = [
        UBatchSlice(slice(0, 1), slice(0, 5)),
        UBatchSlice(slice(0, 2), slice(5, 14)),
    ]
    parent = ForwardContext(
        attn_metadata=metadata,
        context=Context(
            positions=metadata.step.positions,
            is_prefill=True,
            scheduled_bs=2,
            scheduled_tokens=14,
            running_bs=2,
            running_tokens=14,
        ),
        ubatch_slices=slices,
    )
    wrapper = UBatchWrapper(torch.nn.Identity(), builder)
    part = slices[ubatch_idx]
    child = wrapper._make_ubatch_context(
        parent, part, part.request_slice.stop, ubatch_idx=ubatch_idx
    )
    assert child.ubatch_slices is None
    width = child.attn_metadata.step.width
    assert replay_rows(child, width) == builder.geometry.ring_slots
    monkeypatch.setitem(
        ubatching._THREAD_ID_TO_CONTEXT, threading.get_ident(), ubatch_idx
    )
    assert replay_rows(child, width) is None


def test_tbo_rejects_compacted_scheduler_rows_before_slicing():
    builder, parent = make_parent("cpu", lengths=(0, 4))
    assert parent.scheduler_rows == (1,)
    with pytest.raises(ValueError, match="uncompacted scheduler"):
        builder.build_ubatch_prefill_metadata(
            parent, UBatchSlice(slice(0, 1), slice(0, 4)), 1
        )


@pytest.mark.parametrize("backend,expected", [("none", 0), ("v4", 0), ("v41", -1)])
def test_tbo_communication_priority_is_backend_owned(monkeypatch, backend, expected):
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    cls = {
        "v4": DeepseekV4AttentionMetadataBuilder,
        "v41": DeepseekV41MetadataBuilder,
    }.get(backend)
    builder = None if cls is None else cls.__new__(cls)
    priorities = []
    stream = object()

    def create_stream(*, priority):
        priorities.append(priority)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", create_stream)
    wrapper = UBatchWrapper(torch.nn.Identity(), builder)
    wrapper._ensure_comm_stream()
    wrapper._ensure_comm_stream()
    assert priorities == [expected]
    assert wrapper.comm_stream is stream


def test_tbo_preserves_v4_forward_signature(monkeypatch):
    from atom.models import deepseek_v4

    monkeypatch.setattr(deepseek_v4, "_pcp_active", lambda: False)
    monkeypatch.setattr(deepseek_v4, "_moe_pcp_merge_active", lambda: False)
    context = SimpleNamespace(context=SimpleNamespace(input_ids=None))
    monkeypatch.setattr(deepseek_v4, "get_forward_context", lambda: context)

    class Model(torch.nn.Module):
        forward = deepseek_v4.DeepseekV4ForCausalLM.forward
        _need_ids_gather = False

        def model(self, input_ids, positions):
            return input_ids + positions

    ids = torch.arange(4)
    positions = ids + 10
    monkeypatch.setattr(
        _forward_context_local,
        "ctx",
        SimpleNamespace(ubatch_slices=None),
        raising=False,
    )
    torch.testing.assert_close(UBatchWrapper(Model())(ids, positions), ids + positions)
    torch.testing.assert_close(context.context.input_ids, ids)


def test_fp4_page_budget_includes_persistent_microbatch_metadata(monkeypatch):
    import json
    from pathlib import Path

    from atom.model_ops.attentions.backends import CommonAttentionBuilder
    from atom.models.deepseek_v41.config import normalize_hf_config

    fixture = Path(__file__).parents[2] / "models/deepseek_v41/fixtures/config.json"
    hf = normalize_hf_config(json.loads(fixture.read_text()))
    runner = SimpleNamespace(
        block_size=256,
        forward_vars=metadata_buffers(2, 16, 8, "cpu"),
        config=SimpleNamespace(
            hf_config=hf,
            speculative_config=None,
            kv_cache_dtype="bf16",
            index_cache_dtype="fp4",
            enable_tbo=True,
            load_dummy=True,
            gpu_memory_utilization=0.75,
        ),
    )

    def initialize_base(self, model_runner):
        self.model_runner = model_runner
        self.device = "cpu"
        self.max_bs, self.max_num_batched_tokens, self.block_table_cols = 2, 16, 8

    monkeypatch.setattr(CommonAttentionBuilder, "__init__", initialize_base)
    recorded = []

    class BudgetBuilder(DeepseekV41MetadataBuilder):
        def _page_bound(self, utilization):
            # These allocations must already exist when memory_allocated is
            # subtracted from the PAGE budget; otherwise FP4 over-reserves.
            assert len(self._ubatch_indptrs) == 2
            for i in range(2):
                assert f"ub{i}_positions" in runner.forward_vars
                for name in self.step_planners[0]._names:
                    assert f"ub{i}_{name}" in runner.forward_vars
            used = sum(
                v.gpu.numel() * v.gpu.element_size()
                for v in runner.forward_vars.values()
            )
            used += sum(
                t.numel() * t.element_size()
                for indptrs in self._ubatch_indptrs
                for pair in indptrs.values()
                for t in pair
            )
            monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: used)
            total = self.geometry.paged_bytes * 100
            monkeypatch.setattr(
                torch.cuda, "mem_get_info", lambda device: (total - used, total)
            )
            pages = super()._page_bound(utilization)
            assert pages == max(
                1, (int(total * utilization) - used) // self.geometry.paged_bytes
            )
            recorded.append(pages)
            return pages

    builder = BudgetBuilder(runner)
    assert len(recorded) == 1
    assert builder.score_workspace.packed
    assert builder.step_planners[0].workspace is builder.score_workspace
