# SPDX-License-Identifier: MIT
"""Prefill microbatches preserve histories, cache plans and cross-layer state."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.attentions.deepseek_v41.backend import DeepseekV41MetadataBuilder
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.models.deepseek_v41.config import validate_runtime_config
from atom.utils.forward_context import Context, ForwardContext, _forward_context_local
from atom.utils.tbo.ubatch_splitting import UBatchSlice, _split_prefill_token_midpoint
from atom.utils.tbo.ubatch_wrapper import UBatchWrapper
from tests.attentions.deepseek_v41.helpers import metadata_buffers
from tests.attentions.deepseek_v41.test_runtime_contract import runtime_config


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
    parent, _ = builder._prepare(batch, len(lengths), sum(lengths))
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
        assert child.step.width == end - start
        torch.testing.assert_close(
            child.step.positions, parent.step.positions[start:end]
        )
        assert child.step.slots.tolist() == parent.step.slots[rs].tolist()
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
    for name, value in builder.model_runner.forward_vars.items():
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


def test_prefill_tbo_admission_keeps_decode_tbo_rejected():
    for dpa in (False, True):
        validate_runtime_config(
            runtime_config(enable_tbo=True, enable_dp_attention=dpa)
        )
        with pytest.raises(ValueError, match="decode TBO"):
            validate_runtime_config(
                runtime_config(
                    enable_tbo=True, enable_tbo_decode=True, enable_dp_attention=dpa
                )
            )


def test_parent_engram_join_runs_when_a_microbatch_fails():
    events = []
    parent = SimpleNamespace(
        engram_embeddings=SimpleNamespace(
            stage=lambda: events.append("stage"), join=lambda: events.append("join")
        )
    )
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    with pytest.raises(RuntimeError, match="failed child"):
        with builder.ubatch_forward(parent):
            raise RuntimeError("failed child")
    assert events == ["stage", "join"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_real_tbo_workers_share_one_uva_prefetch_and_keep_cross_layer_state(
    monkeypatch,
):
    from atom.utils.forward_context import get_forward_context
    from atom.utils.tbo.ubatching import (
        tbo_yield_and_switch_from_compute_to_comm,
        tbo_switch_to_compute_sync,
    )
    from tests.model_ops.engram.test_overlap import make_batch, make_staging
    from tests.model_ops.engram.test_hash_bounds import build, tiny_config

    builder, parent = make_parent("cuda", lengths=(9, 5))
    mapping = build(tiny_config())
    staging, backing = make_staging(mapping)
    calls = []
    start = staging.start
    join = staging.join
    monkeypatch.setattr(
        staging, "start", lambda width: (calls.append("stage"), start(width))[-1]
    )
    monkeypatch.setattr(staging, "join", lambda: (calls.append("join"), join())[-1])

    class Model(torch.nn.Module):
        def forward(self, ids, positions):
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
        batch, _, _, _ = make_batch(mapping, [9, 5], seed)
        # Keep the original history in snapshot even after the caller commits it.
        rows = staging.prepare(batch, 14)
        rows.stage()
        expected = (
            rows.get(mapping.config.layer_ids[-1])[0].float().clone()
            + ids.float()[:, None] * 2
        )
        rows.join()
        calls.clear()
        parent.engram_embeddings = rows
        batch.history.fill_(99)
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
def test_split_attention_reads_preceding_microbatch_window(cut):
    from atom.model_ops.v4_kernels import (
        sparse_attn_v4_paged_decode,
        sparse_attn_v4_paged_prefill,
    )

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
        if step.decode:
            cache.write_window(0, values, step)
            return sparse_attn_v4_paged_decode(
                q, cache.pool, prefix, pptr, sink, 512**-0.5
            )
        output = sparse_attn_v4_paged_prefill(
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
    actual = []
    for i, part in enumerate(parts):
        child = builder.build_ubatch_prefill_metadata(
            parent, part, part.request_slice.stop - part.request_slice.start, i
        )
        actual.append(attend(child.step, part.token_slice.start, part.token_slice.stop))
    torch.testing.assert_close(torch.cat(actual), expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(cache.state.view("window"), final_window, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("layers", [1, 8])
@pytest.mark.parametrize("stress_lifetime", [False, True])
def test_dp_moe_original_schedule_preserves_outputs(
    monkeypatch, unified, layers, stress_lifetime
):
    """Exercise the real MoE scheduler with delayed, local collective stand-ins.

    Keep the original MoE yield-before-communication schedule.
    Exercise the actual V4.1 output hook with delayed compute and enough
    partner allocations to reuse an unprotected comm output.
    """
    from atom.model_ops import moe
    from atom.models.deepseek_v41.model import Block
    from atom.models.deepseek_v41.runtime import DeepseekV41RuntimeModel, RuntimeBlock
    from atom.utils.tbo.ubatching import tbo_current_ubatch_id

    builder, parent = make_parent("cuda")
    calls = []
    width = 5120 if stress_lifetime else 1
    config = SimpleNamespace(enable_dp_attention=True)
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

    class RoutedExperts(torch.nn.Module):
        def forward(self, hidden, router):
            return moe.FusedMoE.forward_impl_graph(layer, hidden, router)

    def block_init(self):
        torch.nn.Module.__init__(self)
        self.ffn = torch.nn.Module()
        self.ffn.experts = RoutedExperts()

    monkeypatch.setattr(Block, "__init__", block_init)

    class Model(torch.nn.Module):
        tbo_comm_stream_priority = DeepseekV41RuntimeModel.tbo_comm_stream_priority

        def __init__(self):
            super().__init__()
            # Use the real runtime constructor to install the ownership hook.
            self.block = RuntimeBlock()

        def forward(self, ids, positions):
            from atom.utils.forward_context import get_forward_context

            get_forward_context().context.running_tokens_are_unified = unified
            calls.append((tbo_current_ubatch_id(), "prepare"))
            hidden = ids.float()[:, None].expand(-1, width).contiguous()
            result = hidden
            for _ in range(layers):
                result = self.block.ffn.experts(result, result + 3)
            # Force consumption on compute after reduce-scatter on comm.
            if stress_lifetime:
                torch.cuda._sleep(100_000_000)
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
    wrapper = UBatchWrapper(Model(), builder)
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


@pytest.mark.parametrize("unified", [False, True])
def test_prefill_microbatch_preserves_dp_shape_mode(monkeypatch, unified):
    import atom.config as config_module
    from atom.utils.forward_context import DPMetadata

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
            running_tokens_are_unified=unified,
        ),
        ubatch_slices=_split_prefill_token_midpoint(2, [10, 4], 2, None),
        dp_metadata=object(),
        ub_max_tokens_across_dp=[11, 12],
        ub_tokens_across_dp=((7, 3, 5, 11), (7, 4, 6, 12)),
    )
    monkeypatch.setattr(
        config_module,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=4, data_parallel_rank=0)
        ),
    )

    def unexpected_collective(*args):
        pytest.fail("precomputed token counts must not trigger another collective")

    monkeypatch.setattr(DPMetadata, "num_tokens_across_dp", unexpected_collective)
    wrapper = UBatchWrapper(torch.nn.Identity(), builder, dp_gather_scatter=True)
    counts = wrapper._compute_ub_running_tokens(ctx, 2, 2, 4, torch.device("cpu"))
    assert counts == ([11, 12] if unified else [7, 7])
    metadata = wrapper._make_ubatch_dp_metadata(ctx, 2)
    for i, part in enumerate(ctx.ubatch_slices):
        child = wrapper._make_ubatch_context(
            ctx,
            part,
            part.request_slice.stop - part.request_slice.start,
            i,
            ub_running_tokens=counts[i],
            dp_metadata=metadata[i],
            running_tokens_across_dp=ctx.ub_tokens_across_dp[i],
        )
        assert child.context.running_tokens_are_unified is unified
        assert child.context.running_tokens == counts[i]
        assert child.dp_metadata.get_sizes_across_dp() == list(
            ctx.ub_tokens_across_dp[i]
        )


@pytest.mark.parametrize("dp_attention", [False, True])
def test_runtime_image_scope_preserves_tp_but_rejects_dpa(monkeypatch, dp_attention):
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
    if dp_attention:
        with pytest.raises(NotImplementedError, match="DP attention supports text"):
            v41_begin_forward(hidden)
    else:
        v41_begin_forward(hidden)
