# SPDX-License-Identifier: MIT
"""V4/V4.1 metadata producers, isolated from unsupported full-model modes."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import CpuGpuBuffer
from atom.utils.h2d import PublicationError
from tests.attentions.deepseek_v41.helpers import (
    PagedRequest,
    prepare_step,
    publish_tables,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


def v41_runner(monkeypatch, transport, pp_size):
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from tests.attentions.deepseek_v41.helpers import metadata_buffers

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.config = SimpleNamespace(pipeline_parallel_size=pp_size)
    runner.enforce_eager = True
    geometry = V41PoolGeometry(
        2, ((0, 1), (1, 2)), 32, 4, 128, 32, speculative_tokens=3
    )
    runner.forward_vars = metadata_buffers(4, 32, 4, runner.device, geometry)
    for name in ("positions", "batch_id_per_q_token"):
        runner.forward_vars[name].publication_group = "v41_step"
    runner.forward_vars["cu_seqlens_q"].publication_group = "early"
    runner.forward_vars["block_tables"].publication_group = "prefill"
    runner.forward_vars["decode_src"] = CpuGpuBuffer(
        4, dtype=torch.int32, device="cuda"
    )
    runner.tokenID_processor = SimpleNamespace()
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.geometry = geometry
    builder.model_runner = runner
    runner.attn_metadata_builder = builder
    monkeypatch.setenv("ATOM_H2D_BACKEND", transport)
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    return runner, builder


@pytest.mark.parametrize("transport", ["direct", "packed"])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_v41_plans_and_step_replay_preserve_padding_and_slots(
    monkeypatch, transport, pp_size
):

    runner, builder = v41_runner(monkeypatch, transport, pp_size)
    graphs, copies = [], []
    for variables in runner._fv_ring:
        destinations = {
            name: torch.empty_like(buf.gpu) for name, buf in variables.items()
        }
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for name, buf in variables.items():
                destinations[name].copy_(buf.gpu)
        graphs.append(graph)
        copies.append(destinations)
    torch.cuda.synchronize()
    saved = []
    for step_id, lengths in enumerate(([4, 2, 1], [], [1], [3, 4], [2, 1, 4], [])):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        variables = runner.forward_vars
        for buf in variables.values():
            buf.gpu.fill_(77)
        lengths = np.asarray(lengths, dtype=np.int32)
        starts = np.arange(len(lengths), dtype=np.int32) * 5 + step_id
        spans, offset = [], 0
        for i, (start, length) in enumerate(zip(starts, lengths)):
            spans.append(PagedRequest(i, int(start), offset, int(length), i, (i,)))
            offset += int(length)
        torch.cuda._sleep(2_000_000)
        plans = builder._build_compress_plans(
            lengths, starts + lengths, running_bs=4, max_q_len=4, extra_write=3
        )
        metadata = prepare_step(
            spans,
            runner.device,
            buffers=variables,
            running_bs=4,
            running_tokens=16,
            max_q_len=4,
            ratios=(1, 2),
            state_slot_out=torch.arange(4, device="cuda", dtype=torch.int32),
            publication_group=runner.h2d_groups["v41_step"],
        )
        expected = {}
        for group_name in ("v4_plans", "v41_step"):
            group = runner.h2d_groups[group_name]
            for member, count in zip(group.members, group.counts):
                expected[member.name] = (member.source[:count].clone(), count)
        # The second invocation must fail BEFORE it can overwrite a live source.
        before = {n: variables[n].cpu.clone() for n in expected}
        with pytest.raises(PublicationError, match="republish_reason"):
            builder._build_compress_plans(
                lengths, starts + lengths + 100, extra_write=0
            )
        with pytest.raises(PublicationError, match="republish_reason"):
            prepare_step(
                spans,
                runner.device,
                buffers=variables,
                ratios=(1, 2),
                publication_group=runner.h2d_groups["v41_step"],
            )
        assert all(torch.equal(variables[n].cpu, value) for n, value in before.items())
        graphs[runner._fv_idx].replay()
        saved.append(
            (
                {n: copies[runner._fv_idx][n].clone() for n in expected},
                expected,
                metadata.positions.clone(),
                metadata.batch_ids.clone(),
                spans,
            )
        )
        for plan in plans.values():
            assert plan.write_plan_gpu.shape[0] == 16
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for actual, expected, positions, batch_ids, spans in saved:
        for name, (values, count) in expected.items():
            assert torch.equal(
                actual[name][:count].cpu().view(torch.uint8), values.view(torch.uint8)
            )
            assert torch.all(actual[name][count:].cpu() == 77)
        expected_positions = [
            p for span in spans for p in range(span.position, span.end)
        ]
        expected_ids = [i for i, span in enumerate(spans) for _ in range(span.length)]
        n = len(expected_positions)
        assert positions.cpu().tolist() == expected_positions + [0] * (16 - n)
        assert batch_ids.cpu().tolist() == expected_ids + [-1] * (16 - n)


@pytest.mark.parametrize("transport", ["direct", "packed"])
def test_v4_tbo_plan_allocations_follow_actual_runner_slots(monkeypatch, transport):
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.config = SimpleNamespace(
        pipeline_parallel_size=2, enable_tbo=True, enable_tbo_decode=True
    )
    runner.enforce_eager = True
    runner.forward_vars = {
        name: CpuGpuBuffer(32, dtype=torch.int32, device="cuda")
        for name in ("input_ids", "decode_src")
    }
    runner.tokenID_processor = SimpleNamespace()
    builder = DeepseekV4AttentionMetadataBuilder.__new__(
        DeepseekV4AttentionMetadataBuilder
    )
    builder.device = runner.device
    builder.model_runner = runner
    builder.max_num_batched_tokens = 32
    builder.max_bs = 4
    builder.window_size = 4
    builder.max_decode_tokens = 16
    builder.index_topk = 4
    builder.max_committed_hca = 4
    builder.block_table_cols = 4
    builder._indexer_fp4 = False
    builder._unique_compress_ratios_overlap = [(4, True)]
    builder._alloc_v4_metadata_buffers()
    monkeypatch.setenv("ATOM_H2D_BACKEND", transport)
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    previous = None
    results = []
    for step in range(5):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        current = builder._get_ubatch_compress_plan_buffers(0)[4]["compress"]
        assert current is runner.forward_vars["ub0_v4_compress_plan_4"]
        assert current is not previous
        previous = current
        torch.cuda._sleep(2_000_000)
        per_step = []
        for index in range(2):
            context = np.array([4 + step * 4 + index], dtype=np.int32)
            plans = builder._build_compress_plans(
                np.array([4], dtype=np.int32),
                context,
                extra_write=0,
                buf_prefix_ubatch=f"ub{index}_",
            )
            per_step.append(
                (plans[4].compress_plan_gpu.clone(), plans[4].compress_plan_cpu.copy())
            )
        results.extend(per_step)
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for actual, expected in results:
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)


def test_v4_mtp_uses_immutable_prefix_after_verify_owner_is_sealed(monkeypatch):
    from atom.model_engine.kv_block import STATE_SLOT_CLASS
    from atom.model_ops.attentions import deepseek_v4_attn as v4
    from atom.utils.h2d import PublicationOwner

    buf = CpuGpuBuffer(9, dtype=torch.int32, device="cuda")
    owner = PublicationOwner("cuda", torch.cuda.Event())
    owner.bind(buf, "v4_qo_indptr")
    constants = torch.arange(9, device="cuda", dtype=torch.int32)
    variables = {
        "v4_qo_indptr": buf,
        "v4_draft_qo_indptr": constants,
        "v4_empty_kv_indptr": torch.zeros(9, device="cuda", dtype=torch.int32),
        "v4_kv_indptr_swa": torch.zeros(9, device="cuda", dtype=torch.int32),
        "v4_kv_indices_swa": torch.zeros(32, device="cuda", dtype=torch.int32),
        "batch_id_per_q_token": CpuGpuBuffer(8, dtype=torch.int32, device="cuda"),
    }
    owner.completion.record()
    owner.begin()
    buf.np[:] = [0, 1, 2, 3, 4, 4, 4, 4, 4]
    torch.cuda._sleep(2_000_000)
    buf.copy_to_gpu()
    owner.finish()
    builder = v4.DeepseekV4AttentionMetadataBuilder.__new__(
        v4.DeepseekV4AttentionMetadataBuilder
    )
    builder.model_runner = SimpleNamespace(
        forward_vars=variables, pool_plan=SimpleNamespace(entries={STATE_SLOT_CLASS: 2})
    )
    builder._mtp_layers_are_swa_only = True
    builder.window_size = 4
    builder._kv_fp8 = True
    builder.row_ids = torch.arange(8, device="cuda", dtype=torch.int32)
    builder.pool_geometry = object()
    builder._dest_row_buffers = dict
    metadata = SimpleNamespace(
        context_lens=torch.full((4,), 7, device="cuda", dtype=torch.int32),
        state_slot_out=torch.arange(4, device="cuda", dtype=torch.int32),
    )
    monkeypatch.setattr(
        v4, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    monkeypatch.setattr(v4, "write_v4_paged_decode_indices", lambda **kwargs: None)
    builder.prepare_mtp_decode(2, 1, 7, torch.arange(4, device="cuda"))
    assert metadata.qo_indptr.data_ptr() == constants.data_ptr()
    assert metadata.qo_indptr.cpu().tolist() == [0, 1, 2, 3, 4]
    assert buf.gpu.cpu().tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 4]
    assert buf.cpu.tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 4]


@pytest.mark.parametrize("pp_size", [1, 2])
def test_v41_combined_publication_precedes_indptr_consumer(monkeypatch, pp_size):
    from atom.model_ops.attentions.deepseek_v41 import cache as cache_module

    runner, builder = v41_runner(monkeypatch, "packed", pp_size)
    builder.device = runner.device
    builder.block_size = builder.geometry.block_size
    builder.cache = cache_module.PagedAttentionCache(
        builder.geometry, 8, 4, runner.device, max_tokens=32
    )
    original = cache_module.fill_step_indptrs
    observed = []

    def consume(step, geometry, buffers):
        group = runner.h2d_groups["v41_metadata"]
        assert group._backend.kernel is not None
        assert runner.h2d_groups["v4_plans"]._backend is None
        assert runner.h2d_groups["v41_step"]._backend is None
        # Clone before the real first consumer, on the same stream: this sees
        # the bytes available to indptr construction, including all padding.
        observed.append(
            [
                (member.destination[:count].clone(), member.source[:count].clone())
                for member, count in zip(group.members, group.counts)
                if count is not None
            ]
        )
        return original(step, geometry, buffers)

    monkeypatch.setattr(cache_module, "fill_step_indptrs", consume)
    keys = {}
    for iteration, lengths in enumerate(([4, 1, 2],) * 4 + ([1, 3],) * 4):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        slots = [2, 0, 1][: len(lengths)]
        blocks = tuple((slot,) for slot in slots)
        batch = SimpleNamespace(
            is_dummy_run=False,
            req_ids=tuple(100 + slot for slot in slots),
            num_scheduled_tokens=np.asarray(lengths, dtype=np.int32),
            context_lens=np.asarray(lengths, dtype=np.int32) + 8 + iteration,
            state_slots_committed=slots,
            block_tables=blocks,
            total_seqs_num=len(lengths),
            total_tokens_num=sum(lengths),
        )
        torch.cuda._sleep(2_000_000)
        metadata, _ = builder._prepare(batch, 4, 16, max_q_len=4, tentative=True)
        group = runner.h2d_groups["v41_metadata"]
        table_count = group.counts[group.indices["block_tables"]]
        assert table_count == (None if keys.get(runner._fv_idx) == blocks else 4)
        keys[runner._fv_idx] = blocks
        assert metadata.state_slot_out.shape == (4,)
        assert metadata.step.positions.shape == (16,)
        # A plan was already included in the combined publication: its own
        # group still rejects duplicate writes before touching pinned sources.
        with pytest.raises(PublicationError, match="republish_reason"):
            builder._build_compress_plans(
                batch.num_scheduled_tokens, batch.context_lens, extra_write=0
            )
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for fields in observed:
        for gpu, cpu in fields:
            assert torch.equal(gpu.cpu().view(torch.uint8), cpu.view(torch.uint8))


@pytest.mark.parametrize("source", ["query_prefix", "block_tables"])
def test_v41_standalone_producer_rejects_before_source_rewrite(monkeypatch, source):

    runner, _ = v41_runner(monkeypatch, "direct", 1)
    var = runner.forward_vars
    spans = [PagedRequest(0, 4, 0, 1, 0, (7,))]
    runner._gate_staging_reuse()
    if source == "query_prefix":
        buf = var["cu_seqlens_q"]
        buf.np[:3] = [0, 2, 2]
    else:
        buf = var["block_tables"]
        buf.np[:2] = 3
    before = buf.cpu.clone()
    torch.cuda._sleep(20_000_000)
    buf.copy_to_gpu(2 if source == "block_tables" else 3)
    observed = buf.gpu.clone()
    try:
        with pytest.raises(PublicationError, match="republish_reason"):
            if source == "block_tables":
                publish_tables(buf, spans, 2)
            else:
                prepare_step(
                    spans,
                    runner.device,
                    buffers=var,
                    running_bs=2,
                    running_tokens=4,
                    state_slot_out=torch.zeros(2, dtype=torch.int32, device="cuda"),
                    publication_group=runner.h2d_groups["v41_step"],
                )
    finally:
        runner._mark_staging_h2d_enqueued()
        torch.cuda.synchronize()
    assert torch.equal(buf.cpu, before)
    count = 2 if source == "block_tables" else 3
    assert torch.equal(observed[:count].cpu(), before[:count])


def test_dummy_storage_isolation_survives_shared_metadata():
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from tests.attentions.deepseek_v41.helpers import metadata_buffers

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.geometry = V41PoolGeometry(1, ((0, 2),), 32, 4, 512, 32)
    builder.block_size, builder.device = 32, "cuda"
    builder.model_runner = SimpleNamespace(
        forward_vars=metadata_buffers(4, 36, 4, "cuda", builder.geometry)
    )
    serving = builder.cache = PagedAttentionCache(builder.geometry, 8, 4, "cuda")
    serving.backing.fill_(17)
    before = serving.backing.clone()
    batch = SimpleNamespace(
        is_dummy_run=True,
        req_ids=(-1, -2),
        num_scheduled_tokens=(34, 2),
        context_lens=(34, 2),
        state_slots_committed=(),
        block_tables=((0,), (0,)),
        total_seqs_num=2,
        total_tokens_num=36,
    )
    metadata, _ = builder._prepare(batch, 4, 36)
    private, step = metadata.cache, metadata.step
    assert private is not serving and metadata.dummy
    assert not hasattr(step.requests[0], "block_ids")
    assert step.block_tables[:2, :2].tolist() == [[0, 1], [2, 0]]
    kv = torch.full((1, step.width, 512), 3, dtype=torch.bfloat16, device="cuda")
    old_state, old_pages = private.state_bytes.clone(), private.page_bytes.clone()
    private.write_window(0, kv, step)
    n = step.plans[2].compress_plan_gpu.shape[0]
    compressed = torch.full((1, n, 512), 5, dtype=torch.bfloat16, device="cuda")
    private._scatter_rows(private.pages.view("main_0")[0], step, compressed, 2)
    torch.cuda.synchronize()
    assert torch.equal(serving.backing, before)
    assert not torch.equal(private.state_bytes, old_state)
    assert not torch.equal(private.page_bytes, old_pages)
