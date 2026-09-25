# SPDX-License-Identifier: MIT
"""V4 indexer/PCP producers use the forward slot through late TBO prepare."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import CpuGpuBuffer
from atom.utils.h2d import PublicationError

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


def make_runner(monkeypatch, *, pp_size=1, opus=False, base=None):
    import atom.config
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )
    from atom.model_ops.attentions.pool_layout.v4_pool_fields import FP4_GFX1250_NATURAL

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.config = SimpleNamespace(
        pipeline_parallel_size=pp_size, enable_tbo=True, enable_tbo_decode=True
    )
    monkeypatch.setattr(atom.config, "_current_atom_config", runner.config)
    runner.enforce_eager = True
    runner.tokenID_processor = SimpleNamespace()
    runner.pool_plan = SimpleNamespace(entries={})
    runner.forward_vars = {}
    for name, capacity, dtype, group in (
        ("input_ids", 32, torch.int32, None),
        ("decode_src", 32, torch.int32, None),
        ("positions", 32, torch.int64, "positions"),
        ("context_lens", 4, torch.int32, "context"),
        ("cu_seqlens_q", 5, torch.int32, "early"),
        ("cu_seqlens_k", 5, torch.int32, "cu_k"),
        ("batch_id_per_q_token", 32, torch.int32, None),
    ):
        runner.forward_vars[name] = CpuGpuBuffer(
            capacity, dtype=dtype, device=runner.device, publication_group=group
        )
    runner.forward_vars["block_tables"] = CpuGpuBuffer(
        4, 4, dtype=torch.int32, device=runner.device, publication_group="blocks"
    )
    cls = base or DeepseekV4AttentionMetadataBuilder
    builder = cls.__new__(cls)
    builder.model_runner = runner
    builder.device = runner.device
    builder.max_num_batched_tokens = 32
    builder.max_bs = 4
    builder.max_decode_tokens = 16
    builder.window_size = 4
    builder.index_topk = 4
    builder.max_committed_hca = 4
    builder.block_table_cols = 4
    builder.max_spec_steps = 2
    builder._unique_compress_ratios_overlap = [(4, True)]
    builder._indexer_fp4 = False
    builder.indexer_layout = FP4_GFX1250_NATURAL if opus else "fp8"
    builder.pool_geometry = SimpleNamespace(slot_positions=4)
    builder._alloc_v4_metadata_buffers()
    runner.attn_metadata_builder = builder
    monkeypatch.setenv("ATOM_H2D_BACKEND", "direct")
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    return runner, builder


def make_metadata(builder, lengths=(3, 5, 3), starts=(8, 16, 0), *, stage_map=True):
    from atom.model_ops.attentions.deepseek_v4_attn import AttentionMetaData_DSV4
    from atom.utils.forward_context import AttentionMetaData, AttnState
    from atom.utils.tbo.ubatch_splitting import attach_tbo_cpu_lens

    lengths = np.asarray(lengths, dtype=np.int32)
    starts = np.asarray(starts, dtype=np.int32)
    cu = np.r_[np.int32(0), np.cumsum(lengths, dtype=np.int32)]
    positions = np.concatenate(
        [
            np.arange(start, start + n, dtype=np.int64)
            for start, n in zip(starts, lengths)
        ]
    )
    pos_gpu = builder._stage("positions", positions)
    cu_gpu = builder._stage("cu_seqlens_q", cu)
    ctx_gpu = builder._stage("context_lens", starts + lengths)
    tables = builder._stage("block_tables", np.ones((len(lengths), 4), dtype=np.int32))
    slots = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    md = AttentionMetaData(
        cu_seqlens_q=cu_gpu,
        cu_seqlens_k=cu_gpu,
        max_seqlen_q=int(lengths.max()),
        max_seqlen_k=int((starts + lengths).max()),
        context_lens=ctx_gpu,
        block_tables=tables,
        state=AttnState.PREFILL_NATIVE,
        has_cached=False,
    )
    # Serving constructs the base metadata before promoting it to DSV4.
    md.__class__ = AttentionMetaData_DSV4
    md.state_slot_out = slots
    md.state_slot_in = slots
    md.state_slot_out_cpu = np.arange(len(lengths), dtype=np.int32)
    if stage_map:
        builder._attach_v4_per_fwd_meta(
            md,
            np.repeat(np.arange(len(lengths), dtype=np.int32), lengths),
            md.state_slot_out_cpu,
            len(lengths),
            len(positions),
        )
    md.indexer_meta = {}
    for name, value in (
        ("cu_seqlens_q", cu),
        ("cu_seqlens_k", cu),
        ("context_lens", starts + lengths),
    ):
        attach_tbo_cpu_lens(md, name, value.copy())
    return md, pos_gpu, cu, lengths


def set_pcp(monkeypatch, size, rank):
    from atom.distributed.pcp_utils import pcp_round_robin_query_indices
    from atom.model_ops.attentions import deepseek_v4_attn as module

    monkeypatch.setattr(module, "pcp_is_enabled", lambda: True)
    monkeypatch.setattr(module, "get_pcp_world_size", lambda: size)
    monkeypatch.setattr(
        module,
        "pcp_round_robin_query_indices",
        lambda total, width: pcp_round_robin_query_indices(total, width, rank),
    )


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("rank", [0, 3])
def test_pcp_ragged_queries_publish_indexer_once(monkeypatch, pp_size, rank):
    runner, builder = make_runner(monkeypatch, pp_size=pp_size)
    set_pcp(monkeypatch, 4, rank)
    saved = []
    for step in range(5):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        md, positions, cu, lengths = make_metadata(
            builder, starts=(8 + 4 * step, 16, 0)
        )
        md.kv_indptr_extend = torch.arange(12, dtype=torch.int32, device="cuda")
        md.kv_indices_extend = torch.arange(11, dtype=torch.int32, device="cuda")
        buf = runner.forward_vars["v4_indexer_cu_committed"]
        buf.gpu.fill_(77)
        torch.cuda._sleep(5_000_000)
        local = builder._apply_pcp_reindex(md, positions, 3, 11, cu)
        expected_cu = np.r_[0, np.cumsum(md.n_committed_csa_per_seq_cpu)].astype(
            np.int32
        )
        expected_cu[-1] = max(expected_cu[-1], 1)
        saved.append(
            (
                md.indexer_meta["cu_committed_gpu"].clone(),
                expected_cu,
                md.batch_id_per_q_token.clone(),
                md.kv_indices_extend.clone(),
                local.clone(),
            )
        )
        before = buf.cpu.clone()
        with pytest.raises(PublicationError, match="republish_reason"):
            builder._build_v4_indexer_meta(
                attn_metadata=md,
                positions_gpu=local,
                scheduled_bs=3,
                total_tokens=3,
                device=runner.device,
            )
        assert torch.equal(buf.cpu, before)
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    ids = np.r_[np.repeat(np.arange(3), lengths), -1][rank::4]
    for actual, expected, bids, indices, local in saved:
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)
        np.testing.assert_array_equal(bids.cpu().numpy(), ids)
        assert indices.cpu().tolist() == list(range(rank, 11, 4))
        assert len(local) == 3
    assert torch.all(buf.gpu[4:] == 77)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
def test_prepare_prefill_defers_indexer_only_for_real_pcp(monkeypatch, balanced, dummy):
    from atom.model_ops.attentions.backends import CommonAttentionBuilder

    runner, builder = make_runner(monkeypatch)
    set_pcp(monkeypatch, 4, 0)
    runner._pcp_tbo_balanced_active = balanced
    runner._gate_staging_reuse()
    md, pos, _, _ = make_metadata(builder, stage_map=False)
    monkeypatch.setattr(
        CommonAttentionBuilder, "prepare_prefill", lambda *args: (md, pos)
    )
    batch = SimpleNamespace(
        total_seqs_num_prefill=3,
        total_tokens_num_prefill=11,
        is_dummy_run=dummy,
        state_slots_committed=[],
        state_fork_srcs=None,
    )
    actual, _ = builder.prepare_prefill(batch, 3)
    binding = runner.forward_vars["v4_indexer_cu_committed"]._publication
    assert (binding._epoch == runner.h2d_owner.epoch) == (dummy or not balanced)
    assert len(actual.batch_id_per_q_token) == (11 if dummy or balanced else 3)
    runner._mark_staging_h2d_enqueued()


@pytest.mark.parametrize("pp_size", [1, 2])
def test_token_split_tbo_resumes_epoch_and_preserves_both_prefixes(
    monkeypatch, pp_size
):
    from atom.utils.tbo.ubatch_splitting import UBatchSlice

    runner, builder = make_runner(monkeypatch, pp_size=pp_size)
    saved = []
    for step in range(5):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        md, _, _, _ = make_metadata(builder, (5, 4), (8 + 4 * step, 20))
        runner._mark_staging_h2d_enqueued()
        epoch = runner.h2d_owner.epoch
        ubatches = []
        for index, sl in enumerate(
            (
                UBatchSlice(slice(0, 1), slice(0, 4)),
                UBatchSlice(slice(0, 2), slice(4, 9)),
            )
        ):
            torch.cuda._sleep(5_000_000)
            ub = builder.build_ubatch_prefill_metadata(
                md, sl, sl.request_slice.stop, index
            )
            assert runner.h2d_owner.epoch == epoch
            assert runner.h2d_owner._state == "sealed"
            buf = runner.forward_vars[f"ub{index}_cu_seqlens_q"]
            assert ub.cu_seqlens_q.data_ptr() == buf.gpu.data_ptr()
            ubatches.append(ub)
        # Read after BOTH preparations. A snapshot taken before the second
        # producer could conceal a shared-storage overwrite.
        for index, ub in enumerate(ubatches):
            assert (
                ub.batch_id_per_q_token.data_ptr()
                == runner.forward_vars[f"ub{index}_batch_id_per_q_token"].gpu.data_ptr()
            )
            assert (
                ub.indexer_meta["cu_committed_gpu"].data_ptr()
                == runner.forward_vars[
                    f"ub{index}_v4_indexer_cu_committed"
                ].gpu.data_ptr()
            )
            saved.append(
                (
                    index,
                    ub.cu_seqlens_q.clone(),
                    ub.indexer_meta["cu_committed_gpu"].clone(),
                    step,
                )
            )
        before = runner.forward_vars["ub0_context_lens"].cpu.clone()
        with pytest.raises(PublicationError, match="republish_reason"):
            builder.build_ubatch_prefill_metadata(
                md, UBatchSlice(slice(0, 1), slice(0, 4)), 1, 0
            )
        assert runner.h2d_owner._state == "sealed"
        assert torch.equal(runner.forward_vars["ub0_context_lens"].cpu, before)
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for index, query, committed, step in saved:
        assert query.cpu().tolist() == ([0, 4] if index == 0 else [0, 1, 5])
        assert committed.cpu().tolist() == (
            [0, 3 + step] if index == 0 else [0, 3 + step, 9 + step]
        )


@pytest.mark.parametrize("rank", [0, 3])
def test_balanced_pcp_tbo_uses_separate_indexer_buffers(monkeypatch, rank):
    runner, builder = make_runner(monkeypatch)
    set_pcp(monkeypatch, 4, rank)
    runner._pcp_tbo_balanced_active = True
    runner._pcp_bal_groups = [
        SimpleNamespace(req_start=0, req_stop=2, tok_start=0, tok_end=8),
        SimpleNamespace(req_start=2, req_stop=4, tok_start=8, tok_end=15),
    ]
    runner._gate_staging_reuse()
    md, _, _, _ = make_metadata(builder, (3, 5, 3, 4), (8, 16, 28, 40))
    runner._mark_staging_h2d_enqueued()
    ubatches = []
    for index in range(2):
        torch.cuda._sleep(5_000_000)
        ubatches.append(builder.build_ubatch_prefill_metadata(md, None, 2, index))
    runner.h2d_owner.completion.synchronize()
    assert ubatches[0].indexer_meta["cu_committed_gpu"].cpu().tolist() == [0, 2, 7]
    assert ubatches[1].indexer_meta["cu_committed_gpu"].cpu().tolist() == [0, 7, 18]
    assert [ub.max_seqlen_q for ub in ubatches] == [5, 4]
    for index, ub in enumerate(ubatches):
        assert (
            ub.indexer_meta["cu_committed_gpu"].data_ptr()
            == runner.forward_vars[f"ub{index}_v4_indexer_cu_committed"].gpu.data_ptr()
        )
    assert runner.forward_vars["v4_indexer_cu_committed"]._publication._epoch == -1
    for index in range(2):
        assert (
            runner.forward_vars[
                f"ub{index}_v4_indexer_cu_committed"
            ]._publication._epoch
            == runner.h2d_owner.epoch
        )


def mock_opus(monkeypatch):
    name = "aiter.ops.opus.pa_mqa_logits_mxfp4"

    def plan(cu, ends, **kwargs):
        return (cu.clone(), ends.clone(), kwargs["row_to_batch"].clone())

    monkeypatch.setitem(
        sys.modules, name, SimpleNamespace(pa_mqa_logits_mxfp4_plan=plan)
    )


@pytest.mark.parametrize("prefix", ["", "ub0_", "ub1_"])
def test_opus_chunks_are_one_checked_upload_with_independent_ranges(
    monkeypatch, prefix
):
    from atom.model_ops.attentions import deepseek_v4_attn as module

    mock_opus(monkeypatch)
    monkeypatch.setattr(module, "sparse_indexer_row_chunk", lambda *args: 2)
    runner, builder = make_runner(monkeypatch, opus=True)
    runner._gate_staging_reuse()
    buf = runner.forward_vars[f"{prefix}v4_indexer_chunk_cu"]
    buf.gpu.fill_(77)
    cu = np.array([0, 0, 3, 3, 8], dtype=np.int32)
    md = SimpleNamespace(
        n_committed_csa_per_seq_cpu=np.array([0, 2, 0, 4], dtype=np.int32),
        batch_id_per_q_token=torch.tensor(
            [1, 1, 1, 3, 3, 3, 3, 3], device="cuda", dtype=torch.int32
        ),
    )
    ends = torch.arange(8, dtype=torch.int32, device="cuda")
    meta = {}
    torch.cuda._sleep(5_000_000)
    builder._build_fp4_opus_prefill_plans(
        attn_metadata=md,
        meta=meta,
        total_tokens=8,
        scheduled_bs=4,
        visible_end_gpu=ends,
        cu_seqlens_q_cpu=cu,
        reuse_cu_seqlens_q=False,
        plan_total_tokens=None,
        buf_prefix_ubatch=prefix,
    )
    assert buf._publication._epoch == runner.h2d_owner.epoch
    runner._mark_staging_h2d_enqueued()
    runner.h2d_owner.completion.synchronize()
    entries = 0
    for start, end, plan in meta["fp4_opus_prefill_chunks"]:
        expected = module._chunk_cu_seqlens(cu, start, end)
        np.testing.assert_array_equal(plan[0].cpu().numpy(), expected)
        entries += len(expected)
    assert torch.all(buf.gpu[entries:] == 77)


def test_opus_full_prefix_reuse_does_not_publish_chunk_storage(monkeypatch):
    from atom.model_ops.attentions import deepseek_v4_attn as module

    mock_opus(monkeypatch)
    monkeypatch.setattr(module, "sparse_indexer_row_chunk", lambda *args: 32)
    runner, builder = make_runner(monkeypatch, opus=True)
    runner._gate_staging_reuse()
    md, pos, cu, _ = make_metadata(builder)
    meta = {}
    builder._build_fp4_opus_prefill_plans(
        attn_metadata=md,
        meta=meta,
        total_tokens=11,
        scheduled_bs=3,
        visible_end_gpu=pos.to(torch.int32),
        cu_seqlens_q_cpu=cu,
        reuse_cu_seqlens_q=True,
        plan_total_tokens=None,
    )
    assert runner.forward_vars["v4_indexer_chunk_cu"]._publication._epoch == -1
    assert torch.equal(meta["fp4_opus_prefill_chunks"][0][2][0], md.cu_seqlens_q)
    runner._mark_staging_h2d_enqueued()


def test_opus_packed_source_reuse_and_capacity_check(monkeypatch):
    from atom.model_ops.attentions import deepseek_v4_attn as module

    mock_opus(monkeypatch)
    monkeypatch.setattr(module, "sparse_indexer_row_chunk", lambda *args: 1)
    runner, builder = make_runner(monkeypatch, opus=True)
    saved = []
    for step in range(5):
        runner._gate_staging_reuse()
        md, pos, cu, _ = make_metadata(builder, (step + 1, 4), (8, 20))
        buf = runner.forward_vars["v4_indexer_chunk_cu"]
        before = buf.cpu.clone()
        kwargs = {
            "attn_metadata": md,
            "total_tokens": len(pos),
            "scheduled_bs": 2,
            "visible_end_gpu": pos.to(torch.int32),
            "cu_seqlens_q_cpu": cu,
            "reuse_cu_seqlens_q": False,
        }
        with pytest.raises(ValueError, match="capacity"):
            builder._build_fp4_opus_prefill_plans(
                meta={}, plan_total_tokens=100, **kwargs
            )
        assert torch.equal(buf.cpu, before)
        assert buf._publication._epoch != runner.h2d_owner.epoch
        torch.cuda._sleep(5_000_000)
        meta = {}
        builder._build_fp4_opus_prefill_plans(
            meta=meta, plan_total_tokens=None, **kwargs
        )
        saved.append(meta["fp4_opus_prefill_chunks"])
        runner._mark_staging_h2d_enqueued()
    torch.cuda.synchronize()
    for chunks in saved:
        assert all(plan[0].cpu().tolist() == [0, 1] for _, _, plan in chunks)


def test_late_ubatch_failure_poisoning_and_wrong_stream_rejection(monkeypatch):
    from atom.utils.tbo.ubatch_splitting import UBatchSlice

    runner, builder = make_runner(monkeypatch)
    runner._gate_staging_reuse()
    md, _, _, _ = make_metadata(builder, (5, 4), (8, 20))
    runner._mark_staging_h2d_enqueued()
    sl = UBatchSlice(slice(0, 1), slice(0, 4))
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(PublicationError, match="original owner stream"),
    ):
        builder.build_ubatch_prefill_metadata(md, sl, 1, 0)
    assert runner.h2d_owner._state == "sealed"

    def fail(*args, **kwargs):
        raise RuntimeError("indexer consumer failed")

    monkeypatch.setattr(builder, "_attach_v4_indexer_meta", fail)
    with pytest.raises(RuntimeError, match="consumer failed"):
        builder.build_ubatch_prefill_metadata(md, sl, 1, 0)
    assert runner.h2d_owner._state == "failed"
    before = runner.forward_vars["ub0_context_lens"].cpu.clone()
    with pytest.raises(PublicationError, match="failed"):
        builder.build_ubatch_prefill_metadata(md, sl, 1, 0)
    assert torch.equal(runner.forward_vars["ub0_context_lens"].cpu, before)
    runner.h2d_owner.drain()


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("kv_fp8", [False, True])
def test_decode_ubatch_inputs_padding_and_graph_consumers(
    monkeypatch, pp_size, padded, kv_fp8
):
    from atom.model_ops.attentions import deepseek_v4_attn as module
    from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
        UnifiedPoolGeometry,
    )
    from atom.utils.tbo.ubatch_wrapper import UBatchWrapper

    runner, builder = make_runner(monkeypatch, pp_size=pp_size)
    builder.pool_geometry = UnifiedPoolGeometry([0, 4, 128], 8, 4, 6, 256)
    builder.hca_rows_per_block = 2
    builder._kv_fp8 = kv_fp8
    runner.enforce_eager = not padded
    if padded:
        monkeypatch.setattr(module, "get_forward_context", lambda: None)
        monkeypatch.setattr(UBatchWrapper, "_decode_ub_running_bs", lambda *args: 2)
    names = [
        f"ub{i}_{name}"
        for i in range(2)
        for name in (
            "positions",
            "context_lens",
            "cu_seqlens_q",
            "block_tables",
            "v4_meta_state_slot_out",
            "v4_meta_state_slot_in",
            "batch_id_per_q_token",
        )
    ]
    if kv_fp8:
        names.extend(f"ub{i}_v4_qo_indptr" for i in range(2))
    graphs, outputs = [], []
    for var in runner._fv_ring:
        out = {name: torch.empty_like(var[name].gpu) for name in names}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for name in names:
                out[name].copy_(var[name].gpu)
        graphs.append(graph)
        outputs.append(out)
    saved = []
    for step, lens in enumerate(((1, 3, 2), (2,), (), (3, 1, 2, 1), (1, 2, 3))):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        var = runner.forward_vars
        for name in names:
            var[name].gpu.fill_(77)
            # Test-only storage poisoning also discards the published revision.
            if hasattr(var[name], "_block_table"):
                del var[name]._block_table
        lengths = np.asarray(lens, dtype=np.int32)
        starts = np.arange(len(lens), dtype=np.int32) * 8 + 8 + step
        positions = (
            np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lens)])
            if lens
            else np.empty(0, dtype=np.int64)
        )
        slots = np.arange(len(lens), dtype=np.int32)
        var["block_tables"].np[: len(lens)] = slots[:, None] + 1
        kwargs = {
            "scheduled_bs": len(lens),
            "running_bs": 4 if padded else len(lens),
            "max_seqlen_q": 3,
            "context_lens_np": starts + lengths,
            "state_slot_np": slots,
            "state_slot_in_np": slots[::-1].copy(),
            "positions_np": positions,
            "extend_lens_np": lengths,
        }
        global_qo = np.minimum(np.arange(13, dtype=np.int32), len(positions))
        if kv_fp8:
            builder._stage("v4_qo_indptr", global_qo)
        torch.cuda._sleep(5_000_000)
        builder._prepare_ubatch_decode(**kwargs)
        if kv_fp8:
            # The actual serving path prepared the global verify prefix first.
            np.testing.assert_array_equal(var["v4_qo_indptr"].np[:13], global_qo)
            saved.append((var["v4_qo_indptr"].gpu[:13].clone(), global_qo.tolist()))
        graphs[runner._fv_idx].replay()
        split = min(len(lens), 2) if padded else len(lens) // 2
        for index, (lo, hi) in enumerate(((0, split), (split, len(lens)))):
            width = 2 if padded else hi - lo
            local_lens = lengths[lo:hi]
            tokens = int(local_lens.sum())
            local_pos = positions[
                int(lengths[:lo].sum()) : int(lengths[:hi].sum())
            ].tolist()
            expected = {
                "positions": local_pos + [0] * (3 * width - tokens),
                "context_lens": (starts + lengths)[lo:hi].tolist()
                + [0] * (width - hi + lo),
                "cu_seqlens_q": [0]
                + np.cumsum(local_lens).tolist()
                + [tokens] * (width - hi + lo),
                "block_tables": [[i + 1] * 4 for i in range(lo, hi)]
                + [[0] * 4] * (width - hi + lo),
                "v4_meta_state_slot_out": slots[lo:hi].tolist()
                + [0] * (width - hi + lo),
                "v4_meta_state_slot_in": slots[::-1][lo:hi].tolist()
                + [0] * (width - hi + lo),
                "batch_id_per_q_token": np.repeat(
                    np.arange(hi - lo), local_lens
                ).tolist()
                + [-1] * (3 * width - tokens),
            }
            if kv_fp8:
                # Empty ubatches use the existing prefill fallback and never
                # publish a decode query prefix.
                expected["v4_qo_indptr"] = (
                    list(range(tokens + 1)) + [tokens] * (3 * width - tokens)
                    if tokens
                    else []
                )
            for suffix, values in expected.items():
                name = f"ub{index}_{suffix}"
                saved.append((outputs[runner._fv_idx][name].clone(), values))
        before = {name: var[name].cpu.clone() for name in names}
        with pytest.raises(PublicationError, match="republish_reason"):
            builder._prepare_ubatch_decode(**kwargs)
        assert all(torch.equal(var[name].cpu, before[name]) for name in names)
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for actual, values in saved:
        assert actual[: len(values)].cpu().tolist() == values
        assert torch.all(actual[len(values) :] == 77)


def test_v4_decode_rejects_borrowed_positions_before_writing(monkeypatch):
    from tests.test_h2d_attention_publication import decode_batch

    runner, builder = make_runner(monkeypatch)
    # The smaller indexer fixture names its common buffers separately.
    runner.h2d_groups["prefill"] = runner.h2d_owner.group(
        "prefill",
        [
            runner.forward_vars[name]._publication
            for name in ("context_lens", "block_tables")
        ],
    )
    runner.arange_np = np.arange(32, dtype=np.int64)
    batch = decode_batch(2)
    batch.num_spec_step = 0
    positions = runner.forward_vars["positions"]
    tables = runner.forward_vars["block_tables"]
    positions.cpu.fill_(7)
    cu = runner.forward_vars["cu_seqlens_q"]
    cu.cpu[:3] = torch.tensor([0, 1, 2])
    runner._gate_staging_reuse()
    torch.cuda._sleep(20_000_000)
    positions.copy_to_gpu(4)
    tables.copy_to_gpu(2)
    observed = positions.gpu[:4].clone()
    try:
        with pytest.raises(PublicationError, match="republish_reason"):
            builder.prepare_decode(batch, 4, 4, 1)
    finally:
        runner._mark_staging_h2d_enqueued()
        torch.cuda.synchronize()
    assert positions.cpu[:4].tolist() == [7] * 4
    assert observed.tolist() == [7] * 4
