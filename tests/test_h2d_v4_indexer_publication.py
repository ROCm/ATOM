# SPDX-License-Identifier: MIT
"""V4 indexer/PCP producers use the forward slot through late TBO prepare."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import CpuGpuBuffer
from atom.utils.h2d import PublicationError

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


def make_runner(monkeypatch, *, pp_size=1, base=None):
    import atom.config
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.config = SimpleNamespace(
        pipeline_parallel_size=pp_size,
        enable_tbo=True,
        enable_tbo_decode=True,
        prefill_context_parallel_size=1,
        max_num_seqs=4,
    )
    monkeypatch.setattr(atom.config, "_current_atom_config", runner.config)
    runner.enforce_eager = True
    runner.tokenID_processor = SimpleNamespace()
    runner.pool_plan = SimpleNamespace(entries={})
    runner.forward_vars = {}
    for name, capacity, dtype, group in (
        ("input_ids", 32, torch.int32, None),
        ("decode_src", 32, torch.int32, None),
        ("batch_info", 32, torch.int32, None),
        ("positions", 32, torch.int64, "positions"),
        ("context_lens", 4, torch.int32, "context"),
        ("cu_seqlens_q", 5, torch.int32, "early"),
        ("cu_seqlens_k", 5, torch.int32, "cu_k"),
        ("batch_id_per_q_token", 32, torch.int32, "early"),
        ("slot_mapping", 32, torch.int64, "prefill"),
        ("num_cached_tokens", 4, torch.int32, "prefill"),
        ("seq_starts", 4, torch.int32, "prefill"),
        # A decode TBO runner's (`prepare_query_metadata`).
        ("ubatch_token_order", 32, torch.int64, "early"),
        ("ubatch_scheduled_tokens", 2, torch.int32, "early"),
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
    builder.max_model_len_idx = 8
    # The fp8 decode kernel's split plan and the indexer's decode rows are
    # not under test here.
    builder._decode_split_plan = lambda rows, kv_len: None
    builder._index_rows = lambda *args, **kwargs: None
    builder._unique_compress_ratios_overlap = [(4, True)]
    builder._indexer_fp4 = False
    builder.csa_layers = [0]
    builder.indexer_layout = "fp8"
    builder._indexer_qsplit_world = 1
    builder.pool_geometry = SimpleNamespace(slot_positions=4)
    builder._alloc_v4_metadata_buffers()
    runner.attn_metadata_builder = builder
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    return runner, builder


def make_metadata(builder, lengths=(3, 5, 3), starts=(8, 16, 0), *, stage_map=True):
    from atom.model_ops.attentions.deepseek_v4_attn import AttentionMetaData_DSV4
    from atom.utils.forward_context import AttentionMetaData, AttnState

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
    md.n_committed_csa_per_seq_cpu = (starts + lengths) // 4
    if stage_map:
        builder._attach_v4_per_fwd_meta(
            md,
            builder._stage(
                "batch_id_per_q_token",
                np.repeat(np.arange(len(lengths), dtype=np.int32), lengths),
            ),
            md.state_slot_out_cpu,
            len(lengths),
            len(positions),
            running_tokens=len(positions),
        )
    return md, pos_gpu, cu, lengths


def prefill_batch(builder, bs, **fields):
    """The ``bs``-request prefill batch `make_metadata` staged the step of."""
    var = builder.model_runner.forward_vars
    return SimpleNamespace(
        total_seqs_num_prefill=bs,
        context_lens=var["context_lens"].np[:bs].copy(),
        num_scheduled_tokens=np.diff(var["cu_seqlens_q"].np[: bs + 1]),
        block_tables=[],
        is_dummy_run=False,
        **fields,
    )


def split_prefill(builder, md, slices, pcp_groups=None):
    """`prepare_prefill`'s micro-batch half over the step ``md``: stage, the
    one upload, finish. Returns the micro-batches' metadata."""
    batch = prefill_batch(builder, len(md.state_slot_out_cpu))
    ubatches = builder._prefill_ubatches(batch, slices, pcp_groups)
    for group in builder.stage_ubatch_prefill(batch, md, ubatches, pcp_groups):
        group.publish()
    builder.finish_ubatch_prefill(batch, md, ubatches, pcp_groups)
    return md.ubatches


def set_pcp(monkeypatch, size, rank):
    from atom.model_ops.attentions import deepseek_v4_attn as module

    monkeypatch.setattr(module, "pcp_is_enabled", lambda: True)
    monkeypatch.setattr(module, "get_pcp_world_size", lambda: size)
    monkeypatch.setattr(module, "get_pcp_rank", lambda: rank)


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
        local = builder._apply_pcp_reindex(md, positions, 3, 3, 11, cu)
        expected_cu = np.r_[0, np.cumsum(md.n_committed_csa_per_seq_cpu)].astype(
            np.int32
        )
        expected_cu[-1] = max(expected_cu[-1], 1)
        saved.append(
            (
                md.index_rows.concat.cu_committed.clone(),
                expected_cu,
                md.batch_id_per_q_token.clone(),
                md.kv_indices_extend.clone(),
                local.clone(),
            )
        )
        before = buf.cpu.clone()
        with pytest.raises(PublicationError, match="republish_reason"):
            builder._stage_prefill_index_keys(md.n_committed_csa_per_seq_cpu, 3)
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
    runner._gate_staging_reuse()
    md, pos, _, _ = make_metadata(builder, stage_map=False)
    # What `stage_ubatch_prefill` leaves on a PCP prefill cut into groups.
    md.pcp_groups = (SimpleNamespace(),) * 2 if balanced else None
    monkeypatch.setattr(
        CommonAttentionBuilder, "stage_prefill", lambda *args: (md, pos)
    )
    batch = prefill_batch(
        builder,
        3,
        total_tokens_num_prefill=11,
        state_slots_committed=[],
        state_fork_srcs=None,
    )
    batch.is_dummy_run = dummy
    actual, _ = builder.prepare_prefill(batch, 3, batch.total_tokens_num_prefill)
    binding = runner.forward_vars["v4_indexer_cu_committed"]._publication
    assert (binding._epoch == runner.h2d_owner.epoch) == (dummy or not balanced)
    assert len(actual.batch_id_per_q_token) == (11 if dummy or balanced else 3)
    runner._mark_staging_h2d_enqueued()


@pytest.mark.parametrize("pp_size", [1, 2])
def test_token_split_tbo_publishes_with_the_step_and_preserves_both_prefixes(
    monkeypatch, pp_size
):
    from atom.utils.tbo.ubatch_splitting import UBatchSlice

    runner, builder = make_runner(monkeypatch, pp_size=pp_size)
    saved = []
    slices = (
        UBatchSlice(slice(0, 1), slice(0, 4)),
        UBatchSlice(slice(0, 2), slice(4, 9)),
    )
    for step in range(5):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        md, _, _, _ = make_metadata(builder, (5, 4), (8 + 4 * step, 20))
        epoch = runner.h2d_owner.epoch
        torch.cuda._sleep(5_000_000)
        ubatches = split_prefill(builder, md, slices)
        assert runner.h2d_owner.epoch == epoch
        for index, ub in enumerate(ubatches):
            buf = runner.forward_vars[f"ub{index}_cu_seqlens_q"]
            assert ub.cu_seqlens_q.data_ptr() == buf.gpu.data_ptr()
        # Read after BOTH preparations. A snapshot taken before the second
        # producer could conceal a shared-storage overwrite.
        for index, ub in enumerate(ubatches):
            assert (
                ub.batch_id_per_q_token.data_ptr()
                == runner.forward_vars[f"ub{index}_batch_id_per_q_token"].gpu.data_ptr()
            )
            assert (
                ub.index_rows.concat.cu_committed.data_ptr()
                == runner.forward_vars[
                    f"ub{index}_v4_indexer_cu_committed"
                ].gpu.data_ptr()
            )
            saved.append(
                (
                    index,
                    ub.cu_seqlens_q.clone(),
                    ub.index_rows.concat.cu_committed.clone(),
                    step,
                )
            )
        before = runner.forward_vars["ub0_cu_seqlens_q"].cpu.clone()
        batch = prefill_batch(builder, 2)
        ubatches = builder._prefill_ubatches(batch, slices, None)
        with pytest.raises(PublicationError, match="republish_reason"):
            builder.stage_ubatch_prefill(batch, md, ubatches, None)
        assert torch.equal(runner.forward_vars["ub0_cu_seqlens_q"].cpu, before)
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for index, query, committed, step in saved:
        assert query.cpu().tolist() == ([0, 4] if index == 0 else [0, 1, 5])
        assert committed.cpu().tolist() == (
            [0, 3 + step] if index == 0 else [0, 3 + step, 9 + step]
        )


def test_tbo_prefill_state_slots_span_the_micro_batch_running_bs(monkeypatch):
    """A micro-batch's state slots run out to its `running_bs`, as its query
    prefix does: one with fewer requests (an idle rank's, none) still hands
    `swa_write` a slot for every request its prefix describes."""
    from atom.utils.tbo.ubatch_splitting import UBatchSlice

    runner, builder = make_runner(monkeypatch)
    runner._advance_forward_vars()
    runner._gate_staging_reuse()
    md, _, _, _ = make_metadata(builder, (5, 4), (8, 20))
    [ub] = split_prefill(
        builder, md, (UBatchSlice(slice(0, 1), slice(0, 4), running_bs=2),)
    )
    runner._mark_staging_h2d_enqueued()
    assert ub.cu_seqlens_q.shape[0] == 3
    assert ub.state_slot_out.shape[0] == ub.state_slot_in.shape[0] == 2
    torch.cuda.synchronize()


@pytest.mark.parametrize("rank", [0, 3])
def test_balanced_pcp_tbo_uses_separate_indexer_buffers(monkeypatch, rank):
    from atom.utils.tbo.ubatch_splitting import UBatchSlice

    runner, builder = make_runner(monkeypatch)
    set_pcp(monkeypatch, 4, rank)
    groups = (
        SimpleNamespace(req_start=0, req_stop=2, tok_start=0, tok_end=8),
        SimpleNamespace(req_start=2, req_stop=4, tok_start=8, tok_end=15),
    )
    slices = tuple(
        UBatchSlice(slice(g.req_start, g.req_stop), slice(g.tok_start, g.tok_end))
        for g in groups
    )
    runner._gate_staging_reuse()
    md, _, _, _ = make_metadata(builder, (3, 5, 3, 4), (8, 16, 28, 40))
    torch.cuda._sleep(5_000_000)
    ubatches = split_prefill(builder, md, slices, groups)
    runner._mark_staging_h2d_enqueued()
    runner.h2d_owner.completion.synchronize()
    assert ubatches[0].index_rows.concat.cu_committed.cpu().tolist() == [0, 2, 7]
    assert ubatches[1].index_rows.concat.cu_committed.cpu().tolist() == [0, 7, 18]
    assert [ub.max_seqlen_q for ub in ubatches] == [5, 4]
    for index, ub in enumerate(ubatches):
        assert (
            ub.index_rows.concat.cu_committed.data_ptr()
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


def test_ubatch_staging_rejects_a_wrong_stream_before_writing(monkeypatch):
    from atom.utils.tbo.ubatch_splitting import UBatchSlice

    runner, builder = make_runner(monkeypatch)
    runner._gate_staging_reuse()
    md, _, _, _ = make_metadata(builder, (5, 4), (8, 20))
    batch = prefill_batch(builder, 2)
    ubatches = builder._prefill_ubatches(
        batch, (UBatchSlice(slice(0, 1), slice(0, 4)),), None
    )
    before = runner.forward_vars["ub0_cu_seqlens_q"].cpu.clone()
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(PublicationError, match="owner's current compute stream"),
    ):
        builder.stage_ubatch_prefill(batch, md, ubatches, None)
    assert torch.equal(runner.forward_vars["ub0_cu_seqlens_q"].cpu, before)
    runner._mark_staging_h2d_enqueued()


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("kv_fp8", [False, True])
def test_decode_ubatch_inputs_padding_and_graph_consumers(
    monkeypatch, pp_size, padded, kv_fp8
):
    from atom.model_ops.attentions import deepseek_v4_attn as module
    from atom.model_ops.attentions.deepseek_v4_attn import AttentionMetaData_DSV4
    from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
        UnifiedPoolGeometry,
    )
    from atom.utils.tbo.ubatch_splitting import cut_decode_ubatches

    runner, builder = make_runner(monkeypatch, pp_size=pp_size)
    builder.pool_geometry = UnifiedPoolGeometry([0, 4, 128], 8, 4, 6, 256)
    builder.hca_rows_per_block = 2
    builder._kv_fp8 = kv_fp8
    runner.enforce_eager = not padded
    if padded:
        monkeypatch.setattr(module, "get_forward_context", lambda: None)
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
        # The step's host mirrors, as `stage_decode` leaves them.
        var["positions"].np[: len(positions)] = positions
        var["context_lens"].np[: len(lens)] = starts + lengths
        var["v4_meta_state_slot_out"].np[: len(lens)] = slots
        var["v4_meta_state_slot_in"].np[: len(lens)] = slots[::-1]
        cu = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
        running_bs = 4 if padded else len(lens)
        args = (
            len(lens),
            lengths,
            cu,
            cut_decode_ubatches(running_bs, 3 * running_bs, lengths),
            3,
        )
        global_qo = np.minimum(np.arange(13, dtype=np.int32), len(positions))
        if kv_fp8:
            builder._stage("v4_qo_indptr", global_qo)
        torch.cuda._sleep(5_000_000)
        md = AttentionMetaData_DSV4()
        for group in builder.stage_ubatch_decode(md, *args):
            group.publish()
        builder.finish_ubatch_decode(md, args[0], args[3], args[4])
        if kv_fp8:
            # The actual serving path prepared the global verify prefix first.
            np.testing.assert_array_equal(var["v4_qo_indptr"].np[:13], global_qo)
            saved.append((var["v4_qo_indptr"].gpu[:13].clone(), global_qo.tolist()))
        graphs[runner._fv_idx].replay()
        for index, ub in enumerate(args[3]):
            lo, hi = ub.request_slice.start, ub.request_slice.stop
            width, token_width = ub.running_bs, ub.running_tokens
            local_lens = lengths[lo:hi]
            tokens = int(local_lens.sum())
            local_pos = positions[
                int(lengths[:lo].sum()) : int(lengths[:hi].sum())
            ].tolist()
            expected = {
                "positions": local_pos + [0] * (token_width - tokens),
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
                + [-1] * (token_width - tokens),
            }
            if kv_fp8:
                # An empty ubatch still publishes its prefix out to its running
                # rows: a replayed graph reads every one of them.
                expected["v4_qo_indptr"] = list(range(tokens + 1)) + [tokens] * (
                    token_width - tokens
                )
            for suffix, values in expected.items():
                name = f"ub{index}_{suffix}"
                saved.append((outputs[runner._fv_idx][name].clone(), values))
        before = {name: var[name].cpu.clone() for name in names}
        with pytest.raises(PublicationError, match="republish_reason"):
            builder.stage_ubatch_decode(AttentionMetaData_DSV4(), *args)
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
    runner.h2d_groups["seq_metadata"] = runner.h2d_owner.group(
        "seq_metadata",
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
