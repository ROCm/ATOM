# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Mixed metadata uses independent publication banks across steps and slots."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.scheduler import ScheduledBatch
from atom.spec_decode.dspark_scheduler import resize_decode_tokens
from atom.utils import CpuGpuBuffer
from tests.test_mixed_decode_view import make_batch


@pytest.fixture(scope="module", autouse=True)
def kernel_imports():
    with atom_without_aiter():
        yield


def _cpu_query_runner(monkeypatch, slots=1):
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )
    from atom.utils.h2d import PublicationOwner, PublicationRegistry

    registry = PublicationRegistry()

    def bind(variables, event=None, *, partial=False):
        owner = PublicationOwner("cpu", registry=registry)
        bindings = [owner.bind(buffer, name) for name, buffer in variables.items()]
        early = owner.group("early", bindings)
        return owner, {"early": early, "token_inputs": early}

    ring = [
        {
            name: CpuGpuBuffer(
                32,
                dtype=torch.int32,
                device="cpu",
                pin_memory=False,
                publication_group="early",
            )
            for name in ("cu_seqlens_q", "batch_id_per_q_token")
        }
        for _ in range(slots)
    ]
    owners, groups = zip(*(bind(variables) for variables in ring))
    runner = SimpleNamespace(
        forward_vars=ring[0],
        _fv_ring=ring,
        _fv_idx=0,
        h2d_owner=owners[0],
        h2d_groups=groups[0],
        build_h2d_slot=bind,
    )
    builder = DeepseekV4AttentionMetadataBuilder.__new__(
        DeepseekV4AttentionMetadataBuilder
    )
    builder.model_runner = runner
    runner.attn_metadata_builder = builder
    monkeypatch.setattr(torch.cuda, "Event", lambda: None)
    builder.build_mixed_prefill_banks()
    return runner, builder, owners, groups


@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("fail", [False, True])
def test_bank_publishes_prefill_spans_and_restores_live_owner(monkeypatch, slots, fail):
    from atom.utils.h2d import PublicationError

    runner, builder, owners, groups = _cpu_query_runner(monkeypatch, slots)
    batch = make_batch()
    prefill = batch.slice_rows(0, batch.total_seqs_num_prefill)
    for slot in range(slots):
        runner._fv_idx = slot
        live = runner.forward_vars = runner._fv_ring[slot]
        live_owner = runner.h2d_owner = owners[slot]
        live_groups = runner.h2d_groups = groups[slot]
        live_owner.begin()
        builder.publish_query_metadata(np.array([2, 1, 2]), 3, 5)
        bank, owner, bank_groups = builder._get_mixed_prefill_bank()

        def prepare(bank=bank, owner=owner, bank_groups=bank_groups):
            with builder.mixed_prefill_bank_active(prefill):
                assert runner.forward_vars is bank
                assert runner.h2d_owner is owner and runner.h2d_groups is bank_groups
                assert bank["cu_seqlens_q"].gpu[:3].tolist() == [0, 8, 16]
                assert (
                    bank["batch_id_per_q_token"].gpu[:16].tolist() == [0] * 8 + [1] * 8
                )
                with pytest.raises(PublicationError):
                    builder.publish_query_metadata(np.array([8, 8]), 2, 16)
                if fail:
                    raise ValueError("prefill failed")

        if fail:
            with pytest.raises(ValueError, match="prefill failed"):
                prepare()
        else:
            prepare()
        assert runner.forward_vars is live
        assert runner.h2d_owner is live_owner and runner.h2d_groups is live_groups
        assert live["cu_seqlens_q"].gpu[:4].tolist() == [0, 2, 3, 5]
        assert live["batch_id_per_q_token"].gpu[:5].tolist() == [0, 0, 1, 2, 2]
        # finish seals a successful bank; failure prevents reuse altogether.
        with pytest.raises(PublicationError, match="failed" if fail else "sealed"):
            bank_groups["early"].check_writable()
        if fail:
            with pytest.raises(PublicationError, match="failed"):
                owner.begin()
        else:
            with builder.mixed_prefill_bank_active(prefill):
                assert bank["cu_seqlens_q"].gpu[:3].tolist() == [0, 8, 16]
        live_owner.finish()


def _runner(monkeypatch, slots):
    from atom.model_engine.model_runner import ModelRunner, tokenIDProcessor
    from atom.model_ops.attentions import backends
    from atom.model_ops.attentions import deepseek_v4_attn as v4

    runner = ModelRunner.__new__(ModelRunner)
    runner._return_routed_experts = False
    runner.device = torch.device("cuda", 0)
    runner.block_size = 256
    runner.max_num_batched_tokens = 32
    runner.max_bs = 8
    runner.arange_np = np.arange(32, dtype=np.int32)
    runner.enforce_eager = True
    runner.use_mrope = False
    runner.config = SimpleNamespace(
        pipeline_parallel_size=slots,
        enable_tbo=False,
        enable_tbo_decode=False,
        prefill_context_parallel_size=1,
        max_num_seqs=8,
        max_model_len=512,
        hf_config=SimpleNamespace(num_attention_heads=8),
    )
    runner.pool_plan = SimpleNamespace(entries={STATE_SLOT_CLASS: 64})
    runner.attn_family = SimpleNamespace(has_recurrent_state=False)
    runner.tokenID_processor = tokenIDProcessor(
        runner, 32, use_spec=True, num_spec_tokens=1
    )
    runner.forward_vars = {
        name: getattr(runner.tokenID_processor, name)
        for name in ("input_ids", "decode_src", "batch_info")
    }
    runner.forward_vars["positions"] = CpuGpuBuffer(
        32, dtype=torch.int64, device=runner.device, publication_group="positions"
    )
    monkeypatch.setattr(backends, "attn_head_shard_size", lambda: 1)
    monkeypatch.setattr(backends, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(backends, "get_dcp_rank", lambda: 0)
    monkeypatch.setattr(v4, "pcp_is_enabled", lambda: False)
    builder = v4.DeepseekV4AttentionMetadataBuilder.__new__(
        v4.DeepseekV4AttentionMetadataBuilder
    )
    backends.CommonAttentionBuilder.__init__(builder, runner)
    builder.max_decode_tokens = 32
    builder.window_size = 4
    builder.index_topk = 4
    builder.max_committed_hca = 4
    builder.max_spec_steps = 1
    builder.max_model_len_idx = 8
    builder._unique_compress_ratios_overlap = [(4, True)]
    builder._indexer_fp4 = False
    builder._kv_fp8 = True
    builder.csa_layers = [0]
    builder._indexer_qsplit_world = 1
    builder.pool_geometry = SimpleNamespace(slot_positions=64)
    # Only attention-index kernel generation is outside this publication test.
    # All row slicing, staging, uploads and plans run unmodified.
    builder._attach_v4_paged_decode_meta = lambda **kw: None
    builder._attach_v4_index_rows = lambda *a, **kw: None
    builder._build_paged_prefill_meta = lambda *a, **kw: None
    builder._alloc_v4_metadata_buffers()
    runner.attn_metadata_builder = builder
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    builder.build_mixed_prefill_banks()
    return runner, builder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU H2D")
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("query_width", [1, 2])
def test_idle_v4_decode_publishes_padding_with_previous_spec_status(
    monkeypatch, slots, query_width
):
    runner, builder = _runner(monkeypatch, slots)
    previous_rejected = np.array([0, 1, 1], dtype=np.int32)
    processor = runner.tokenID_processor
    processor.num_rejected = previous_rejected
    processor.num_bonus = np.zeros(3, dtype=np.int32)
    processor.prev_batch = SimpleNamespace(req_ids=[10, 11, 12], is_dummy_run=False)
    processor.prev_token_ids = torch.zeros(3, dtype=torch.int32, device=runner.device)
    processor.prev_rejected_num = previous_rejected
    processor.prev_bonus_num = processor.num_bonus
    batch = ScheduledBatch({}, [], 0, num_spec_step=1, is_dummy_run=True)
    assert batch.total_seqs_num_decode == 0 and not batch.is_mixed

    runner._advance_forward_vars()
    runner._gate_staging_reuse()
    builder.publish_query_metadata(batch.num_scheduled_tokens, 4, 4 * query_width)
    group = runner.h2d_groups["input_ids"]
    # prepare_model assembles tokens before attention, rebinding status to
    # this empty batch instead of carrying the previous rows into metadata.
    ids = processor.prepare_input_ids(batch, query_width, group)
    metadata, positions = builder.prepare_decode(batch, 4, 4 * query_width, query_width)
    runner._mark_staging_h2d_enqueued()
    runner._record_forward_vars_event()

    assert metadata.context_lens.numel() == 0
    assert metadata.cu_seqlens_q.cpu().tolist() == [0] * 5
    assert positions.cpu().tolist() == [0] * (4 * query_width)
    assert metadata.state_slot_out.cpu().tolist() == [0] * 4
    assert ids.numel() == 0
    assert processor.num_rejected.size == processor.num_bonus.size == 0
    np.testing.assert_array_equal(previous_rejected, [0, 1, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU H2D")
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("spec,query_width", [(0, 1), (1, 1), (1, 2)])
def test_mixed_stage_publish_finish_preserves_both_segments(
    monkeypatch, slots, spec, query_width
):
    runner, builder = _runner(monkeypatch, slots)
    for step in range(3):
        batch = make_batch(num_spec_step=int(spec))
        if spec and query_width == 1:
            resize_decode_tokens(batch, np.ones(3, dtype=np.int32))
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        live = runner.forward_vars
        runner.tokenID_processor.num_rejected = np.array(
            [0, 0, 1, 0, 1] if spec else [0] * 5
        )
        lengths = batch.num_scheduled_tokens[2:]
        builder.publish_query_metadata(lengths, 3, int(lengths.sum()))
        torch.cuda._sleep(2_000_000)
        metadata, positions = builder.prepare_mixed(batch, 8, batch.total_tokens_num)
        prefill, decode = metadata.prefill_attn_metadata, metadata.decode_attn_metadata
        snapshots = [
            positions.clone(),
            prefill.cu_seqlens_q.clone(),
            decode.cu_seqlens_q.clone(),
            prefill.context_lens.clone(),
            decode.context_lens.clone(),
            metadata.state_slot_out.clone(),
        ]
        assert runner.forward_vars is live
        bank, _, _ = builder._get_mixed_prefill_bank()
        for name in (
            "cu_seqlens_q",
            "context_lens",
            "block_tables",
            "v4_meta_state_slot_out",
        ):
            assert bank[name].gpu.data_ptr() != live[name].gpu.data_ptr()
        assert decode.cu_seqlens_q.data_ptr() == live["cu_seqlens_q"].gpu.data_ptr()
        assert prefill.cu_seqlens_q.data_ptr() == bank["cu_seqlens_q"].gpu.data_ptr()
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
        actual_pos, pcu, dcu, pctx, dctx, state = [t.cpu().tolist() for t in snapshots]
        assert pcu == [0, 8, 16]
        assert dcu == [0, *np.cumsum(lengths)]
        assert pctx == [8, 8]
        shift = 1 if spec and slots == 2 else runner.tokenID_processor.num_rejected[2:]
        expected_ctx = np.array(batch.context_lens[2:]) - shift
        assert dctx == expected_ctx.tolist()
        expected_decode = [
            p
            for ctx, length in zip(expected_ctx, lengths)
            for p in range(
                int(ctx) - batch.num_spec_step - 1,
                int(ctx) - batch.num_spec_step - 1 + int(length),
            )
        ]
        assert actual_pos == list(range(8)) * 2 + expected_decode
        assert state == [63 - v for v in batch.state_slots_committed] + [0] * 3


@pytest.mark.parametrize("spec", [0, 1])
def test_prepare_model_publishes_decode_local_query_spans(monkeypatch, spec):
    from atom.model_engine import model_runner

    batch = make_batch(num_spec_step=spec)
    runner, _, _, _ = _cpu_query_runner(monkeypatch)
    mode = SimpleNamespace(
        sync=SimpleNamespace(any_rank_has_prefill=True),
        is_prefill=True,
        running_bs=batch.total_seqs_num,
        running_tokens=batch.total_tokens_num,
        max_seqlen_q=spec + 1,
        decode_ubatch_slices=None,
    )
    monkeypatch.setattr(model_runner.ForwardMode, "decide", lambda **kwargs: mode)
    runner.config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_tbo=False,
        enable_tbo_decode=False,
    )
    runner._dspark_apply_q_bucket = lambda batch: None
    runner._local_tbo_eligibility = lambda batch: False
    runner._producer_graph_shapes = lambda: None
    runner._cut_ubatches = lambda mode, batch: mode
    runner._verify_spans = lambda batch: None
    runner.prepare_sample = lambda batch, group: (None, None, None, True, False)
    runner.enforce_eager = True
    runner.capture_sizes_np = np.array([], dtype=np.int32)
    runner.running_tokens_per_dp_rank = None
    runner.tbo_decode_graphs = False
    calls = []

    def assemble(batch, width, group):
        # Token assembly is the first consumer of the real query publication.
        group.publish()
        calls.append("tokens")
        cu = runner.forward_vars["cu_seqlens_q"].gpu[:4].tolist()
        assert cu == ([0, 2, 3, 5] if spec else [0, 1, 2, 3])
        return object(), model_runner.TokenLocations(
            np.array([2, 3, 4]), np.array([1, 2, 0]), np.array([0, 1])
        )

    def inputs(batch, input_ids, *, forward_mode, spec_decode_indices):
        assert forward_mode is mode
        calls.append("attention")
        # The carried decode rows must use the previous verify's rejections
        # before attention derives their positions, despite is_prefill=True.
        decode_ctx = np.asarray(batch.context_lens[2:])
        np.testing.assert_array_equal(
            processor.verify_ctx_lens(decode_ctx, processor.num_rejected[2:]),
            decode_ctx - np.array([0, 1, 1]) * spec,
        )

    processor = model_runner.tokenIDProcessor.__new__(model_runner.tokenIDProcessor)
    processor.runner = runner
    processor.batch_info = SimpleNamespace(np=np.zeros(3, dtype=np.int32))
    processor.draft_tokens_per_seq = spec
    processor.prev_batch = SimpleNamespace(total_seqs_num=3)
    processor.is_deferred_out = True
    processor.spec_status_owed = False
    processor._stage_input_ids = assemble
    processor.recv_spec_status = lambda: (
        np.array([1, 0, 1], dtype=np.int32) * spec,
        np.zeros(3, dtype=np.int32),
    )
    runner.h2d_groups["input_ids"] = runner.h2d_groups["token_inputs"]
    runner.tokenID_processor = processor
    runner.prepare_inputs = inputs
    runner.h2d_owner.begin()
    model_runner.ModelRunner.prepare_model(runner, batch)
    assert calls == ["tokens", "attention"]
    ids = runner.forward_vars["batch_id_per_q_token"].gpu[
        : batch.total_tokens_num_decode
    ]
    assert ids.tolist() == ([0, 0, 1, 2, 2] if spec else [0, 1, 2])
    runner.h2d_owner.finish()
