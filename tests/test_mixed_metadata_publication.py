# SPDX-License-Identifier: MIT
"""Mixed metadata uses independent publication banks and prebuilt TBO children."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("aiter")

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.model_runner import ModelRunner, tokenIDProcessor
from atom.model_ops.attentions import backends
from atom.model_ops.attentions import deepseek_v4_attn as v4
from atom.spec_decode.dspark_scheduler import resize_decode_tokens
from atom.utils import CpuGpuBuffer
from atom.utils.tbo.ubatch_splitting import split_mixed_token_midpoint
from tests.test_mixed_decode_view import make_batch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU H2D")


def _runner(monkeypatch, slots):
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.block_size = 256
    runner.max_num_batched_tokens = 32
    runner.max_bs = 8
    runner.arange_np = np.arange(32, dtype=np.int32)
    runner.enforce_eager = True
    runner.use_mrope = False
    runner.config = SimpleNamespace(
        pipeline_parallel_size=slots,
        enable_tbo=True,
        enable_tbo_decode=True,
        prefill_context_parallel_size=1,
        max_num_seqs=8,
        max_model_len=512,
        hf_config=SimpleNamespace(num_attention_heads=8),
    )
    runner.pool_plan = SimpleNamespace(entries={STATE_SLOT_CLASS: 64})
    runner.tokenID_processor = SimpleNamespace(
        num_rejected=None, is_deferred_out=slots == 1
    )
    runner.tokenID_processor.verify_ctx_lens = tokenIDProcessor.verify_ctx_lens.__get__(
        runner.tokenID_processor
    )
    runner.forward_vars = {
        name: CpuGpuBuffer(32, dtype=torch.int32, device=runner.device)
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
    # All row slicing, staging, uploads, plans and TBO assembly run unmodified.
    builder._attach_v4_paged_decode_meta = lambda **kw: None
    builder._attach_v4_index_rows = lambda *a, **kw: None
    builder._build_paged_prefill_meta = lambda *a, **kw: None
    builder._alloc_v4_metadata_buffers()
    runner.attn_metadata_builder = builder
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    builder.build_mixed_prefill_banks()
    return runner, builder


@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("tbo", [False, True])
@pytest.mark.parametrize("spec,query_width", [(0, 1), (1, 1), (1, 2)])
def test_mixed_stage_publish_finish_preserves_both_segments(
    monkeypatch, slots, tbo, spec, query_width
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
        parts = (
            split_mixed_token_midpoint(5, 2, 16, batch.num_scheduled_tokens)
            if tbo
            else None
        )
        torch.cuda._sleep(2_000_000)
        metadata, positions = builder.prepare_mixed(
            batch, 8, batch.total_tokens_num, parts
        )
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
        if tbo:
            assert len(metadata.ubatches) == 2
            assert not metadata.ubatches[0].is_mixed
            tail = metadata.ubatches[-1]
            assert tail.decode_attn_metadata is decode
            assert tail.num_prefill_tokens == 16 - parts[0].running_tokens
            assert tail.cu_seqlens_q[-1].item() == parts[-1].running_tokens
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
