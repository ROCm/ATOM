# SPDX-License-Identifier: MIT
"""Production V4 FP4 writers, paged logits/top-k, and persistent graph plans."""

import importlib
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires gfx950")


def _opus_scale(natural):
    groups = natural.numel() // 256
    return (
        natural.reshape(groups, 2, 32, 2, 2)
        .permute(0, 4, 2, 3, 1)
        .reshape(groups, 2, 32, 4)
        .contiguous()
    )


def _dequant(packed, scales):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    values = lut[nibbles].flatten(-2)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


@pytest.fixture(params=[1, 4], ids=["single_wave", "k_split"])
def written_inputs(request, monkeypatch):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("requires gfx950")
    from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
        FP4_GFX950_FLYDSL,
        FP4_GFX950_OPUS,
        FP4_GFX1250_NATURAL,
    )
    from atom.models.deepseek_v4 import Indexer

    torch.manual_seed(901)
    dev = "cuda"
    t, nb = 9, 6
    angles = torch.randn(1024, 32, device=dev)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    q = torch.randn(t, 64, 128, device=dev, dtype=torch.bfloat16)
    q *= torch.exp2(torch.arange(64, device=dev) % 7 - 3).view(1, 64, 1)
    weights = torch.randn(t, 64, device=dev, dtype=torch.bfloat16)
    positions = torch.arange(t, device=dev, dtype=torch.int64)
    stub = SimpleNamespace(
        _indexer_fp4=True,
        n_heads=64,
        head_dim=128,
        rope_head_dim=64,
        rotary_emb=SimpleNamespace(cos_cache=cos, sin_cache=sin),
        wq_b=lambda *args, **kwargs: q,
        weights_proj=lambda x: weights,
    )
    quantized = {}
    for layout in (FP4_GFX1250_NATURAL, FP4_GFX950_FLYDSL, FP4_GFX950_OPUS):
        stub.indexer_layout = layout
        quantized[layout] = Indexer.forward_pre(stub, q, q, positions)
    q_nat, _, qs_nat = quantized[FP4_GFX1250_NATURAL]
    q_old, _, qs_old = quantized[FP4_GFX950_FLYDSL]
    q_opus, _, qs_opus = quantized[FP4_GFX950_OPUS]
    torch.testing.assert_close(q_opus, q_nat, rtol=0, atol=0)
    torch.testing.assert_close(q_opus, q_old, rtol=0, atol=0)
    torch.testing.assert_close(qs_opus, _opus_scale(qs_nat), rtol=0, atol=0)
    torch.testing.assert_close(
        qs_old.reshape(t, 4, 16, 4).permute(0, 3, 2, 1).reshape(t, 64, 4),
        qs_nat,
        rtol=0,
        atol=0,
    )

    # Two sequences cross a page boundary; shuffled physical pages and sentinel plans.
    context = 512
    rows = []
    for b in range(2):
        # Start with a complete overlap window. Slot zero is an existing zero
        # key; subsequent pages still exercise every physical slot, including 0.
        for pos in range(7, context, 4):
            rows.append([b * context + pos, b, pos, 4 if pos == 7 else 0])
    plan = torch.tensor(rows + [[-1, -1, -1, -1]] * 3, device=dev, dtype=torch.int32)
    bt = torch.tensor([[3, 1, 0, 0], [4, 2, 0, 0]], device=dev, dtype=torch.int32)
    source = torch.randn(2 * context, 512, device=dev, dtype=torch.bfloat16)
    compress = importlib.import_module("atom.model_ops.v4_kernels.fused_compress")
    from aiter.ops.flydsl.kernels.fused_compress_attn import flydsl_fused_compress_attn

    monkeypatch.setattr(
        compress,
        "flydsl_fused_compress_attn",
        partial(flydsl_fused_compress_attn, k_split_num_waves=request.param),
    )
    args = {
        "kv_in": source[:, :256],
        "score_in": source[:, 256:],
        "kv_state": torch.randn(2, 8, 256, device=dev),
        "score_state": torch.randn(2, 8, 256, device=dev),
        "plan": SimpleNamespace(compress_plan_gpu=plan, num_compress=len(rows)),
        "state_slot_mapping": torch.tensor([1, 0], device=dev, dtype=torch.int32),
        "ape": torch.randn(4, 256, device=dev),
        "rms_weight": torch.exp2(torch.arange(128, device=dev) // 32 - 2).float(),
        "rms_eps": 1e-6,
        "cos_cache": cos,
        "sin_cache": sin,
        "block_tables": bt,
        "k_per_block": 64,
        "overlap": True,
        "ratio": 4,
        "head_dim": 128,
        "rope_head_dim": 64,
    }
    kv_old = torch.zeros(nb, 1, 4, 64, 16, device=dev, dtype=torch.uint8)
    ks_old = torch.zeros(nb, 1, 4, 64, device=dev, dtype=torch.uint8)
    kv = torch.zeros(nb, 4, 64, 16, device=dev, dtype=torch.uint8)
    ks = torch.zeros(nb, 2, 32, 4, device=dev, dtype=torch.uint8)
    compress.fused_compress_attn(
        **args, kv_cache=kv_old, cache_scale=ks_old, quant_mode="fp4_gfx950_flydsl"
    )
    compress.fused_compress_attn(
        **args, kv_cache=kv, cache_scale=ks, quant_mode="fp4_gfx950_opus"
    )
    torch.testing.assert_close(kv, kv_old.squeeze(1), rtol=0, atol=0)
    ks_nat = ks_old.reshape(nb, 4, 16, 4).permute(0, 3, 2, 1).reshape(nb, 64, 4)
    torch.testing.assert_close(ks, _opus_scale(ks_nat), rtol=0, atol=0)
    kv_nat = kv.permute(0, 2, 1, 3).reshape(nb, 64, 64)
    k_dq = _dequant(kv_nat, ks_nat)
    k_seq = k_dq[bt[:, :2].long()].reshape(2, 128, 128)
    q_dq = _dequant(q_nat, qs_nat)
    return SimpleNamespace(
        q=q_opus,
        qs=qs_opus,
        kv=kv,
        ks=ks,
        bt=bt,
        weights=weights,
        q_dq=q_dq,
        k_seq=k_seq,
        qs_flydsl=qs_old,
        kv_flydsl=kv_old,
        ks_flydsl=ks_old,
    )


def _reference(inp, row_to_batch, ends):
    logits = torch.einsum("thd,tkd->thk", inp.q_dq, inp.k_seq[row_to_batch.long()])
    logits = (logits.relu() * inp.weights.float().unsqueeze(-1)).sum(1) * 0.125
    mask = torch.arange(128, device="cuda")[None, :] < ends[:, None]
    return logits.masked_fill(~mask, -torch.inf)


def test_writers_logits_topk_and_graph_replay(written_inputs, monkeypatch):
    from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
        pa_mqa_logits_mxfp4,
        pa_mqa_logits_mxfp4_plan,
        pa_mqa_logits_mxfp4_plan_buffers,
    )

    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder as Builder,
    )
    from atom.models.deepseek_v4 import Indexer

    inp = written_inputs
    cu = torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32)
    row_map = torch.tensor([0] * 4 + [1] * 5, device="cuda", dtype=torch.int32)
    ends = torch.tensor(
        [0, 1, 63, 64, 65, 100, 127, 128, 0], device="cuda", dtype=torch.int32
    )
    buffers = pa_mqa_logits_mxfp4_plan_buffers("cuda", 9, 2, variant="qlen1_kv64")
    builder = object.__new__(Builder)
    builder.indexer_layout = "fp4-gfx950-opus"
    builder.device = torch.device("cuda")
    builder.model_runner = SimpleNamespace(
        forward_vars=builder._indexer_staging_buffers(2, 9)
    )
    builder._v4_fp4_opus_plan_buffers = {"": {"qlen1_kv64": buffers}}
    md = SimpleNamespace(
        cu_seqlens_q=cu,
        csa_n_committed_per_token=ends,
        batch_id_per_q_token=row_map,
        max_seqlen_q=5,
        n_committed_csa_per_seq_cpu=np.array([128, 128]),
    )
    scorer = object.__new__(Indexer)
    torch.nn.Module.__init__(scorer)
    scorer.indexer_layout = "fp4-gfx950-opus"
    scorer.kv_cache, scorer.kv_scale = inp.kv, inp.ks
    scorer._max_model_len_idx, scorer._weights_scale = 128, 0.125
    positions = torch.empty(9, device="cuda")
    meta = {}
    builder._refresh_fp4_opus_decode_plan(md, positions, meta)

    # The per-layer scorer only reads the visible window, including -1 for empty rows.
    scorer._score_topk_decode_fp4(inp.q, inp.qs, inp.bt, inp.weights, meta, 8)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_top = scorer._score_topk_decode_fp4(
            inp.q, inp.qs, inp.bt, inp.weights, meta, 8
        )
    for iteration in range(3):
        if iteration:
            ends.copy_(torch.roll(ends, 1))
            row_map.copy_(1 - row_map)
        builder._refresh_fp4_opus_decode_plan(md, positions, meta)
        graph.replay()
        eager = scorer._score_topk_decode_fp4(
            inp.q, inp.qs, inp.bt, inp.weights, meta, 8
        )
        torch.testing.assert_close(graph_top, eager, rtol=0, atol=0)
        ref = _reference(inp, row_map, ends)
        assert torch.isfinite(inp.q_dq).all(), "nonfinite Q reference"
        assert torch.isfinite(inp.k_seq).all(), "nonfinite K reference"
        assert not torch.isnan(ref).any(), "nonfinite logits reference"
        for row, end in enumerate(ends.cpu().tolist()):
            valid = min(end, 8)
            selected = graph_top[row, :valid].long()
            if valid:
                assert ((selected >= 0) & (selected < end)).all(), (
                    iteration,
                    row,
                    end,
                    selected.cpu().tolist(),
                )
                torch.testing.assert_close(
                    ref[row, selected].sort().values,
                    ref[row].topk(valid).values.sort().values,
                    rtol=2e-3,
                    atol=2e-3,
                )
            assert (graph_top[row, valid:] == -1).all()

        for variant in ("qlen1_kv64", "qlen1_kv256"):
            buf = pa_mqa_logits_mxfp4_plan_buffers("cuda", 9, 2, variant=variant)
            plan = pa_mqa_logits_mxfp4_plan(
                cu, ends, buffers=buf, total_q=9, row_to_batch=row_map
            )
            logits = pa_mqa_logits_mxfp4(
                inp.q,
                inp.qs,
                inp.kv,
                inp.ks,
                inp.bt,
                inp.weights,
                plan,
                128,
                weight_scale=0.125,
            )
            torch.testing.assert_close(logits, ref, rtol=2e-3, atol=2e-3)

    # Prefill chunks cross a sequence boundary and omit a trailing PCP dummy row.
    attn = importlib.import_module("atom.model_ops.attentions.deepseek_v4_attn")
    monkeypatch.setattr(attn, "sparse_indexer_row_chunk", lambda *args: 3)
    md.csa_n_committed_per_token = ends
    md.batch_id_per_q_token = row_map.clone()
    md.batch_id_per_q_token[-1] = -1
    prefill = {}
    builder._build_fp4_opus_prefill_plans(
        attn_metadata=md,
        meta=prefill,
        total_tokens=9,
        scheduled_bs=2,
        visible_end_gpu=ends,
        cu_seqlens_q_cpu=np.array([0, 4, 9], dtype=np.int32),
        reuse_cu_seqlens_q=True,
        plan_total_tokens=8,
    )
    prefill["visible_end_gpu"] = ends
    assert [(start, end) for start, end, _ in prefill["fp4_opus_prefill_chunks"]] == [
        (0, 3),
        (3, 6),
        (6, 8),
    ]
    pre_top = scorer._score_topk_prefill_fp4(
        inp.q, inp.qs, inp.bt, inp.weights, prefill, 8
    )
    torch.testing.assert_close(pre_top[:8], eager[:8], rtol=0, atol=0)
    # The dummy row is skipped by consumers; its top-k storage is unspecified.

    # Exercise the explicit FlyDSL escape hatch with the same quantized values.
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
    )

    from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
        fp4_indexer_layout_for_arch,
    )
    from atom.utils import envs

    monkeypatch.setenv("ATOM_V4_UNIFIED_MQA", "flydsl")
    scorer.indexer_layout = fp4_indexer_layout_for_arch(
        "gfx950", envs.ATOM_V4_UNIFIED_MQA
    )
    scorer.kv_cache, scorer.kv_scale = inp.kv_flydsl, inp.ks_flydsl
    model = importlib.import_module("atom.models.deepseek_v4")
    monkeypatch.setattr(
        model, "get_forward_context", lambda: SimpleNamespace(attn_metadata=md)
    )
    starts = torch.zeros_like(ends)
    _, cta, ncta = compute_prefill_schedule(row_map, starts, ends, 256, 4096, 128)
    old_meta = {
        "fp4_local_starts": starts,
        "fp4_local_ends": ends,
        "fp4_cta_info": cta,
        "fp4_n_ctas": ncta,
        "visible_end_gpu": ends,
        "fp4_prefill_local_starts": starts,
        "fp4_prefill_cta_info": cta,
        "fp4_prefill_n_ctas": ncta,
        "fp4_prefill_max_seq_len": 128,
    }
    for score in (scorer._score_topk_decode_fp4, scorer._score_topk_prefill_fp4):
        selected = score(inp.q, inp.qs_flydsl, inp.bt, inp.weights, old_meta, 8)
        for row, end in enumerate(ends.cpu().tolist()):
            valid = min(end, 8)
            assert (selected[row, valid:] == -1).all()
            if valid:
                torch.testing.assert_close(
                    ref[row, selected[row, :valid].long()].sort().values,
                    ref[row].topk(valid).values.sort().values,
                    rtol=2e-3,
                    atol=2e-3,
                )


@pytest.mark.parametrize("variant", ["qlen1_kv64", "qlen1_kv256"])
def test_opus_plan_bounds_oversized_query_buffers(written_inputs, variant):
    from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
        pa_mqa_logits_mxfp4,
        pa_mqa_logits_mxfp4_plan,
        pa_mqa_logits_mxfp4_plan_buffers,
    )

    inp = written_inputs
    # The last Q/window row is allocated but outside this forward's row domain.
    cu = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
    row_map = torch.tensor([0] * 4 + [1] * 5, device="cuda", dtype=torch.int32)
    ends = torch.tensor(
        [1, 63, 64, 65, 100, 127, 128, 63, 128], device="cuda", dtype=torch.int32
    )
    buffers = pa_mqa_logits_mxfp4_plan_buffers("cuda", 9, 2, variant=variant)
    plan = pa_mqa_logits_mxfp4_plan(
        cu, ends, buffers=buffers, total_q=8, row_to_batch=row_map
    )
    assert plan.num_rows == 8
    assert plan.variant.arch == "gfx950"
    logits = pa_mqa_logits_mxfp4(
        inp.q,
        inp.qs,
        inp.kv,
        inp.ks,
        inp.bt,
        inp.weights,
        plan,
        128,
        weight_scale=0.125,
    )
    ref = _reference(inp, row_map, ends)
    torch.testing.assert_close(logits[:8], ref[:8], rtol=2e-3, atol=2e-3)
    assert torch.isneginf(logits[8:]).all()
