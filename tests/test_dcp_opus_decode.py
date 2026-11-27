# SPDX-License-Identifier: MIT
"""CPU checks of the DCP OPUS scorer's tensor and launch contracts."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
from atom.model_ops import dcp_ops


@pytest.mark.parametrize("heads,qlen,block", [(32, 1, 64), (32, 6, 64), (64, 6, 128)])
def test_opus_decode_flattens_queries_and_preserves_planes(
    monkeypatch, heads, qlen, block
):
    bs, world, topk = 2, 8, 4
    rows = bs * qlen
    seen = {}
    opus = ModuleType("aiter.ops.opus.pa_mqa_logits_mxfp4")
    topk_mod = ModuleType("aiter.ops.topk")
    fp8 = ModuleType("aiter.ops.triton.pa_mqa_logits")
    ragged = ModuleType("atom.model_ops.fp4_mqa_ragged_metadata")

    def configure(device, **kwargs):
        seen["config"] = kwargs
        return SimpleNamespace(device=device)

    def score(q, qs, cache, scales, weights, tables, starts, ends, **kwargs):
        seen["score"] = (q, qs, cache, scales, weights, tables, starts, ends, kwargs)
        kwargs["out"].zero_()
        return kwargs["out"]

    def select(logits, next_n, ends, idx, count, *args, values, **kwargs):
        seen["select"] = (logits, next_n, ends, count)
        idx.zero_()
        values.zero_()

    def merge(scores, idx, tables, out_idx, out_ptr, counts, *args):
        seen["merge"] = (scores, tables, out_idx, out_ptr, counts)
        out_idx.fill_(17)
        out_ptr.zero_()
        counts.zero_()

    def wrong_scorer(*args, **kwargs):
        pytest.fail("MFMA FP4 decode reached the FP8 scorer")

    def gather(group, x, dim):
        assert dim == -1
        return torch.cat([x] * world, dim=dim)

    opus.mqa_logits_mxfp4_config = configure
    opus.mqa_logits_mxfp4 = score
    topk_mod.top_k_per_row_decode = select
    topk_mod.flydsl_dcp_topk_merge = merge
    fp8.deepgemm_fp8_paged_mqa_logits = wrong_scorer
    ragged.Fp4MqaRaggedMetadata = lambda *args, **kwargs: pytest.fail(
        "MFMA FP4 decode reached the PRESHUFFLE planner"
    )
    for module in (opus, topk_mod, fp8, ragged):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(dcp_ops, "get_dcp_world_size", lambda: world)
    monkeypatch.setattr(dcp_ops, "get_dcp_group", lambda: object())
    monkeypatch.setattr(dcp_ops, "dcp_all_gather", gather)
    # Run the real dispatcher and adapter on CPU; only their allocations and
    # GPU kernels are replaced. No GPU is needed to verify these arguments.
    monkeypatch.setattr(
        dcp_ops,
        "torch",
        SimpleNamespace(
            empty=lambda *args, **kw: torch.empty(*args, **dict(kw, device="cpu")),
            float32=torch.float32,
            int32=torch.int32,
        ),
    )
    query = (
        torch.arange(rows * heads * 64, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
    )
    query = query.view(bs, qlen, heads, 64)
    qs = torch.zeros(rows + 1, 2, 32, heads // 16, dtype=torch.uint8)
    cache = torch.zeros(3, 4, block, 16, dtype=torch.uint8)
    scales = torch.zeros(3, block * 4, dtype=torch.uint8)
    weights = torch.zeros(rows + 1, heads, dtype=torch.bfloat16)
    tables = torch.arange((rows + 1) * 3, dtype=torch.int32).view(rows + 1, 3)
    starts = torch.arange(rows + 2, dtype=torch.int32)
    windows = torch.arange(1, rows + 1, dtype=torch.int32)
    metadata = SimpleNamespace(
        dcp_local_context_lens=windows,
        token_block_tables=tables,
        sparse_cu_seqlens_q=starts,
        context_lens=torch.full((bs,), 64, dtype=torch.int32),
    )
    out = torch.empty(rows * topk, dtype=torch.int32)
    ptr = torch.empty(rows + 1, dtype=torch.int32)
    counts = torch.empty(rows, dtype=torch.int32)
    dcp_ops.dcp_decode_candidate_exchange_fused(
        metadata,
        query,
        cache,
        weights,
        dcp_rank=0,
        num_decode_tokens=rows,
        topk_tokens=topk,
        max_model_len=257,
        runner_block_size=block,
        stable_topk=True,
        cp_kv_cache_interleave_size=1,
        out_kv_indices=out,
        out_kv_indptr=ptr,
        owned_counts=counts,
        q_scale=qs,
        kv_scale=scales,
        weights_scale=0.125,
        cache_layout="mfma",
    )
    q, actual_qs, actual_cache, actual_scales, w, bt, cu, ends, kw = seen["score"]
    assert torch.equal(q, query.view(rows, heads, 64))
    assert actual_qs.data_ptr() == qs.data_ptr() and actual_qs.shape[0] == rows
    assert actual_cache is cache and actual_scales is scales
    assert w.data_ptr() == weights.data_ptr() and w.shape == (rows, heads)
    assert torch.equal(bt, tables[:rows]) and torch.equal(cu, starts[: rows + 1])
    assert ends is windows and kw["weight_scale"] == 0.125
    assert seen["config"] == {
        "heads": heads,
        "page_size": 64,
        "pages_per_block": block // 64,
        "max_rows": rows,
        "max_seqs": rows,
        "max_qlen": 1,
        "max_seq_len": 33,
    }
    logits, next_n, local_ends, count = seen["select"]
    assert logits.shape == (rows, 33) and next_n == 1 and count == rows
    assert local_ends is windows
    gathered, merged_tables, actual_out, actual_ptr, actual_counts = seen["merge"]
    assert gathered.shape == (rows, world * topk)
    assert torch.equal(merged_tables, tables[:rows])
    assert actual_out is out and actual_ptr is ptr and actual_counts is counts
    assert (out == 17).all()
