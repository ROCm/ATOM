# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 NVFP4 sparse prefill: the lean per-step launches (``prefill_fast.StepPlan``) give
ATOM's top-16 ids item by item and bit-identical attention outputs over ragged batches
(``ak16``: aiter's ``sparse_pa_nvfp4_prefill``; ``ak16lazy`` / ``v2``: their own host functions). Needs gfx950 and amd-kernels on PYTHONPATH (skipped otherwise)."""

import math

import pytest
import torch

pytest.importorskip("amd_kernels")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers  # noqa: E402
from atom.model_ops.minimax_m3.index_topk import (  # noqa: E402
    build_n_valid_column_per_row,
    minimax_m3_index_topk,
)
from atom.model_ops.minimax_m3.prefill_fast import StepPlan  # noqa: E402
from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import prefill_row_positions  # noqa: E402
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace  # noqa: E402

HQ, D, BLK, TOPK = 16, 128, 128, 16
SM = 1.0 / math.sqrt(D)


def _pages(n, gen):
    """Random NVFP4 pages: any e2m1 nibble, e4m3 scales in a finite positive range."""
    idx = torch.randint(0, 256, (n, 9216), dtype=torch.uint8, device="cuda", generator=gen)
    idx[:, 8192:] = torch.randint(0x30, 0x48, (n, 1024), dtype=torch.uint8, device="cuda",
                                  generator=gen)
    kv = torch.randint(0, 256, (n, 2 * 9216), dtype=torch.uint8, device="cuda", generator=gen)
    for side in range(2):
        b = side * 9216 + 8192
        kv[:, b: b + 1024] = torch.randint(0x30, 0x48, (n, 1024), dtype=torch.uint8,
                                           device="cuda", generator=gen)
    return idx, kv


@pytest.mark.parametrize("attn", ["ak16", "ak16lazy", "v2"])
@pytest.mark.parametrize(
    "reqs,tied",
    [
        ([(700, 20000)], False),
        ([(300, 5000), (1200, 70000), (64, 129)], False),
        # past 1024 blocks ATOM selects with aiter's per-row selector (its own ties at the cut)
        ([(500, 140000), (90, 3000)], False),
        ([(500, 140000), (90, 3000)], True),
    ],
)
def test_plan_matches_the_host_functions(attn, reqs, tied):
    from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.host import (
        nvfp4_sparse_prefill,
    )

    from atom.model_ops.minimax_m3.kernels.sparse_pa_prefill import sparse_pa_prefill

    gen = torch.Generator(device="cuda").manual_seed(7)
    n_new = [n for n, _ in reqs]
    seq = [n + p for n, p in reqs]
    mb = [-(-s // BLK) for s in seq]
    pages = sum(mb) + 3
    idx_pages, kv_pages = _pages(pages, gen)
    if tied:  # every block the same keys: every score of a row ties
        idx_pages[:] = idx_pages[0]
    perm = torch.randperm(pages, device="cuda", generator=gen).to(torch.int32)
    bt = torch.zeros(len(reqs), max(mb), dtype=torch.int32, device="cuda")
    o = 0
    for b, m in enumerate(mb):
        bt[b, :m] = perm[o: o + m]
        o += m
    rows = sum(n_new)
    cu = torch.tensor([0] + list(torch.tensor(n_new).cumsum(0)), dtype=torch.int32, device="cuda")
    seq_t = torch.tensor(seq, dtype=torch.int32, device="cuda")
    pre = torch.tensor([p for _, p in reqs], dtype=torch.int32, device="cuda")
    iq = (torch.randn(rows, 1, D, device="cuda", generator=gen) * 2).to(torch.float8_e4m3fn)
    q = torch.randn(rows, HQ, D, device="cuda", generator=gen).to(torch.bfloat16)
    ks = torch.tensor([0.05], device="cuda")
    vs = torch.tensor([0.07], device="cuda")
    n_valid = build_n_valid_column_per_row(cu, pre, batch=len(reqs), total_q=rows,
                                           num_idx_heads=1, decode_max_q=0)
    req, kvl = prefill_row_positions(cu, pre, rows)
    ws = LogitsWorkspace(rows * 8192, "cuda")
    ref_ids = minimax_m3_index_topk(
        iq, idx_pages, bt, cu, seq_t, pre, max(n_new), max(seq), TOPK, 0, 1, 1, SM,
        emit_sparse_block_table=False, n_valid_column_per_row=n_valid,
        index_scorer=IndexScoreBuffers(scores=ws, nvfp4=True)).clone()
    width = -(-max(seq) // BLK)
    ref_scores = ws.logits(rows, width).clone()
    plan = StepPlan(attn=attn, rows=rows, num_kv_heads=1, num_heads=HQ, cu_seqlens_q=cu,
                    seq_lens=seq_t, block_table=bt, n_valid=n_valid, row_req_id=req,
                    row_kv_lens=kvl, max_query_len=max(n_new), max_seq_len=max(seq),
                    init_blocks=0, local_blocks=1, sm_scale=SM, scores=ws, device=q.device,
                    prefix_lens=pre)
    ids = plan.index(iq, idx_pages)
    assert torch.equal(ids, ref_ids)
    # aiter's scores (amd-kernels writes its forced-block sentinels into the local block)
    valid = torch.arange(width, device="cuda")[None, :] < (n_valid - 1)[:, None]
    got_scores = plan._score[0, :, :width]
    assert torch.equal(got_scores[valid].view(torch.int32), ref_scores[valid].view(torch.int32))
    out = torch.empty_like(q)
    plan.attention(q, kv_pages, out, ks, vs)
    ref = torch.empty_like(q)
    if attn == "ak16":
        from aiter.ops.flydsl import sparse_pa_nvfp4_prefill

        sparse_pa_nvfp4_prefill(q, kv_pages, ref_ids, bt, req, kvl, SM, out=ref, k_scale=ks,
                                v_scale=vs)
    elif attn == "ak16lazy":
        nvfp4_sparse_prefill(q, kv_pages.view(-1, 2, 128, 72), ref_ids, bt, req,
                             (kvl - 1).contiguous(), 1, SM, ref, ks, vs, compute_dtype="fp16",
                             lazy_rescale=True)
    else:
        sparse_pa_prefill(q, kv_pages, ref_ids, bt, req, kvl, SM, out=ref, k_scale=ks,
                          v_scale=vs)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert not torch.isnan(out).any()
    if attn == "ak16":  # the same ids through ATOM_M3_PREFILL_INDEX=amdk (no fast path)
        from atom.model_ops.minimax_m3.prefill_kernels import index_topk_amdk

        ids2 = index_topk_amdk(iq, idx_pages, bt, cu, seq_t, max(n_new), max(seq), TOPK, 0, 1,
                               n_valid, IndexScoreBuffers(scores=ws, nvfp4=True), SM)
        assert torch.equal(ids2, ref_ids)
