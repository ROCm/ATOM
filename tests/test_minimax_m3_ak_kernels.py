# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""amd-kernels switches of the NVFP4 sparse layer (atom/model_ops/minimax_m3/ak_kernels.py) against today's aiter
path, one TP4 rank (16 q / 1 KV / 1 index head): the insert bit for bit (pages, q, index-q), the decode block's
top-16 ids equal and its attention within rel L2 5e-3. Needs one gfx950 GPU and amd-kernels on PYTHONPATH."""

import math

import pytest
import torch

amd_kernels = pytest.importorskip("amd_kernels")

from atom.model_ops.minimax_m3 import ak_kernels  # noqa: E402

HQ, HKV, HIDX, D, ROT, BLK = 16, 1, 1, 128, 64, 128
ROW = HQ * D + 2 * HKV * D + HIDX * D + D
SM = D**-0.5


def _cos_sin(n, dev):
    inv = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, device=dev).float() / ROT))
    f = torch.outer(torch.arange(n, device=dev).float(), inv)
    return torch.cat([f.cos(), f.sin()], -1).bfloat16().contiguous()


def _aiter_insert(qkv, w, cs, pos, slots, kv, idx, ks, vs, q_out, iq_out):
    import aiter

    aiter.fused_qknorm_idxrqknorm(
        qkv, w[0], w[1], cs, pos, HQ, HKV, ROT, 1e-6, w[2], w[3], HIDX, slots, kv, kv, idx, BLK, q_out, iq_out,
        slots, kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4", k_scale=ks, v_scale=vs, asm_layout=False,
        skip_index_branch=False)


@pytest.mark.parametrize("n", [8, 64, 256, 512, 1537, 4096])
def test_insert_bitwise(n):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(n)
    prefix = 1000
    pages = math.ceil((prefix + n) / BLK) + 4
    perm = torch.randperm(pages, device=dev, generator=g)
    pos = torch.arange(prefix, prefix + n, device=dev)
    slots = perm[pos // BLK] * BLK + pos % BLK
    slots[-1] = -1  # a padding row at the tail
    qkv = torch.randn(n, ROW, device=dev, generator=g).bfloat16()
    w = [(0.1 * torch.randn(D, device=dev, generator=g)).bfloat16() for _ in range(4)]
    cs = _cos_sin(prefix + n + 1, dev)
    ks = torch.tensor([0.05], device=dev)
    vs = torch.tensor([0.07], device=dev)
    outs = []
    for path in ("aiter", "ak"):
        kv = torch.zeros(pages, 2 * HKV * 9216, dtype=torch.uint8, device=dev)
        idx = torch.zeros(pages, 9216, dtype=torch.uint8, device=dev)
        q_out = torch.empty(n, HQ * D, dtype=torch.bfloat16, device=dev)
        iq_out = torch.empty(n, HIDX * D, dtype=torch.float8_e4m3fn, device=dev)
        if path == "aiter":
            _aiter_insert(qkv, w, cs, pos, slots, kv, idx, ks, vs, q_out, iq_out)
        else:
            assert ak_kernels.insert_blocker(HQ, HKV, HIDX, ROT) is None
            ak_kernels.insert(qkv, *w, cs, pos, slots, kv, idx, ks, vs, q_out, iq_out, HQ, HKV, HIDX, 1e-6)
        torch.cuda.synchronize()
        outs.append((kv, idx, q_out[:-1], iq_out[:-1].view(torch.uint8)))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


@pytest.mark.parametrize("n,k", [(2560, 6144), (6144, 2048), (1536, 6144), (6144, 768), (6144, 6144), (6144, 3072)])
def test_mxfp8_gemm(n, k):
    from aiter.ops.shuffle import shuffle_weight

    from atom.model_ops import ak_gemm
    from atom.model_ops.blockscale import native_quant_linear

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(n + k)
    w = (torch.randn(n, k, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(119, 124, (n, k // 32), device=dev, dtype=torch.uint8, generator=g).view(torch.float8_e8m0fnu)
    w = shuffle_weight(w, layout=(16, 16))
    ak_s = ak_gemm.prepare(w, s, 4096)
    for m in (1, 8, 33, 128, 256, 512, 1500, 4096):
        x = torch.randn(m, k, device=dev, generator=g).bfloat16()
        ref = native_quant_linear(x, w, s, weight_group_rows=1, weight_preshuffled=True, dtype=torch.bfloat16)
        y = ak_gemm.ak_mxfp8_linear(x, w, s, ak_s)
        rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 1e-4, (m, rel)


@pytest.mark.parametrize("reqs,ctx", [(4, 20000), (16, 70000), (64, 33000)])
def test_decode_block(reqs, ctx):
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers
    from atom.model_ops.minimax_m3.index_topk import (
        build_n_valid_column_per_row, flydsl_ops, minimax_m3_index_topk_decode)
    from atom.model_ops.minimax_m3.nvfp4_pages import index_page_views  # noqa: F401
    from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace
    from aiter.ops.flydsl import sparse_pa_nvfp4_decode, sparse_pa_nvfp4_decode_alloc

    dev = torch.device("cuda")
    ql = 8
    g = torch.Generator(device=dev).manual_seed(reqs)
    lens = [ctx + 997 * i for i in range(reqs)]
    seq = torch.tensor(lens, dtype=torch.int32, device=dev)
    maxblk = 8192
    pages = sum(math.ceil(s / BLK) for s in lens) + 8
    kv = torch.randint(0, 256, (pages, 2 * 9216), dtype=torch.uint8, device=dev, generator=g)
    for off in (8192, 9216 + 8192):
        kv[:, off:off + 1024] = torch.randint(0x30, 0x40, (pages, 1024), dtype=torch.uint8, device=dev, generator=g)
    ix = torch.randint(0, 256, (pages, 9216), dtype=torch.uint8, device=dev, generator=g)
    ix[:, 8192:] = torch.randint(0x30, 0x40, (pages, 1024), dtype=torch.uint8, device=dev, generator=g)
    tab = torch.randint(0, pages, (reqs, maxblk), dtype=torch.int32, device=dev, generator=g)
    rows = reqs * ql
    iq = ((torch.randn(reqs, 1, 1, D, device=dev, generator=g)
           + 0.5 * torch.randn(reqs, ql, 1, D, device=dev, generator=g)) * 0.5).reshape(rows, 1, D)
    iq = iq.to(torch.float8_e4m3fn)
    q = (torch.randn(rows, HQ, D, device=dev, generator=g) * 0.7).bfloat16()
    one = torch.ones(1, device=dev)

    ws = LogitsWorkspace(max(rows * maxblk, 1 << 20), dev)
    flyd = flydsl_ops()
    cap = flyd.minimax_m3_index_score_work_map_capacity(reqs, maxblk, ql, 1)
    wm = flyd.minimax_m3_index_score_work_map(seq, maxblk, ql, 1,
                                              out=torch.empty((cap, 2), dtype=torch.int32, device=dev))
    scorer = IndexScoreBuffers(scores=ws, work_map=wm, max_block=maxblk, nvfp4=True)
    nvalid = build_n_valid_column_per_row(seq, seq, batch=reqs, total_q=rows, num_idx_heads=1, decode_max_q=ql)
    sparse_pa_nvfp4_decode_alloc(dev)
    ids_ref = minimax_m3_index_topk_decode(iq, ix, tab, seq, maxblk * BLK, 16, 0, 1, 1, SM,
                                           emit_sparse_block_table=False, max_query_len=ql,
                                           n_valid_column_per_row=nvalid, index_scorer=scorer)
    out_ref = torch.empty(rows, HQ, D, dtype=torch.bfloat16, device=dev)
    sparse_pa_nvfp4_decode(q, kv, ids_ref, tab, seq, SM, out=out_ref, k_scale=one, v_scale=one, query_len=ql)

    out = torch.empty_like(out_ref)
    ids = ak_kernels.index_sparse_decode(iq, ix, q, kv, tab, seq, 0, 1, SM, out, one, one, ql, maxblk * BLK, HKV)
    out2 = torch.empty_like(out_ref)
    ak_kernels.sparse_decode(q, kv, ids, tab, seq, SM, out2, one, one, ql, HKV)
    torch.cuda.synchronize()
    ref_sorted = ids_ref.reshape(-1, 16)[:rows].sort(-1).values
    assert torch.equal(ids.reshape(-1, 16)[:rows].sort(-1).values, ref_sorted)
    for o in (out, out2):
        rel = ((o.float() - out_ref.float()).norm() / out_ref.float().norm()).item()
        assert rel < 5e-3, rel
