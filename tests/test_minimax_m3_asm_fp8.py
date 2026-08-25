# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""fp8 KV cache for the MiniMax-M3 ASM path: decode (gluon) + prefill (pa_fwd_asm)
fp8 vs bf16 within fp8 tolerance.

Writes the SAME K/V into a bf16 SHUFFLE cache (reshape_and_cache) and an fp8
SHUFFLE cache with per-token dynamic quant (reshape_and_cache_with_pertoken_quant),
then runs the M3 ASM decode/prefill wrappers on each and compares (cos > 0.99).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from atom.model_ops.minimax_m3.sparse_attn import (  # noqa: E402
    minimax_m3_sparse_attn_decode_asm,
    minimax_m3_sparse_attn_prefill_asm,
    ASM_PAGE_SIZE,
    PAGES_PER_SPARSE_BLOCK,
    SPARSE_BLOCK_SIZE,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/ROCm"
)

NH, NKV, HD = 16, 1, 128
SM = HD**-0.5
fp8 = dtypes.fp8


def _make_caches(kv_dtype, nblk):
    x = 16 // torch.tensor([], dtype=kv_dtype).element_size()
    nphys = nblk * PAGES_PER_SPARSE_BLOCK
    kc = torch.zeros(
        nphys, NKV, HD // x, ASM_PAGE_SIZE, x, dtype=kv_dtype, device="cuda"
    )
    vc = torch.zeros(
        nphys, NKV, ASM_PAGE_SIZE // x, HD, x, dtype=kv_dtype, device="cuda"
    )
    return kc, vc, nphys


def _fill(kc, vc, k, v, slot, nphys, fp8_on):
    if fp8_on:
        ks = torch.zeros(nphys, NKV, ASM_PAGE_SIZE, dtype=torch.float32, device="cuda")
        vs = torch.zeros(nphys, NKV, ASM_PAGE_SIZE, dtype=torch.float32, device="cuda")
        aiter.reshape_and_cache_with_pertoken_quant(
            k, v, kc, vc, ks, vs, slot, asm_layout=True
        )
        return ks, vs
    aiter.reshape_and_cache(
        k,
        v,
        kc,
        vc,
        slot,
        kv_cache_dtype="auto",
        k_scale=None,
        v_scale=None,
        asm_layout=True,
    )
    return None, None


def test_decode_fp8_matches_bf16():
    torch.manual_seed(0)
    dev = "cuda"
    batch = 4
    seq_lens = torch.tensor([300, 128, 777, 50], dtype=torch.int32, device=dev)
    max_block = (int(seq_lens.max()) + 127) // 128
    nblk = batch * max_block + 4
    bt = (
        torch.randperm(nblk, device=dev)[: batch * max_block]
        .view(batch, max_block)
        .to(torch.int32)
    )
    ntok = nblk * SPARSE_BLOCK_SIZE
    k = torch.randn(ntok, NKV, HD, dtype=torch.bfloat16, device=dev) * 0.3
    v = torch.randn(ntok, NKV, HD, dtype=torch.bfloat16, device=dev) * 0.3
    slot = torch.arange(ntok, dtype=torch.int64, device=dev)
    q = torch.randn(batch, NH, HD, dtype=torch.bfloat16, device=dev)
    topk = 16
    topk_idx = torch.full((1, batch, topk), -1, dtype=torch.int32, device=dev)
    for b in range(batch):
        nb = int((int(seq_lens[b]) + 127) // 128)
        kk = min(topk, nb)
        topk_idx[0, b, :kk] = torch.arange(kk, dtype=torch.int32, device=dev)

    def run(kv_dtype, fp8_on):
        kc, vc, nphys = _make_caches(kv_dtype, nblk)
        ks, vs = _fill(kc, vc, k, v, slot, nphys, fp8_on)
        out = torch.empty(batch, NH, HD, dtype=torch.bfloat16, device=dev)
        minimax_m3_sparse_attn_decode_asm(
            q, kc, vc, topk_idx, bt, seq_lens, NKV, SM, out, k_scale=ks, v_scale=vs
        )
        return out

    o_bf = run(torch.bfloat16, False)
    o_f8 = run(fp8, True)
    cos = F.cosine_similarity(o_bf.float().flatten(), o_f8.float().flatten(), dim=0)
    assert cos.item() > 0.99, f"decode fp8 vs bf16 cos {cos.item():.5f}"


def test_prefill_fp8_matches_bf16():
    torch.manual_seed(0)
    dev = "cuda"
    seq_lens_list = [200, 129, 300]
    prefix_list = [0, 0, 0]
    batch = len(seq_lens_list)
    prefix_lens = torch.tensor(prefix_list, dtype=torch.int32, device=dev)
    q_lens = [seq_lens_list[b] - prefix_list[b] for b in range(batch)]
    total_q = sum(q_lens)
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device=dev).cumsum(0)
    max_block = (max(seq_lens_list) + 127) // 128
    nblk = batch * max_block + 4
    bt = (
        torch.randperm(nblk, device=dev)[: batch * max_block]
        .view(batch, max_block)
        .to(torch.int32)
    )
    ntok = nblk * SPARSE_BLOCK_SIZE
    k = torch.randn(ntok, NKV, HD, dtype=torch.bfloat16, device=dev) * 0.3
    v = torch.randn(ntok, NKV, HD, dtype=torch.bfloat16, device=dev) * 0.3
    slot = torch.arange(ntok, dtype=torch.int64, device=dev)
    q = torch.randn(total_q, NH, HD, dtype=torch.bfloat16, device=dev)
    topk = 16
    topk_idx = torch.full((1, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(batch):
        qs, qe = int(cu[b]), int(cu[b + 1])
        for n in range(qs, qe):
            p = prefix_list[b] + (n - qs)
            nb = p // 128 + 1
            kk = min(topk, nb)
            sel = torch.randperm(nb)[:kk].sort().values
            topk_idx[0, n, : len(sel)] = sel.to(torch.int32).to(dev)

    def run(kv_dtype, fp8_on):
        kc, vc, nphys = _make_caches(kv_dtype, nblk)
        ks, vs = _fill(kc, vc, k, v, slot, nphys, fp8_on)
        out = torch.empty(total_q, NH, HD, dtype=torch.bfloat16, device=dev)
        minimax_m3_sparse_attn_prefill_asm(
            q,
            kc,
            vc,
            topk_idx,
            bt,
            None,
            None,
            None,
            NKV,
            SM,
            out,
            k_scale=ks,
            v_scale=vs,
            cu_seqlens_q=cu,
            prefix_lens=prefix_lens,
        )
        return out

    o_bf = run(torch.bfloat16, False)
    o_f8 = run(fp8, True)
    cos = F.cosine_similarity(o_bf.float().flatten(), o_f8.float().flatten(), dim=0)
    assert cos.item() > 0.99, f"prefill fp8 vs bf16 cos {cos.item():.5f}"
