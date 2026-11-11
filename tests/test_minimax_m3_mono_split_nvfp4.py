# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono's split-K sparse attention over NVFP4 K/V pages (gfx950, single-buffer 18432 B
pages, f16 MFMA over the exact dequantized products): the partitions' output, max
and sum against a torch float64 reference of the same pages, and the merged
attention against aiter's ``minimax_m3_sparse_pa_nvfp4_decode``.

Needs ``ATOM_NVFP4_VLLM_REF`` (vLLM's ``tests/kernels/attention``: the page writer) and,
for the aiter comparison, ``ATOM_NVFP4_PA_KERNELS`` (the aiter
``aiter/ops/flydsl/kernels`` directory of the nvfp4 sparse pa); skips without them."""

import os
import sys

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
VLLM_REF = os.environ.get("ATOM_NVFP4_VLLM_REF")
PA_KERNELS = os.environ.get("ATOM_NVFP4_PA_KERNELS")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not VLLM_REF,
    reason="needs gfx950 and the vLLM reference tree (see the module doc)",
)

from atom.models.minimax_m3.mono.config import HEAD_DIM
from atom.models.minimax_m3.mono.kernels.common import KV_V_SCALE_AT, nvfp4_k_scale_at
from atom.models.minimax_m3.mono.kernels.split_nvfp4 import build_split_nvfp4_kernel
from atom.models.minimax_m3.mono.layout import N_SPLIT, O_K, SPLIT_KEYS, H

SM_SCALE = HEAD_DIM**-0.5


def _ref():
    sys.path.insert(0, VLLM_REF)
    import minimax_m3_nvfp4_ref as ref

    return ref


def _case(seed, n_logical, n_sel, tail, k_gs, v_gs, tokens=1):
    """A cache of ``n_logical`` blocks (random keys / values written by vLLM's
    writer), a request of ``n_logical`` logical blocks whose last holds ``tail`` keys,
    ``n_sel`` selected blocks (the last logical one among them)."""
    ref = _ref()
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks = n_logical
    n = blocks * 128
    k = torch.randn(n, 1, HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
    v = (torch.randn(n, 1, HEAD_DIM, device="cuda", generator=g) * 2).to(torch.bfloat16)
    cache = torch.zeros(blocks, 2, 128, 72, device="cuda", dtype=torch.uint8)
    ref.insert_nvfp4_kv(k, v, cache, torch.arange(n, device="cuda"), k_gs, v_gs)
    page_of = torch.randperm(blocks, device="cuda", generator=g)  # logical -> page
    # the cache holds logical block b's keys at page b; remap by writing through
    # page_of: physical page page_of[b] gets logical block b's bytes
    phys = torch.empty_like(cache)
    phys[page_of] = cache
    sel = sorted(
        torch.randperm(n_logical - 1, generator=torch.Generator().manual_seed(seed))[
            : n_sel - 1
        ].tolist()
    )
    sel.append(n_logical - 1)
    q = torch.randn(tokens, 16, HEAD_DIM, device="cuda", generator=g).to(torch.bfloat16)
    ctx = (n_logical - 1) * 128 + tail
    return ref, phys.view(blocks, -1), page_of.to(torch.int32), sel, q, ctx, (k, v)


def _tables(page_of, sel, tail, tokens):
    """The split tasks' page lists: partition t = list blocks 2 t, 2 t + 1."""
    n_keys = (len(sel) - 1) * 128 + tail
    n_part = (n_keys + SPLIT_KEYS - 1) // SPLIT_KEYS
    tbl = torch.zeros(tokens * N_SPLIT, 32, dtype=torch.int32)
    for tok in range(tokens):
        for t in range(n_part):
            for i in range(2):
                if 2 * t + i < len(sel):
                    pg = int(page_of[sel[2 * t + i]])
                    for j in range(8):
                        tbl[tok * N_SPLIT + t, 8 * i + j] = pg * 8 + j
            tbl[tok * N_SPLIT + t, 16] = n_keys
    return tbl.cuda(), n_keys, n_part


def _reference(ref, pages, page_of, sel, q, n_keys, k_gs, v_gs):
    """Per partition t: (o [16, 128], m [16], l [16]) in float64 from the exactly
    dequantized pages."""
    kd, ksc, vd, vsc = ref.nvfp4_kv_cache_token_major(pages.view(-1, 2, 128, 72))
    kx = ref.dequant(kd, ksc).double() * k_gs  # [pages, 1, 128, 128]
    vx = ref.dequant(vd, vsc).double() * v_gs
    keys = torch.cat([kx[page_of[b], 0] for b in sel])[:n_keys]
    vals = torch.cat([vx[page_of[b], 0] for b in sel])[:n_keys]
    out = []
    for t in range((n_keys + SPLIT_KEYS - 1) // SPLIT_KEYS):
        kk, vv = keys[t * 256 : (t + 1) * 256], vals[t * 256 : (t + 1) * 256]
        s = (q[0].double() @ kk.T) * SM_SCALE  # [16, keys]
        m = s.max(-1).values
        p = torch.exp(s - m[:, None])
        out.append(((p @ vv) / p.sum(-1)[:, None], m, p.sum(-1)))
    return out


@pytest.mark.parametrize(
    "n_logical, n_sel, tail, k_gs, v_gs",
    [
        (40, 16, 77, 1.0, 1.0),
        (40, 16, 128, 0.7, 1.9),
        (20, 5, 3, 2.0, 0.4),
        (3, 2, 50, 1.0, 1.0),
    ],
)
@pytest.mark.parametrize("poison", [False, True])
def test_split_partitions_match_the_exact_reference(
    n_logical, n_sel, tail, k_gs, v_gs, poison
):
    ref, pages, page_of, sel, q, _ctx, _ = _case(1, n_logical, n_sel, tail, k_gs, v_gs)
    if poison:  # the tail block's keys past the context: NaN scales (stale bytes)
        last = pages[int(page_of[sel[-1]])]
        for t in range(tail, 128):
            for j in range(8):
                last[8192 + nvfp4_k_scale_at(t, j)] = 0xFF
            for j in range(8):
                last[KV_V_SCALE_AT + j * 128 + t] = 0xFF
    tbl, n_keys, _n_part = _tables(page_of, sel, tail, 1)
    gs = torch.tensor([k_gs, v_gs], device="cuda", dtype=torch.float32)
    sp_o = torch.zeros(N_SPLIT, O_K, device="cuda", dtype=torch.bfloat16)
    sp_m = torch.zeros(N_SPLIT, H, device="cuda", dtype=torch.float32)
    sp_l = torch.zeros(N_SPLIT, H, device="cuda", dtype=torch.float32)
    launch = build_split_nvfp4_kernel(SM_SCALE, N_SPLIT)
    launch(
        tbl.data_ptr(), q.data_ptr(), pages.data_ptr(), gs.data_ptr(),
        sp_o.data_ptr(), sp_m.data_ptr(), sp_l.data_ptr(), 0,
    )  # fmt: skip
    torch.cuda.synchronize()
    want = _reference(ref, pages, page_of, sel, q, n_keys, k_gs, v_gs)
    for t, (o, m, l) in enumerate(want):
        got_o = sp_o[t].view(16, HEAD_DIM).double()
        torch.testing.assert_close(sp_m[t].double(), m, rtol=2e-5, atol=1e-5)
        torch.testing.assert_close(sp_l[t].double(), l, rtol=2e-3, atol=0)
        # P is f16 and the output bf16: ~1e-3 relative to the row's scale
        err = (got_o - o).abs().max() / o.abs().max()
        assert err < 6e-3, (t, err.item())


@pytest.mark.skipif(not PA_KERNELS, reason="needs the aiter nvfp4 sparse pa kernels")
def test_merged_split_equals_aiter_sparse_pa_decode():
    import aiter.ops.flydsl.kernels as pkg

    pkg.__path__.insert(0, PA_KERNELS)
    from aiter.ops.flydsl.kernels import minimax_m3_sparse_pa_nvfp4_decode as pa

    n_logical, n_sel, tail, k_gs, v_gs = 40, 16, 77, 0.8, 1.3
    _ref, pages, page_of, sel, q, ctx, _ = _case(2, n_logical, n_sel, tail, k_gs, v_gs)
    tbl, _n_keys, n_part = _tables(page_of, sel, tail, 1)
    gs = torch.tensor([k_gs, v_gs], device="cuda", dtype=torch.float32)
    sp_o = torch.zeros(N_SPLIT, O_K, device="cuda", dtype=torch.bfloat16)
    sp_m = torch.zeros(N_SPLIT, H, device="cuda", dtype=torch.float32)
    sp_l = torch.zeros(N_SPLIT, H, device="cuda", dtype=torch.float32)
    build_split_nvfp4_kernel(SM_SCALE, N_SPLIT)(
        tbl.data_ptr(), q.data_ptr(), pages.data_ptr(), gs.data_ptr(),
        sp_o.data_ptr(), sp_m.data_ptr(), sp_l.data_ptr(), 0,
    )  # fmt: skip
    torch.cuda.synchronize()
    m, l, o = (
        sp_m[:n_part].double(),
        sp_l[:n_part].double(),
        sp_o[:n_part].view(n_part, 16, HEAD_DIM).double(),
    )
    big = m.max(0).values
    w = torch.exp(m - big) * l
    mine = ((w[..., None] * o).sum(0) / w.sum(0)[:, None]).float()
    topk = torch.full((1, 1, 16), -1, dtype=torch.int32, device="cuda")
    topk[0, 0, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device="cuda")
    bt = page_of.view(1, -1).contiguous()
    seq = torch.tensor([ctx], dtype=torch.int32, device="cuda")
    theirs = pa.minimax_m3_sparse_pa_nvfp4_decode(
        q,
        pages,
        topk,
        bt,
        seq,
        SM_SCALE,
        k_scale=gs[:1].clone(),
        v_scale=gs[1:].clone(),
    ).float()[0]
    err = (mine - theirs).abs().max() / theirs.abs().max()
    assert err < 8e-3, err.item()
