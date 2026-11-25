# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3 FP8 sparse prefill on aiter's token-tiled kernel: the layers and steps it
serves, its ragged tile map, and on a gfx950 its error against ASM and an fp32 reference.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest
import torch

from atom.model_ops.minimax_m3 import tt_prefill as tt

TP1 = {
    "num_heads": 64,
    "num_kv_heads": 4,
    "head_dim": 128,
    "topk": 16,
    "kv_cache_dtype": "fp8",
}
OP = "aiter.ops.m3_tt_sparse_prefill"


@pytest.fixture
def fresh_ops():
    tt._ops.cache_clear()
    yield
    tt._ops.cache_clear()


class TestLayerEligible:
    def test_on_by_default(self, monkeypatch):
        monkeypatch.delenv("ATOM_M3_TT_PREFILL", raising=False)
        monkeypatch.setattr(tt, "_ops", lambda: object())
        assert tt.layer_eligible(**TP1)

    def test_off_by_flag(self, monkeypatch):
        monkeypatch.setenv("ATOM_M3_TT_PREFILL", "0")
        monkeypatch.setattr(tt, "_ops", lambda: pytest.fail("must not reach aiter"))
        assert not tt.layer_eligible(**TP1)

    @pytest.mark.parametrize(
        "change",
        [
            {"num_kv_heads": 2, "num_heads": 32},  # TP2
            {"num_kv_heads": 1, "num_heads": 16},  # TP4
            {"kv_cache_dtype": "fp4"},
            {"kv_cache_dtype": "bf16"},
            {"head_dim": 64},
            {"topk": 32},
        ],
    )
    def test_only_the_tp1_fp8_layer(self, monkeypatch, change):
        monkeypatch.setenv("ATOM_M3_TT_PREFILL", "1")
        monkeypatch.setattr(tt, "_ops", lambda: pytest.fail("must not reach aiter"))
        assert not tt.layer_eligible(**{**TP1, **change})

    def test_an_aiter_without_the_op_keeps_asm_and_says_so_once(
        self, monkeypatch, caplog, fresh_ops
    ):
        monkeypatch.setenv("ATOM_M3_TT_PREFILL", "1")
        monkeypatch.setitem(sys.modules, OP, None)
        with caplog.at_level(logging.WARNING, logger="atom"):
            assert not any(tt.layer_eligible(**TP1) for _ in range(3))
        assert sum("token-tiled" in r.getMessage() for r in caplog.records) == 1

    @pytest.mark.parametrize("gfx950", [True, False])
    def test_only_where_the_op_runs(self, monkeypatch, fresh_ops, gfx950):
        monkeypatch.setenv("ATOM_M3_TT_PREFILL", "1")
        op = types.ModuleType(OP)
        op.is_supported = lambda: gfx950
        monkeypatch.setitem(sys.modules, OP, op)
        assert tt.layer_eligible(**TP1) is gfx950


@pytest.mark.parametrize(
    "rows,scheduled,max_seq_len,want",
    [
        (8192, 8192, 8192, True),
        (1536, 1536, 32768, True),
        (8192, 8192, 65_536, True),
        (1535, 1535, 8192, False),  # too few rows: most CUs idle
        (8192, 8192, 65_537, False),  # past the context bound
        (8192, 8192, 599_040, False),  # also past the kernel's 524,288-token bitmap
        (8192, 8000, 8192, False),  # pad rows past the scheduled tokens
    ],
)
def test_step_gate(rows, scheduled, max_seq_len, want):
    md = types.SimpleNamespace(
        max_seq_len=max_seq_len, qo_indptr=torch.empty(scheduled, dtype=torch.int32)
    )
    assert tt.step_takes(md, rows) is want


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
def test_tile_map_covers_every_row_once_without_straddling(monkeypatch):
    monkeypatch.setattr(tt, "_ops", lambda: types.SimpleNamespace(TT_TQ=12))
    qlens, prefixes = [25, 1, 12, 0, 40], [100, 7, 0, 3, 5000]
    i32 = {"dtype": torch.int32, "device": "cuda"}
    cu = torch.tensor([0, *torch.tensor(qlens).cumsum(0).tolist()], **i32)
    md = types.SimpleNamespace(
        cu_seqlens_q=cu, context_lens=torch.tensor(prefixes, **i32)
    )
    total_q = sum(qlens)
    req, kv_lens, t0, n, ntiles = tt._tile_map(md, total_q)
    assert ntiles == -(-total_q // 12) + len(qlens)
    assert t0.shape == n.shape == (ntiles,)
    req = req.cpu()
    seen = torch.zeros(total_q, dtype=torch.int32)
    for start, length in zip(t0.tolist(), n.tolist()):
        assert 0 <= length <= 12
        if length:
            rows = list(range(start, start + length))
            seen[rows] += 1
            assert len(set(req[rows].tolist())) == 1  # one request a tile
    assert torch.equal(seen, torch.ones(total_q, dtype=torch.int32))
    want_kv = [p + i + 1 for q, p in zip(qlens, prefixes) for i in range(q)]
    assert kv_lens.tolist() == want_kv
    # built once a step: every later layer reads it back
    assert tt._tile_map(md, total_q)[2] is t0


def _tile_map_torch(cu, prefix, total_q, tq):
    """The tile map in torch ops: searchsorted rows and a cumsum of request tiles."""
    nreq = prefix.shape[0]
    rows = torch.arange(total_q, dtype=cu.dtype, device=cu.device)
    req = torch.searchsorted(cu[1:], rows, right=True).clamp_(max=nreq - 1)
    kv_lens = prefix[req] + (rows - cu[req]) + 1
    lens = (cu[1 : nreq + 1] - cu[:nreq]).to(torch.int64)
    tiles = (lens + tq - 1) // tq
    ends = torch.cumsum(tiles, 0)
    ntiles = (total_q + tq - 1) // tq + nreq
    k = torch.arange(ntiles, device=cu.device, dtype=torch.int64)
    b = torch.searchsorted(ends, k, right=True)
    live = b < nreq
    b = b.clamp(max=nreq - 1)
    local = k - (ends[b] - tiles[b])
    t0 = torch.where(live, cu[b].to(torch.int64) + local * tq, 0)
    n = torch.where(live, torch.clamp(lens[b] - local * tq, 0, tq), 0)
    return [x.to(torch.int32) for x in (req, kv_lens, t0, n)], ntiles


@cuda
@pytest.mark.parametrize("nreq", [1, 2, 7, 64, 65, 130, 513])
@pytest.mark.parametrize("max_prefix", [32768, 524_288])
def test_tile_map_kernel_matches_torch(monkeypatch, nreq, max_prefix):
    """One launch for the whole map, the torch ops' bytes: ragged, empty and tile-multiple
    requests, batches past one 64-request pass of the kernel, prefixes up to 512K."""
    monkeypatch.setattr(tt, "_ops", lambda: types.SimpleNamespace(TT_TQ=12))
    g = torch.Generator().manual_seed(nreq)
    lens = torch.randint(0, 3000, (nreq,), generator=g)
    lens[:: max(1, nreq // 3)] = 12
    lens[1::5] = 0
    lens[0] = max(int(lens[0]), 1)
    i32 = {"dtype": torch.int32, "device": "cuda"}
    cu = torch.tensor([0, *lens.cumsum(0).tolist()], **i32)
    prefix = torch.randint(0, max_prefix, (nreq,), generator=g).to(**i32)
    total_q = int(lens.sum())
    md = types.SimpleNamespace(cu_seqlens_q=cu, context_lens=prefix)
    got = tt._tile_map(md, total_q)
    want, ntiles = _tile_map_torch(cu, prefix, total_q, 12)
    assert got[4] == ntiles
    for a, b in zip(got[:4], want):
        assert a.dtype == b.dtype and torch.equal(a, b)


# ---------------------------------------------------------------------------
# The kernel through ATOM's call, on a gfx950 with aiter's op.
# ---------------------------------------------------------------------------
gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or tt._ops() is None,
    reason="needs a gfx950 and aiter.ops.m3_tt_sparse_prefill",
)


def _case(qlens, prefixes, seed, dev="cuda"):
    """A ragged prefill step over a random FP8 page-16 SHUFFLE KV pool and its sparse
    selection from ATOM's own top-k + emit."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk

    g = torch.Generator(device=dev).manual_seed(seed)
    total_q = sum(qlens)
    seqs = [p + q for q, p in zip(qlens, prefixes)]
    nblk = max(-(-s // 128) for s in seqs)
    batch = len(qlens)
    pool = batch * nblk + 3
    perm = torch.randperm(pool, generator=g, device=dev)[: batch * nblk]
    bt = perm.to(torch.int32).view(batch, nblk)
    i32 = {"dtype": torch.int32, "device": dev}
    cu = torch.tensor([0, *torch.tensor(qlens).cumsum(0).tolist()], **i32)
    seq_lens = torch.tensor(seqs, **i32)
    prefix = torch.tensor(prefixes, **i32)
    idx_q = (torch.randn(total_q, 4, 128, generator=g, device=dev) * 0.3).bfloat16()
    index_cache = (torch.randn(pool, 128, 128, generator=g, device=dev) * 0.3).to(
        torch.float8_e4m3fn
    )
    topk_idx, sparse_bt, sparse_ctx = minimax_m3_index_topk(
        idx_q,
        index_cache,
        bt,
        cu,
        seq_lens,
        prefix,
        max(qlens),
        max(seqs),
        16,
        0,
        1,
        4,
        128**-0.5,
        emit_sparse_block_table=True,
    )
    # KV: page-16 SHUFFLE e4m3 (x = 16) with fp32 per-token scales
    pages16 = pool * 8
    k16 = (torch.randn(pages16, 4, 8, 16, 16, generator=g, device=dev) * 2).to(
        torch.float8_e4m3fn
    )
    v16 = (torch.randn(pages16, 4, 1, 128, 16, generator=g, device=dev) * 2).to(
        torch.float8_e4m3fn
    )
    ks = torch.rand(pages16, 4, 16, generator=g, device=dev) * 0.05 + 0.01
    vs = torch.rand(pages16, 4, 16, generator=g, device=dev) * 0.05 + 0.01
    q = (torch.randn(total_q, 64, 128, generator=g, device=dev) * 0.5).bfloat16()
    md = types.SimpleNamespace(
        cu_seqlens_q=cu,
        context_lens=prefix,
        max_seq_len=max(seqs),
        qo_indptr=torch.arange(total_q, **i32),
    )
    return {
        "q": q,
        "k16": k16,
        "v16": v16,
        "ks": ks,
        "vs": vs,
        "topk": topk_idx,
        "bt": bt,
        "sbt": sparse_bt,
        "sctx": sparse_ctx,
        "md": md,
        "cu": cu,
        "prefix": prefix,
    }


def _reference(c):
    """fp32 attention of every row over its emitted page-16 table (dequantized KV),
    whose ids index (page, KV head) records: page16 * 4 + head."""
    T = c["q"].shape[0]
    k = c["k16"].float()  # [P, 4, 8, 16, 16]: (page, kvh, d/16, tok, d%16)
    k = k.permute(0, 1, 3, 2, 4).reshape(-1, 16, 128) * c["ks"].reshape(-1, 16, 1)
    v = c["v16"].float()  # [P, 4, 1, 128, 16]: (page, kvh, -, d, tok)
    v = v[:, :, 0].permute(0, 1, 3, 2).reshape(-1, 16, 128) * c["vs"].reshape(-1, 16, 1)
    out = torch.empty(T, 64, 128, device=c["q"].device)
    sbt = c["sbt"].view(T, 4, -1)
    sctx = c["sctx"].view(T, 4)
    for t0 in range(0, T, 64):
        t1 = min(T, t0 + 64)
        for h in range(4):
            pages = sbt[t0:t1, h].long()  # unused slots hold page 0
            kk = k[pages].reshape(t1 - t0, -1, 128)  # [t, 2048, 128]
            vv = v[pages].reshape(t1 - t0, -1, 128)
            qq = c["q"][t0:t1, 16 * h : 16 * h + 16].float()
            s = torch.einsum("thd,tkd->thk", qq, kk) * 128**-0.5
            live = (
                torch.arange(kk.shape[1], device=s.device)[None] < sctx[t0:t1, h, None]
            )
            s = s.masked_fill(~live[:, None], float("-inf"))
            out[t0:t1, 16 * h : 16 * h + 16] = torch.einsum(
                "thk,tkd->thd", s.softmax(-1), vv
            )
    return out


def _err(o, ref):
    d = (o.float() - ref).abs()
    return d.max().item(), (d.square().sum() / ref.square().sum()).sqrt().item()


@gpu
@pytest.mark.parametrize(
    "qlens,prefixes",
    [
        ([1000], [3000]),
        ([300, 7, 650], [0, 4000, 1500]),
        ([1, 13, 2500], [0, 9, 700]),
        ([1536], [62_000]),  # past the old 32K bound
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_token_tiled_error_is_not_worse_than_asm(qlens, prefixes, seed):
    from atom.model_ops.minimax_m3.sparse_attn import (
        minimax_m3_sparse_attn_prefill_asm,
    )

    c = _case(qlens, prefixes, seed)
    args = (c["q"], c["k16"], c["v16"], c["ks"], c["vs"], c["topk"], c["bt"], c["md"])
    o_tt = torch.full_like(c["q"], float("nan"))
    tt.run(*args, 128**-0.5, o_tt)
    o_asm = torch.full_like(c["q"], float("nan"))
    minimax_m3_sparse_attn_prefill_asm(
        c["q"],
        c["k16"],
        c["v16"],
        c["topk"],
        c["bt"],
        None,
        None,
        c["md"].qo_indptr,
        4,
        128**-0.5,
        o_asm,
        k_scale=c["ks"],
        v_scale=c["vs"],
        cu_seqlens_q=c["cu"],
        prefix_lens=c["prefix"],
        sparse_bt=c["sbt"],
        sparse_ctx=c["sctx"],
    )
    torch.cuda.synchronize()
    assert not torch.isnan(o_tt).any()
    ref = _reference(c)
    (tt_max, tt_l2), (asm_max, asm_l2) = _err(o_tt, ref), _err(o_asm, ref)
    assert tt_l2 <= asm_l2 * 1.02
    assert tt_max <= asm_max * 1.25
    # deterministic: a second run writes the same bytes
    again = torch.empty_like(o_tt)
    tt.run(*args, 128**-0.5, again)
    assert torch.equal(again, o_tt)
