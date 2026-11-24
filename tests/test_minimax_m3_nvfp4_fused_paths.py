# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse KV: the fused paths against the unfused ones they replace.

The NVFP4 attention (gfx1250) used to cost three launches a sparse layer more than
the FP8 one. Each is now folded into a launch the layer runs anyway, and each test
pins the fused result bit for bit to the unfused one:

- the page-128 table: emitted by the top-k selectors (TP decode / prefill, the
  standalone emit after aiter's selector, the parallel prefill selector, the
  indexer-CP merge) instead of ``nvfp4_page_table``'s own launch;
- the K/V write: aiter's ``fused_qknorm_idxrqknorm`` quantizes K/V into the NVFP4
  pool (``kv_cache_dtype="nvfp4"`` with the scale planes) instead of writing them to
  a BF16 staging buffer that ``kv_nvfp4_store`` quantizes;
- o_proj's FP8 input: the paged decode writes it (``quantized_output``) instead of
  the BF16 output plus a per-token quant;
- the FP4 index key write: the same aiter launch quantizes it into the FP4 index
  planes (``index_cache_dtype="nvfp4"`` with ``index_scale``) instead of a BF16
  staging buffer that ``index_fp4_store`` quantizes.

The table tests are plain Triton: without a GPU they run on the CPU under
``TRITON_INTERPRET=1`` (``TRITON_INTERPRET=1 GPU_ARCHS=gfx1250 pytest ...``). The
others need a gfx1250 GPU (the write also gfx950).
"""

import os

import pytest
import torch

pytest.importorskip("triton", reason="the kernels are triton")


def _gfx() -> str:
    if not torch.cuda.is_available():
        return ""
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", "").split(":")[0]


GFX = _gfx()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cuda" if GFX else "cpu"
D, PAGE = 128, 128

table_kernels = pytest.mark.skipif(
    not GFX and not INTERPRET,
    reason="needs a GPU, or TRITON_INTERPRET=1 to run on the CPU",
)
gfx1250_only = pytest.mark.skipif(GFX != "gfx1250", reason="needs a gfx1250 GPU")
writer_archs = pytest.mark.skipif(
    GFX not in ("gfx950", "gfx1250"), reason="needs a gfx950 or gfx1250 GPU"
)


@pytest.fixture
def page_table_ref(monkeypatch):
    """``nvfp4_page_table`` (the unfused launch) on any device: its gfx950
    short-cut is an arch query, which a CPU run cannot answer."""
    from atom.model_ops.minimax_m3 import sparse_attn_nvfp4

    monkeypatch.setattr(sparse_attn_nvfp4, "_gluon_kernel", lambda: False)
    return sparse_attn_nvfp4.nvfp4_page_table


def _block_table(nreq, maxb, seed):
    g = torch.Generator().manual_seed(seed)
    bt = torch.stack(
        [torch.randperm(4 * maxb, generator=g)[:maxb] for _ in range(nreq)]
    )
    return bt.to(torch.int32).to(DEV)


def _check_table(page_table_ref, topk_idx, bt, row_req, row_seq, table, ctx, rows):
    want_t, want_c = page_table_ref(topk_idx, bt, row_req, row_seq)
    hkv = topk_idx.shape[0]
    assert torch.equal(ctx[: rows * hkv], want_c[: rows * hkv])
    assert torch.equal(table[: rows * hkv], want_t[: rows * hkv])


@table_kernels
@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("q_rows", [1, 4])
def test_standalone_emit_is_the_page_table(hkv, q_rows, page_table_ref):
    """The emit that follows aiter's selector (and shares the fused selectors'
    row helper), decode-shaped, with requests shorter than the verify window."""
    from atom.model_ops.minimax_m3.index_topk import (
        NVFP4_PAGES_PER_BLOCK,
        _alloc_emit,
        _launch_sparse_block_table_emit,
    )
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import decode_rows

    torch.manual_seed(0)
    nreq, topk, maxb = 6, 16, 40
    seq = torch.randint(1, maxb * PAGE, (nreq,), dtype=torch.int32)
    seq[:3] = torch.tensor([1, 2, PAGE + 1], dtype=torch.int32)
    if q_rows > 1:
        seq[0] = 0  # a graph pad request: every row before its first token
    seq = seq.to(DEV)
    rows = nreq * q_rows
    bt = _block_table(nreq, maxb, 1)
    idx = torch.full((hkv, rows, topk), -1, dtype=torch.int32)
    for h in range(hkv):
        for r in range(rows):
            sel = torch.randperm(maxb)[: torch.randint(0, topk + 1, ()).item()]
            idx[h, r, : sel.numel()] = sel.to(torch.int32)
    idx = idx.to(DEV)
    out, args = _alloc_emit(rows, hkv, topk, bt, True, DEV, NVFP4_PAGES_PER_BLOCK)
    _launch_sparse_block_table_emit(
        idx,
        row_starts=seq,
        row_prefix=seq,
        batch=nreq,
        topk=topk,
        block_table=bt,
        emit_args=args,
        num_idx_heads=hkv,
        rows_per_req=q_rows,
        decode_max_q=q_rows,
        block_page_stride=NVFP4_PAGES_PER_BLOCK,
        pages_per_block=NVFP4_PAGES_PER_BLOCK,
    )
    row_req, row_seq = decode_rows(seq, rows, q_rows)
    _check_table(page_table_ref, idx, bt, row_req, row_seq, *out, rows)


@table_kernels
@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_fused_selector_emits_the_page_table(phase, page_table_ref):
    """The Triton selector's fused emission over synthetic block scores: the
    same top-k ids as the page-16 emission (and as no emission), and the page
    table nvfp4_page_table builds from them."""
    from atom.model_ops.minimax_m3.index_topk import (
        DECODE_TOPK_BLOCK_SIZE_K,
        DECODE_TOPK_NUM_WARPS,
        NVFP4_PAGES_PER_BLOCK,
        PAGES_PER_SPARSE_BLOCK,
        PREFILL_TOPK_NUM_WARPS,
        _alloc_emit,
        _launch_select,
        _prefill_topk_block_size_k,
    )
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import decode_rows, prefill_rows

    torch.manual_seed(1)
    topk, init_blocks, local_blocks, maxb = 16, 1, 2, 64
    if phase == "decode":
        nreq, q_rows = 5, 4
        seq = torch.randint(1, maxb * PAGE, (nreq,), dtype=torch.int32).to(DEV)
        rows = nreq * q_rows
        starts = prefix = seq
        decode_max_q, block_k, warps = (
            q_rows,
            DECODE_TOPK_BLOCK_SIZE_K,
            DECODE_TOPK_NUM_WARPS,
        )
        row_req, row_seq = decode_rows(seq, rows, q_rows)
    else:
        qlens = torch.tensor([3, 130, 1, 40])
        nreq = qlens.numel()
        prefix = torch.tensor([0, 900, 5000, 7900], dtype=torch.int32).to(DEV)
        starts = torch.cat([torch.zeros(1), qlens.cumsum(0)]).to(torch.int32).to(DEV)
        rows, q_rows = int(qlens.sum()), int(qlens.max())
        decode_max_q, block_k, warps = (
            0,
            _prefill_topk_block_size_k(maxb),
            PREFILL_TOPK_NUM_WARPS,
        )
        row_req, row_seq = prefill_rows(starts, prefix, rows)
    bt = _block_table(nreq, maxb, 2)
    score = torch.randn(1, rows, maxb, dtype=torch.float32).to(DEV)

    def select(emit, ppb):
        idx = torch.empty((1, rows, topk), dtype=torch.int32, device=DEV)
        out, args = _alloc_emit(rows, 1, topk, bt, emit, DEV, ppb)
        _launch_select(
            score, idx, starts, prefix, nreq, topk, init_blocks, local_blocks, bt,
            args, 1, q_rows, decode_max_q, block_k, warps, emit, ppb,
            pages_per_block=ppb,
        )  # fmt: skip
        return idx, out

    idx, (table, ctx) = select(True, NVFP4_PAGES_PER_BLOCK)
    idx16, _ = select(True, PAGES_PER_SPARSE_BLOCK)
    idx0, _ = select(False, PAGES_PER_SPARSE_BLOCK)
    assert torch.equal(idx, idx16) and torch.equal(idx, idx0)
    _check_table(page_table_ref, idx, bt, row_req, row_seq, table, ctx, rows)


@table_kernels
def test_parallel_prefill_selector_emits_the_page_table(page_table_ref):
    """The two-pass prefill selector keeps its partial keys in the page-16 table;
    the page-128 one is too narrow, so it gets a workspace of its own."""
    from atom.model_ops.minimax_m3.index_topk import (
        NVFP4_PAGES_PER_BLOCK,
        PAGES_PER_SPARSE_BLOCK,
        PREFILL_TOPK_MAX_BLOCK_SIZE_K,
        _alloc_emit,
        _launch_parallel_prefill_selector,
    )
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import prefill_rows

    torch.manual_seed(2)
    topk, maxb = 16, 2 * PREFILL_TOPK_MAX_BLOCK_SIZE_K + 3
    qlens = torch.tensor([2, 5])
    prefix = torch.tensor([maxb * PAGE - 9, 1000], dtype=torch.int32).to(DEV)
    starts = torch.cat([torch.zeros(1), qlens.cumsum(0)]).to(torch.int32).to(DEV)
    rows = int(qlens.sum())
    bt = _block_table(2, maxb, 3)
    score = torch.randn(1, rows, maxb, dtype=torch.float32).to(DEV)

    def select(ppb):
        idx = torch.empty((1, rows, topk), dtype=torch.int32, device=DEV)
        out, args = _alloc_emit(rows, 1, topk, bt, True, DEV, ppb)
        _launch_parallel_prefill_selector(
            score, idx, starts, prefix, bt, args, int(qlens.max()), topk, 1, 2,
            ppb, pages_per_block=ppb,
        )  # fmt: skip
        return idx, out

    idx, (table, ctx) = select(NVFP4_PAGES_PER_BLOCK)
    idx16, _ = select(PAGES_PER_SPARSE_BLOCK)
    assert torch.equal(idx, idx16)
    row_req, row_seq = prefill_rows(starts, prefix, rows)
    _check_table(page_table_ref, idx, bt, row_req, row_seq, table, ctx, rows)


@gfx1250_only
@pytest.mark.parametrize("q_rows", [1, 8])
def test_cp_merge_emits_the_page_table(q_rows, page_table_ref):
    """Indexer CP: the candidate merge's emission (the near-term NVFP4 KV + FP8
    index config runs it every decode step)."""
    from atom.model_ops.minimax_m3.index_topk import NVFP4_PAGES_PER_BLOCK
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        local_candidate_keys,
        merge_candidate_keys,
    )
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import decode_rows

    torch.manual_seed(4)
    nreq, topk, maxb, world = 6, 16, 300, 4
    seq = torch.randint(1, maxb * PAGE, (nreq,), dtype=torch.int32, device=DEV)
    seq[0] = 3
    rows = nreq * q_rows
    bt = _block_table(nreq, maxb, 5)
    local = -(-maxb // world)
    keys = []
    for r in range(world):
        scores = torch.randn(1, rows, local, device=DEV)
        keys.append(
            local_candidate_keys(scores, seq, topk, r, world, q_rows, maxb, 1, 2)
        )
    keys = torch.cat(keys)
    idx, table, ctx = merge_candidate_keys(
        keys, bt, seq, topk, 1, 2, q_rows, pages_per_block=NVFP4_PAGES_PER_BLOCK
    )
    idx16, _, _ = merge_candidate_keys(keys, bt, seq, topk, 1, 2, q_rows)
    assert torch.equal(idx, idx16)
    row_req, row_seq = decode_rows(seq, rows, q_rows)
    _check_table(page_table_ref, idx, bt, row_req, row_seq, table, ctx, rows)


@writer_archs
@pytest.mark.parametrize("index_dtype", ["fp8", "auto"])
@pytest.mark.parametrize("tokens", [37, 2000])
def test_fused_nvfp4_write_equals_staging_then_store(index_dtype, tokens):
    """aiter's NVFP4 K/V write into the pool (strided field views) against the path
    it replaces: "auto" K/V into the BF16 staging buffer, then kv_nvfp4_store. The
    pool bytes, q_out, the index query and the index cache are identical."""
    import aiter
    from aiter import dtypes

    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
    from atom.model_ops.minimax_m3.index_fp4 import kv_nvfp4_store

    torch.manual_seed(6)
    heads, hkv, rot = 16, 1, 64
    blocks = -(-tokens // PAGE) + 4
    g = torch.Generator(device=DEV).manual_seed(7)
    row = (heads + 3 * hkv + 1) * D
    qkv = torch.randn(tokens, row, generator=g, device=DEV)
    qkv = (
        qkv.view(tokens, -1, D)
        * torch.exp2(torch.randint(-6, 5, (tokens, row // D, 1), device=DEV))
    ).view(tokens, row)
    qkv = qkv.to(torch.bfloat16)
    w = [
        torch.randn(D, generator=g, device=DEV).to(torch.bfloat16) * 0.3 + 1
        for _ in range(4)
    ]
    cos_sin = torch.randn(8192, rot, generator=g, device=DEV).to(torch.bfloat16)
    pos = torch.randint(0, 8192, (tokens,), device=DEV)
    slots = torch.randperm(blocks * PAGE, device=DEV)[:tokens].to(torch.int64)
    slots[2::29] = -1
    idx_t = dtypes.fp8 if index_dtype == "fp8" else torch.bfloat16

    def fused(k, v, ks, vs, index, kv_dtype, kv_slots):
        q_out = torch.empty(tokens, heads * D, dtype=torch.bfloat16, device=DEV)
        iq = torch.empty(tokens, hkv * D, dtype=torch.bfloat16, device=DEV)
        aiter.fused_qknorm_idxrqknorm(
            qkv.clone(), w[0], w[1], cos_sin, pos, heads, hkv, rot, 1e-6, w[2], w[3],
            hkv, kv_slots, k, v, index, PAGE, q_out, iq, slots,
            kv_cache_dtype=kv_dtype, index_cache_dtype=index_dtype,
            k_scale=ks, v_scale=vs, asm_layout=False,
        )  # fmt: skip
        return q_out, iq

    def pool_views():
        pool = MhaKvPool(
            layers=2, block_size=PAGE, num_kv_heads=hkv, head_dim=D,
            kv_dtype=torch.float8_e4m3fn, kv_format="nvfp4",
        )  # fmt: skip
        pool.allocate(blocks, torch.device(DEV))
        (kp, vp), (ks, vs) = pool.kv_views(1), pool.scale_views(1)
        for t in (kp, vp, ks, vs):
            t.fill_(0x5A)
        return kp, vp, ks, vs

    # The unfused path: identity-slot staging, then the Triton store.
    stage_pages = -(-tokens // PAGE)
    stage = torch.zeros(stage_pages, 2, PAGE, hkv, D, dtype=torch.bfloat16, device=DEV)
    ref_index = torch.zeros(blocks * PAGE, D, dtype=idx_t, device=DEV)
    ident = torch.arange(stage_pages * PAGE, dtype=torch.int64, device=DEV)
    ref_q, ref_iq = fused(
        stage[:, 0], stage[:, 1], None, None, ref_index, "auto", ident[:tokens]
    )
    rkp, rvp, rks, rvs = pool_views()
    kv_nvfp4_store(stage, slots, rkp, rks, rvp, rvs)

    kp, vp, ks, vs = pool_views()
    index = torch.zeros_like(ref_index)
    q, iq = fused(kp, vp, ks, vs, index, "nvfp4", slots)
    torch.cuda.synchronize()
    assert torch.equal(q, ref_q) and torch.equal(iq, ref_iq)
    assert torch.equal(index.view(torch.uint8), ref_index.view(torch.uint8))
    for got, want in ((kp, rkp), (vp, rvp), (ks, rks), (vs, rvs)):
        assert torch.equal(got, want)


@writer_archs
@pytest.mark.parametrize("tokens", [37, 2000])
@pytest.mark.parametrize("splits", [0, 4])
def test_fused_fp4_index_write_equals_staging_then_store(tokens, splits):
    """aiter's NVFP4 index rows (index_cache_dtype="nvfp4" + index_scale, strided
    pool fields) in the NVFP4 K/V launch, against what attention_mha did before:
    the index key "auto" into the token-major staging buffer, then index_fp4_store
    at the real slots. Index planes bitwise -- zero- and subnormal-scale blocks
    included -- and the K/V pool, q_out and the index query as the "auto"-index
    call writes them."""
    import aiter

    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
    from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store

    heads, hkv, rot = 16, 1, 64
    blocks = -(-tokens // PAGE) + 4
    g = torch.Generator(device=DEV).manual_seed(17 + splits)
    row = (heads + 3 * hkv + 1) * D
    qkv = torch.randn(tokens, row, generator=g, device=DEV)
    qkv = (
        qkv.view(tokens, -1, D)
        * torch.exp2(torch.randint(-6, 5, (tokens, row // D, 1), device=DEV))
    ).view(tokens, row)
    ik = slice((heads + 3 * hkv) * D, row)
    qkv[3::7, ik] *= 1e-4  # index keys whose blocks quantize to zero / subnormal scales
    qkv = qkv.to(torch.bfloat16)
    partials = None
    if splits:
        partials = torch.randn(splits, tokens, row, generator=g, device=DEV)
        acc = partials[0].clone()
        for i in range(1, splits):
            acc += partials[i]
        qkv = acc.to(torch.bfloat16)  # what the kernel reduces them to
    w = [
        torch.randn(D, generator=g, device=DEV).to(torch.bfloat16) * 0.3 + 1
        for _ in range(4)
    ]
    cos_sin = torch.randn(8192, rot, generator=g, device=DEV).to(torch.bfloat16)
    pos = torch.randint(0, 8192, (tokens,), device=DEV)
    slots = torch.randperm(blocks * PAGE, device=DEV)[:tokens].to(torch.int64)
    slots[2::29] = -1

    def pool_views():
        pool = MhaKvPool(
            layers=2, block_size=PAGE, num_kv_heads=hkv, head_dim=D,
            kv_dtype=torch.float8_e4m3fn, kv_format="nvfp4",
        )  # fmt: skip
        pool.allocate(blocks, torch.device(DEV))
        (kp, vp), (ks, vs) = pool.kv_views(1), pool.scale_views(1)
        for t in (kp, vp, ks, vs):
            t.fill_(0x5A)
        return kp, vp, ks, vs

    def index_planes():
        # Two fields of one entry-major region, as the pool binds them.
        pay, sc = PAGE * D // 2, PAGE * D // 16
        region = torch.full(
            (blocks, pay + sc + 32), 0x5A, dtype=torch.uint8, device=DEV
        )
        return region[:, :pay].unflatten(1, (PAGE, D // 2)), region[
            :, pay : pay + sc
        ].unflatten(1, (PAGE, D // 16))

    def fused(index, index_dtype, index_slots, index_scale=None, with_partials=False):
        kp, vp, ks, vs = pool_views()
        q_out = torch.empty(tokens, heads * D, dtype=torch.bfloat16, device=DEV)
        iq = torch.empty(tokens, hkv * D, dtype=torch.bfloat16, device=DEV)
        src = torch.zeros_like(qkv) if with_partials else qkv.clone()
        aiter.fused_qknorm_idxrqknorm(
            src, w[0], w[1], cos_sin, pos, heads, hkv, rot, 1e-6, w[2], w[3],
            hkv, slots, kp, vp, index, PAGE, q_out, iq, index_slots,
            kv_cache_dtype="nvfp4", index_cache_dtype=index_dtype,
            k_scale=ks, v_scale=vs, asm_layout=False,
            **({"qkv_partials": partials} if with_partials else {}),
            **({} if index_scale is None else {"index_scale": index_scale}),
        )  # fmt: skip
        return q_out, iq, (kp, vp, ks, vs)

    # Before: the key "auto" into identity-slot staging, then index_fp4_store.
    stage = torch.zeros(-(-tokens // PAGE) * PAGE, D, dtype=torch.bfloat16, device=DEV)
    ident = torch.arange(stage.shape[0], dtype=torch.int64, device=DEV)
    ref_q, ref_iq, ref_kv = fused(stage, "auto", ident[:tokens])
    rp, rs = index_planes()
    index_fp4_store(stage, slots, rp, rs)

    ip, isc = index_planes()
    q, iq, kv = fused(ip, "nvfp4", slots, isc, with_partials=bool(splits))
    torch.cuda.synchronize()
    assert torch.equal(q, ref_q) and torch.equal(iq, ref_iq)
    assert torch.equal(ip, rp) and torch.equal(isc, rs)
    for got, want in zip(kv, ref_kv):
        assert torch.equal(got, want)


@gfx1250_only
@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.parametrize("rows, pad", [(64, 0), (37, 11), (600, 0)])
def test_nvfp4_attention_fp8_output_is_bf16_then_quant(phase, rows, pad, monkeypatch):
    """o_proj's FP8 input from the attention epilogue equals the BF16 output of the
    same split quantized per token, bit for bit (aiter's FP8 page16 test holds the
    epilogue to the same oracle); a padded bucket's extra rows are FP8 zeros with a
    unit descale. Decode rows take the cluster-of-8 split, prefill rows one task
    each, 600 rows aiter's fallback.

    The oracle is the BF16 output that call itself stores (store_output forced on):
    the FP8 split differs from the BF16-only call's (prefill: one task a row instead
    of get_recommended_splits), and a different split merges the softmax in another
    order, so that output differs from it by BF16 rounding. It is checked against it
    with a tolerance only."""
    import aiter.ops.flydsl.pa_decode as flydsl_pa
    from aiter import dtypes
    from aiter.ops.quant import dynamic_per_token_scaled_quant

    from atom.model_ops.minimax_m3.index_fp4 import kv_nvfp4_store
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import (
        decode_rows,
        nvfp4_page_table,
        nvfp4_sparse_attention,
    )

    torch.manual_seed(9)
    blocks, topk, maxb = 4096, 16, 48
    stage = torch.randn(blocks, 2, PAGE, 1, D, device=DEV).to(torch.bfloat16)
    kp = torch.zeros(blocks, 1, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    ks = torch.zeros(blocks, 1, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    vp, vs = torch.zeros_like(kp), torch.zeros_like(ks)
    kv_nvfp4_store(stage, torch.arange(blocks * PAGE, device=DEV), kp, ks, vp, vs)
    seq = torch.randint(PAGE, maxb * PAGE, (rows,), dtype=torch.int32, device=DEV)
    bt = torch.stack([torch.randperm(blocks, device=DEV)[:maxb] for _ in range(rows)])
    bt = bt.to(torch.int32)
    row_req, row_seq = decode_rows(seq, rows, 1)
    idx = torch.full((1, rows, topk), -1, dtype=torch.int32, device=DEV)
    for r in range(rows):
        n = int((row_seq[r] + PAGE - 1) // PAGE)
        sel = torch.randperm(max(n - 1, 1), device=DEV)[: topk - 1]
        idx[0, r, : sel.numel()] = sel.to(torch.int32)
        idx[0, r, topk - 1] = n - 1
    q = torch.randn(rows, 16, D, device=DEV).to(torch.bfloat16)
    table = nvfp4_page_table(idx, bt, row_req, row_seq)
    args = (q, kp, ks, vp, vs, idx, bt, row_req, row_seq, D**-0.5)
    bf16_only = nvfp4_sparse_attention(*args, page_table=table)

    pa_decode = flydsl_pa.pa_decode

    def keep_bf16(*a, **kw):
        kw["store_output"] = True
        return pa_decode(*a, **kw)

    monkeypatch.setattr(flydsl_pa, "pa_decode", keep_bf16)
    fp8 = torch.full((rows + pad, 16, D), 0x7F, dtype=torch.uint8, device=DEV)
    fp8 = fp8.view(dtypes.fp8)
    scales = torch.full((rows + pad, 1), float("nan"), device=DEV)
    out = nvfp4_sparse_attention(
        *args, page_table=table, quantized_output=fp8, output_scale=scales,
        quantized_direct=phase == "prefill",
    )  # fmt: skip
    ref_q = torch.empty(rows, 16 * D, dtype=dtypes.fp8, device=DEV)
    ref_s = torch.empty(rows, 1, dtype=torch.float32, device=DEV)
    dynamic_per_token_scaled_quant(ref_q, out.view(rows, -1), ref_s)
    torch.cuda.synchronize()
    assert torch.equal(
        fp8[:rows].view(rows, -1).view(torch.uint8), ref_q.view(torch.uint8)
    )
    assert torch.equal(scales[:rows], ref_s)
    assert not fp8[rows:].view(torch.uint8).any()
    assert torch.equal(scales[rows:], torch.ones_like(scales[rows:]))
    torch.testing.assert_close(out, bf16_only, atol=2e-2, rtol=2e-2)


@gfx1250_only
@pytest.mark.parametrize("hkv", [1, 4], ids=["hkv1", "hkv4"])
@pytest.mark.parametrize("phase", ["decode", "prefill", "prefill-reuse"])
@pytest.mark.parametrize("rows, pad", [(8, 0), (64, 0), (37, 11), (512, 0), (2048, 3)])
def test_nvfp4_attention_mxfp8_output_is_bf16_then_quantize_fp8(
    phase, rows, pad, hkv, monkeypatch
):
    """o_proj's MXFP8 input from the attention (ATOM_MINIMAX_M3_PA_MXFP8_OUT)
    equals the split path it replaces bit for bit: the BF16-only attention, then
    o_proj's own ``quantize_fp8``. The MXFP8 output keeps the BF16 call's split
    (aiter writes it in every output mode), so the oracle is the BF16-only call
    itself, not a re-split one. A padded bucket's extra rows are FP8 zeros with
    E8M0 1.0 scales. ``hkv`` 4: PP4 x TP1's 64 query heads over 4 KV heads a GPU,
    the (row, kv head) sequences writing the row-major ``[rows, 64 * D]``
    activation. ``prefill-reuse``: with ATOM_M3_SPARSE_PREFILL_REUSE."""
    from aiter import dtypes

    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.minimax_m3.index_fp4 import kv_nvfp4_store
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import (
        decode_rows,
        nvfp4_page_table,
        nvfp4_sparse_attention,
    )

    torch.manual_seed(11)
    reuse = phase == "prefill-reuse"
    monkeypatch.setenv("ATOM_M3_SPARSE_PREFILL_REUSE_MIN_ROWS", "1")
    # The image's aiter pads the reuse kernel's MXFP8 bucket for one kv head
    # only, so with more the MXFP8 prefill runs the paged decode (sparse_attn_nvfp4):
    # its oracle is that kernel's BF16 output.
    monkeypatch.setenv("ATOM_M3_SPARSE_PREFILL_REUSE", str(int(reuse and hkv == 1)))
    blocks, topk, maxb, heads = 4096 // hkv, 16, 48, 16 * hkv
    stage = torch.randn(blocks, 2, PAGE, hkv, D, device=DEV).to(torch.bfloat16)
    kp = torch.zeros(blocks, hkv, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    ks = torch.zeros(blocks, hkv, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    vp, vs = torch.zeros_like(kp), torch.zeros_like(ks)
    kv_nvfp4_store(stage, torch.arange(blocks * PAGE, device=DEV), kp, ks, vp, vs)
    seq = torch.randint(PAGE, maxb * PAGE, (rows,), dtype=torch.int32, device=DEV)
    bt = torch.randint(0, blocks, (rows, maxb), dtype=torch.int32, device=DEV)
    row_req, row_seq = decode_rows(seq, rows, 1)
    n = (row_seq + PAGE - 1) // PAGE
    # Each kv head its own selection.
    idx = (torch.rand(hkv, rows, topk, device=DEV) * (n[None, :, None] - 1)).int()
    idx[:, :, topk - 1] = n - 1
    q = torch.randn(rows, heads, D, device=DEV).to(torch.bfloat16)
    q[:, ::5] *= 16  # heads of very different output magnitude
    table = nvfp4_page_table(idx, bt, row_req, row_seq)
    args = (q, kp, ks, vp, vs, idx, bt, row_req, row_seq, D**-0.5)
    direct = phase != "decode"
    bf16_only = nvfp4_sparse_attention(*args, page_table=table, quantized_direct=direct)
    ref_q, ref_s = quantize_fp8(bf16_only.view(rows, -1))
    monkeypatch.setenv("ATOM_M3_SPARSE_PREFILL_REUSE", str(int(reuse)))

    fp8 = torch.full((rows + pad, heads * D), 0x7F, dtype=torch.uint8, device=DEV)
    fp8 = fp8.view(dtypes.fp8)
    scales = torch.full(
        (rows + pad, heads * D // 32), 0xFF, dtype=torch.uint8, device=DEV
    ).view(torch.float8_e8m0fnu)
    nvfp4_sparse_attention(
        *args, page_table=table, quantized_output=fp8, output_scale=scales,
        quantized_direct=direct,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(fp8[:rows].view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(scales[:rows].view(torch.uint8), ref_s.view(torch.uint8))
    assert not fp8[rows:].view(torch.uint8).any()
    assert (scales[rows:].view(torch.uint8) == 127).all()


@gfx1250_only
@pytest.mark.parametrize("rows, pad", [(24, 0), (64, 0), (192, 5), (256, 0), (40, 3)])
@pytest.mark.parametrize("mxfp8", [True, False], ids=["mxfp8", "bf16"])
def test_nvfp4_attention_fused_reduce(rows, pad, mxfp8, monkeypatch):
    """ATOM_M3_PA_FUSED_REDUCE: decode rows merge their top-k window inside the
    paged attention (a cluster a row while every row's cluster is resident,
    else one CTA a row) instead of the PS reduce kernel. The MXFP8 bytes are
    o_proj's ``quantize_fp8`` of the BF16 output of that same call (store_output
    forced on), which matches the unfused split's BF16 output to BF16 rounding
    (a different split merges the softmax in another order); padded bucket rows
    are FP8 zeros with E8M0 1.0 scales."""
    import aiter.ops.flydsl.pa_decode as flydsl_pa
    from aiter import dtypes

    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.minimax_m3.index_fp4 import kv_nvfp4_store
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import (
        decode_rows,
        nvfp4_page_table,
        nvfp4_sparse_attention,
    )

    torch.manual_seed(13)
    blocks, topk, maxb, ql = 4096, 16, 48, 8
    stage = torch.randn(blocks, 2, PAGE, 1, D, device=DEV).to(torch.bfloat16)
    kp = torch.zeros(blocks, 1, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    ks = torch.zeros(blocks, 1, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    vp, vs = torch.zeros_like(kp), torch.zeros_like(ks)
    kv_nvfp4_store(stage, torch.arange(blocks * PAGE, device=DEV), kp, ks, vp, vs)
    reqs = rows // ql
    seq = torch.randint(PAGE, maxb * PAGE, (reqs,), dtype=torch.int32, device=DEV)
    seq[::3] = 200  # short requests: fewer than 16 valid blocks
    bt = torch.randint(0, blocks, (reqs, maxb), dtype=torch.int32, device=DEV)
    row_req, row_seq = decode_rows(seq, rows, ql)
    n = (row_seq + PAGE - 1) // PAGE
    idx = (torch.rand(1, rows, topk, device=DEV) * (n[None, :, None] - 1)).int()
    idx[0, :, topk - 1] = n - 1
    q = torch.randn(rows, 16, D, device=DEV).to(torch.bfloat16)
    q[:, ::5] *= 16
    table = nvfp4_page_table(idx, bt, row_req, row_seq)
    args = (q, kp, ks, vp, vs, idx, bt, row_req, row_seq, D**-0.5)
    bf16_split = nvfp4_sparse_attention(*args, page_table=table, quantized_direct=False)

    monkeypatch.setenv("ATOM_M3_PA_FUSED_REDUCE", "1")
    pa_decode = flydsl_pa.pa_decode
    seen = []

    def keep_bf16(*a, **kw):
        seen.append(kw)
        kw["store_output"] = True
        return pa_decode(*a, **kw)

    monkeypatch.setattr(flydsl_pa, "pa_decode", keep_bf16)
    kw = {}
    if mxfp8:
        fp8 = torch.full((rows + pad, 16 * D), 0x7F, dtype=torch.uint8, device=DEV)
        fp8 = fp8.view(dtypes.fp8)
        scales = torch.full(
            (rows + pad, 16 * D // 32), 0xFF, dtype=torch.uint8, device=DEV
        ).view(torch.float8_e8m0fnu)
        kw = {"quantized_output": fp8, "output_scale": scales}
    out = nvfp4_sparse_attention(
        *args, page_table=table, quantized_direct=False, rows_per_request=ql, **kw
    )
    torch.cuda.synchronize()
    assert seen and seen[-1].get("plan_free") and seen[-1].get("xcd_rows") == ql
    torch.testing.assert_close(out, bf16_split, atol=2e-2, rtol=2e-2)
    if mxfp8:
        ref_q, ref_s = quantize_fp8(out.view(rows, -1))
        assert torch.equal(fp8[:rows].view(torch.uint8), ref_q.view(torch.uint8))
        assert torch.equal(scales[:rows].view(torch.uint8), ref_s.view(torch.uint8))
        assert not fp8[rows:].view(torch.uint8).any()
        assert (scales[rows:].view(torch.uint8) == 127).all()
