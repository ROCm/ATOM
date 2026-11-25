# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse KV: the fused paths against the unfused ones they replace.

The NVFP4 attention (gfx1250) used to cost extra launches a sparse layer over the
FP8 one. Each is now folded into a launch the layer runs anyway, and each test
pins the fused result bit for bit to the unfused one:

- the page-128 table: emitted by the top-k selectors (TP decode / prefill, the
  emit after aiter's selector, the indexer-CP merge) instead of
  ``nvfp4_page_table``'s own launch;
- the K/V write: aiter's ``fused_qknorm_idxrqknorm`` quantizes K/V into the NVFP4
  pool (``kv_cache_dtype="nvfp4"`` with the scale planes) instead of writing them to
  a BF16 staging buffer that ``kv_nvfp4_store`` quantizes;
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
        SPARSE_BLOCK_SIZE,
        _alloc_emit,
        _emit_sparse_block_table_kernel,
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
    sbt, sctx, bt_stride0, sbt_stride0 = args
    # As `_launch_select` launches it after aiter's selector.
    _emit_sparse_block_table_kernel[(q_rows, nreq, hkv)](
        idx, seq, seq, topk, *idx.stride(), bt, sbt, sctx, bt_stride0, sbt_stride0,
        1, SPARSE_BLOCK_SIZE,
        NUM_KV_HEADS=hkv,
        DECODE_MAX_Q=q_rows,
        pages_per_block=NVFP4_PAGES_PER_BLOCK,
        block_page_stride=NVFP4_PAGES_PER_BLOCK,
        num_warps=1,
    )  # fmt: skip
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
def test_fused_fp4_index_write_equals_staging_then_store(tokens):
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
    g = torch.Generator(device=DEV).manual_seed(17)
    row = (heads + 3 * hkv + 1) * D
    qkv = torch.randn(tokens, row, generator=g, device=DEV)
    qkv = (
        qkv.view(tokens, -1, D)
        * torch.exp2(torch.randint(-6, 5, (tokens, row // D, 1), device=DEV))
    ).view(tokens, row)
    ik = slice((heads + 3 * hkv) * D, row)
    qkv[3::7, ik] *= 1e-4  # index keys whose blocks quantize to zero / subnormal scales
    qkv = qkv.to(torch.bfloat16)
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

    def fused(index, index_dtype, index_slots, index_scale=None):
        kp, vp, ks, vs = pool_views()
        q_out = torch.empty(tokens, heads * D, dtype=torch.bfloat16, device=DEV)
        iq = torch.empty(tokens, hkv * D, dtype=torch.bfloat16, device=DEV)
        aiter.fused_qknorm_idxrqknorm(
            qkv.clone(), w[0], w[1], cos_sin, pos, heads, hkv, rot, 1e-6, w[2], w[3],
            hkv, slots, kp, vp, index, PAGE, q_out, iq, index_slots,
            kv_cache_dtype="nvfp4", index_cache_dtype=index_dtype,
            k_scale=ks, v_scale=vs, asm_layout=False,
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
    q, iq, kv = fused(ip, "nvfp4", slots, isc)
    torch.cuda.synchronize()
    assert torch.equal(q, ref_q) and torch.equal(iq, ref_iq)
    assert torch.equal(ip, rp) and torch.equal(isc, rs)
    for got, want in zip(kv, ref_kv):
        assert torch.equal(got, want)


@gfx1250_only
@pytest.mark.parametrize("rows", [8, 37, 64, 512, 2048])
def test_nvfp4_attention_mxfp8_output_is_bf16_then_quantize_fp8(rows):
    """o_proj's MXFP8 input from the attention (ATOM_MINIMAX_M3_PA_MXFP8_OUT)
    equals the split path it replaces bit for bit: the BF16-only attention, then
    o_proj's own ``quantize_fp8``. The MXFP8 output keeps the BF16 call's split
    (aiter writes it in every output mode), so the oracle is the BF16-only call
    itself, not a re-split one. Rows run from one cluster to past one resident
    wave of clusters."""
    from aiter import dtypes

    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.minimax_m3.index_fp4 import kv_nvfp4_store
    from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import (
        decode_rows,
        nvfp4_page_table,
        nvfp4_sparse_attention,
    )

    torch.manual_seed(11)
    blocks, topk, maxb = 4096, 16, 48
    stage = torch.randn(blocks, 2, PAGE, 1, D, device=DEV).to(torch.bfloat16)
    kp = torch.zeros(blocks, 1, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    ks = torch.zeros(blocks, 1, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    vp, vs = torch.zeros_like(kp), torch.zeros_like(ks)
    kv_nvfp4_store(stage, torch.arange(blocks * PAGE, device=DEV), kp, ks, vp, vs)
    seq = torch.randint(PAGE, maxb * PAGE, (rows,), dtype=torch.int32, device=DEV)
    bt = torch.randint(0, blocks, (rows, maxb), dtype=torch.int32, device=DEV)
    row_req, row_seq = decode_rows(seq, rows, 1)
    n = (row_seq + PAGE - 1) // PAGE
    idx = (torch.rand(1, rows, topk, device=DEV) * (n[None, :, None] - 1)).int()
    idx[0, :, topk - 1] = n - 1
    q = torch.randn(rows, 16, D, device=DEV).to(torch.bfloat16)
    q[:, ::5] *= 16  # heads of very different output magnitude
    table = nvfp4_page_table(idx, bt, row_req, row_seq)
    args = (q, kp, ks, vp, vs, idx, bt, row_req, row_seq, D**-0.5)
    bf16_only = nvfp4_sparse_attention(*args, page_table=table)
    ref_q, ref_s = quantize_fp8(bf16_only.view(rows, -1))

    fp8 = torch.full((rows, 16 * D), 0x7F, dtype=torch.uint8, device=DEV)
    fp8 = fp8.view(dtypes.fp8)
    scales = torch.full((rows, 16 * D // 32), 0xFF, dtype=torch.uint8, device=DEV)
    scales = scales.view(torch.float8_e8m0fnu)
    nvfp4_sparse_attention(
        *args, page_table=table, quantized_output=fp8, output_scale=scales,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(fp8.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), ref_s.view(torch.uint8))


@gfx1250_only
def test_nvfp4_prefill_reuse_matches_the_paged_decode(monkeypatch):
    from atom.model_ops.minimax_m3 import sparse_attn_nvfp4 as sa
    from atom.model_ops.minimax_m3.index_fp4 import kv_nvfp4_store

    torch.manual_seed(17)
    blocks, topk, maxb, rows = 1024, 16, 48, 512
    stage = torch.randn(blocks, 2, PAGE, 1, D, device=DEV).to(torch.bfloat16)
    kp = torch.zeros(blocks, 1, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    ks = torch.zeros(blocks, 1, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    vp, vs = torch.zeros_like(kp), torch.zeros_like(ks)
    kv_nvfp4_store(stage, torch.arange(blocks * PAGE, device=DEV), kp, ks, vp, vs)
    # One request: 512 new tokens on a 4096-token prefix (33-36 valid pages).
    cu = torch.tensor([0, rows], dtype=torch.int32, device=DEV)
    prefix = torch.tensor([4096], dtype=torch.int32, device=DEV)
    row_req, row_seq = sa.prefill_rows(cu, prefix, rows)
    bt = torch.randperm(blocks, device=DEV)[:maxb].int().unsqueeze(0)
    idx = torch.rand(rows, 32, device=DEV).argsort(-1)[:, :topk].int()
    idx[:, -1] = (row_seq + PAGE - 1) // PAGE - 1
    idx = idx.sort(-1).values.unsqueeze(0)
    q = torch.randn(rows, 16, D, device=DEV).to(torch.bfloat16)
    table = sa.nvfp4_page_table(idx, bt, row_req, row_seq)
    args = (q, kp, ks, vp, vs, idx, bt, row_req, row_seq, D**-0.5)
    paged = sa.nvfp4_sparse_attention(*args, page_table=table)

    kernel, served = sa._prefill_reuse_kernel()
    used = []
    monkeypatch.setattr(
        sa,
        "_prefill_reuse_kernel",
        lambda: (lambda *a, **kw: used.append(1) or kernel(*a, **kw), served),
    )
    out = sa.nvfp4_sparse_attention(*args, page_table=table, prefill=True)
    torch.cuda.synchronize()
    assert used, "the K/V-reuse kernel did not take the prefill"
    torch.testing.assert_close(out, paged, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "prefill, rows, served, reuse",
    [(True, 256, True, True), (False, 512, True, False), (True, 255, True, False),
     (True, 512, False, False)],
    ids=["long-prefill", "decode", "short-prefill", "unserved"],
)  # fmt: skip
def test_prefill_reuse_dispatch(monkeypatch, prefill, rows, served, reuse):
    import importlib

    from atom.model_ops import base_attention
    from atom.model_ops.minimax_m3 import sparse_attn_nvfp4

    calls = {"reuse": [], "pa_decode": []}
    monkeypatch.setattr(
        sparse_attn_nvfp4,
        "_prefill_reuse_kernel",
        lambda: (lambda *a, **kw: calls["reuse"].append(kw), lambda *_: served),
    )
    monkeypatch.setattr(
        importlib.import_module("aiter.ops.flydsl.pa_decode"),
        "pa_decode",
        lambda *a, **kw: calls["pa_decode"].append(kw),
    )
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits", lambda *_: 3
    )
    monkeypatch.setattr(
        base_attention, "_flydsl_prepare_explicit_plan", lambda **kw: None
    )
    monkeypatch.setattr(
        base_attention, "_flydsl_plan_scratch", lambda *a, **kw: (None,) * 3
    )
    payload = torch.zeros(4, 1, 128, 64, dtype=torch.uint8)
    scales = torch.zeros(4, 1, 128, 8, dtype=torch.uint8)
    e8m0 = torch.empty(rows, 64, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sparse_attn_nvfp4._flydsl_attention(
        torch.randn(rows, 16, 128, dtype=torch.bfloat16), payload, scales, payload,
        scales, torch.zeros(rows, 1, dtype=torch.int32),
        torch.ones(rows, dtype=torch.int32), 128**-0.5, None, None, prefill=prefill,
        quantized_output=torch.empty(rows, 16 * 128, dtype=torch.uint8),
        output_scale=e8m0,
    )  # fmt: skip
    assert (len(calls["reuse"]), len(calls["pa_decode"])) == (
        (1, 0) if reuse else (0, 1)
    )
    if reuse:
        kw = calls["reuse"][0]
        assert kw["tile_rows"] == 2 and kw["impl"] == "v2"
        assert kw["quantized_output"].shape == (rows, 16, 128)
        assert kw["output_scale"] is e8m0
