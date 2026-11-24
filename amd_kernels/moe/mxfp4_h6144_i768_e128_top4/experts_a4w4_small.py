# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The W4A4 expert host on 16 / 32-row blocks (to 512 tokens) with the shared expert's tiles leading both routed
launches: rows routed inline (``*_inline_sort``), built by GEMM1 from ``topk_ids`` (``*_id_rows``) or sorted by
``sort_a16w4``."""
import torch
from .sort_a16w4 import moe_sort_a16w4
from .common.launch import _run_compiled

# inline routing to here (from 17 an expert may need two 16-row blocks): -0.27..-0.38 us at 16, -1.0..-2.4 at 12-15
INLINE_SORT_MAX = 16

# BF16 entry with the shared expert: GEMM1 builds its 16-row blocks from topk_ids (``*_id_rows``) at these rows
ID_ROWS_FROM, ID_ROWS_MAX = 16, 96


def id_rows_for(n):
    return ID_ROWS_FROM <= n <= ID_ROWS_MAX


def id_pairs_for(n):
    """Routing pair slots GEMM1 scans in the id-rows tier (a trace-time int, >= 4 * n)."""
    return 64 if n <= 16 else 128 if n <= 32 else 256 if n <= 64 else 512


def a4w4_experts(x_quant, w13, w2, scales, topk_weights, topk_ids, *, shared, kw=None, pf=3, tn=256, bm=None,
                 inline=None, id_rows=False):
    """``shared`` = (MXFP8 codes, scales, SharedWeights): the shared gate / up tiles run first in the routed GEMM1
    launch, the shared down tiles first in the routed GEMM2 launch (tile rows and K-waves by ``shared_fold_tile_for``);
    returns (slots, shared)."""
    xq, xs = x_quant
    n = xq.shape[0]
    selected_kw, selected_bm = small_config_for(n)
    if kw is None:
        kw = selected_kw
    if bm is None:
        bm = selected_bm
    if inline is None:
        inline = n <= INLINE_SORT_MAX and not id_rows
    if inline:
        assert n <= 16 and bm == 16
        from .gemm1_a4w4_inline_sort import compile_gemm1_shared_first
        from .gemm2_a4w4_inline_sort import compile_gemm2_shared_first as get_gemm2_sf
        ids, wt, eids, nv = topk_ids.data_ptr(), topk_weights.view(-1), topk_ids.view(-1), topk_ids.view(-1)
        rows = n * 4 * 16
    elif id_rows:
        assert n <= ID_ROWS_MAX and kw == 4 and bm == 16
        from .gemm1_a4w4_id_rows import compile_gemm1_shared_first
        from .gemm2_a4w4_id_rows import compile_gemm2_shared_first as get_gemm2_sf
        from .sort_a16w4 import max_sorted_rows
        rows = max_sorted_rows(n, 128, 4, bm)
        # GEMM1 reads topk_ids, writes the block -> expert map (+ block count) and the rows
        eids = torch.empty(rows // 16 + 1, device=xq.device, dtype=torch.int32)
        nv = torch.empty(rows, device=xq.device, dtype=torch.int32)
        ids, wt = topk_ids.view(-1), topk_weights.view(-1)
    else:
        from .gemm1_a4w4_small import compile_gemm1_shared_first
        from .gemm2_a4w4_small import compile_gemm2_shared_first as get_gemm2_sf
        ids, wt, eids, nv = moe_sort_a16w4(topk_ids, topk_weights, 128, bm)
        rows = ids.numel()
    q = torch.empty((rows, 384), device=xq.device, dtype=torch.uint8)
    s = torch.empty((rows, 24), device=xq.device, dtype=torch.uint8)
    out = torch.empty((n, 4, 6144), device=xq.device, dtype=torch.bfloat16)
    routing_args = (ids,) if inline else (ids, eids, nv)
    ids2 = nv if id_rows else ids  # GEMM2's rows: GEMM1's published table in the id-rows tier
    stream = torch.cuda.current_stream()
    sh_bm, sh_kw, sh_pf = shared_fold_tile_for(n)
    assert kw == 4 and (bm == 16 or not inline) and ((sh_bm, sh_kw, sh_pf) == (16, 4, 2) or not inline)
    q8, s8, sw = shared
    hq = torch.empty((n, 768), device=xq.device, dtype=torch.uint8)
    hs = torch.empty((n, 24), device=xq.device, dtype=torch.uint8)
    nsh = (n + sh_bm - 1) // sh_bm * (24 // (4 // sh_kw))
    g1_kw = {} if inline else {"sh_bm": sh_bm, "sh_kw": sh_kw, "sh_pf": sh_pf}
    if id_rows:
        g1_kw["pairs"] = id_pairs_for(n)
    _run_compiled(compile_gemm1_shared_first(kw=kw, pf=pf, bm=bm, **g1_kw), xq.view(-1),
                  xs.view(torch.uint8).view(-1), w13.view(-1), scales.w13.view(-1), q.view(-1), s.view(-1),
                  *routing_args, q8.view(-1), s8.view(-1), sw.w13.view(-1), sw.s13.view(-1),
                  hq.view(-1), hs.view(-1), n, rows, nsh, stream)
    sh_tn = shared_g2_columns_for(n)
    sh_out = torch.empty((n, 6144), device=xq.device, dtype=torch.bfloat16)
    g2_bm = shared_fold_down_rows_for(sh_bm, bm)
    g2_kw = {} if inline else {"sh_bm": g2_bm}
    _run_compiled(get_gemm2_sf(tn=tn, bm=bm, sh_tn=sh_tn, **g2_kw), q.view(-1), s.view(-1), w2.view(-1),
                  scales.w2.view(-1), out.view(-1), ids2, wt, *((eids, nv) if not inline else ()),
                  hq.view(-1), hs.view(-1), sw.w2.view(-1), sw.s2.view(-1), sh_out.view(-1),
                  n, rows, (n + g2_bm - 1) // g2_bm * (6144 // sh_tn), stream)
    return out, sh_out


def shared_g2_columns_for(n):
    """Columns per shared down tile inside the routed GEMM2 launch."""
    return 64 if n <= 16 else 256


def shared_fold_tile_for(n):
    """(rows, GEMM1 K-waves, GEMM1 prefetch depth) of the shared tiles that lead the routed launches: fewer K-waves
    give a shared workgroup more columns, so the activations are read fewer times; 64-row tiles read the gate / up
    weight a quarter as often as 16-row ones, but need prefetch depth 1 to keep the launch at two waves per SIMD, and
    their idle rows cost more at <= 128 rows."""
    if n <= 96:
        return 16, 4, 2
    if n <= 128:
        return 16, 2, 2
    return (64, 4, 1) if n <= 384 else (64, 2, 1)


def shared_fold_down_rows_for(sh_bm, bm):
    """Rows per folded shared down tile: at most the routed tile's rows, so the launch keeps the routed GEMM2's LDS
    (rows x 1 KB) and registers (16-row down tiles next to 16-row routed tiles: -0.4..-1.0% at 129-256 rows; next to
    32-row routed tiles: +0.9..+1.1%)."""
    return min(sh_bm, bm)


def small_config_for(n):
    return 4, (16 if n <= 256 else 32)
