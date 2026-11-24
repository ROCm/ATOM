# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1 decode MoE block (``ak_moe``) at 17..128 rows with retuned shared-expert tiles
(``ATOM_P1_MOE_TILES``, default off).

The same launches and kernel bodies as ``amd_kernels`` ``mxfp4_moe`` (router, routed GEMM1 with the shared gate/up
tiles first, routed GEMM2 with the shared down tiles first, slot + shared sum), only the tiling of the shared tiles
inside the routed launches (rows per tile, columns per down tile, prefetch depth) and the routed GEMM2's prefetch depth
differ. No K split changes, so every output is the same bits as the entry (bit-identical on every slot, eager and graph,
p1/small-g-moe sgm_bench). Measured per MoE layer on one MI355X (cold weights, real routing): -0.4..-2.4 us at 24-128 rows vs the entry.

``ATOM_P1_MOE_IDROWS`` (with ``ATOM_P1_MOE_TILES``, default off, p1/moe-gemm-headroom): 97-128 rows run the id-rows host
(GEMM1 builds its 16-row blocks from topk_ids: no sort launch) with the 65-96 shared tiling instead of the sorted host
(-7.4..-8.9 us per layer), 8 rows the id-rows host with shared gate/up prefetch 3 (-1.5 us), 16 rows routed GEMM1
prefetch 6 + 128-column GEMM2 tiles (-0.8..-1.1 us). Same kernel bodies, same K order: bit-identical on every slot.

``ATOM_P1_MBF_ASM`` (with ``ATOM_P1_MOE_IDROWS``, default off, p1/mbf-build-asm): at 32-128 rows the routed GEMM2 prefetches
3 K-steps instead of 2, and GEMM1's routed waves run their prologue (routing-id load, histogram, first weight loads) at
issue priority 3 (``ak_moe_g1prio``: the id-rows GEMM1 plus two ``s_setprio``). Measured per MoE layer on one MI355X (cold
weights, real routing, graph): -0.3..-0.6 us at 32-56 rows, -0.7..-1.2 us at 64-128 rows; 16/24 rows unchanged (both knobs
lose there). Same tiles, same K order: bit-identical on every slot.
"""

import torch

H = 6144
ROWS = (17, 128)


IDROWS_SMALL = (8, 16)


def _idrows() -> bool:
    from atom.utils import envs

    return envs.ATOM_P1_MOE_IDROWS


def _asm() -> bool:
    from atom.utils import envs

    return envs.ATOM_P1_MBF_ASM and envs.ATOM_P1_MOE_IDROWS


ASM_ROWS = (32, 128)


def tiles_for(n: int) -> dict:
    cfg = _tiles_for(n)
    if ASM_ROWS[0] <= n <= ASM_ROWS[1] and _asm():
        cfg = dict(cfg, g2_pf=3, g1_prio=(3, 0))
    return cfg


def _tiles_for(n: int) -> dict:
    """Per-row-count tiling (p1/small-g-moe b4): shared down tiles 128 columns + 32-row shared gate/up tiles to 64 rows,
    64-row gate/up tiles with prefetch 1 to 96, routed GEMM2 prefetch 3 above."""
    if n <= 16:  # ATOM_P1_MOE_IDROWS only (IDROWS_SMALL)
        return {"sh_pf": 3, "id_rows": True} if n == 8 else {"g1_pf": 6, "g2_tn": 128}
    if n <= 64:
        return {"sh_tn": 128, "sh_bm": 32}
    if n <= 96 or _idrows():
        return {"sh_tn": 128, "sh_bm": 64, "sh_pf": 1, "id_rows": True}
    return {"g2_pf": 3}


def supported(n: int) -> bool:
    return ROWS[0] <= n <= ROWS[1] or (n in IDROWS_SMALL and _idrows())


class _Plan:
    """``experts_a4w4_small.a4w4_experts``'s setup (id-rows host 16-96, sorted host 97-512) with the tiling above."""

    def __init__(self, n: int):
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4_small as XS
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sort_a16w4 import max_sorted_rows

        cfg = tiles_for(n)
        kw, bm = XS.small_config_for(n)
        assert kw == 4 and bm == 16 and (n > XS.INLINE_SORT_MAX or n in IDROWS_SMALL)
        self.n, self.bm = n, bm
        self.id_rows = XS.id_rows_for(n) or cfg.get("id_rows", False)
        if self.id_rows:
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm1_a4w4_id_rows import compile_gemm1_shared_first
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm2_a4w4_id_rows import compile_gemm2_shared_first
            if "g1_prio" in cfg:
                from .ak_moe_g1prio import compile_gemm1_shared_first as _g1prio

                def compile_gemm1_shared_first(**kw):
                    return _g1prio(prio=cfg["g1_prio"], **kw)
            self.rows = max_sorted_rows(n, 128, 4, bm)
        else:
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm1_a4w4_small import compile_gemm1_shared_first
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm2_a4w4_small import compile_gemm2_shared_first
            self.rows = max_sorted_rows(n, 128, 4, bm)
        sh_bm, sh_kw, sh_pf = XS.shared_fold_tile_for(n)
        sh_bm, sh_pf = cfg.get("sh_bm", sh_bm), cfg.get("sh_pf", sh_pf)
        self.nsh1 = (n + sh_bm - 1) // sh_bm * (24 // (4 // sh_kw))
        sh_tn = cfg.get("sh_tn", XS.shared_g2_columns_for(n))
        g2_bm = XS.shared_fold_down_rows_for(sh_bm, bm)
        self.nsh2 = (n + g2_bm - 1) // g2_bm * (H // sh_tn)
        g1_kw = {"sh_bm": sh_bm, "sh_kw": sh_kw, "sh_pf": sh_pf}
        if self.id_rows:
            g1_kw["pairs"] = XS.id_pairs_for(n)
        self.k1 = compile_gemm1_shared_first(kw=kw, pf=cfg.get("g1_pf", 3), bm=bm, **g1_kw)
        self.k2 = compile_gemm2_shared_first(tn=cfg.get("g2_tn", 256), bm=bm, pf=cfg.get("g2_pf", 2), sh_tn=sh_tn,
                                             sh_bm=g2_bm)


_plans: dict = {}


def mxfp4_moe_tiles(x: torch.Tensor, w) -> torch.Tensor:
    """``mxfp4_moe(x, w)`` (BF16 reduced entry) for 17..128 rows with ``tiles_for``'s tiling: same bits."""
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import chain
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.launch import _run_compiled
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.reduce_slots_shared import reduce_slots_shared
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sort_a16w4 import moe_sort_a16w4

    n, dev = x.shape[0], x.device
    p = _plans.get(n)
    if p is None:
        p = _plans[n] = _Plan(n)
    sw = w.shared_weights
    (xq, xs), (q8, s8), wt, ids = chain._route_quant_small(x, w)
    if p.id_rows:
        rows = p.rows
        eids = torch.empty(rows // 16 + 1, device=dev, dtype=torch.int32)
        nv = torch.empty(rows, device=dev, dtype=torch.int32)
        rargs, ids2, wt2 = (ids.view(-1), eids, nv), nv, wt.view(-1)
    else:
        sids, wt2, eids, nv = moe_sort_a16w4(ids, wt, 128, p.bm)
        rows = sids.numel()
        rargs, ids2 = (sids, eids, nv), sids
    q = torch.empty((rows, 384), device=dev, dtype=torch.uint8)
    s = torch.empty((rows, 24), device=dev, dtype=torch.uint8)
    out = torch.empty((n, 4, H), device=dev, dtype=torch.bfloat16)
    hq = torch.empty((n, 768), device=dev, dtype=torch.uint8)
    hs = torch.empty((n, 24), device=dev, dtype=torch.uint8)
    sh_out = torch.empty((n, H), device=dev, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(dev)
    _run_compiled(p.k1, xq.view(-1), xs.view(torch.uint8).view(-1), w.w13.view(-1), w.a4w4_scales.w13.view(-1),
                  q.view(-1), s.view(-1), *rargs, q8.view(-1), s8.view(-1), sw.w13.view(-1), sw.s13.view(-1),
                  hq.view(-1), hs.view(-1), n, rows, p.nsh1, stream)
    _run_compiled(p.k2, q.view(-1), s.view(-1), w.w2.view(-1), w.a4w4_scales.w2.view(-1), out.view(-1), ids2, wt2,
                  eids, nv, hq.view(-1), hs.view(-1), sw.w2.view(-1), sw.s2.view(-1), sh_out.view(-1),
                  n, rows, p.nsh2, stream)
    return reduce_slots_shared(out, sh_out)


def warm_up(w) -> None:
    """Compiles and loads every tiling variant before graph capture (one call per row count, zeros)."""
    sizes = list(range(ROWS[0], ROWS[1] + 1))
    if _idrows():
        sizes = list(IDROWS_SMALL) + sizes
    for n in sizes:
        mxfp4_moe_tiles(torch.zeros((n, H), device=w.router_weight.device, dtype=torch.bfloat16), w)
    torch.cuda.synchronize()
