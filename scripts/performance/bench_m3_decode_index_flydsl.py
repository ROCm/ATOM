# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Op bench: MiniMax-M3 decode index score + top-16 on one GPU (gfx1250, FP4 index).

  T    Triton `_decode_index_score_transposed_kernel` alone (today's scorer)
  F    aiter's FlyDSL scorer alone (ATOM_M3_DECODE_SCORE_FLYDSL)
  TT   today's step: Triton scorer + Triton selector (+ page-128 table emit)
  FT   FlyDSL scorer + Triton selector
  FA   FlyDSL scorer + aiter `topk_per_row_small_k` + the emit launch
       (ATOM_M3_AITER_TOPK_MIN_WIDTH=1)
  TA   Triton scorer + aiter selector + emit
       (FA / TA are dropped where aiter's selector declines the device: it has
       no wave32 build yet)
  Ff   aiter's FlyDSL scorer on its flat (length-proportional) plan alone
  Tk   the Triton selector + emit alone (on a scored plane)
  Sk   aiter's FlyDSL selector + emit alone (index_select_decode)
  TS   Triton scorer + FlyDSL selector + emit
  XX   the whole decode indexer through aiter (ATOM_M3_DECODE_INDEX_TOPK_FLYDSL:
       flat FlyDSL scorer + FlyDSL selector + emit)

Each variant is captured in a CUDA graph and replayed (--direct: launched
directly, events around each call); times are the median of --reps runs (us). Every replay scores another working set (--sets disjoint
page sets a request, rotated through the block table outside the timed region),
so the index pages come from HBM, not the L2. BW = live index bytes (9216 B a
block: e2m1 rows + E4M3 scales) / time, as % of 17.5 TB/s. Before timing, F's
live scores must equal T's bit for bit, FT's outputs TT's exactly and FA's / TA's
block sets TT's (--no-check skips). XX's ids /
table / lengths must equal TT's exactly, Ff's scores T's bit for bit, TS's
outputs TT's and Sk's Tk's.

  python scripts/performance/bench_m3_decode_index_flydsl.py                  # B 4/8/14/16 x 150K, 4 heads x 8 rows
  python scripts/performance/bench_m3_decode_index_flydsl.py --ctx 50000 300000
  python scripts/performance/bench_m3_decode_index_flydsl.py --varlen 0.66    # lengths ctx x U(1-0.66, 1+0.66)
  python scripts/performance/bench_m3_decode_index_flydsl.py --sweep          # F over ring x workgroups
"""

import argparse
import json
import os
import statistics

import torch

from atom.model_ops.minimax_m3 import index_topk as m
from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store

DEV, D, PAGE, HBM = "cuda", 128, 128, 17.5e12
BLOCK_BYTES = PAGE * (D // 2 + D // 16)
FLAG, WIDTH = "ATOM_M3_DECODE_SCORE_FLYDSL", "ATOM_M3_AITER_TOPK_MIN_WIDTH"
XFLAG = "ATOM_M3_DECODE_INDEX_TOPK_FLYDSL"


def make(lens, bound, mq, heads, sets, seed):
    """Index planes for ``sets`` disjoint page sets of every request, and one
    block table a set."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = [-(-n // PAGE) for n in lens]
    pages = sets * sum(nb) + 1
    payload = torch.empty(pages, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    scale = torch.empty(pages, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    step = 1 << 16  # quantize in slices: the bf16 keys of 8M tokens would not fit
    for s in range(0, pages * PAGE, step):
        n = min(step, pages * PAGE - s)
        keys = (torch.randn(n, D, device=DEV, generator=g) * 3).to(torch.bfloat16)
        index_fp4_store(keys, torch.arange(s, s + n, device=DEV), payload, scale)
    perm = torch.randperm(pages, device=DEV, generator=g).to(torch.int32)
    tables = torch.zeros(
        sets, len(lens), -(-bound // PAGE), dtype=torch.int32, device=DEV
    )
    off = 0
    for s in range(sets):
        for b, n in enumerate(nb):
            tables[s, b, :n] = perm[off : off + n]
            off += n
    q = torch.randn(len(lens) * mq, heads, D, device=DEV, generator=g).to(
        torch.bfloat16
    )
    seq = torch.tensor(lens, dtype=torch.int32, device=DEV)
    return q, payload, scale, tables, seq


def timed_direct(fn, reps, rotate):
    rotate(0)
    fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(reps):
        rotate(i + 1)
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def timed(fn, reps, rotate):
    rotate(0)
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for i in range(reps):
        rotate(i + 1)
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def bench(a, ctx, bs, rows):
    """One (context, batch) cell: check, then time every variant."""
    from aiter.ops.flydsl.index_score_decode import (
        flat_programs,
        index_score_decode_nvfp4,
        index_select_decode,
        index_topk_decode_score_plane,
    )

    sm = a.sm
    bound = max(a.bound, ctx)
    blocks = -(-bound // PAGE)
    g = torch.Generator().manual_seed(bs)
    if a.varlen:
        u = torch.rand(bs, generator=g) * 2 - 1
        lens = [max(a.mq, min(bound, int(ctx * (1 + a.varlen * float(x))))) for x in u]
    else:
        lens = [ctx] * bs
    live = sum(min(-(-n // PAGE), blocks) for n in lens)
    nbytes = live * BLOCK_BYTES
    sets = a.sets or max(2, -(-400_000_000 // nbytes))
    q, payload, scale, tables, seq = make(lens, bound, a.mq, a.heads, sets, bs)
    bt = tables[0].clone()

    def rotate(i, bt=bt, tables=tables, sets=sets):
        bt.copy_(tables[i % sets])

    # rows padded as aiter's selector reads them (the Triton kernels take strides)
    score = index_topk_decode_score_plane(a.heads, bs * a.mq, blocks, DEV)
    bounds = m.build_n_valid_column_per_row(
        seq,
        seq,
        batch=bs,
        total_q=bs * a.mq,
        num_idx_heads=a.heads,
        decode_max_q=a.mq,
    )

    def triton_score(q=q, payload=payload, scale=scale, bt=bt, seq=seq):
        chunks = m._native_decode_score_chunks(len(seq), blocks)
        m._decode_index_score_transposed_kernel[(len(seq), chunks)](
            q, payload, scale, score, bt, seq, D, sm, -(-blocks // chunks),
            *q.stride(), *payload.stride(), scale.stride(0), scale.stride(1),
            *score.stride(), bt.stride(0), NUM_IDX_HEADS=a.heads, MAX_Q=a.mq,
            BLOCK_SIZE_K=PAGE, IS_FP4=True, FP4_SCALED_DOT=True,
            num_warps=m.DECODE_SCORE_FP4_NUM_WARPS,
            num_stages=m.DECODE_SCORE_FP4_NUM_STAGES,
        )  # fmt: skip
        return score

    def flydsl_score(workgroups=None, ring=4, q=q, payload=payload,
                     scale=scale, bt=bt, seq=seq):  # fmt: skip
        index_score_decode_nvfp4(
            q, payload, scale, score, bt, seq, sm, a.mq,
            workgroups=workgroups, ring=ring,
        )  # fmt: skip
        return score

    def flat_score(workgroups=None, ring=a.ring, q=q, payload=payload,
                   scale=scale, bt=bt, seq=seq):  # fmt: skip
        index_score_decode_nvfp4(
            q, payload, scale, score, bt, seq, sm, a.mq,
            workgroups=workgroups, ring=ring, flat=True,
        )  # fmt: skip
        return score

    total_q = bs * a.mq
    topk_idx = torch.empty(a.heads, total_q, 16, dtype=torch.int32, device=DEV)
    sbt = torch.empty(total_q * a.heads, 16, dtype=torch.int32, device=DEV)
    sctx = torch.empty(total_q * a.heads, dtype=torch.int32, device=DEV)
    sel_score = index_topk_decode_score_plane(a.heads, bs * a.mq, blocks, DEV)

    def triton_select(seq=seq, bt=bt):
        m._launch_select(
            sel_score, topk_idx, seq, seq, len(seq), 16, a.init, a.keep, bt,
            (sbt, sctx, bt.stride(0), sbt.stride(0)), a.heads,
            rows_per_req=a.mq, decode_max_q=a.mq,
            block_size_k=m._decode_topk_block_size_k("gfx1250", blocks),
            num_warps=m.DECODE_TOPK_NUM_WARPS, emit=True,
            block_page_stride=m.NVFP4_PAGES_PER_BLOCK,
            pages_per_block=m.NVFP4_PAGES_PER_BLOCK,
        )  # fmt: skip
        return topk_idx, sbt, sctx

    def flydsl_select(plane=sel_score, seq=seq, bt=bt):
        index_select_decode(
            plane, seq, topk_idx, a.mq, a.init, a.keep, block_table=bt,
            sparse_bt=sbt, sparse_ctx=sctx,
        )  # fmt: skip
        return topk_idx, sbt, sctx

    def triton_then_flydsl_select():
        triton_score()
        return flydsl_select(plane=score)

    def step(flydsl, aiter_select, q=q, payload=payload, scale=scale, bt=bt,
             seq=seq, bounds=bounds, whole=False):  # fmt: skip
        os.environ[XFLAG] = "1" if whole else "0"
        os.environ[FLAG] = "1" if flydsl else "0"
        os.environ[WIDTH] = "1" if aiter_select else ""
        return m.minimax_m3_index_topk_decode(
            q, payload, bt, seq, bound, 16, a.init, a.keep, a.heads, sm,
            emit_sparse_block_table=True, max_query_len=a.mq,
            block_page_stride=m.NVFP4_PAGES_PER_BLOCK,
            n_valid_column_per_row=bounds, index_kv_scale=scale,
            emit_pages_per_block=m.NVFP4_PAGES_PER_BLOCK,
        )  # fmt: skip

    # Does aiter's selector serve this device and shape? (ATOM falls back to the
    # Triton selector where it declines.)
    served = []
    select = m._select_with_aiter
    m._select_with_aiter = lambda *x: served.append(select(*x)) or served[-1]
    rotate(0)
    step(False, True)
    m._select_with_aiter = select
    aiter_selects = served == [True]
    if not aiter_selects:
        print("aiter's selector declines this device: FA / TA dropped")

    ref = mask = None
    if not a.no_check:
        rotate(0)
        mask = (
            torch.arange(blocks, device=DEV)[None]
            < ((seq.long() + PAGE - 1) // PAGE).clamp(max=blocks)[:, None]
        )
        mask = mask.repeat_interleave(a.mq, 0)[None].expand(a.heads, -1, -1)
        ref = triton_score().clone()
        got = flydsl_score()
        torch.cuda.synchronize()
        bad = ref.view(torch.int32)[mask] != got.view(torch.int32)[mask]
        assert not bad.any(), f"score errRatio {bad.float().mean().item():.3e}"
        got = flat_score()
        torch.cuda.synchronize()
        bad = ref.view(torch.int32)[mask] != got.view(torch.int32)[mask]
        assert not bad.any(), f"flat score errRatio {bad.float().mean().item():.3e}"
        tt = [t.clone() for t in step(False, False)]
        for name, x, y in zip(("topk", "bt", "ctx"), tt, step(True, False)):
            assert torch.equal(x, y), f"FT {name}"
        for name, x, y in zip(
            ("topk", "bt", "ctx"), tt, step(False, False, whole=True)
        ):
            assert torch.equal(x, y), f"XX {name}"
        for name, x, y in zip(("topk", "bt", "ctx"), tt, triton_then_flydsl_select()):
            assert torch.equal(x, y), f"TS {name}"
        sel_score.copy_(ref)
        tk = [t.clone() for t in triton_select()]
        for name, x, y in zip(("topk", "bt", "ctx"), tk, flydsl_select()):
            assert torch.equal(x, y), f"Sk {name}"
        for tag, fly in (("FA", True), ("TA", False)) if aiter_selects else ():
            out = step(fly, True)
            for name, x, y in zip(("topk", "bt"), tt, out):
                n = x.shape[-1]
                assert torch.equal(
                    x.reshape(-1, n).sort(1).values,
                    y.reshape(-1, n).sort(1).values,
                ), f"{tag} {name}"
            assert torch.equal(tt[2], out[2]), f"{tag} ctx"

    rotate(0)
    sel_score.copy_(triton_score())  # Tk / Sk select from a scored plane
    variants = [
        ("T  triton score", triton_score, True),
        ("F  flydsl score", flydsl_score, True),
        ("Ff flydsl flat score", flat_score, True),
        ("Tk triton select+emit", triton_select, False),
        ("Sk flydsl select+emit", flydsl_select, False),
        ("TT triton + triton", lambda: step(False, False), False),
        ("FT flydsl + triton", lambda: step(True, False), False),
        ("TS triton + flydsl sel", triton_then_flydsl_select, False),
        ("XX flydsl indexer", lambda: step(False, False, whole=True), False),
    ]
    if aiter_selects:
        variants += [
            ("FA flydsl + aiter + emit", lambda: step(True, True), False),
            ("TA triton + aiter + emit", lambda: step(False, True), False),
        ]
    if a.sweep:
        variants = [
            (
                f"Ff ring {ring} grid {flat_programs(bs, blocks, wg)}",
                lambda ring=ring, wg=wg: flat_score(wg, ring),
                True,
            )
            for ring in (4, 6, 8)
            for wg in (512, 768, 1024, 1536, 2048)
        ]
    for name, fn, is_score in variants:
        if a.sweep and ref is not None:  # every ring / budget: the same bits
            rotate(0)
            got = fn()
            torch.cuda.synchronize()
            bad = ref.view(torch.int32)[mask] != got.view(torch.int32)[mask]
            assert not bad.any(), f"{name}: errRatio {bad.float().mean().item():.3e}"
        us = (timed_direct if a.direct else timed)(fn, a.reps, rotate)
        bw = 100 * nbytes / (us * 1e-6) / HBM
        mode = "direct" if a.direct else "graph"
        print(f"{bs:>4} {ctx:>7} {mode:<6} {name:<26} {us:8.1f} {bw:6.1f}")
        rows.append(
            {"bs": bs, "ctx": ctx, "varlen": a.varlen, "mode": mode,
             "variant": name, "us": us, "bw_pct": bw}
        )  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[4, 8, 14, 16])
    ap.add_argument("--ctx", type=int, nargs="+", default=[150_000])
    ap.add_argument(
        "--varlen", type=float, default=0.0, help="lengths ctx x U(1-v, 1+v)"
    )
    ap.add_argument(
        "--bound", type=int, default=350_000, help="max context of the graph"
    )
    ap.add_argument("--mq", type=int, default=8, help="query rows a request")
    ap.add_argument("--heads", type=int, default=4, help="index heads (TP1: 4)")
    ap.add_argument(
        "--sm", type=float, default=D**-0.5, help="sm_scale (the model's: 128**-0.5)"
    )
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--keep", type=int, default=1)
    ap.add_argument("--sets", type=int, default=0, help="0: enough for 400 MB")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--direct", action="store_true", help="no graph: direct launches")
    ap.add_argument("--ring", type=int, default=4, help="the flat scorer's ring")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    print(f"{'bs':>4} {'ctx':>7} {'mode':<6} {'variant':<26} {'us':>8} {'BW%':>6}")
    for ctx in a.ctx:
        for bs in a.bs:
            bench(a, ctx, bs, rows)
            torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
