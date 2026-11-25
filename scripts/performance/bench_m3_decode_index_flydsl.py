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

Each variant is captured in a CUDA graph and replayed; times are the median of
--reps replays (us). Every replay scores another working set (--sets disjoint
page sets a request, rotated through the block table outside the timed region),
so the index pages come from HBM, not the L2. BW = live index bytes (9216 B a
block: e2m1 rows + E4M3 scales) / time, as % of 17.5 TB/s. Before timing, F's
live scores must equal T's bit for bit, FT's outputs TT's exactly and FA's / TA's
block sets TT's (--no-check skips).

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
    from aiter.ops.flydsl.index_score_decode import index_score_decode_nvfp4

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

    score = torch.empty(a.heads, bs * a.mq, blocks, device=DEV)
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

    def step(flydsl, aiter_select, q=q, payload=payload, scale=scale, bt=bt,
             seq=seq, bounds=bounds):  # fmt: skip
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
        tt = [t.clone() for t in step(False, False)]
        for name, x, y in zip(("topk", "bt", "ctx"), tt, step(True, False)):
            assert torch.equal(x, y), f"FT {name}"
        for tag, fly in (("FA", True), ("TA", False)) if aiter_selects else ():
            out = step(fly, True)
            for name, x, y in zip(("topk", "bt"), tt, out):
                n = x.shape[-1]
                assert torch.equal(
                    x.reshape(-1, n).sort(1).values,
                    y.reshape(-1, n).sort(1).values,
                ), f"{tag} {name}"
            assert torch.equal(tt[2], out[2]), f"{tag} ctx"

    variants = [
        ("T  triton score", triton_score, True),
        ("F  flydsl score", flydsl_score, True),
        ("TT triton + triton", lambda: step(False, False), False),
        ("FT flydsl + triton", lambda: step(True, False), False),
    ]
    if aiter_selects:
        variants += [
            ("FA flydsl + aiter + emit", lambda: step(True, True), False),
            ("TA triton + aiter + emit", lambda: step(False, True), False),
        ]
    if a.sweep:
        variants = [
            (
                f"F  ring {ring} wg {wg}",
                lambda ring=ring, wg=wg: flydsl_score(wg, ring),
                True,
            )
            for ring in (3, 4, 5, 6)
            for wg in (64, 256, 512, 768, 1024, 1536)
        ]
    for name, fn, is_score in variants:
        if a.sweep and ref is not None:  # every ring / budget: the same bits
            rotate(0)
            got = fn()
            torch.cuda.synchronize()
            bad = ref.view(torch.int32)[mask] != got.view(torch.int32)[mask]
            assert not bad.any(), f"{name}: errRatio {bad.float().mean().item():.3e}"
        us = timed(fn, a.reps, rotate)
        bw = 100 * nbytes / (us * 1e-6) / HBM
        print(f"{bs:>4} {ctx:>7} {name:<26} {us:8.1f} {bw:6.1f}")
        rows.append({"bs": bs, "ctx": ctx, "variant": name, "us": us, "bw_pct": bw})


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
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    print(f"{'bs':>4} {'ctx':>7} {'variant':<26} {'us':>8} {'BW%':>6}")
    for ctx in a.ctx:
        for bs in a.bs:
            bench(a, ctx, bs, rows)
            torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
