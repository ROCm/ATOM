"""Check wide verify row descriptors and packed piece prefixes on the GPU."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import Int32, Int64

from atom.mono.device.ops import rsrc
from atom.mono.dsa_indexer import kernel as kix
from atom.mono.dsa_indexer import select
from atom.mono.dsa_indexer.plan import IndexerShape, local_len


def planner(key):
    s, ql = key.tokens, key.qlen
    pieces = s // ql
    words = 3 * s + 4 + pieces + s
    keyed = (s, ql, key.replicated, key.world, key.max_ctx, kix.SOURCES)

    @fx.struct
    class Smem:
        rows: fx.Array[fx.Int32, 3 * s, 16]
        plan: fx.Array[fx.Int32, 4 + 2 * s, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel(ctx: Int64, bids: Int64, out: Int64, rank: Int32):
        _ = keyed
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        select.step_table({
            "S": s, "shape": key.shape, "tid": tid, "lane": tid % 64,
            "rank": rank, "world": key.world, "inter": key.interleave,
            "own_rows": key.own_rows, "short_owned_verify": key.short_owned_verify,
            "local_short": key.short_owned_verify, "short_score": False,
            "max_ctx": key.max_ctx, "qlen": ql, "rg": ql, "ppr": 1,
            "args": {"ctx": ctx, "bids": bids},
            "rows": lds.rows.ptr, "plan": lds.plan.ptr,
        })
        for i in range(tid, 3 * s, 512):
            bo.buffer_store(fx.ptr_load(lds.rows.ptr + i), rsrc(out), i)
        for i in range(tid, words - 3 * s, 512):
            bo.buffer_store(fx.ptr_load(lds.plan.ptr + i), rsrc(out), 3 * s + i)

    @flyc.jit
    def launch(ctx: Int64, bids: Int64, out: Int64, rank: Int32):
        _ = keyed
        kernel(ctx, bids, out, rank).launch(grid=(1,), block=(512,))

    return launch, words


def expected(key, lengths, rank):
    short = key.short_owned_verify and max(lengths) <= 8192
    rows = []
    for r in range(key.tokens):
        req, draft = divmod(r, key.qlen)
        context = lengths[req] - (key.qlen - 1 - draft)
        split = not key.own_rows
        if key.own_rows:
            split = lengths[req] > 262144 and 2 * key.world * lengths[req] > sum(lengths)
        count = local_len(max(context, 0), rank, key.world, key.interleave) if split else (
            max(context, 0) if req % key.world == rank else 0)
        if short:
            count, split = max(context, 0), False
        long = context > 2048 and count > 0
        flags = int(long) | (2 * int(split) if key.own_rows else 0) | 4 * int(short)
        rows.append((context, count, flags))
    units = [(n + 127) // 128 if flags & 1 else 0 for _, n, flags in rows[key.qlen - 1::key.qlen]]
    lg = next((w for w in range(3) if sum((u + (1 << w) - 1) >> w for u in units) <= 256), 3)
    scores = [(u + (1 << lg) - 1) >> lg for u in units]
    chunks = [(n + 4095) // 4096 if flags & 1 else 0 for _, n, flags in rows]
    room = 256 - sum(bool(flags & 1) for _, _, flags in rows)
    slg = 0
    if sum(chunks) > 256:
        for w in range(1, 11):
            if sum(chunks) > room << (w - 1):
                slg = w
    segments = [(ch + (1 << slg) - 1) >> slg for ch in chunks]
    assert sum(segments) <= 256
    plan = [sum(scores), lg, sum(segments), slg]
    for counts in (scores, segments):
        prefix = 0
        for count in counts:
            plan.append(prefix)
            prefix += count
    return [v for row in rows for v in row] + plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("planner_result.json"))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    shape = IndexerShape(32, 128, 2048, 2048, "bf16", "fp8", 128, 64)
    base = kix.IxBuild(shape, 6, 8, 163840, qlen=6)
    for replica in (False, True):
        for rows in range(6, 253, 6):
            assert not kix.unfit(replace(base, tokens=rows, replicated=replica))
        assert kix.unfit(replace(base, tokens=256, replicated=replica))
        assert kix.unfit(replace(base, tokens=258, replicated=replica))
    assert kix.unfit(replace(base, tokens=72, qlen=8))
    checked = 0
    for replica in (False, True):
        for batch, qlen in ((11, 6), (16, 6), (22, 6), (32, 6), (40, 6), (42, 6), (256, 1)):
            key = replace(base, tokens=batch * qlen, qlen=qlen, replicated=replica)
            launch, words = planner(key)
            ctx = torch.empty(batch, device="cuda", dtype=torch.int32)
            bids = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(qlen)
            out = torch.empty(words, device="cuda", dtype=torch.int32)
            patterns = [[length] * batch for length in (0, 2048, 8192, 8193, 163840)]
            patterns += [[8192] * (batch - 1) + [163840],
                         [[0, 8192, 8193, 163840][r % 4] for r in range(batch)]]
            for lengths in patterns:
                ctx.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
                for rank in (0, 3, 7):
                    launch(ctx.data_ptr(), bids.data_ptr(), out.data_ptr(), rank)
                    got = out.cpu().tolist()
                    want = expected(key, lengths, rank)
                    assert got == want, (replica, batch, qlen, rank, lengths,
                        [(i, a, b) for i, (a, b) in enumerate(zip(got, want)) if a != b][:12])
                    checked += 1
            print(f"PLANNER_PASS replica={replica} batch={batch} qlen={qlen}", flush=True)
    result = {"planner_scenarios": checked, "admission_checks": True, "max_verify_rows": 252}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PLANNER_DONE", result, flush=True)


if __name__ == "__main__":
    main()
