# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3's local full-table embedding (``step_gap.embed_local``) against the
vocabulary-sharded embedding and its TP all-reduce, on one GPU, bit for bit.

Production cannot run here (its all-reduce needs eight ranks), so this process
plays them: the table is cut into TP shards, every rank's lookup is production's
own ``masked_embedding`` on its shard, and the eight rows are added in rank
order in fp32 by a HIP kernel whose float mode is the all-reduce kernels'
(``add_rmsnorm_quant``'s residual add: FLOAT_DENORM_MODE_32 0, the same kernel
descriptor flags; a partial sum of one real row and zeros is exact in BF16, so
rounding after each add changes nothing).

1. Every BF16 bit pattern x: ``canonicalize_``(x) against that kernel's
   x + 0.0 and 0.0 + x (all 65536).
2. ``bind`` as the installer runs it, the collectives replaced by this
   process's shards: the table every rank would hold, against
   ``canonicalize_`` of the whole table; its refusals.
3. The whole vocabulary: ``K3LocalEmbedding.forward`` on the bound table
   against the eight ranks' masked lookups summed in rank order, every id, and
   batches with repeated and out-of-range ids; then CUDA-graph replays.
4. The draft block, NOT wired into the model (an existing test pins the
   standalone drafts to ``DSparkDraftModel.block_backbone``): a tool-only
   kernel that embeds ``[anchor, MASK, ...]`` in one launch from the anchors,
   against the lookup of the id buffer production builds and against the
   emulated ranks; eager and CUDA-graph replays.
5. CUDA-graph time of the lookup: the shard's masked lookup (production's
   first launch; its all-reduce is not run here) against the local table's;
   and the draft block's fill + copy + lookup against that kernel.

The table is synthetic (no checkpoint on the node): normal values, and rows
and scattered elements of -0.0, +0.0, denormals of both signs, NaNs (quiet and
signalling), infinities and the largest and smallest normals.

    python tools/kimi_k3_step_gap_embed_check.py
    env: VOCAB (163840), TP (8), RANK (5), CHUNK (4096), REPLAYS (20), BENCH (1)
"""

import json
import os
import statistics
import time
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from aiter import QuantType, dtypes
from torch import nn

from atom.model_ops.embed_head import masked_embedding
from atom.model_ops.layernorm import _aiter_rms_quant
from atom.models.kimi_k3.step_gap import embed_local as E

VOCAB = int(os.environ.get("VOCAB", "163840"))
TP = int(os.environ.get("TP", "8"))
RANK = int(os.environ.get("RANK", "5"))
CHUNK = int(os.environ.get("CHUNK", "4096"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
BENCH = os.environ.get("BENCH", "1") == "1"
HIDDEN = 7168
SHARD = VOCAB // TP
GN = 16
KEEP = []
SPECIAL = (
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x007F,
    0x807F,
    0x0040,
    0x0080,
    0x8080,
    0x7F7F,
    0xFF7F,
    0x7F80,
    0xFF80,
    0x7FC0,
    0xFFC0,
    0x7FA0,
    0xFF81,
    0x7FFF,
    0x3F80,
    0xBF80,
)

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(41)
ONES = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)


def i16(values):
    return torch.tensor(
        [v - 65536 if v >= 32768 else v for v in values], dtype=torch.int16, device=dev
    )


def hip_add(a, b):
    """a + b in fp32, rounded to BF16, by a HIP kernel in the all-reduce
    kernels' float mode (rows of HIDDEN)."""
    return _aiter_rms_quant(
        a, ONES, 1e-6, QuantType.per_1x128.value, True, b, dtypes.fp8, True
    )[2]


def patterns():
    """Every BF16 bit pattern, as rows of HIDDEN (the tail repeats +1.0)."""
    rows = -(-65536 // HIDDEN)
    raw = torch.full((rows * HIDDEN,), 0x3F80, dtype=torch.int32, device=dev)
    raw[:65536] = torch.arange(65536, device=dev, dtype=torch.int32)
    return raw.to(torch.int16).view(rows, HIDDEN).view(torch.bfloat16).clone()


def every_pattern():
    x = patterns()
    zero = torch.zeros_like(x)
    want_a = hip_add(x, zero).view(torch.int16).reshape(-1)[:65536]
    want_b = hip_add(zero, x).view(torch.int16).reshape(-1)[:65536]
    got = E.canonicalize_(x.clone()).view(torch.int16).reshape(-1)[:65536]
    torch.cuda.synchronize()
    src = x.view(torch.int16).reshape(-1)[:65536]
    bad_a, bad_b = (
        (got != want_a).nonzero().flatten(),
        (got != want_b).nonzero().flatten(),
    )
    changed = (got != src).sum()
    show = [
        {
            "x": hex(int(src[i]) & 0xFFFF),
            "canonical": hex(int(got[i]) & 0xFFFF),
            "x_plus_0": hex(int(want_a[i]) & 0xFFFF),
            "0_plus_x": hex(int(want_b[i]) & 0xFFFF),
        }
        for i in torch.cat([bad_a[:6], bad_b[:6]]).unique().tolist()
    ]
    res = {
        "check": "every BF16 pattern", "patterns": 65536, "changed_by_canonicalize": int(changed),
        "differ_from_x_plus_0": int(bad_a.numel()), "differ_from_0_plus_x": int(bad_b.numel()), "first": show,
        "the_two_orders_agree": bool(torch.equal(want_a, want_b)),
    }  # fmt: skip
    print(json.dumps(res), flush=True)
    return bad_a.numel() == 0 and bad_b.numel() == 0


def synthetic_table():
    table = torch.empty(VOCAB, HIDDEN, dtype=torch.bfloat16, device=dev)
    for start in range(0, VOCAB, 8192):
        n = min(8192, VOCAB - start)
        table[start : start + n] = (
            torch.randn(n, HIDDEN, generator=gen, device=dev) * 0.02
        ).to(torch.bfloat16)
    raw = table.view(torch.int16)
    special = i16(SPECIAL)
    # whole rows of one special value, in every shard
    for r in range(TP):
        for k, value in enumerate(special):
            raw[r * SHARD + 3 + 7 * k] = value
        # rows of all the special values in turn
        raw[r * SHARD + SHARD - 2] = special[
            torch.arange(HIDDEN, device=dev) % len(special)
        ]
    # and scattered elements
    for _ in range(64):
        rows = torch.randint(0, VOCAB, (4096,), generator=gen, device=dev)
        cols = torch.randint(0, HIDDEN, (4096,), generator=gen, device=dev)
        raw[rows, cols] = special[
            torch.randint(0, len(special), (4096,), generator=gen, device=dev)
        ]
    return table


def embedding(shard, rank):
    """A ``K3LocalEmbedding`` as its constructor and loader leave rank
    ``rank``'s, without the TP group."""
    e = E.K3LocalEmbedding.__new__(E.K3LocalEmbedding)
    nn.Module.__init__(e)
    e.prefix = ""
    e.tp_rank, e.tp_size = rank, TP
    e.num_embeddings, e.num_embeddings_per_partition = VOCAB, SHARD
    e.vocab_start_idx, e.vocab_end_idx = rank * SHARD, (rank + 1) * SHARD
    e.weight = nn.Parameter(shard, requires_grad=False)
    e.local_table = None
    return e


def production(ids, shards):
    """The eight ranks' masked lookups, added in rank order."""
    acc = None
    for r, shard in enumerate(shards):
        y = masked_embedding(ids, shard, r * SHARD, (r + 1) * SHARD)
        acc = y if acc is None else hip_add(acc, y)
    return acc


def bound(table, shards):
    """``bind`` with the collectives played here: the gather returns the
    shards, the agreement is this rank's own."""
    emb = embedding(shards[RANK], RANK)
    real = dist.all_gather_into_tensor
    calls = []

    def gather(out, piece, group=None):
        start = (piece.data_ptr() - emb.weight.data_ptr()) // (HIDDEN * 2)
        calls.append((start, piece.shape[0]))
        for r in range(TP):
            out[r].copy_(shards[r][start : start + piece.shape[0]])

    dist.all_gather_into_tensor = gather
    try:
        torch.cuda.synchronize()
        held, t0 = torch.cuda.memory_allocated(), time.perf_counter()
        ok = E.bind(emb, NS(cpu_group=None, device_group=None))
        torch.cuda.synchronize()
        bind_stats = {
            "bind_seconds_gather_played_locally": round(time.perf_counter() - t0, 2),
            "bound_table_mib": (torch.cuda.memory_allocated() - held) >> 20,
        }
    finally:
        dist.all_gather_into_tensor = real
    want = E.canonicalize_(table.clone())
    covered = sorted(calls) == [
        (s, min(E.GATHER_ROWS, SHARD - s)) for s in range(0, SHARD, E.GATHER_ROWS)
    ]
    res = {
        "check": "bind", "bound": ok, "gathers": len(calls), "every_row_gathered_once": covered,
        "table_is_canonical_whole_table": ok and bool(torch.equal(emb.local_table.view(torch.int16), want.view(torch.int16))),
        "rows": list(emb.local_table.shape) if ok else None,
        "changed_elements": int((want.view(torch.int16) != table.view(torch.int16)).sum()),
        "checksum_differs_from_raw": E.checksum(want) != E.checksum(table),
    }  # fmt: skip

    # a gather that returns another rank's rows in this rank's place: refused
    other = embedding(shards[RANK], RANK)

    def wrong(out, piece, group=None):
        start = (piece.data_ptr() - other.weight.data_ptr()) // (HIDDEN * 2)
        for r in range(TP):
            out[r].copy_(shards[(r + 1) % TP][start : start + piece.shape[0]])

    dist.all_gather_into_tensor = wrong
    try:
        res["a_wrong_gather_is_refused"] = (
            E.bind(other, NS(cpu_group=None, device_group=None)) is False
            and other.local_table is None
        )
    finally:
        dist.all_gather_into_tensor = real
    f32 = embedding(shards[RANK].float(), RANK)
    res["a_float_shard_is_refused"] = (
        E.bind(f32, NS(cpu_group=None, device_group=None)) is False
        and f32.local_table is None
    )
    res.update(bind_stats)
    print(json.dumps(res), flush=True)
    good = (
        ok
        and covered
        and res["table_is_canonical_whole_table"]
        and res["a_wrong_gather_is_refused"]
        and res["a_float_shard_is_refused"]
    )
    return emb, good


def whole_vocabulary(emb, shards):
    bad_rows = bad_values = 0
    for start in range(0, VOCAB, CHUNK):
        ids = torch.arange(
            start, min(VOCAB, start + CHUNK), dtype=torch.int32, device=dev
        )
        want, got = production(ids, shards), emb(ids)
        diff = want.view(torch.int16) != got.view(torch.int16)
        bad_values += int(diff.sum())
        bad_rows += int(diff.any(dim=1).sum())
    odd = torch.randint(0, VOCAB, (CHUNK,), generator=gen, device=dev).to(torch.int32)
    odd[::7] = -1
    odd[3::11] = VOCAB
    odd[5::13] = VOCAB + 5
    odd[1::2] = odd[0::2]  # repeats
    want, got = production(odd, shards), emb(odd)
    outside = (odd < 0) | (odd >= VOCAB)
    res = {
        "check": "whole vocabulary", "ids": VOCAB, "mismatched_rows": bad_rows, "mismatched_values": bad_values,
        "mixed_batch_equal": bool(torch.equal(want.view(torch.int16), got.view(torch.int16))),
        "out_of_range_rows_are_plus_zero": bool((got.view(torch.int16)[outside] == 0).all()),
        "out_of_range_rows": int(outside.sum()),
    }  # fmt: skip
    # the lookup inside a CUDA graph, its ids rewritten between replays
    ids = torch.zeros(64, dtype=torch.int32, device=dev)
    graph = torch.cuda.CUDAGraph()
    emb(ids)
    with torch.cuda.graph(graph):
        out = emb(ids)
    KEEP.append((graph, ids, out))
    replays_bad = 0
    for _ in range(REPLAYS):
        ids.copy_(
            torch.randint(-1, VOCAB + 1, (64,), generator=gen, device=dev).to(
                torch.int32
            )
        )
        graph.replay()
        replays_bad += not torch.equal(
            out.view(torch.int16), production(ids, shards).view(torch.int16)
        )
    res["replays_bad"], res["replays"] = int(replays_bad), REPLAYS
    print(json.dumps(res), flush=True)
    return (
        bad_values == 0
        and res["mixed_batch_equal"]
        and res["out_of_range_rows_are_plus_zero"]
        and replays_bad == 0
    )


NOISE = 163838 if VOCAB > 163838 else VOCAB - 2


def block_ids(anchors, drafts):
    """The id buffer ``DSparkDraftModel.block_backbone`` builds."""
    ids = anchors.new_full((anchors.shape[0], drafts), NOISE)
    ids[:, 0] = anchors
    return ids.view(-1)


@triton.jit
def _block_embedding_kernel(
    anchors_ptr,  # [requests] int32
    weight_ptr,  # [vocab, D]: the whole table
    out_ptr,  # [requests * T, D]
    noise_id,
    vocab,
    T,
    stride_w_row,
    stride_out_row,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    """``_masked_embedding_kernel`` over the full id range, for row
    ``request * T + k`` of a draft block: the request's anchor at k == 0,
    the mask id elsewhere."""
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    if pid_row >= N:
        return

    anchor = tl.load(anchors_ptr + pid_row // T)
    token_id = tl.where(pid_row % T == 0, anchor, noise_id)
    in_range = (token_id >= 0) & (token_id < vocab)

    col_start = pid_col * BLOCK_D
    cols = col_start + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    emb = tl.load(
        weight_ptr + token_id.to(tl.int64) * stride_w_row + cols,
        mask=in_range & col_mask,
        other=0.0,
    )

    tl.store(out_ptr + pid_row * stride_out_row + cols, emb, mask=col_mask)


def embed_block(
    anchors: torch.Tensor, noise_id: int, drafts: int, table: torch.Tensor, vocab: int
) -> torch.Tensor:
    """The embedding of ``[anchor, noise_id, ..., noise_id]`` (``drafts``
    wide) a request, ``[requests * drafts, D]``: what the masked lookup
    returns for that id buffer, without building it."""
    rows = anchors.numel() * drafts
    dim = table.shape[1]
    out = torch.empty(rows, dim, dtype=table.dtype, device=table.device)
    grid = (rows, triton.cdiv(dim, 1024))
    _block_embedding_kernel[grid](
        anchors,
        table,
        out,
        noise_id,
        vocab,
        drafts,
        table.stride(0),
        out.stride(0),
        rows,
        dim,
        BLOCK_D=1024,
    )
    return out


def draft_block(emb, shards):
    bad = cases = 0
    for requests, drafts in ((1, 7), (3, 7), (8, 7), (16, 4), (8, 4), (5, 1)):
        for _ in range(8):
            anchors = torch.randint(
                0, VOCAB, (requests,), generator=gen, device=dev
            ).to(torch.int32)
            if requests > 2:
                anchors[1] = -1
                anchors[2] = VOCAB
            ids = block_ids(anchors, drafts)
            got = embed_block(anchors, NOISE, drafts, emb.local_table, VOCAB)
            cases += 1
            bad += not (
                torch.equal(got.view(torch.int16), emb(ids).view(torch.int16))
                and torch.equal(
                    got.view(torch.int16), production(ids, shards).view(torch.int16)
                )
                and tuple(got.shape) == (requests * drafts, HIDDEN)
            )
    anchors = torch.zeros(8, dtype=torch.int32, device=dev)
    graph = torch.cuda.CUDAGraph()
    embed_block(anchors, NOISE, 7, emb.local_table, VOCAB)
    with torch.cuda.graph(graph):
        out = embed_block(anchors, NOISE, 7, emb.local_table, VOCAB)
    KEEP.append((graph, anchors, out))
    replays_bad = 0
    for _ in range(REPLAYS):
        anchors.copy_(
            torch.randint(-1, VOCAB + 1, (8,), generator=gen, device=dev).to(
                torch.int32
            )
        )
        graph.replay()
        replays_bad += not torch.equal(
            out.view(torch.int16),
            production(block_ids(anchors, 7), shards).view(torch.int16),
        )
    res = {
        "check": "draft block (tool-only kernel)",
        "cases": cases,
        "mismatched": int(bad),
        "replays_bad": int(replays_bad),
        "replays": REPLAYS,
    }
    print(json.dumps(res), flush=True)
    return bad == 0 and replays_bad == 0


def graph_of(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    KEEP.append((g, fn))
    return g


def timed(g):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def bench(emb, shards):
    for rows in (8, 24, 56, 64):
        sets = [
            torch.randint(0, VOCAB, (rows,), generator=gen, device=dev).to(torch.int32)
            for _ in range(GN)
        ]
        KEEP.append(sets)

        def shard_lookup(sets=sets):
            for ids in sets:
                KEEP.append(
                    masked_embedding(
                        ids, shards[RANK], RANK * SHARD, (RANK + 1) * SHARD
                    )
                )

        def local_lookup(sets=sets):
            for ids in sets:
                KEEP.append(emb(ids))

        graphs = {
            "shard_masked_lookup": graph_of(shard_lookup),
            "local_table_lookup": graph_of(local_lookup),
        }
        times = {n: [] for n in graphs}
        for _ in range(9):
            for n, g in graphs.items():
                times[n].append(timed(g))
        print(
            json.dumps(
                {
                    "rows": rows,
                    "us_per_lookup": {
                        n: {
                            "median": round(statistics.median(t), 2),
                            "min": round(min(t), 2),
                            "max": round(max(t), 2),
                        }
                        for n, t in times.items()
                    },
                }
            ),
            flush=True,
        )
    for requests in (1, 3, 8):
        sets = [
            torch.randint(0, VOCAB, (requests,), generator=gen, device=dev).to(
                torch.int32
            )
            for _ in range(GN)
        ]
        KEEP.append(sets)

        def production_block(sets=sets):
            for anchors in sets:
                KEEP.append(
                    masked_embedding(
                        block_ids(anchors, 7),
                        shards[RANK],
                        RANK * SHARD,
                        (RANK + 1) * SHARD,
                    )
                )

        def local_block(sets=sets):
            for anchors in sets:
                KEEP.append(embed_block(anchors, NOISE, 7, emb.local_table, VOCAB))

        graphs = {
            "fill_copy_shard_lookup": graph_of(production_block),
            "embed_block": graph_of(local_block),
        }
        times = {n: [] for n in graphs}
        for _ in range(9):
            for n, g in graphs.items():
                times[n].append(timed(g))
        print(
            json.dumps(
                {
                    "requests": requests,
                    "drafts": 7,
                    "us_per_block": {
                        n: {
                            "median": round(statistics.median(t), 2),
                            "min": round(min(t), 2),
                            "max": round(max(t), 2),
                        }
                        for n, t in times.items()
                    },
                }
            ),
            flush=True,
        )


ok = every_pattern()
table = synthetic_table()
shards = [table[r * SHARD : (r + 1) * SHARD].clone() for r in range(TP)]
emb, good = bound(table, shards)
ok = ok and good
if emb.local_table is not None:
    ok = whole_vocabulary(emb, shards) and ok
    ok = draft_block(emb, shards) and ok
    if BENCH:
        bench(emb, shards)
print(json.dumps({"all_ok": bool(ok)}), flush=True)
