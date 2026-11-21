# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250: the Kimi-K3 DSpark draft's context KV projection, every layer in
one resident launch.

After a target forward each draft layer projects the same context tensor with
its own ``fused_qkv_a_proj`` and writes the kv_a and rope rows to its cache
(``K3DSparkMLAAttention.write_context_kv``): in production an E8M0 activation
quant, the full 2112-row FP8 GEMM (and its split-K reduce) and the cache write,
a layer. Here the context is quantized once (production's kernel: the layers
quantize the same tensor) and one launch computes all layers' kv_a and rope
rows -- 576 of the 2112 -- to ``out`` [layers, rows, 576] BF16, which
production's cache write then takes a layer.

The arithmetic is the draft front's GEMV (``mla_draft_front_gfx1250``), which
is production's FlyDSL ``gemm_a8w8_mxfp8_128`` at the width's split-K: a
cluster of four CTAs owns a 64-row group of the preshuffled weight, the scaled
FP8 WMMA (16x16x128, E8M0 codes on weights and tokens) chained over K in order,
at split-K 4 the four quarters' sums added in quarter order by the task's owner
past one cluster barrier, the sum rounded to BF16. Only what a CTA does with
its sums differs: plain stores to ``out`` (nothing in this launch reads them).

A layer's weight and its block scales come from a pointer table (``tab``:
[layer] (w_a, s_a) i64), read once a wave.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_gfx1250 as mf
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as flay
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER
from atom.models.kimi_k3.mono.kernels.stages.mla_absorb_gfx1250 import _store_words
from atom.models.kimi_k3.mono.kernels.stages.mla_front_gfx1250 import _pack
from atom.models.kimi_k3.step_gap import ctx_kv_layout as lay
from atom.models.kimi_k3.step_gap.sources import SOURCES
from atom.mono.device.ops import load_ptr64, rsrc, traced, uniform
from atom.mono.runtime.abi import KernelAbi

THREADS = flay.THREADS
WAVES = flay.WAVES
LANES = flay.LANES
ACC = mf.ACC
SCOPE_SE = gemv_tdm.SCOPE_SE
GROUP_WORDS = flay.GROUP_BYTES // 4

CTX_KV_ABI = KernelAbi(("tab", "x", "x_scale", "out", "partials"))


@dataclass(frozen=True)
class CtxKvKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    rows: int
    ksplit: int
    layers: int
    abi: KernelAbi = CTX_KV_ABI
    launch_count: int = 1


@traced
def _store_rows(c, words, tile, task, lane):
    """A task's 16 rows of a token tile, BF16, a lane its token's eight rows,
    to the token's context row of the CTA's layer."""
    rows = c["rows"]
    token = tile * ROWS + lane % ROWS
    col = (
        (c["cluster"] - lay.FIRST_CLUSTER) * flay.GROUP_ROWS
        + task * ROWS
        + (lane // ROWS) * ACC
    )
    if token < rows:
        _store_words(
            words, c["out"], ((c["layer"] * rows + token) * lay.ENTRY + col) // 2
        )


@traced
def _sum_quarters(c, lane, wave):
    """The owned task's four quarter sums, in quarter order (the split-K
    reduce: ``((p0 + p1) + p2) + p3``), a wave a tile: the draft front's."""
    tiles = c["plan"].tiles
    cta, rank = c["bid"], c["rank"]
    for r in range_constexpr(-(-tiles // WAVES)):
        tile = wave + WAVES * r
        if tile < tiles:
            total = None
            for q in range_constexpr(flay.TASKS):
                at = (((cta * flay.TASKS + q) * tiles + tile) * LANES + lane) * ACC
                halves = [
                    fx.Vector(
                        bo.buffer_load(
                            rsrc(c["partials"]),
                            at + h * 4,
                            vec_width=4,
                            dtype=T.f32,
                            cache_modifier=SCOPE_SE,
                        )
                    )
                    for h in range(2)
                ]
                part = [fx.Float32(halves[i // 4][i % 4]) for i in range(ACC)]
                total = part if total is None else [a + b for a, b in zip(total, part)]
            _store_rows(c, [_pack(total[2 * i], total[2 * i + 1]) for i in range(4)],
                        tile, rank, lane)  # fmt: skip


@traced
def _project(c, lds, lane, wave):
    """A CTA: its weight runs copied, its chains, then (ksplit 4) the quarters'
    sums exchanged and the owned task's summed in order. The draft front's
    ``_project`` but for where the sums are stored."""
    rank, cluster_id, ksplit = c["rank"], c["cluster"], c["ksplit"]
    run, half = wave % flay.RUNS, wave // flay.RUNS
    if const_expr(ksplit == 4):
        task, group = run, rank * flay.RUN_GROUPS
    else:
        task, group = rank, run * flay.RUN_GROUPS
    group = group + half * flay.PIECE_GROUPS
    source = fx.Int64(cluster_id * flay.GROUP_ROWS + task * ROWS) * fx.Int64(
        HIDDEN
    ) + fx.Int64(group) * fx.Int64(flay.GROUP_BYTES)
    gemv_tdm.load_rows(
        c["w_a"] + source,
        flay.GROUP_BYTES // 2,
        lds + (run * flay.RUN_GROUPS + half * flay.PIECE_GROUPS) * GROUP_WORDS,
        flay.PIECE_GROUPS,
        flay.PIECE_GROUPS,
        padded=False,
        cache_policy=gemv_tdm._WEIGHT_CACHE_POLICY,
        whole=True,
        width=flay.GROUP_BYTES // 2,
        uniform=True,
    )
    c["w_codes"] = mf._weight_codes(c, cluster_id * flay.GROUP_ROWS // 128)
    plan = c["plan"]
    pieces = flay.pieces(ksplit)
    tokens = [lds + flay.token_offset(b, plan.tiles) for b in range(2)]
    codes_lds = [lds + flay.code_offset(b, plan.tiles) for b in range(2)]
    tid = wave * LANES + lane
    mf._store_piece(mf._load_piece(c, 0, tid), tokens[0], codes_lds[0])
    ahead = mf._load_piece(c, 1, tid) if const_expr(pieces > 1) else None
    rounds = -(-plan.chain_count // WAVES)
    chains = []
    for r in range_constexpr(rounds):
        index = wave + WAVES * r
        if const_expr(ksplit == 4):
            chains.append(
                (index % flay.TASKS, index // flay.TASKS, index < plan.chain_count)
            )
        else:
            chains.append((fx.Int32(0), index, index < plan.chain_count))
    accs = [fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(rounds)]
    for p in range_constexpr(pieces):
        mf._wait_weights(c, wave, p)
        gpu.barrier()
        for r in range_constexpr(rounds):
            task, tile, live = chains[r]
            accs[r] = mf._piece(c, lds, tokens[p % 2], codes_lds[p % 2], lane, task,
                                tile, p, accs[r], live)  # fmt: skip
        if const_expr(p + 1 < pieces):
            mf._store_piece(ahead, tokens[(p + 1) % 2], codes_lds[(p + 1) % 2])
            if const_expr(p + 2 < pieces):
                ahead = mf._load_piece(c, p + 2, tid)
    results = [(accs[r], *chains[r]) for r in range(rounds)]
    if const_expr(ksplit == 1):
        for acc, _, tile, live in results:
            if live:
                _store_rows(c, mf._bf16_words(acc), tile, rank, lane)
        return
    tiles = plan.tiles
    cta = c["bid"]
    for acc, chain_task, tile, live in results:
        if live:
            owner = cta - rank + chain_task
            at = (((owner * flay.TASKS + rank) * tiles + tile) * LANES + lane) * ACC
            for h in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [acc[h * 4 + i] for i in range(4)], fx.Float32
                    ),
                    rsrc(c["partials"]),
                    at + h * 4,
                    cache_modifier=SCOPE_SE,
                )
    rocdl.s_wait_storecnt(0)
    cluster.cluster_barrier()
    _sum_quarters(c, lane, wave)


def ctx_kv_name(rows: int, ksplit: int, layers: int) -> str:
    return f"k3_draft_ctx_kv_gfx1250_s{rows}_k{ksplit}_l{layers}"


def build_ctx_kv_proj(rows: int, ksplit: int, layers: int) -> CtxKvKernel:
    """The context KV projection at ``rows`` context rows for ``layers`` draft
    layers, ``fused_qkv_a_proj``'s sum split in ``ksplit`` (production's
    split_k at the width: 1 or 4). ``tab``: [layers] (w_a, s_a) i64 -- a
    layer's preshuffled FP8 [2176, HIDDEN] and its [17, 56] E8M0 block scales;
    ``x`` [rows, HIDDEN] E4M3 and ``x_scale`` its [56, rows] E8M0 codes (the
    transposed per-1x128 layout); ``partials``
    ``ctx_kv_layout.partial_bytes(layers)``. Out: ``out`` [layers, rows, 576]
    BF16, a row kv_a (512) then rope (64)."""
    why = lay.refusal(rows, ksplit, layers)
    if why:
        raise ValueError(why)
    plan = flay.DraftFrontPlan(rows, ksplit)
    grid = lay.grid(layers)
    name = ctx_kv_name(rows, ksplit, layers)
    sources = SOURCES
    lds_count = flay.lds_words(rows)

    @fx.struct
    class Lds:
        words: fx.Array[fx.Int32, lds_count, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(tab: Int64, x: Int64, x_scale: Int64, out: Int64, partials: Int64):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % LANES
        wave = uniform(tid // LANES)
        group = uniform(bid // TDM_CLUSTER)
        layer = group // lay.CLUSTERS
        c = {
            "rows": rows,
            "ksplit": ksplit,
            "plan": plan,
            "lane": lane,
            "bid": uniform(bid),
            "rank": uniform(bid % TDM_CLUSTER),
            "cluster": group % lay.CLUSTERS + lay.FIRST_CLUSTER,
            "layer": layer,
            "x": x,
            "x_scale": x_scale,
            "w_a": load_ptr64(tab, layer * 2),
            "s_a": load_ptr64(tab, layer * 2 + 1),
            "out": out,
            "partials": partials,
        }
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        _project(c, lds.words.ptr, lane, wave)

    @flyc.jit
    def launch(
        tab: Int64,
        x: Int64,
        x_scale: Int64,
        out: Int64,
        partials: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            tab,
            x,
            x_scale,
            out,
            partials,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    CTX_KV_ABI.check(kernel, launch)
    return CtxKvKernel(name, launch, (grid,), (THREADS,), rows, ksplit, layers)
