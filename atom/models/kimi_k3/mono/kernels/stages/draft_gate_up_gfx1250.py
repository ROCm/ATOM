# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A DSpark draft layer's gate_up_proj and SiLU * mul in one gfx1250 launch:
from the post-attention norm's FP8 row (``add_rmsnorm_quant``'s E4M3 and its
transposed per-1x128 E8M0 codes) to the MLP's BF16 activation
``[rows, inter]`` -- in place of the FlyDSL ``gemm_a8w8_mxfp8_128`` GEMM, its
split-K reduce and ``act_and_mul`` (``silu_and_mul``), bit for bit.

The geometry is ``mla_draft_front_gfx1250``'s at ``ksplit`` 4, its token
loaders and its chain unchanged (``draft_mlp_layout``): a cluster of four
CTAs owns 32 intermediate columns -- two gate tasks and the two up tasks of
the same columns -- and CTA ``r`` chains every task over K quarter ``r``:
its weights four 28 KB runs of the rank's ``[2 inter, HIDDEN]`` FP8 shard
(AITER's preshuffle) copied to LDS by the TDM, a wave half a run, its token
quarter staged in the WMMA operand layout a seven-group piece at a time.
A chain is the scaled FP8 WMMA (16x16x128, the E8M0 codes on the weights and
the tokens) over its K groups in order from zero, as production's GEMM runs
each of its ``split_k`` splits: at 4 the CTA's quarter, at 8 each of its two
pieces. The splits' f32 sums go to ``partials`` at SE scope; past one
cluster barrier CTA 0 and CTA 1 each sum the gate task and the up task of
their 16 columns in split order (the split-K reduce's ``((p0 + p1) + ..)``)
and round them to BF16 once.

The activation is ``act_and_mul_kernel<bf16, silu>``'s arithmetic as the
compiler left it: ``exp(-x)`` the device libm's (``2 ** (x * -log2e)`` with
the product in double-float -- ``fma(x, -log2e_hi, -n)`` for the fraction,
``n`` the product rounded to even -- the hardware ``2 ** fraction`` scaled
by ``n``, 0 past x = 103.28 and +inf below -88.72),
``v_rcp_f32(1 + exp(-x))``, then ``(up * gate) * rcp`` rounded to f32 at
each product (``v_pk_mul_f32``) and to BF16 once -- all of it with f32
denormals flushed, the HIP kernel's float mode (``s_denorm_mode`` around it:
a gate near -88 has a denormal reciprocal).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as flay
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER
from atom.models.kimi_k3.mono.kernels.stages.mla_absorb_gfx1250 import (
    _pack,
    _scalar_byte,
    _store_words,
)
from atom.models.kimi_k3.mono.kernels.stages.mla_draft_front_gfx1250 import (
    _load_piece,
    _piece,
    _store_piece,
    _wait_weights,
)
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import bf_hi, bf_lo, hw_exp2, hw_rcp, rsrc, traced, uniform
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.consensus import MonoUnsupported

THREADS = lay.THREADS
WAVES = lay.WAVES
LANES = lay.LANES
ACC = 8
SLOTS = lay.GATE_UP_SLOTS
SCOPE_SE = gemv_tdm.SCOPE_SE
assert lay.GATE_UP_CLUSTER == TDM_CLUSTER and flay.THREADS == THREADS
assert flay.GROUP_BYTES == lay.GROUP_BYTES

DRAFT_GATE_UP_ABI = KernelAbi(("x", "x_scale", "w", "w_scale", "act", "partials", "gu"))


@dataclass(frozen=True)
class DraftGateUpKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    rows: int
    inter: int
    split_k: int
    abi: KernelAbi = DRAFT_GATE_UP_ABI
    launch_count: int = 1


def _slot_codes(c, slot):
    """The E8M0 codes of a task slot's weights over the CTA's K quarter:
    its 128-row block's bytes."""
    block = uniform(lay.gate_up_slot_row(c["cluster"], slot, c["inter"]) // 128)
    first = block * lay.GATE_UP_GROUPS + c["rank"] * lay.QUARTER_GROUPS
    return [_scalar_byte(c["w_scale"], first + j) for j in range(lay.QUARTER_GROUPS)]


@traced
def _store_split(c, acc, slot, split, tile, lane, live):
    """A chain's f32 sums of a split, to ``partials`` for the columns'
    owner."""
    if live:
        at = lay.gate_up_partial_index(
            c["cluster"], slot, split, tile, lane, c["split_k"], c["rows"]
        )
        for h in range_constexpr(2):
            bo.buffer_store(
                fx.Vector.from_elements([acc[h * 4 + i] for i in range(4)], fx.Float32),
                rsrc(c["partials"]),
                at + h * 4,
                cache_modifier=SCOPE_SE,
            )


def _f32(bits: int):
    return fx.Int32(bits - (1 << 32) if bits >= 1 << 31 else bits).bitcast(fx.Float32)


def _fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def _rcp_one_plus_exp_neg(x):
    """``v_rcp_f32(1 + exp(-x))`` as ``act_and_mul_kernel<bf16, silu>`` runs
    it: the device libm's ``expf`` of ``-x`` with the negation folded into
    its constants and the fraction ``fma(x, -log2e_hi, -n)``."""
    hi = _f32(lay.SILU_NEG_LOG2E_HI_BITS)
    lo = _f32(lay.SILU_NEG_LOG2E_LO_BITS)
    product = x * hi
    low = _fma(hi, x, -product)
    whole = fx.Float32(fmath.roundeven(product))
    low = _fma(lo, x, low)
    fraction = _fma(hi, x, -whole)
    r = hw_exp2(fraction + low)
    exponent = fx.Int32(
        llvm.call_intrinsic(
            T.i32, "llvm.fptosi.sat.i32.f32", [whole.ir_value()], [], []
        )
    )
    r = fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.ldexp.f32.i32", [r.ir_value(), exponent.ir_value()], [], []
        )
    )
    r = (x > _f32(lay.SILU_EXP_ZERO_BITS)).select(fx.Float32(0.0), r)
    r = (x < _f32(lay.SILU_EXP_INF_BITS)).select(_f32(0x7F800000), r)
    return hw_rcp(r + fx.Float32(1.0))


def _silu_mul(gate, up):
    """A BF16 pair word of ``silu(gate) * up``: ``(up * gate) * rcp``, each
    product rounded to f32 by a ``v_pk_mul_f32`` (LLVM would fold the second
    into the BF16 conversion: one rounding instead of two)."""
    gates = [bf_lo(gate), bf_hi(gate)]
    ups = [bf_lo(up), bf_hi(up)]
    rcps = [_rcp_one_plus_exp_neg(g) for g in gates]
    out = fx.Vector(
        llvm.inline_asm(
            T.vec(2, T.f32),
            [
                fx.Vector.from_elements(pair, fx.Float32).ir_value()
                for pair in (ups, gates, rcps)
            ],
            "v_pk_mul_f32 $0, $1, $2\n v_pk_mul_f32 $0, $0, $3",
            "=&v,v,v,v",
            has_side_effects=False,
        )
    )
    return _pack(fx.Float32(out[0]), fx.Float32(out[1]))


def _flush_enter(word):
    """``word``, past an ``s_denorm_mode`` that flushes f32 denormals
    (operands and results): ``act_and_mul_kernel``'s code object runs
    FLOAT_DENORM_MODE_32 0, this kernel 3. Every value the activation's
    arithmetic reads goes through one, so none of it is scheduled ahead."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(word).ir_value()],
            "s_denorm_mode 12\n v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def _flush_leave(words):
    """``words``, all computed, then the kernel's own denormal mode back."""
    first = fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(word).ir_value() for word in words],
            "v_mov_b32 $0, $1\n s_denorm_mode 15",
            "=v" + ",v" * len(words),
            has_side_effects=True,
        )
    )
    return [first, *words[1:]]


def _summed(c, slot, tile, lane):
    """A task slot's splits of a token tile summed in split order -- the
    split-K reduce: ``((p0 + p1) + p2) + ..`` -- as BF16 pair words."""
    total = None
    for split in range_constexpr(c["split_k"]):
        at = lay.gate_up_partial_index(
            c["cluster"], slot, split, tile, lane, c["split_k"], c["rows"]
        )
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
    return [_pack(total[2 * i], total[2 * i + 1]) for i in range_constexpr(ACC // 2)]


@traced
def _finish(c, lane, wave):
    """An owning CTA (0, 1), a wave a token tile: its 16 columns' gate and
    up sums, then SiLU(gate) * up, stored."""
    rows, inter, owner = c["rows"], c["inter"], c["rank"]
    tile = wave
    token = tile * ROWS + lane % ROWS
    if (owner < lay.GATE_UP_OWNERS) & (tile < lay.tiles(rows)) & (token < rows):
        gates = _summed(c, owner, tile, lane)
        ups = _summed(c, lay.GATE_UP_OWNERS + owner, tile, lane)
        column = lay.gate_up_column(c["cluster"], owner) + (lane // ROWS) * ACC
        if c["gu"] != 0:
            _store_words(gates, c["gu"], (token * (2 * inter) + column) // 2)
            _store_words(ups, c["gu"], (token * (2 * inter) + inter + column) // 2)
        gates = [_flush_enter(word) for word in gates]
        ups = [_flush_enter(word) for word in ups]
        out = _flush_leave(
            [_silu_mul(gates[i], ups[i]) for i in range_constexpr(ACC // 2)]
        )
        _store_words(out, c["act"], (token * inter + column) // 2)


@traced
def _project(c, lds, lane, wave):
    """A CTA: its weight runs copied, its chains over its K quarter, their
    splits' sums traded past the cluster barrier."""
    rows, inter, split_k = c["rows"], c["inter"], c["split_k"]
    rank, cluster_id = c["rank"], c["cluster"]
    slot, half = wave % flay.RUNS, wave // flay.RUNS
    source = fx.Int64(lay.gate_up_slot_row(cluster_id, slot, inter)) * fx.Int64(
        HIDDEN
    ) + fx.Int64(
        (rank * lay.QUARTER_GROUPS + half * lay.PIECE_GROUPS) * lay.GROUP_BYTES
    )
    gemv_tdm.load_rows(
        c["w"] + source,
        lay.GROUP_BYTES // 2,
        lds + (slot * lay.QUARTER_GROUPS + half * lay.PIECE_GROUPS) * lay.GROUP_WORDS,
        lay.PIECE_GROUPS,
        lay.PIECE_GROUPS,
        padded=False,
        cache_policy=gemv_tdm._WEIGHT_CACHE_POLICY,
        whole=True,
        width=lay.GROUP_BYTES // 2,
        uniform=True,
    )
    tile_count = lay.tiles(rows)
    tokens = [lds + flay.token_offset(b, tile_count) for b in range(2)]
    codes = [lds + flay.code_offset(b, tile_count) for b in range(2)]
    tid = wave * LANES + lane
    _store_piece(_load_piece(c, 0, tid), tokens[0], codes[0])
    ahead = _load_piece(c, 1, tid)
    count = lay.gate_up_chains(rows)
    chains = []
    for r in range_constexpr(-(-count // WAVES)):
        index = wave + WAVES * r
        chain_slot = index % SLOTS
        chains.append(
            (chain_slot, index // SLOTS, index < count, _slot_codes(c, chain_slot))
        )
    accs = [fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in chains]
    for p in range_constexpr(2):
        _wait_weights(c, wave, p)
        gpu.barrier()
        for r in range_constexpr(len(chains)):
            chain_slot, tile, live, w_codes = chains[r]
            c["w_codes"] = w_codes
            accs[r] = _piece(c, lds, tokens[p], codes[p], lane, chain_slot, tile, p,
                             accs[r], live)  # fmt: skip
            split = lay.gate_up_piece_split(split_k, rank, p)
            if const_expr(split is not None):
                _store_split(c, accs[r], chain_slot, split, tile, lane, live)
                accs[r] = fx.Vector.filled(ACC, 0.0, fx.Float32)
        if const_expr(p == 0):
            # the second piece's tokens, its loads in flight since the entry
            _store_piece(ahead, tokens[1], codes[1])
    rocdl.s_wait_storecnt(0)
    cluster.cluster_barrier()
    _finish(c, lane, wave)


def build_draft_gate_up(rows: int, inter: int, split_k: int) -> DraftGateUpKernel:
    """gate_up_proj and SiLU * mul at ``rows`` token rows for a rank's
    intermediate shard ``inter``, the GEMM's sum split in ``split_k``
    (production's at the width: 4 or 8). ``x`` [rows, HIDDEN] E4M3 and
    ``x_scale`` its [56, rows] E8M0 codes (the norm's transposed per-1x128
    layout); ``w``/``w_scale`` gate_up_proj's preshuffled FP8
    [2 inter, HIDDEN] (gate rows first) and its [2 inter / 128, 56] E8M0
    block scales; ``partials`` ``draft_mlp_layout.gate_up_partial_bytes``
    of scratch. Out: ``act`` [rows, inter] BF16. ``gu`` (checks only; 0: not
    written): the GEMM's [rows, 2 inter] BF16 rows the activation read."""
    why = lay.gate_up_refusal(rows, inter, split_k)
    if why is not None:
        raise MonoUnsupported(f"draft gate_up: {why}")
    grid = lay.gate_up_ctas(inter)
    lds_count = lay.gate_up_lds_words(rows)
    name = f"k3_draft_gate_up_gfx1250_s{rows}_i{inter}_sk{split_k}"
    sources = SOURCES

    @fx.struct
    class Lds:
        words: fx.Array[fx.Int32, lds_count, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        x: Int64,
        x_scale: Int64,
        w: Int64,
        w_scale: Int64,
        act: Int64,
        partials: Int64,
        gu: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % LANES
        wave = uniform(tid // LANES)
        c = {
            "rows": rows,
            "inter": inter,
            "split_k": split_k,
            # the draft front's token loaders and chain: a CTA's K quarter
            "ksplit": 4,
            "rank": uniform(bid % TDM_CLUSTER),
            "cluster": uniform(bid // TDM_CLUSTER),
            "x": x,
            "x_scale": x_scale,
            "w": w,
            "w_scale": w_scale,
            "act": act,
            "partials": partials,
            "gu": gu,
        }
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        _project(c, lds.words.ptr, lane, wave)

    @flyc.jit
    def launch(
        x: Int64,
        x_scale: Int64,
        w: Int64,
        w_scale: Int64,
        act: Int64,
        partials: Int64,
        gu: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            x,
            x_scale,
            w,
            w_scale,
            act,
            partials,
            gu,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    DRAFT_GATE_UP_ABI.check(kernel, launch)
    return DraftGateUpKernel(name, launch, (grid,), (THREADS,), rows, inter, split_k)
