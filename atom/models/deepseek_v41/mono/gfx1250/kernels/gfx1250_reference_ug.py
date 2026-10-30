# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 UG: paired tiles, four K slices and two-round prefetch.

Weights retain the mother's tile16 GUGU and shuffle_scale layout. Native
wave32 fragments restore the original red slots before its own epilogue.
"""
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import HIDDEN
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_moe import fp4_weight_offset
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_route import load_x8r_c1, load_route_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import publish_plain
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import (
    ug_rounds, ug_wave_pair, ug_red_slot, ug_epilogue,
)
from .reference_layout import wsl_words
from atom.mono.device.mx import scale_index
from atom.mono.device.ops import CM_NT_GFX1250, rsrc, traced
from atom.mono.plan.execution import BLOCKS, first_task
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_route import C1RouteShape
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import pairwise_sum, LIMIT, LOG2E, st_dev, UG_PART
from atom.mono.device.mx import FP8_MAX, code_ceil, clamp_fp8, rcp_pow2
from atom.mono.device.ops import CM_DEV, butterfly, hw_rcp, hw_exp2, bf16_round, lane_gather, fp8_pack4

THREADS = 256


@traced
def ug_pair_loads_c1(c, e, rg0, st0):
    """One original two-K128 round's scales and both tile operands."""
    lane, half, d = c["lane"], c["lane"]//16, c["d"]
    rw, rs = rsrc(c["args"]["w13"]), rsrc(c["args"]["w13_s"])
    # The original 64 lanes each load one scale dword: physical lane l loads
    # l and l+32. No n32k4 scale transform or per-step global scale reread.
    block = (e*(2*d.inter)+rg0*16)//32*(HIDDEN//32*8)
    sv = [fx.Int32(bo.buffer_load(rs,block+32*st0+lane+32*h,
                                vec_width=1,dtype=T.i32,cache_modifier=CM_NT_GFX1250))
          for h in range(2)]
    # Fold the static byte-to-word factor before the dynamic multiply. The
    # expert383 byte base exceeds signed32; its dword base remains positive.
    base = e*((2*d.inter)*(HIDDEN//2)//4)
    weights = []
    for t in range_constexpr(2):
        tile = []
        for i in range_constexpr(2):
            pieces = [fx.Vector(bo.buffer_load(
                rw,base+fp4_weight_offset(rg0+t,st0+i,half,lane%16,j,HIDDEN)//4,
                vec_width=4,dtype=T.i32,cache_modifier=CM_NT_GFX1250)) for j in range(2)]
            tile.append(pieces[0].shuffle(pieces[1],list(range(8))))
        weights.append(tile)
    return sv, weights


@traced
def ug_pair_matrix_c1(c, e, rg0, ks):
    """Mother two rounds in flight, input/scale reused across gate and up."""
    lane, wave = c["lane"], c["wave"]
    sl = c["wsl"]+wave*wsl_words(c["d"])
    rounds = ug_rounds(HIDDEN,ks,4)
    assert len(rounds)==5
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(
        16,16,128,fx.Float4E2M1FN,fx.Float8E4M3FN,fx.Float32))
    accs = [fx.make_rmem_tensor(8,fx.Float32) for _ in range(2)]
    for acc in accs:
        acc.store(fx.full(8,0.,fx.Float32))
    ops = [ug_pair_loads_c1(c,e,rg0,rounds[q][1]) for q in range(2)]
    for r in range_constexpr(len(rounds)):
        sc0,st0,n,sc_steps = rounds[r]
        assert n==2 and sc_steps==2
        sv,weights = ops[r]
        for h in range_constexpr(2):
            fx.ptr_store(sv[h],sl+lane+32*h)
        for i in range_constexpr(2):
            # All native K groups in one WMMA operand; each activation and
            # input code word serves both original projection tiles.
            chunks = [fx.Vector(fx.ptr_load(c["xl"]+(st0+i)*32+lane//16*4+j*8,
                      result_type=fx.Vector.make_type(4,fx.Int32))) for j in range(4)]
            xv = chunks[0].shuffle(chunks[1],list(range(8))).shuffle(
                chunks[2].shuffle(chunks[3],list(range(8))),list(range(16)))
            sx = fx.Int32(0)
            for j in range_constexpr(4):
                sx = sx | (fx.ptr_load(c["xsl"]+(st0+i)*4+j)<<(8*j))
            x = fx.make_rmem_tensor(16,fx.Int32); x.store(xv)
            for t in range_constexpr(2):
                sw = fx.Int32(0)
                for j in range_constexpr(4):
                    at = scale_index(t*16+lane%16,(st0+i)*4+j,HIDDEN//32)-128*sc0
                    sw = sw | (((fx.ptr_load(sl+at//4)>>(8*(at%4)))&255)<<(8*j))
                w = fx.make_rmem_tensor(8,fx.Int32); w.store(weights[t][i])
                fx.gemm(atom,accs[t],w,x,accs[t],scale_a=sw,scale_b=sx)
        if r+2 < len(rounds):
            ops.append(ug_pair_loads_c1(c,e,rg0,rounds[r+2][1]))
    return [acc.load() for acc in accs]


@traced
def ug_epilogue_compact(c, part0, idx, first, count):
    """Thread ``idx``'s value of an ug task: silu x up of its (group, pick j,
    column), the group's MXFP8 scale, its byte of MID (the pick's row, ``first``
    + j in the pick order)."""
    lane, red, d, rs = c["lane"], c["red"], c["d"], c["rs"]
    cols = rs.tile_picks
    if idx < rs.ug_groups * cols * UG_PART:
        p = idx // (cols * UG_PART)
        part = part0 + p
        t = idx % (cols * UG_PART) // UG_PART
        i = idx % UG_PART
        tile = i // 16
        r = i % 16
        at = (16 * (r // 4) + t) * 4 + r % 4
        g, up = [
            pairwise_sum(
                [
                    fx.ptr_load(red + ug_red_slot(rs, p, tile, ks, t) * 64 * 4 + at)
                    for ks in range(rs.ug_k_slices)
                ]
            )
            for t in range(2)
        ]
        gc = fx.min(g, fx.Float32(LIMIT))
        uc = fx.max(fx.min(up, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
        sig = hw_rcp(fx.Float32(1.0) + hw_exp2(gc * fx.Float32(-LOG2E)))
        # the stage-1 kernel stores the intermediate as bf16; the stage-2 input
        # quant (fused_mx_quant_moe_sort) makes it MXFP8: amax floored 1e-10,
        # the ceil code, x / scale clamped, RNE
        v = bf16_round(gc * sig * uc)
        amax = fx.max(butterfly(abs(v), (1, 2, 4, 8, 16), fx.max), fx.Float32(1e-10))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        q8 = clamp_fp8(v * rcp_pow2(code))
        nb = [
            lane_gather(q8.bitcast(fx.Int32), fx.min(lane + q, 63)).bitcast(fx.Float32)
            for q in range(1, 4)
        ]
        # 4 threads' bytes -> one word
        word = fp8_pack4(q8, *nb)
        row = first + t
        if (i % 4 == 0) & (t < count):
            st_dev(word, c["mid"], row * d.mid_words + (part // 4) * 32 + (i // 16) * 16 + (part % 4) * 4 + (i % 16) // 4)
        if (i == 0) & (t < count):
            bo.buffer_store(code.to(fx.Int8), rsrc(c["mids"].value),
                            row * (d.inter // 32) + part, cache_modifier=CM_DEV)


@traced
def stage_ug_c1(c, task):
    rs, d, tid, lane, wave = c["rs"],c["d"],c["tid"],c["lane"],c["wave"]
    tab = c["tab"]
    u = task//rs.ug_parts
    if u < rs.n_union(tab,1):
        e = c["rs"].union_expert(tab,1,u)
        first = c["rs"].slot_first(tab,u)
        count = c["rs"].slot_first(tab,u+1)-first
        part0 = task%rs.ug_parts
        p,rg0,ks,real = ug_wave_pair(c,part0)
        accs = [fx.Vector.filled(8,0.,fx.Float32) for _ in range(2)]
        if real:
            accs = ug_pair_matrix_c1(c,e,rg0,ks)
        if lane%16==0:
            for t in range_constexpr(2):
                slot = ug_red_slot(rs,p,wave%2,ks,t)
                for j in range_constexpr(8):
                    row = lane//16*8+j
                    at = (16*(row//4))*4+row%4
                    fx.ptr_store(accs[t][j],c["red"]+slot*64*4+at)
        gpu.barrier()
        # c1 has one live pick and one group: the mother's epilogue's 32
        # lanes retain its four-slice pairwise sum and plain MID/MIDS ABI.
        if const_expr(isinstance(rs, C1RouteShape)):
            # Each task owns eight MID words and one scale byte. Four tasks
            # write disjoint byte addresses; down waits all four UGF flags
            # before loading their combined K128 scale word.
            ug_epilogue_compact(c,part0,tid,first,count)
        else:
            ug_epilogue(c,part0,tid,first,count)
        publish_plain(c["put"],c["ugf"],task,1,tid==0)
    gpu.barrier()


@traced
def run_ug_c1(c, bid):
    rs = c["rs"]
    assert rs.tokens==1 and rs.ug_groups==1 and rs.ug_k_slices==4
    task = first_task(bid,rs.ug0)
    if task < rs.max_u*rs.ug_parts:
        load_x8r_c1(c,c["xl"],c["xsl"])
        load_route_c1(c,c["tab"])
        stage_ug_c1(c,task)
