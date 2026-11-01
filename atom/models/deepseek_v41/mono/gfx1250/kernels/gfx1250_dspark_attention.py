# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Original attention score tasks and tagged outputs, translated to wave32.

Eight waves retain their original split ownership. Each physical lane holds
two of the mother's logical lane fragments; native TDM copies each complete key
row to the original padded LDS layout. Query and key operands stay
in registers, and the next split's DMA starts after current LDS read-back and
before current matrix work. AM/AL/AP keep the original pair addresses.
"""
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T
from flydsl.expr.rocdl import tdm_ops

from atom.models.deepseek_v41.mono import attention_plan as aplan
from atom.models.deepseek_v41.mono.gfx950.kernels.attention import (
    BK, HEAD_DIM, HEAD_TILE, NEG, SCORE_LDS_ROW, SCORE_LDS_WORDS,
    _ap, _key_row, _keys, _ms, await_selection, scratch_pairs,
    tile_head, tile_task,
)
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_attention import _wmma
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_index import vector_address
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import fresh, hw_exp2, kernel_symbol, launcher, readlane, rsrc, traced
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import BLOCKS, first_task
from atom.mono.plan.layout import pair_layout

THREADS, WAVES = 256, 8



from flydsl.expr import math as fmath
from atom.models.deepseek_v41.mono.gfx950.kernels.attention import (
    COLS, GROUPS, LOG2E, _ld_row4, _stat_specs,
)
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_post import irq_group, irq_rope_loads
from atom.mono.device.ops import bf16_round, lane_gather, row_shr, uniform
@traced
def rows_to_lds(c, pool, slots, kv):
    for row_index in range_constexpr(BK):
        row = readlane(slots, row_index)
        if const_expr(c.get("defer_tdm_rows", False)):
            # TDM requires a scalar row descriptor. Materialize each row
            # at its copy: hoisting all sixteen across the IRDY seam spilled
            # SGPRs in the joint index/K2 binary. Copy order/LDS are unchanged.
            row = fx.Int32(llvm.inline_asm(T.i32, [row.ir_value()],
                "s_mov_b32 $0, $1", "=s,s", has_side_effects=True))
        # The mother raw buffer-to-LDS intrinsic cannot be selected on gfx1250.
        # One wave-scoped native TDM row copy keeps its direct-to-LDS lifetime.
        pointer = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type,
            fx.AddressSpace.Global,16),pool + fx.Int64(row) * HEAD_DIM * 2)
        source = fx.Tensor(fx.make_view(pointer,fx.make_layout(HEAD_DIM//2, 1)))
        target = fx.Tensor(fx.make_view(kv + row_index * SCORE_LDS_ROW,
                                       fx.make_layout(HEAD_DIM//2, 1)))
        atom = fx.rocdl.make_tdm_atom(source,[HEAD_DIM//2],num_warps=1)
        fx.copy(atom,source,target)


def rows_from_lds(c, kv):
    lane = c["lane"]
    at = kv + (lane % BK) * SCORE_LDS_ROW + (lane // 16) * 8
    return [fx.Vector(fx.ptr_load(at + chunk * 16,
                                 result_type=fx.Vector.make_type(8, fx.Int32)))
            for chunk in range(HEAD_DIM // 32)]


@traced
def score_split(c, t, tile, split, head, kv_len, kws, qws):
    lane, d, a = c["lane"], c["d"], c["args"]
    if split < aplan.written_splits(kv_len):
        key = split * BK + lane % BK
        ok = key < kv_len
        acc = fx.make_rmem_tensor(8, fx.Float32)
        acc.store(fx.full(8, 0., fx.Float32))
        for chunk in range_constexpr(HEAD_DIM // 32):
            kw = fx.Vector.from_elements(
                [ok.select(kws[chunk][i], fx.Int32(0)) for i in range(8)], fx.Int32)
            _wmma(acc, qws[chunk].bitcast(fx.BFloat16), kw.bitcast(fx.BFloat16))
        result = acc.load()
        scale = fx.Int32(a["qk_scale"]).bitcast(fx.Float32)
        sv = [fx.Float32((split * BK + (lane // 16) * 8 + i < kv_len).select(
              result[i] * scale, fx.Float32(NEG))) for i in range(8)]
        quarter_max = [fx.max(fx.max(sv[4*q], sv[4*q+1]),
                             fx.max(sv[4*q+2], sv[4*q+3])) for q in range(2)]
        m = fx.max(quarter_max[0], quarter_max[1])
        m = fx.max(m, m.shuffle_xor(16, 32))
        probs = [hw_exp2(v - m) for v in sv]
        quarters = [(probs[4*q] + probs[4*q+2]) +
                    (probs[4*q+1] + probs[4*q+3]) for q in range(2)]
        # Mother's xor32 then xor16: (quarter0+quarter2)+(quarter1+quarter3).
        total = (quarters[0] + quarters[0].shuffle_xor(16, 32)) + \
                (quarters[1] + quarters[1].shuffle_xor(16, 32))
        if lane < 16:
            c["put"](c["am"], _ms(d, t, split, head), m)
            c["put"](c["al"], _ms(d, t, split, head), total)
        logical_lane = lane % 16 + (lane // 16) * 32
        for part in range_constexpr(2):
            packed = fx.Vector.from_elements(probs[part*4:part*4+4], fx.Float32).to(
                fx.BFloat16).bitcast(fx.Int32)
            c["put_words"](c["ap"],
                (_ap(d, t, tile, split) + logical_lane + part*16) * 2,
                [packed[0], packed[1]])


@traced
def stage_attn_score(c, task, per):
    lane, wave, d, a = c["lane"], c["wave"], c["d"], c["args"]
    t, tile, sg = tile_task(task, per, d.head_tiles)
    head, live = tile_head(d, tile, lane % 16)
    await_selection(c, t)
    kv_len, meta = _keys(a, t)
    splits = [aplan.score_split(sg, per, k, wave)
              for k in range(aplan.score_splits_a_wave(per))]
    slots = [_key_row(a, t, meta, fx.min(sp * BK + lane % 16, kv_len - 1))
             for sp in splits]
    qws = []
    for chunk in range_constexpr(HEAD_DIM // 32):
        at = ((t * d.heads + head) * HEAD_DIM + chunk*32 + (lane//16)*16)//2
        halves = [fx.Vector(bo.buffer_load(rsrc(a["q"]), at + part*4,
                                          vec_width=4, dtype=T.i32)) for part in range(2)]
        qws.append(fx.Vector.from_elements([halves[p][i] for p in range(2)
                                           for i in range(4)], fx.Int32))
    kv = c["att"]["score_kv"] + wave * SCORE_LDS_WORDS
    rows_to_lds(c, a["pool"], slots[0], kv)
    for k in range_constexpr(len(splits)):
        # gfx1250 names load and LDS counters separately from legacy vmcnt/lgkmcnt.
        rocdl.s_wait_loadcnt(0)
        tdm_ops.tensor_wait(0)
        kws = rows_from_lds(c, kv)
        if const_expr(k + 1 < len(splits)):
            rocdl.s_wait_dscnt(0)
            rows_to_lds(c, a["pool"], slots[k+1], kv)
        score_split(c, t, tile, splits[k], head, kv_len, kws, qws)


@traced
def l_tree(ls, alphas, lane):
    """Mother DPP tree over 64 splits, two 32-lane register streams."""
    vs = [fx.Float32(fmath.fma(ls[p], alphas[p],
              row_shr(ls[p] * alphas[p], lane, 8))) for p in range(2)]
    for k in (4, 2, 1):
        vs = [row_shr(v, lane, k) + v for v in vs]
    vs = [fx.Float32((lane // 16 == 1).select(
          lane_gather(v.bitcast(fx.Int32), 15).bitcast(fx.Float32) + v, v))
          for v in vs]
    b31 = lane_gather(vs[0].bitcast(fx.Int32), 31).bitcast(fx.Float32)
    upper = b31 + vs[1]
    return lane_gather(upper.bitcast(fx.Int32), 31).bitcast(fx.Float32)


@traced
def pv_stats(got, act, lane, wave, att):
    for k in range_constexpr(2):
        h = 2 * wave + k
        ms = [fx.Float32((lane + p * 32 < act).select(
              got[p][k][0].bitcast(fx.Float32), fx.Float32(NEG))) for p in range(2)]
        ls = [fx.Float32((lane + p * 32 < act).select(
              got[p][2+k][0].bitcast(fx.Float32), fx.Float32(0.))) for p in range(2)]
        maximum = fx.max(ms[0], ms[1])
        for off in (16, 8, 4, 2, 1):
            maximum = fx.max(maximum, maximum.shuffle_xor(off, 32))
        alphas = [hw_exp2(m - maximum) for m in ms]
        combined = l_tree(ls, alphas, lane)
        for p in range_constexpr(2):
            fx.ptr_store(alphas[p], att["alpha"] + ((lane+p*32) * HEAD_TILE + h))
        if lane == 0:
            fx.ptr_store(maximum, att["stat"] + h)
            fx.ptr_store(combined, att["stat"] + HEAD_TILE + h)


@traced
def mhtjstest_pv_row(a, t, meta, kv_len, act, sp, virtual):
    """Resolve a live split's row before any payload issue; empty KV is safe."""
    row = fx.Int32(0)
    if sp < act:
        row = _key_row(a, t, meta, fx.min(sp*BK+virtual//4, kv_len-1))
    return row


@traced
def mhtjstest_pv_payload(a, row, act, sp, byte):
    """Keep inactive payloads zero without touching the pool."""
    words = fx.Vector.filled(4, 0, fx.Int32)
    if sp < act:
        words = _ld_row4(dict(kv_fp8=False), a["pool"], row, byte // 2)
    return words


@traced
def pv_split_matrix(c, kv_len, act, sp, tiles, pws):
    """Unchanged valid-split math; caller enters with the complete wave32."""
    lane, wave, att = c["lane"], c["wave"], c["att"]
    head, half = lane % 16, lane // 16
    kt = att["kv"] + wave * (BK * COLS // 2)
    # Original two lane halves write their original quarter rows.
    for p in range_constexpr(2):
        virtual = lane + p * 32
        ok = sp * BK + virtual // 4 < kv_len
        fx.ptr_store(fx.Vector.from_elements(
            [ok.select(tiles[p][e], fx.Int32(0)) for e in range(4)], fx.Int32),
            kt + (virtual // 4 * (COLS // 2) + virtual % 4 * 4))
    rocdl.s_wait_dscnt(0)
    # AP virtual halves contain keys [0:8] and [8:16] for each head.
    words = [pws[p][e] for p in range(2) for e in range(2)]
    own = fx.Vector.from_elements(words, fx.Int32)
    other = fx.Vector.from_elements(
        [word.shuffle_xor(16, 32) for word in words], fx.Int32)
    # Reassemble the four mother quarters, then zero-pad the second K16.
    q0 = [fx.Int32((half == 0).select(own[e], other[e])) for e in range(4)]
    q1 = [fx.Int32((half == 0).select(other[e], own[e])) for e in range(4)]
    # q0 holds quarters0/2 and q1 quarters1/3 (logical lane l, l+32).
    ordered = [q0[0], q0[1], q1[0], q1[1], q0[2], q0[3], q1[2], q1[3]]
    pv = fx.Vector.from_elements(
        [(half == 0).select(word, fx.Int32(0)) for word in ordered],
        fx.Int32).bitcast(fx.BFloat16)
    row_accs = []
    for tt in range_constexpr(2):
        col = tt * 16 + head
        kw = fx.Vector.from_elements([
            (half == 0).select(
                fx.Int32(fx.ptr_load(kt + key * (COLS // 2) + col // 2))
                >> (col % 2 * 16) & 0xFFFF, fx.Int32(0))
            for key in range(BK)], fx.Int32).to(fx.Int16).bitcast(fx.BFloat16)
        accum = fx.make_rmem_tensor(8, fx.Float32)
        accum.store(fx.full(8, 0., fx.Float32))
        _wmma(accum, pv, kw)
        result = accum.load()
        row_accs.append(fx.Vector.from_elements([fx.Float32((sp < act).select(
            result[j], fx.Float32(0.))) for j in range(8)], fx.Float32))
    # A wave's next split may reuse kt only after all reads complete.
    rocdl.s_wait_dscnt(0)
    return row_accs


@traced
def pv_group(c, kv_len, act, splits, tiles, pws, al, sink, i):
    lane, wave, tid, att = c["lane"], c["wave"], c["tid"], c["att"]
    head, half = lane % 16, lane // 16
    kt = att["kv"] + wave * (BK * COLS // 2)
    accs = []
    for r in range_constexpr(8):
        sp = splits[r]
        row_accs = [fx.Vector.filled(8, 0., fx.Float32) for _ in range(2)]
        # Both values are wave-uniform; no CTA barrier is conditional.
        if sp < act:
            row_accs = pv_split_matrix(c, kv_len, act, sp, tiles[r],
                                         [pws[p][r] for p in range(2)])
        accs.append(row_accs)
    for tt in range_constexpr(2):
        for j in range_constexpr(8):
            av = [accs[r][tt][j] for r in range(8)]
            pairs = [fx.Float32(fmath.fma(av[2*m], al[2*m], av[2*m+1] * al[2*m+1]))
                     for m in range(4)]
            node = (pairs[0] + pairs[1]) + (pairs[2] + pairs[3])
            virtual = head + (half * 2 + j // 4) * 16
            k = tt * 4 + j % 4
            fx.ptr_store(node, att["tree"] + ((wave * 64 + virtual) * 8 + k))
    gpu.barrier()
    # Original owner mapping and wave-combine tree; each thread emits two cols.
    for p in range_constexpr(2):
        virtual_tid = tid + p * THREADS
        h, col = virtual_tid // COLS, virtual_tid % COLS
        owner = col % 16 // 4 * 16 + h
        k = col // 16 * 4 + col % 4
        ws = [fx.ptr_load(att["tree"] + ((ww * 64 + owner) * 8 + k)) for ww in range(8)]
        xs = [ws[2*m] + ws[2*m+1] for m in range(4)]
        total = (xs[0] + xs[2]) + (xs[1] + xs[3])
        maximum = fx.ptr_load(att["stat"] + h)
        combined = fx.ptr_load(att["stat"] + HEAD_TILE + h)
        sink_value = sink[p]
        final_m = fx.max(maximum, sink_value * fx.Float32(LOG2E))
        alpha_kv = hw_exp2(maximum - final_m)
        alpha_sink = hw_exp2(fx.Float32(fmath.fma(sink_value, fx.Float32(LOG2E), -final_m)))
        final_l = fx.Float32(fmath.fma(combined, alpha_kv, alpha_sink))
        value = (total * alpha_kv) / fx.max(final_l, fx.Float32(1e-30))
        value = fx.Float32((final_l > fx.Float32(0.)).select(value, fx.Float32(0.)))
        fx.ptr_store(bf16_round(value), att["out"] + ((i*HEAD_TILE+h)*COLS+col))
    gpu.barrier()


@traced
def stage_attn_pv(c, task, groups):
    lane, wave, d, att, a = c["lane"], c["wave"], c["d"], c["att"], c["args"]
    t, tile, g0 = tile_task(task, GROUPS // groups, d.head_tiles)
    await_selection(c, t)
    kv_len, meta = _keys(a, t)
    act, last = uniform((kv_len+BK-1)//BK), aplan.last_split(kv_len)
    split_wave = uniform(wave)
    splits = [split_wave + 8*r for r in range(8)]
    # Match gfx950's lifetime: all row lookups precede payload issue,
    # and those rows are reused by every column group of this PV task.
    rows = [[mhtjstest_pv_row(a, t, meta, kv_len, act, sp, lane+p*32)
             for p in range(2)] for sp in splits]
    def group_tiles(i):
        dc = g0 * groups + i
        return [[mhtjstest_pv_payload(a, rows[r][p], act, splits[r],
                    (dc*COLS+8*(lane%4))*2)
                 for p in range(2)] for r in range(8)]
    tiles = group_tiles(0)
    sink = [fx.Float32(bo.buffer_load(rsrc(a["sink"]),
              tile_head(d, tile, (c["tid"]+p*THREADS)//COLS)[0], vec_width=1,
              dtype=T.f32)) for p in range(2)]
    stats = [_stat_specs(c,t,tile,last,lane+p*32,wave) for p in range(2)]
    if wave == 0:
        for p in range_constexpr(2):
            c["poll"](stats[p])
    gpu.barrier()
    got = [c["poll"](stats[p] + [
        (c["ap"], (_ap(d,t,tile,aplan.polled_split(sp,last))+lane+p*32)*2, 2)
        for sp in splits], batch=12) for p in range(2)]
    pv_stats(got,act,lane,wave,att)
    pws = [values[4:] for values in got]
    gpu.barrier()
    al = [fx.ptr_load(att["alpha"]+(sp*HEAD_TILE+lane%16)) for sp in splits]
    for i in range_constexpr(groups):
        nxt = group_tiles(i+1) if i+1 < groups else None
        pv_group(c,kv_len,act,splits,tiles,pws,al,sink,i)
        tiles = nxt


@traced
def stage_attn_irq(c, task, groups, debug=False):
    t,tile,g0 = tile_task(task,GROUPS//groups,c["d"].head_tiles)
    ropes = irq_rope_loads(c,t,[g0*groups+i for i in range(groups)])
    stage_attn_pv(c,task,groups)
    for i in range_constexpr(groups):
        if const_expr(debug):
            h, pair = c["tid"]//16, c["tid"]%16
            for j in range_constexpr(2):
                value = fx.ptr_load(c["att"]["out"]+((i*HEAD_TILE+h)*COLS+pair*2+j))
                fx.ptr_store(value.to(fx.BFloat16),c["debug_out"]+
                    ((t*c["d"].heads+tile*HEAD_TILE+h)*HEAD_DIM+(g0*groups+i)*COLS+pair*2+j))
        irq_group(c,t,tile,g0*groups+i,c["att"]["out"]+i*HEAD_TILE*COLS,ropes[i])
