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
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_indexer import vector_address
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import fresh, hw_exp2, kernel_symbol, launcher, readlane, rsrc, traced
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import BLOCKS, first_task
from atom.mono.plan.layout import pair_layout

THREADS, WAVES = 256, 8


@traced
def rows_to_lds_c1(c, pool, slots, kv):
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


def rows_from_lds_c1(c, kv):
    lane = c["lane"]
    at = kv + (lane % BK) * SCORE_LDS_ROW + (lane // 16) * 8
    return [fx.Vector(fx.ptr_load(at + chunk * 16,
                                 result_type=fx.Vector.make_type(8, fx.Int32)))
            for chunk in range(HEAD_DIM // 32)]


@traced
def score_split_c1(c, t, tile, split, head, kv_len, kws, qws):
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
def stage_attn_score_c1(c, task, per):
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
    rows_to_lds_c1(c, a["pool"], slots[0], kv)
    for k in range_constexpr(len(splits)):
        # gfx1250 names load and LDS counters separately from legacy vmcnt/lgkmcnt.
        rocdl.s_wait_loadcnt(0)
        tdm_ops.tensor_wait(0)
        kws = rows_from_lds_c1(c, kv)
        if const_expr(k + 1 < len(splits)):
            rocdl.s_wait_dscnt(0)
            rows_to_lds_c1(c, a["pool"], slots[k+1], kv)
        score_split_c1(c, t, tile, splits[k], head, kv_len, kws, qws)


def attention_score_layout_c1():
    return pair_layout(scratch_pairs(1, Dims(2)).items())


def attention_score_symbol_c1(per=8):
    return kernel_symbol("gfx1250_reference_attention_score", b=BLOCKS,t=THREADS,per=per)


@lru_cache(None)
def build_reference_attention_score_c1(per=8):
    if per not in (1, 2, 4, 8):
        raise ValueError("original score tasks per tile must divide eight")
    layout = attention_score_layout_c1()
    dims = Dims(2)
    name = attention_score_symbol_c1(per)
    @fx.struct
    class Smem:
        kv: fx.Array[fx.Int32, WAVES * SCORE_LDS_WORDS, 16]
    @flyc.kernel(name=name, known_block_size=[THREADS,1,1])
    def kernel(q:Int64,pool:Int64,sel:Int64,sbound:Int64,kmeta:Int64,
               scratch:Int64,diag:Int64,layer:Int64,config:Int64):
        tid,bid = fx.thread_idx.x,fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        iptr = fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4)
        cfg = fx.inttoptr(iptr,config)
        fail = fx.inttoptr(iptr,diag + fx.Int64(BLOCKS*8*4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(iptr,layer))+64,diag,
                               {name:i+1 for i,name in enumerate(layout)},fail)
        args = dict(q=q,pool=pool,sel=sel,sbound=sbound,kmeta=kmeta,topk=512,
                    ring_off=fx.ptr_load(cfg+1),ring_slots=512,
                    iact=fx.ptr_load(cfg),qk_scale=fx.ptr_load(cfg+2))
        c = dict(S=1,tid=tid,bid=bid,lane=tid%32,wave=tid//32,d=dims,args=args,
                 att=dict(score_kv=lds.kv.ptr),put=mb.put,put_words=mb.put_words,poll=mb.poll)
        scratch = vector_address(scratch)
        for region in layout:
            c[region] = sreg(scratch,layout[region][0],region)
        for task in range(first_task(bid,0),2*per,BLOCKS):
            stage_attn_score_c1(c,task,per)
    @launcher(name)
    def launch(q:Int64,pool:Int64,sel:Int64,sbound:Int64,kmeta:Int64,
               scratch:Int64,diag:Int64,layer:Int64,config:Int64,stream:fx.Stream):
        kernel(q,pool,sel,sbound,kmeta,scratch,diag,layer,config).launch(
            grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
