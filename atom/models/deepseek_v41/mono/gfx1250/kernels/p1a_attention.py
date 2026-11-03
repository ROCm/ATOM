# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CT1-P1A: single-pass split-K attention inside the complete K2 (replaces P40's
score tasks and PV half of the IRQ tasks). Standalone kernel and its acceptance:
``fused_attn.py`` / ``result.md`` in this directory.

stage_fused(c, task):  task = (token t, split s). Waits for t's selection (as the
    original score/PV did), resolves its keys with the original ``_key_row`` order,
    one TDM read a KV row into LDS, QK + online softmax + PV, then writes FP32
    partial O [16, 512] and m, l [16] to the P1A state tail and publishes a tagged
    ready pair (store drain, CTA join, one lane: the original ``finish_irq_task``
    publication shape).
stage_irq_p1a(c, task, groups): the original IRQ task (token, 4 column groups of
    32): polls the token's S ready pairs, merges the splits and applies the sink
    exactly as the original ``pv_group`` finalization into the original PV ``out``
    LDS layout, then the original ``raw_irq_group`` per group and the original
    ``finish_irq_task``. WO and everything after it are unchanged.

State tail (appended after the original STATE_BYTES; original offsets unchanged):
  P1A_READY: tokens*S tagged pairs; P1A_ML: tokens*S*16 (m, l); P1A_O: FP32 partial O.
"""
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo

from atom.models.deepseek_v41.mono.gfx1250.kernels import dspark_mhtjstest_state as state
from atom.models.deepseek_v41.mono.kernels.attention import (
    COLS, GROUPS, HEAD_DIM, HEAD_TILE, LOG2E, NEG, _key_row, _keys, await_selection, tile_task,
)
from atom.models.deepseek_v41.mono.kernels.attn_post import irq_rope_loads
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import _wmma
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_irq import raw_irq_group as irq_group
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_handoff import finish_irq_task
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_handoff import raw_address
from atom.mono.device.ops import CM_DEV, bf16_round, hw_exp2, readlane, rsrc, traced
from atom.mono.device.sync import sreg

S = 5                         # key splits a token: 48*5 = 240 tasks, one round of 256 CTAs
TOKENS = 48
HEADS = 16
TILE = 64
ROW = HEAD_DIM // 2 + 4
KMAX = 1024
MAXT = -(-(KMAX // TILE) // S)
assert HEAD_TILE == HEADS and HEAD_DIM == 512 and COLS * 4 == 128

P1A_READY_OFFSET = state.STATE_BYTES
P1A_ML_OFFSET = P1A_READY_OFFSET + TOKENS * S * 8
P1A_O_OFFSET = P1A_ML_OFFSET + TOKENS * S * HEADS * 8
P1A_STATE_BYTES = P1A_O_OFFSET + TOKENS * S * HEADS * HEAD_DIM * 4
P1A_STATE_WORDS = P1A_STATE_BYTES // 4
LDS_WORDS = dict(kv=2 * TILE * ROW, red=8 * 32 * 8, p=HEADS * TILE // 2, mx=4 * HEADS, ls=4 * HEADS)


def _sreg(v):
    return fx.Int32(llvm.inline_asm(T.i32, [fx.Int32(v).ir_value()], "s_mov_b32 $0, $1", "=s,s",
                                    has_side_effects=True))


def _ready(c):
    return sreg(raw_address(c, 0), P1A_READY_OFFSET, 'p1a_ready')


@traced
def stage_fused(c, task):
    lane, wave, a, L = c["lane"], c["wave"], c["args"], c["p1a"]
    half, hl = lane // 16, lane % 16
    t, s = task // S, task % S
    await_selection(c, t)
    n_keys, meta = _keys(a, t)
    n_tiles = (n_keys + TILE - 1) // TILE
    per = (n_tiles + S - 1) // S
    first = s * per
    count = fx.max(fx.min(per, n_tiles - first), 0)
    scale = fx.Int32(a["qk_scale"]).bitcast(fx.Float32)
    sub, kh = wave % 4, wave // 4

    qws = []
    for ch in range_constexpr(8):
        at = ((t * HEADS + hl) * HEAD_DIM + (kh * 8 + ch) * 32 + half * 16) // 2
        parts = [fx.Vector(bo.buffer_load(rsrc(a["q"]), at + p * 4, vec_width=4, dtype=T.i32)) for p in range(2)]
        qws.append(fx.Vector.from_elements([parts[p][i] for p in range(2) for i in range(4)], fx.Int32))

    def issue(tile_index, buf):
        key = fx.min((first + tile_index) * TILE + wave * 8 + fx.min(lane, 7), n_keys - 1)
        row = _key_row(a, t, meta, key)
        for r in range_constexpr(8):
            rr = _sreg(readlane(row, r))
            pointer = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16),
                                  a["pool"] + fx.Int64(rr) * HEAD_DIM * 2)
            source = fx.Tensor(fx.make_view(pointer, fx.make_layout(HEAD_DIM // 2, 1)))
            target = fx.Tensor(fx.make_view(L["kv"] + buf * TILE * ROW + (wave * 8 + r) * ROW,
                                            fx.make_layout(HEAD_DIM // 2, 1)))
            atom = fx.rocdl.make_tdm_atom(source, [HEAD_DIM // 2], num_warps=1)
            fx.copy(atom, source, target)

    m_run = fx.Float32(NEG)
    l_run = fx.Float32(0.)
    acc = [fx.Vector.filled(8, 0., fx.Float32) for _ in range(4)]
    if count > 0:
        issue(0, 0)
    for it in range_constexpr(MAXT):
        buf = it % 2
        if it < count:
            kv = L["kv"] + buf * TILE * ROW
            rocdl.s_wait_loadcnt(0)
            tdm_ops.tensor_wait(0)
            gpu.barrier()
            if const_expr(it + 1 < MAXT):
                if it + 1 < count:
                    issue(it + 1, 1 - buf)
            part = fx.make_rmem_tensor(8, fx.Float32)
            part.store(fx.full(8, 0., fx.Float32))
            for ch in range_constexpr(8):
                at = kv + (sub * 16 + hl) * ROW + (kh * 8 + ch) * 16 + half * 8
                kw = fx.Vector(fx.ptr_load(at, result_type=fx.Vector.make_type(8, fx.Int32)))
                _wmma(part, qws[ch].bitcast(fx.BFloat16), kw.bitcast(fx.BFloat16))
            pv = part.load()
            fx.ptr_store(fx.Vector.from_elements([pv[i] for i in range(8)], fx.Float32),
                         L["red"] + (wave * 32 + lane) * 8)
            gpu.barrier()
            base = (first + it) * TILE + sub * 16 + half * 8
            sv = [fx.Float32(NEG) for _ in range(8)]
            if wave < 4:
                other = fx.Vector(fx.ptr_load(L["red"] + ((wave + 4) * 32 + lane) * 8,
                                              result_type=fx.Vector.make_type(8, fx.Float32)))
                sv = [fx.Float32((base + i < n_keys).select((pv[i] + other[i]) * scale, fx.Float32(NEG)))
                      for i in range(8)]
                mloc = sv[0]
                for i in range_constexpr(1, 8):
                    mloc = fx.max(mloc, sv[i])
                mloc = fx.max(mloc, mloc.shuffle_xor(16, 32))
                if lane < 16:
                    fx.ptr_store(mloc, L["mx"] + wave * HEADS + hl)
            gpu.barrier()
            m_tile = fx.ptr_load(L["mx"] + hl)
            for w in range_constexpr(1, 4):
                m_tile = fx.max(m_tile, fx.ptr_load(L["mx"] + w * HEADS + hl))
            m_new = fx.max(m_run, m_tile)
            alpha = hw_exp2(m_run - m_new)
            if wave < 4:
                probs = [hw_exp2(v - m_new) for v in sv]
                psum = ((probs[0] + probs[2]) + (probs[1] + probs[3])) + ((probs[4] + probs[6]) + (probs[5] + probs[7]))
                psum = psum + psum.shuffle_xor(16, 32)
                if lane < 16:
                    fx.ptr_store(psum, L["ls"] + wave * HEADS + hl)
                packed = fx.Vector.from_elements(probs, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                fx.ptr_store(packed, L["p"] + hl * (TILE // 2) + (sub * 16 + half * 8) // 2)
            gpu.barrier()
            tile_sum = ((fx.ptr_load(L["ls"] + hl) + fx.ptr_load(L["ls"] + 2 * HEADS + hl))
                        + (fx.ptr_load(L["ls"] + HEADS + hl) + fx.ptr_load(L["ls"] + 3 * HEADS + hl)))
            l_run = l_run * alpha + tile_sum
            m_run = m_new
            acc = [fx.Vector.from_elements([acc[j][i] * alpha for i in range(8)], fx.Float32) for j in range(4)]
            for ks in range_constexpr(TILE // 32):
                aw = fx.Vector(fx.ptr_load(L["p"] + hl * (TILE // 2) + ks * 16 + half * 8,
                                           result_type=fx.Vector.make_type(8, fx.Int32)))
                for j in range_constexpr(4):
                    col_words = (wave * 64 + j * 16) // 2
                    loads = []
                    for blk in range_constexpr(2):
                        pointer = fx.to_llvm_ptr(kv + ((ks * 32 + blk * 16 + 8 * half + lane % 8) * ROW
                                                       + (lane // 8 % 2) * 4 + col_words))
                        loads.append(fx.Vector(rocdl.ds_load_tr16_b128(T.vec(8, T.bf16), pointer)).bitcast(fx.Int32))
                    o0 = [loads[0][e].shuffle_xor(16, 32) for e in range(4)]
                    o1 = [loads[1][e].shuffle_xor(16, 32) for e in range(4)]
                    bw = fx.Vector.from_elements(
                        [(half == 0).select(loads[0][e], o1[e]) for e in range(4)] +
                        [(half == 0).select(o0[e], loads[1][e]) for e in range(4)], fx.Int32)
                    cacc = fx.make_rmem_tensor(8, fx.Float32)
                    cacc.store(acc[j])
                    _wmma(cacc, aw.bitcast(fx.BFloat16), bw.bitcast(fx.BFloat16))
                    acc[j] = cacc.load()
    base_o = raw_address(c, P1A_O_OFFSET)
    for j in range_constexpr(4):
        col = wave * 64 + j * 16 + half * 8
        ptr = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16),
                          base_o + fx.Int64(((task * HEADS + hl) * HEAD_DIM + col) * 4))
        fx.ptr_store(fx.Vector.from_elements([acc[j][i] for i in range(8)], fx.Float32), ptr)
    if (wave == 0) & (lane < 16):
        bo.buffer_store(fx.Vector.from_elements([m_run, l_run], fx.Float32),
                        rsrc(raw_address(c, P1A_ML_OFFSET)), (task * HEADS + hl) * 2)
    # The original task publication: every wave's stores retire, then one lane.
    state.drain()
    if c["tid"] == 0:
        c["put"](_ready(c), task, 1)


@traced
def stage_irq_p1a(c, task, groups):
    assert groups == 4 and c["S"] == TOKENS and c["d"].head_tiles.count == 1
    tid, wave, a, att = c["tid"], c["wave"], c["args"], c["att"]
    t, tile, g0 = tile_task(task, GROUPS // groups, c["d"].head_tiles)
    ropes = irq_rope_loads(c, t, [g0 * groups + i for i in range(groups)])
    if wave == 0:
        c["poll"]([(_ready(c), t * S + s, 1) for s in range(S)])
    gpu.barrier()
    h = tid // 16
    col = (tid % 16) * 8                         # within this task's 128 columns
    ml = raw_address(c, P1A_ML_OFFSET)
    # Another CTA's plain stores, published by its drained tagged pair: read at device
    # scope (past this CU's non-coherent caches), as the original hand-offs' payloads are.
    pairs = [fx.Vector(bo.buffer_load(rsrc(ml), ((t * S + s) * HEADS + h) * 2, vec_width=2, dtype=T.f32,
                                      cache_modifier=CM_DEV)) for s in range(S)]
    ms = [fx.Float32(pairs[s][0]) for s in range(S)]
    ls = [fx.Float32(pairs[s][1]) for s in range(S)]
    mx = ms[0]
    for s in range_constexpr(1, S):
        mx = fx.max(mx, ms[s])
    w = [hw_exp2(ms[s] - mx) for s in range(S)]
    total = [fx.Float32(0.) for _ in range(8)]
    combined = fx.Float32(0.)
    base_o = raw_address(c, P1A_O_OFFSET)
    for s in range_constexpr(S):
        at = ((t * S + s) * HEADS + h) * HEAD_DIM + g0 * 128 + col
        halves = [fx.Vector(bo.buffer_load(rsrc(base_o), at + q * 4, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV))
                  for q in range(2)]
        o = [halves[i // 4][i % 4] for i in range(8)]
        total = [total[i] + o[i] * w[s] for i in range(8)]
        combined = combined + ls[s] * w[s]
    sink_value = fx.Float32(bo.buffer_load(rsrc(a["sink"]), h, vec_width=1, dtype=T.f32))
    final_m = fx.max(mx, sink_value * fx.Float32(LOG2E))
    alpha_kv = hw_exp2(mx - final_m)
    alpha_sink = hw_exp2(sink_value * fx.Float32(LOG2E) - final_m)
    final_l = combined * alpha_kv + alpha_sink
    group, gcol = col // COLS, col % COLS
    for i in range_constexpr(8):
        value = (total[i] * alpha_kv) / fx.max(final_l, fx.Float32(1e-30))
        value = fx.Float32((final_l > fx.Float32(0.)).select(value, fx.Float32(0.)))
        fx.ptr_store(bf16_round(value), att["out"] + ((group * HEAD_TILE + h) * COLS + gcol + i))
    gpu.barrier()
    for i in range_constexpr(groups):
        irq_group(c, t, tile, g0 * groups + i, att["out"] + i * HEAD_TILE * COLS, ropes[i])
        if const_expr(i + 1 < groups):
            gpu.barrier()
    finish_irq_task(c, task)
