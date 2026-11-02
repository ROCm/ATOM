"""240 resident paired-N32 producers, all-row TP1 and in-place norm/quant.

Every producer computes both N32 splits before waiting. Each 16-row gang has
80 producers, so no retained LDS input waits for an undispatched split.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhc_dataflow_seam import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_mhc import row_view
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_mhc import tile_prefetch
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_mhc import stage_tile
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_mhc import norm_square_total
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_moe_front import st_dev
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_moe_front import rcp_pow2
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_state import *
from atom.mono.device.ops import lane_gather, bf16_pair, bf2_f32, rsrc
from atom.mono.device.sync import _noted, _spin, preg
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_wo import ATTN_GROUPS, attn_group_word
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1


@traced
def finish_pair(root, co, task0, rowbase):
    tid = co["tid"]
    row, cc = tid // 16, 2 * (tid % 16)
    global_row = rowbase + row
    epoch = scalar_word(fx.ptr_load(root["late_meta"] + 219))
    gang = rowbase // 16
    norm = seam_context(root, "norm")
    # Norm weights first: one packed bf16 pair per half, in flight during the rstd wait.
    wpairs = [fx.Int32(bo.buffer_load(rsrc(norm["args"]["attn_w"]), ((task0 + half) * COLS + cc) // 2,
                                      vec_width=1, dtype=T.i32)) for half in range(2)]
    gpu.barrier()
    for begin in range_constexpr(0, 16, 8):
        token = rowbase + begin + tid // 32
        total = norm_square_total(co, token)
        r8 = hw_rsq(fma(total, fx.Float32(1. / HIDDEN), fx.Float32(EPS)))
        if tid % 32 == 0:
            fx.ptr_store(r8, co["red"] + 64 + begin + tid // 32)
    gpu.barrier()
    r = fx.ptr_load(co["red"] + 64 + row)
    cb = x_context(root)
    cf = x_context(root)
    for name in ("x8", "x8s", "x8rdy"):
        cf[name] = sreg(late_pointer(root, 24), root["scratch_layout"][name][0], name)
    for half in range_constexpr(2):
        task = task0 + half
        ws = bf2_f32(wpairs[half])
        ys = []
        for d in range_constexpr(2):
            x = fx.ptr_load(root["split_x"] + (half * 16 + row) * COLS + cc + d)
            ys.append(bf16_round((x * r) * ws[d]))
        amax = fx.max(abs(ys[0]), abs(ys[1]))
        for off in (1, 2, 4, 8):
            amax = fx.max(amax, amax.shuffle_xor(off, 32))
        code = code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1. / FP8_MAX))
        inv = rcp_pow2(code)
        q0, q1 = [clamp_fp8(y * inv) for y in ys]
        lane = tid % 32
        q2 = lane_gather(q0.bitcast(fx.Int32), fx.min(lane + 1, 31)).bitcast(fx.Float32)
        q3 = lane_gather(q1.bitcast(fx.Int32), fx.min(lane + 1, 31)).bitcast(fx.Float32)
        word = fp8_pack4(q0, q1, q2, q3)
        col = task * COLS + cc
        for d in range_constexpr(2):
            bo.buffer_store(ys[d].to(fx.BFloat16), rsrc(norm["args"]["normed"]),
                            global_row * HIDDEN + col + d, cache_modifier=CM_DEV)
        if tid % 2 == 0:
            st_dev(word, cf["x8"], global_row * X8_WORDS + col // 4)
        if tid % 16 == 0:
            st_dev(code, cf["x8s"], global_row * (HIDDEN // 32) + task)
    drain()
    if tid == 0:
        last = arrive(root, MHC_COUNTS_OFFSET, ROW_BLOCKS, epoch, gang) == 79
        fx.ptr_store(last.to(fx.Int32), co["red"])
    gpu.barrier()
    if fx.ptr_load(co["red"]) != 0:
        if tid < 16:
            cb["put"](cb["xrdy"], rowbase + tid, 1)
            cf["put"](cf["x8rdy"], rowbase + tid, 1)
    gpu.barrier()


@traced
def reduce_attn_pair_packed(c, task, idx):
    s = c['S']
    c0 = task * k1.COLS
    if idx < s * k1.COLS // 2:
        t = idx // (k1.COLS // 2)
        col = c0 + 2 * (idx % (k1.COLS // 2))
        own = preg(c['sym'], c['peer']['attn'], 'attn')
        mb = c['poll'].__self__
        cm = mb.scope_cm[own.space]
        base = _noted(own, 'poll')
        block = col // 64
        pair = (col % 64) // 2
        g = fx.min(pair // 3, ATTN_GROUPS - 1)
        k = pair - g * 3
        tag = fx.Int32(mb.tag)

        def load_all():
            words = []
            for src in range_constexpr(4):
                w = fx.Vector(bo.buffer_load(rsrc(base), attn_group_word(src, t, s, block, g),
                                             vec_width=4, dtype=T.i32, cache_modifier=cm))
                words += [w[e] for e in range(4)]
            return fx.Vector.from_elements(words, fx.Int32)

        def pending(v):
            bad = v[3] != tag
            for src in range_constexpr(1, 4):
                bad = bad | (v[4 * src + 3] != tag)
            return bad

        v = _spin(load_all, pending)
        acc0 = fx.Float32(0.)
        acc1 = fx.Float32(0.)
        for src in range_constexpr(4):
            w = (k == 0).select(v[4 * src], (k == 1).select(v[4 * src + 1], v[4 * src + 2]))
            lo, hi = bf2_f32(w)
            if const_expr(src == 0):
                acc0, acc1 = lo, hi
            else:
                acc0 = acc0 + lo
                acc1 = acc1 + hi
        lds = c['pend_lds'] + (t * k1.COLS + col - c0)
        fx.ptr_store(bf16_round(acc0), lds)
        fx.ptr_store(bf16_round(acc1), lds + 1)

@traced
def run_seam(root, key, bid):
    assert key.tokens == 48 and not key.tok_par
    unit = first_task(bid, 0)
    enter_stage("k2a.paired_split")
    if unit < 240:
        rowbase = (unit // 80) * 16
        task0 = (unit % 80) * 2
        co = seam_context(root, "slice")
        for half in range_constexpr(2):
            task = task0 + half
            local = row_view(co, rowbase)
            local["split_x"] = root["split_x"] + half * 16 * COLS
            pre = tile_prefetch(local, task)
            reduce_attn_pair_packed(co, task, co["tid"] + rowbase * 16)
            gpu.barrier()
            stage_tile(local, task, pre)
        finish_pair(root, local, task0, rowbase)
