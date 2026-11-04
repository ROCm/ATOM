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
from atom.mono.device.ops import lane_gather, bf16_pair


@traced
def finish_pair(root, co, task0, rowbase):
    tid = co["tid"]
    row, cc = tid // 16, 2 * (tid % 16)
    global_row = rowbase + row
    epoch = scalar_word(fx.ptr_load(root["late_meta"] + 219))
    gang = rowbase // 16
    rstd = sreg(late_pointer(root, 220), RSTD_OFFSET, "mhc_sq")
    # Both complete N32 partial sets and their payloads precede this ticket.
    drain()
    if tid == 0:
        last = arrive(root, MHC_COUNTS_OFFSET, ROW_BLOCKS, epoch, gang) == 79
        fx.ptr_store(last.to(fx.Int32), co["red"])
    gpu.barrier()
    if fx.ptr_load(co["red"]) != 0:
        for begin in range_constexpr(0, 16, 8):
            token = rowbase + begin + tid // 32
            total = norm_square_total(co, token)
            r = hw_rsq(fma(total, fx.Float32(1. / HIDDEN), fx.Float32(EPS)))
            if tid % 32 == 0:
                co["put"](rstd, token, r)
    (rr,) = co["poll"]([(rstd, global_row, 1)])
    r = rr[0].bitcast(fx.Float32)
    norm = seam_context(root, "norm")
    cb = x_context(root)
    cf = x_context(root)
    for name in ("x8", "x8s", "x8rdy"):
        cf[name] = sreg(late_pointer(root, 24), root["scratch_layout"][name][0], name)
    for half in range_constexpr(2):
        task = task0 + half
        ys = []
        for d in range_constexpr(2):
            x = fx.ptr_load(root["split_x"] + (half * 16 + row) * COLS + cc + d)
            w = ld_bf(norm["args"]["attn_w"], task * COLS + cc + d)
            ys.append(bf16_round((x * r) * w))
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
        last = arrive(root, MHC_COUNTS_OFFSET, ROW_BLOCKS, epoch, gang) == 159
        fx.ptr_store(last.to(fx.Int32), co["red"])
    gpu.barrier()
    if fx.ptr_load(co["red"]) != 0:
        if tid < 16:
            cb["put"](cb["xrdy"], rowbase + tid, 1)
            cf["put"](cf["x8rdy"], rowbase + tid, 1)
    gpu.barrier()


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
            reduce_attn_pair(co, task, co["tid"] + rowbase * 16)
            gpu.barrier()
            stage_tile(local, task, pre)
        finish_pair(root, local, task0, rowbase)
