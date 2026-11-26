# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: ATOM extraction and explicit BF16 NoPE QAT profile, 2026-10-13.
# FlyDSL traces closures and constexpr guards during kernel construction.
# ruff: noqa: B023, SIM102

"""Attention input projections of the DeepSeek-V4 MonoKernel: q_a / kv, the KV ring and
compressor, and q_b."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v4.mono.config import EPS, FP8_MAX, ROPE_DIM
from atom.models.deepseek_v4.mono.kernels import buffer_ops as bo
from atom.models.deepseek_v4.mono.kernels.common import pow2_ceil
from atom.models.deepseek_v4.mono.kernels.helpers import traced
from atom.models.deepseek_v4.mono.kernels.plan import (
    Q_B_TILE,
    QKV_A_TILE,
    THREADS,
    WAVES,
    q_b_groups,
    qkv_a_groups,
)

NEG = -1.0e30
from atom.models.deepseek_v4.mono.kernels.primitives import (
    bf2_f32,
    bf16_round,
    exp,
    ld_bf16,
    ld_f32,
    lds_ld,
    lds_st,
    rcp,
    read_lane_i32,
    rsq,
    rsrc,
    wave_max,
    wave_sum,
    xred,
    xshfl,
)


def kv_quant(nv):
    """One KV channel through ATOM's NoPE FP8 QAT round trip.

    Kept at module scope because the split attention-core launcher reuses the
    exact helper without replaying qkv/cache producer stages.
    """
    amax = wave_max(fmath.absf(nv))
    sc = pow2_ceil(fx.max(amax, fx.Float32(FP8_MAX * 2.0**-126)) * (1.0 / FP8_MAX))
    q = fx.min(fx.max(nv * rcp(sc), -FP8_MAX), FP8_MAX)
    word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, q, fx.Int32(0), False))
    v2 = fx.Vector.make_type(2, fx.Float32)
    d = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))[0]
    return d * sc, word & 0xFF, (sc.bitcast(fx.Int32) >> 23) & 0xFF


def draft_bf16_kv_quant(nv):
    """Native DSpark BF16 NoPE QAT, including materialized BF16 seams.

    Native first stores the QK result as BF16, then performs each PyTorch
    fake-quant operation in that dtype. This is not the FP32 packed-cache
    quantizer above; keep it specific to the BF16 draft arithmetic profile.
    """
    nv = bf16_round(nv)
    amax = bf16_round(fx.max(wave_max(fmath.absf(nv)), bf16_round(fx.Float32(1.0e-4))))
    ratio = bf16_round(amax * (1.0 / FP8_MAX))
    exponent = fmath.ceil(bf16_round(fmath.log2(ratio)))
    scale = bf16_round(fmath.exp2(exponent))
    q = bf16_round(nv * rcp(scale))
    q = fx.min(fx.max(q, -FP8_MAX), FP8_MAX)
    word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, q, fx.Int32(0), False))
    v2 = fx.Vector.make_type(2, fx.Float32)
    d = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))[0]
    return bf16_round(d * scale)


@traced
def compressor_helpers(ctx):
    """Return compressor ring/pooling helpers without emitting producer stages.

    The ONE kernel and split attention phases share these exact closures.  Keeping
    helper construction separate lets a query/index phase consume qkv mailboxes
    without replaying cache or compressor writes.
    """
    CMP_CHUNK = ctx["CMP_CHUNK"]
    CMP_CHUNK_T = ctx["CMP_CHUNK_T"]
    C_RING = ctx["C_RING"]
    C_ROWS = ctx["C_ROWS"]
    TOK = ctx["TOK"]

    def pool_fold(acc, svs, kvs):
        m, den, num = acc
        for e in range_constexpr(len(svs)):
            m_new = fx.max(m, svs[e])
            rescale = exp(m - m_new)
            w = exp(svs[e] - m_new)
            den = den * rescale + w
            num = num * rescale + w * kvs[e]
            m = m_new
        return [m, den, num]

    def window_pool(p, j_tok, ring, launch, own):
        """Online softmax over the compressor window ending at ``p``."""
        n_ring = C_ROWS if const_expr(TOK == 1) else C_ROWS - TOK
        chunk = CMP_CHUNK if const_expr(TOK == 1) else CMP_CHUNK_T
        for _i, acc in range(
            0,
            n_ring // chunk,
            fx.Int32(1),
            init=[fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)],
        ):
            ib = fx.Int32(_i) * chunk
            lds_ = [ring(ib + e) for e in range(chunk)]
            res = yield pool_fold(
                [fx.Float32(acc[0]), fx.Float32(acc[1]), fx.Float32(acc[2])],
                [a for a, _ in lds_],
                [b for _, b in lds_],
            )
        acc = [fx.Float32(res[0]), fx.Float32(res[1]), fx.Float32(res[2])]
        if const_expr(TOK > 1):
            svs, kvs = [], []
            for i in range_constexpr(C_ROWS - TOK, C_ROWS):
                d = C_ROWS - 1 - i
                if const_expr(d == 0):
                    sv, kv = own(i)
                else:
                    r_sv, r_kv = ring(fx.Int32(i))
                    l_sv, l_kv = launch(d, i)
                    inl = fx.Int32(d) <= j_tok
                    sv, kv = inl.select(l_sv, r_sv), inl.select(l_kv, r_kv)
                svs.append(sv)
                kvs.append(kv)
            acc = pool_fold(acc, svs, kvs)
        return acc[1], acc[2]

    def split_window_pool(j_tok, ring, launch, own, splits):
        """Match native main-cache K-split pooling, retaining packed inputs.

        Each partial folds a contiguous window interval in source order;
        partials merge in increasing native wave order. Launch rows must
        still come from mailboxes rather than concurrently updated state.
        """
        parts = []
        width = C_ROWS // splits
        for w in range_constexpr(splits):
            start, stop = w * width, (w + 1) * width
            ring_stop = min(stop, max(start, C_ROWS - TOK))
            for i, acc in range(
                start,
                ring_stop,
                fx.Int32(1),
                init=[fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)],
            ):
                sv, kv = ring(fx.Int32(i))
                res = yield pool_fold(
                    [fx.Float32(acc[0]), fx.Float32(acc[1]), fx.Float32(acc[2])],
                    [sv],
                    [kv],
                )
            acc = [fx.Float32(res[0]), fx.Float32(res[1]), fx.Float32(res[2])]
            for i in range_constexpr(ring_stop, stop):
                d = C_ROWS - 1 - i
                if const_expr(d == 0):
                    sv, kv = own(i)
                else:
                    r_sv, r_kv = ring(fx.Int32(i))
                    l_sv, l_kv = launch(d, i)
                    inl = fx.Int32(d) <= j_tok
                    sv, kv = inl.select(l_sv, r_sv), inl.select(l_kv, r_kv)
                acc = pool_fold(acc, [sv], [kv])
            parts.append(acc)
        m = fx.Float32(NEG)
        for w in range_constexpr(splits):
            m = fx.max(m, parts[w][0])
        den, num = fx.Float32(0.0), fx.Float32(0.0)
        for w in range_constexpr(splits):
            scale = exp(parts[w][0] - m)
            num = num + parts[w][2] * scale
            den = den + parts[w][1] * scale
        return num * rcp(den)

    def ring_row(p, i):
        """State-ring row of window element i (position p + 1 - C_ROWS + i)."""
        return (p + 1 + i + (C_RING - C_ROWS)) % C_RING

    return {
        "pool_fold": pool_fold,
        "ring_row": ring_row,
        "window_pool": window_pool,
        "split_window_pool": split_window_pool,
    }


@traced
def qkv_handoff_stages(ctx):
    """QKV projections fed by one tagged, packed-BF16 normalized-input leg."""
    CW = ctx["CW"]
    G = ctx["G"]
    HC = ctx["HC"]
    HEAD_DIM = ctx["HEAD_DIM"]
    HIDDEN = ctx["HIDDEN"]
    IW = ctx["IW"]
    N_QKV_A = ctx["N_QKV_A"]
    N_QKV_C = ctx["N_QKV_C"]
    QKV_A_ROWS = ctx["QKV_A_ROWS"]
    Q_LORA = ctx["Q_LORA"]
    S = ctx["S"]
    _rmsnorm_tail_ks = ctx["_rmsnorm_tail_ks"]
    attention_fp8_bf16 = ctx["attention_fp8_bf16"]
    compressor_projection_bf16 = ctx["compressor_projection_bf16"]
    emit_out = ctx["emit_out"]
    g_in = ctx["g_in"]
    get2_many = ctx["get2_many"]
    lane = ctx["lane"]
    lds = ctx["xs"]
    load_x_rmsnorm = ctx["load_x_rmsnorm"]
    mb = ctx["mb"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    put = ctx["put"]
    put_bf = ctx["put_bf"]
    qkv_batch_limit = ctx["qkv_batch_limit"]
    qkv_samples_per_pass = ctx["qkv_samples_per_pass"]
    r_h = ctx["r_h"]
    red = ctx["red"]
    reduce_rows = ctx["reduce_rows"]
    run_qkv_a = ctx["run_qkv_a"]
    run_qkv_c = ctx["run_qkv_c"]
    run_units = ctx["run_units"]
    s_qkv_a = ctx["s_qkv_a"]
    stage_x_pairs = ctx["stage_x_pairs"]
    stage_x_quant_e8m0 = ctx["stage_x_quant_e8m0"]
    stage_x_rmsnorm = ctx["stage_x_rmsnorm"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16 = ctx["unit_bf16"]
    unit_fp8 = ctx["unit_fp8"]
    w_qkv_a = ctx["w_qkv_a"]
    w_qkv_c = ctx["w_qkv_c"]
    wave = ctx["wave"]

    r_wqa, r_sqa = rsrc(w_qkv_a), rsrc(s_qkv_a)
    r_wqc = rsrc(w_qkv_c)
    QA_NKC = HIDDEN // 64

    def norm_inputs(sample0):
        def ld_h(sks):
            if const_expr(HC > 1):
                vals = poll(
                    [(mb("xin"), ((sample0 + s) * HIDDEN + k) // 2, 2) for s, k in sks]
                )
                res = []
                for i in range_constexpr(len(sks)):
                    a0, a1 = bf2_f32(vals[i][0])
                    b0, b1 = bf2_f32(vals[i][1])
                    res.append([a0, a1, b0, b1])
                return res
            res = []
            for s, k in sks:
                w = fx.Vector(
                    bo.buffer_load(
                        r_h,
                        ((sample0 + s) * HIDDEN + k) // 2,
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                res.append([v[j] for j in range(4)])
            return res

        h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in, count=1)
        norm_loaded = None
        rstds = None
        if const_expr(attention_fp8_bf16 and HC > 1):
            if const_expr(HC == 4 and HIDDEN == 7168):
                ss = fx.Float32(0.0)
                if (tid < 256) & (tid % 4 == 0):
                    group = tid // 4
                    raw = get2_many(
                        [
                            (
                                mb("xin_raw"),
                                sample0 * HIDDEN + i * 1024 + group * 16 + j,
                            )
                            for i in range(7)
                            for j in range(0, 16, 2)
                        ]
                    )
                    for pair in raw:
                        for a in pair:
                            ss = ss + a * a
                wave_ss = wave_sum(ss)
                if (lane == 63) & (wave < 4):
                    lds_st(red, wave, wave_ss)
                gpu.barrier()
                part = (lane < 4).select(lds_ld(red, lane), fx.Float32(0.0))
                for off in (1, 2):
                    part = xred(part, off, lambda a, b: a + b)
                total = read_lane_i32(part.bitcast(fx.Int32), 3).bitcast(fx.Float32)
                rstds = [rsq(total * (1.0 / HIDDEN) + EPS)]
                gpu.barrier()
                norm_loaded = []
            else:
                ks, _ = _rmsnorm_tail_ks(HIDDEN)
                raw = get2_many(
                    [
                        (mb("xin_raw"), sample0 * HIDDEN + k + j)
                        for k in ks
                        for j in (0, 2)
                    ]
                )
                norm_loaded = [
                    list(raw[2 * i]) + list(raw[2 * i + 1]) for i in range(len(ks))
                ]
        return ld_h, h_ld, norm_loaded, rstds

    if const_expr(run_qkv_a or run_qkv_c):
        for ss in range(start("qkv_norm"), S, G):
            ss = fx.Int32(ss)
            stamp("qkv_norm", ss, 0)
            ld_h, h_ld, norm_loaded, rstds = norm_inputs(ss)
            stage_x_rmsnorm(
                ld_h,
                HIDDEN,
                g_in,
                loaded=h_ld,
                count=1,
                norm_loaded=norm_loaded,
                rstds=rstds,
            )
            gpu.barrier()
            for i in range_constexpr((HIDDEN // 2 + THREADS - 1) // THREADS):
                word = tid + i * THREADS
                if word < HIDDEN // 2:
                    a0, a1 = bf2_f32(lds_ld(lds, word).bitcast(fx.Int32))
                    put_bf(mb("qkv_x"), ss * HIDDEN + 2 * word, [a0, a1])
            if const_expr(attention_fp8_bf16):
                stage_x_quant_e8m0(HIDDEN)
                for i in range_constexpr((HIDDEN // 2 + THREADS - 1) // THREADS):
                    word = tid + i * THREADS
                    if word < HIDDEN // 2:
                        a0, a1 = bf2_f32(lds_ld(lds, word).bitcast(fx.Int32))
                        put_bf(mb("qkv_xq"), ss * HIDDEN + 2 * word, [a0, a1])
            stamp("qkv_norm", ss, 3)
            stamp("qkv_norm", ss, 4)

    def qkv_stage(name, n_tiles, row0, bf16_w):
        QA_R = qkv_a_groups(n_tiles)
        QA_WPR = WAVES // QA_R
        QA_UPW = QA_NKC // QA_WPR
        default_batch = max(1, QA_NKC // WAVES // (4 if bf16_w else 1))
        QA_BATCH = (
            default_batch
            if qkv_batch_limit is None
            else min(default_batch, qkv_batch_limit)
        )
        QA_SPP = S if qkv_samples_per_pass is None else qkv_samples_per_pass
        QA_ROWS = QKV_A_TILE * QA_R
        for t in range(start(name), n_tiles // QA_R, G):
            t = fx.Int32(t)
            stamp(name, t, 0)
            qa_rg = t * QA_R + wave // QA_WPR
            for sample0 in range(0, S, QA_SPP):

                def u_qa(c):
                    kc = (wave % QA_WPR) * QA_UPW + c
                    xoff = n_sel(QA_SPP) * HIDDEN + kc * 64
                    if const_expr(bf16_w):
                        return unit_bf16(r_wqc, qa_rg, kc, QA_NKC, xoff // 2)
                    return unit_fp8(
                        r_wqa, r_sqa, qa_rg, kc, QA_NKC, HIDDEN, 128, xoff // 2
                    )

                pre = [u_qa(c) for c in range(QA_BATCH)]
                input_leg = "qkv_xq" if (not bf16_w and attention_fp8_bf16) else "qkv_x"
                stage_x_pairs(
                    input_leg, QA_SPP * HIDDEN, lambda k: sample0 * HIDDEN + k
                )
                gpu.barrier()
                stamp(name, t, 2)
                acc = run_units(u_qa, QA_UPW, QA_BATCH, pre)
                reduce_rows(QA_R, acc, emit_out(QA_ROWS), count=QA_SPP)
                stamp(name, t, 3)
                gpu.barrier()
                if tid < QA_SPP * QA_ROWS:
                    s = sample0 + tid // QA_ROWS
                    row = row0 + t * QA_ROWS + tid % QA_ROWS
                    v = lds_ld(outs, tid)
                    if const_expr(
                        (bf16_w and compressor_projection_bf16)
                        or (not bf16_w and attention_fp8_bf16)
                    ):
                        v = bf16_round(v)
                    if row < Q_LORA:
                        put(mb("q_a"), s * Q_LORA + row, v)
                    elif row < Q_LORA + HEAD_DIM:
                        put(mb("kv_a"), s * HEAD_DIM + row - Q_LORA, v)
                    elif row < Q_LORA + HEAD_DIM + CW:
                        put(mb("c_kv"), s * CW + row - Q_LORA - HEAD_DIM, v)
                    elif row < Q_LORA + HEAD_DIM + 2 * CW:
                        put(mb("c_gate"), s * CW + row - Q_LORA - HEAD_DIM - CW, v)
                    elif row < Q_LORA + HEAD_DIM + 2 * CW + IW:
                        put(mb("i_kv"), s * IW + row - Q_LORA - HEAD_DIM - 2 * CW, v)
                    else:
                        put(
                            mb("i_gate"),
                            s * IW + row - Q_LORA - HEAD_DIM - 2 * CW - IW,
                            v,
                        )
            stamp(name, t, 4)

    if const_expr(run_qkv_a):
        qkv_stage("qkv_a", N_QKV_A, 0, False)
    if const_expr(run_qkv_c and N_QKV_C):
        qkv_stage("qkv_c", N_QKV_C, QKV_A_ROWS, True)
    return {}


@traced
def qkv_stages(ctx):
    """1-2b. q_a / kv GEMV, KV RMSNorm + RoPE -> sliding-window ring, and the KV compressor."""
    ctx["CMP_CHUNK"]
    ctx["CMP_CHUNK_T"]
    CR = ctx["CR"]
    CW = ctx["CW"]
    C_COFF = ctx["C_COFF"]
    C_RING = ctx["C_RING"]
    C_ROWS = ctx["C_ROWS"]
    G = ctx["G"]
    HC = ctx["HC"]
    HEAD_DIM = ctx["HEAD_DIM"]
    HIDDEN = ctx["HIDDEN"]
    IW = ctx["IW"]
    KV_FP8 = ctx["KV_FP8"]
    KV_QAT = ctx["KV_QAT"]
    DRAFT_BLOCK = ctx["DRAFT_BLOCK"]
    KV_ROW_BYTES = ctx["KV_ROW_BYTES"]
    NOPE_DIM = ctx["NOPE_DIM"]
    N_QKV_A = ctx["N_QKV_A"]
    N_QKV_C = ctx["N_QKV_C"]
    OVERLAP = ctx["OVERLAP"]
    QKV_A_ROWS = ctx["QKV_A_ROWS"]
    Q_LORA = ctx["Q_LORA"]
    S = ctx["S"]
    TOK = ctx["TOK"]
    ape = ctx["ape"]
    block_sum = ctx["block_sum"]
    block_sums = ctx["block_sums"]
    comp_row = ctx["comp_row"]
    emit_out = ctx["emit_out"]
    g_ckv = ctx["g_ckv"]
    g_in = ctx["g_in"]
    g_kv = ctx["g_kv"]
    getf = ctx["getf"]
    getf_many = ctx["getf_many"]
    hint_wait = ctx["hint_wait"]
    kv_cache = ctx["kv_cache"]
    kv_rope = ctx["kv_rope"]
    kv_state = ctx["kv_state"]
    lane = ctx["lane"]
    ld_dest = ctx["ld_dest"]
    ld_pos = ctx["ld_pos"]
    load_x_rmsnorm = ctx["load_x_rmsnorm"]
    _rmsnorm_tail_ks = ctx["_rmsnorm_tail_ks"]
    get2_many = ctx["get2_many"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    put = ctx["put"]
    red = ctx["red"]
    r_h = ctx["r_h"]
    reduce_rows = ctx["reduce_rows"]
    rope_cos = ctx["rope_cos"]
    rope_sin = ctx["rope_sin"]
    row_rsrc = ctx["row_rsrc"]
    run_units = ctx["run_units"]
    s_qkv_a = ctx["s_qkv_a"]
    score_state = ctx["score_state"]
    slot_rsrc = ctx["slot_rsrc"]
    input_slot_rsrc = ctx["input_slot_rsrc"]
    st_kv = ctx["st_kv"]
    st_sc = ctx["st_sc"]
    stage_x_rmsnorm = ctx["stage_x_rmsnorm"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16 = ctx["unit_bf16"]
    unit_fp8 = ctx["unit_fp8"]
    w_qkv_a = ctx["w_qkv_a"]
    w_qkv_c = ctx["w_qkv_c"]
    compressor_projection_bf16 = ctx["compressor_projection_bf16"]
    attention_fp8_bf16 = ctx["attention_fp8_bf16"]
    stage_x_quant_e8m0 = ctx["stage_x_quant_e8m0"]
    wave = ctx["wave"]
    pool_fold = ctx["pool_fold"]
    ring_row = ctx["ring_row"]
    window_pool = ctx["window_pool"]
    split_window_pool = ctx["split_window_pool"]
    run_qkv_a = ctx["run_qkv_a"]
    run_qkv_c = ctx["run_qkv_c"]
    run_attention_cache = ctx["run_attention_cache"]
    qkv_batch_limit = ctx["qkv_batch_limit"]
    qkv_samples_per_pass = ctx["qkv_samples_per_pass"]
    # ================================================= 1. q_a / kv GEMV
    r_wqa, r_sqa = rsrc(w_qkv_a), rsrc(s_qkv_a)
    r_wqc = rsrc(w_qkv_c)
    QA_NKC = HIDDEN // 64

    def qkv_stage(name, n_tiles, row0, bf16_w):
        """One fused-GEMV stage over n_tiles 16-row groups from global row row0: the
        FP8 q_a | kv rows (qkv_a) or the compressors' BF16 rows (qkv_c)."""
        QA_R = qkv_a_groups(n_tiles)
        QA_WPR = WAVES // QA_R
        QA_UPW = QA_NKC // QA_WPR
        # units in flight per wave (larger BF16 batches spill VGPRs)
        default_batch = max(1, QA_NKC // WAVES // (4 if bf16_w else 1))
        QA_BATCH = (
            default_batch
            if qkv_batch_limit is None
            else min(default_batch, qkv_batch_limit)
        )
        QA_SPP = S if qkv_samples_per_pass is None else qkv_samples_per_pass
        QA_ROWS = QKV_A_TILE * QA_R
        for t in range(start(name), n_tiles // QA_R, G):
            t = fx.Int32(t)
            stamp(name, t, 0)
            qa_rg = t * QA_R + wave // QA_WPR

            for sample0 in range(0, S, QA_SPP):

                def u_qa(c):
                    kc = (wave % QA_WPR) * QA_UPW + c
                    xoff = n_sel(QA_SPP) * HIDDEN + kc * 64
                    if const_expr(bf16_w):
                        return unit_bf16(r_wqc, qa_rg, kc, QA_NKC, xoff // 2)
                    return unit_fp8(
                        r_wqa, r_sqa, qa_rg, kc, QA_NKC, HIDDEN, 128, xoff // 2
                    )

                def ld_h(sks):
                    if const_expr(HC > 1):  # hc_pre already contracted the streams
                        vals = poll(
                            [
                                (mb("xin"), ((sample0 + s) * HIDDEN + k) // 2, 2)
                                for s, k in sks
                            ]
                        )
                        res = []
                        for i in range_constexpr(len(sks)):
                            a0, a1 = bf2_f32(vals[i][0])
                            b0, b1 = bf2_f32(vals[i][1])
                            res.append([a0, a1, b0, b1])
                        return res
                    res = []
                    for s, k in sks:
                        w = fx.Vector(
                            bo.buffer_load(
                                r_h,
                                ((sample0 + s) * HIDDEN + k) // 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append([v[j] for j in range(4)])
                    return res

                # input loads go out before the weight stream (loads complete in order)
                h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in, count=QA_SPP)
                norm_loaded = None
                rstds = None
                if const_expr(attention_fp8_bf16 and HC > 1):
                    if const_expr(HC == 4 and HIDDEN == 7168):
                        rstds = []
                        for s in range_constexpr(QA_SPP):
                            ss = fx.Float32(0.0)
                            if (tid < 256) & (tid % 4 == 0):
                                group = tid // 4
                                raw = get2_many(
                                    [
                                        (
                                            mb("xin_raw"),
                                            (sample0 + s) * HIDDEN
                                            + i * 1024
                                            + group * 16
                                            + j,
                                        )
                                        for i in range(7)
                                        for j in range(0, 16, 2)
                                    ]
                                )
                                for pair in raw:
                                    for a in pair:
                                        ss = ss + a * a
                            wave_ss = wave_sum(ss)
                            if (lane == 63) & (wave < 4):
                                lds_st(red, wave, wave_ss)
                            gpu.barrier()
                            part = (lane < 4).select(lds_ld(red, lane), fx.Float32(0.0))
                            for off in (1, 2):
                                part = xred(part, off, lambda a, b: a + b)
                            total = read_lane_i32(part.bitcast(fx.Int32), 3).bitcast(
                                fx.Float32
                            )
                            rstds.append(rsq(total * (1.0 / HIDDEN) + EPS))
                            gpu.barrier()
                        norm_loaded = []
                    else:
                        ks, _ = _rmsnorm_tail_ks(HIDDEN)
                        raw = get2_many(
                            [
                                (mb("xin_raw"), (sample0 + s) * HIDDEN + k + j)
                                for s in range(QA_SPP)
                                for k in ks
                                for j in (0, 2)
                            ]
                        )
                        norm_loaded = [
                            list(raw[2 * i]) + list(raw[2 * i + 1])
                            for i in range(QA_SPP * len(ks))
                        ]
                pre = [u_qa(c) for c in range(QA_BATCH)]
                stage_x_rmsnorm(
                    ld_h,
                    HIDDEN,
                    g_in,
                    loaded=h_ld,
                    count=QA_SPP,
                    norm_loaded=norm_loaded,
                    rstds=rstds,
                )
                if const_expr(not bf16_w and attention_fp8_bf16):
                    stage_x_quant_e8m0(QA_SPP * HIDDEN)
                gpu.barrier()
                stamp(name, t, 2)
                acc = run_units(u_qa, QA_UPW, QA_BATCH, pre)
                reduce_rows(QA_R, acc, emit_out(QA_ROWS), count=QA_SPP)
                stamp(name, t, 3)
                gpu.barrier()
                if tid < QA_SPP * QA_ROWS:
                    s = sample0 + tid // QA_ROWS
                    row = row0 + t * QA_ROWS + tid % QA_ROWS
                    v = lds_ld(outs, tid)
                    if const_expr(
                        (bf16_w and compressor_projection_bf16)
                        or (not bf16_w and attention_fp8_bf16)
                    ):
                        v = bf16_round(v)
                    if row < Q_LORA:
                        put(mb("q_a"), s * Q_LORA + row, v)
                    elif row < Q_LORA + HEAD_DIM:
                        put(mb("kv_a"), s * HEAD_DIM + row - Q_LORA, v)
                    elif row < Q_LORA + HEAD_DIM + CW:
                        put(mb("c_kv"), s * CW + row - Q_LORA - HEAD_DIM, v)
                    elif row < Q_LORA + HEAD_DIM + 2 * CW:
                        put(mb("c_gate"), s * CW + row - Q_LORA - HEAD_DIM - CW, v)
                    elif row < Q_LORA + HEAD_DIM + 2 * CW + IW:
                        put(mb("i_kv"), s * IW + row - Q_LORA - HEAD_DIM - 2 * CW, v)
                    else:
                        put(
                            mb("i_gate"),
                            s * IW + row - Q_LORA - HEAD_DIM - 2 * CW - IW,
                            v,
                        )
            stamp(name, t, 4)

    if const_expr(run_qkv_a):
        qkv_stage("qkv_a", N_QKV_A, 0, False)
    if const_expr(run_qkv_c and N_QKV_C):
        qkv_stage("qkv_c", N_QKV_C, QKV_A_ROWS, True)

    def put_kv_row(row, kvn, byte, e8):
        """Write one KV row (thread = channel; ``kvn`` its value, ``byte`` / ``e8``
        from kv_quant). CTA-uniform: the fp8 layout trades scale bytes through LDS."""
        if const_expr(KV_FP8):
            r_nope, r_rope = (
                row_rsrc(kv_cache, row, KV_ROW_BYTES),
                row_rsrc(kv_rope, row, ROPE_DIM * 2),
            )
            # four lanes' FP8 bytes to one dword, channel tid in byte tid % 4
            wb = byte << ((tid % 4) * 8)
            for off in (1, 2):
                wb = xred(wb, off, lambda a, b: a | b)
            if (tid < NOPE_DIM) & (tid % 4 == 0):
                bo.buffer_store(wb, r_nope, tid // 4)
            if tid >= NOPE_DIM:
                bo.buffer_store(kvn.to(fx.BFloat16), r_rope, tid - NOPE_DIM)
            if (lane == 0) & (tid < NOPE_DIM):
                lds_st(misc, wave, e8.bitcast(fx.Float32))
            gpu.barrier()
            # scale dword k: groups 2k and 2k + 1, each byte twice
            NG = NOPE_DIM // 64
            if tid < (NG + 1) // 2:
                lo = lds_ld(misc, fx.min(2 * tid, fx.Int32(NG - 1))).bitcast(fx.Int32)
                hi = (2 * tid + 1 < NG).select(
                    lds_ld(misc, fx.min(2 * tid + 1, fx.Int32(NG - 1))).bitcast(
                        fx.Int32
                    ),
                    fx.Int32(0),
                )
                sw = lo | (lo << 8) | (hi << 16) | (hi << 24)
                bo.buffer_store(sw, r_nope, NOPE_DIM // 4 + tid)
            gpu.barrier()
        else:
            if tid < HEAD_DIM:
                bo.buffer_store(
                    kvn.to(fx.BFloat16), row_rsrc(kv_cache, row, HEAD_DIM * 2), tid
                )

    # ====== 2. KV RMSNorm + RoPE + FP8 round trip -> sliding-window ring cache
    # the NOPE_DIM lanes are FP8 round-tripped in 64-blocks, matching the checkpoint's QAT
    if const_expr(run_attention_cache):
        for t in range(start("cache"), 1, G):
            stamp("cache", t, 0)
            g = ld_bf16(rsrc(g_kv), fx.min(tid, HEAD_DIM - 1))
            ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
            ps = [ld_pos(sx) for sx in range(S)]
            cs = [
                ld_f32(rsrc(rope_cos), ps[sx] * (ROPE_DIM // 2) + ri) for sx in range(S)
            ]
            sns = [
                ld_f32(rsrc(rope_sin), ps[sx] * (ROPE_DIM // 2) + ri) for sx in range(S)
            ]
            hint_wait(
                HEAD_DIM // QKV_A_TILE,
                lambda k: (
                    mb("kv_a"),
                    (S - 1) * HEAD_DIM + k * QKV_A_TILE + QKV_A_TILE - 1,
                ),
                mark=("cache", t),
            )
            vs = getf_many(
                [
                    (mb("kv_a"), s * HEAD_DIM + fx.min(tid, HEAD_DIM - 1))
                    for s in range(S)
                ]
            )
            stamp("cache", t, 2)
            live = tid < HEAD_DIM
            ssq = block_sums([live.select(v * v, fx.Float32(0.0)) for v in vs])
            for s in range_constexpr(S):
                nv = vs[s] * rsq(ssq[s] * (1.0 / HEAD_DIM) + EPS) * g
                # rope tail: lane ^ 1 is the other half of this interleaved (2i, 2i+1) pair
                partner = xshfl(nv, 1)
                even = tid % 2 == 0
                rot = even.select(
                    nv * cs[s] - partner * sns[s], partner * sns[s] + nv * cs[s]
                )
                if const_expr(DRAFT_BLOCK and not KV_FP8):
                    dq = draft_bf16_kv_quant(nv)
                    byte, e8 = fx.Int32(0), fx.Int32(0)
                elif const_expr(KV_QAT):
                    dq, byte, e8 = kv_quant(nv)
                else:
                    dq, byte, e8 = nv, fx.Int32(0), fx.Int32(0)
                kvn = bf16_round((tid < NOPE_DIM).select(dq, rot))
                # Native BF16 draft block KV is temporary, unlike FP8's
                # scatter path. Consumers patch every block row from this
                # epoch's kvnew mailbox, including future bidirectional keys;
                # destination rows are identity metadata, not persistent writes.
                if const_expr(not DRAFT_BLOCK or KV_FP8):
                    put_kv_row(ld_dest(0, s), kvn, byte, e8)
                if live:
                    put(mb("kvnew"), s * HEAD_DIM + tid, kvn)
            stamp("cache", t, 4)

    # ============= 2b. KV compressor (HCA): rolling state, pooled on the boundary
    # each CR window's last token emits a per-channel softmax pool, normed / RoPE'd / FP8'd
    def csa_ksplit4_pool(ring, own):
        """Ordinary CSA Main K=8 arithmetic: four waves fold two rows each, then
        wave 0 merges the four online-softmax partials in increasing wave order."""
        rows = [ring(fx.Int32(i)) for i in range(7)] + [own(7)]
        parts = []
        for w in range_constexpr(4):
            pair = rows[2 * w : 2 * w + 2]
            parts.append(
                pool_fold(
                    [fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)],
                    [a for a, _ in pair],
                    [b for _, b in pair],
                )
            )
        m = fx.Float32(NEG)
        for w in range_constexpr(4):
            m = m.maximumf(parts[w][0])
        den = fx.Float32(0.0)
        num = fx.Float32(0.0)
        for w in range_constexpr(4):
            scale = exp(parts[w][0] - m)
            num = num + parts[w][2] * scale
            den = den + parts[w][1] * scale
        return num * rcp(den)

    if const_expr(CR and run_attention_cache):
        REAL_CSA_KSPLIT4 = (
            compressor_projection_bf16
            and attention_fp8_bf16
            and CR == 4
            and C_COFF == 2
            and C_ROWS == 8
            and HEAD_DIM == 512
            and TOK == 1
            and not KV_FP8
        )
        PACKED_MAIN_KSPLIT = (
            compressor_projection_bf16
            and TOK > 1
            and HEAD_DIM == 512
            and ((CR == 4 and C_ROWS == 8) or (CR == 128 and C_ROWS == 128))
            and not KV_FP8
        )
        for tt in range(start("cmp"), S, G):
            tt = fx.Int32(tt)
            stamp("cmp", tt, 0)
            ch = fx.min(tid, HEAD_DIM - 1)
            live = tid < HEAD_DIM
            # tt is the sample; per-sample state keeps these unordered tasks from racing
            p = ld_pos(tt)
            rs_kv, rs_sc, sb = (
                input_slot_rsrc(kv_state, tt, st_kv),
                input_slot_rsrc(score_state, tt, st_sc),
                0,
            )
            ws_kv = slot_rsrc(kv_state, tt, st_kv)
            ws_sc = slot_rsrc(score_state, tt, st_sc)
            slot = p % CR
            ap0 = [
                ld_f32(rsrc(ape), slot * CW + j * HEAD_DIM + ch) for j in range(C_COFF)
            ]
            g = ld_bf16(rsrc(g_ckv), ch)
            # the window's first position (clamped: loaded unconditionally)
            anchor = fx.max(p + 1 - CR, fx.Int32(0))
            ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
            # one rope table per layer: a compressing layer's is on compress_rope_theta for all it rotates
            rc = ld_f32(rsrc(rope_cos), anchor * (ROPE_DIM // 2) + ri)
            rs = ld_f32(rsrc(rope_sin), anchor * (ROPE_DIM // 2) + ri)
            kvv = [
                getf(mb("c_kv"), (tt * C_COFF + j) * HEAD_DIM + ch)
                for j in range(C_COFF)
            ]
            gtv = [
                getf(mb("c_gate"), (tt * C_COFF + j) * HEAD_DIM + ch)
                for j in range(C_COFF)
            ]
            stamp("cmp", tt, 2)

            def emit_compressed(pooled, native_vec8_norm=False):
                if const_expr(not compressor_projection_bf16):
                    pooled = bf16_round(pooled)
                if const_expr(native_vec8_norm):
                    # Native CSA/HCA wave 0 owns eight consecutive D512 channels per
                    # lane, sums those squares serially, then uses a wave64 XOR
                    # tree. Broadcast its reciprocal RMS to the 512 Mono lanes.
                    if live:
                        lds_st(red, tid, pooled)
                    gpu.barrier()
                    if tid < 64:
                        ssq = fx.Float32(0.0)
                        for i in range_constexpr(8):
                            v = lds_ld(red, tid * 8 + i)
                            ssq = ssq + v * v
                        ssq = wave_sum(ssq)
                        if lane == 0:
                            lds_st(red, HEAD_DIM, rsq(ssq * (1.0 / HEAD_DIM) + EPS))
                    gpu.barrier()
                    rrms = lds_ld(red, HEAD_DIM)
                    gpu.barrier()
                else:
                    ssq = block_sum(live.select(pooled * pooled, fx.Float32(0.0)))
                    rrms = rsq(ssq * (1.0 / HEAD_DIM) + EPS)
                # Current ordinary fused pool/norm/RoPE keeps FP32 until
                # the cache/quant store; the PR oracle inserts BF16 seams.
                nv = pooled * rrms * g
                if const_expr(not compressor_projection_bf16):
                    nv = bf16_round(nv)
                partner = xshfl(nv, 1)
                even = tid % 2 == 0
                rot = even.select(nv * rc - partner * rs, partner * rs + nv * rc)
                if const_expr(KV_QAT):
                    dq, byte, e8 = kv_quant(nv)
                else:
                    dq, byte, e8 = nv, fx.Int32(0), fx.Int32(0)
                cv = bf16_round((tid < NOPE_DIM).select(dq, rot))
                put_kv_row(comp_row(tt, p // CR), cv, byte, e8)
                if live:
                    put(mb("cnew"), tt * HEAD_DIM + tid, cv)

            # overlap: previous window's rows from their first half, current from the second
            def c_half(i):
                return (i >= CR).select(fx.Int32(1), fx.Int32(0)) if OVERLAP else 0

            def c_ring(i):
                wi = sb + ring_row(p, i) * CW + c_half(i) * HEAD_DIM + ch
                sv, kv = ld_f32(rs_sc, wi), ld_f32(rs_kv, wi)
                # Native pads positions before the beginning of the request;
                # modulo-addressed bytes in a recycled slot are not history.
                padding = p + 1 - C_ROWS + i < 0
                sv = padding.select(fx.Float32(NEG), sv)
                kv = padding.select(fx.Float32(0.0), kv)
                if const_expr(TOK == 1):
                    h = 1 if OVERLAP else 0
                    current = i == C_ROWS - 1
                    sv = current.select(gtv[h] + ap0[h], sv)
                    kv = current.select(kvv[h], kv)
                return sv, kv

            def c_own(i):
                h = 1 if (OVERLAP and i >= CR) else 0
                return gtv[h] + ap0[h], kvv[h]

            if const_expr(REAL_CSA_KSPLIT4):
                if (p + 1) % CR == 0:  # uniform across the CTA
                    # Ordinary reads the previous state first; the current row
                    # is a phase-2 input and is published only after emission.
                    emit_compressed(csa_ksplit4_pool(c_ring, c_own), True)

            if const_expr(not REAL_CSA_KSPLIT4):
                if (p + 1) % CR == 0:  # uniform across the CTA
                    j_tok = tt % TOK

                    def c_launch(d, i):
                        h = 1 if (OVERLAP and i >= CR) else 0
                        sm = tt - fx.min(fx.Int32(d), j_tok)
                        q = fx.max(p - d, fx.Int32(0))
                        apq = ld_f32(rsrc(ape), (q % CR) * CW + h * HEAD_DIM + ch)
                        return (
                            getf(mb("c_gate"), (sm * C_COFF + h) * HEAD_DIM + ch) + apq,
                            getf(mb("c_kv"), (sm * C_COFF + h) * HEAD_DIM + ch),
                        )

                    if const_expr(PACKED_MAIN_KSPLIT):
                        pooled = split_window_pool(
                            j_tok, c_ring, c_launch, c_own, 4 if CR == 4 else 8
                        )
                        emit_compressed(pooled, True)
                    else:
                        den_, num_ = window_pool(p, j_tok, c_ring, c_launch, c_own)
                        emit_compressed(num_ * rcp(den_))
            # Match native's read-history -> pool/scatter -> update order.
            # Current launch rows come from mailboxes, never a new state store.
            if live:
                for j in range_constexpr(C_COFF):
                    w = sb + (p % C_RING) * CW + j * HEAD_DIM + ch
                    bo.buffer_store(kvv[j], ws_kv, w)
                    bo.buffer_store(gtv[j] + ap0[j], ws_sc, w)
            stamp("cmp", tt, 4)
    return {"kv_quant": kv_quant, "ring_row": ring_row, "window_pool": window_pool}


@traced
def q_b_stage(ctx):
    """3. q_a RMSNorm -> q_b (raw f32 query)."""
    G = ctx["G"]
    H = ctx["H"]
    HEAD_DIM = ctx["HEAD_DIM"]
    N_QB = ctx["N_QB"]
    Q_LORA = ctx["Q_LORA"]
    S = ctx["S"]
    emit_out = ctx["emit_out"]
    g_q = ctx["g_q"]
    get2_many = ctx["get2_many"]
    hint_wait = ctx["hint_wait"]
    mb = ctx["mb"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    put = ctx["put"]
    reduce_rows = ctx["reduce_rows"]
    run_units = ctx["run_units"]
    s_q_b = ctx["s_q_b"]
    stage_x_rmsnorm = ctx["stage_x_rmsnorm"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_fp8 = ctx["unit_fp8"]
    w_q_b = ctx["w_q_b"]
    wave = ctx["wave"]
    attention_fp8_bf16 = ctx["attention_fp8_bf16"]
    # ==================================== 3. q_a RMSNorm -> q_b (raw f32 query)
    r_wqb, r_sqb = rsrc(w_q_b), rsrc(s_q_b)
    QB_NKC = Q_LORA // 64
    QB_R = q_b_groups(N_QB)
    QB_WPR = WAVES // QB_R
    QB_UPW = QB_NKC // QB_WPR
    QB_BATCH = QB_NKC // WAVES
    QB_ROWS = Q_B_TILE * QB_R
    for t in range(start("q_b"), N_QB // QB_R, G):
        t = fx.Int32(t)
        stamp("q_b", t, 0)
        qb_rg = t * QB_R + wave // QB_WPR

        def u_qb(c):
            kc = (wave % QB_WPR) * QB_UPW + c
            return unit_fp8(
                r_wqb,
                r_sqb,
                qb_rg,
                kc,
                QB_NKC,
                Q_LORA,
                128,
                (n_sel() * Q_LORA + kc * 64) // 2,
            )

        pre = [u_qb(c) for c in range(QB_BATCH)]
        hint_wait(
            Q_LORA // QKV_A_TILE,
            lambda k: (mb("q_a"), (S - 1) * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1),
            mark=("q_b", t),
        )

        def ld_qa(sks):
            v = get2_many(
                [(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)]
            )
            return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

        if const_expr(
            S > WAVES
            or (
                S in (2, 3, 4)
                and ctx["TOK"] == 1
                and ctx["IHD"]
                and ctx["index_topk_refresh"]
            )
        ):
            # Bound query input lifetimes in wide verify and ordinary CSA
            # refresh. Staging every ordinary row together can spill at the
            # native 1M capacity. Each row keeps its RMS reduction and
            # 128-channel FP8 groups; the all-column MFMA still follows
            # publication of every row.
            rows_per_group = 2 if S == 4 else 1
            for sample in range_constexpr(0, S, rows_per_group):
                stage_x_rmsnorm(
                    lambda sks: ld_qa([(sample + s, k) for s, k in sks]),
                    Q_LORA,
                    g_q,
                    count=rows_per_group,
                    output_row_offset=sample,
                    quant_e8m0=attention_fp8_bf16,
                )
        else:
            stage_x_rmsnorm(ld_qa, Q_LORA, g_q, quant_e8m0=attention_fp8_bf16)
        stamp("q_b", t, 2)
        gpu.barrier()
        acc = run_units(u_qb, QB_UPW, QB_BATCH, pre)
        reduce_rows(QB_R, acc, emit_out(QB_ROWS))
        stamp("q_b", t, 3)
        gpu.barrier()
        # PR sees FP32; ordinary sees the GEMM's BF16 output before head RMS.
        if tid < S * QB_ROWS:
            s = tid // QB_ROWS
            row = t * QB_ROWS + tid % QB_ROWS
            v = lds_ld(outs, tid)
            if const_expr(attention_fp8_bf16):
                v = bf16_round(v)
            put(mb("q_raw"), (s * H + row // HEAD_DIM) * HEAD_DIM + row % HEAD_DIM, v)
        stamp("q_b", t, 4)
    return {}
