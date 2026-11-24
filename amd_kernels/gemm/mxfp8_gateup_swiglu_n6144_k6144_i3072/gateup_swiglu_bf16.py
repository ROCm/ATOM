# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Scaled-MFMA gate/up with paired weight tiles and BF16 SwiGLU output.

The existing packed gate|up weight is read without another layout or copy.
Every GEMM dot rounds to BF16 before the caller-selected activation; the
BF16 activation feeds the unchanged down projection quantizer.
Two K-waves sum in the materialized GEMM's order, so ``bf16_chain`` gives the draft's bf16 bits; its SwiGLU rounds
to bf16 at every op, as the draft's aten chain.
Based on mxfp8_gemm_decode.py.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

W_CACHE_MOD = 2  # non-temporal W loads: a decode GEMM reads every weight byte once
# VGPRs the kernel may hold at once. An MFMA operand is 8 contiguous VGPRs, so the register
# allocator runs out well before all 256 are live (measured: clean at 199 live, parking
# operands in AGPRs at 202). A parked operand is copied back with v_accvgpr_read right
# before its MFMA and races the AGPR write: wrong sums, some of them only now and then.
VGPR_BUDGET = 192
# kernel arguments preloaded into SGPRs instead of s_load'ed before the first loads
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _addr(t):
    """A tensor argument's device address (in a launch body)."""
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _lookahead(SW, RT, NI, RP, CP, vgpr_budget=None, a_vgprs=8, reserve=0, fold=1):
    """(W steps, A blocks) issued ahead of the MFMAs: the longest common horizon whose
    registers fit. Held at once: the operands in flight plus the one in use of each
    queue (8 VGPRs per 128-K operand), a scale dword per 32-row / 32-column group for
    every other W step, and the per-lane offset vector of every row / column tile.
    An A block is one (step, row tile) pair, or ``fold`` steps of the folded bf16 mode."""
    budget = (vgpr_budget or VGPR_BUDGET) - (RT + NI) - reserve
    best = None
    for la_a in range(1, SW * RT // fold + 1):
        la_w = min(SW, -(-la_a * fold // RT))
        regs = a_vgprs * (la_a + 1) + 8 * NI * (la_w + 1) + (CP + RP) * (la_w // 2 + 2)
        if regs > budget:
            break
        best = (la_w, la_a)
    assert best is not None, f"RT {RT} x NI {NI} leaves no VGPRs for the operands"
    return best


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _pack8(h0, h1):
    """two 4 x i32 halves -> one 8 x i32 MFMA operand (VGPRs 0-3 = K [16*klane, +16),
    VGPRs 4-7 = K [64 + 16*klane, +16))."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements(
        [a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32
    )


def _mfma(w_op, a_op, acc, sw, sa, jw, ia, k2):
    """One scaled 16x16x128 fp8 MFMA, accumulator pinned in AGPR (``=a`` / tied ``0``).
    Operand order (W, A): lane (q, r) accumulates out[row r, columns 4q .. 4q+3].
    Scale byte of a dword = op_sel + 2 * op_sel_hi (16-row / 16-col half, 128-K half)."""
    src2 = "$0" if acc is not None else "0"
    asm = (
        f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
        f"op_sel:[{jw},{ia},0] op_sel_hi:[{k2},{k2},0] cbsz:0 blgp:0"
    )
    ops = [
        fx.as_ir_value(w_op),
        fx.as_ir_value(a_op),
        fx.as_ir_value(sw),
        fx.as_ir_value(sa),
    ]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(fx.as_ir_value(acc))
        cons += ",0"
    return llvm.inline_asm(
        Vec.make_type(4, fx.Float32), ops, asm, cons, has_side_effects=True
    )


def _pk_max_u16(a, b):
    """per-16-bit-half unsigned max (v_pk_max_u16): on sign-cleared bf16 patterns that is
    the value order."""
    va = llvm.BitcastOp(T.i16x2, _raw(fx.Int32(a))).result
    vb = llvm.BitcastOp(T.i16x2, _raw(fx.Int32(b))).result
    return fx.Int32(llvm.BitcastOp(T.i32, llvm.intr_umax(va, vb)).result)


def _amax_bf16(dwords):
    """max |x| over the 2 * len(dwords) bf16 values, as a 16-bit bf16 pattern in an i32."""
    m = None
    for d in dwords:
        a = fx.Int32(d) & fx.Int32(0x7FFF7FFF)
        m = a if m is None else _pk_max_u16(m, a)
    lo, hi = m & fx.Int32(0xFFFF), m >> 16
    return (lo > hi).select(lo, hi)


def _i32_max(a, b):
    return (a > b).select(a, b)


def _swap16(v):
    """(even-row value, odd-row value) of this lane's 16-lane row pair (klane q, q ^ 1)."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(fx.Int32(v))
    res = fx.rocdl.permlane16_swap(pair_ty, raw, raw, False, False)
    return fx.Int32(llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        llvm.extractvalue(T.i32, res, [1])
    )


def _swap32(v):
    """(low-half value, high-half value): the values of klanes 0/1 and 2/3."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(fx.Int32(v))
    res = fx.rocdl.permlane32_swap(pair_ty, raw, raw, False, False)
    return fx.Int32(llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        llvm.extractvalue(T.i32, res, [1])
    )


def _e8m0_bf16(amax_bits):
    """(scale byte, conversion scale) of a block from the bf16 pattern of its maximum. The
    byte is the Triton MXFP8 quantizer's, clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0,
    254): with amax = 2^e * m, ceil(log2(amax / 448)) = e - 8 + [m > 1.75], so it is the
    exponent field - 8 + [mantissa > 0x60], which (a + 0x1F) >> 7 gives in one carry;
    zero, denormal and tiny maxima land on 0 as in that quantizer. The conversion divides by
    2^(byte - 127); byte 0 would make that a denormal, so such a block (maximum below
    2^-119) is converted with 2^-126 instead."""
    t = (fx.Int32(amax_bits) + fx.Int32(0x1F)) >> 7
    sb = _i32_max(t, fx.Int32(8)) - fx.Int32(8)
    scale = ((_i32_max(t, fx.Int32(9)) - fx.Int32(8)) << 23).bitcast(fx.Float32)
    return sb, scale


def _e8m0_bf16_pp2(amax_bits):
    """NVIDIA-consistent variant of ``_e8m0_bf16``: the stored
    byte IS the conversion exponent (``max(t, 9) - 8``, never 0), so the stored
    scale always equals the 2^(byte - 127) its consumer decodes -- ``_e8m0_bf16``
    stores 0 for a block maximum below ~1.75*2^-119 while converting with 2^-126
    (a 2x error under an OCP consumer, a zeroed block under a ``(byte << 23)``
    one). Identical output for every bf16 block maximum above 1.75*2^-119, which
    includes every nonzero normal activation."""
    t = (fx.Int32(amax_bits) + fx.Int32(0x1F)) >> 7
    sb = _i32_max(t, fx.Int32(9)) - fx.Int32(8)
    scale = (sb << 23).bitcast(fx.Float32)
    return sb, scale


def _row_shl(vals, n):
    """Each lane takes the value of lane + n of its 16-lane row (v_mov_b32 dpp row_shl;
    the last n lanes of a row get 0)."""
    zero = _raw(fx.Int32(0))
    return [
        fx.Int32(
            fx.rocdl.update_dpp(
                T.i32, zero, _raw(fx.Int32(v)), 0x100 + n, 0xF, 0xF, True
            )
        )
        for v in vals
    ]


def _quant_bf16(dwords, scale):
    """dwords of bf16 pairs -> half as many dwords of fp8 codes: x / scale, round to nearest
    even (v_cvt_scalef32_pk_fp8_bf16, the scale is 2^(sb - 127))."""
    out = []
    for i in range(0, len(dwords), 2):
        pair0 = llvm.BitcastOp(T.bf16x2, _raw(fx.Int32(dwords[i]))).result
        pair1 = llvm.BitcastOp(T.bf16x2, _raw(fx.Int32(dwords[i + 1]))).result
        # the first conversion's other half is overwritten by the second
        w = fx.rocdl.cvt_scalef32_pk_fp8_bf16(
            T.i16x2, llvm.mlir_undef(T.i16x2), pair0, _raw(scale), False
        )
        w = fx.rocdl.cvt_scalef32_pk_fp8_bf16(T.i16x2, w, pair1, _raw(scale), True)
        out.append(fx.Int32(llvm.BitcastOp(T.i32, w).result))
    return Vec.from_elements(out, fx.Int32)


def _pin_accumulators(values):
    """Route the accumulators through an inline asm that outputs them again (tied
    ``=a`` / ``0``) with the XDL wait states inside: the reads that follow depend on
    this fence and cannot be hoisted right behind the last inline-asm MFMA (the
    compiler sees no hazard through the asm; without it the last MFMA of a wave was
    missing from half of each accumulator)."""
    one = len(values) == 1  # an inline asm with one output returns it bare
    ty = ir.Type.parse(
        "vector<4xf32>"
        if one
        else "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(values)) + ")>"
    )
    constraints = ",".join(["=a"] * len(values) + [str(i) for i in range(len(values))])
    r = llvm.inline_asm(
        ty,
        [_raw(v) for v in values],
        "s_nop 15\ns_nop 15",
        constraints,
        has_side_effects=True,
    )
    if one:
        return [r]
    return [llvm.extractvalue(T.vec(4, T.f32), r, [i]) for i in range(len(values))]


def compile_mxfp8_gateup_swiglu_bf16(form='fp32'):
    """Fixed 1-8-row, paired 6144x6144 gate/up epilogue into 3072 BF16 activations.

    The cold whole-FFN sweep uses NI2/KW2 and a (7,7) W/A lookahead. Each output
    keeps the incumbent's K partition and sum order. Scalar activation parameters
    stay runtime arguments so constant folding cannot change their rounding.
    """
    N,K,RT,NI,KW,KS=6144,6144,1,2,2,1
    vgpr_budget=None
    sched_barriers=False
    LA=(7,7)
    A_BF16=False
    FOLD=1
    W_CM=None
    assert form in ('fp32','bf16_chain')
    w_cm = W_CACHE_MOD if W_CM is None else W_CM
    assert w_cm in (0, 1, 2), w_cm
    assert K % 256 == 0 and N % (16 * NI) == 0
    KSTEPS = K // 128  # 128-K steps
    assert KSTEPS % (KS * KW) == 0 and (KSTEPS // (KS * KW)) % 2 == 0, (K, KS, KW)
    SW = KSTEPS // (KS * KW)  # steps per K-wave (even: starts on a 256-K scale block)
    # Odd RT is legal only on the quantized-A path, where the A-scale dwords are
    # re-paired to the m-block's parity at load time (module docstring). A_BF16 builds
    # each scale dword itself with the byte at (rt % 2) and stays even-RT.
    assert RT == 1 or RT % 2 == 0 or not A_BF16
    BM = 16 * RT // FOLD  # rows per m-block
    NB = N // (16 * NI)
    SCB = K // 256  # scale blocks (64 dwords) per 32-row / 32-column group
    RP = max(1, RT // 2)  # 32-row scale groups per workgroup
    if RT > 1 and RT % 2 == 1:
        RP = (RT + 1) // 2  # the parity-straddling block needs one group more
    # W scale dwords per step: one per 32-column group, or one per 16-column tile when
    # NI is odd (a group can then straddle two workgroups: each tile shifts its half down)
    CP = NI
    LA_W, LA_A = LA or _lookahead(
        SW, RT, NI, RP, CP, vgpr_budget, a_vgprs=16 if A_BF16 else 8,
        reserve=(24 + (8 if FOLD > 1 else 0)) if A_BF16 else 0, fold=FOLD,
    )  # fmt: skip
    assert FOLD in (1, 2, 4) and (
        FOLD == 1 or (A_BF16 and RT == 1 and SW % FOLD == 0)
    ), (FOLD, RT, SW)
    RPT = 16 // FOLD  # rows per 16-row tile
    NA = SW * RT // FOLD  # A blocks per K-wave
    assert 1 <= LA_W <= SW and 1 <= LA_A <= NA, (LA_W, LA_A, SW, RT, FOLD)
    RED_SLOT = RT * NI * 64 * 16  # bytes parked per K-wave
    LDS_BYTES = max(16, KW * RED_SLOT)
    assert LDS_BYTES <= 160 * 1024, "K-reduce scratch exceeds the LDS"
    assert N * K <= 0xFFFFFFFF

    @fx.struct
    class Shared:
        red: fx.Array[fx.Uint8, LDS_BYTES, 16]

    name = (
        f"mxfp8_gateup_swiglu_bf16_{form}_i3072_" +
        f"flydsl_mxfp8_dec_n{N}_k{K}_rt{RT}_ni{NI}_kw{KW}_ks{KS}_la{LA_W}x{LA_A}"
        + ("sb" if sched_barriers else "")
        + (("_fq" + (str(FOLD) if FOLD > 1 else "")) if A_BF16 else "")
        + (f"_wcm{w_cm}" if w_cm != W_CACHE_MOD else "")
    )

    # addresses, not tensors (a tensor argument ends the preload): every argument
    # arrives preloaded in SGPRs
    @flyc.kernel(name=name, known_block_size=[64 * KW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_mb: fx.Int32,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_as: fx.Int64,
        arg_a: fx.Int64,
        arg_out: fx.Int64,
        i32_alpha: fx.Int32, i32_beta: fx.Int32, i32_limit: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        if const_expr(KS > 1):
            kz, wg = bx // (n_mb * NB), bx % (n_mb * NB)
            kwave = kz * KW + wave
        else:
            kz, wg, kwave = 0, bx, wave
        mb, nb = wg // NB, wg % NB
        lane_off = q16 * 256 + l16 * 16  # a lane's 16 B inside a 1 KB block
        k_base = kwave * (SW * 2048)  # this K-wave's first block pair (bytes)

        if const_expr(A_BF16):
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_a, num_records_bytes=i32_m * (2 * K)
            )
        else:
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_a, num_records_bytes=n_mb * (BM * K)
            )
        w_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_w, num_records_bytes=N * K
        )
        as_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_as, num_records_bytes=a_scale_bytes
        )
        ws_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_ws, num_records_bytes=(N // 32) * SCB * 256
        )
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=i32_m * ((N//2) * 2)
        )

        n16 = [nb + ni*(N//32) for ni in range_constexpr(NI)]
        r16 = [mb * RT + rt for rt in range_constexpr(RT)]
        w_dw = [
            (n16[ni] * (16 * K) + k_base + lane_off) // 4 for ni in range_constexpr(NI)
        ]
        if const_expr(A_BF16):
            # row-major bf16: this lane's row, its 32-byte K segment, this K-wave's first
            # step (folded: the lane's row r and step j of each block)
            a_row, a_fold = l16 % RPT, l16 // RPT
            a_dw = [
                (
                    (r16[rt] * RPT + a_row) * (2 * K)
                    + kwave * (SW * 256)
                    + a_fold * 256
                    + q16 * 32
                )
                // 4
                for rt in range_constexpr(RT)
            ]
            sa_shift = (
                ((a_fold % 2) * 16) if FOLD > 1 else None
            )  # the block step's parity
        else:
            a_dw = [
                (r16[rt] * (16 * K) + k_base + lane_off) // 4
                for rt in range_constexpr(RT)
            ]
        sc_lane = q16 * 16 + l16
        sc_k = kwave * (SW // 2 * 64)  # this K-wave's first 256-K scale block (dwords)
        ws_dw = [
            ((n16[cp] // 2) * SCB) * 64 + sc_k + sc_lane
            for cp in range_constexpr(CP)
        ]
        if const_expr(RT > 1 and RT % 2 == 1):
            # odd RT: address the 32-row groups from the m-block's first tile rounded
            # down to even (r16[0] // 2 floors); the parity mix below re-pairs them
            # (const_expr: a bare if is rewritten to a dynamic scf.if whose branch
            # assignments never bind in this scope -> as_dw NameError)
            as_dw = [
                ((r16[0] // 2 + rp) * SCB) * 64 + sc_k + sc_lane
                for rp in range_constexpr(RP)
            ]
        else:
            as_dw = [
                ((r16[2 * rp] // 2) * SCB) * 64 + sc_k + sc_lane
                for rp in range_constexpr(RP)
            ]
        a_par = r16[0] % 2 if RT > 1 and RT % 2 == 1 else None  # m-block parity
        # a 16-column (row) tile of an odd count sits in either half of its 32 group:
        # move its bytes (half, half + 2) down to (0, 2) and select half 0
        w_shift = [(n16[ni] % 2) * 8 for ni in range_constexpr(NI)]
        a_shift = (r16[0] % 2) * 8 if RT == 1 else None

        def load_pair(rsrc, dw, s, nt):
            kw = dict(vec_width=4, dtype=fx.Int32)
            if const_expr(nt):
                kw["cache_modifier"] = w_cm
            return _pack8(
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048, **kw),
                buffer_ops.buffer_load(rsrc, dw, soffset_bytes=s * 2048 + 1024, **kw),
            )

        def load_scales(rsrc, dws, s, shift):
            vals = [
                fx.Int32(
                    buffer_ops.buffer_load(
                        rsrc,
                        dw,
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=(s // 2) * 256,
                    )
                )
                for dw in dws
            ]
            if const_expr(shift is None):
                return vals
            if const_expr(isinstance(shift, list)):
                return [v.shrui(sh) for v, sh in zip(vals, shift)]
            return [v.shrui(shift) for v in vals]

        def repair_a_scale(vals, p):
            """Odd RT: re-pair scale dword p to the even-parity layout the MFMA immediates
            assume -- dword p must carry block tiles 2p (bytes 0/2) and 2p+1 (bytes 1/3).
            An odd-parity block shifts every atom by one half: bytes 0/2 come from half 1
            of group p, bytes 1/3 from half 0 of group p+1 (the last pair's byte 1/3 name a
            tile past the block and are never consumed). Called at the dword's first MFMA,
            not at its load, so the load keeps its full look-ahead (VMLEAD)."""
            nxt = vals[p + 1] if p + 1 < RP else vals[p]
            return (a_par == 1).select(
                (vals[p].shrui(fx.Int32(8)) & fx.Int32(0x00FF00FF))
                | ((nxt & fx.Int32(0x00FF00FF)) << 8),
                vals[p],
            )

        def load_w_step(s):
            bb = [load_pair(w_rsrc, w_dw[ni], s, True) for ni in range_constexpr(NI)]
            if const_expr(s % 2 == 0):
                return bb, (
                    load_scales(ws_rsrc, ws_dw, s, w_shift),
                    None if A_BF16 else load_scales(as_rsrc, as_dw, s, a_shift),
                )
            return bb, None

        def load_a(t):
            rt, s = t % RT, (t // RT) * FOLD  # block t: row tile, first step
            if const_expr(A_BF16):
                # the two 16-K segments (K 16q and 64 + 16q of the step), 32 B each, bf16
                return [
                    buffer_ops.buffer_load(
                        a_rsrc,
                        a_dw[rt],
                        vec_width=4,
                        dtype=fx.Int32,
                        soffset_bytes=s * 256 + off,
                    )
                    for off in (0, 16, 128, 144)
                ]
            return load_pair(a_rsrc, a_dw[rt], s, False)

        def quantize(raw, rt, s):
            """raw bf16 of (row tile rt, step s) -> (fp8 operand, this MFMA's scale dword)."""
            seg0 = [Vec(raw[0])[i] for i in range(4)] + [
                Vec(raw[1])[i] for i in range(4)
            ]
            seg1 = [Vec(raw[2])[i] for i in range(4)] + [
                Vec(raw[3])[i] for i in range(4)
            ]
            # per-32 groups of this row: pair (q, q ^ 1) holds group q // 2 in segment 0 and
            # group 2 + q // 2 in segment 1; the other pair's values come with the 32-swap
            e0, o0 = _swap16(_amax_bf16(seg0))
            e1, o1 = _swap16(_amax_bf16(seg1))
            own0, own1 = _i32_max(e0, o0), _i32_max(e1, o1)
            (sb0, sc0), (sb1, sc1) = _e8m0_bf16(own0), _e8m0_bf16(own1)
            # the scale byte the MFMA reads from this lane is group q's
            lo0, hi0 = _swap32(sb0)
            lo1, hi1 = _swap32(sb1)
            sbm = (q16 == 0).select(
                sb0, (q16 == 1).select(hi0, (q16 == 2).select(lo1, sb1))
            )
            a_op = _pack8(_quant_bf16(seg0, sc0), _quant_bf16(seg1, sc1))
            if const_expr(FOLD > 1):
                return a_op, sbm << sa_shift
            return a_op, sbm << (8 * (2 * (s % 2) + (rt % 2)))

        # Both operands run ahead of the MFMAs: W by LA_W steps, A by LA_A (step, row
        # tile) blocks in the order the MFMAs consume them; with everything in flight
        # from the start a wave ends when its last block lands, not after one memory
        # latency per step.
        acc = [[None] * NI for _ in range(RT)]
        w_q = [None] * SW
        a_q = [None] * NA
        for s in range_constexpr(LA_W):
            w_q[s] = load_w_step(s)
        for t in range_constexpr(LA_A):
            a_q[t] = load_a(t)
        if const_expr(sched_barriers):
            fx.rocdl.sched_barrier(0)
        sw = sa = None
        held = None  # folded mode: the block's quantized operand and scale dword
        for s in range_constexpr(SW):
            if const_expr(s + LA_W < SW):
                w_q[s + LA_W] = load_w_step(s + LA_W)
            bb, sc = w_q[s]
            if const_expr(s % 2 == 0):
                sw, sa = sc
                if const_expr(RT > 1 and RT % 2 == 1):
                    sa_raw, sa = sa, [None] * RP  # repaired at first use (both steps)
            for rt in range_constexpr(RT):
                if const_expr(FOLD == 1):
                    t = s * RT + rt
                    if const_expr(t + LA_A < NA):
                        a_q[t + LA_A] = load_a(t + LA_A)
                    if const_expr(A_BF16):
                        a_op, sa_v = quantize(a_q[t], rt, s)
                    else:
                        if const_expr(RT > 1 and RT % 2 == 1 and sa[rt // 2] is None):
                            sa[rt // 2] = repair_a_scale(sa_raw, rt // 2)
                        a_op, sa_v = a_q[t], sa[rt // 2]
                    a_q[t] = None
                else:
                    b, j = s // FOLD, s % FOLD
                    if const_expr(j == 0):
                        if const_expr(b + LA_A < NA):
                            a_q[b + LA_A] = load_a(b + LA_A)
                        held = quantize(a_q[b], rt, s)
                        a_q[b] = None
                    if const_expr(j == 0):
                        a_op, sa_v = held
                    else:
                        moved = _row_shl(
                            [Vec(held[0])[i] for i in range(8)] + [held[1]], RPT * j
                        )
                        a_op, sa_v = Vec.from_elements(moved[:8], fx.Int32), moved[8]
                for ni in range_constexpr(NI):
                    acc[rt][ni] = _mfma(
                        bb[ni],
                        a_op,
                        acc[rt][ni],
                        sw[ni],
                        sa_v,
                        0,
                        rt % 2,
                        s % 2,
                    )
                # the loads depend on nothing, so the scheduler would pull later steps'
                # loads up to here: more operands alive than _lookahead budgeted
                if const_expr(sched_barriers):
                    fx.rocdl.sched_barrier(0)
            w_q[s] = None

        # Keep GEMM's BF16 materialization before the model's activation.
        flat = _pin_accumulators([acc[rt][ni] for rt in range(RT) for ni in range(NI)])
        lds_base = fx.Int32(fx.ptrtoint(smem.red.ptr))
        def lds_ptr(i,w):
            off=w*RED_SLOT+i*1024
            return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"),_raw(fx.Int64(lds_base+off+lane*16)))
        for i in range_constexpr(RT*NI):
            llvm.StoreOp(_raw(Vec(flat[i]).bitcast(fx.Int32)),lds_ptr(i,wave),alignment=16)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        if wave==0:
            vals=[]
            for i in range_constexpr(2):
                v=Vec(flat[i])
                for w in range_constexpr(1,KW):
                    v=v+Vec(llvm.load(T.vec(4,T.i32),lds_ptr(i,w),alignment=16)).bitcast(fx.Float32)
                vals.append(v.to(fx.BFloat16).to(fx.Float32))
            gate,up=vals
            alpha,beta,limit=i32_alpha.bitcast(fx.Float32),i32_beta.bitcast(fx.Float32),i32_limit.bitcast(fx.Float32)
            def bf(v):return v.to(fx.BFloat16).to(fx.Float32)
            ys=[]
            for e in range_constexpr(4):
                gv=fx.Float32(gate[e]).minimumf(limit)
                uv=fx.Float32(up[e]).maximumf(-limit).minimumf(limit)
                if const_expr(form=="bf16_chain"):
                    ag=bf(alpha*gv)
                    sig=bf(fx.Float32(1.0)/(fx.Float32(1.0)+fx.math.exp(-ag)))
                    y=bf(bf(gv*sig)*bf(uv+beta))
                else:
                    sig=fx.Float32(1.0)/(fx.Float32(1.0)+fx.math.exp(-(alpha*gv)))
                    y=(gv*sig)*(uv+beta)
                ys.append(y)
            row=mb*16+l16
            col=nb*16+q16*4
            packed=Vec.from_elements(ys,fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            buffer_ops.buffer_store(packed,out_rsrc,(row*(N//2)+col)//2,mask=row<i32_m)

    @flyc.jit
    def launch_mxfp8_gateup_swiglu_bf16_n6144_k6144(
        A: fx.Tensor,
        W: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W_scale: fx.Tensor,
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_mb: fx.Int32,
        grid: fx.Int32,
        i32_alpha: fx.Int32, i32_beta: fx.Int32, i32_limit: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, a_scale_bytes, n_mb, _addr(W), _addr(W_scale), _addr(A_scale), _addr(A),
            _addr(OUT), i32_alpha, i32_beta, i32_limit,
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(64 * KW, 1, 1), stream=stream)  # fmt: skip

    launch_mxfp8_gateup_swiglu_bf16_n6144_k6144.kernel_name = name
    launch_mxfp8_gateup_swiglu_bf16_n6144_k6144.compile_hints = PRELOAD_HINTS
    launch_mxfp8_gateup_swiglu_bf16_n6144_k6144.k_split = KS
    launch_mxfp8_gateup_swiglu_bf16_n6144_k6144.n_blocks = NB
    launch_mxfp8_gateup_swiglu_bf16_n6144_k6144.block_m = BM
    launch_mxfp8_gateup_swiglu_bf16_n6144_k6144.fold = FOLD
    return launch_mxfp8_gateup_swiglu_bf16_n6144_k6144
