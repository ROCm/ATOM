# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""gfx950 MXFP8 x MXFP8 GEMM for decode-size M, one launch, no LDS staging of A or W:

    out[m, n] = sum_k A[m, k] * W[n, k]          (E4M3 codes, per-32 E8M0 scales, bf16 out)

A is taken in the same (16, 16) preshuffled layout as W (16-row tiles instead of
16-column tiles), so one ``buffer_load`` brings a contiguous 1 KB block of either
operand straight into the 64 lanes of a wave: lane (klane, r) gets K bytes
``[16*klane, +16)`` of row / column r of a 64-K block, and two blocks (K ``+0`` and
``+64``) make one 8-VGPR operand of ``v_mfma_scale_f32_16x16x128_f8f6f4``.

A workgroup is ``KW`` K-waves over the same ``NI`` 16-column tiles and ``RT`` 16-row
tiles: wave w covers K-steps ``[w*SW, (w+1)*SW)`` (128 K each); W (non-temporal
loads) and A blocks are issued ahead of the MFMAs as far as the registers allow
(a decode wave has all of them in flight from the start). After the last step every
K-wave parks the fp32 partial sums of the tiles it does not own in LDS; the owner of a
tile (K-wave tile % KW) adds the KW partials in wave order (deterministic) and stores
it: the MFMA is fed (W, A), so a lane ends up with 4 consecutive columns of one row =
one 8 B store.

Layouts (bytes):
  A        [RT*16, K]          shuffle_weight(A padded to 16 rows, (16, 16))
  A_scale  [ceil(M/32)*32, K/32]  shuffle_scale_a16w4: [M/32, K/256, klane 4, r 16] dwords,
                                  bytes = (16-row half) + 2 * (128-K half)
  W        [N, K], W_scale     the same two shuffles of the weight
  OUT      [M, N] bf16

Odd RT (a 16*RT-row tile with RT % 2 == 1, e.g. RT=7 for a 112-row tile): the A_scale
atom pairs two 16-row tiles per dword, so the MFMA's half-select (op_sel, a trace-time
immediate) would flip meaning with the m-block's parity. The kernel re-pairs instead:
scale dwords are addressed from the m-block's first tile rounded DOWN to even
(r16[0] // 2), and when the block starts on the odd half, each dword consumed by the
MFMAs is rebuilt at load time as (d[p] >> 8 & 0x00FF00FF) | ((d[p+1] & 0x00FF00FF) << 8)
-- bytes 0/2 from the odd half of group p, bytes 1/3 from the even half of group p+1 --
which restores the even-parity layout the immediates assume. The last pair mixes the
last dword with itself (its byte 1/3 name a tile past the block, never consumed). RT=1
keeps its own shift path; A_BF16 builds its own scale dwords and stays even-RT.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .mxfp8_gemm_4wave import _acc_read

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


# (N, K, RT, NI, KW, KS) rows that run the park-all K-reduce (``PARK_ALL``). It moves
# KW / (KW - 1) times the owner form's LDS bytes; measured in graph replay on the TP4 shapes
# (bitwise equal outputs): QKV 64x32 kw3 / kw4 -0.6..-1.3%,
# 96x32 kw4 -2.8..-3.7%, 96..160x80 kw3 -0.9..-1.6%, deferred 48x64 / 64x64 kw3 split 2
# -0.3..-0.6%, o_proj 80x48 kw4 -1.7..-2.5%. It is flat or slower on 64x64 / 64x80 / 96x64
# kw3 and 80x80 kw4 standalone rows (-0.1..+0.4%) and loses 2-5% with two K-waves (o_proj
# 64x96 / 80x96 kw2). The deferred 64x80 / 96x64 split-2 rows take four K-waves with it
# (``gemm._deferred_qkv_config``), the o_proj 96x48 (161-192) and 80x96 (257-320) rows four
# instead of two (-1.4..-1.5%, -2.9..-3.3%). Other shapes keep the owner form until they are measured.
_PARK_ALL_ROWS = frozenset(
    [(2560, 6144, rt, 2, kw, 1) for rt, kw in ((4, 3), (4, 4), (6, 4))]
    + [(2560, 6144, rt, 5, 3, 1) for rt in (6, 7, 8, 9, 10)]
    + [(2560, 6144, 3, 4, 3, 2), (2560, 6144, 4, 4, 3, 2), (2560, 6144, 4, 5, 4, 2),
       (2560, 6144, 6, 4, 4, 2)]
    + [(6144, 2048, 5, 3, 4, 1), (6144, 2048, 6, 3, 4, 1), (6144, 2048, 5, 6, 4, 1)]
)


# Read owned accumulators inside the owning K-wave's branch. Explicit reads avoid
# LLVM copying them before both wave branches (144 -> 96 static reads, same LDS writes).
# The public-entry A/B at M193-256 / 321-512 is 2.0-2.9% faster, bitwise equal.
_OWNER_READ_ROWS = frozenset((6144, 2048, rt, 6, 2, 1) for rt in (4, 6, 7, 8))

# (N, K, RT, NI, KW, KS) rows whose output stores are non-temporal (``OUT_CM``). The next launch
# reads these outputs at once: the deferred TP4 QKV partial slabs (fp32, read by the split-K
# reduce-insert) at 81-288 rows. With default stores the slab's dirty lines wait in the writing
# XCD's L2 and the hand-off cost the chain ~0.5 us at 128-256 rows (the same reduce-insert on a
# slab written long before ran 0.5 us faster). Chain -1.7..-4.2% at
# 81-288 rows (3 x 5 rounds, all faster), bitwise; the 16x32 .. 80x32 rows (1-80) lose 0.2-3.1%
# with them, and the bf16 standalone rows 0.1-1.5% (96x32 four K-waves mixed at 193-288).
_NT_OUT_ROWS = frozenset([
    (2560, 6144, 3, 4, 1, 3), (2560, 6144, 4, 4, 1, 3),  # 81-96 48x64, 97-128 64x64, split 3
    (2560, 6144, 3, 4, 3, 2), (2560, 6144, 4, 4, 3, 2),  # 129-144 48x64, 145-192 64x64, split 2
    (2560, 6144, 4, 5, 4, 2), (2560, 6144, 6, 4, 4, 2),  # 193-256 64x80, 257-288 96x64, split 2
])


# At the same lookahead, interleave one W step with its A fragments; A first for o_proj.
# Deferred QKV M97-128 chain -2.3..-2.4%; o_proj M257-320 -1.6..-1.9%,
# M385-448 -4.2..-4.7%. Other rows keep the W-first burst.
_INTERLEAVED_PRIME_ROWS = frozenset((
    (2560, 6144, 4, 4, 1, 3), (6144, 2048, 5, 6, 4, 1), (6144, 2048, 7, 6, 2, 1),
))


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
    compiler sees no hazard through the asm; without it the last MFMA of a wave is
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


def compile_mxfp8_gemm_decode(
    *,
    N: int,
    K: int,
    RT: int,
    NI: int,
    KW: int,
    KS: int = 1,
    vgpr_budget: int | None = None,
    sched_barriers: bool = True,
    LA: tuple[int, int] | None = None,
    A_BF16: bool = False,
    FOLD: int = 1,
    W_CM: int | None = None,
    PARK_ALL: bool | None = None,
    OWNER_READ: bool | None = None,
    OUT_CM: int | None = None,
    INTERLEAVE_PRIME: bool | None = None,
):
    """Kernel for one (N, K) weight, ``RT`` 16-row tiles per workgroup (M <= 16*RT per
    m-block), ``NI`` 16-column tiles per workgroup, ``KW`` K-waves. Grid =
    ``KS * ceil(M / (16*RT)) * launch.n_blocks`` workgroups of ``64*KW`` threads.

    ``KS`` > 1 splits K over workgroups as well (a workgroup then reads only its
    ``K / KS`` of A: what bounds a workgroup at M >= 64 is the A it has to ingest);
    OUT is then the fp32 ``[KS, M, N]`` partial slabs for the split-K reduce kernel.

    ``A_BF16``: A is the bf16 activation itself, row-major [M, K] (rows beyond M read
    zero), quantized in registers right before the MFMA -- the per-32 amax needs the two
    16-K segments a lane holds and their neighbours' (klane q ^ 1 through a 16-lane row
    swap, q ^ 2 through a 32-lane swap), the scale byte is the Triton quantizer's, the
    codes are x / 2^(sb - 127) rounded to nearest even (v_cvt_scalef32_pk_fp8_bf16, the
    per-32 maxima on packed bf16 patterns); A_scale is unused. Saves the quantizer launch at
    decode M (2.1 us of 7.5 at M 4). Every workgroup quantizes its rows again, so the cost
    is the VALU work per step (~70 instructions per lane, for the 16 rows of a tile);
    ``FOLD`` 2 / 4 (M <= 8 / 4, ``RT`` 1): a tile holds 16 / FOLD rows and the lanes of
    MFMA row r + (16 / FOLD) * j quantize step j of every FOLD-step block for row r --
    one load and a FOLD-th of the VALU work per step, the codes and the scale byte move
    to MFMA row r with a DPP row shift right before the MFMA; MFMA rows beyond 16 / FOLD
    compute junk that the store masks.

    ``LA`` = (W steps, A blocks) in flight ahead of the MFMAs, by default the deepest
    pair that fits ``vgpr_budget`` VGPRs; ``sched_barriers`` keeps the loads where the
    source issues them (off, the scheduler hoists them: a little faster while the
    registers last). The tuned rows carry the values the sweep found clean and fastest.

    ``INTERLEAVE_PRIME=False`` runs the preceding W-first burst prologue for A/B tests.
    Measured rows otherwise alternate one W step and its A fragments (A first for o_proj).

    ``OWNER_READ=False`` runs the preceding owner-copy epilogue for A/B tests.

    ``OUT_CM`` is the output stores' cache modifier (default: non-temporal, 2, on the rows of
    ``_NT_OUT_ROWS``, else 0 = the default policy); it changes no byte of the output.

    ``PARK_ALL`` (default: the measured rows of ``_PARK_ALL_ROWS``) is the K-reduce form
    with ``KW`` > 1: every K-wave parks every tile in LDS straight from its AGPRs, those of a
    row tile right after the next row tile's MFMAs of the last step, and wave w then sums
    tiles w, w + KW, ... from LDS in wave order (all LDS reads of a group of tiles before
    its first store; the tile's part of the store offset in the scalar offset) -- the sums
    of the owner form below, bit for bit, without its per-wave code copies and the
    AGPR-to-VGPR copies of the owned tiles. ``False``: the owner form (the owner of tile
    i = K-wave i % KW keeps its own partial in registers and adds the others' from LDS)."""
    interleave_auto = ((N, K, RT, NI, KW, KS) in _INTERLEAVED_PRIME_ROWS
                       and not A_BF16 and LA is None)
    a_first_prime = (N, K) == (6144, 2048)
    interleave_prime = interleave_auto if INTERLEAVE_PRIME is None else bool(INTERLEAVE_PRIME)
    assert not interleave_prime or (not A_BF16 and FOLD == 1)
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
    CP = NI if NI % 2 else NI // 2
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
    # the measured rows (the module's _PARK_ALL_ROWS)
    park_auto = (N, K, RT, NI, KW, KS) in _PARK_ALL_ROWS and KW * RED_SLOT <= 160 * 1024
    park_all = park_auto if PARK_ALL is None else (bool(PARK_ALL) and KW > 1)
    owner_auto = not A_BF16 and not park_all and (N, K, RT, NI, KW, KS) in _OWNER_READ_ROWS
    owner_read = owner_auto if OWNER_READ is None else (bool(OWNER_READ) and KW > 1 and not park_all)
    out_cm = (W_CACHE_MOD if (N, K, RT, NI, KW, KS) in _NT_OUT_ROWS else 0) if OUT_CM is None else OUT_CM
    assert out_cm in (0, 1, 2, 16, 17, 18), out_cm
    out_kw = dict(cache_modifier=out_cm) if out_cm else {}
    LDS_BYTES = KW * RED_SLOT if park_all else max(16, (KW - 1) * RED_SLOT)
    assert LDS_BYTES <= 160 * 1024, "K-reduce scratch exceeds the LDS"
    assert N * K <= 0xFFFFFFFF

    # The 80-column N2560 rows share both operands within each XCD's L2 when
    # its contiguous eighth of a column-major, four-row group is a 2D tile block.
    # The same four-row order improves N6144 K2048's 96-column, two-K-wave
    # rows at M385-512. Keep the preceding rows and the four-K-wave QKV tile.
    grouped = (
        KS == 1 and not A_BF16 and (
            (N == 2560 and K == 6144 and NI == 5
             and RT in (4, 6, 7, 8, 9, 10, 11, 12) and KW == 3)
            or (N == 6144 and K == 2048 and NI == 6
                and RT in (7, 8) and KW == 2 and w_cm == W_CACHE_MOD)
        )
    )

    @fx.struct
    class Shared:
        red: fx.Array[fx.Uint8, LDS_BYTES, 16]

    name = (
        f"flydsl_mxfp8_dec_n{N}_k{K}_rt{RT}_ni{NI}_kw{KW}_ks{KS}_la{LA_W}x{LA_A}"
        + ("sb" if sched_barriers else "")
        + (("_fq" + (str(FOLD) if FOLD > 1 else "")) if A_BF16 else "")
        + (f"_wcm{w_cm}" if w_cm != W_CACHE_MOD else "")
        + ("_g4" if grouped else "")
        + ("_ip" if interleave_prime else "")
        + ({0: "", W_CACHE_MOD: "_ont"}.get(out_cm, f"_ocm{out_cm}"))
        + ("_rk0" if park_auto and not park_all else "")
        + ("_ownread" if owner_read else ("_ownread0" if owner_auto else ""))
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
        if const_expr(grouped):
            # Workgroup w runs on XCD w % 8. Deal each XCD a contiguous
            # eighth of the grouped order, with columns inside four-row groups.
            total = n_mb * NB
            x, q = wg % 8, wg // 8
            base, extra = total // 8, total % 8
            idx = x * base + (x < extra).select(x, extra) + q
            first = (idx // (4 * NB)) * 4
            rows_in = (n_mb - first < 4).select(n_mb - first, fx.Int32(4))
            rem = idx % (4 * NB)
            mb, nb = first + rem % rows_in, rem // rows_in
        else:
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
            arg_out, num_records_bytes=i32_m * (N * (2 if KS == 1 else 4 * KS))
        )

        n16 = [nb * NI + ni for ni in range_constexpr(NI)]
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
            ((n16[cp if NI % 2 else 2 * cp] // 2) * SCB) * 64 + sc_k + sc_lane
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
        w_shift = [(n16[ni] % 2) * 8 for ni in range_constexpr(NI)] if NI % 2 else None
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
        if const_expr(park_all):
            # K-wave `wave` parks tile i at (wave * RT * NI + i) KB, a lane's 16 B at lane * 16
            park_base = fx.Int32(fx.ptrtoint(smem.red.ptr)) + lane * 16

            def park_tile(i, v):
                llvm.StoreOp(
                    _raw(Vec(v).bitcast(fx.Int32)),
                    llvm.inttoptr(
                        ir.Type.parse("!llvm.ptr<3>"),
                        _raw(fx.Int64(park_base + (wave * (RT * NI) + i) * 1024)),
                    ),
                    alignment=16,
                )
        if const_expr(interleave_prime):
            for p in range_constexpr(max(LA_W, (LA_A + RT - 1) // RT)):
                if const_expr(not a_first_prime and p < LA_W):
                    w_q[p] = load_w_step(p)
                    fx.rocdl.sched_barrier(0)
                for rt in range_constexpr(RT):
                    if const_expr(p * RT + rt < LA_A):
                        a_q[p * RT + rt] = load_a(p * RT + rt)
                fx.rocdl.sched_barrier(0)
                if const_expr(a_first_prime and p < LA_W):
                    w_q[p] = load_w_step(p)
                    fx.rocdl.sched_barrier(0)
        else:
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
                        sw[ni if NI % 2 else ni // 2],
                        sa_v,
                        0 if NI % 2 else ni % 2,
                        rt % 2,
                        s % 2,
                    )
                # the loads depend on nothing, so the scheduler would pull later steps'
                # loads up to here: more operands alive than _lookahead budgeted
                if const_expr(sched_barriers):
                    fx.rocdl.sched_barrier(0)
                if const_expr(park_all and s == SW - 1 and rt >= 1):
                    # row tile rt - 1 is final: park it behind row tile rt's NI MFMAs (the
                    # >= 2 * 32 cycles they issue in cover the XDL-write -> LDS-read wait
                    # states, which the compiler cannot see through the asm); the fences
                    # keep the writes there
                    if const_expr(not sched_barriers):
                        fx.rocdl.sched_barrier(0)
                    for ni in range_constexpr(NI):
                        park_tile((rt - 1) * NI + ni, acc[rt - 1][ni])
                    fx.rocdl.sched_barrier(0)
            w_q[s] = None

        def store_tile(i, v):
            rt, ni = i // NI, i % NI
            row = mb * BM + rt * 16 + l16
            valid = row < i32_m if FOLD == 1 else (row < i32_m) & (l16 < RPT)
            col = (nb * NI + ni) * 16 + q16 * 4
            if const_expr(KS == 1):
                buffer_ops.buffer_store(
                    v.to(fx.BFloat16).bitcast(fx.Int32),
                    out_rsrc,
                    (row * N + col) * 2,
                    offset_is_bytes=True,
                    mask=valid,
                    **out_kw,
                )
            else:
                buffer_ops.buffer_store(
                    v.bitcast(fx.Int32),
                    out_rsrc,
                    ((kz * i32_m + row) * N + col) * 4,
                    offset_is_bytes=True,
                    mask=valid,
                    **out_kw,
                )

        def store_tile_dyn(i, v, live):
            """store_tile for a wave-uniform runtime tile index (``live``: i < RT * NI): the
            lane's part of the offset is fixed, the tile's part goes in the scalar offset"""
            rt_d, ni_d = i // NI, i % NI
            esz = 2 if KS == 1 else 4
            # rows past M: the lane offset is masked out of the buffer (as store_tile's)
            valid = l16 < i32_m - (mb * BM + rt_d * 16)
            if const_expr(FOLD > 1):
                valid = valid & (l16 < RPT)
            if const_expr(live is not None):
                valid = valid & live
            data = v.to(fx.BFloat16).bitcast(fx.Int32) if KS == 1 else v.bitcast(fx.Int32)
            buffer_ops.buffer_store(
                data,
                out_rsrc,
                lane_out,
                offset_is_bytes=True,
                mask=valid,
                soffset_bytes=(rt_d * (16 * N) + ni_d * 16) * esz,
                **out_kw,
            )

        if const_expr(park_all):
            last = _pin_accumulators([acc[RT - 1][ni] for ni in range(NI)])
            for ni in range_constexpr(NI):
                park_tile((RT - 1) * NI + ni, last[ni])
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            if const_expr(KS == 1):
                lane_out = ((mb * BM + l16) * N + nb * (NI * 16) + q16 * 4) * 2
            else:
                lane_out = (((kz * i32_m) + mb * BM + l16) * N + nb * (NI * 16) + q16 * 4) * 4
            n_t = RT * NI
            n_j = -(-n_t // KW)  # tiles per wave (the last one only for waves < n_t % KW)
            red_base = park_base + wave * 1024  # this wave's first tile, K-wave 0's slot
            # a wave whose last tile index runs past the tiles reads tile n_t - 1 instead
            # and masks the store
            last_i = (wave + (n_j - 1) * KW < n_t).select(
                wave + (n_j - 1) * KW, fx.Int32(n_t - 1)
            )

            def lds_vec(addr):
                return Vec(
                    llvm.load(
                        T.vec(4, T.i32),
                        llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(addr))),
                        alignment=16,
                    )
                ).bitcast(fx.Float32)

            # all LDS reads of a group of tiles go out before its first store (the stores
            # would otherwise order every read behind the previous tile's store)
            GROUP = max(1, 24 // KW)
            for j0 in range_constexpr(0, n_j, GROUP):
                js = list(range(j0, min(n_j, j0 + GROUP)))
                parts = []
                for j in js:
                    if const_expr(j * KW + KW <= n_t):
                        base = red_base + j * KW * 1024
                    else:
                        base = park_base + last_i * 1024
                    parts.append([lds_vec(base + w * n_t * 1024) for w in range_constexpr(KW)])
                for j, ps in zip(js, parts):
                    v = ps[0]
                    for w in range_constexpr(1, KW):
                        v = v + ps[w]
                    full = j * KW + KW <= n_t
                    store_tile_dyn(
                        wave + j * KW if full else last_i,
                        v,
                        None if full else (wave + j * KW < n_t),
                    )
        else:
            flat = _pin_accumulators([acc[rt][ni] for rt in range(RT) for ni in range(NI)])
            if const_expr(KW > 1):
                # K-reduce: tile i belongs to K-wave i % KW; the other K-waves park their
                # partial sums of it in LDS, the owner adds the KW partials in wave order
                # (the sums one reducing wave made) and stores the tile
                lds_base = fx.Int32(fx.ptrtoint(smem.red.ptr))

                def lds_ptr(i, w):
                    """where K-wave w parks tile i (w != the owner)"""
                    o = i % KW
                    off = (w if w < o else w - 1) * RED_SLOT + i * 1024
                    return llvm.inttoptr(
                        ir.Type.parse("!llvm.ptr<3>"),
                        _raw(fx.Int64(lds_base + off + lane * 16)),
                    )

                def park(w):
                    for i in range_constexpr(RT * NI):
                        if const_expr(i % KW != w):
                            llvm.StoreOp(
                                _raw(Vec(flat[i]).bitcast(fx.Int32)),
                                lds_ptr(i, w),
                                alignment=16,
                            )

                def owned_value(v):
                    if const_expr(owner_read):
                        # Side-effecting AGPR reads stay after the ownership branch.
                        # Other rows retain their compiler-selected copies and ISA.
                        return Vec.from_elements([_acc_read(v, z) for z in range(4)], fx.Float32)
                    return Vec(v)

                def reduce_store(o):
                    for i in range_constexpr(RT * NI):
                        if const_expr(i % KW == o):
                            v = None
                            for w in range_constexpr(KW):
                                p = (
                                    owned_value(flat[i])
                                    if w == o
                                    else Vec(
                                        llvm.load(
                                            T.vec(4, T.i32), lds_ptr(i, w), alignment=16
                                        )
                                    ).bitcast(fx.Float32)
                                )
                                v = p if v is None else v + p
                            store_tile(i, v)

                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                for w in range_constexpr(KW):
                    if wave == w:
                        park(w)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                for w in range_constexpr(KW):
                    if wave == w:
                        reduce_store(w)
            else:
                for i in range_constexpr(RT * NI):
                    store_tile(i, Vec(flat[i]))

    @flyc.jit
    def launch(
        A: fx.Tensor,
        W: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W_scale: fx.Tensor,
        i32_m: fx.Int32,
        a_scale_bytes: fx.Int32,
        n_mb: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, a_scale_bytes, n_mb, _addr(W), _addr(W_scale), _addr(A_scale), _addr(A),
            _addr(OUT),
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(64 * KW, 1, 1), stream=stream)  # fmt: skip

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    launch.k_split = KS
    launch.n_blocks = NB
    launch.block_m = BM
    launch.fold = FOLD
    return launch
