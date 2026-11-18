"""Fused MoE monolayer, v2 prelude (user-approved departures from native, 2026-11-03):

  (1) MXFP4 quant per TOKEN (not per route) into a token-major payload [T, K/2] + row-major e8m0 [T, K/32] at
      kernel start (x is an input): the same 1x32 e8m0 math (aiter's quant helper), only layout / order change.
  (2) top-k in the router's epilogue: router tiles (16 x 16, K split 4: 68 a 16-token block) arrive on a per-block
      counter; the last ntok arrivals each do one token's softmax / top-8 / renorm and claim its route slots
      (global atomics), the last token to finish runs the prefix sum. No separate top-k stage.
  (3) router operands by TDM at kernel start (x tile + 1 MB of router weights split over the tiles), WMMA from LDS.
  (4) gate_up gathers its A rows (and row-major A scales) by token index from (1) with TDM gather loads
      (16-bit indices, 16 rows an op) through the expert-major slot -> token map top-k writes; native tile math.
  The routing (prefix sum) is published as 16 copies of 256 tagged words ((tag & 0xFFFF) << 16 | psum[e]): one
  poll gives every CTA the whole prefix sum (no second round trip); expert / start / bound of any row from it.

Stages after the routing (gate_up, down pipelining, gather) as in mono_moe.py (see NOTES.md).
"""

import os
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _emit_quant_block_loop,
    _quant_layout,
)
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import gpu, rocdl, tdm_ops
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen3_5_mono import prims
from atom.models.qwen3_5_mono.prims import (
    AGENT,
    CM_DEV,
    ld_dw,
    rsrc_at,
    st_dw,
    st_f32,
    uniform,
    wmma_bf16,
)

from . import gemm_tdm
from .mono_moe import _swz, fence_acq, fence_rel

NCTA = 1792
E, TOPK, D, I = 256, 8, 2048, 128
LGS = 264
TMAX = 128
CMMAX = 5120
NRN, RKS = (
    17,
    4,
)  # RKS: the shared gate_up's K split (and the partial buffers' depth bound)
RKS_R = int(
    os.environ.get("MONO_RKS_R", "8")
)  # (3) the router's K split (8: T <= 112 only, role ranges)
NRT = NRN * RKS_R  # router tiles a 16-token block
RKS_MAX = 8
# flag words (int32)
F_CNT = 0  # (unused)
F_DONE = 2  # CTAs done this launch (the last resets it)
F_EPOCH = 4  # launch epoch
F_QC1 = 32  # 16 lines
F_QC2 = 544
F_TKC = 576
F_QFR = 1024  # 64 copies of the quant-done flag
F_RC = (
    3072  # [m] router arrival counters, a line each (monotonic: 68 arrivals a launch)
)
F_EC = 3328  # [256] expert route counters
F_GU = 4096  # [m][4]
F_DN = F_GU + (CMMAX // 16) * 4  # [m][16]
NPSC = 16
F_PSC = 11264  # NPSC copies x 256 tagged prefix-sum words
F_SC = (
    F_PSC + NPSC * 256
)  # [m][8] shared gate_up arrival counters, a line each (8 arrivals a launch)
F_ACT = F_SC + 64 * 32  # [m][8] shared act tile flags
F_SD = F_ACT + 64  # [m][16] shared down tile flags
F_RC8 = (
    F_SD + 128
)  # router arrival counters of the K-split-8 build (reset by each block's last tile)
F_RD = F_RC8 + 8 * 32  # [m] router block done flag (tag), a line each
F_TKD = F_RD + 8 * 32  # [t] top-k token stores-done flags (TKO)
F_DBG = F_TKD + 128  # debug-spin record: site, bid, hit
F_WORDS = F_DBG + 32
SH = 128  # shared intermediate a rank
NSTAMP = 24
STAMP_NAMES = [
    "start",
    "router_end",
    "topk_wait",
    "topk_end",
    "psum_wait",
    "psum_end",
    "quant_wait",
    "quant_end",
    "gu_wait",
    "gu_end",
    "dn_wait",
    "dn_end",
    "gather_wait",
    "gather_end",
    "end",
    "ps_seen",
    "router_tdm",
    "gu_b_issued",
    "sh_act_end",
    "sh_down_end",
    "ar_rows_in",
    "norm_end",
    "x22",
    "x23",
]
DN_SLOT = 10240


from flydsl.compiler.ast_rewriter import ASTRewriter

from .tspin import DEADLINE_TICKS, spin_t
from .tspin import clk as _clk

POLL_SLEEP = int(os.environ.get("MONO_POLL_SLEEP", "0"))
# MONO_IPF=<n>: n s_prefetch_inst_pc_rel of 32 lines each, MONO_IPF_STEP bytes apart, at kernel start (the code
# after the hardware's launch prefetch, refilled at once when another large kernel ran in between). Experiment.
IPF = int(os.environ.get("MONO_IPF", "0"))
IPF_STEP = int(os.environ.get("MONO_IPF_STEP", "4096"))
# MONO_INSTPREF=<lines>: the descriptor's INST_PREF_SIZE (launch prefetch, 128 B lines; LLVM sets 255 = ~32 KB) set to
# <lines> after compile (instpref.py). -1: LLVM's value.
INSTPREF = int(os.environ.get("MONO_INSTPREF", "-1"))
# MONO_ROLES (code-size accounting only: a build without some roles does not work): bits 1 quant, 2 shared expert,
# 4 router + top-k + prefix sum, 8 gate_up (+ its early B), 16 down (+ early B), 32 gather + all-reduce + norm
ROLES = int(os.environ.get("MONO_ROLES", "63"))
# MONO_ROLL: rolled (unroll=1) loops instead of straight-line code, smaller instruction-cache footprint, same
# arithmetic order: bits 1 top-k selection, 2 gather (per-lane route lookups + rolled accumulate), 4 shared-expert
# down sub-tiles, 8 all-reduce norm stores, 16 down tiles (one issue_b / issue_a and two finish call sites, runtime
# LDS slot, instead of three copies of each)
ROLL = int(os.environ.get("MONO_ROLL", "0"))
# MONO_WCTA=W (7): W waves a CTA, NCTA / W CTAs (one a CU: W x 44 KB of LDS), wave w of CTA b = virtual one-wave CTA
# W b + w: every role range (quant, shared expert, router, tiles, gather) is contiguous a CU, so a CU executes only
# its roles' code. 0: NCTA one-wave CTAs.
WCTA = int(os.environ.get("MONO_WCTA", "0"))
# With MONO_WCTA: MONO_WTILE=1 (default) gives the gate_up / down tiles an id interleaved over the CUs (wave w of CTA
# b: w * NCTA / W + b) so every CU gets tiles (the CU-contiguous id would put all 1,192 gate_up tiles on 2/3 of the
# CUs); MONO_GLO=1 (default) puts the gather-reduce on the lowest ids (with quant / the first shared-expert CUs)
# instead of the top ones (the router CUs); 2: on interleaved ids (the top tbid: one gather wave on each of 2T CUs).
WTILE = int(os.environ.get("MONO_WTILE", "1"))
GLO = int(os.environ.get("MONO_GLO", "1"))


def _sel(vals, i):
    """vals[i] for an index i (uniform or per lane) over a short Python list."""
    v = vals[0]
    for k in range(1, len(vals)):
        v = (i == k).select(vals[k], v)
    return v


def _readlane_f32(x, k):
    return fx.Int32(
        rocdl.readlane(
            T.i32, fx.Float32(x).bitcast(fx.Int32).ir_value(), fx.Int32(k).ir_value()
        )
    ).bitcast(fx.Float32)


if INSTPREF >= 0:
    from .instpref import install as _install_instpref

    _install_instpref(
        "moe_mono2", INSTPREF
    )  # s_sleep between the routing polls (idle CTAs)


@ASTRewriter.transform
def spin_sleep(load_all, pending, k):
    """prims.spin with s_sleep k between polls (bounded, traps)."""
    v = load_all()
    t0 = _clk()
    late = fx.Int32(0)
    while pending(v) & (late == 0):
        rocdl.s_sleep(k)
        v = load_all()
        late = ((_clk() - t0) > DEADLINE_TICKS).select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        prims._trap()
    return v


DEBUG_SPIN = int(os.environ.get("MONO_DEBUG_SPIN", "0"))


@ASTRewriter.transform
def _spin_dbg(load_all, pending, site, rfl, bid_):
    """Debug spin: on exhausting a short limit, record (site, bid) in the flag buffer's debug words and RETURN
    (the launch's results are then garbage, but the host can read where it hung)."""
    v = load_all()
    n = fx.Int32(0)
    while pending(v) & (n < (1 << 21)):
        prims.compiler_fence()
        v = load_all()
        n = n + 1
    if n >= (1 << 21):
        bo.buffer_store(
            fx.Int32(site).ir_value(), rfl, fx.Int32(F_DBG), cache_modifier=CM_DEV
        )
        bo.buffer_store(
            fx.Int32(bid_).ir_value(), rfl, fx.Int32(F_DBG + 1), cache_modifier=CM_DEV
        )
        bo.buffer_store(
            fx.Int32(1).ir_value(), rfl, fx.Int32(F_DBG + 2), cache_modifier=CM_DEV
        )
    return v


def _make_spin(dbg):
    def spin(load_all, pending):
        if DEBUG_SPIN and "rfl" in dbg:
            dbg["site"] = dbg.get("site", 0) + 1
            return _spin_dbg(load_all, pending, dbg["site"], dbg["rfl"], dbg["bid"])
        return spin_t(load_all, pending)

    return spin


def _gu_task():
    return gemm_tdm.build_gemm_task(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        2 * I,
        D,
        16,
        64,
        512,
        1,
        1,
        0,
        2,
        1,
        None,
        E,
        1,
        0,
        None,
        float("inf"),
        stage1_quant_out=1,
        quant_wmma_rep=1,
        cluster_n=1,
        cluster_m=1,
        num_waves_per_tensor_tdm=1,
        persistent_workers=1,
        row_major_ascale=1,
    )


def _dn_task():
    return gemm_tdm.build_gemm_task(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        D,
        I,
        16,
        128,
        128,
        1,
        1,
        0,
        1,
        1,
        None,
        E,
        0,
        0,
        None,
        float("inf"),
        cluster_n=1,
        cluster_m=1,
        num_waves_per_tensor_tdm=1,
        persistent_workers=1,
    )


# M3: inbox (cached hipMalloc memory, IPC-mapped by every rank, as aiter's Gluon all-reduce) in int32 dwords:
# [2 parity][4 src rank][TMAX][2 halves][6 lines][32 src lanes][4 dwords] (a store instruction = 512 contiguous
# bytes, full lines over the fabric): a sender lane's 16 data dwords go out
# as 6 16-byte lines of 3 data dwords + the launch tag (one atomic store each; no flag, no ack wait), the receiver
# polls the lines until every tag matches. parity = tag & 1 (a rank writes launch L + 2 only after every rank has
# finished launch L's gather, so 2 slots suffice).
TP = 4
LANE_DW = 24
TOK_DW = 2 * 32 * LANE_DW
PEER_BYTES = 2 * TP * TMAX * TOK_DW * 4
EPS = 1e-6
# (1) L2 prefetch of the expert weights by prelude-idle CTAs (after the shared-down CTAs, below the router CTAs):
# PF 0 off, 1 all experts (gate_up weights + scales first, then down); PF_DELAY_US after the CTA's start
PF = int(os.environ.get("MONO_PF", "0"))
DNMAP = int(os.environ.get("MONO_DNMAP", "1"))
GUDN = int(os.environ.get("MONO_GUDN", "0"))
TKO = int(
    os.environ.get("MONO_TKO", "0")
)  # (3) top-k arrival before its stores complete (per-token done flags)
SH_DELAY_US = float(
    os.environ.get("MONO_SH_DELAY_US", "0")
)  # (3) delay the shared-expert TDM (measured: worse)
DN_BASE = (
    11776 if GUDN else 0
)  # down slots [11776 + k 10240): clear of the gate_up epilogue staging and buffer 1
PF_DELAY_US = float(os.environ.get("MONO_PF_DELAY_US", "0"))


def build(
    timing=False, upto=99, shared=True, ar=False, rks_r=None, gudn=None, tko=None
):
    TKO_ = TKO if tko is None else tko
    GUDN_ = GUDN if gudn is None else gudn
    DN_BASE_ = 11776 if GUDN_ else 0
    RKS_R_ = rks_r or RKS_R
    NRT_ = NRN * RKS_R_
    F_RC_ = F_RC if RKS_R_ == 4 else F_RC8
    dbg_ = {}
    spin = _make_spin(dbg_)
    QL = _quant_layout(D, "fp4", 1)

    @flyc.kernel(
        name=f"moe_mono2{'_timing' if timing else ''}{'_sh' if shared else ''}"
        f"{'_ar' if ar else ''}_pf{int(PF)}_{int(PF_DELAY_US * 100)}"
        f"_dm{int(DNMAP)}_gd{int(GUDN_)}_sd{int(SH_DELAY_US * 100)}_ps{int(POLL_SLEEP)}"
        f"_rk{int(RKS_R_)}_dbg{int(DEBUG_SPIN)}_tko{int(TKO_)}"
        + (f"_ipf{int(IPF)}_{int(IPF_STEP)}" if IPF else "")
        + (f"_ip{int(INSTPREF)}" if INSTPREF >= 0 else "")
        + (f"_roles{int(ROLES)}" if ROLES != 63 else "")
        + (f"_rl{int(ROLL)}" if ROLL else "")
        + (f"_w{int(WCTA)}{'t' if WTILE else ''}g{int(GLO)}" if WCTA else ""),
        known_block_size=[32 * max(WCTA, 1), 1, 1],
    )
    def kernel(
        x: fx.Pointer,
        gate: fx.Pointer,
        w13: fx.Pointer,
        s13: fx.Pointer,
        w2: fx.Pointer,
        s2: fx.Pointer,
        lg: fx.Pointer,
        ti: fx.Pointer,
        tw: fx.Pointer,
        tsl: fx.Pointer,
        etok: fx.Pointer,
        a1t: fx.Pointer,
        a1ts: fx.Pointer,
        a2: fx.Pointer,
        a2s: fx.Pointer,
        gout: fx.Pointer,
        out: fx.Pointer,
        fl: fx.Pointer,
        stp: fx.Pointer,
        shgu: fx.Pointer,
        shdn: fx.Pointer,
        shp: fx.Pointer,
        act: fx.Pointer,
        shout: fx.Pointer,
        sg: fx.Pointer,
        res: fx.Pointer,
        xo: fx.Pointer,
        nw: fx.Pointer,
        pr0: fx.Int64,
        pr1: fx.Int64,
        pr2: fx.Int64,
        pr3: fx.Int64,
        rank: fx.Int32,
        T_: fx.Int32,
        cm: fx.Int32,
    ):
        gu_task, gu_arena = _gu_task()
        dn_task, dn_arena = _dn_task()
        assert dn_arena <= DN_SLOT and 3 * DN_SLOT <= gu_arena
        ARENA = (max(gu_arena, dn_arena) + 255) // 256 * 256
        gemm_tdm.WAVE_CTA[0] = WCTA > 0
        base_ptr = fx.SharedAllocator().allocate(ARENA * max(WCTA, 1))._ptr
        p8_shared = fx.PointerType.get(
            elem_ty=fx.Int8.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
        )
        if fx.const_expr(WCTA > 0):
            wave_id = uniform(fx.Int32(fx.thread_idx.x) >> 5)
            L0 = fx.Int32(fx.ptrtoint(base_ptr)) + wave_id * ARENA
            base_ptr = fx.inttoptr(p8_shared, L0)
            bid = fx.Int32(fx.block_idx.x) * WCTA + wave_id
            tbid = (
                (wave_id * (NCTA // WCTA) + fx.Int32(fx.block_idx.x))
                if fx.const_expr(WTILE)
                else bid
            )
            lane = fx.Int32(fx.thread_idx.x) & 31
        else:
            L0 = fx.Int32(fx.ptrtoint(base_ptr))
            bid = fx.Int32(fx.block_idx.x)
            tbid = bid
            lane = fx.Int32(fx.thread_idx.x)
        gemm_tdm.LDS_BASE[0] = base_ptr
        base_i8 = fx.recast_iter(p8_shared, base_ptr)
        T_ = fx.Int32(T_)

        def _lds_sync():
            if fx.const_expr(WCTA > 0):
                rocdl.s_wait_dscnt(0)
            else:
                prims.lds_barrier()

        def addr(p):
            return fx.Int64(fx.ptrtoint(p))

        r_fl = rsrc_at(addr(fl))
        dbg_["rfl"] = r_fl
        dbg_["bid"] = bid
        dbg_["site"] = 0
        r_lg = rsrc_at(addr(lg))
        r_ti = rsrc_at(addr(ti))
        r_tw = rsrc_at(addr(tw))
        r_tsl = rsrc_at(addr(tsl))
        r_etok = rsrc_at(addr(etok))
        r_st = rsrc_at(addr(stp))
        r_shp = rsrc_at(addr(shp))
        r_act = rsrc_at(addr(act))
        r_shout = rsrc_at(addr(shout))
        r_sg = rsrc_at(addr(sg))
        fl_p = fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4), fl
        )

        def stamp(i):
            if fx.const_expr(timing):
                clk = fx.Int64(
                    _llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], [])
                )
                st_dw(
                    r_st,
                    (bid * NSTAMP + i) * 2,
                    [fx.Int32(clk & 0xFFFFFFFF), fx.Int32(clk >> 32)],
                )

        stamp(0)
        if fx.const_expr(IPF > 0):
            _llvm.InlineAsmOp(
                res=None, operands_=[], has_side_effects=True, is_align_stack=False, constraints="",
                asm_string="\n".join(f"s_prefetch_inst_pc_rel {k * IPF_STEP}, null, 31" for k in range(IPF)),
            )  # fmt: skip
        t_start = fx.Int64(
            _llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], [])
        )
        MT = (T_ + 15) // 16
        n_rt = MT * NRT_
        n_q = T_  # quant tasks: one token a CTA (all 8 k groups)
        # roles: quant [0, T); shared gate_up tiles [T, T + 64 MT); shared down tiles next 16 MT; router from the top
        sg_id = bid - T_
        is_sg = (sg_id >= 0) & (sg_id < MT * 64) if shared else (bid < 0)
        sd_id = bid - T_ - MT * 64
        is_sd = (sd_id >= 0) & (sd_id < MT * 16) if shared else (bid < 0)

        # ------------------------------------------------------------ router operands: TDM at kernel start
        r_id = NCTA - 1 - bid
        is_rt = r_id < n_rt
        rm = r_id // NRT_
        rrem = r_id - rm * NRT_
        rn = rrem // RKS_R_
        rks = rrem - rn * RKS_R_
        XP = 1040  # LDS row pitch (1024 B + 16 B pad)
        KBR = 2 * D // RKS_R_  # router K bytes a tile
        XPR = KBR + 16

        def tdm_rows(gptr, row0, ext, rows, row_b, g_stride_b, col0_b, lds_off, pitch):
            """TDM 2D load: ``rows`` rows x ``row_b`` bytes from byte (row0 * stride + col0) into LDS rows of ``pitch``."""
            g8 = fx.recast_iter(fx.Int8, gptr)
            gt = fx.Tensor(
                fx.make_view(
                    g8 + (fx.Int64(row0) * g_stride_b + fx.Int64(col0_b)),
                    fx.make_layout((rows, row_b), (g_stride_b, 1)),
                )
            )
            atom = fx.rocdl.make_tdm_atom(
                gt,
                [ext, None],
                strides=[g_stride_b, None],
                num_warps=1,
                pad_interval=row_b,
                pad_amount=pitch - row_b,
            )
            dst = fx.Tensor(
                fx.make_view(
                    base_i8 + lds_off, fx.make_layout((rows, row_b), (pitch, 1))
                )
            )
            fx.copy(atom, gt, dst)

        def tile16_mma(kb=1024):
            """[16 x kb/2] x [16 x kb/2]^T from LDS (x at 0, w at 16 pitch): lane column / rows 8 h + i of D."""
            row = lane % 16
            h = lane // 16
            pit = kb + 16
            acc = [fx.Float32(0.0) for _ in range(8)]
            for s_ in fx.range_constexpr(kb // 64):
                dwi = (row * pit + h * (kb // 2) + 32 * s_) // 4
                acc = wmma_bf16(
                    prims.lds_ld8w(L0, dwi), prims.lds_ld8w(L0 + 16 * pit, dwi), acc
                )
            return acc

        if is_rt & fx.const_expr(ROLES & 4 != 0):
            tdm_rows(x, rm * 16, T_ - rm * 16, 16, KBR, 2 * D, rks * KBR, 0, XPR)
            tdm_rows(
                gate,
                rn * 16,
                fx.Int32(E + 1) - rn * 16,
                16,
                KBR,
                2 * D,
                rks * KBR,
                16 * XPR,
                XPR,
            )
        # shared expert operands, early TDM (x and weights are inputs)
        sm = sg_id // 64
        srem = sg_id - sm * 64
        sn = srem // RKS
        sks = srem - sn * RKS
        dm_ = sd_id // 16
        dn_s = sd_id - dm_ * 16
        SDP = 272  # shared down LDS row pitch (256 B + 16)
        if fx.const_expr(  # noqa: SIM102 (a compile-time test, then a runtime one)
            SH_DELAY_US > 0
        ):
            # (3) let the router's operand loads go first: the shared expert is off the critical path
            if is_sg | is_sd:
                tgt_ = t_start + int(SH_DELAY_US * 100)
                spin(
                    lambda: fx.Int64(
                        _llvm.call_intrinsic(
                            T.i64, "llvm.readsteadycounter", [], [], []
                        )
                    ),
                    lambda c_: c_ < tgt_,
                )
        if is_sg & fx.const_expr(ROLES & 2 != 0):
            tdm_rows(x, sm * 16, T_ - sm * 16, 16, 1024, 2 * D, sks * 1024, 0, XP)
            tdm_rows(
                shgu,
                sn * 16,
                fx.Int32(2 * SH) - sn * 16,
                16,
                1024,
                2 * D,
                sks * 1024,
                16 * XP,
                XP,
            )
        if is_sd & fx.const_expr(ROLES & 2 != 0):
            tdm_rows(shdn, dn_s * 128, fx.Int32(128), 128, 2 * SH, 2 * SH, 0, 0, SDP)

        # launch tag: the epoch word (advanced at the end of every launch by its last CTA) + 1, in [1, 2^30]:
        # positive, consecutive launches alternate parity, a flag word aliases only after 2^30 launches
        ep = uniform(
            fx.Int32(
                bo.buffer_load(
                    r_fl,
                    fx.Int32(F_EPOCH),
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
        )
        tag = (ep & 0x3FFFFFFF) + 1
        tag16 = tag & 0x7FFF  # psum words: (tag16 << 16) | psum stays a positive int32

        def acquire():
            fence_acq()

        def publish(word):
            fence_rel()
            bo.buffer_store(tag.ir_value(), r_fl, fx.Int32(word), cache_modifier=CM_DEV)

        def wave_any(b):
            v = b.select(fx.Int32(1), fx.Int32(0))
            for o in (16, 8, 4, 2, 1):
                v = v | v.shuffle_xor(o, 32)
            return v != 0

        def wait_words(word_of_lane, live):
            def load_all():
                return fx.Int32(
                    bo.buffer_load(
                        r_fl,
                        word_of_lane,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )

            spin(load_all, lambda v: wave_any(live & (v != tag)))
            acquire()

        def wait4(word):
            def load_all():
                return fx.Vector(
                    bo.buffer_load(
                        r_fl,
                        fx.Int32(word),
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )

            spin(
                load_all,
                lambda v: (fx.Int32(v[0]) != tag)
                | (fx.Int32(v[1]) != tag)
                | (fx.Int32(v[2]) != tag)
                | (fx.Int32(v[3]) != tag),
            )
            acquire()

        def arrive2(j1, n_total, done_flag_base):
            """Two-level arrival (16 counters, then one); the last arriver resets both and publishes 64 flag copies."""
            fence_rel()
            n1 = (n_total - j1 + 15) // 16
            a1 = fx.Int32(0)
            if lane == 0:
                a1 = fx.Int32(
                    fx.atomic_add(
                        fl_p + (F_QC1 + 32 * j1), fx.Int32(1), syncscope=AGENT
                    )
                )
            a1 = uniform(a1)
            if a1 == n1 - 1:
                st_dw(r_fl, F_QC1 + 32 * j1, [fx.Int32(0)], cm=CM_DEV)
                fence_rel()
                a2_ = fx.Int32(0)
                if lane == 0:
                    a2_ = fx.Int32(
                        fx.atomic_add(fl_p + F_QC2, fx.Int32(1), syncscope=AGENT)
                    )
                a2_ = uniform(a2_)
                if a2_ == (n_total < 16).select(n_total, fx.Int32(16)) - 1:
                    acquire()
                    st_dw(r_fl, F_QC2, [fx.Int32(0)], cm=CM_DEV)
                    fence_rel()
                    for j in fx.range_constexpr(2):
                        st_dw(
                            r_fl,
                            done_flag_base + 32 * (lane + 32 * j),
                            [tag],
                            cm=CM_DEV,
                        )

        # ------------------------------------------------------------ quant per token (x is an input): (t, k group)
        if fx.const_expr(upto < 1):
            return
        if (bid < n_q) & fx.const_expr(ROLES & 1 != 0):
            stamp(6)
            t = bid
            block_in_wave = fx.Uint32(lane) // fx.Uint32(QL.lanes_per_mx_block)
            lane_in_block = fx.Uint32(lane) - block_in_wave * QL.lanes_per_mx_block
            from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor

            qc = SimpleNamespace(
                i32=T.i32,
                f32=T.f32,
                block_iters=QL.block_iters,
                payload_base=addr(a1t),
                payload_bytes_per_row=QL.payload_bytes_per_row,
                hidden_base=addr(x),
                feat_bytes_per_row=D * 2,
                feat_row_i32=fx.Uint32(t),
                prequantized=False,
                payload_dwords_per_lane=QL.payload_bytes_per_lane // 4,
                src_scale_base=addr(x),
                src_scale_bytes_per_row=0,
                mx_blocks_per_wave_iter=QL.mx_blocks_per_wave_iter,
                mx_blocks_per_row=QL.mx_blocks_per_row,
                amax_shuffle_dists=QL.amax_shuffle_dists,
                is_fp8=False,
                use_native=QL.use_native,
                use_pk8=QL.use_pk8,
                mx_dtype=QL.mx_dtype,
                c0_i32=fx.Int32(0),
                c1_i32=fx.Int32(1),
                c4_i32=fx.Int32(4),
                c23_i32=fx.Int32(23),
                c254_i32=fx.Int32(254),
                c0_f32=fx.Float32(0.0),
                c_wave=fx.Int32(32),
                c_elems_per_lane=fx.Int32(QL.elems_per_lane),
                c_payload_bytes_per_block=fx.Int32(QL.payload_bytes_per_block),
                c_payload_bytes_per_lane=fx.Int32(QL.payload_bytes_per_lane),
                c_wmma_rep=fx.Int32(1),
                block_in_wave=block_in_wave,
                lane_in_block=lane_in_block,
                is_block_lead=lane_in_block == 0,
                dests=[
                    SimpleNamespace(
                        payload_row_i32=fx.Uint32(t), scale_row_dword_base=fx.Uint32(0)
                    )
                ],
                scale_t=ptr_buf_tensor(a1ts, fx.Int8),
                scale_t_i32=ptr_buf_tensor(a1ts, fx.Int32),
                lane=fx.Uint32(lane),
                wmma_rep=1,
                row_major_scale=False,
                compact_scale=True,
                scale_rsrc=rsrc_at(addr(a1ts)),
                c_scale_dwords_per_row=fx.Int32(D // 128),
            )
            _emit_quant_block_loop(qc)
            arrive2(bid % 16, n_q, F_QFR)
            stamp(7)

        # ------------------------------------------------------------ shared expert (idle CTAs during the prelude)
        if is_sg & fx.const_expr(ROLES & 2 != 0):
            # gate_up [16 tok x 16 cols] K split 4 -> fp32 partials; the last of the 8 tiles of (cols n8, n8 + 128)
            # sums them (bf16 GEMM outputs) and writes SiLU(gate) * up (bf16) for those 16 x 16
            tdm_ops.tensor_wait(0)
            acc = tile16_mma()
            col = sn * 16 + lane % 16
            for i in fx.range_constexpr(8):
                t2 = sm * 16 + (lane // 16) * 8 + i
                if t2 < T_:
                    st_f32(r_shp, (sks * TMAX + t2) * (2 * SH) + col, [acc[i]])
            fence_rel()
            n8 = sn % 8
            sa = fx.Int32(0)
            if lane == 0:
                sa = fx.Int32(
                    fx.atomic_add(
                        fl_p + (F_SC + 32 * (sm * 8 + n8)), fx.Int32(1), syncscope=AGENT
                    )
                )
            sa = uniform(sa)
            if sa == 7:
                st_dw(
                    r_fl, F_SC + 32 * (sm * 8 + n8), [fx.Int32(0)], cm=CM_DEV
                )  # reset for the next launch
                acquire()
                tr = sm * 16 + lane % 16
                c0 = n8 * 16 + (lane // 16) * 8
                [fx.Float32(0.0)] * 8
                [fx.Float32(0.0)] * 8
                gl = []
                ul = []
                for k4 in fx.range_constexpr(RKS):
                    o = (k4 * TMAX + tr) * (2 * SH) + c0
                    gl.append(
                        prims.ld_f32(r_shp, o, cm=CM_DEV)
                        + prims.ld_f32(r_shp, o + 4, cm=CM_DEV)
                    )
                    ul.append(
                        prims.ld_f32(r_shp, o + SH, cm=CM_DEV)
                        + prims.ld_f32(r_shp, o + SH + 4, cm=CM_DEV)
                    )
                av = []
                for i in fx.range_constexpr(8):
                    gv = gl[0][i]
                    uv = ul[0][i]
                    for k4 in fx.range_constexpr(1, RKS):
                        gv = gv + gl[k4][i]
                        uv = uv + ul[k4][i]
                    av.append(
                        prims.silu(prims.bf16r(gv)) * prims.bf16r(uv)
                    )  # act_and_mul: fp32, bf16 out
                if tr < T_:
                    st_dw(
                        r_act,
                        (tr * SH + c0) // 2,
                        [prims.bf16_pack(av[2 * j], av[2 * j + 1]) for j in range(4)],
                    )
                publish(F_ACT + sm * 8 + n8)
                stamp(18)
        if is_sd & fx.const_expr(ROLES & 2 != 0):
            # down: [16 tok x 128 in] x [128 out x 128 in]^T, weights already in flight
            wait4(F_ACT + dm_ * 8)
            wait4(F_ACT + dm_ * 8 + 4)
            tdm_rows(
                act, dm_ * 16, T_ - dm_ * 16, 16, 2 * SH, 2 * SH, 0, 128 * SDP, SDP
            )
            tdm_ops.tensor_wait(0)
            row = lane % 16
            h = lane // 16
            if fx.const_expr(ROLL & 4):
                # one loop body for the 8 sub-tiles, each staged right away above the act rows (the weights at L0
                # are still read by the later sub-tiles)
                LST = L0 + 128 * SDP + 16 * SDP
                for sub_ in range(8, unroll=1):
                    sub = fx.Int32(sub_)
                    acc = [fx.Float32(0.0) for _ in range(8)]
                    for s_ in fx.range_constexpr(4):
                        a_ = prims.lds_ld8w(
                            L0 + 128 * SDP, row * (SDP // 4) + 16 * s_ + 8 * h
                        )
                        b_ = prims.lds_ld8w(
                            L0, (sub * 16 + row) * (SDP // 4) + 16 * s_ + 8 * h
                        )
                        acc = wmma_bf16(a_, b_, acc)
                    for i in fx.range_constexpr(8):
                        v = prims.bf16_bits(acc[i])
                        nb = v.shuffle_xor(1, 32)
                        if lane % 2 == 0:
                            prims.lds_sti(
                                LST,
                                ((h * 8 + i) * 128 + sub * 16 + row) // 2,
                                v | (nb << 16),
                            )
                _lds_sync()
                LSD = LST
            else:
                accs = []
                for sub in fx.range_constexpr(8):
                    acc = [fx.Float32(0.0) for _ in range(8)]
                    for s_ in fx.range_constexpr(4):
                        a_ = prims.lds_ld8w(
                            L0 + 128 * SDP, row * (SDP // 4) + 16 * s_ + 8 * h
                        )
                        b_ = prims.lds_ld8w(
                            L0, (sub * 16 + row) * (SDP // 4) + 16 * s_ + 8 * h
                        )
                        acc = wmma_bf16(a_, b_, acc)
                    accs.append(acc)
                _lds_sync()
                # stage the bf16 tile [16 x 128] in LDS (pairs of lanes -> dwords), then row-contiguous stores
                for sub in fx.range_constexpr(8):
                    for i in fx.range_constexpr(8):
                        v = prims.bf16_bits(accs[sub][i])
                        nb = v.shuffle_xor(1, 32)
                        if lane % 2 == 0:
                            prims.lds_sti(
                                L0,
                                ((h * 8 + i) * 128 + sub * 16 + row) // 2,
                                v | (nb << 16),
                            )
                _lds_sync()
                LSD = L0
            orow = lane // 2
            for q in fx.range_constexpr(8):
                wv = prims.lds_ld4w(LSD, orow * 64 + (lane % 2) * 32 + 4 * q)
                if dm_ * 16 + orow < T_:
                    st_dw(
                        r_shout,
                        (dm_ * 16 + orow) * (D // 2)
                        + dn_s * 64
                        + (lane % 2) * 32
                        + 4 * q,
                        wv,
                    )
            publish(F_SD + dm_ * 16 + dn_s)
            stamp(19)

        # ------------------------------------------------------------ router tile + top-k in its epilogue
        if fx.const_expr(upto < 2):
            return
        if is_rt & fx.const_expr(ROLES & 4 != 0):
            tdm_ops.tensor_wait(0)
            stamp(16)
            h = lane // 16
            acc = tile16_mma(KBR)
            col = rn * 16 + lane % 16
            for i in fx.range_constexpr(8):
                t2 = rm * 16 + h * 8 + i
                if (col < E + 1) & (t2 < T_):
                    st_f32(r_lg, (rks * TMAX + t2) * LGS + col, [acc[i]])
            stamp(1)
            fence_rel()
            ra = fx.Int32(0)
            if lane == 0:
                ra = fx.Int32(
                    fx.atomic_add(
                        fl_p + (F_RC_ + 32 * rm), fx.Int32(1), syncscope=AGENT
                    )
                )
            ra = uniform(ra)
            if ra == NRT_ - 1:
                # the block's last router tile: reset the counter for the next launch, flag the block done
                st_dw(r_fl, F_RC_ + 32 * rm, [fx.Int32(0)], cm=CM_DEV)
                bo.buffer_store(
                    tag.ir_value(),
                    r_fl,
                    fx.Int32(F_RD + 32 * rm),
                    cache_modifier=CM_DEV,
                )
            ntok_m = (T_ - rm * 16 < 16).select(T_ - rm * 16, fx.Int32(16))
            kk = ra - (NRT_ - ntok_m)
            if kk >= 0:
                # one of the block's last ntok arrivals: wait for the other tiles, then token rm * 16 + kk
                spin(
                    lambda: fx.Int32(
                        bo.buffer_load(
                            r_fl,
                            F_RD + 32 * rm,
                            vec_width=1,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    ),
                    lambda v: v != tag,
                )
                acquire()
                stamp(2)
                t = rm * 16 + kk
                parts = []
                for k4 in fx.range_constexpr(RKS_R_):
                    parts.append(
                        prims.ld_f32(r_lg, (k4 * TMAX + t) * LGS + lane * 8)
                        + prims.ld_f32(r_lg, (k4 * TMAX + t) * LGS + lane * 8 + 4)
                    )
                vals = []
                for i in fx.range_constexpr(8):
                    v = parts[0][i]
                    for k4 in fx.range_constexpr(1, RKS_R_):
                        v = v + parts[k4][i]
                    vals.append(prims.bf16r(v))
                if fx.const_expr(ROLL & 1):
                    # one loop body for the 8 picks: lane k keeps pick k (value, expert)
                    vv = fx.Vector.from_elements(vals, fx.Float32)
                    my_e = fx.Int32(0)
                    my_v = fx.Float32(0.0)
                    for k_ in range(TOPK, unroll=1):
                        kq = fx.Int32(k_)
                        bv = fx.Float32(vv[0])
                        bi = lane * 8
                        for i in fx.range_constexpr(1, 8):
                            vi = fx.Float32(vv[i])
                            take = vi > bv
                            bv = take.select(vi, bv)
                            bi = take.select(lane * 8 + i, bi)
                        for o_ in (16, 8, 4, 2, 1):
                            ov = bv.shuffle_xor(o_, 32)
                            oi = bi.shuffle_xor(o_, 32)
                            take = (ov > bv) | ((ov == bv) & (oi < bi))
                            bv = take.select(ov, bv)
                            bi = take.select(oi, bi)
                        my_e = (lane == kq).select(bi, my_e)
                        my_v = (lane == kq).select(bv, my_v)
                        mine = lane == bi // 8
                        vv = fx.Vector.from_elements(
                            [
                                (mine & (bi % 8 == i)).select(
                                    fx.Float32(-3.0e38), fx.Float32(vv[i])
                                )
                                for i in range(8)
                            ],
                            fx.Float32,
                        )
                    my_w = fmath.exp(my_v - _readlane_f32(my_v, 0))
                    tot = _readlane_f32(my_w, 0)
                    for k in fx.range_constexpr(1, TOPK):
                        tot = tot + _readlane_f32(my_w, k)
                else:
                    picked_v = []
                    picked_i = []
                    for k in fx.range_constexpr(TOPK):
                        bv = vals[0]
                        bi = lane * 8
                        for i in fx.range_constexpr(1, 8):
                            take = vals[i] > bv
                            bv = take.select(vals[i], bv)
                            bi = take.select(lane * 8 + i, bi)
                        for o_ in (16, 8, 4, 2, 1):
                            ov = bv.shuffle_xor(o_, 32)
                            oi = bi.shuffle_xor(o_, 32)
                            take = (ov > bv) | ((ov == bv) & (oi < bi))
                            bv = take.select(ov, bv)
                            bi = take.select(oi, bi)
                        picked_v.append(bv)
                        picked_i.append(bi)
                        mine = lane == bi // 8
                        for i in fx.range_constexpr(8):
                            vals[i] = (mine & (bi % 8 == i)).select(
                                fx.Float32(-3.0e38), vals[i]
                            )
                    es = [fmath.exp(v - picked_v[0]) for v in picked_v]
                    tot = es[0]
                    for k in fx.range_constexpr(1, TOPK):
                        tot = tot + es[k]
                    my_e = picked_i[0]
                    my_w = es[0]
                    for k in fx.range_constexpr(1, TOPK):
                        my_e = (lane == k).select(picked_i[k], my_e)
                        my_w = (lane == k).select(es[k], my_w)
                if lane < TOPK:
                    slot = fx.Int32(
                        fx.atomic_add(
                            fl_p + (F_EC + my_e), fx.Int32(1), syncscope=AGENT
                        )
                    )
                    bo.buffer_store(my_e.ir_value(), r_ti, t * 8 + lane)
                    bo.buffer_store((my_w / tot).ir_value(), r_tw, t * 8 + lane)
                    bo.buffer_store(slot.ir_value(), r_tsl, t * 8 + lane)
                    bo.buffer_store(t.ir_value(), r_etok, my_e * TMAX + slot)
                if fx.const_expr(shared):
                    gl_ = prims.ld_f32(r_lg, t * LGS + E, n=1)[0]
                    for k4 in fx.range_constexpr(1, RKS_R_):
                        gl_ = (
                            gl_ + prims.ld_f32(r_lg, (k4 * TMAX + t) * LGS + E, n=1)[0]
                        )
                    sgv = prims.bf16r(
                        prims.sigmoid(prims.bf16r(gl_))
                    )  # sigmoid of the bf16 logit, rounded to bf16
                    if lane == 0:
                        st_f32(r_sg, t, [sgv])
                stamp(3)
                if fx.const_expr(TKO_ == 0):
                    fence_rel()
                ko = fx.Int32(0)
                if lane == 0:
                    ko = fx.Int32(
                        fx.atomic_add(fl_p + F_TKC, fx.Int32(1), syncscope=AGENT)
                    )
                ko = uniform(ko)
                if fx.const_expr(TKO_ == 1):
                    # arrive first (the last arriver's prefix sum overlaps the stores' completion), then flag this
                    # token's stores done; the prefix sum publishes only once every token's flag is up
                    fence_rel()
                    bo.buffer_store(
                        tag.ir_value(), r_fl, fx.Int32(F_TKD + t), cache_modifier=CM_DEV
                    )
                if ko == T_ - 1:
                    # the last token: route prefix sum, published as NPSC copies of tagged words
                    acquire()
                    stamp(4)
                    cn = [
                        fx.Int32(c_)
                        for c_ in ld_dw(r_fl, F_EC + lane * 8, cm=CM_DEV)
                        + ld_dw(r_fl, F_EC + lane * 8 + 4, cm=CM_DEV)
                    ]
                    st_dw(r_fl, F_EC + lane * 8, [fx.Int32(0)] * 4, cm=CM_DEV)
                    st_dw(r_fl, F_EC + lane * 8 + 4, [fx.Int32(0)] * 4, cm=CM_DEV)
                    if lane == 0:
                        st_dw(r_fl, F_TKC, [fx.Int32(0)], cm=CM_DEV)
                    gn = [(c_ + 15) // 16 * 16 for c_ in cn]
                    gs = gn[0]
                    for i in fx.range_constexpr(1, 8):
                        gs = gs + gn[i]
                    gi = gs
                    for o_ in (1, 2, 4, 8, 16):
                        pg = fx.Int32(gpu.shuffle_up(gi, o_, 32))
                        gi = (lane >= o_).select(gi + pg, gi)
                    st = gi - gs
                    words = []
                    for i in fx.range_constexpr(8):
                        words.append((tag16 << 16) | (st + cn[i]))
                        st = st + gn[i]
                    if fx.const_expr(TKO_ == 1):
                        for q_ in fx.range_constexpr(TMAX // 32):
                            wait_words(F_TKD + 32 * q_ + lane, lane + 32 * q_ < T_)
                    fence_rel()
                    for c in fx.range_constexpr(NPSC):
                        st_dw(r_fl, F_PSC + 256 * c + lane * 8, words[:4], cm=CM_DEV)
                        st_dw(
                            r_fl, F_PSC + 256 * c + lane * 8 + 4, words[4:], cm=CM_DEV
                        )
                    stamp(5)

        if fx.const_expr(PF == 1 and upto >= 3):
            pf_lo = T_ + MT * 80 if shared else T_
            n_pf = NCTA - n_rt - pf_lo
            pf_i = bid - pf_lo
            if (pf_i >= 0) & (pf_i < n_pf):
                if fx.const_expr(PF_DELAY_US > 0):
                    tgt = t_start + int(PF_DELAY_US * 100)
                    spin(
                        lambda: fx.Int64(
                            _llvm.call_intrinsic(
                                T.i64, "llvm.readsteadycounter", [], [], []
                            )
                        ),
                        lambda c_: c_ < tgt,
                    )
                stamp(18)
                pf_regs = (
                    (addr(w13), E * 2 * I * D // 2),
                    (addr(s13), E * 2 * I * D // 32),
                    (addr(w2), E * D * I // 2),
                    (addr(s2), E * D * I // 32),
                )
                for ri in fx.range_constexpr(4):
                    pbase = pf_regs[ri][0]
                    nl = pf_regs[ri][1] // 128
                    per = (nl + n_pf - 1) // n_pf
                    l0 = pf_i * per
                    for j in range((per + 31) // 32):
                        ln = j * 32 + lane
                        if (ln < per) & (l0 + ln < nl):
                            prims.prefetch_line(pbase + fx.Int64(l0 + ln) * 128)
                stamp(19)

        # ------------------------------------------------------------ every CTA: the prefix sum (one poll)
        if fx.const_expr(upto < 3):
            return
        pc = F_PSC + 256 * (bid % NPSC) + lane * 8

        def load_psc():
            w = ld_dw(r_fl, pc, cm=CM_DEV) + ld_dw(r_fl, pc + 4, cm=CM_DEV)
            return fx.Vector.from_elements([fx.Int32(v_) for v_ in w], fx.Int32)

        def psc_pending(v):
            bad = ((fx.Int32(v[0]) >> 16) & 0x7FFF) != tag16
            for i in fx.range_constexpr(1, 8):
                bad = bad | (((fx.Int32(v[i]) >> 16) & 0x7FFF) != tag16)
            return wave_any(bad)

        pvv = (
            spin_sleep(load_psc, psc_pending, POLL_SLEEP)
            if POLL_SLEEP > 0
            else spin(load_psc, psc_pending)
        )
        acquire()
        stamp(15)
        pv = [fx.Int32(pvv[i]) & 0xFFFF for i in range(8)]
        nrows = fx.Int32(rocdl.readlane(T.i32, pv[7].ir_value(), 31))
        mt = (nrows + 15) // 16

        def psum_at(idx):
            v_ = pv[0]
            for i in fx.range_constexpr(1, 8):
                v_ = (idx % 8 == i).select(pv[i], v_)
            return uniform(prims.wave_sum((lane == idx // 8).select(v_, fx.Int32(0))))

        def expert_of(blk_m):
            c_ = fx.Int32(0)
            for i in fx.range_constexpr(8):
                c_ = c_ + (pv[i] <= blk_m).select(fx.Int32(1), fx.Int32(0))
            return uniform(prims.wave_sum(c_))

        def start_of(e_):
            prev = psum_at((e_ > 0).select(e_ - 1, fx.Int32(0)))
            return (e_ > 0).select((prev + 15) // 16 * 16, fx.Int32(0))

        def expert_fn(blk_m):
            e_ = expert_of(blk_m)
            return e_, psum_at((e_ < E).select(e_, fx.Int32(E - 1)))

        has_gu = tbid < mt * 4
        n_dn = mt * 16
        if fx.const_expr(DNMAP == 1):
            # down tiles go first to CTAs without a gate_up tile (3 slots each, weights issued at routing time),
            # the rest to the gate_up CTAs (whose down weights can only be issued after their gate_up tile)
            G_ = mt * 4
            N1_ = NCTA - G_

            def dts_at(k):  # measured: better from T 85, worse at T 16
                return (T_ > 64).select(
                    (tbid >= G_).select((tbid - G_) + k * N1_, 3 * N1_ + tbid + k * G_),
                    tbid + k * NCTA,
                )

        else:

            def dts_at(k):
                return tbid + k * NCTA

        dts = [dts_at(k) for k in range(3)]

        def gemm_call(
            task, which, mode, tid_, slot=0, late=0, gather=None, drain_cb=None
        ):
            gemm_tdm.HOOKS.clear()
            gemm_tdm.HOOKS.update(expert_fn=expert_fn, mode=mode, outstanding=late)
            if fx.const_expr(gather is not None):
                gemm_tdm.HOOKS["a_gather"] = gather
            if fx.const_expr(drain_cb is not None and GUDN_ == 1):
                gemm_tdm.HOOKS["drain_cb"] = drain_cb
                gemm_tdm.HOOKS["drain_extra"] = 2
            if fx.const_expr(drain_cb is not None and GUDN_ == 2):
                gemm_tdm.HOOKS["store_cb"] = drain_cb
                gemm_tdm.HOOKS["store_extra"] = 4
            if fx.const_expr(which == "dn" and not isinstance(slot, int)):
                # runtime slot (MONO_ROLL 16): a 16-byte aligned LDS pointer rebuilt from the address
                gemm_tdm.LDS_BASE[0] = fx.inttoptr(
                    p8_shared, L0 + DN_BASE_ + slot * DN_SLOT
                )
            else:
                gemm_tdm.LDS_BASE[0] = fx.inttoptr(
                    p8_shared,
                    L0 + ((DN_BASE_ + slot * DN_SLOT) if which == "dn" else 0),
                )
            if fx.const_expr(which == "gu"):
                task(
                    a2,
                    a1t,
                    w13,
                    a1ts,
                    s13,
                    a1t,
                    a1t,
                    a2s,
                    a2,
                    cm,
                    fx.Int32(2 * I),
                    fx.Float32(float("inf")),
                    fx.Float32(1.0),
                    fx.Float32(1.0),
                    tid_,
                    nrows,
                )
            else:
                task(
                    gout,
                    a2,
                    w2,
                    a2s,
                    s2,
                    a2,
                    a2,
                    a2,
                    a2,
                    cm,
                    fx.Int32(D),
                    fx.Float32(float("inf")),
                    fx.Float32(1.0),
                    fx.Float32(1.0),
                    tid_,
                    nrows,
                )
            gemm_tdm.HOOKS.clear()
            gemm_tdm.LDS_BASE[0] = base_ptr

        def issue_down_b(skip0=0):
            if fx.const_expr(ROLL & 16):
                for k_ in range(skip0, 3, unroll=1):
                    kq = fx.Int32(k_)
                    d_ = dts_at(kq)
                    if d_ < n_dn:
                        gemm_call(dn_task, "dn", "issue_b", d_, slot=kq)
            else:
                for k in fx.range_constexpr(skip0, 3):
                    if dts[k] < n_dn:
                        gemm_call(dn_task, "dn", "issue_b", dts[k], slot=k)

        def gu_drain_cb():
            # always exactly 2 (GUDN_ 1) / 4 (GUDN_ 2) TDM ops (the gate_up fences count them): real tiles, clamped
            # to a valid one when this CTA has fewer
            for k in fx.range_constexpr(1 if GUDN_ == 1 else 2):
                gemm_call(
                    dn_task,
                    "dn",
                    "issue_b",
                    (dts[k] < n_dn).select(dts[k], tbid % n_dn),
                    slot=k,
                )

        # EARLY TDM: expert weights as soon as the routing is known
        if fx.const_expr(upto < 4):
            return
        if has_gu & fx.const_expr(ROLES & 8 != 0):
            gemm_call(gu_task, "gu", "issue_b", tbid)
            if fx.const_expr(PF == 2):
                # L2 prefetch of this tile's k tiles 2, 3 of B (the halves of its 4 preshuffled 16 KB rows that the
                # 2-buffer pipeline loads mid-tile)
                pm, pn = _swz(tbid, nrows, 4)
                pe_ = expert_of(pm * 16)
                rowb = (
                    addr(w13) + (fx.Int64(pe_) * 16 + fx.Int64(pn) * 4) * 16384 + 8192
                )
                for i in fx.range_constexpr(8):
                    li = i * 32 + lane
                    prims.prefetch_line(
                        rowb + fx.Int64(li // 64) * 16384 + fx.Int64(li % 64) * 128
                    )
            stamp(17)
        elif fx.const_expr(ROLES & 16 != 0):
            issue_down_b()

        # ------------------------------------------------------------ gate_up: A rows gathered by token
        if has_gu & fx.const_expr(ROLES & 8 != 0):
            gm, gn_ = _swz(tbid, nrows, 4)
            blk_m = gm * 16
            ge = expert_of(blk_m)
            gst = start_of(ge)
            gpe = psum_at((ge < E).select(ge, fx.Int32(E - 1)))
            r_ = blk_m + lane
            rv = (r_ < gpe) & (lane < 16)
            tokv = fx.Int32(
                bo.buffer_load(
                    r_etok,
                    rv.select(ge * TMAX + (r_ - gst), fx.Int32(0)),
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            tokv = rv.select(tokv, T_)
            rows16 = [
                fx.Int32(rocdl.readlane(T.i32, tokv.ir_value(), i)) for i in range(16)
            ]
            gat = {"rows": rows16, "ntok": T_, "a_i64": addr(a1t), "sa_i64": addr(a1ts)}
            wait_words(F_QFR + 32 * (bid % 64), lane == 0)
            stamp(8)
            gemm_call(gu_task, "gu", "issue_a", tbid, gather=gat)
            gemm_call(
                gu_task,
                "gu",
                "rest",
                tbid,
                gather=gat,
                drain_cb=gu_drain_cb if GUDN_ else None,
            )
            publish(F_GU + gm * 4 + gn_)
            stamp(9)
            if fx.const_expr(ROLES & 16 != 0):
                issue_down_b(skip0=GUDN_)

        # ------------------------------------------------------------ down: up to 3 tiles a CTA in flight
        if fx.const_expr(upto < 6):
            return

        def dn_wait_issue_a(k):
            d_ = dts_at(k) if fx.const_expr(ROLL & 16) else dts[k]
            dm, _ = _swz(d_, nrows, 16)
            wait4(F_GU + dm * 4)
            gemm_call(dn_task, "dn", "issue_a", d_, slot=k)

        def dn_finish(k, late):
            d_ = dts_at(k) if fx.const_expr(ROLL & 16) else dts[k]
            dm, dn_ = _swz(d_, nrows, 16)
            gemm_call(dn_task, "dn", "finish", d_, slot=k, late=late)
            publish(F_DN + dm * 16 + dn_)
            stamp(11)

        if fx.const_expr((ROLL & 16) != 0 and (ROLES & 16) != 0):
            # events e = 0..3: tile e's A issued (after its m tile's gate_up flags), then tile e - 1 finished with
            # tile e's loads still in flight (late 2) or, the last one, with none (late 0); one call site each
            for e_ in range(4, unroll=1):
                eq = fx.Int32(e_)
                nxt = (eq < 3) & (dts_at(eq) < n_dn)
                if nxt:
                    dn_wait_issue_a(eq)
                if (eq >= 1) & (dts_at(eq - 1) < n_dn):
                    if nxt:
                        dn_finish(eq - 1, 2)
                    else:
                        dn_finish(eq - 1, 0)
        elif (dts[0] < n_dn) & fx.const_expr(ROLES & 16 != 0):
            dn_wait_issue_a(0)
            stamp(10)
            if dts[1] < n_dn:
                dn_wait_issue_a(1)
                dn_finish(0, 2)
                if dts[2] < n_dn:
                    dn_wait_issue_a(2)
                    dn_finish(1, 2)
                    dn_finish(2, 0)
                else:
                    dn_finish(1, 0)
            else:
                dn_finish(0, 0)

        # ------------------------------------------------------------ gather-reduce: (token, half) a CTA
        if fx.const_expr(
            WCTA > 0 and GLO == 2
        ):  # interleaved over the CUs, as the tiles (one gather wave a CU)
            g_id = NCTA - 1 - tbid
        elif fx.const_expr(WCTA > 0 and GLO == 1):
            g_id = bid
        else:
            g_id = NCTA - 1 - bid
        if (g_id < T_ * 2) & fx.const_expr(ROLES & 32 != 0):
            t = g_id // 2
            hh = g_id - t * 2
            if fx.const_expr(ROLL & 2):
                # route k = lane % 8 on every lane: its expert's start from the prefix sum held by lane (e - 1) // 8
                kl = lane % 8
                ek_l = fx.Int32(
                    bo.buffer_load(
                        r_ti,
                        t * 8 + kl,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                sk_l = fx.Int32(
                    bo.buffer_load(
                        r_tsl,
                        t * 8 + kl,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                wk_l = fx.Int32(
                    bo.buffer_load(
                        r_tw,
                        t * 8 + kl,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                ).bitcast(fx.Float32)
                pidx = (ek_l > 0).select(ek_l - 1, fx.Int32(0))
                cand = [
                    fx.Int32(gpu.shuffle_idx(pv[i], pidx // 8, 32)) for i in range(8)
                ]
                rk_l = (ek_l > 0).select(
                    (_sel(cand, pidx % 8) + 15) // 16 * 16, fx.Int32(0)
                ) + sk_l
                rk = [
                    fx.Int32(
                        rocdl.readlane(T.i32, rk_l.ir_value(), fx.Int32(k).ir_value())
                    )
                    for k in range(TOPK)
                ]
                wk = [_readlane_f32(wk_l, k) for k in range(TOPK)]
                my_r = fx.Int32(gpu.shuffle_idx(rk_l, lane // 4, 32))
            else:
                rk = []
                wk = []
                for k in fx.range_constexpr(TOPK):
                    ek = uniform(
                        fx.Int32(
                            bo.buffer_load(
                                r_ti,
                                t * 8 + k,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                    )
                    sk = uniform(
                        fx.Int32(
                            bo.buffer_load(
                                r_tsl,
                                t * 8 + k,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                    )
                    rk.append(start_of(ek) + sk)
                    wk.append(
                        uniform(
                            fx.Int32(
                                bo.buffer_load(
                                    r_tw,
                                    t * 8 + k,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )
                        ).bitcast(fx.Float32)
                    )
                my_r = rk[0]
                for k in fx.range_constexpr(1, TOPK):
                    my_r = (lane // 4 == k).select(rk[k], my_r)
            wbase = F_DN + (my_r // 16) * 16 + hh * 8 + (lane % 4) * 2

            def load_dn():
                return fx.Vector(
                    bo.buffer_load(
                        r_fl, wbase, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV
                    )
                )

            spin(
                load_dn,
                lambda v: wave_any((fx.Int32(v[0]) != tag) | (fx.Int32(v[1]) != tag)),
            )
            if fx.const_expr(shared):
                wait_words(F_SD + (t // 16) * 16 + hh * 8 + lane % 8, lane >= 0)
            acquire()
            stamp(12)
            acc = [fx.Float32(0.0) for _ in range(32)]
            raws = []
            for k in fx.range_constexpr(TOPK):
                rg = rsrc_at(addr(gout) + fx.Int64(rk[k]) * (2 * D))
                raw = []
                for q in fx.range_constexpr(4):
                    raw += ld_dw(rg, hh * 512 + lane * 16 + q * 4)
                raws.append(raw)
            if fx.const_expr(ROLL & 2):
                # one loop body for the 8 routes (all 8 rows' loads are already in flight), same order
                accv = fx.Vector.from_elements(acc, fx.Float32)
                for k_ in range(TOPK, unroll=1):
                    kq = fx.Int32(k_)
                    rw = [
                        _sel([fx.Int32(raws[k][n]) for k in range(TOPK)], kq)
                        for n in range(16)
                    ]
                    wq = _sel(wk, kq)
                    vals = prims.bf16_unpack([fx.Uint32(w_) for w_ in rw])
                    accv = fx.Vector.from_elements(
                        [fx.Float32(accv[j]) + wq * vals[j] for j in range(32)],
                        fx.Float32,
                    )
                acc = [fx.Float32(accv[j]) for j in range(32)]
            else:
                for k in fx.range_constexpr(TOPK):
                    vals = prims.bf16_unpack(raws[k])
                    for j in fx.range_constexpr(32):
                        acc[j] = acc[j] + wk[k] * vals[j]
            if fx.const_expr(shared):
                # shared_expert_gate_add: bf16(bf16(shared * bf16(sigmoid(g))) + routed_bf16)
                rsh = rsrc_at(addr(shout) + fx.Int64(t) * (2 * D))
                shr = []
                for q in fx.range_constexpr(4):
                    shr += ld_dw(rsh, hh * 512 + lane * 16 + q * 4, cm=CM_DEV)
                shv = prims.bf16_unpack(shr)
                gv = uniform(
                    fx.Int32(
                        bo.buffer_load(
                            r_sg, t, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
                        )
                    )
                ).bitcast(fx.Float32)
                for j in fx.range_constexpr(32):
                    acc[j] = prims.bf16r(shv[j] * gv) + prims.bf16r(acc[j])
            words = [prims.bf16_pack(acc[2 * j], acc[2 * j + 1]) for j in range(16)]
            if fx.const_expr(not ar):
                ro = rsrc_at(addr(out) + fx.Int64(t) * (2 * D))
                for q in fx.range_constexpr(4):
                    st_dw(ro, hh * 512 + lane * 16 + q * 4, words[4 * q : 4 * q + 4])
                stamp(13)
            else:
                # ---- M3: this (token, half) to every rank's inbox: 6 tagged 16-byte lines a lane
                rk_ = fx.Int32(rank)
                par = tag & 1
                peers = [rsrc_at(fx.Int64(p_)) for p_ in (pr0, pr1, pr2, pr3)]
                lbase = (((par * TP + rk_) * TMAX + t) * 2 + hh) * (
                    32 * LANE_DW
                ) + lane * 4
                wpad = words + [fx.Uint32(0), fx.Uint32(0)]
                for P_ in fx.range_constexpr(TP):
                    for i in fx.range_constexpr(6):
                        st_dw(
                            peers[P_],
                            lbase + 128 * i,
                            [wpad[3 * i], wpad[3 * i + 1], wpad[3 * i + 2], tag],
                            cm=prims.CM_SYS,
                        )
                stamp(13)
                if hh == 0:
                    # ---- the token's all-reduce (rank order, fp32) + residual + Gemma RMSNorm, as the native kernel
                    own_base = fx.Int64(pr0)
                    for P_ in fx.range_constexpr(1, TP):
                        own_base = (rk_ == P_).select(
                            fx.Int64([pr0, pr1, pr2, pr3][P_]), own_base
                        )
                    own = rsrc_at(own_base)

                    # this lane: source lane l of both halves (row dwords h * 512 + 16 l .. + 15): contiguous reads
                    def line_off(r_, j):
                        return (
                            (((par * TP + r_) * TMAX + t) * 2 + j // 6) * (32 * LANE_DW)
                            + (j % 6) * 128
                            + lane * 4
                        )

                    def load_lines():
                        els = []
                        for r_ in fx.range_constexpr(TP):
                            for j in fx.range_constexpr(12):
                                v_ = fx.Vector(
                                    bo.buffer_load(
                                        own,
                                        line_off(r_, j),
                                        vec_width=4,
                                        dtype=T.i32,
                                        cache_modifier=prims.CM_SYS,
                                    )
                                )
                                els += [v_[k] for k in range(4)]
                        return fx.Vector.from_elements(els, fx.Int32)

                    def lines_pending(v):
                        bad = fx.Int32(v[3]) != tag
                        for j in fx.range_constexpr(1, TP * 12):
                            bad = bad | (fx.Int32(v[4 * j + 3]) != tag)
                        return wave_any(bad)

                    lv = spin(load_lines, lines_pending)
                    stamp(20)
                    srcs = []
                    for r_ in fx.range_constexpr(TP):
                        dw = []
                        for j2 in fx.range_constexpr(2):
                            d16 = []
                            for i in fx.range_constexpr(6):
                                b_ = 4 * (r_ * 12 + j2 * 6 + i)
                                d16 += [
                                    fx.Uint32(lv[b_]),
                                    fx.Uint32(lv[b_ + 1]),
                                    fx.Uint32(lv[b_ + 2]),
                                ]
                            dw += d16[:16]
                        srcs.append(prims.bf16_unpack(dw))
                    rres = rsrc_at(addr(res) + fx.Int64(t) * (2 * D))
                    rdw = []
                    wdw = []

                    def cdw(q):  # row dword of this lane's q-th 4-dword chunk (q < 8)
                        return (q // 4) * 512 + lane * 16 + 4 * (q % 4)

                    for q in fx.range_constexpr(8):
                        rdw += ld_dw(rres, cdw(q))
                        wdw += ld_dw(rsrc_at(addr(nw)), cdw(q))
                    rv = prims.bf16_unpack(rdw)
                    wv = prims.bf16_unpack(wdw)
                    sv = [
                        ((srcs[0][j] + srcs[1][j]) + srcs[2][j]) + srcs[3][j] + rv[j]
                        for j in range(64)
                    ]
                    sq = sv[0] * sv[0]
                    for j in fx.range_constexpr(1, 64):
                        sq = fmath.fma(sv[j], sv[j], sq)
                    sq = prims.wave_sum(sq)
                    inv = fmath.rsqrt(sq / float(D) + EPS)
                    rxo = rsrc_at(addr(xo) + fx.Int64(t) * (2 * D))
                    for q in fx.range_constexpr(8):
                        st_dw(
                            rres,
                            cdw(q),
                            [
                                prims.bf16_pack(
                                    sv[8 * q + 2 * j], sv[8 * q + 2 * j + 1]
                                )
                                for j in range(4)
                            ],
                        )
                        st_dw(
                            rxo,
                            cdw(q),
                            [
                                prims.bf16_pack(
                                    sv[8 * q + 2 * j] * inv * (wv[8 * q + 2 * j] + 1.0),
                                    sv[8 * q + 2 * j + 1]
                                    * inv
                                    * (wv[8 * q + 2 * j + 1] + 1.0),
                                )
                                for j in range(4)
                            ],
                        )
                    stamp(21)
        stamp(14)
        # end of launch: the last CTA advances the epoch (read by the next launch's CTAs at their start)
        dn_ = fx.Int32(0)
        if lane == 0:
            dn_ = fx.Int32(fx.atomic_add(fl_p + F_DONE, fx.Int32(1), syncscope=AGENT))
        dn_ = uniform(dn_)
        if dn_ == NCTA - 1:
            st_dw(r_fl, F_DONE, [fx.Int32(0)], cm=CM_DEV)
            st_dw(r_fl, F_EPOCH, [ep + 1], cm=CM_DEV)

    @flyc.jit
    def launch(
        x: fx.Pointer,
        gate: fx.Pointer,
        w13: fx.Pointer,
        s13: fx.Pointer,
        w2: fx.Pointer,
        s2: fx.Pointer,
        lg: fx.Pointer,
        ti: fx.Pointer,
        tw: fx.Pointer,
        tsl: fx.Pointer,
        etok: fx.Pointer,
        a1t: fx.Pointer,
        a1ts: fx.Pointer,
        a2: fx.Pointer,
        a2s: fx.Pointer,
        gout: fx.Pointer,
        out: fx.Pointer,
        fl: fx.Pointer,
        stp: fx.Pointer,
        shgu: fx.Pointer,
        shdn: fx.Pointer,
        shp: fx.Pointer,
        act: fx.Pointer,
        shout: fx.Pointer,
        sg: fx.Pointer,
        res: fx.Pointer,
        xo: fx.Pointer,
        nw: fx.Pointer,
        pr0: fx.Int64,
        pr1: fx.Int64,
        pr2: fx.Int64,
        pr3: fx.Int64,
        rank: fx.Int32,
        T_: fx.Int32,
        cm: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008 (the FlyDSL launcher's default stream)
            None
        ),
    ):
        kernel(
            x,
            gate,
            w13,
            s13,
            w2,
            s2,
            lg,
            ti,
            tw,
            tsl,
            etok,
            a1t,
            a1ts,
            a2,
            a2s,
            gout,
            out,
            fl,
            stp,
            shgu,
            shdn,
            shp,
            act,
            shout,
            sg,
            res,
            xo,
            nw,
            pr0,
            pr1,
            pr2,
            pr3,
            rank,
            T_,
            cm,
        ).launch(
            grid=(NCTA // max(WCTA, 1), 1, 1),
            block=(32 * max(WCTA, 1), 1, 1),
            stream=stream,
        )

    return launch


class MonoMoE:
    def __init__(
        self,
        dev,
        timing=False,
        upto=99,
        stamp_sets=1,
        shared=True,
        peers=None,
        rank=0,
        opts=None,
    ):
        """``peers``: a qwen3_5_mono PeerBuffer of >= PEER_BYTES (all-reduce + norm fused: M3), else None."""
        import torch

        self.shared = shared
        self.ar = peers is not None
        self.peers = peers
        self.rank = rank
        self.peer_bases = (
            [int(a) for a in peers.addresses.cpu().tolist()]
            if peers is not None
            else [0] * 4
        )
        # router K split 8 (shorter router, measured) where the role ranges allow it (T <= 112), else 4
        self._bargs = (timing, upto, shared, self.ar)
        self.builds = {}
        self.opts = dict(opts or {})
        self.launch8 = self._build(8, self.opts)
        self.launch4 = self._build(4, self.opts)
        rf = os.environ.get("MONO_RKS_FORCE")
        self.rks_force = int(rf) if rf else None
        i32, u8 = torch.int32, torch.uint8
        self.lg = torch.zeros(RKS_MAX, TMAX, LGS, dtype=torch.float32, device=dev)
        self.ti = torch.zeros(TMAX * TOPK, dtype=i32, device=dev)
        self.tw = torch.zeros(TMAX * TOPK, dtype=torch.float32, device=dev)
        self.tsl = torch.zeros(TMAX * TOPK, dtype=i32, device=dev)
        self.etok = torch.zeros(E * TMAX, dtype=i32, device=dev)
        self.a1t = torch.zeros(TMAX, D // 2, dtype=u8, device=dev)
        self.a1ts = torch.zeros(TMAX, D // 32, dtype=u8, device=dev)
        self.a2 = torch.zeros(CMMAX, I // 2, dtype=u8, device=dev)
        self.a2s = torch.zeros(CMMAX, I // 32, dtype=u8, device=dev)
        self.gout = torch.zeros(CMMAX, D, dtype=torch.bfloat16, device=dev)
        self.out = torch.zeros(TMAX, D, dtype=torch.bfloat16, device=dev)
        self.fl = torch.zeros(F_WORDS, dtype=i32, device=dev)
        self.stamps = torch.zeros(
            stamp_sets, NCTA, NSTAMP, dtype=torch.int64, device=dev
        )
        self.shp = torch.zeros(RKS, TMAX, 2 * SH, dtype=torch.float32, device=dev)
        self.act = torch.zeros(TMAX, SH, dtype=torch.bfloat16, device=dev)
        self.shout = torch.zeros(TMAX, D, dtype=torch.bfloat16, device=dev)
        self.sg = torch.zeros(TMAX, dtype=torch.float32, device=dev)

    def _build(self, rks, opts):
        key = (rks, tuple(sorted(opts.items())))
        if key not in self.builds:
            self.builds[key] = build(*self._bargs, rks_r=rks, **opts)
        return self.builds[key]

    def _args(self, L, x, sidx, res, xo):
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

        T_ = x.shape[0]
        assert T_ <= TMAX
        cm = max(16, (T_ * TOPK + E * 16 - TOPK + 15) // 16 * 16)
        P = lambda t: ptr_arg(t, fx.Int32)
        u8 = torch.uint8
        return (
            P(x),
            P(L.gate),
            P(L.w13.view(u8)),
            P(L.s13),
            P(L.w2.view(u8)),
            P(L.s2),
            P(self.lg),
            P(self.ti),
            P(self.tw),
            P(self.tsl),
            P(self.etok),
            P(self.a1t),
            P(self.a1ts),
            P(self.a2),
            P(self.a2s),
            P(self.gout),
            P(self.out),
            P(self.fl),
            P(self.stamps[sidx]),
            P(L.sh_gu),
            P(L.sh_down),
            P(self.shp),
            P(self.act),
            P(self.shout),
            P(self.sg),
            P(res if res is not None else self.out),
            P(xo if xo is not None else self.out),
            P(L.norm_w if self.ar else self.out),
            *self.peer_bases,
            self.rank,
            T_,
            cm,
        )

    def precompile(self, L, x, res=None, xo=None):
        """Compile both builds without launching: with the 2 s spin deadline, a rank still compiling while its
        peers already spin in the all-reduce would trap them. Call on every rank, then barrier.
        """
        import torch
        from flydsl.compiler import jit_function as _jf

        var = (
            type(_jf.env.compile).options["compile_only"].env_var
        )  # read from os.environ on every access
        old = os.environ.get(var)
        os.environ[var] = "1"
        try:
            for fn in list(self.builds.values()):
                fn(*self._args(L, x, 0, res, xo), stream=torch.cuda.current_stream())
        finally:
            if old is None:
                del os.environ[var]
            else:
                os.environ[var] = old

    def __call__(self, L, x, sidx=0, res=None, xo=None):
        """routed + shared (+ M3: all-reduce over the 4 ranks + residual (in place) + Gemma RMSNorm into ``xo``)."""
        import torch

        T_ = x.shape[0]
        use8 = (
            (T_ <= 112)
            if getattr(self, "rks_force", None) is None
            else (self.rks_force == 8)
        )
        var = getattr(
            self, "variant", None
        )  # in-process A/B: a dict of build options on THIS instance's buffers
        fn = (
            (self.launch8 if use8 else self.launch4)
            if var is None
            else self._build(8 if use8 else 4, {**self.opts, **var})
        )
        fn(*self._args(L, x, sidx, res, xo), stream=torch.cuda.current_stream())
        return self.out[:T_]
