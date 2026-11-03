"""Fused MoE "monolayer" for Qwen3.5-35B-A3B-MXFP4 decode, TP4 rank slice, gfx1250: ONE persistent FlyDSL kernel
for the routed MoE (M1: router -> top-k -> route count / prefix sum -> MXFP4 quant + preshuffle -> gate_up -> down ->
gather-reduce), 1792 one-wave CTAs (7 a CU x 256 CUs) with one 44 KB LDS arena each (the native gate_up kernel's
occupancy). Every stage keeps its native kernel's work decomposition:

  stage    native kernel                              tasks (T tokens)            here
  router   gemm_a16w16 bandwidth_bound 16x8x512       ceil(T/16) x 33 tiles       bf16 WMMA, a 16x8 tile a CTA
  top-k    topkGatingSoftmax                          (rows)                      2 tokens a CTA (16 lanes a token)
  psum     moe_route_psum_fused (1 WG)                1                           one CTA (LDS atomics + scan)
  quant    moe_fused_quant_preshuffle_routeks         T x 8 (route block, k grp)  same helper (8 dests: the token's routes)
  gate_up  a8w4_tdm_fp4_t16x64x512 (act1, q1r1)       ceil(rows/16) x 4           the native tile function (gemm_tdm.py)
  down     a8w4_tdm_fp4_t16x128x128                   ceil(rows/16) x 16          the native tile function
  gather   moe_gather_reduce (v2, 2 WGs a token)      T x 2                       one CTA a (token, half)

Hand-offs: tag words in global memory (tag = launch index + 1, so nothing is reset between launches), written after
an agent-scope release, polled (bounded spin, traps) then an agent-scope acquire. A many-to-one join (quant -> gate_up)
is an arrival counter whose last arriver resets it and publishes the tag.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import _emit_quant_one_k_group, _quant_layout
from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE  # noqa: F401  (the quant helper's rounding)
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import gpu, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from types import SimpleNamespace

from . import gemm_tdm
from atom.models.qwen3_5_mono import prims
from atom.models.qwen3_5_mono.prims import (AGENT, CM_DEV, lds_atomic_add, lds_barrier, lds_ldi, lds_sti, ld_dw,
                                           rsrc_at, spin, st_dw, st_f32, uniform, wmma_bf16)

NCTA = 1792
E, TOPK, D, I = 256, 8, 2048, 128
NR = 33  # router n tiles (257 columns, 8 a tile)
LGS = 264  # logits row stride (f32)
TMAX = 128
CMMAX = 5120  # contiguous_m bound: align16(TMAX * 8 + 256 * 16 - 8)
# flag words (int32)
F_CNT, F_QCNT, F_PS, F_QF = 0, 1, 4, 5
F_RT = 64  # [m][72]: router (n, k split) tiles
F_TK = 1024  # [t]
F_GU = 1280  # [m][4]
F_DN = F_GU + (CMMAX // 16) * 4  # [m][16]
F_PSR = 8192  # 64 copies (one a 128 B line) of {tag, rows}: the psum flag, spread over L2 channels
F_QFR = F_PSR + 64 * 32  # 64 copies of the quant-done flag
F_QC1 = F_QFR + 64 * 32  # 16 first-level quant arrival counters (one a line)
F_QC2 = F_QC1 + 16 * 32  # second-level quant arrival counter
F_TKC = F_QC2 + 32  # top-k arrival counter (the last top-k CTA runs the prefix sum)
F_EC = F_TKC + 32  # [256] per-expert route counters (top-k claims slots; the prefix sum reads and zeroes them)
F_ST = F_EC + 256  # [256] expert start rows
F_WORDS = F_ST + 256
NRN = 17  # router n tiles of 16 (257 columns)
RKS = 4  # router k splits (512 each)
NSTAMP = 24
STAMP_NAMES = ["start", "router_end", "topk_wait", "topk_end", "psum_wait", "psum_end", "quant_wait", "quant_end",
               "gu_wait", "gu_end", "dn_wait", "dn_end", "gather_wait", "gather_end", "end", "ps_seen",
               "ps_counted", "ps_scanned", "ps_remapped", "q_stored", "x20", "x21", "x22", "x23"]

prims.SPIN_LIMIT = 1 << 22
import os  # noqa: E402

# FENCE: 0 = agent-scope release / acquire fences (global_wb / global_inv SCOPE_DEV); 1 = workgroup-scope fences
# only (ordering + s_wait), relying on the flags' scope:DEV loads / stores (experiment: is L2 device-coherent?)
# 2 (default) = release: the stores retired (s_wait_storecnt 0; the GEMM tiles also wait their TDM stores), acquire:
# global_inv scope:SCOPE_SE (L0 + GL1). probe_coherence.py: plain stores are visible device-wide without global_wb;
# a reader's stale copy sits in L0/GL1 (SCOPE_CU inv is not enough, SCOPE_SE is). The agent-scope fences (0) cost
# ~35 us a launch here (global_wb / global_inv SCOPE_DEV at every hand-off).
FENCE = int(os.environ.get("MONO_FENCE", "2"))
EARLY = int(os.environ.get("MONO_EARLY", "1"))
PRELOAD_X = int(os.environ.get("MONO_PRELOAD_X", "1"))
GATHER_CM = int(os.environ.get("MONO_GATHER_CM", "0"))
GATHER_DST = int(os.environ.get("MONO_GATHER_DST", "0"))


def _asm(txt):
    _llvm.InlineAsmOp(res=None, operands_=[], asm_string=txt, constraints="~{memory}", has_side_effects=True,
                      is_align_stack=False)


def fence_rel():
    rocdl.s_wait_storecnt(0)
    if FENCE == 2:
        prims.compiler_fence()
    else:
        fx.memory_fence(syncscope=AGENT if FENCE == 0 else prims.WORKGROUP, ordering=fx.AtomicOrdering.Release)


def fence_acq():
    if FENCE == 2:
        _asm("global_inv scope:SCOPE_SE\n s_wait_loadcnt 0x0")
    else:
        fx.memory_fence(syncscope=AGENT if FENCE == 0 else prims.WORKGROUP, ordering=fx.AtomicOrdering.Acquire)


def _gu_task():
    return gemm_tdm.build_gemm_task(None, None, None, None, None, None, None, 2 * I, D, 16, 64, 512, 1, 1, 0, 2, 1,
                                    None, E, 1, 0, None, float("inf"), stage1_quant_out=1, quant_wmma_rep=1,
                                    cluster_n=1, cluster_m=1, num_waves_per_tensor_tdm=1, persistent_workers=1)


def _dn_task():
    return gemm_tdm.build_gemm_task(None, None, None, None, None, None, None, D, I, 16, 128, 128, 1, 1, 0, 1, 1,
                                    None, E, 0, 0, None, float("inf"), cluster_n=1, cluster_m=1,
                                    num_waves_per_tensor_tdm=1, persistent_workers=1)


def _swz(task, rows, n_tiles):
    """(m tile, n tile) of persistent GEMM task ``task``: the native DeepGEMM swizzle (16-tile groups, cluster 1)."""
    total_m = (rows + 15) // 16
    bpg = n_tiles * 16
    group = task // bpg
    in_group = task - group * bpg
    rem = total_m - group * 16
    gt = (rem < 16).select(rem, fx.Int32(16))
    return group * 16 + in_group % gt, in_group // gt


def build(timing=False, upto=99):
    QL = _quant_layout(D, "fp4", 1)

    @flyc.kernel(name="moe_mono_routed%s_f%d_e%d_x%d" % ("_timing" if timing else "", FENCE, EARLY, PRELOAD_X) + "_g%d_d%d" % (GATHER_CM + 1, GATHER_DST), known_block_size=[32, 1, 1])
    def kernel(x: fx.Pointer, gate: fx.Pointer, w13: fx.Pointer, s13: fx.Pointer, w2: fx.Pointer, s2: fx.Pointer,
               lg: fx.Pointer, ti: fx.Pointer, tw: fx.Pointer, rows_p: fx.Pointer, mm: fx.Pointer, ps: fx.Pointer,
               a1: fx.Pointer, a1s: fx.Pointer, a2: fx.Pointer, a2s: fx.Pointer, gout: fx.Pointer, out: fx.Pointer,
               fl: fx.Pointer, stp: fx.Pointer, T_: fx.Int32, cm: fx.Int32):
        gu_task, gu_arena = _gu_task()
        dn_task, dn_arena = _dn_task()
        base_ptr = fx.SharedAllocator().allocate(max(gu_arena, dn_arena))._ptr
        gemm_tdm.LDS_BASE[0] = base_ptr
        L0 = fx.Int32(fx.ptrtoint(base_ptr))

        bid = fx.Int32(fx.block_idx.x)
        lane = fx.Int32(fx.thread_idx.x)
        T_ = fx.Int32(T_)

        def addr(p):
            return fx.Int64(fx.ptrtoint(p))

        r_fl = rsrc_at(addr(fl))
        r_lg = rsrc_at(addr(lg))
        r_ti = rsrc_at(addr(ti))
        r_tw = rsrc_at(addr(tw))
        r_rows = rsrc_at(addr(rows_p))
        r_mm = rsrc_at(addr(mm))
        r_ps = rsrc_at(addr(ps))
        r_st = rsrc_at(addr(stp))

        def stamp(i):
            if fx.const_expr(timing):
                clk = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))
                st_dw(r_st, (bid * NSTAMP + i) * 2, [fx.Int32(clk & 0xFFFFFFFF), fx.Int32(clk >> 32)])

        stamp(0)
        # launch index: every CTA arrives once a launch on a monotonic counter
        cnt_p = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4), fl)
        fl_p = cnt_p
        old = fx.Int32(0)
        if lane == 0:
            old = fx.Int32(fx.atomic_add(cnt_p, fx.Int32(1), syncscope=AGENT))
        tag = uniform(old) // NCTA + 1

        def publish(word):
            """Release this CTA's stores, then flag ``word`` = tag."""
            rocdl.s_wait_storecnt(0)
            fence_rel()
            bo.buffer_store(tag.ir_value(), r_fl, fx.Int32(word), cache_modifier=CM_DEV)

        def acquire():
            fence_acq()

        def wave_any(b):
            v = b.select(fx.Int32(1), fx.Int32(0))
            for o in (16, 8, 4, 2, 1):
                v = v | v.shuffle_xor(o, 32)
            return v != 0

        def wait_words(word_of_lane, live):
            """Spin until every live lane's flag word equals tag (lane-parallel poll), then acquire."""
            def load_all():
                return fx.Int32(bo.buffer_load(r_fl, word_of_lane, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))

            def pending(v):
                return wave_any(live & (v != tag))

            spin(load_all, pending)
            acquire()

        def wait4(word):
            def load_all():
                return fx.Vector(bo.buffer_load(r_fl, fx.Int32(word), vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))

            def pending(v):
                return (fx.Int32(v[0]) != tag) | (fx.Int32(v[1]) != tag) | (fx.Int32(v[2]) != tag) | (fx.Int32(v[3]) != tag)

            spin(load_all, pending)
            acquire()

        MT = (T_ + 15) // 16
        n_rt = MT * NRN * RKS
        n_tk = T_
        n_q = T_ * TOPK
        n_ga = T_ * 2

        # ------------------------------------------------------------ router: 16 x 16 tiles, K split 4 (512 each)
        r_id = NCTA - 1 - bid
        if fx.const_expr(upto < 1):
            return
        if r_id < n_rt:
            m = r_id // (NRN * RKS)
            rem = r_id - m * (NRN * RKS)
            n = rem // RKS
            ks = rem - n * RKS
            row = lane % 16
            h = lane // 16
            tok = m * 16 + row
            col = n * 16 + row
            rx = bo.create_buffer_resource_from_addr(addr(x), num_records_bytes=(T_ * (2 * D)).ir_value())
            rw = bo.create_buffer_resource_from_addr(addr(gate), num_records_bytes=fx.Int32((E + 1) * 2 * D).ir_value())
            xo = (tok < T_).select(tok * (D // 2), fx.Int32(1 << 28)) + ks * 256 + h * 128
            wo = (col < E + 1).select(col * (D // 2), fx.Int32(1 << 28)) + ks * 256 + h * 128
            acc = [fx.Float32(0.0) for _ in range(8)]
            KS = 8
            for c in fx.range_constexpr(16 // KS):
                av = []
                bv = []
                for s_ in fx.range_constexpr(KS):
                    off = (c * KS + s_) * 8
                    av.append(ld_dw(rx, xo + off) + ld_dw(rx, xo + off + 4))
                    bv.append(ld_dw(rw, wo + off) + ld_dw(rw, wo + off + 4))
                for s_ in fx.range_constexpr(KS):
                    acc = wmma_bf16(av[s_], bv[s_], acc)
            # D: column lane % 16 (expert col), rows 8 h + i: fp32 partials [ks][t][LGS]
            for i in fx.range_constexpr(8):
                t2 = m * 16 + h * 8 + i
                if (col < E + 1) & (t2 < T_):
                    st_f32(r_lg, (ks * TMAX + t2) * LGS + col, [acc[i]])
            publish(F_RT + m * 72 + n * RKS + ks)
            stamp(1)

        # ------------------------------------------------------------ top-k softmax: one token a CTA, 8 experts a lane
        # + route slots (a global atomic a route on its expert's counter); the LAST top-k CTA to arrive runs the
        # route prefix sum (the native moe_route_psum_fused's one-workgroup scan) and publishes the routing.
        k_id = NCTA - 1 - n_rt - bid
        if fx.const_expr(upto < 2):
            return
        if (k_id >= 0) & (k_id < n_tk):
            t = k_id
            m = t // 16

            def load_rt():
                a0 = fx.Int32(bo.buffer_load(r_fl, F_RT + m * 72 + lane, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                a1_ = fx.Int32(bo.buffer_load(r_fl, F_RT + m * 72 + 32 + lane, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                a2_ = fx.Int32(bo.buffer_load(r_fl, F_RT + m * 72 + 64 + (lane % 4), vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                return fx.Vector.from_elements([a0, a1_, a2_], fx.Int32)

            spin(load_rt, lambda v: wave_any((fx.Int32(v[0]) != tag) | (fx.Int32(v[1]) != tag) | (fx.Int32(v[2]) != tag)))
            acquire()
            stamp(2)
            parts = []
            for k4 in fx.range_constexpr(RKS):
                parts.append(prims.ld_f32(r_lg, (k4 * TMAX + t) * LGS + lane * 8) + prims.ld_f32(r_lg, (k4 * TMAX + t) * LGS + lane * 8 + 4))
            vals = []
            for i in fx.range_constexpr(8):
                v = parts[0][i]
                for k4 in fx.range_constexpr(1, RKS):
                    v = v + parts[k4][i]
                vals.append(prims.bf16r(v))  # the native router GEMM's bf16 logits
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
                    vals[i] = (mine & (bi % 8 == i)).select(fx.Float32(-3.0e38), vals[i])
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
                slot = fx.Int32(fx.atomic_add(fl_p + (F_EC + my_e), fx.Int32(1), syncscope=AGENT))
                bo.buffer_store(my_e.ir_value(), r_ti, t * 8 + lane)
                bo.buffer_store((my_w / tot).ir_value(), r_tw, t * 8 + lane)
                bo.buffer_store(slot.ir_value(), r_rows, t * 8 + lane)
            stamp(3)
            fence_rel()
            kold = fx.Int32(0)
            if lane == 0:
                kold = fx.Int32(fx.atomic_add(fl_p + F_TKC, fx.Int32(1), syncscope=AGENT))
            kold = uniform(kold)
            if kold == n_tk - 1:
                acquire()
                stamp(4)
                cn = ld_dw(r_fl, F_EC + lane * 8, cm=CM_DEV) + ld_dw(r_fl, F_EC + lane * 8 + 4, cm=CM_DEV)
                cn = [fx.Int32(c_) for c_ in cn]
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
                starts = []
                for i in fx.range_constexpr(8):
                    starts.append(st)
                    st = st + gn[i]
                st_dw(r_fl, F_ST + lane * 8, starts[:4], cm=CM_DEV)
                st_dw(r_fl, F_ST + lane * 8 + 4, starts[4:], cm=CM_DEV)
                st_dw(r_mm, lane * 8, cn[:4])
                st_dw(r_mm, lane * 8 + 4, cn[4:])
                st_dw(r_ps, lane * 8, [starts[i] + cn[i] for i in range(4)])
                st_dw(r_ps, lane * 8 + 4, [starts[i] + cn[i] for i in range(4, 8)])
                nr_all = fx.Int32(rocdl.readlane(T.i32, (starts[7] + cn[7]).ir_value(), 31))
                fence_rel()
                for j in fx.range_constexpr(2):
                    st_dw(r_fl, F_PSR + 32 * (lane + 32 * j), [tag, nr_all], cm=CM_DEV)
                stamp(5)

        # quant CTAs: their token's activation chunk is an input: load it now, before the routing is known
        q_id0 = NCTA - 1 - n_rt - n_tk - bid
        q_id0 = (q_id0 < 0).select(q_id0 + NCTA, q_id0)
        q_t0 = (q_id0 < n_q).select(q_id0 // 8, fx.Int32(0))
        q_g0 = q_id0 % 8
        rxq = bo.create_buffer_resource_from_addr(addr(x), num_records_bytes=(T_ * (2 * D)).ir_value())
        q_mx_block = q_g0 * QL.mx_blocks_per_wave_iter + lane // QL.lanes_per_mx_block
        q_col = q_mx_block * 32 + (lane % QL.lanes_per_mx_block) * QL.elems_per_lane
        xq_pre = bo.buffer_load(rxq, q_t0 * (D // 2) + (q_col >> 1), vec_width=4, dtype=T.i32)

        # every CTA: the routing extent (GEMM task counts)
        def load_ps():
            return fx.Vector(bo.buffer_load(r_fl, F_PSR + 32 * (bid % 64), vec_width=2, dtype=T.i32, cache_modifier=CM_DEV))

        psv = spin(load_ps, lambda v: fx.Int32(v[0]) != tag)
        acquire()
        stamp(15)
        nrows = uniform(fx.Int32(psv[1]))
        mt = (nrows + 15) // 16

        # the prefix sum in registers (8 experts a lane): a GEMM tile's expert = #{e: psum[e] <= blk_m} (what the
        # native in-kernel bisect finds, without its 9 dependent global loads), its row bound psum[expert]
        pv = [fx.Int32(v_) for v_ in ld_dw(r_ps, lane * 8, cm=CM_DEV) + ld_dw(r_ps, lane * 8 + 4, cm=CM_DEV)]

        def expert_fn(blk_m):
            c_ = fx.Int32(0)
            for i in fx.range_constexpr(8):
                c_ = c_ + (pv[i] <= blk_m).select(fx.Int32(1), fx.Int32(0))
            e_ = uniform(prims.wave_sum(c_))
            ec = (e_ < E).select(e_, fx.Int32(E - 1))
            v_ = pv[0]
            for i in fx.range_constexpr(1, 8):
                v_ = (ec % 8 == i).select(pv[i], v_)
            pe = uniform(prims.wave_sum((lane == ec // 8).select(v_, fx.Int32(0))))
            return e_, pe

        DN_SLOT = 10240
        has_gu = bid < mt * 4
        n_dn = mt * 16
        dts = [bid + k * NCTA for k in range(3)]

        def gemm_call(task, which, mode, tid_, slot=0, late=0):
            gemm_tdm.HOOKS.clear()
            gemm_tdm.HOOKS.update(expert_fn=expert_fn, mode=mode, outstanding=late)
            gemm_tdm.LDS_BASE[0] = base_ptr + slot * DN_SLOT
            if fx.const_expr(which == "gu"):
                task(a2, a1, w13, a1s, s13, ps, a1, a2s, a2, cm, fx.Int32(2 * I), fx.Float32(float("inf")),
                     fx.Float32(1.0), fx.Float32(1.0), tid_, nrows)
            else:
                task(gout, a2, w2, a2s, s2, ps, a2, a2, a2, cm, fx.Int32(D), fx.Float32(float("inf")),
                     fx.Float32(1.0), fx.Float32(1.0), tid_, nrows)
            gemm_tdm.HOOKS.clear()
            gemm_tdm.LDS_BASE[0] = base_ptr

        def issue_down_b():
            for k in fx.range_constexpr(3):
                if dts[k] < n_dn:
                    gemm_call(dn_task, "dn", "issue_b", dts[k], slot=k)

        # EARLY TDM: weights as soon as the routing is known, before (and under) the quant work
        # (EARLY=1: a CTA with a quant task issues after it; down weights after the quant-done flag, so the
        # gate_up weights and the quant's own traffic go first)
        q_id = NCTA - 1 - n_rt - n_tk - bid
        q_id = (q_id < 0).select(q_id + NCTA, q_id)
        if fx.const_expr(upto >= 5):
            if fx.const_expr(EARLY == 0):
                if has_gu:
                    gemm_call(gu_task, "gu", "issue_b", bid)
                else:
                    issue_down_b()
            else:
                if has_gu & (q_id >= n_q):
                    gemm_call(gu_task, "gu", "issue_b", bid)

        # ------------------------------------------------------------ quant: (token, k group) a CTA, 8 dests
        if fx.const_expr(upto < 4):
            return
        q_id = (q_id < 0).select(q_id + NCTA, q_id)
        if q_id < n_q:
            stamp(6)
            t = q_id // 8
            g = q_id - t * 8
            dests = []
            for k in fx.range_constexpr(TOPK):
                ek = uniform(fx.Int32(bo.buffer_load(r_ti, t * 8 + k, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)))
                sk = uniform(fx.Int32(bo.buffer_load(r_rows, t * 8 + k, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)))
                row = uniform(fx.Int32(bo.buffer_load(r_fl, F_ST + ek, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))) + sk
                scale_tile = row // 16
                dests.append(SimpleNamespace(payload_row_i32=fx.Uint32(row),
                                             scale_row_dword_base=fx.Uint32(scale_tile * 16 * 16 + (row - scale_tile * 16))))
            block_in_wave = fx.Uint32(lane) // fx.Uint32(QL.lanes_per_mx_block)
            lane_in_block = fx.Uint32(lane) - block_in_wave * QL.lanes_per_mx_block
            from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor
            qc = SimpleNamespace(
                i32=T.i32, f32=T.f32, block_iters=1, payload_base=addr(a1), payload_bytes_per_row=QL.payload_bytes_per_row,
                hidden_base=addr(x), feat_bytes_per_row=D * 2, feat_row_i32=fx.Uint32(t), prequantized=False,
                payload_dwords_per_lane=QL.payload_bytes_per_lane // 4, src_scale_base=addr(x), src_scale_bytes_per_row=0,
                mx_blocks_per_wave_iter=QL.mx_blocks_per_wave_iter, mx_blocks_per_row=QL.mx_blocks_per_row,
                amax_shuffle_dists=QL.amax_shuffle_dists, is_fp8=False, use_native=QL.use_native, use_pk8=QL.use_pk8,
                mx_dtype=QL.mx_dtype, c0_i32=fx.Int32(0), c1_i32=fx.Int32(1), c4_i32=fx.Int32(4), c23_i32=fx.Int32(23),
                c254_i32=fx.Int32(254), c0_f32=fx.Float32(0.0), c_wave=fx.Int32(32), c_elems_per_lane=fx.Int32(QL.elems_per_lane),
                c_payload_bytes_per_block=fx.Int32(QL.payload_bytes_per_block),
                c_payload_bytes_per_lane=fx.Int32(QL.payload_bytes_per_lane), c_wmma_rep=fx.Int32(1),
                block_in_wave=block_in_wave, lane_in_block=lane_in_block, is_block_lead=lane_in_block == 0,
                dests=dests, scale_t=ptr_buf_tensor(a1s, fx.Int8), scale_t_i32=ptr_buf_tensor(a1s, fx.Int32),
                lane=fx.Uint32(lane), wmma_rep=1, row_major_scale=False,
                hidden_lds_load=(lambda _i, _o: xq_pre) if PRELOAD_X else None, hidden_lds_idx=fx.Int32(0),
                hidden_lds_row_off=fx.Int32(0), hidden_slot_bytes=0)
            _emit_quant_one_k_group(qc, fx.Uint32(g))
            stamp(19)
            # arrive; the last of the n_q tasks resets the counter and publishes
            rocdl.s_wait_storecnt(0)
            fence_rel()
            j1 = q_id % 16
            n1 = (n_q - j1 + 15) // 16
            qold = fx.Int32(0)
            if lane == 0:
                qold = fx.Int32(fx.atomic_add(fl_p + (F_QC1 + 32 * j1), fx.Int32(1), syncscope=AGENT))
            qold = uniform(qold)
            if qold == n1 - 1:
                st_dw(r_fl, F_QC1 + 32 * j1, [fx.Int32(0)], cm=CM_DEV)
                fence_rel()
                q2 = fx.Int32(0)
                if lane == 0:
                    q2 = fx.Int32(fx.atomic_add(fl_p + F_QC2, fx.Int32(1), syncscope=AGENT))
                q2 = uniform(q2)
                if q2 == (n_q < 16).select(n_q, fx.Int32(16)) - 1:
                    acquire()
                    st_dw(r_fl, F_QC2, [fx.Int32(0)], cm=CM_DEV)
                    fence_rel()
                    for j in fx.range_constexpr(2):
                        st_dw(r_fl, F_QFR + 32 * (lane + 32 * j), [tag], cm=CM_DEV)
            stamp(7)
        if fx.const_expr(upto >= 5 and EARLY == 1):
            if has_gu & (q_id < n_q):
                gemm_call(gu_task, "gu", "issue_b", bid)
            if bid >= mt * 4:
                wait_words(F_QFR + 32 * (bid % 64), lane == 0)
                issue_down_b()

        # ------------------------------------------------------------ gate_up: native tile (bid), SiLU*up + MXFP4 out
        if fx.const_expr(upto < 5):
            return
        if has_gu:
            wait_words(F_QFR + 32 * (bid % 64), lane == 0)
            stamp(8)
            gemm_call(gu_task, "gu", "issue_a", bid)
            gemm_call(gu_task, "gu", "rest", bid)
            gm, gn_ = _swz(bid, nrows, 4)
            publish(F_GU + gm * 4 + gn_)
            stamp(9)
            issue_down_b()  # the arena is free again: this CTA's down weights

        # ------------------------------------------------------------ down: up to 3 tiles a CTA in flight
        # B (weights) already issued into slot k; A (gate_up output) once its m tile's 4 gate_up flags are up;
        # tile k+1's A is issued before tile k computes (its fence leaves those 2 TDM ops in flight).
        if fx.const_expr(upto < 6):
            return

        def dn_wait_issue_a(k):
            dm, _ = _swz(dts[k], nrows, 16)
            wait4(F_GU + dm * 4)
            gemm_call(dn_task, "dn", "issue_a", dts[k], slot=k)

        def dn_finish(k, late):
            dm, dn_ = _swz(dts[k], nrows, 16)
            gemm_call(dn_task, "dn", "finish", dts[k], slot=k, late=late)
            publish(F_DN + dm * 16 + dn_)
            stamp(11)

        if dts[0] < n_dn:
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
        g_id = NCTA - 1 - bid
        if fx.const_expr(upto < 7):
            return
        if g_id < n_ga:
            t = g_id // 2
            hh = g_id - t * 2
            rk = []
            wk = []
            for k in fx.range_constexpr(TOPK):
                ek = uniform(fx.Int32(bo.buffer_load(r_ti, t * 8 + k, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)))
                sk = uniform(fx.Int32(bo.buffer_load(r_rows, t * 8 + k, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)))
                rk.append(uniform(fx.Int32(bo.buffer_load(r_fl, F_ST + ek, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))) + sk)
                wk.append(uniform(fx.Int32(bo.buffer_load(r_tw, t * 8 + k, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))).bitcast(fx.Float32))
            # lane l polls route l // 4's two down flags (l % 4) * 2 .. + 1 of this half
            my_r = rk[0]
            for k in fx.range_constexpr(1, TOPK):
                my_r = (lane // 4 == k).select(rk[k], my_r)
            wbase = F_DN + (my_r // 16) * 16 + hh * 8 + (lane % 4) * 2

            def load_dn():
                return fx.Vector(bo.buffer_load(r_fl, wbase, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV))

            spin(load_dn, lambda v: wave_any((fx.Int32(v[0]) != tag) | (fx.Int32(v[1]) != tag)))
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
            if fx.const_expr(timing):
                rocdl.s_wait_loadcnt(0)
                stamp(20)
            for k in fx.range_constexpr(TOPK):
                vals = prims.bf16_unpack(raws[k])
                for j in fx.range_constexpr(32):
                    acc[j] = acc[j] + wk[k] * vals[j]
            ro = rsrc_at(addr(out) + fx.Int64(t) * (2 * D)) if GATHER_DST == 0 else rsrc_at(addr(lg) + fx.Int64(t) * (2 * D))
            for q in fx.range_constexpr(4 if GATHER_CM >= 0 else 0):
                st_dw(ro, hh * 512 + lane * 16 + q * 4, [prims.bf16_pack(acc[8 * q + 2 * j], acc[8 * q + 2 * j + 1]) for j in range(4)],
                      cm=max(GATHER_CM, 0))
            if fx.const_expr(timing):
                rocdl.s_wait_storecnt(0)
            stamp(13)
        stamp(14)

    @flyc.jit
    def launch(x: fx.Pointer, gate: fx.Pointer, w13: fx.Pointer, s13: fx.Pointer, w2: fx.Pointer, s2: fx.Pointer,
               lg: fx.Pointer, ti: fx.Pointer, tw: fx.Pointer, rows_p: fx.Pointer, mm: fx.Pointer, ps: fx.Pointer,
               a1: fx.Pointer, a1s: fx.Pointer, a2: fx.Pointer, a2s: fx.Pointer, gout: fx.Pointer, out: fx.Pointer,
               fl: fx.Pointer, stp: fx.Pointer, T_: fx.Int32, cm: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        kernel(x, gate, w13, s13, w2, s2, lg, ti, tw, rows_p, mm, ps, a1, a1s, a2, a2s, gout, out, fl, stp, T_, cm).launch(
            grid=(NCTA, 1, 1), block=(32, 1, 1), stream=stream)

    return launch


class MonoMoE:
    """Workspace + launcher of the routed monolayer kernel (one per device)."""

    def __init__(self, dev, timing=False, upto=99, stamp_sets=1):
        import torch

        self.dev = dev
        self.timing = timing
        self.launch = build(timing, upto)
        i32, u8 = torch.int32, torch.uint8
        self.lg = torch.zeros(RKS, TMAX, LGS, dtype=torch.float32, device=dev)
        self.ti = torch.zeros(TMAX * TOPK, dtype=i32, device=dev)
        self.tw = torch.zeros(TMAX * TOPK, dtype=torch.float32, device=dev)
        self.rows = torch.zeros(TMAX * TOPK, dtype=i32, device=dev)
        self.mm = torch.zeros(E, dtype=i32, device=dev)
        self.ps = torch.zeros(E, dtype=i32, device=dev)
        self.a1 = torch.zeros(CMMAX, D // 2, dtype=u8, device=dev)
        self.a1s = torch.zeros(CMMAX, D // 32, dtype=u8, device=dev)
        self.a2 = torch.zeros(CMMAX, I // 2, dtype=u8, device=dev)
        self.a2s = torch.zeros(CMMAX, I // 32, dtype=u8, device=dev)
        self.gout = torch.zeros(CMMAX, D, dtype=torch.bfloat16, device=dev)
        self.out = torch.zeros(TMAX, D, dtype=torch.bfloat16, device=dev)
        self.fl = torch.zeros(F_WORDS, dtype=i32, device=dev)
        self.stamps = torch.zeros(stamp_sets, NCTA, NSTAMP, dtype=torch.int64, device=dev)

    def __call__(self, L, x, sidx=0):
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

        T_ = x.shape[0]
        assert T_ <= TMAX
        cm = max(16, (T_ * TOPK + E * 16 - TOPK + 15) // 16 * 16)
        P = lambda t: ptr_arg(t, fx.Int32)  # noqa: E731
        self.launch(P(x), P(L.gate), P(L.w13.view(__import__("torch").uint8)), P(L.s13), P(L.w2.view(__import__("torch").uint8)),
                    P(L.s2), P(self.lg), P(self.ti), P(self.tw), P(self.rows), P(self.mm), P(self.ps), P(self.a1),
                    P(self.a1s), P(self.a2), P(self.a2s), P(self.gout), P(self.out), P(self.fl), P(self.stamps[sidx]), T_, cm,
                    stream=torch.cuda.current_stream())
        return self.out[:T_]
