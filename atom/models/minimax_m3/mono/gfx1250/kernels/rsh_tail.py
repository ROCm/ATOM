# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Launch A's tail phase (MiniMax-M3.1 C4 decode, TP4, one request of 8 rows, gfx1250): the MoE router
(bf16 GEMV + biased-sigmoid top-4 + the MXFP4 token rows) and the shared expert's gate_up (MXFP8 GEMV +
SwiGLU-OAI + group-32 MXFP8 quant), both from the normed hidden state x that R1 (the attention AR receive +
residual + RMSNorm) publishes. Its outputs are launch B's inputs; launch B starts straight into the routed
weight stream.

``RshTail.run(role, ...)`` is called by 104 CTAs (13 whole clusters of 8) of the persistent launch, each as
soon as it has no other work. It first issues the CTA's weight loads (VGPRs, and LDS by TDM for the router),
so the weights stream while the launch's earlier phases run, then polls R1's x-ready count, then computes.

Roles (``role`` 0..103; the 8 CTAs of a role cluster must be one launch cluster, ``role % 8`` = its rank):

* router, roles 0-7 (CTA = K split ``s`` of production's split-K 8, K [768 s, +768)): wave ``w`` runs the
  WMMA chains of expert slices ``w`` (weights in VGPRs) and ``w + 4`` (weights in LDS), 24 bf16 16x16x32
  steps each from a zero accumulator, on x from LDS. Partials -> cluster scratch, ``_cluster_sync``, then
  CTA ``s`` gates token row ``s`` with ``moe_router_topk``'s arithmetic (split-order fp32 sum, bf16 logit,
  ``rcp(1 + exp2(-x log2e))``, bias, 5-comparator network, 4 wave-argmax rounds, renorm). Before the
  exchange each CTA also writes its K slice of every token row as MXFP4 (the router's ``quant_rows``).
* shared, roles 8-103 (cluster ``c`` = h columns [64 c, +64), CTA rank ``p``: 32-column group
  ``g = 2 c + p / 4`` x K split ``s = p % 4`` of production's split-K 4, K [1536 s, +1536)): wave ``t``
  runs the 12-step scaled-fp8 16x16x128 chain of tile ``t`` = gate rows ``32 g + 8 t + 0..7`` and the same
  up rows (weights in VGPRs; x quantized to MXFP8 by the CTA into LDS, ``quantize_fp8``'s bytes).
  Partials -> cluster scratch, ``_cluster_sync``, then every CTA of the group sums the four splits in split
  order (``gemm_a8w8 ... _sk4_fsk``'s fp32 order, one bf16 rounding), applies SwiGLU-OAI as Triton's
  ``_swiglu_oai_mxfp8_kernel`` does (bf16 boundary), takes the group's 32-column amax and stores tile
  ``s``'s E4M3 bytes (and, at ``s == 0``, the group's E8M0 scale).

The exchanges are production PA's cluster merge sync (cluster-scope release, cluster barrier, cluster-scope
acquire): every exchange stays inside one cluster (one XCD), no device-scope hand-off inside the phase, no
other cluster or barrier configuration.

The x hand-off from R1: an epoch counter in uncached memory (never reset; R1's 8 receiver CTAs each do
device-scope x stores, ``s_wait_storecnt 0``, then one relaxed add), polled by thread 0 with a count bound and
``s_sleep``; x is then read with device-scope loads. Option ``x_flag``: x is its own flag (k5's sentinel rule,
see ``x_flag_word``), the CTAs poll the x words they need, and the router CTAs re-arm the previous launch's x
buffer. Every poll is bounded and counts timeouts. The outputs reach launch B through the launch boundary.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.utility.mx_types import MxDtypeInt

from aiter.ops.flydsl.kernels import tdm_ops_gfx1250 as tdm_ops
from aiter.ops.flydsl.kernels.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _MXFP8_COPY_AMAX_FLOOR,
    _ROUND_MODE,
    _cvt_scalef32_pk8_fp4_bf16,
    _cvt_scalef32_pk8_fp8_bf16,
)
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale

HIDDEN = 6144
EXPERTS = 128
TOPK = 4
ROWS = 8
WAVE = 32
CLUSTER = 8
SCOPE_DEV = 2 << 3  # gfx12 CPol SCOPE field: device
W_NT = 1  # gfx12 CPol TH_LOAD_NT: weights stream non-temporally (production's wcpol; checklist 7)
POLL_SLEEP = 16  # x-ready poll: normal memory + device-scope loads + s_sleep >= 16 (coordinator rule)
SENT_LO = 0x7FFF7FFF
SENT_HI = -0x7FFF8000  # 0x80008000 as int32
LOG2E = 1.4426950408889634
RENORM_SUM_FLOOR = 1e-20
FLT_MIN = 1.1754943508222875e-38

# router: production moe_router_topk at 8 rows (split_k 8, bf16 WMMA 16x16x32)
R_SPLIT = 8
R_CHUNK = HIDDEN // R_SPLIT  # 768
R_STEPS = R_CHUNK // 32  # 24
R_ROWB = R_CHUNK * 2  # 1536 bytes of a weight row's K slice
R_PADI, R_PADA = 512, 16  # TDM LDS pad: 16 B after every 512 B (oproj_ar's activation chunk pad)
R_PITCH = R_ROWB + (R_ROWB // R_PADI) * R_PADA  # 1584
R_WB = 0  # LDS: chain-B weights, [wave][16 rows][R_PITCH]
R_XB = 4 * 16 * R_PITCH  # 101376: x B fragments [step][half][kgrp][tok] x 16 B
R_LDS = R_XB + R_STEPS * 2 * 16 * 16  # 113664

# shared gate_up: production gemm_a8w8_mx32_t16x32x512_mw1_nw2_nb2_sk4 ..._fsk (scaled fp8 WMMA 16x16x128)
S_INTER = 768  # h columns a rank
S_SPLIT = 4
S_CHUNK = HIDDEN // S_SPLIT  # 1536
S_STEPS = S_CHUNK // 128  # 12
GROUP = 32
N_GROUPS = S_INTER // GROUP  # 24
S_CLUSTERS = N_GROUPS // 2  # 12
S_XB = 0  # LDS: x MXFP8 B fragments [step][jj][kgrp][tok] x 16 B
S_SC = S_STEPS * 4 * 16 * 16  # 12288: x E8M0 words [tok][step]
S_WS = S_SC + ROWS * S_STEPS * 4  # 12672: weight E8M0 words [wave][step][lane]
S_LDS = S_WS + 4 * S_STEPS * WAVE * 4  # 18816
S_KB = HIDDEN // 128  # scale words of a weight row (n32k4: [N/32][K/128][32][4])

N_ROLES = CLUSTER * (1 + S_CLUSTERS)  # 104
LDS_BYTES = max(R_LDS, S_LDS)

# scratch (fp32, cluster-private): router [split][tok][expert], then shared [cluster][rank][tile][tok][16 rows]
SCR_R = R_SPLIT * ROWS * EXPERTS
SCR_S = S_CLUSTERS * CLUSTER * 4 * ROWS * 16
SCR_WORDS = SCR_R + SCR_S

# device table (int64 words)
(T_X, T_LDX, T_RW, T_RB, T_NVR, T_SW, T_SS, T_IDS, T_WTS, T_XQ, T_XS, T_H, T_HS, T_SCR, T_TL, T_XPREV) = range(16)
T_WORDS = 16
TL_SLOTS = 8  # int64 stamps a role CTA (tl builds)


def _cluster_sync():
    """Production PA's cluster merge sync: publish global stores to the cluster, observe the others'."""
    fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Release)
    cluster.cluster_barrier()
    fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Acquire)


def _rsrc(addr, nbytes=None):
    return create_buffer_resource_from_addr(fx.Int64(addr), num_records_bytes=nbytes)


def _ld(rsrc, idx, words, cpol=0):
    return buffer_load(rsrc, fx.Int32(idx), vec_width=words, dtype=T.i32, cache_modifier=cpol)


def _uni(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(fx.Int32(v))))


def _word(tab, i):
    w = Vec(_ld(tab, 2 * i, 2))
    return (fx.Int64(_uni(w[1])) << fx.Int64(32)) | fx.Int64(fx.Uint32(_uni(w[0])))


def _now():
    return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))


def _rmem(v, n, dtype):
    t = fx.make_rmem_tensor(n, dtype)
    t.store(v if isinstance(v, fx.Vector) else fx.Vector(v))
    return t


def _cat2(lo, hi):
    return Vec(lo).shuffle(Vec(hi), list(range(8)))


def _cat4(v):
    v01 = Vec(v[0]).shuffle(Vec(v[1]), list(range(8)))
    v23 = Vec(v[2]).shuffle(Vec(v[3]), list(range(8)))
    return v01.shuffle(v23, list(range(16)))


def _gptr(addr, elem=fx.Int32, align=4):
    """Global pointer (iterator) at i64 address ``addr``."""
    return fx.recast_iter(elem, fx.inttoptr(fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, align), fx.Int64(addr)))


def _load_agent_i32(addr):
    ptr = _gptr(addr)
    return fx.Int32(
        fx.generic_load(
            ptr, volatile=True, memory_order=fx.AtomicOrdering.Monotonic, syncscope=fx.rocdl.SyncScope.AgentOneAs
        )
    )


def _atomic_add_agent(addr, v):
    return fx.atomic_add(_gptr(addr), fx.Int32(v), syncscope="agent")


def _amax_bf16(chunks):
    """max |x| (fp32) over the bf16 pairs of ``chunks`` (vec4 i32 each) on the bit patterns: |x| of a bf16 is
    its low 15 bits, and non-negative floats order as their bits, so this is the fp32 max of the values
    without unpacking them (fewer live VGPRs than ``fx.max`` over fp32 copies; the same result)."""
    m = fx.Int32(0)
    for v in chunks:
        for d in range(4):
            w = fx.Int32(Vec(v)[d])
            m = fx.max(m, w & fx.Int32(0x7FFF))
            m = fx.max(m, (w >> fx.Int32(16)) & fx.Int32(0x7FFF))
    return (m << fx.Int32(16)).bitcast(fx.Float32)


def sentinel_bits(w):
    """k5's sentinel_mask: one bit (the half-word's top bit) per half-word of ``w`` that is 0xffff."""
    x = w ^ fx.Int32(-1)
    t = ((x & fx.Int32(SENT_LO)) + fx.Int32(SENT_LO)) | x
    return (t & fx.Int32(SENT_HI)) ^ fx.Int32(SENT_HI)


def x_flag_word(w):
    """What R1 stores for an x word in data-as-flag mode (k5's sender rule): 0xffff half-words -> 0x7fff."""
    return w ^ sentinel_bits(w)


def _pending(chunks):
    m = fx.Int32(0)
    for v in chunks:
        for d in range(4):
            m = m | sentinel_bits(fx.Int32(Vec(v)[d]))
    return fx.Int32(rocdl.ballot(T.i32, m != 0)) != 0


def _mem_clobber():
    """Keeps the poll loop's loads inside the loop (k5: asm volatile("" ::: "memory"))."""
    _llvm.InlineAsmOp(
        res=None, operands_=[], asm_string="", constraints="~{memory}", has_side_effects=True, is_align_stack=False
    )


def _lds_barrier():
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


class RshTail:
    """The phase; ``run`` is called inside the persistent launch (see the module docstring).

    ``alpha`` / ``beta`` / ``limit``: the config's SwiGLU-OAI constants (MiniMax-M3.1: 1.702, 1.0, 7.0).
    ``tl``: timeline build (int64 stamps per role CTA into the table's ``T_TL`` array)."""

    def __init__(self, *, alpha=1.702, beta=1.0, limit=7.0, need_renorm=True, tl=False, only=None, x_flag=False,
                 poll_sleep=POLL_SLEEP, standalone=False, cl=CLUSTER):
        self.poll_sleep = int(poll_sleep)
        # cl: the cluster size. 8 (launch A's clusters): the router is one cluster, 2 chains a wave (one from LDS);
        # 16 (own launch): the router is production's moe_router_topk geometry, 16 workgroups = 8 K splits x 2
        # expert groups, one chain a wave with its weights in VGPRs, and a shared cluster holds 4 h groups
        assert cl in (8, 16)
        self.cl = int(cl)
        # standalone: the phase is its own launch after production's k5 AR + RMSNorm; x is ready at entry, so
        # there is no x-ready wait and the x loads go out before the weights (the loads return in order)
        self.standalone = bool(standalone)
        self.only = only  # compile-time debug: "router" / "shared" builds one role only
        # x_flag: x is its own ready signal (R1 stores ``x_flag_word`` of each word at device scope; the phase
        # polls the words until none is 0xffff; the router CTAs re-arm the previous user's buffer, T_XPREV).
        # Otherwise R1 counts on ``x_ready`` after its stores drained.
        self.x_flag = bool(x_flag)
        self.alpha, self.beta, self.limit = float(alpha), float(beta), float(limit)
        self.need_renorm = bool(need_renorm)
        self.tl = bool(tl)

    # ------------------------------------------------------------------------------------------ entry
    @flyc.jit
    def run(self, role, lds_ptr, arg_tab, x_ready_addr, x_target, err_addr, maxit, x_ptr=None):
        """``role``: 0..103 (uniform in the CTA; ``role // 8`` uniform in the launch cluster).
        ``lds_ptr``: shared Uint8 pointer to >= ``LDS_BYTES`` bytes this CTA no longer uses.
        ``arg_tab``: the device table (``T_*`` int64 words). ``x_ready_addr``: i64 address of R1's int32
        x-ready count (uncached memory, never reset: each launch's 8 receiver CTAs add 1 each after their x
        stores drained), polled until it reaches ``x_target`` = (launch epoch + 1) * 8, compared mod 2^26 (so
        an epoch taken from ATT's arrival count, count >> 9, is enough). Unused with ``x_flag``.
        ``err_addr``: i64 address of an int32 timeout counter. ``maxit``: poll bound. ``x_ptr``: x as a kernel
        argument (contiguous rows of 6144; overrides the table's ``T_X`` / ``T_LDX``), for a graph-captured caller
        whose x buffer is only known at capture time."""
        tab = _rsrc(fx.ptrtoint(arg_tab))
        if const_expr(self.only == "router"):
            self._router(role % CLUSTER, lds_ptr, tab, x_ready_addr, x_target, err_addr, maxit, x_ptr)
        elif const_expr(self.only == "shared"):
            self._shared(role % (N_ROLES - CLUSTER), lds_ptr, tab, x_ready_addr, x_target, err_addr, maxit, x_ptr)
        else:
            if role < self.cl:
                if const_expr(self.cl == 16):
                    self._router16(role, lds_ptr, tab, x_ptr)
                else:
                    self._router(role, lds_ptr, tab, x_ready_addr, x_target, err_addr, maxit, x_ptr)
            else:
                self._shared(role - self.cl, lds_ptr, tab, x_ready_addr, x_target, err_addr, maxit, x_ptr)

    @flyc.jit
    def _wait_x(self, rank, x_ready_addr, x_target, err_addr, maxit):
        """One poller a cluster (CTA rank 0, thread 0) on the x-ready count, then the cluster barrier: 13
        pollers on the line instead of 104. The count lives in normal memory (device-scope loads, ``poll_sleep``
        >= 16) or, measured as the option, uncached memory (few pollers only)."""
        tid = fx.Int32(gpu.thread_id("x"))
        if const_expr(self.x_flag or self.standalone):
            return
        if (rank == 0) & (tid == 0):
            # epoch counter (never reset): wait until (cur - target) mod 2^26 >= 0 (wrap-safe, see run())
            it = fx.Int32(0)
            cur = _load_agent_i32(x_ready_addr)
            while (((cur - x_target) << fx.Int32(6)) < fx.Int32(0)) & (it < maxit):
                rocdl.s_sleep(self.poll_sleep)
                cur = _load_agent_i32(x_ready_addr)
                it = it + 1
            if ((cur - x_target) << fx.Int32(6)) < fx.Int32(0):
                _atomic_add_agent(err_addr, 1)
        cluster.cluster_barrier()

    def _tl_base(self, tab, idx):
        return _word(tab, T_TL) + fx.Int64(idx * (TL_SLOTS * 8)) if self.tl else None

    @flyc.jit
    def _stamp(self, tl_base, slot):
        if const_expr(self.tl):
            tid = fx.Int32(gpu.thread_id("x"))
            if const_expr(slot == TL_SLOTS - 2):  # the end stamp: after every thread's stores drained
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
            if tid == 0:
                fx.ptr_store(_now(), _gptr(tl_base, fx.Int64, 8) + slot)

    # ------------------------------------------------------------------------------------------ router
    @flyc.jit
    def _router(self, s, lds_ptr, tab, x_ready_addr, x_target, err_addr, maxit, x_ptr=None):
        tid = fx.Int32(gpu.thread_id("x"))
        wave = _uni(tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        tl_base = self._tl_base(tab, s)
        self._stamp(tl_base, 0)
        lds_idx = fx.Int64(fx.ptrtoint(lds_ptr))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        w_addr = _word(tab, T_RW)
        w_rs = _rsrc(w_addr, EXPERTS * HIDDEN * 2)
        # x: token rows 0..7 x K [768 s, +768) (in launch A device scope: R1 wrote it in this launch). Thread i: MX
        # blocks i and i + 128 (< 192) of 32 values (tok = b // 24, j = b % 24) -> the B fragments in LDS + MXFP4.
        x_addr = _word(tab, T_X) if x_ptr is None else fx.Int64(fx.ptrtoint(x_ptr))
        ldx = fx.Int32(_word(tab, T_LDX)) if x_ptr is None else fx.Int32(HIDDEN)
        x_rs = _rsrc(x_addr)
        xq_rs = _rsrc(_word(tab, T_XQ), ROWS * (HIDDEN // 2))
        xs_rs = _rsrc(_word(tab, T_XS), ROWS * (HIDDEN // 32))
        rb = [tid, tid + 128]
        rlive = [rb[0] < ROWS * R_STEPS, rb[1] < ROWS * R_STEPS]
        rtok = [rlive[k].select(rb[k], fx.Int32(0)) // R_STEPS for k in range_constexpr(2)]
        rj = [rlive[k].select(rb[k], fx.Int32(0)) % R_STEPS for k in range_constexpr(2)]
        rdw = [(rtok[k] * ldx + s * R_CHUNK + rj[k] * 32) // 2 for k in range_constexpr(2)]
        x_cpol = 0 if self.standalone else SCOPE_DEV

        def load_x():
            return [Vec(_ld(x_rs, rdw[k] + c * 4, 4, x_cpol)) for k in range_constexpr(2) for c in range_constexpr(4)]

        if const_expr(self.standalone):
            c0, c1, c2, c3, c4, c5, c6, c7 = load_x()
            rocdl.sched_barrier(0)

        # chain A (slice ``wave``): 24 steps x (lo, hi) 16 B of row ``16 wave + lane16`` into VGPRs.
        # Production's fragment: K [32 j + 8 kgrp, +8) and [32 j + 16 + 8 kgrp, +8) of the split's chunk.
        a_dw = ((wave * 16 + lane16) * HIDDEN + s * R_CHUNK + kgrp * 8) // 2
        wa = [
            [Vec(_ld(w_rs, a_dw + j * 16 + h * 8, 4, W_NT)) for h in range_constexpr(2)] for j in range_constexpr(R_STEPS)
        ]
        # chain B (slice ``wave + 4``): TDM rows [64 + 16 wave, +16) x K bytes [1536 s, +1536) -> LDS, 16 B pad
        # after every 512 B (the oproj_ar activation chunk's pad), one descriptor a wave.
        gW = fx.recast_iter(fx.Int8, fx.inttoptr(fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Global, 16), w_addr))
        b_row0 = 64 + wave * 16
        desc = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=fx.Tensor(
                fx.make_view(
                    fx.add_offset(gW, fx.Int64(b_row0) * (HIDDEN * 2) + fx.Int64(s * R_ROWB)),
                    fx.make_layout((16, R_ROWB), (HIDDEN * 2, 1)),
                )
            ),
            lds_memref=fx.Tensor(
                fx.make_view(fx.add_offset(lds_ptr, R_WB + wave * 16 * R_PITCH), fx.make_layout((16, R_ROWB), (R_PITCH, 1)))
            ),
            global_offset=(0, 0),
            tensor_shape=(16, R_ROWB),
            strides=(HIDDEN * 2, 1),
            tile_shape=(16, R_ROWB),
            elem_bytes=1,
            pad_interval=R_PADI,
            pad_amount=R_PADA,
            num_warps=1,
            workgroup_mask=0,
        )
        tdm_ops.tensor_load_2d(desc, cache_policy=W_NT)
        rocdl.sched_barrier(0)
        if const_expr(self.x_flag):
            # re-arm the previous user's x buffer (its launch is over; its next writer is a later launch):
            # 8 rows x 768 16-B chunks over the 8 router CTAs
            prev = _word(tab, T_XPREV)
            if prev != fx.Int64(0):
                prs = _rsrc(prev)
                pldx = fx.Int32(_word(tab, T_LDX))
                ones = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)
                for k in range_constexpr(ROWS * (HIDDEN // 8) // (CLUSTER * 128)):
                    cidx = (s * 128 + tid) + k * (CLUSTER * 128)
                    buffer_store(ones, prs, (cidx // (HIDDEN // 8) * pldx + cidx % (HIDDEN // 8) * 8) // 2,
                                 cache_modifier=SCOPE_DEV)
        self._stamp(tl_base, 1)

        self._wait_x(s, x_ready_addr, x_target, err_addr, maxit)

        if const_expr(not self.standalone):
            c0, c1, c2, c3, c4, c5, c6, c7 = load_x()
        if const_expr(self.x_flag):
            pend = _pending((c0, c1, c2, c3, c4, c5, c6, c7))
            it = fx.Int32(0)
            while pend & (it < maxit):
                rocdl.s_sleep(1)
                _mem_clobber()
                c0, c1, c2, c3, c4, c5, c6, c7 = load_x()
                pend = _pending((c0, c1, c2, c3, c4, c5, c6, c7))
                it = it + 1
            if pend & (lane == 0):
                _atomic_add_agent(err_addr, 1)
        self._stamp(tl_base, 2)
        xch = [[c0, c1, c2, c3], [c4, c5, c6, c7]]
        for rnd in range_constexpr(2):
            rocdl.sched_barrier(0)
            tok = rtok[rnd]
            j = rj[rnd]
            ch = xch[rnd]
            if rlive[rnd]:
                for c in range_constexpr(4):
                    # chunk c = K [8 c, +8): half c // 2, kgrp c % 2
                    off = R_XB + ((j * 2 + c // 2) * 16 + (c % 2) * 8 + tok) * 16
                    lds_store_b128(lds_idx, off, ch[c])
                bfs = [ch[c].bitcast(fx.BFloat16) for c in range_constexpr(4)]
                amax = _amax_bf16(ch)
                e8m0 = fx.Int32(emit_mx_e8m0_scale(amax, mode=_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1))
                sc = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                p = [fx.Int32(_cvt_scalef32_pk8_fp4_bf16(bfs[c], sc, i32_ty=T.i32)) for c in range_constexpr(4)]
                buffer_store(
                    Vec.from_elements(p, fx.Int32), xq_rs, tok * (HIDDEN // 2) + s * (R_CHUNK // 2) + j * 16,
                    offset_is_bytes=True,
                )
                buffer_store(fx.Uint8(e8m0), xs_rs, tok * (HIDDEN // 32) + s * R_STEPS + j, offset_is_bytes=True)
        tdm_ops.tensor_wait(0)
        _lds_barrier()
        self._stamp(tl_base, 3)

        # chains: x fragment of token lane16 % 8 (lanes 8..15 duplicate; their columns are never read)
        tok16 = lane16 % 8
        b_base = R_WB + (wave * 16 + lane16) * R_PITCH

        def x_frag(j):
            # (lanes 8..15 read token lane16 - 8 again: a branch to skip them measured slower)
            return _cat2(
                lds_load_b128(lds_idx, R_XB + ((j * 2 + 0) * 16 + kgrp * 8 + tok16) * 16),
                lds_load_b128(lds_idx, R_XB + ((j * 2 + 1) * 16 + kgrp * 8 + tok16) * 16),
            ).bitcast(fx.BFloat16)

        def wb_frag(j):
            # logical row byte 64 j + 32 h + 16 kgrp; the TDM pad adds 16 B per 512 B (a step never crosses one)
            pj = b_base + j * 64 + (j * 64 // R_PADI) * R_PADA + kgrp * 16
            return _cat2(lds_load_b128(lds_idx, pj), lds_load_b128(lds_idx, pj + 32)).bitcast(fx.BFloat16)

        wmma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
        acc_a = fx.full(8, 0.0, fx.Float32)
        acc_b = fx.full(8, 0.0, fx.Float32)
        PF = 3
        pend = [(x_frag(j), wb_frag(j)) for j in range_constexpr(PF)]
        for j in range_constexpr(R_STEPS):
            if const_expr(j + PF < R_STEPS):
                pend.append((x_frag(j + PF), wb_frag(j + PF)))
            rocdl.sched_barrier(0)
            xf, wbf = pend[j]
            x_t = _rmem(xf, 16, fx.BFloat16)
            ta = _rmem(acc_a, 8, fx.Float32)
            fx.gemm(wmma, ta, _rmem(_cat2(wa[j][0], wa[j][1]).bitcast(fx.BFloat16), 16, fx.BFloat16), x_t, ta)
            acc_a = ta.load()
            tb = _rmem(acc_b, 8, fx.Float32)
            fx.gemm(wmma, tb, _rmem(wbf, 16, fx.BFloat16), x_t, tb)
            acc_b = tb.load()
            pend[j] = None
            rocdl.sched_barrier(0)

        # acc[i]: expert 16 slice + 8 kgrp + i of token lane16 -> scratch [s][tok][expert]
        scr = _word(tab, T_SCR)
        scr_rs = _rsrc(scr)
        if lane16 < ROWS:
            for ci, acc in ((0, acc_a), (1, acc_b)):
                slc = wave + 4 * ci
                av = Vec(acc)
                for half in range_constexpr(2):
                    part = Vec.from_elements([av[half * 4 + i] for i in range_constexpr(4)], fx.Float32)
                    buffer_store(part, scr_rs, (s * ROWS + lane16) * EXPERTS + slc * 16 + kgrp * 8 + half * 4)
        # the gate's inputs that do not depend on the exchange, loaded before it
        bias_rs = _rsrc(_word(tab, T_RB))
        bias_v = [fx.Float32(buffer_load(bias_rs, lane + WAVE * i, vec_width=1, dtype=T.f32)) for i in range_constexpr(4)]
        nvr_addr = _word(tab, T_NVR)
        has_nvr = nvr_addr != fx.Int64(0)
        nvr = fx.Int32(buffer_load(_rsrc(has_nvr.select(nvr_addr, scr)), fx.Int32(0), vec_width=1, dtype=T.i32))
        self._stamp(tl_base, 4)
        _cluster_sync()
        self._stamp(tl_base, 5)

        # gate row s (wave 0): moe_router_topk's _gate_row on the 8 splits' partials
        if wave == 0:
            out_w, out_id = self._gate_row(scr_rs, s, lane, bias_v, has_nvr, nvr)
            if lane < TOPK:
                buffer_store(out_w, _rsrc(_word(tab, T_WTS)), s * TOPK + lane)
                buffer_store(out_id, _rsrc(_word(tab, T_IDS)), s * TOPK + lane)
        self._stamp(tl_base, 6)

    # ------------------------------------------------------------------------------------------ router, cluster 16
    @flyc.jit
    def _router16(self, c, lds_ptr, tab, x_ptr=None):
        """moe_router_topk's own geometry (split_k 8, 2 expert groups, cluster 16): CTA c = (split s = c // 2,
        group ng = c % 2); wave w runs slice 4 ng + w's 24-step chain, weights in VGPRs. x of the split goes to LDS
        (B fragments); the CTA writes MX blocks i (ng 0) / i + 128 (ng 1) of the split's MXFP4 rows. Then the
        cluster sync, and CTA c < 8 gates token row c."""
        tid = fx.Int32(gpu.thread_id("x"))
        wave = _uni(tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        s = c // 2
        ng = c % 2
        tl_base = self._tl_base(tab, c)
        self._stamp(tl_base, 0)
        lds_idx = fx.Int64(fx.ptrtoint(lds_ptr))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        x_addr = _word(tab, T_X) if x_ptr is None else fx.Int64(fx.ptrtoint(x_ptr))
        ldx = fx.Int32(_word(tab, T_LDX)) if x_ptr is None else fx.Int32(HIDDEN)
        x_rs = _rsrc(x_addr)
        xq_rs = _rsrc(_word(tab, T_XQ), ROWS * (HIDDEN // 2))
        xs_rs = _rsrc(_word(tab, T_XS), ROWS * (HIDDEN // 32))
        rb = [tid, tid + 128]
        rlive = [rb[0] < ROWS * R_STEPS, rb[1] < ROWS * R_STEPS]
        rtok = [rlive[k].select(rb[k], fx.Int32(0)) // R_STEPS for k in range_constexpr(2)]
        rj = [rlive[k].select(rb[k], fx.Int32(0)) % R_STEPS for k in range_constexpr(2)]
        rdw = [(rtok[k] * ldx + s * R_CHUNK + rj[k] * 32) // 2 for k in range_constexpr(2)]
        x_cpol = 0 if self.standalone else SCOPE_DEV
        xc = [Vec(_ld(x_rs, rdw[k] + cc * 4, 4, x_cpol)) for k in range_constexpr(2) for cc in range_constexpr(4)]
        rocdl.sched_barrier(0)
        w_rs = _rsrc(_word(tab, T_RW), EXPERTS * HIDDEN * 2)
        slc = ng * 4 + wave
        a_dw = ((slc * 16 + lane16) * HIDDEN + s * R_CHUNK + kgrp * 8) // 2
        wa = [
            [Vec(_ld(w_rs, a_dw + j * 16 + h * 8, 4, W_NT)) for h in range_constexpr(2)] for j in range_constexpr(R_STEPS)
        ]
        rocdl.sched_barrier(0)
        self._stamp(tl_base, 1)
        self._stamp(tl_base, 2)
        xch = [xc[0:4], xc[4:8]]
        for rnd in range_constexpr(2):
            rocdl.sched_barrier(0)
            tok = rtok[rnd]
            j = rj[rnd]
            ch = xch[rnd]
            if rlive[rnd]:
                for cc in range_constexpr(4):
                    lds_store_b128(lds_idx, ((j * 2 + cc // 2) * 16 + (cc % 2) * 8 + tok) * 16, ch[cc])
                if ng == rnd:
                    bfs = [ch[cc].bitcast(fx.BFloat16) for cc in range_constexpr(4)]
                    amax = _amax_bf16(ch)
                    e8m0 = fx.Int32(emit_mx_e8m0_scale(amax, mode=_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1))
                    sc = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                    pq = [fx.Int32(_cvt_scalef32_pk8_fp4_bf16(bfs[cc], sc, i32_ty=T.i32)) for cc in range_constexpr(4)]
                    buffer_store(
                        Vec.from_elements(pq, fx.Int32), xq_rs, tok * (HIDDEN // 2) + s * (R_CHUNK // 2) + j * 16,
                        offset_is_bytes=True,
                    )
                    buffer_store(fx.Uint8(e8m0), xs_rs, tok * (HIDDEN // 32) + s * R_STEPS + j, offset_is_bytes=True)
        _lds_barrier()
        self._stamp(tl_base, 3)

        tok16 = lane16 % 8

        def x_frag(j):
            return _cat2(
                lds_load_b128(lds_idx, ((j * 2 + 0) * 16 + kgrp * 8 + tok16) * 16),
                lds_load_b128(lds_idx, ((j * 2 + 1) * 16 + kgrp * 8 + tok16) * 16),
            ).bitcast(fx.BFloat16)

        wmma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
        acc = fx.full(8, 0.0, fx.Float32)
        PF = 3
        pend = [x_frag(j) for j in range_constexpr(PF)]
        for j in range_constexpr(R_STEPS):
            if const_expr(j + PF < R_STEPS):
                pend.append(x_frag(j + PF))
            rocdl.sched_barrier(0)
            ta = _rmem(acc, 8, fx.Float32)
            fx.gemm(wmma, ta, _rmem(_cat2(wa[j][0], wa[j][1]).bitcast(fx.BFloat16), 16, fx.BFloat16),
                    _rmem(pend[j], 16, fx.BFloat16), ta)
            acc = ta.load()
            pend[j] = None
            rocdl.sched_barrier(0)

        scr = _word(tab, T_SCR)
        scr_rs = _rsrc(scr)
        if lane16 < ROWS:
            av = Vec(acc)
            for half in range_constexpr(2):
                part = Vec.from_elements([av[half * 4 + i] for i in range_constexpr(4)], fx.Float32)
                buffer_store(part, scr_rs, (s * ROWS + lane16) * EXPERTS + slc * 16 + kgrp * 8 + half * 4)
        bias_rs = _rsrc(_word(tab, T_RB))
        bias_v = [fx.Float32(buffer_load(bias_rs, lane + WAVE * i, vec_width=1, dtype=T.f32)) for i in range_constexpr(4)]
        nvr_addr = _word(tab, T_NVR)
        has_nvr = nvr_addr != fx.Int64(0)
        nvr = fx.Int32(buffer_load(_rsrc(has_nvr.select(nvr_addr, scr)), fx.Int32(0), vec_width=1, dtype=T.i32))
        self._stamp(tl_base, 4)
        _cluster_sync()
        self._stamp(tl_base, 5)
        if (wave == 0) & (c < ROWS):
            out_w, out_id = self._gate_row(scr_rs, c, lane, bias_v, has_nvr, nvr)
            if lane < TOPK:
                buffer_store(out_w, _rsrc(_word(tab, T_WTS)), c * TOPK + lane)
                buffer_store(out_id, _rsrc(_word(tab, T_IDS)), c * TOPK + lane)
        self._stamp(tl_base, 6)

    def _gate_row(self, scr_rs, row, lane, bias_v, has_nvr, nvr):
        """(weight, id) of top-k slot ``lane`` (lanes < 4) of token ``row``: moe_router_topk's arithmetic."""
        neg_inf = float("-inf")
        ept = EXPERTS // WAVE
        parts = [
            [
                fx.Float32(
                    buffer_load(scr_rs, (sidx * ROWS + row) * EXPERTS + lane + WAVE * i, vec_width=1, dtype=T.f32)
                )
                for i in range_constexpr(ept)
            ]
            for sidx in range_constexpr(R_SPLIT)
        ]
        vals, orig, idxs = [], [], []
        for i in range_constexpr(ept):
            total = parts[0][i]
            for sidx in range_constexpr(1, R_SPLIT):
                total = total + parts[sidx][i]
            logit = total.to(fx.BFloat16).to(fx.Float32)
            e2 = fx.exp2((-logit) * fx.Float32(LOG2E))
            score = fx.Float32(rocdl.rcp(T.f32, fx.Float32(1.0) + e2))
            score = (score < fx.Float32(FLT_MIN)).select(fx.Float32(0.0), score)
            is_nan = score != score  # noqa: PLR0124 -- traced value: the NaN test
            orig.append(is_nan.select(fx.Float32(0.0), score))
            v = score + bias_v[i]
            v = (v != v).select(fx.Float32(neg_inf), v)  # noqa: PLR0124
            vals.append(v)
            idxs.append(lane + WAVE * i)

        def cas(a, b):
            sw = vals[a] < vals[b]
            vals[a], vals[b] = sw.select(vals[b], vals[a]), sw.select(vals[a], vals[b])
            orig[a], orig[b] = sw.select(orig[b], orig[a]), sw.select(orig[a], orig[b])
            idxs[a], idxs[b] = sw.select(idxs[b], idxs[a]), sw.select(idxs[a], idxs[b])

        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
            cas(a, b)

        cursor = fx.Int32(0)
        wsum = fx.Float32(0.0)
        my_id = fx.Int32(0)
        my_w = fx.Float32(0.0)
        for k in range_constexpr(TOPK):
            head_v = fx.Float32(neg_inf)
            head_i = fx.Int32(0)
            head_o = fx.Float32(0.0)
            for i in range_constexpr(ept - 1, -1, -1):
                at = cursor == i
                head_v = at.select(vals[i], head_v)
                head_i = at.select(idxs[i], head_i)
                head_o = at.select(orig[i], head_o)
            mx = head_v
            for off in (16, 8, 4, 2, 1):
                mx = fx.maxnumf(mx, mx.shuffle_xor(fx.Int32(off), fx.Int32(WAVE)))
            mask = fx.Int32(rocdl.ballot(T.i32, head_v == mx))
            win = (mask != 0).select(fx.Int32(fx.cttz(mask)), fx.Int32(0))
            win_id = fx.Int32(rocdl.readlane(T.i32, head_i, win))
            won = (cursor < ept) & (head_i == win_id)
            mine = won.select(head_o, fx.Float32(0.0))
            cursor = cursor + won.select(fx.Int32(1), fx.Int32(0))
            weight = fx.Int32(rocdl.readlane(T.i32, mine.bitcast(fx.Int32), win_id & (WAVE - 1))).bitcast(fx.Float32)
            is_k = lane == k
            my_id = is_k.select(win_id, my_id)
            my_w = is_k.select(weight, my_w)
            if const_expr(self.need_renorm):
                wsum = wsum + weight
        if const_expr(self.need_renorm):
            inv = fx.Float32(1.0) / fx.maxnumf(wsum, fx.Float32(RENORM_SUM_FLOOR))
        else:
            inv = fx.Float32(1.0)
        out_w = my_w * inv
        pad = has_nvr & (row >= nvr)
        out_w = pad.select(fx.Float32(0.0), out_w)
        out_id = pad.select(fx.Int32(-1), my_id)
        return out_w, out_id

    # ------------------------------------------------------------------------------------------ shared
    @flyc.jit
    def _shared(self, r, lds_ptr, tab, x_ready_addr, x_target, err_addr, maxit, x_ptr=None):
        tid = fx.Int32(gpu.thread_id("x"))
        wave = _uni(tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        cl = self.cl
        c = r // cl
        p = r % cl
        gl = p // S_SPLIT
        g = c * (cl // S_SPLIT) + gl
        s = p % S_SPLIT
        tl_base = self._tl_base(tab, cl + r)
        self._stamp(tl_base, 0)
        lds_idx = fx.Int64(fx.ptrtoint(lds_ptr))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)

        # weights of tile ``wave``: lane16 < 8 -> gate row 32 g + 8 wave + lane16, else the same up row;
        # preshuffled (16, 16) E4M3 (16 B unit (R // 16) * K + (k // 16) * 16 + R % 16) and n32k4 E8M0
        # (word (R // 32) * K / 4 + k128 * 32 + R % 32), the layouts production's GEMM reads.
        # x -> MXFP8 (quantize_fp8's bytes: amax floor 1e-10, ceil pow2 of amax / 448, RNE E4M3), into the B
        # fragment layout. Thread i: MX blocks i, i + 128, i + 256 of the 384 (token b // 48, block b % 48).
        x_rs = _rsrc(_word(tab, T_X) if x_ptr is None else fx.Int64(fx.ptrtoint(x_ptr)))
        ldx = fx.Int32(_word(tab, T_LDX)) if x_ptr is None else fx.Int32(HIDDEN)
        xb = [tid + m * 128 for m in range_constexpr(3)]
        xtok = [b // (S_CHUNK // 32) for b in xb]
        xkb = [b % (S_CHUNK // 32) for b in xb]
        x_cpol = 0 if self.standalone else SCOPE_DEV

        def load_x():
            return [
                Vec(_ld(x_rs, (xtok[m] * ldx + s * S_CHUNK + xkb[m] * 32) // 2 + u * 4, 4, x_cpol))
                for m in range_constexpr(3)
                for u in range_constexpr(4)
            ]

        if const_expr(self.standalone):
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = load_x()
            rocdl.sched_barrier(0)

        row = (lane16 < 8).select(g * GROUP + wave * 8 + lane16, S_INTER + g * GROUP + wave * 8 + lane16 - 8)
        w_rs = _rsrc(_word(tab, T_SW), 2 * S_INTER * HIDDEN)
        ws_rs = _rsrc(_word(tab, T_SS), 2 * S_INTER * (HIDDEN // 32))
        # scales first (they are parked in LDS once they land, freeing their VGPRs), then the weights
        ws_w = (row // 32) * (HIDDEN // 4) + row % 32
        wsc = [fx.Int32(_ld(ws_rs, ws_w + (s * S_STEPS + ks) * 32, 1, W_NT)) for ks in range_constexpr(S_STEPS)]
        rocdl.sched_barrier(0)
        w_u = (row // 16) * HIDDEN + row % 16
        wf = [
            [Vec(_ld(w_rs, (w_u + ((s * S_STEPS + ks) * 8 + jj * 2 + kgrp) * 16) * 4, 4, W_NT)) for jj in range_constexpr(4)]
            for ks in range_constexpr(S_STEPS)
        ]
        rocdl.sched_barrier(0)
        for ks in range_constexpr(S_STEPS):
            lds_store_b32(lds_idx, S_WS + ((wave * S_STEPS + ks) * WAVE + lane) * 4, Vec.from_elements([wsc[ks]], fx.Int32))
        self._stamp(tl_base, 1)

        self._wait_x(p, x_ready_addr, x_target, err_addr, maxit)

        if const_expr(not self.standalone):
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = load_x()
        if const_expr(self.x_flag):
            pend = _pending((c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11))
            it = fx.Int32(0)
            while pend & (it < maxit):
                rocdl.s_sleep(1)
                _mem_clobber()
                c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = load_x()
                pend = _pending((c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11))
                it = it + 1
            if pend & (lane == 0):
                _atomic_add_agent(err_addr, 1)
        self._stamp(tl_base, 2)
        xch = [[c0, c1, c2, c3], [c4, c5, c6, c7], [c8, c9, c10, c11]]
        for m in range_constexpr(3):
            rocdl.sched_barrier(0)
            bfs = [xch[m][cc].bitcast(fx.BFloat16) for cc in range_constexpr(4)]
            xamax = _amax_bf16(xch[m])
            xe8 = fx.Int32(
                emit_mx_e8m0_scale(
                    fx.max(xamax, fx.Float32(_MXFP8_COPY_AMAX_FLOOR)), mode=_ROUND_MODE, dtype=MxDtypeInt.FP8_E4M3
                )
            )
            xsc8 = (xe8 << fx.Int32(23)).bitcast(fx.Float32)
            qv = [Vec(_cvt_scalef32_pk8_fp8_bf16(bfs[cc], xsc8, v2i32_ty=T.vec(2, T.i32))) for cc in range_constexpr(4)]
            xks = xkb[m] // 4
            xq = xkb[m] % 4
            # bytes [32 xq, +16) of step xks -> (jj = xq, kgrp 0), [32 xq + 16, +16) -> (jj = xq, kgrp 1)
            for kg in range_constexpr(2):
                unit = Vec.from_elements([qv[2 * kg][0], qv[2 * kg][1], qv[2 * kg + 1][0], qv[2 * kg + 1][1]], fx.Int32)
                lds_store_b128(lds_idx, S_XB + ((xks * 4 + xq) * 16 + kg * 8 + xtok[m]) * 16, unit)
            fx.ptr_store(fx.Uint8(xe8), fx.add_offset(lds_ptr, S_SC + (xtok[m] * S_STEPS + xks) * 4 + xq))
        _lds_barrier()
        self._stamp(tl_base, 3)

        tok16 = lane16 % 8
        atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0, opsel_b=0)
        )

        def x_frag(ks):
            return [lds_load_b128(lds_idx, S_XB + ((ks * 4 + jj) * 16 + kgrp * 8 + tok16) * 16) for jj in range_constexpr(4)]

        def x_scale(ks):
            return (
                fx.Int32(Vec(lds_load_b32(lds_idx, S_SC + (tok16 * S_STEPS + ks) * 4))[0]),
                fx.Int32(Vec(lds_load_b32(lds_idx, S_WS + ((wave * S_STEPS + ks) * WAVE + lane) * 4))[0]),
            )

        acc = fx.make_rmem_tensor(8, fx.Float32)
        acc.store(fx.full(8, 0.0, fx.Float32))
        PF = 2
        pend = [(x_frag(ks), x_scale(ks)) for ks in range_constexpr(PF)]
        for ks in range_constexpr(S_STEPS):
            if const_expr(ks + PF < S_STEPS):
                pend.append((x_frag(ks + PF), x_scale(ks + PF)))
            rocdl.sched_barrier(0)
            xf, (xs, wsk) = pend[ks]
            fx.gemm(
                atom, acc, _rmem(_cat4(wf[ks]), 16, fx.Int32), _rmem(_cat4(xf), 16, fx.Int32), acc,
                scale_a=wsk, scale_b=xs,
            )
            pend[ks] = None
            rocdl.sched_barrier(0)

        # acc[i]: tile row 8 kgrp + i (kgrp 0: gate, 1: up) of token lane16 -> scratch
        # [cluster][rank p][tile][tok][16 rows]
        scr_rs = _rsrc(_word(tab, T_SCR))
        av = Vec(acc.load())
        if lane16 < ROWS:
            base = SCR_R + (((c * cl + p) * 4 + wave) * ROWS + lane16) * 16 + kgrp * 8
            for half in range_constexpr(2):
                part = Vec.from_elements([av[half * 4 + i] for i in range_constexpr(4)], fx.Float32)
                buffer_store(part, scr_rs, base + half * 4)
        self._stamp(tl_base, 4)
        _cluster_sync()
        self._stamp(tl_base, 5)

        # thread: token tid // 16, tile (tid % 16) // 4, h columns 8 tile + 2 q + {0, 1} (q = tid % 4)
        tok = tid // 16
        tile = (tid % 16) // 4
        q = tid % 4
        gate = [None, None]
        up = [None, None]
        for sp in range_constexpr(S_SPLIT):
            base = SCR_R + (((c * cl + gl * S_SPLIT + sp) * 4 + tile) * ROWS + tok) * 16 + 2 * q
            gv = Vec(buffer_load(scr_rs, base, vec_width=2, dtype=T.f32))
            uv = Vec(buffer_load(scr_rs, base + 8, vec_width=2, dtype=T.f32))
            for e in range_constexpr(2):
                gate[e] = fx.Float32(gv[e]) if sp == 0 else gate[e] + fx.Float32(gv[e])
                up[e] = fx.Float32(uv[e]) if sp == 0 else up[e] + fx.Float32(uv[e])
        hb = []
        for e in range_constexpr(2):
            gt = gate[e].to(fx.BFloat16).to(fx.Float32)
            ut = up[e].to(fx.BFloat16).to(fx.Float32)
            gt = fx.minnumf(gt, fx.Float32(self.limit))
            ut = fx.minnumf(fx.maxnumf(ut, fx.Float32(-self.limit)), fx.Float32(self.limit))
            # Triton: gate * sigmoid(alpha * gate) * (up + beta); sigmoid = 1 / (1 + exp(-x)),
            # exp = exp2(x * log2 e) (llvm.exp2.f32), the division IEEE
            a = fx.Float32(self.alpha) * gt
            ex = fx.exp2((-a) * fx.Float32(LOG2E))
            sig = fx.Float32(1.0) / (fx.Float32(1.0) + ex)
            out = (gt * sig) * (ut + fx.Float32(self.beta))
            hb.append(out.to(fx.BFloat16))
        hv = Vec.from_elements(hb, fx.BFloat16)
        hf = hv.to(fx.Float32)
        amax = fx.max(abs(fx.Float32(hf[0])), abs(fx.Float32(hf[1])))
        for off in (1, 2, 4, 8):
            amax = fx.max(amax, amax.shuffle_xor(fx.Int32(off), fx.Int32(WAVE)))
        e8 = fx.Int32(
            emit_mx_e8m0_scale(fx.max(amax, fx.Float32(_MXFP8_COPY_AMAX_FLOOR)), mode=_ROUND_MODE, dtype=MxDtypeInt.FP8_E4M3)
        )
        sc8 = (e8 << fx.Int32(23)).bitcast(fx.Float32)
        # gather the tile's 8 columns (4 lanes x 2) into every lane of the 4, convert, lane q == 0 stores
        hw = hv.bitcast(fx.Int32)
        w0 = fx.Int32(Vec(hw)[0])
        ws_ = [w0, w0.shuffle_xor(fx.Int32(1), fx.Int32(WAVE)), w0.shuffle_xor(fx.Int32(2), fx.Int32(WAVE)),
               w0.shuffle_xor(fx.Int32(3), fx.Int32(WAVE))]
        # lane q holds pair q; lane q ^ m holds pair q ^ m: order the four pairs by column
        pairs = []
        for want in range_constexpr(4):
            pick = ws_[0]
            for m in range_constexpr(1, 4):
                pick = ((q ^ m) == want).select(ws_[m], pick)
            pick = (q == want).select(ws_[0], pick)
            pairs.append(pick)
        v8 = Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16)
        q8 = Vec(_cvt_scalef32_pk8_fp8_bf16(v8, sc8, v2i32_ty=T.vec(2, T.i32)))
        if (q == 0) & (tile == s):
            buffer_store(
                Vec.from_elements([q8[0], q8[1]], fx.Int32), _rsrc(_word(tab, T_H), ROWS * S_INTER),
                tok * S_INTER + g * GROUP + tile * 8, offset_is_bytes=True,
            )
        if (q == 0) & (tile == 0) & (s == 0):
            buffer_store(fx.Uint8(e8), _rsrc(_word(tab, T_HS), ROWS * N_GROUPS), tok * N_GROUPS + g, offset_is_bytes=True)
        self._stamp(tl_base, 6)


# ------------------------------------------------------------------------------------------------ test launch
_CACHE = {}


def compile_rsh_test(*, tl=False, delay_ticks=0, alpha=1.702, beta=1.0, limit=7.0, only=None, x_flag=False,
                     poll_sleep=POLL_SLEEP):
    """Standalone launch for the single-GPU check and timing: launch A's shape (512 CTAs x 128 threads,
    clusters of 8, 132 KB LDS a CTA). Clusters 0-12 run roles 0-103; CTA 104 plays R1: after
    ``delay_ticks`` steady-counter ticks (the weights stream meanwhile) it copies x_src -> x with device-scope
    stores, drains, and adds 8 to ``ctl[0]`` (uncached: the polled line). Counters are epoch counters, never
    reset: every CTA reads ``arr[0]`` (arrivals, normal memory: 512 atomics a launch on an uncached line
    serialize for hundreds of us) at entry, epoch = value >> 9, and adds 1 at exit; the x-ready target is
    (epoch + 1) * 8. ``ctl[2]``: poll timeouts. ``pub`` (int64[2]): publish stamps (tl builds)."""
    key = (tl, delay_ticks, alpha, beta, limit, only, x_flag, poll_sleep)
    if key in _CACHE:
        return _CACHE[key]
    phase = RshTail(alpha=alpha, beta=beta, limit=limit, tl=tl, only=only, x_flag=x_flag, poll_sleep=poll_sleep)
    GRID = 512
    LDS_WORDS = 132 * 1024 // 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, LDS_WORDS, 16]

    @flyc.kernel(name=f"rsh_tail_test_gfx1250{'_tl' if tl else ''}_d{delay_ticks}{'_' + only if only else ''}"
                      f"{'_xf' if x_flag else ''}_s{poll_sleep}",
                 known_block_size=(128, 1, 1))
    def rsh_test_kernel(arg_tab: fx.Pointer, arg_ctl: fx.Pointer, arg_xsrc: fx.Pointer, arg_pub: fx.Pointer,
                        arg_arr: fx.Pointer, maxit: fx.Int32):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        lds_all = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_ptr = fx.recast_iter(fx.Uint8, lds_all.buf.ptr)
        ctl = fx.Int64(fx.ptrtoint(arg_ctl))
        arr = fx.Int64(fx.ptrtoint(arg_arr))
        epoch = _uni(_load_agent_i32(arr)) >> fx.Int32(9)
        if bid < N_ROLES:
            phase.run(bid, lds_ptr, arg_tab, ctl, (epoch + 1) * ROWS, ctl + fx.Int64(8), maxit)
        if bid == N_ROLES:
            # R1 stand-in
            if const_expr(delay_ticks > 0):
                t0 = _now()
                t1 = _now()
                it = fx.Int32(0)
                while ((t1 - t0) < fx.Int64(delay_ticks)) & (it < maxit):
                    rocdl.s_sleep(8)
                    t1 = _now()
                    it = it + 1
            tab = _rsrc(fx.ptrtoint(arg_tab))
            x_rs = _rsrc(_word(tab, T_X))
            src_rs = _rsrc(fx.ptrtoint(arg_xsrc))
            ldx = fx.Int32(_word(tab, T_LDX))
            # 8 rows x 6144 bf16 = 6144 16-B chunks, 48 a thread
            for k in range_constexpr(48):
                cidx = tid + k * 128
                rr = cidx // (HIDDEN // 8)
                cc = cidx % (HIDDEN // 8)
                v = Vec(_ld(src_rs, cidx * 4, 4))
                if const_expr(x_flag):
                    v = Vec.from_elements([x_flag_word(fx.Int32(v[e])) for e in range_constexpr(4)], fx.Int32)
                buffer_store(v, x_rs, (rr * ldx + cc * 8) // 2, cache_modifier=SCOPE_DEV)
            if const_expr(tl):
                # pub[0]: the last x store issued (the data-as-flag origin); pub[1]: the count (counter mode)
                _lds_barrier()
                if tid == 0:
                    fx.ptr_store(_now(), _gptr(fx.ptrtoint(arg_pub), fx.Int64, 8))
            if const_expr(not x_flag):
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
                if tid == 0:
                    if const_expr(tl):
                        fx.ptr_store(_now(), _gptr(fx.ptrtoint(arg_pub), fx.Int64, 8) + 1)
                    _atomic_add_agent(ctl, ROWS)
        _lds_barrier()
        if tid == 0:
            _atomic_add_agent(arr, 1)

    @flyc.jit
    def launch(arg_tab: fx.Pointer, arg_ctl: fx.Pointer, arg_xsrc: fx.Pointer, arg_pub: fx.Pointer,
               arg_arr: fx.Pointer, maxit: fx.Int32, stream: fx.Stream):
        rsh_test_kernel(arg_tab, arg_ctl, arg_xsrc, arg_pub, arg_arr, maxit,
                        value_attrs={"rocdl.cluster_dims": f"{CLUSTER},1,1", "rocdl.waves_per_eu": 2}).launch(
            grid=(GRID, 1, 1), block=(128, 1, 1), stream=stream, cluster=(CLUSTER, 1, 1))

    return _CACHE.setdefault(key, launch)


def compile_rsh_launch(*, tl=False, alpha=1.702, beta=1.0, limit=7.0, x_arg=False, cl=16, shared_only=False):
    """The phase as its own launch after production's ``k5_ar_rms`` (its bf16 output is x): router GEMV + biased
    sigmoid top-4 + the MXFP4 token rows, and the shared gate_up + SwiGLU-OAI + MXFP8 h, replacing production's
    router launch and the shared expert's quantize / gate_up / SwiGLU launches (its down stays where it is).
    104 CTAs x 128 threads in clusters of 8 (the roles of ``RshTail``), LDS ``LDS_BYTES``. Launcher
    ``(arg_tab, stream)``: the table of ``rsh_tail.rsh_table``; ``x_arg``: ``(arg_tab, arg_x, stream)``, x (contiguous
    bf16 [8, 6144]) passed per call instead of the table's word (graph capture: x is allocated in the capture)."""
    key = ("launch", tl, alpha, beta, limit, x_arg, cl, shared_only)
    if key in _CACHE:
        return _CACHE[key]
    phase = RshTail(alpha=alpha, beta=beta, limit=limit, tl=tl, standalone=True, cl=cl)
    # the router's cluster + 96 shared CTAs; shared_only: the 96 shared CTAs (h only, e.g. on a side stream)
    n_roles = (N_ROLES - CLUSTER) if shared_only else cl + N_ROLES - CLUSTER
    role0 = cl if shared_only else 0

    lds_bytes = max(S_LDS, R_STEPS * 2 * 16 * 16) if cl == 16 else LDS_BYTES

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, lds_bytes // 4, 16]

    @flyc.kernel(name=f"moe_router_shared_gu_gfx1250_c{cl}{'_tl' if tl else ''}{'_xa' if x_arg else ''}"
                      f"{'_so' if shared_only else ''}",
                 known_block_size=(128, 1, 1))
    def rsh_launch_kernel(arg_tab: fx.Pointer, arg_x: fx.Pointer):
        bid = fx.Int32(gpu.block_id("x"))
        lds_all = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_ptr = fx.recast_iter(fx.Uint8, lds_all.buf.ptr)
        phase.run(bid + role0, lds_ptr, arg_tab, fx.Int64(0), fx.Int32(0), fx.Int64(0), fx.Int32(1),
                  arg_x if x_arg else None)

    if x_arg:

        @flyc.jit
        def launch_x(arg_tab: fx.Pointer, arg_x: fx.Pointer, stream: fx.Stream):
            rsh_launch_kernel(arg_tab, arg_x, value_attrs={"rocdl.cluster_dims": f"{cl},1,1"}).launch(
                grid=(n_roles, 1, 1), block=(128, 1, 1), stream=stream, cluster=(cl, 1, 1))

        return _CACHE.setdefault(key, launch_x)

    @flyc.jit
    def launch(arg_tab: fx.Pointer, stream: fx.Stream):
        rsh_launch_kernel(arg_tab, arg_tab, value_attrs={"rocdl.cluster_dims": f"{cl},1,1"}).launch(
            grid=(n_roles, 1, 1), block=(128, 1, 1), stream=stream, cluster=(cl, 1, 1))

    return _CACHE.setdefault(key, launch)


__all__ = ["RshTail", "compile_rsh_test", "compile_rsh_launch", "LDS_BYTES", "N_ROLES", "SCR_WORDS", "T_WORDS", "TL_SLOTS"] + [
    n for n in dir() if n.startswith("T_")
]
