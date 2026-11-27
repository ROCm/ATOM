# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Sparse PA + o_proj GEMV + TP4 one-shot all-reduce + residual + RMSNorm in one launch
(gfx1250, MiniMax-M3.1 decode: one request, 8 rows, TP4).

Production runs three launches: the NVFP4 PA tile kernel (plan-free, cluster-8 merge, MXFP8 output),
the MXFP8 o_proj decode GEMM and the k5 one-shot all-reduce + norm. This launch keeps the PA
launch's configuration (grid, 8-CTA clusters, 128 threads, the same tile body and merge) and gives
its padded clusters, which production leaves almost idle, the rest of the chain:

* PA CTAs (clusters of the request's rows): the production tile body and cluster merge, unchanged.
  Then each publishes: its stores drained, a device-scope release, one count on ``ctl[PA_DONE]``.
* o_proj CTAs (the first 384 padded CTAs, 16 output columns each): at entry, the column block's
  MXFP8 weights (32 KB, 64 VGPRs a lane) and scales go to registers, so the weight stream overlaps
  PA. Each wave owns K steps 4w..4w+3. After all PA CTAs have counted, the attention output
  (device-scope loads) goes through the same scaled WMMAs as the decode GEMM. Every K step is
  its own plane, and the 16 planes are summed in K order through LDS, as the decode GEMM's 16
  K-waves are: the bf16 partial has production's bits. The packs then go to every rank's k5 inbox,
  as in ``oproj_ar_gfx1250``.
* receiver CTAs (the next 8 padded CTAs, one row each, a wave two slabs): k5's receiver and Gluon
  epilogue (``oproj_ar_gfx1250``), out + residual_out, the row's 8 slab counters advanced.

Every CTA counts once on ``ctl[EXIT]``; the last one resets the launch's counters for the next.
Polls are bounded (``i32_maxit``) and count timeouts in ``ctl[ERR]``.
Requires: one request of 8 rows (8 PA sequences), xcd_rows 8 (the active clusters are those with
index % 8 == 0), 512 CTAs.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from aiter.ops.flydsl.kernels.communication_ops_utils import fence_release
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.mxfp8_decode_gemm_gfx1250 import P_PAD, WAVE, WMMA_K
from aiter.ops.flydsl.kernels.oproj_ar_gfx1250 import (
    BUFS,
    HIDDEN,
    RANKS,
    ROW_PACKS,
    RWAVES,
    SCOPE_SYS,
    SLAB,
    _bf_hi,
    _bf_lo,
    _bits,
    _f2bf2,
    _lds_barrier,
    _mem_clobber,
    _now32,
    _rcp,
    _rsq,
)
from aiter.ops.flydsl.kernels.pa_decode.context import PaDecodeContext
from .rsh_tail import LDS_BYTES as RSH_LDS
from .rsh_tail import RshTail
from aiter.ops.flydsl.kernels.pa_decode.op_merge import Gfx1250ClusterMerge, Gfx1250OutputMxQuant
from aiter.ops.flydsl.kernels.pa_decode.op_mxfp8 import FP8_MAX, GROUP, MxFp8Groups, group_code, group_reciprocal
from aiter.ops.flydsl.kernels.pa_decode.pipeline_nvfp4 import Gfx1250Nvfp4WavePipeline
from aiter.ops.flydsl.kernels.pa_decode.traits import KV_COMPUTE_BLOCK
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_base_i64,
    buf_copy_load,
    ptr_buf_tensor,
)

SCOPE_SE = 1 << 3  # gfx12 CPol SCOPE field: shader engine (a cluster's CTAs share it)
SCOPE_DEV = 2 << 3  # gfx12 CPol SCOPE field: device


class _DevGroups(MxFp8Groups):
    """``MxFp8Groups`` whose E4M3 words and E8M0 bytes are stored at device scope (``scope:SCOPE_DEV``),
    so a consumer on another XCD reads them after the producer's store drain alone. A release fence would
    write back the producer XCD's whole L2 instead (raw buffer stores: the typed pointer store has no scope
    operand)."""

    def __init__(self, quantized, scales):
        super().__init__(quantized, scales)
        self.q_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(quantized)))
        self.s_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(scales)))

    def _store_bytes(self, floats, offset, reciprocal):
        scaled = []
        for value in floats:
            value = value * reciprocal
            value = fx.maxnumf(value, fx.Float32(-FP8_MAX))
            scaled.append(fx.minnumf(value, fx.Float32(FP8_MAX)).ir_value())
        assert len(scaled) % 4 == 0, "device-scope MXFP8 stores take 4n values a lane"
        words = []
        for base in range_constexpr(0, len(scaled), 4):
            word = fx.Int32(0).ir_value()
            for half in range_constexpr(2):
                word = fx.rocdl.cvt_pk_fp8_f32(T.i32, scaled[base + 2 * half], scaled[base + 2 * half + 1], word, half)
            words.append(word)
        data = words[0] if len(words) == 1 else fx.Vector.from_elements(words, dtype=fx.Int32).ir_value()
        buffer_store(data, self.q_rs, fx.Int32(offset), offset_is_bytes=True, cache_modifier=SCOPE_DEV)

    @flyc.jit
    def store(self, values, offsets, masks, wave):
        lane_max = self._abs_max(values[0])
        for value in values[1:]:
            lane_max = fx.maxnumf(lane_max, self._abs_max(value))
        for mask in masks:
            lane_max = fx.maxnumf(lane_max, lane_max.shuffle_xor(fx.Int32(mask), fx.Int32(wave)))
        code = group_code(lane_max)
        reciprocal = group_reciprocal(code)
        for value, offset in zip(values, offsets):
            self._store_bytes(self._floats(value), offset, reciprocal)
        s_rs, first, byte = self.s_rs, fx.Int32(offsets[0]), code.to(fx.Int8)
        if (first & (GROUP - 1)) == 0:
            buffer_store(byte.ir_value(), s_rs, first // GROUP, offset_is_bytes=True, cache_modifier=SCOPE_DEV)


class _DevMxQuant(Gfx1250OutputMxQuant):
    """``Gfx1250OutputMxQuant`` with device-scope stores of the merged rows (``_DevGroups``)."""

    def __init__(self, t, quantized_output, output_scale):
        super().__init__(t, quantized_output, output_scale)
        self.groups = _DevGroups(quantized_output, output_scale)
POLL_SLEEP = 16  # guide §6: s_sleep 16 in polls. TP4 runs with s_sleep 2 / 4 desynced the k5 protocol (bit mismatch, stalled server); 16 runs clean
# ctl (int32): the launch's counters, reset by its last CTA; ERR accumulates
PA_DONE, EXIT, ERR, GEN = 0, 1, 2, 3
X_READY = 4  # r1: receiver CTAs that published x (the router phase's input); epoch counter, +8 a launch, never reset
N_COLS = 16  # o_proj columns a CTA
K_STEPS = 16  # K 2048 / 128
N_OPROJ = HIDDEN // N_COLS  # 384
N_RECV = 8
GRID = 512
# timeline builds (tl=1): int32 [GRID, TLW] steady-counter stamps a CTA. Slots 0-3: entry and three phase points;
# o_proj CTAs: 4 = own pushes issued; receivers (wave 0, slab 0): 4 + q = the last poll that found source rank q's
# packs incomplete (0: complete at the first look)
TLW = 16
CPUSH_BYTES = (N_OPROJ // 8) * 8 * 16 * 16  # cpush staging: o_proj clusters x rows x 16 packs x 16 B (98 KB of pout)

_CACHE = {}


def _load_agent_i32(ptr):
    return fx.Int32(
        fx.generic_load(
            ptr,
            volatile=True,
            memory_order=fx.AtomicOrdering.Monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        )
    )


def compile_pa_oproj_ar(t, *, max_rows=256, tl=0, dev_out=1, poll_sleep=POLL_SLEEP, gen_sync=1, cluster_poll=0,
                        w_delay=0, a_l2=0, cpush=0, w_cpol=1, recv_late=0, w_late=0, exit_cl=0, role_mode=0, r1=0, r1_router=1):
    """``t``: the production PA launch's ``Gfx1250Traits`` (NVFP4 wave tile, plan-free, cluster 8,
    MXFP8 output only, xcd_rows 8)."""
    key = (t, max_rows, tl, dev_out, poll_sleep, gen_sync, cluster_poll, w_delay, a_l2, cpush, w_cpol, recv_late, w_late, exit_cl, role_mode, r1, r1_router)
    if key in _CACHE:
        return _CACHE[key]
    if not (
        t.nvfp4_wave
        and t.plan_free
        and t.cluster_merge == 8
        and t.cluster_rows == 1
        and t.quant_output
        and t.quant_mx
        and not t.store_output
        and t.xcd_rows == 8
        and t.xcds == 8
        and t.query_splits == 1
        and t.BLOCK_THREADS in (4 * WAVE, 8 * WAVE)
        and t.head_dim * t.query_group_size == 2048
    ):
        raise ValueError(f"pa_oproj_ar: unsupported PA traits {t}")
    K = 2048
    KB4 = K // 128
    P_ROW = N_COLS * 4 + P_PAD
    P_PLANE = 16 * P_ROW
    TAIL_LDS = K_STEPS * P_PLANE + 2 * RWAVES * 4
    # r1: no PA in the launch (production's PA ran before it); o_proj + AR + the router (RSH's phase) on idle clusters
    r1_router = int(r1 and r1_router)
    lds_words = (max(t.total_bytes, TAIL_LDS, RSH_LDS if r1_router else 0) + 3) // 4
    rsh_router = RshTail(alpha=1.702, beta=1.0, limit=7.0, only="router", cl=8) if r1_router else None
    ctr_recv = max_rows * RWAVES
    ctr_err = 2 * max_rows * RWAVES
    block = t.BLOCK_THREADS
    # a_l2 (diagnostic): the attention output read through this XCD's L2 (one miss an XCD) instead of at
    # device scope; correct only if no line of it is cached here from an earlier launch
    A_CPOL = 0 if a_l2 else SCOPE_DEV
    NW = block // WAVE  # 4 (page_groups 1) or 8 (page_groups 2)
    KPW = K_STEPS // NW  # o_proj K steps a wave
    SPW = RWAVES // NW  # receiver slabs a wave

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, lds_words, 16]

    kernel_name = (f"pa_oproj_ar_k5_gfx1250{'_tl' if tl else ''}{'' if dev_out else '_wb'}_s{poll_sleep}"
                   f"{'_gen' if gen_sync else ''}{'_cp' if cluster_poll else ''}{f'_wd{w_delay}' if w_delay else ''}{'_al2' if a_l2 else ''}{'_cpush' if cpush else ''}{'' if w_cpol == 1 else f'_wc{w_cpol}'}{'_rl' if recv_late else ''}{'_wl' if w_late else ''}{'_ecl' if exit_cl else ''}{f'_rm{role_mode}' if role_mode else ''}{('_r1' if r1_router else '_r1n') if r1 else ''}"
                   f"{f'_pg{t.page_groups}' if t.page_groups != 1 else ''}")

    @flyc.kernel(name=kernel_name, known_block_size=(block, 1, 1))
    def pa_oproj_ar_kernel(
        work: fx.Pointer,
        table: fx.Pointer,
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        max_blocks: fx.Int32,
        sk_block: fx.Int32,
        sk_head: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        sequences: fx.Int32,
        capacity: fx.Int32,
        kv_heads: fx.Int32,
        reduce_info: fx.Pointer,
        output: fx.Pointer,
        counters: fx.Pointer,
        quantized_output: fx.Pointer,
        output_scale: fx.Pointer,
        quant_rows: fx.Int32,
        arg_w: fx.Pointer,
        arg_ws: fx.Pointer,
        arg_res: fx.Pointer,
        arg_nw: fx.Pointer,
        arg_out: fx.Pointer,
        arg_res_out: fx.Pointer,
        arg_ctr: fx.Pointer,
        arg_ctl: fx.Pointer,
        arg_tl: fx.Pointer,
        arg_rsh: fx.Pointer,
        i64_in0: fx.Int64,
        i64_in1: fx.Int64,
        i64_in2: fx.Int64,
        i64_in3: fx.Int64,
        i32_m: fx.Int32,
        i32_lda: fx.Int32,
        i32_slot: fx.Int32,
        i32_rank: fx.Int32,
        i32_maxit: fx.Int32,
        f64_eps: fx.Float64,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        i32_global = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        tlp = fx.recast_iter(i32_global, arg_tl)
        ctl = fx.recast_iter(i32_global, arg_ctl)
        ctr = fx.recast_iter(i32_global, arg_ctr)

        def stamp(slot):
            if const_expr(tl):
                if tid == 0:
                    fx.ptr_store(_now32(), tlp + (bid * TLW + slot))

        stamp(0)
        quant = (_DevMxQuant if dev_out else Gfx1250OutputMxQuant)(t, quantized_output, output_scale)
        lds_base = fx.recast_iter(fx.Uint8, fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr)
        lds_idx = fx.Int64(fx.ptrtoint(lds_base))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

        # ------------------------------------------------------------------ PA (production body)
        def zero_output(seq):
            quant.store_zero(seq, sq_row, sq_head)

        def pad_output(row):
            quant.store_pad(row, sq_row, sq_head)

        def xcd_remap(index):
            xcd, slot = index % t.xcds, index // t.xcds
            return ((slot // t.xcd_rows) * t.xcds + xcd) * t.xcd_rows + slot % t.xcd_rows

        def preload_pages(seq_id):
            entry = seq_id * max_blocks + fx.min(tid % t.WAVE, max_blocks - 1)
            src = create_llvm_ptr(buf_base_i64(table) + fx.Int64(entry) * 4, address_space=1)
            dst = create_llvm_ptr(
                fx.Int64(fx.ptrtoint(lds_base)) + fx.Int64(t.page_ids_offset + tid * 4), address_space=3
            )
            fx.rocdl.global_load_async_to_lds_b32(src, dst, 0)
            return None

        parts = t.cluster_merge
        cl = bid // parts
        part = bid % parts
        seq = xcd_remap(cl)
        pad_cluster = seq >= sequences
        plan_seq = pad_cluster.select(fx.Int32(0), seq)
        def pa_work():
            page_ids = preload_pages(plan_seq)
            context = pad_cluster.select(fx.Int32(0), fx.Int32(fx.recast_iter(fx.Int32, work)[plan_seq]))
            tiles = (context + (KV_COMPUTE_BLOCK - 1)) // KV_COMPUTE_BLOCK
            count = fx.min(tiles, fx.Int32(parts))
            first = plan_seq * parts
            n_tasks = fx.max(count, fx.Int32(1))
            task = [plan_seq, part * tiles // n_tasks, (part + 1) * tiles // n_tasks, context]
            task_slot = first + part
            active = part < count
            if active:
                ctx = PaDecodeContext(
                    t, SharedStorage, pmax, psum, pout, query, key, value, table, ks, vs, max_blocks,
                    sk_block, sk_head, sq_row, sq_head, sequences,
                    fx.Int32(task[0]), fx.Int32(task[1]), fx.Int32(task[2]), fx.Int32(task[3]),
                )
                ctx.task_slot, ctx.capacity = task_slot, capacity
                ctx.page_ids = page_ids
                ctx.lds_base = lds_base
                ctx.n_kv = kv_heads
                ctx.quant = quant
                ctx.task_count = count
                ctx.output_ptr = output
                stamp(1)
                Gfx1250Nvfp4WavePipeline(ctx).run()
                stamp(2)
            if count > 1:
                merge = Gfx1250ClusterMerge(t, pmax, psum, pout, output, sq_row, sq_head, quant)
                merge.run(seq, part, first, count, capacity)
            elif seq >= sequences:
                row = sequences + (seq - sequences) * parts + part
                if row < quant_rows:
                    pad_output(row)
            elif (count == 0) & (part == 0):
                zero_output(seq)
            fx.rocdl.s_wait_asynccnt(0)

        if const_expr(not r1):
            pa_work()

        if seq < sequences:
            if const_expr(not r1):
                # publish: this CTA's output rows drained, written back to device scope, then counted
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
                if tid == 0:
                    if const_expr(not dev_out):
                        fence_release("agent")
                    fx.atomic_add(ctl + PA_DONE, fx.Int32(1), syncscope="agent")
                stamp(3)
        else:
            # padded clusters (index % 8 != 0 with 8 sequences): the o_proj and receiver roles
            tail = (cl - cl // 8 - 1) * parts + part
            wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
            lane = tid % WAVE
            inbox_bytes = fx.Int32(BUFS * RANKS) * i32_slot * fx.Int32(16)
            ins = [i64_in0, i64_in1, i64_in2, i64_in3]
            # role_mode (diagnostic, the neighbour-slowdown bisect): 0 = the full launch; 1 = PA only (the padded
            # CTAs exit at once, like production's PA launch); 2 = PA + the 392 o_proj / receiver CTAs stay resident
            # and wait for the PA handoff (tid 0 polls, the other waves wait in the barrier), then exit; 3 = 2 +
            # the o_proj CTAs stream their weights first; 4 = 2 with 48 waiting CTAs. In 1-4 o_proj and the AR run as
            # production's launches.
            n_op = N_OPROJ if const_expr(role_mode == 0) else 0
            n_rv = N_RECV if const_expr(role_mode == 0) else 0
            if const_expr(role_mode >= 2):
                # role_mode 4: only 48 of them (6 clusters) stay resident and wait, the others exit at once
                if tail < (48 if const_expr(role_mode == 4) else N_OPROJ + N_RECV):
                    if const_expr(role_mode == 3):
                        # the o_proj CTAs' weight stream; the buffer is sized to the weight (HIDDEN x K bytes), so
                        # the receivers' rows (tail 384..391) fall past it and read 0 (11-25: an unsized buffer here
                        # read past the weight and page-faulted)
                        keep = fx.Int32(0)
                        wb_buf = ptr_buf_tensor(arg_w, fx.Int32, unit_elems=4, num_records_bytes=HIDDEN * K)
                        wave_ = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
                        for i in range_constexpr(K_STEPS // (block // WAVE)):
                            for jj in range_constexpr(4):
                                v_ = Vec(buf_copy_load(wb_buf, tail * K + (wave_ * (K_STEPS // (block // WAVE)) + i)
                                                       * 128 + jj * 32 + lane, fx.Int32, 4, cache_modifier=w_cpol))
                                keep = keep ^ fx.Int32(v_[0])
                        if (keep == i32_rank + 0x7531) & (lane == 0):  # never true; keeps the loads
                            fx.atomic_add(ctl + ERR, fx.Int32(1), syncscope="agent")
                    if tid == 0:
                        need_d = sequences * parts
                        if const_expr(gen_sync):
                            need_d = (_load_agent_i32(ctl + GEN) + 1) * need_d
                        itd = fx.Int32(0)
                        cur_d = _load_agent_i32(ctl + PA_DONE)
                        while ((cur_d - need_d) < 0) & (itd < i32_maxit):
                            rocdl.s_sleep(poll_sleep)
                            cur_d = _load_agent_i32(ctl + PA_DONE)
                            itd = itd + 1
                    _lds_barrier()
                    stamp(1)  # handoff seen (timeline builds)

            if tail < n_op:
                # ---------------------------------------------------------- o_proj, 16 columns
                n16 = tail
                lane16 = lane % 16
                kgrp = lane // 16
                b_buf = ptr_buf_tensor(arg_w, fx.Int32, unit_elems=4)
                sb_buf = ptr_buf_tensor(arg_ws, fx.Int32)
                ksteps = [wave * KPW + i for i in range_constexpr(KPW)]
                if const_expr(cpush):
                    stg_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(pout)), num_records_bytes=CPUSH_BYTES)
                if const_expr(w_delay):
                    # diagnostic: hold the weight stream back so the PA CTAs' first pages go first
                    for _ in range_constexpr(w_delay):
                        rocdl.s_sleep(32)  # ~1 us each
                # the weights of this CTA's columns and the wave's K steps, issued before the wait (w_late: after
                # it, as production's o_proj reads them; diagnostic); non-temporal (w_cpol 1, production's wcpol)
                def load_weights():
                    bfr_ = [
                        [buf_copy_load(b_buf, n16 * K + ksg * 128 + jj * 32 + lane, fx.Int32, 4, cache_modifier=w_cpol)
                         for jj in range_constexpr(4)]
                        for ksg in ksteps
                    ]
                    sb_col = n16 * 16 + lane16
                    sb_base = (sb_col // 32) * (K // 4) + sb_col % 32
                    return bfr_, [buf_copy_load(sb_buf, sb_base + ksg * 32, fx.Int32) for ksg in ksteps]

                if const_expr(not w_late):
                    bfr, sb = load_weights()
                # every PA CTA has published (cluster_poll: one poller a cluster, the cluster barrier releases)
                poller = (tid == 0) & (part == 0) if const_expr(cluster_poll) else (tid == 0)
                if const_expr(r1):
                    poller = tid < 0  # r1: the attention output is ready at entry (production PA's launch)
                if poller:
                    need = sequences * parts
                    if const_expr(gen_sync):
                        # monotonic counters: this launch's target from the launch count (wrap-safe compare)
                        need = (_load_agent_i32(ctl + GEN) + 1) * need
                    it = fx.Int32(0)
                    cur = _load_agent_i32(ctl + PA_DONE)
                    while ((cur - need) < 0) & (it < i32_maxit):
                        rocdl.s_sleep(poll_sleep)
                        cur = _load_agent_i32(ctl + PA_DONE)
                        it = it + 1
                    if (cur - need) < 0:
                        fx.atomic_add(ctl + ERR, fx.Int32(1), syncscope="agent")
                if const_expr(cluster_poll):
                    cluster.cluster_barrier()  # every CTA of an o_proj cluster takes it once
                _lds_barrier()
                stamp(1)
                if const_expr(w_late):
                    bfr, sb = load_weights()
                a_buf = ptr_buf_tensor(quantized_output, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda)
                sa_buf = ptr_buf_tensor(output_scale, fx.Int32, num_records_bytes=i32_m * (K // 32))
                lda16 = i32_lda // 16
                a_row = lane16 * lda16 + kgrp
                sa_base = lane16 * KB4
                afr = [
                    [
                        buf_copy_load(a_buf, a_row + ksg * 8 + jj * 2, fx.Int32, 4, cache_modifier=A_CPOL)
                        for jj in range_constexpr(4)
                    ]
                    for ksg in ksteps
                ]
                sa = [buf_copy_load(sa_buf, sa_base + ksg, fx.Int32, cache_modifier=A_CPOL) for ksg in ksteps]
                atom = fx.make_mma_atom(
                    fx.rocdl.WMMAScale(
                        16, 16, WMMA_K, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32, opsel_a=0, opsel_b=0
                    )
                )

                def _cat16(v):
                    v01 = v[0].shuffle(v[1], list(range(8)))
                    v23 = v[2].shuffle(v[3], list(range(8)))
                    return v01.shuffle(v23, list(range(16)))

                def _rmem(v):
                    r_ = fx.make_rmem_tensor(16, fx.Int32)
                    r_.store(v)
                    return r_

                for i in range_constexpr(KPW):
                    acc = fx.make_rmem_tensor(8, fx.Float32)
                    acc.store(fx.full(8, 0.0, fx.Float32))
                    fx.gemm(atom, acc, _rmem(_cat16(bfr[i])), _rmem(_cat16(afr[i])), acc, scale_a=sb[i], scale_b=sa[i])
                    v = Vec(acc.load())
                    off = ksteps[i] * P_PLANE + lane16 * P_ROW + kgrp * 8 * 4
                    lds_store_b128(lds_idx, off, v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32))
                    lds_store_b128(lds_idx, off + 16, v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32))
                stamp(3)  # A loaded, planes computed
                _lds_barrier()

                def plane_sum(r, c4):
                    tot = None
                    for k in range_constexpr(K_STEPS):
                        part_ = Vec(lds_load_b128(lds_idx, k * P_PLANE + r * P_ROW + c4 * 16)).bitcast(fx.Float32)
                        tot = part_ if tot is None else tot + part_
                    return tot

                rsrc = [create_buffer_resource_from_addr(fx.Int64(p), num_records_bytes=inbox_bytes) for p in ins]
                r = tid // 2
                c8 = tid % 2

                def push(row, pcol, data):
                    p = row * ROW_PACKS + pcol
                    g = row * RWAVES + pcol // SLAB
                    rnd = fx.Int32(ctr[g])
                    off = ((rnd * RANKS + i32_rank) * i32_slot + p) * 16
                    for d in range_constexpr(RANKS):
                        buffer_store(data.ir_value(), rsrc[d], off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)

                if (tid < 32) & (r < i32_m):
                    lo4 = plane_sum(r, 2 * c8)
                    hi4 = plane_sum(r, 2 * c8 + 1)
                    v8 = lo4.shuffle(hi4, list(range(8)))
                    pk = Vec(v8.to(fx.BFloat16).bitcast(fx.Int32))
                    words = [fx.Int32(pk[c]) for c in range_constexpr(4)]
                    words = [w_ ^ _bits(w_) for w_ in words]
                    data = Vec.from_elements(words, fx.Int32)
                    if const_expr(cpush):
                        # cluster-coalesced push: stage the packs in the cluster's slice of the PA partial buffer
                        # (free once every PA CTA has published; same XCD), then CTA `part` pushes row `part`'s 16
                        # contiguous packs (256 B a peer instead of 8 x 32 B pieces from 8 CTAs)
                        buffer_store(data.ir_value(), stg_rs, (((n16 // 8) * 8 + r) * 16 + part * 2 + c8) * 16,
                                     offset_is_bytes=True, cache_modifier=SCOPE_SE)
                    else:
                        push(r, n16 * 2 + c8, data)
                if const_expr(cpush):
                    rocdl.s_wait_storecnt(0)
                    cluster.cluster_barrier()
                    if (tid < 16) & (part < i32_m):
                        sdata = Vec(buffer_load(stg_rs, ((n16 // 8) * 8 + part) * 64 + tid * 4, vec_width=4, dtype=T.i32,
                                                cache_modifier=SCOPE_SE))
                        push(part, (n16 // 8) * 16 + tid, sdata)
                stamp(4)  # pushes issued (tid 0 is a pushing thread)
                if const_expr(tl):
                    rocdl.s_wait_storecnt(0)
                    _lds_barrier()
                stamp(2)
            if (tail >= N_OPROJ) & (tail < N_OPROJ + n_rv):
                # ---------------------------------------------------------- k5 receiver, one row
                r = tail - N_OPROJ
                live = r < i32_m
                rr = live.select(r, fx.Int32(0))
                res_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res)), num_records_bytes=i32_m * HIDDEN * 2)
                nw_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_nw)), num_records_bytes=HIDDEN * 2)
                out_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_out)), num_records_bytes=i32_m * HIDDEN * 2)
                ro_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res_out)), num_records_bytes=i32_m * HIDDEN * 2)
                my_base = fx.Int64(i64_in0)
                for d in range_constexpr(1, RANKS):
                    my_base = (i32_rank == d).select(fx.Int64(ins[d]), my_base)
                my_rs = create_buffer_resource_from_addr(my_base, num_records_bytes=inbox_bytes)
                sent = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)

                def my_store(off, data):
                    buffer_store(data.ir_value(), my_rs, off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)

                def my_load(off):
                    return Vec(buffer_load(my_rs, off // 4, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_SYS))

                slabs = [wave + NW * h for h in range_constexpr(SPW)]
                g_ = [rr * RWAVES + sl for sl in slabs]
                srnd = [fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[g]))) for g in g_]
                rrnd = [fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[ctr_recv + g]))) for g in g_]
                p0 = [rr * ROW_PACKS + sl * SLAB + lane for sl in slabs]
                rs = [[Vec(buffer_load(res_rs, (p0[h] + 32 * j) * 4, vec_width=4, dtype=T.i32)) for j in range_constexpr(3)]
                      for h in range_constexpr(SPW)]
                wt = [
                    [Vec(buffer_load(nw_rs, (slabs[h] * SLAB + 32 * j + lane) * 4, vec_width=4, dtype=T.i32))
                     for j in range_constexpr(3)]
                    for h in range_constexpr(SPW)
                ]
                if live:
                    for h in range_constexpr(SPW):
                        rearm = (rrnd[h] + 2) % BUFS
                        for i in range_constexpr(1, RANKS):
                            s_ = (i32_rank + i) & (RANKS - 1)
                            for j in range_constexpr(3):
                                my_store(((rearm * RANKS + s_) * i32_slot + p0[h] + 32 * j) * 16, sent)

                if const_expr(recv_late):
                    # the peers' packs cannot come before this rank's PA has published (their o_proj waits on
                    # their own PA, which runs in step): one poller a receiver CTA waits for it, the other waves
                    # sit in the barrier, and only then do all waves poll the inbox (fewer spinning waves)
                    if tid == 0:
                        need_r = sequences * parts
                        if const_expr(gen_sync):
                            need_r = (_load_agent_i32(ctl + GEN) + 1) * need_r
                        itr = fx.Int32(0)
                        cur_r = _load_agent_i32(ctl + PA_DONE)
                        while ((cur_r - need_r) < 0) & (itr < i32_maxit):
                            rocdl.s_sleep(poll_sleep)
                            cur_r = _load_agent_i32(ctl + PA_DONE)
                            itr = itr + 1
                    _lds_barrier()

                def pending(vs):
                    m_ = fx.Int32(0)
                    for v in vs:
                        for c in range_constexpr(4):
                            m_ = m_ | _bits(fx.Int32(v[c]))
                    return fx.Int32(rocdl.ballot(T.i32, m_ != 0)) != 0

                stamp(1)
                s = []
                for h in range_constexpr(SPW):

                    def offs(s_, j, h=h):
                        return ((rrnd[h] * RANKS + s_) * i32_slot + p0[h] + 32 * j) * 16

                    def load_all(offs=offs):
                        _mem_clobber()
                        return tuple(my_load(offs(s_, j)) for s_ in range_constexpr(RANKS) for j in range_constexpr(3))

                    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
                    pend = live & pending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
                    it = fx.Int32(0)
                    while pend & (it < i32_maxit):
                        rocdl.s_sleep(1)
                        (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
                        pend = pending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
                        if const_expr(tl and h == 0):
                            # per source q: the last time its slab-0 packs were still incomplete (wave 0 stamps;
                            # no loop-carried state: the clock read is behind the loads' use, never between a load
                            # and its wait). A source complete at the first look leaves its slot 0.
                            fl = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
                            for q_ in range_constexpr(RANKS):
                                if pending(fl[3 * q_:3 * q_ + 3]) & (tid == 0):
                                    fx.ptr_store(_now32(), tlp + (bid * TLW + 4 + q_))
                        it = it + 1
                    if live & pend & (lane == 0):
                        fx.atomic_add(ctr + ctr_err, fx.Int32(1), syncscope="agent")
                    flat = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
                    sh = [[None] * 8 for _ in range(3)]
                    for j in range_constexpr(3):
                        for k in range_constexpr(8):
                            acc_ = None
                            for q in range_constexpr(RANKS):
                                wq = fx.Int32(flat[q * 3 + j][k // 2])
                                x_ = _bf_hi(wq) if k % 2 else _bf_lo(wq)
                                acc_ = x_ if acc_ is None else acc_ + x_
                            sh[j][k] = acc_
                    s.append(sh)
                stamp(2)
                red = fx.recast_iter(
                    fx.PointerType.get(elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4),
                    lds_base,
                ) + ((K_STEPS * P_PLANE) // 4)
                for h in range_constexpr(SPW):
                    sq = None
                    for j in range_constexpr(3):
                        o = []
                        for c in range_constexpr(4):
                            ab = _f2bf2(s[h][j][2 * c], s[h][j][2 * c + 1])
                            rw = fx.Int32(rs[h][j][c])
                            s[h][j][2 * c] = _bf_lo(ab) + _bf_lo(rw)
                            s[h][j][2 * c + 1] = _bf_hi(ab) + _bf_hi(rw)
                        for k in range_constexpr(8):
                            e = s[h][j][k] * s[h][j][k]
                            sq = e if sq is None else sq + e
                        for c in range_constexpr(4):
                            o.append(_f2bf2(s[h][j][2 * c], s[h][j][2 * c + 1]))
                        if live:
                            buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), ro_rs, (p0[h] + 32 * j) * 16,
                                         offset_is_bytes=True)
                    for d_ in range_constexpr(5):
                        sq = sq + sq.shuffle_xor(fx.Int32(1 << d_), fx.Int32(WAVE))
                    if lane == 0:
                        fx.ptr_store(sq, red + slabs[h])
                _lds_barrier()
                pr = [fx.Float32(red[i]) for i in range_constexpr(RWAVES)]
                total = ((pr[0] + pr[1]) + (pr[2] + pr[3])) + ((pr[4] + pr[5]) + (pr[6] + pr[7]))
                mean = total * _rcp(fx.Float32(float(HIDDEN)))
                rstd = _rsq(fx.Float32(fx.Float64(mean) + f64_eps))
                for h in range_constexpr(SPW):
                    for j in range_constexpr(3):
                        o = []
                        for c in range_constexpr(4):
                            wv = fx.Int32(wt[h][j][c])
                            y0 = (s[h][j][2 * c] * rstd) * (_bf_lo(wv) + fx.Float32(1.0))
                            y1 = (s[h][j][2 * c + 1] * rstd) * (_bf_hi(wv) + fx.Float32(1.0))
                            o.append(_f2bf2(y0, y1))
                        if live:
                            buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), out_rs, (p0[h] + 32 * j) * 16,
                                         offset_is_bytes=True, cache_modifier=SCOPE_DEV if r1 else 0)
                if const_expr(r1):
                    # publish x (the router phase's input): this CTA's out row at device scope, drained, one count
                    rocdl.s_wait_storecnt(0)
                    _lds_barrier()
                    if tid == 0:
                        fx.atomic_add(ctl + X_READY, fx.Int32(1), syncscope="agent")
                if live:
                    for h in range_constexpr(SPW):
                        for j in range_constexpr(3):
                            my_store(((rrnd[h] * RANKS + i32_rank) * i32_slot + p0[h] + 32 * j) * 16, sent)
                        if lane == 0:
                            fx.ptr_store((srnd[h] + 1) % BUFS, ctr + g_[h])
                            fx.ptr_store((rrnd[h] + 1) % BUFS, ctr + (ctr_recv + g_[h]))
                if const_expr(tl):
                    rocdl.s_wait_storecnt(0)
                    _lds_barrier()
                stamp(3)

        if const_expr(r1_router):
            # r1: RSH's router phase (production moe_router_topk's split-K 8 math, weights issued before the x wait)
            # on the first idle padded cluster (tail 392..399 = one launch cluster, rank = part)
            r_tail = (cl - cl // 8 - 1) * parts + part - (N_OPROJ + N_RECV)
            if (seq >= sequences) & (r_tail >= 0) & (r_tail < 8):
                ctl_i64 = fx.Int64(fx.ptrtoint(arg_ctl))
                x_target = (_load_agent_i32(ctl + GEN) + 1) * 8
                rsh_router.run(part, lds_base, arg_rsh, ctl_i64 + 4 * X_READY, x_target, ctl_i64 + 4 * ERR, i32_maxit,
                               arg_out)

        # ------------------------------------------------------------------ exit: the last CTA resets
        # exit_cl: one count a cluster (64 device-scope atomics a launch instead of 512); clusters are role-uniform
        # (PA / o_proj / receivers / idle), so the cluster barrier only waits for CTAs of the same phase
        n_exit = GRID // parts if const_expr(exit_cl) else GRID
        if const_expr(exit_cl):
            if tid == 0:
                rocdl.s_wait_storecnt(0)  # this CTA's PA_DONE / ERR adds land before its cluster counts out
            cluster.cluster_barrier()
        counter = ((tid == 0) & (part == 0)) if const_expr(exit_cl) else (tid == 0)
        if counter:
            # this CTA's earlier non-returning counts (PA_DONE, ERR) are performed before it counts out: the
            # last CTA's reset must not race a late PA_DONE add
            rocdl.s_wait_storecnt(0)
            old = fx.Int32(fx.atomic_add(ctl + EXIT, fx.Int32(1), syncscope="agent"))
            if const_expr(gen_sync):
                # no resets: the last CTA of a launch counts the launch (device-scope atomic)
                if ((old + 1) & (n_exit - 1)) == 0:
                    fx.atomic_add(ctl + GEN, fx.Int32(1), syncscope="agent")
            else:
                if old == n_exit - 1:
                    fx.ptr_store(fx.Int32(0), ctl + PA_DONE)
                    fx.ptr_store(fx.Int32(0), ctl + EXIT)

    @flyc.jit
    def launch(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        table: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        max_blocks: fx.Int32,
        sequences: fx.Int32,
        kv_heads: fx.Int32,
        sk_block: fx.Int32,
        sk_head: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        work: fx.Pointer,
        capacity: fx.Int32,
        output: fx.Pointer,
        reduce_info: fx.Pointer,
        counters: fx.Pointer,
        quantized_output: fx.Pointer,
        output_scale: fx.Pointer,
        quant_rows: fx.Int32,
        arg_w: fx.Pointer,
        arg_ws: fx.Pointer,
        arg_res: fx.Pointer,
        arg_nw: fx.Pointer,
        arg_out: fx.Pointer,
        arg_res_out: fx.Pointer,
        arg_ctr: fx.Pointer,
        arg_ctl: fx.Pointer,
        arg_tl: fx.Pointer,
        arg_rsh: fx.Pointer,
        i64_in0: fx.Int64,
        i64_in1: fx.Int64,
        i64_in2: fx.Int64,
        i64_in3: fx.Int64,
        i32_m: fx.Int32,
        i32_lda: fx.Int32,
        i32_slot: fx.Int32,
        i32_rank: fx.Int32,
        i32_maxit: fx.Int32,
        f64_eps: fx.Float64,
        stream: fx.Stream,
    ):
        pa_oproj_ar_kernel(
            work, table, query, key, value, pmax, psum, pout, ks, vs, max_blocks, sk_block, sk_head, sq_row,
            sq_head, sequences, capacity, kv_heads, reduce_info, output, counters, quantized_output, output_scale,
            quant_rows, arg_w, arg_ws, arg_res, arg_nw, arg_out, arg_res_out, arg_ctr, arg_ctl, arg_tl, arg_rsh,
            i64_in0, i64_in1, i64_in2, i64_in3, i32_m, i32_lda, i32_slot, i32_rank, i32_maxit, f64_eps,
            value_attrs={"rocdl.cluster_dims": f"{t.cluster_merge},1,1"},
        ).launch(
            grid=(GRID, kv_heads, 1),
            block=(block, 1, 1),
            stream=stream,
            cluster=(t.cluster_merge, 1, 1),
        )

    launch.compile_hints["llvm_options"] = {
        "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
        "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    }
    return _CACHE.setdefault(key, launch)
