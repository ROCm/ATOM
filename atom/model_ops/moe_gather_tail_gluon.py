# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 Gluon form of the fused gather-reduce + MoE tail RMSNorm for
prefill-size token counts (ATOM_MOE_GATHER_TAIL_NORM_PIPE): a program a run of
tokens, software-pipelined so that no token waits on a dependent DRAM load.

`moe_tail_norm._moe_gather_tail_rmsnorm_kernel` is one program a token: it
waits on its route-map load before it can issue its MoE2 row loads, so the GPU
moves data only as fast as resident programs cover two back-to-back DRAM
latencies. Here a program owns a contiguous run of output rows (the grid two
programs a CU) and:

* reads the run's whole route map (TOPK row ids + weights a token), its gate
  logits and (for `rows=`) its source rows once, up front, into LDS -- every
  index load of the run is in flight together, and in the loop an index is an
  LDS read, never a DRAM wait;
* moves a token's TOPK MoE2 rows global -> LDS with one TDM gather (its row
  ids read from that LDS table) and its shared-expert and residual rows with
  two TDM loads, into an NSTAGE-deep ring: while token i is reduced and
  normalized, the loads of tokens i + 1 .. i + NSTAGE - 1 are in flight;
* stores the normed row and the new residual from registers.

TDM, not buffer loads, for the 8 x 4 KB rows: a buffer-loaded stage of 8 rows
holds 64 VGPRs a thread until consumed, and a loop that issues the next
token's index loads compiles (Triton, gfx1250) to vector loads that are
`v_readfirstlane`d -- waited on -- right where they are issued, behind every
load in flight on the same in-order counter: the Triton version of this
pipeline measured slower than the one-token kernel. TDM rows sit in LDS on
their own counter (tensorcnt) with explicit waits. Shared / residual rows by
buffer loads into registers instead of TDM measured slower too.

Every token issues the same three TDM ops (predicated off past the run), so
the wait count is static; Triton's async-wait pass would otherwise lower a
conditional issue's wait to 0. The LDS barriers (slot reuse, each warp's share
of a TDM op) are the local-only ones Triton's membar pass inserts -- an
explicit `gl.barrier()` also waits on the previous token's global stores.

The per-token math is the Triton kernel's line for line, in the same row
layout (8 contiguous elements a thread, 32 lanes, 4 warps at 2048 hidden), so
the fp32 sum of squares reduces in the same order: bitwise
moe_gather_reduce + `_moe_tail_rmsnorm_kernel` (padding tokens, dropped
routes and `rows=` included).
"""

from __future__ import annotations

import functools

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _smem_scalar(smem, j, L1: gl.constexpr):
    """Element (j, 0) of a [R, 1] LDS table as a scalar (a one-element reduce
    returns the element itself)."""
    return gl.sum(smem.index(j).load(L1), axis=0)


@gluon.jit
def _issue(
    j,
    n,
    start,
    g_desc,
    sh_desc,
    rs_desc,
    r_tab,
    s_tab,
    rows_ring,
    sh_ring,
    rs_ring,
    IDX_L: gl.constexpr,
    L1: gl.constexpr,
    NSTAGE: gl.constexpr,
    GATHER: gl.constexpr,
):
    """TDM loads of local token j into ring slot j % NSTAGE: its TOPK MoE2 rows
    (one gather), its shared-expert row and its residual row. Past the run
    (j >= n) the same three instructions go out predicated off, so every
    token's wait counts the same number of TDM ops."""
    slot = j % NSTAGE
    p = j < n
    jj = gl.minimum(j, n - 1)
    idx = r_tab.index(jj).load(IDX_L)
    g_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(g_desc, pred=p)
    gl.amd.gfx1250.tdm.async_gather(g_desc, idx, rows_ring.index(slot))
    if GATHER:
        src = _smem_scalar(s_tab, jj, L1)
    else:
        src = start + jj
    gl.amd.gfx1250.tdm.async_load(sh_desc, [src, 0], sh_ring.index(slot), pred=p)
    gl.amd.gfx1250.tdm.async_load(rs_desc, [src, 0], rs_ring.index(slot), pred=p)


@gluon.jit
def moe_gather_tail_rmsnorm_tdm_kernel(
    grouped_ptr,
    route_rows_ptr,
    route_w_ptr,
    nvt_ptr,
    shared_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    rows_ptr,
    out_ptr,
    res_out_ptr,
    n_grouped,
    n_tokens,
    n_out,
    stride_grouped,
    stride_shared,
    stride_gate,
    stride_residual,
    stride_out,
    stride_res_out,
    eps,
    N: gl.constexpr,
    BLOCK: gl.constexpr,
    TOPK: gl.constexpr,
    RMAX: gl.constexpr,
    NSTAGE: gl.constexpr,
    HAS_GATE: gl.constexpr,
    GATHER: gl.constexpr,
    WRITE_RESIDUAL: gl.constexpr,
    HAS_NVT: gl.constexpr,
):
    gl.static_assert(TOPK == 8, "one int32 TDM gather carries 8 rows")
    gl.static_assert(BLOCK == N, "rows are whole powers of two")
    NW: gl.constexpr = gl.num_warps()
    # The Triton tail kernel's row layout: 8 contiguous elements a thread.
    ROW_L: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, NW], [1, 0])
    COL_L: gl.constexpr = gl.SliceLayout(0, ROW_L)
    TAB_L: gl.constexpr = gl.BlockedLayout([1, TOPK], [32, 1], [NW, 1], [1, 0])
    RUN_L: gl.constexpr = gl.SliceLayout(1, TAB_L)
    IDX_L: gl.constexpr = gl.SliceLayout(
        1, gl.BlockedLayout([TOPK, 1], [1, 32], [1, NW], [1, 0])
    )
    L1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    SH1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    pid = gl.program_id(0)
    nprog = gl.num_programs(0)
    start = (pid * n_out) // nprog
    n = ((pid + 1) * n_out) // nprog - start

    # The run's route map, weights, gate logits and source rows, up front.
    rr = gl.arange(0, RMAX, layout=RUN_L)
    kk = gl.arange(0, TOPK, layout=gl.SliceLayout(0, TAB_L))
    in_run = rr < n
    if GATHER:
        src_v = gl.load(rows_ptr + start + rr, mask=in_run, other=0).to(gl.int32)
    else:
        src_v = start + rr
    if HAS_NVT:
        live = in_run & (src_v < gl.load(nvt_ptr))
    else:
        live = in_run
    roff = src_v[:, None] * TOPK + kk[None, :]
    lmask = live[:, None] & (kk[None, :] < TOPK)
    r = gl.load(route_rows_ptr + roff, mask=lmask, other=-1)
    w = gl.load(route_w_ptr + roff, mask=lmask, other=0.0).to(gl.float32)
    # Dropped routes and padding tokens gather a row past the end: TDM fills
    # it with zeros, as the reduce's masked load does.
    r = gl.where(r >= 0, r, n_grouped)
    # Written as 2-D tiles, read back a token's row at a time (the same
    # unswizzled row-major bytes under a per-row view).
    r_tab = gl.allocate_shared_memory(gl.int32, [RMAX, TOPK], SH, r).reinterpret(
        layout=SH1
    )
    w_tab = gl.allocate_shared_memory(gl.float32, [RMAX, TOPK], SH, w).reinterpret(
        layout=SH1
    )
    if HAS_GATE:
        g = gl.load(gate_ptr + src_v.to(gl.int64) * stride_gate, mask=in_run, other=0.0)
        g_tab = gl.allocate_shared_memory(
            gate_ptr.dtype.element_ty, [RMAX, 1], SH, gl.expand_dims(g, 1)
        ).reinterpret(layout=SH1)
    s_tab = gl.allocate_shared_memory(
        gl.int32, [RMAX, 1], SH, gl.expand_dims(src_v, 1)
    ).reinterpret(layout=SH1)

    g_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        grouped_ptr, [n_grouped, N], [stride_grouped, 1], [TOPK, BLOCK], SH
    )
    sh_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        shared_ptr, [n_tokens, N], [stride_shared, 1], [1, BLOCK], SH
    )
    rs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        residual_ptr, [n_tokens, N], [stride_residual, 1], [1, BLOCK], SH
    )
    rows_ring = gl.allocate_shared_memory(
        grouped_ptr.dtype.element_ty, [NSTAGE, TOPK, BLOCK], SH
    )
    sh_ring = gl.allocate_shared_memory(
        shared_ptr.dtype.element_ty, [NSTAGE, 1, BLOCK], SH
    )
    rs_ring = gl.allocate_shared_memory(
        residual_ptr.dtype.element_ty, [NSTAGE, 1, BLOCK], SH
    )

    cols = gl.arange(0, BLOCK, layout=COL_L)
    colmask = cols[None, :] < N
    weight = gl.load(weight_ptr + cols[None, :]).to(gl.float32) + 1.0

    for s in gl.static_range(NSTAGE - 1):
        _issue(
            s,
            n,
            start,
            g_desc,
            sh_desc,
            rs_desc,
            r_tab,
            s_tab,
            rows_ring,
            sh_ring,
            rs_ring,
            IDX_L,
            L1,
            NSTAGE,
            GATHER,
        )

    for i in range(n):
        _issue(
            i + NSTAGE - 1,
            n,
            start,
            g_desc,
            sh_desc,
            rs_desc,
            r_tab,
            s_tab,
            rows_ring,
            sh_ring,
            rs_ring,
            IDX_L,
            L1,
            NSTAGE,
            GATHER,
        )
        # Token i's three TDM ops landed; those of the NSTAGE - 1 tokens after
        # it stay in flight. (The LDS barriers -- slot reuse, every warp's share
        # of a TDM op -- are the ones Triton's membar pass inserts.)
        gl.amd.gfx1250.tdm.async_wait(3 * (NSTAGE - 1))

        slot = i % NSTAGE
        rows = rows_ring.index(slot)
        wi = w_tab.index(i)
        acc = gl.zeros([1, BLOCK], gl.float32, ROW_L)
        for k in gl.static_range(TOPK):
            # moe_gather_reduce's step: a product, then a sum, each rounded.
            wk = gl.sum(wi.slice(k, 1).load(L1), axis=0)
            acc = acc + wk * rows.slice(k, 1).load(ROW_L).to(gl.float32)
        routed = acc.to(out_ptr.dtype.element_ty).to(gl.float32)
        shared = sh_ring.index(slot).load(ROW_L).to(gl.float32)
        if HAS_GATE:
            gate = _smem_scalar(g_tab, i, L1)
            # gl.sum returns the bf16 element as fp32 (exact); round the
            # sigmoid to the logit dtype as sigmoid_mul does.
            gdt: gl.constexpr = gate_ptr.dtype.element_ty
            gs = (1 / (1 + gl.exp(-gate.to(gl.float32)))).to(gdt).to(gl.float32)
            shared = (shared * gs).to(shared_ptr.dtype.element_ty).to(gl.float32)
        moe = (shared + routed).to(out_ptr.dtype.element_ty).to(gl.float32)
        x = moe + rs_ring.index(slot).load(ROW_L).to(gl.float32)
        row = (start + i).to(gl.int64)
        if WRITE_RESIDUAL:
            gl.store(
                res_out_ptr + row * stride_res_out + cols[None, :],
                x.to(res_out_ptr.dtype.element_ty),
                colmask,
            )
        rstd = gl.rsqrt(gl.sum(x * x, axis=1) / N + eps)
        normed = ((x * gl.expand_dims(rstd, 1)) * weight).to(out_ptr.dtype.element_ty)
        gl.store(out_ptr + row * stride_out + cols[None, :], normed, colmask)


# Below this many output rows the one-token kernel wins: up to ~2k rows its
# programs all fit in one wave, and at 4k it is level (gfx1250, measured).
PIPE_MIN_ROWS = 8192
# Two programs a CU, a ring of two tokens each (40 KB a stage): measured best
# from 8k to 20k rows against 1-8 programs a CU and 2-4 stages.
_PROGRAMS_PER_CU = 2
_NSTAGE = 2
_INT32_MAX = 2**31 - 1


@functools.cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def gather_tail_tdm_takes(g, shared, residual, n_out: int) -> bool:
    """Whether the TDM kernel runs these inputs (on top of
    `can_fuse_gather_tail`): gfx1250, a prefill-size row count, 2048-wide
    bf16 / f16 rows, 8 routes, and the 16-byte alignment TDM descriptors take
    (bases and row strides)."""
    if n_out < PIPE_MIN_ROWS or g.route_rows.shape[1] != 8:
        return False
    hidden = shared.shape[1]
    if hidden != 2048 or not _is_gfx1250():
        return False
    es = shared.element_size()
    for t in (g.grouped, shared, residual):
        if t.data_ptr() % 16 or (t.stride(0) * es) % 16:
            return False
    return True


@functools.cache
def _is_gfx1250() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx
    except ImportError:
        return False
    return get_gfx() == "gfx1250"


# Compiled kernels by launch key: eager prefill is host-bound, and Triton's own
# dispatch is most of a launch's host time (as `moe_tail_norm._launch`).
_COMPILED: dict[tuple, object] = {}


def launch_gather_tail_tdm(
    g, shared, gate, residual, weight, eps, rows, out, res_out, grid=None, nstage=None
):
    """`out` (and `res_out` when given) for `GatheredRouted` g: the TDM
    kernel over `grid` programs (default two a CU)."""
    n_out, hidden = out.shape
    if grid is None:
        grid = _PROGRAMS_PER_CU * _num_cus(out.device.index or 0)
    grid = max(1, min(grid, n_out))
    nstage = _NSTAGE if nstage is None else nstage
    rmax = max(16, triton.next_power_of_2(triton.cdiv(n_out, grid)))
    nvt = g.num_valid_tokens
    ptrs = (
        g.grouped,
        g.route_rows,
        g.weights,
        nvt if nvt is not None else g.route_rows,
        shared,
        gate if gate is not None else shared,
        residual,
        weight,
        rows if rows is not None else g.route_rows,
        out,
        res_out if res_out is not None else out,
    )
    ints = (
        g.grouped.shape[0],
        shared.shape[0],
        n_out,
        g.grouped.stride(0),
        shared.stride(0),
        gate.stride(0) if gate is not None else 0,
        residual.stride(0),
        out.stride(0),
        res_out.stride(0) if res_out is not None else 0,
    )
    # Cached only where Triton would not specialize an int away (== 1) and
    # every pointer is 16-byte aligned and reaches no further than 2 GiB.
    reach = (max(ints[0] * ints[3], max(ints[1], n_out) * max(ints[4:])) + hidden) * 2
    cacheable = (
        reach <= _INT32_MAX
        and 1 not in ints
        and not any(t.data_ptr() % 16 for t in ptrs)
    )
    key = (
        hidden,
        shared.dtype,
        g.weights.dtype,
        None if gate is None else gate.dtype,
        None if rows is None else rows.dtype,
        nvt is not None,
        res_out is not None,
        rmax,
        nstage,
        tuple(v % 16 == 0 for v in ints),
    )
    kernel = _COMPILED.get(key) if cacheable else None
    if kernel is not None:
        kernel[(grid, 1, 1)](*ptrs, *ints, eps)
        return
    kernel = moe_gather_tail_rmsnorm_tdm_kernel[(grid,)](
        *ptrs,
        *ints,
        eps,
        N=hidden,
        BLOCK=hidden,
        TOPK=g.route_rows.shape[1],
        RMAX=rmax,
        NSTAGE=nstage,
        HAS_GATE=gate is not None,
        GATHER=rows is not None,
        WRITE_RESIDUAL=res_out is not None,
        HAS_NVT=nvt is not None,
        num_warps=4,
        # Products and sums round where the unfused kernels round.
        enable_fp_fusion=False,
    )
    if cacheable:
        _COMPILED[key] = kernel
