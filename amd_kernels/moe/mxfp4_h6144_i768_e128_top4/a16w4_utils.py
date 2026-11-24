# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Helpers of the MXFP4 MoE GEMMs: raw pointers, the agent-scope ready counters that order a fused GEMM1 -> GEMM2
launch, and the inline-sort routing table."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .common.utils import _global_i32_at, _global_i32_ptr, _raw

LOG2E = 1.4426950408889634


# cache policy bit SC1 (device scope: write-through stores) and the bytes between two m-blocks' counters (ready
# count, then consumer count): each block in its own 256 B line, so the polls of waiting tiles spread over channels
SC1 = 16
READY_STRIDE = 256


def _global_atomic_agent(op, addr_i64, value):
    ptr = _global_i32_ptr(addr_i64)
    return fx.Int32(
        llvm.AtomicRMWOp(
            op,
            ptr.llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def global_atomic_add_agent(addr_i64, value):
    """``old = *addr; *addr += value`` on a global i32, device (agent) scope, relaxed."""
    return _global_atomic_agent(llvm.AtomicBinOp.add, addr_i64, value)


@flyc.jit
def wait_ready(addr, need, consumers):
    """A fused launch's GEMM2 tile: thread 0 spins until the m-block's ready count at ``addr`` reaches ``need`` (every
    GEMM1 tile of the block stored its rows), then counts itself at ``addr + 4``; the block's last consumer resets
    both words to 0 for the next call (every consumer has passed its wait by then, and no GEMM1 tile of the block is
    left). The workgroup waits at a barrier. No acquire fence: the rows were stored write-through and no load of this
    launch touched their lines before the wait; an agent acquire (``buffer_inv sc1``) per tile cost +13 us at 8 tokens
    and +76 us at 32 (it invalidates the XCD's L2 each time)."""
    if fx.thread_idx.x == 0:
        got = global_atomic_add_agent(addr, 0)
        while got < need:
            rocdl.s_sleep(2)
            got = global_atomic_add_agent(addr, 0)
        old = global_atomic_add_agent(addr + fx.Int64(4), 1)
        if old == consumers - 1:
            _global_atomic_agent(llvm.AtomicBinOp.xchg, addr, 0)
            _global_atomic_agent(llvm.AtomicBinOp.xchg, addr + fx.Int64(4), 0)
    gpu.barrier()


def poll_ready(addr, cpol=SC1):
    """Thread 0's first read of a ready counter (device scope; the other lanes' out-of-range offset drops theirs),
    issued early so its latency overlaps the work item's setup; ``wait_ready_from`` consumes the value."""
    rsrc = buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=8)
    off = (fx.thread_idx.x == 0).select(fx.Int32(0), fx.Int32(0x7FFFFFF0))
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.raw.ptr.buffer.atomic.add.i32",
            [_raw(fx.Int32(0)), _raw(rsrc), _raw(off), _raw(fx.Int32(0)), _raw(fx.Int32(cpol))],
            [],
            [],
        )
    )


@flyc.jit
def wait_ready_from(addr, need, consumers, got):
    """``wait_ready`` whose first poll ``got`` (``poll_ready``) was issued earlier."""
    if fx.thread_idx.x == 0:
        while got < need:
            rocdl.s_sleep(2)
            got = global_atomic_add_agent(addr, 0)
        old = global_atomic_add_agent(addr + fx.Int64(4), 1)
        if old == consumers - 1:
            _global_atomic_agent(llvm.AtomicBinOp.xchg, addr, 0)
            _global_atomic_agent(llvm.AtomicBinOp.xchg, addr + fx.Int64(4), 0)
    gpu.barrier()


def inline_sort_max_pairs(n_tokens, topk, bm):
    """Routing pairs the inline-sort table scans (64 per wave pass): 64 when n_tokens * topk <= 64, else bm * topk
    while n_tokens <= bm, else every pair rounded up to whole passes; a trace-time int (part of the kernel name)."""
    n, k = int(n_tokens), int(topk)
    if n * k <= 64:
        return 64
    if n <= int(bm):
        return int(bm) * k
    return (n * k + 63) // 64 * 64


def inline_sort_multi_block(n_tokens, bm):
    """More tokens than block rows: an expert's rows may span several m-blocks."""
    return int(n_tokens) > int(bm)


def inline_sort_table(arg_topk, i32_ntok, TOPK, p_i32, lane, tab, max_pairs=64, bm=16,
                      multi_block=False):
    """Inline sort: each block builds its own sorted-ids table from ``topk_ids`` with a wave ballot, no sort kernel.

    Routing pair q = token * TOPK + slot (row-major topk_ids). Block p owns expert e = topk_ids[p] iff p is the first
    pair with that expert; its rows are all pairs with expert e, in pair order (n_tokens <= bm: one m-block per
    expert). ``tab`` (i32 LDS, ``bm + 1`` entries) holds token | slot << 24 per row, token = n_tokens for padding rows
    (the sort's rows for this block); non-matching pairs are parked in slot ``bm``. Returns (expert id, owner, row
    count, build_table); non-owner blocks (duplicate experts) must exit. Pairs are scanned 64 per wave pass;
    ``max_pairs`` (``inline_sort_max_pairs``) sets the number of passes.

    ``multi_block`` (n_tokens > bm, trace-time): an expert can have more than ``bm`` rows. Block p owns iff the rank r
    of pair p among its expert's pairs is a multiple of ``bm``, and its rows are that expert's ranks r .. r + bm - 1.
    """
    n_chunks = (int(max_pairs) + 63) // 64
    n_pairs = i32_ntok * fx.Int32(TOPK)
    last = n_pairs - fx.Int32(1)
    p_lane = p_i32 % fx.Int32(64)
    p_chunk = p_i32 // fx.Int32(64)
    qs, pvs = [], []
    for c in range_constexpr(n_chunks):
        q = lane + fx.Int32(c * 64)
        idx = fx.Int32(arith.minsi(_raw(q), _raw(last)))
        v = fx.Int32(_global_i32_at(arg_topk, idx))
        qs.append(q)
        pvs.append((q < n_pairs).select(v, fx.Int32(-1)))
    # my expert = value of pair p (uniform)
    e = fx.Int32(rocdl.readlane(T.i32, _raw(pvs[0]), _raw(p_lane)))
    for c in range_constexpr(1, n_chunks):
        e_c = fx.Int32(rocdl.readlane(T.i32, _raw(pvs[c]), _raw(p_lane)))
        e = (p_chunk == fx.Int32(c)).select(e_c, e)
    base = fx.Int32(0)
    rank_p = fx.Int32(0)
    slots, fuseds, ranks, matches = [], [], [], []
    for c in range_constexpr(n_chunks):
        is_match = pvs[c] == e
        mask = rocdl.ballot(T.i64, _raw(is_match))
        mask_lo = arith.trunci(T.i32, mask)
        mask_hi = arith.trunci(T.i32, arith.shrui(mask, arith.constant(32, type=T.i64)))
        below = fx.Int32(
            rocdl.mbcnt_hi(
                T.i32, mask_hi, rocdl.mbcnt_lo(T.i32, mask_lo, _raw(fx.Int32(0)))
            )
        )
        rank = base + below  # row of pair q among this expert's pairs (pair order)
        fuseds.append(
            (qs[c] // fx.Int32(TOPK)) | ((qs[c] % fx.Int32(TOPK)) << fx.Int32(24))
        )
        if multi_block:
            ranks.append(rank)
            matches.append(is_match)
        else:
            slots.append(is_match.select(rank, fx.Int32(bm)))  # non-matching -> slot bm
        r_p = fx.Int32(rocdl.readlane(T.i32, _raw(rank), _raw(p_lane)))
        rank_p = (p_chunk == fx.Int32(c)).select(r_p, rank_p)
        # matches so far = rank at lane 63 + lane 63's own match bit
        tot = rank + is_match.select(fx.Int32(1), fx.Int32(0))
        base = fx.Int32(rocdl.readlane(T.i32, _raw(tot), _raw(fx.Int32(63))))
    if multi_block:
        # block p: ranks rank_p .. rank_p+bm-1 of its expert; other pairs -> slot bm
        owner = (rank_p % fx.Int32(bm)) == fx.Int32(0)
        for c in range_constexpr(n_chunks):
            rel = ranks[c] - rank_p
            mine = matches[c] & (rel >= fx.Int32(0)) & (rel < fx.Int32(bm))
            slots.append(mine.select(rel, fx.Int32(bm)))
    else:
        owner = rank_p == fx.Int32(0)

    def build_table():
        # padding sentinel in every row slot first (0..31 covers bm <= 32), then the rows
        tab[lane % 32] = i32_ntok
        gpu.barrier()
        for c in range_constexpr(n_chunks):
            tab[slots[c]] = fuseds[c]
        gpu.barrier()

    # the kernel calls build_table() under `if owner:` (a uniform scf.if), so duplicate-expert blocks leave after the
    # loads and ballots without the two barriers
    return e, owner, base, build_table
