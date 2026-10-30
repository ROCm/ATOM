"""Each G1 CTA resolves its expert and token membership without a sorted table."""

import hashlib
from functools import cache
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.mxfp4_gemm1 import _bm_constants, default_k_stages
from flydsl.expr import rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

from atom.models.qwen4_exp_mono.owned.native_g1 import _gemm1_body
from atom.mono.device.ops import ballot, kernel_symbol, launcher, popcount, rsrc
from atom.mono.runtime.abi import KernelAbi

SLOTS = 41
GRID = SLOTS * 5
K_STAGES = default_k_stages(16, 256, 128, 10, 1280)
LDS_BYTES = _bm_constants(16, 256, 128, 10, k_stages=K_STAGES)[3]
SOURCE = hashlib.sha256(
    Path(__file__).read_bytes() + Path(__file__).with_name("native_g1.py").read_bytes()
).hexdigest()
ABI = KernelAbi(
    (
        "mixed",
        "ids",
        "weights",
        "sorted_ids",
        "sorted_weights",
        "experts",
        "valid",
        "m_indices",
        "reverse",
        "out",
        "w1",
        "s1",
        "aq",
        "ascale",
        "slot_map",
    )
)
_STREAM = fx.Stream(None)


@fx.struct
class Smem:
    buf: fx.Array[fx.Int8, LDS_BYTES, 16]


@cache
def build():
    name = kernel_symbol("qwen_mtp_expert_owned_g1", rows=4, version=2)

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        mixed: Int64,
        ids: Int64,
        weights: Int64,
        sorted_ids: Int64,
        sorted_weights: Int64,
        experts: Int64,
        valid: Int64,
        m_indices: Int64,
        reverse: Int64,
        out: Int64,
        w1: Int64,
        s1: Int64,
        aq: Int64,
        ascale: Int64,
        slot_map: Int64,
    ):
        tid, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tid % 64
        wave = rocdl.readfirstlane(T.i32, as_ir_value(tid // 64))
        sm = fx.SharedAllocator().allocate(Smem).peek()
        # Native XCD4 remap, with a fixed graph-stable 41-slot upper bound.
        xc = pid % 8
        wgid = xc * 25 + fx.min(xc, 5) + pid // 8
        group = wgid // 20
        first = group * 4
        width = fx.min(fx.Int32(SLOTS) - first, 4)
        within = wgid % 20
        slot = first + within % width
        tile_n = within // width
        tile = slot * 5 + tile_n
        pick = fx.max(slot - 1, 0)
        eid = fx.Int32(
            bo.buffer_load(
                rsrc(ids), pick // 10 * 11 + pick % 10, dtype=T.i32, vec_width=1
            )
        )
        eid = (slot == 0).select(fx.Int32(512), eid)
        # Every wave reads the same 40 picks; ownership is uniform for the CTA.
        j = fx.min(lane, 39)
        seen = fx.Int32(
            bo.buffer_load(rsrc(ids), j // 10 * 11 + j % 10, dtype=T.i32, vec_width=1)
        )
        hits = ballot((lane < 40) & (seen == eid))
        before = (fx.Int64(1) << pick) - fx.Int64(1)
        active = (slot == 0) | ((hits & before) == 0)
        mask = fx.Int32(0)
        for row in fx.range_constexpr(4):
            mask = mask | ((hits >> (row * 10)) & fx.Int64(1023) != 0).select(
                fx.Int32(1 << row), fx.Int32(0)
            )
        mask = (slot == 0).select(fx.Int32(15), mask)
        # Only N tile zero writes metadata; all N tiles derive identical local
        # mappings, so G1 never reads or waits for these global writes.
        if tile_n == 0:
            if tid == 0:
                bo.buffer_store(active.select(eid, fx.Int32(-1)), rsrc(experts), slot)
                if active:
                    counter = fx.inttoptr(
                        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), valid
                    )
                    ticket = (
                        fx.llvm.atomic_add(counter, fx.Int32(16), syncscope="agent")
                        // 16
                    )
                    bo.buffer_store(slot, rsrc(slot_map), ticket)
            if tid < 16:
                row = fx.min(tid, 3)
                row_hits = (hits >> (row * 10)) & fx.Int64(1023)
                low = row_hits & (-row_hits)
                route = popcount(low - fx.Int64(1))
                route = (slot == 0).select(fx.Int32(10), fx.min(route, 9))
                live = active & (tid < 4) & (((mask >> row) & 1) != 0)
                wt = fx.Float32(
                    bo.buffer_load(
                        rsrc(weights), row * 11 + route, dtype=T.f32, vec_width=1
                    )
                )
                at = slot * 16 + tid
                bo.buffer_store(
                    live.select((route << 24) | row, fx.Int32(4)), rsrc(sorted_ids), at
                )
                bo.buffer_store(
                    live.select(wt, fx.Float32(0)), rsrc(sorted_weights), at
                )
                bo.buffer_store(live.select(row, fx.Int32(4)), rsrc(m_indices), at)
                if live:
                    bo.buffer_store(at, rsrc(reverse), row * 11 + route)
        if active:
            _gemm1_body(
                sm.buf.ptr,
                mixed,
                mixed,
                w1,
                s1,
                experts,
                m_indices,
                aq,
                ascale,
                mixed,
                mixed,
                tile,
                lane,
                wave,
                True,
                fx.Int32(4),
                fx.Int32(SLOTS),
                expert_override=eid,
                row_mask=mask,
                BM=16,
                BN=256,
                BK=256,
                inline_quant=True,
                prefetch_hidden=True,
                K=2560,
                N_OUT=1280,
                NE=513,
                native_scale_layout=True,
                swiglu_limit=float("inf"),
                k_stages=K_STAGES,
            )

    @launcher(name)
    def launch(
        mixed: Int64,
        ids: Int64,
        weights: Int64,
        sorted_ids: Int64,
        sorted_weights: Int64,
        experts: Int64,
        valid: Int64,
        m_indices: Int64,
        reverse: Int64,
        out: Int64,
        w1: Int64,
        s1: Int64,
        aq: Int64,
        ascale: Int64,
        slot_map: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCE
        kernel(
            mixed,
            ids,
            weights,
            sorted_ids,
            sorted_weights,
            experts,
            valid,
            m_indices,
            reverse,
            out,
            w1,
            s1,
            aq,
            ascale,
            slot_map,
        ).launch(grid=(GRID,), block=(256,), stream=stream)

    launch.compile_hints["llvm_options"] = {"enable-post-misched": False}
    ABI.check(kernel, launch)
    return launch
