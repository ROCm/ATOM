# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 shared FFN: gate-first prefetch, one input copy, then up.

The original 72 N16 owners, eight K streams, red addresses, BF16/SiLU
ordering and odd-task SMID→SMQ/SMC handoff are retained. Native wave32
matrix fragments and the gfx1250 non-temporal encoding are translations.
"""
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import HIDDEN, lds_row
from atom.models.deepseek_v41.mono.kernels.dims import Dims, ROWS
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_ffn import ffn_input_layout_c1, run_ffn_input_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_projection import reference_fp8_loads, reference_fp8_matrices_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_route import stage_route_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox, publish_plain
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import _silu_mul, smid_quant
from atom.models.deepseek_v41.mono.kernels.moe_shape import RouteShape
from atom.mono.device.ops import CM_DEV, CM_NT_GFX1250, bf16_round, fresh, kernel_symbol, launcher, ld_dev, row_sum, traced
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import BLOCKS, first_task

THREADS, WAVES = 256, 8


@traced
def load_shared_x_c1(c):
    """Original X8RDY wait, then plain X8/X8S copied once into private LDS."""
    tid = c["tid"]
    if tid < 16:
        c["poll"]([(c["x8rdy"], 0, 1)])
    gpu.barrier()
    for name, width, dst in (("x8", HIDDEN // 4, c["xl"]),
                             ("x8s", HIDDEN // 32, c["xsl"])):
        for start in range_constexpr(0, width // 4, THREADS):
            quad = tid + start
            if quad < width // 4:
                words = fx.Vector(ld_dev(c[name].value, quad * 4, 4))
                fx.ptr_store(words, dst + quad * 4)
    gpu.barrier()


@traced
def stage_shared_c1(c, task, gops):
    """Mother gate→up joins and original per-task tagged output/quantization."""
    tid, red, d, a = c["tid"], c["red"], c["d"], c["args"]
    reference_fp8_matrices_c1(c, HIDDEN, gops, mother_red=True, join=False)
    gpu.barrier()
    gate = fx.Float32(0.)
    if tid < ROWS:
        gate = row_sum(red, tid, 0)
    gpu.barrier()
    uops = reference_fp8_loads(c, a["sgu"], a["sgu_s"], HIDDEN,
                               d.shared_tasks + task, CM_NT_GFX1250)
    reference_fp8_matrices_c1(c, HIDDEN, uops, mother_red=True, join=False)
    gpu.barrier()
    if tid < ROWS:
        up = row_sum(red, tid, 0)
        v = fx.Float32(_silu_mul(bf16_round(gate), bf16_round(up)).to(fx.BFloat16))
        c["put"](c["smid"], task * ROWS + tid, v)
        fx.ptr_store(v, c["pair"] + tid)
    gpu.barrier()
    if task % 2 == 1:
        smid_quant(c, task, 0, 1)
    gpu.barrier()


@traced
def run_shared_c1(c, bid):
    """Original one straight-line shared unit per owner, base112 at c1."""
    unit = first_task(bid, c["rs"].shared0)
    if unit < c["rs"].shared_units:
        gops = reference_fp8_loads(c, c["args"]["sgu"], c["args"]["sgu_s"],
                                   HIDDEN, unit, CM_NT_GFX1250)
        load_shared_x_c1(c)
        stage_shared_c1(c, unit, gops)


def shared_symbol_c1():
    return kernel_symbol("gfx1250_reference_ffn_shared", b=BLOCKS, t=THREADS)


@lru_cache(None)
def build_reference_ffn_shared_c1():
    """Isolate original XRDY→XQ/router→route→shared; gate/UG follows later."""
    d, rs, layout = Dims(2), RouteShape(384, 6, 1, 2, 1), ffn_input_layout_c1()
    name = shared_symbol_c1()

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, lds_row(HIDDEN // 4), 16]
        xsl: fx.Array[fx.Int32, lds_row(HIDDEN // 32), 16]
        pair: fx.Array[fx.Float32, ROWS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(x: Int64, gate_w: Int64, bias: Int64, sgu: Int64, sgu_s: Int64,
               scratch: Int64, diag: Int64, layer: Int64):
        tid, bid = fresh(fx.thread_idx.x), fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        iptr = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        failure = fx.inttoptr(iptr, diag + fx.Int64(BLOCKS * 8 * 4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(iptr, layer)) + 128, diag,
                               {key: i + 1 for i, key in enumerate(layout)}, failure)
        c = dict(S=1, tid=tid, bid=bid, lane=tid % 32, wave=tid // 32, d=d, rs=rs,
                 args=dict(x=x, gate_w=gate_w, bias=bias, sgu=sgu, sgu_s=sgu_s),
                 red=lds.red.ptr, xl=lds.xl.ptr, xsl=lds.xsl.ptr, pair=lds.pair.ptr,
                 put=mb.put, put_words=mb.put_words, poll=mb.poll,
                 x_cm=CM_DEV, x_ready=True)
        for key in ("xrdy", "x8rdy", "x8r", "x8rs", "x8", "x8s", "logit", "route",
                    "smid", "smq", "smc"):
            c[key] = sreg(scratch, layout[key][0], key)
        if bid == 168:
            publish_plain(c["put"], c["xrdy"], 0, 1, tid == 0)
        run_ffn_input_c1(c, bid, True)
        for token in range(first_task(bid, rs.route0), 1, BLOCKS):
            stage_route_c1(c, token)
        run_shared_c1(c, bid)

    @launcher(name)
    def launch(x: Int64, gate_w: Int64, bias: Int64, sgu: Int64, sgu_s: Int64,
               scratch: Int64, diag: Int64, layer: Int64, stream: fx.Stream):
        kernel(x, gate_w, bias, sgu, sgu_s, scratch, diag, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream)
    return launch
