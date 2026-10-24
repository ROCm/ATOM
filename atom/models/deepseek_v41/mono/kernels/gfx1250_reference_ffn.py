# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 FFN input and router structure on eight physical wave32s.

The original XQ quantization and plain-region ABI are retained. Router tasks
remain N16/contiguous K1280 parts, each wave's five K32 steps w,w+8,... .
Each native WMMA lane carries two original eight-BF16 operand streams, and
its output is restored to the mother red addresses before the wave sum.
"""
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.model_ops.deepseek_v41.router import router_weight_loads, router_x_loads
from atom.models.deepseek_v41.mono.kernels.gfx1250_attention import _wmma
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_indexer import vector_address
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_sync import reference_mailbox, publish_plain
from atom.models.deepseek_v41.mono.kernels.layer_post import LayerPostBuild, scratch_layout
from atom.models.deepseek_v41.mono.kernels.moe import await_x, stage_xq
from atom.models.deepseek_v41.mono.kernels.moe_shape import RouteShape
from atom.mono.device.ops import CM_DEV, CM_NT_GFX1250, fresh, kernel_symbol, launcher, row_sum, traced
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import BLOCKS, first_task

THREADS, WAVES = 256, 8


def ffn_input_layout_c1(index_bound_max=0, tp=2):
    """Exact mother K2a/K2b partition and final XRDY, including unused regions."""
    return scratch_layout(LayerPostBuild(tokens=1, tp=tp, ug_groups=1,
                                         index_bound_max=index_bound_max))


@traced
def router_weights_c1(c, task):
    tile, part = task//4, task%4
    lane, wave = c["lane"], c["wave"]
    # Two mother virtual lane groups supply the native sixteen-BF16 operand.
    virtual_lane = lane%16+(lane//16)*32
    lo = router_weight_loads(c["args"]["gate_w"], tile, part, virtual_lane, wave, CM_NT_GFX1250)
    hi = router_weight_loads(c["args"]["gate_w"], tile, part, virtual_lane+16, wave, CM_NT_GFX1250)
    return [fx.Vector.from_elements([lo[i][j] for j in range(4)]+[hi[i][j] for j in range(4)],fx.Int32)
            for i in range(5)]


@traced
def stage_router_c1(c, task, weights):
    """Mother part publication after its five original K32 WMMA steps."""
    lane, wave = c["lane"], c["wave"]
    virtual_lane = lane%16+(lane//16)*32
    part, tile = task%4, task//4
    lo = router_x_loads(c["args"]["x"], part, 1, virtual_lane, wave, CM_DEV)
    hi = router_x_loads(c["args"]["x"], part, 1, virtual_lane+16, wave, CM_DEV)
    inputs = [fx.Vector.from_elements([lo[i][j] for j in range(4)]+[hi[i][j] for j in range(4)],fx.Int32)
              for i in range(5)]
    accum = fx.make_rmem_tensor(8, fx.Float32)
    accum.store(fx.full(8, 0., fx.Float32))
    for i in range_constexpr(5):
        _wmma(accum, inputs[i].bitcast(fx.BFloat16), weights[i].bitcast(fx.BFloat16))
    if lane%16 == 0:
        values = accum.load()
        for j in range_constexpr(8):
            row = lane//16*8+j
            at = (wave*64+16*(row//4))*4+row%4
            fx.ptr_store(values[j], c["red"]+at)
    gpu.barrier()
    if c["tid"] < 16:
        row = c["tid"]
        value = row_sum(c["red"], row, 0)
        c["put"](c["logit"], (tile*16+row)*4+part, value)
    gpu.barrier()


@traced
def run_ffn_input_c1(c, bid, router=True):
    """Original c1 XQ0=0, ROUTER0=8 and wait-before-weight-load branch."""
    for token in range(first_task(bid,0), 1, BLOCKS):
        await_x(c, [token])
        stage_xq(c, token)
        publish_plain(c["put"], c["x8rdy"], token, 1, c["tid"]==0)
    if const_expr(router):
        for task in range(first_task(bid,8), 96, BLOCKS):
            # The mother prefetch-before-XRDY branch starts at S>=12.
            await_x(c, [0])
            weights = router_weights_c1(c, task)
            stage_router_c1(c, task, weights)


def ffn_input_symbol_c1(router=True):
    return kernel_symbol("gfx1250_reference_ffn_input", b=BLOCKS, t=THREADS, rt=int(router))


@lru_cache(None)
def build_reference_ffn_input_c1(router=True):
    """Isolate the original XRDY producer and actual XQ/router consumers."""
    layout, name = ffn_input_layout_c1(), ffn_input_symbol_c1(router)
    @fx.struct
    class Smem:
        red:fx.Array[fx.Float32, WAVES*64*4, 16]
    @flyc.kernel(name=name, known_block_size=[THREADS,1,1])
    def kernel(x:Int64, gate_w:Int64, scratch:Int64, diag:Int64, layer:Int64):
        tid, bid = fresh(fx.thread_idx.x), fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        iptr = fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4)
        failure = fx.inttoptr(iptr,diag+fx.Int64(BLOCKS*8*4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(iptr,layer))+128, diag,
            {key:i+1 for i,key in enumerate(layout)}, failure)
        x, gate_w, scratch = map(vector_address, (x,gate_w,scratch))
        c = dict(S=1, tid=tid, bid=bid, lane=tid%32, wave=tid//32,
                 args=dict(x=x,gate_w=gate_w), red=lds.red.ptr, rs=RouteShape(384,6,1,2,1),
                 put=mb.put, poll=mb.poll, x_cm=CM_DEV, x_ready=True)
        for key in ("xrdy","x8rdy","x8r","x8rs","x8","x8s","logit"):
            c[key] = sreg(scratch,layout[key][0],key)
        if bid == 168:
            publish_plain(c["put"], c["xrdy"], 0, 1, tid==0)
        run_ffn_input_c1(c, bid, router)
    @launcher(name)
    def launch(x:Int64, gate_w:Int64, scratch:Int64, diag:Int64, layer:Int64, stream:fx.Stream):
        kernel(x,gate_w,scratch,diag,layer).launch(grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
