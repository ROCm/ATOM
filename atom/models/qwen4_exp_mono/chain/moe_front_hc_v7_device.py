"""Experimental C1 HC up/router -> top-k -> compact task phase, V7.

Keeps production four-FP32-part HC down inputs, BF16 mixed/router endpoints
and the existing eight-wave HC/router arithmetic. Four request-row selectors
consume their own tagged BF16-rounded router values; one wave publishes the compact M16 task table.
Native/prebound G1 and G2 execute after this phase completes on the stream.
The 512-thread HC phase never calls a 256-thread G1 body inside a divergent
wave subset. The new mailbox waits have a diagnostic bounded variant.
V7 preserves the HC payload, arithmetic, geometry and 20-argument ABI. Its
flags allocation is larger: 2052 value/tag pairs plus four selector ready pairs.
Each selector thread polls one routed logit; lane zero also polls shared 512.
Selection-to-task still uses V6's drained publish. No router-ready flag remains.
"""

import hashlib
from functools import cache
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.chain.moe_front_device import (
    SOURCE as TASK_SOURCE,
)
from atom.models.qwen4_exp_mono.chain.moe_front_device import (
    emit_tasks,
)
from atom.models.qwen4_exp_mono.chain.moe_front_hc_v7_contract import (
    CLEAR_FIRST,
    CLEAR_TASKS,
    HC_TASKS,
    ROUTER_TASKS,
    ROUTER_VALUE_OFFSET,
    SELECT_FIRST,
    SELECT_READY_OFFSET,
    SELECT_ROWS,
    TASK_PUBLISHER,
)
from atom.models.qwen4_exp_mono.chain.moe_front_hc_v7_projection import (
    SOURCE as PROJECTION_SOURCE,
)
from atom.models.qwen4_exp_mono.chain.moe_front_hc_v7_projection import (
    router_projection as _router_projection,
)
from atom.models.qwen4_exp_mono.kernels.batch_hc_front import (
    load_payload,
    store_payload,
)
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_hc import Smem as HcSmem
from atom.models.qwen4_exp_mono.stages.batch_hc import emit_up
from atom.models.qwen4_exp_mono.stages.batch_projection import G1Smem
from atom.models.qwen4_exp_mono.stages.hc import _st_bf
from atom.models.qwen4_exp_mono.stages.router import _float_key
from atom.mono.device.ops import CM_DEV, kernel_symbol, launcher, rsrc, traced
from atom.mono.device.select import (
    KEY_LEAST,
    compact,
    count_beaten,
    group_floor,
    put_key,
)
from atom.mono.device.sync import Mailbox, publish, sreg
from atom.mono.plan.layout import DIAG_WORDS
from atom.mono.runtime.abi import KernelAbi

SOURCE = hashlib.sha256(
    Path(__file__).read_bytes()
    + Path(__file__).with_name("moe_front_hc_v7_contract.py").read_bytes()
    + TASK_SOURCE.encode()
    + PROJECTION_SOURCE.encode()
).hexdigest()
ABI = KernelAbi(
    (
        "part",
        "normed",
        "weight",
        "raw",
        "stride",
        "router",
        "logits",
        "mixed",
        "payload",
        "ids",
        "weights",
        "sorted_ids",
        "sorted_weights",
        "experts",
        "valid",
        "m_indices",
        "reverse",
        "out",
        "flags",
        "diag",
    )
)
_STREAM = fx.Stream(None)


@fx.struct
class SelectSmem:
    keys: fx.Array[fx.Int32, 1024, 16]
    candidates: fx.Array[fx.Int32, 256, 16]
    count: fx.Array[fx.Int32, 1, 16]
    floor: fx.Array[fx.Int32, 129, 16]
    values: fx.Array[fx.Float32, 20, 16]


@fx.union
class PhaseSmem:
    # FlyDSL union members may expose struct views through .peek(); nesting
    # a union here would require the unsupported union __peek_from_ptr__.
    hc: HcSmem
    front: G1Smem
    select: SelectSmem


@traced
def release_after_writers(put, region, index, tid):
    # Selector IDs/weights remain plain CM_DEV stores: drain before ready.
    publish(put, region, index, fx.Int32(1), tid == 0)


@traced
def wait_all(poll, region, count, tid):
    if tid == 0:
        poll([(region, i, 1) for i in range(count)])
    gpu.barrier()


@traced
def select_row(poll, router_values, ids, weights, row, sm):
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    logit = poll([(router_values, row * 513 + tid, 1)])[0][0].bitcast(fx.Float32)
    key, logit = _float_key(logit)
    mine = put_key(sm.keys.ptr, tid, key, fx.Int32(0x7FFFFFFF) - tid, fx.Boolean(True))
    if tid < 128:
        fx.ptr_store(fx.Int32(0), sm.candidates.ptr + 2 * tid)
        fx.ptr_store(fx.Int32(KEY_LEAST), sm.candidates.ptr + 2 * tid + 1)
    if tid == 0:
        fx.ptr_store(fx.Int32(0), sm.count.ptr)
    floor = group_floor([key], 16, sm.floor.ptr, lane, wave)
    compact(
        sm.candidates.ptr,
        sm.count.ptr,
        128,
        [(key, fx.Int32(0x7FFFFFFF) - tid, fx.Boolean(True))],
        floor,
    )
    gpu.barrier()
    count = fx.ptr_load(sm.count.ptr)
    every = count > 128
    # Both paths hold signed int64 keys and preserve lower-expert tie order.
    rank = fx.Int32(0)
    if every:
        rank = count_beaten(sm.keys.ptr, [mine], fx.Int32(512))[0]
    else:
        rank = count_beaten(sm.candidates.ptr, [mine], count)[0]
    if rank < 10:
        bo.buffer_store(tid, rsrc(ids), row * 11 + rank, cache_modifier=CM_DEV)
        fx.ptr_store(logit, sm.values.ptr + rank)
    gpu.barrier()
    if tid == 0:
        maximum = fx.ptr_load(sm.values.ptr)
        total = fx.Float32(0)
        for i in range_constexpr(10):
            v = fx.Float32(fmath.exp(fx.ptr_load(sm.values.ptr + i) - maximum))
            fx.ptr_store(v, sm.values.ptr + 10 + i)
            total = total + v
        inv = fx.Float32(1) / total
        for i in range_constexpr(10):
            bo.buffer_store(
                fx.ptr_load(sm.values.ptr + 10 + i) * inv,
                rsrc(weights),
                row * 11 + i,
                cache_modifier=CM_DEV,
            )
        shared = poll([(router_values, row * 513 + 512, 1)])[0][0].bitcast(fx.Float32)
        gate = fx.Float32(1) / (fx.Float32(1) + fx.Float32(fmath.exp(-shared)))
        bo.buffer_store(fx.Int32(512), rsrc(ids), row * 11 + 10, cache_modifier=CM_DEV)
        bo.buffer_store(gate, rsrc(weights), row * 11 + 10, cache_modifier=CM_DEV)
    gpu.barrier()


@cache
def build(debug=True):
    name = kernel_symbol(
        "qwen_mtp_hc_compact_front", rows=4, debug=int(debug), version=7
    )

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        router: Int64,
        logits: Int64,
        mixed: Int64,
        payload: Int64,
        ids: Int64,
        weights: Int64,
        sorted_ids: Int64,
        sorted_weights: Int64,
        experts: Int64,
        valid: Int64,
        m_indices: Int64,
        reverse: Int64,
        out: Int64,
        flags: Int64,
        diag: Int64,
    ):
        tid, task = fx.thread_idx.x, fx.block_idx.x
        sm = fx.SharedAllocator().allocate(PhaseSmem)
        # Resolve the storage views before SCF needs branch live-state values.
        hc_sm = sm.hc.peek()
        front_sm = sm.front.peek()
        select_sm = sm.select.peek()
        epoch = (
            fx.Int32(
                bo.buffer_load(
                    rsrc(payload),
                    task * 16 + 1,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            + 1
        )
        mb = Mailbox(
            epoch - 1, diag if debug else None, {"router_values": 1, "selection": 2}
        )
        # Bound callables are compile-time context, as in sync.publish(put,...).
        put, poll = mb.put, mb.poll
        # Each varying-index value poll owns a diagnostic bank. All banks are
        # absent in release; ordinary selection-ready leader waits use bank 0.
        value_mb = Mailbox(
            epoch - 1,
            diag + fx.Int64(tid + 1) * (HC_TASKS * DIAG_WORDS * 4) if debug else None,
            {"router_values": 1, "selection": 2},
        )
        value_poll = value_mb.poll
        router_values = sreg(flags, ROUTER_VALUE_OFFSET, "router_values")
        selection_ready = sreg(flags, SELECT_READY_OFFSET, "selection")

        def store(col, value):
            _st_bf(mixed, col, value)
            store_payload(payload, col, value, epoch)

        emit_up(part, normed, weight, raw, stride, hc_sm, 4, True, store, down_parts=4)
        gpu.barrier()
        if task < ROUTER_TASKS:
            _router_projection(
                payload,
                router,
                logits,
                task,
                4,
                front_sm,
                load=lambda pair, width: load_payload(payload, pair, epoch, width),
                prefetch=True,
                split_k=1,
                publish_value=lambda index, value: put(router_values, index, value),
            )
        elif task < SELECT_FIRST + SELECT_ROWS:
            select_row(
                value_poll, router_values, ids, weights, task - SELECT_FIRST, select_sm
            )
            release_after_writers(put, selection_ready, task - SELECT_FIRST, tid)
        elif task == TASK_PUBLISHER:
            wait_all(poll, selection_ready, SELECT_ROWS, tid)
            if tid < 64:
                emit_tasks(
                    ids,
                    weights,
                    sorted_ids,
                    sorted_weights,
                    experts,
                    valid,
                    m_indices,
                    reverse,
                    tid,
                    input_cm=CM_DEV,
                )
        elif task < CLEAR_FIRST + CLEAR_TASKS:
            bo.buffer_store(fx.Int32(0), rsrc(out), (task - CLEAR_FIRST) * 512 + tid)

    @launcher(name)
    def launch(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        router: Int64,
        logits: Int64,
        mixed: Int64,
        payload: Int64,
        ids: Int64,
        weights: Int64,
        sorted_ids: Int64,
        sorted_weights: Int64,
        experts: Int64,
        valid: Int64,
        m_indices: Int64,
        reverse: Int64,
        out: Int64,
        flags: Int64,
        diag: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES, SOURCE
        kernel(
            part,
            normed,
            weight,
            raw,
            stride,
            router,
            logits,
            mixed,
            payload,
            ids,
            weights,
            sorted_ids,
            sorted_weights,
            experts,
            valid,
            m_indices,
            reverse,
            out,
            flags,
            diag,
        ).launch(grid=(HC_TASKS,), block=(512,), stream=stream)

    ABI.check(kernel, launch)
    return launch
