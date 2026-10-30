"""HC/PTPC with concurrent K512 BA tiles, reducing inside the PTPC consumers."""

import hashlib
from functools import cache
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.kernels.batch_hc_front import (
    load_payload,
    store_payload,
)
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_hc import emit_up, smem_type
from atom.models.qwen4_exp_mono.stages.hc import _st_bf
from atom.mono.device.mx import FP8_MAX, clamp_fp8
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    block_max,
    fp8_pack4,
    kernel_symbol,
    launcher,
    mfma_bf16,
    row_sum,
    rsrc,
)
from atom.mono.device.sync import Mailbox, sreg
from atom.mono.runtime.abi import KernelAbi

SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ABI = KernelAbi(
    (
        "part",
        "normed",
        "weight",
        "raw",
        "stride",
        "mixed",
        "q",
        "scale",
        "payload",
        "ba_weight",
        "ba",
        "partials",
        "epochs",
        "diag",
    )
)
_STREAM = fx.Stream(None)


@fx.struct
class QuantSmem:
    reduction: fx.Array[fx.Float32, 8, 16]


@fx.struct
class BaSmem:
    activation: fx.Array[fx.Int32, 4 * 256, 16]
    reduction: fx.Array[fx.Float32, 8 * 64 * 4, 16]


@cache
def build(debug=False):
    hc_type = smem_type(4)

    @fx.union
    class Smem:
        hc: hc_type
        quant: QuantSmem
        ba: BaSmem

    name = kernel_symbol("qwen_mtp_hc_ptpc_ba", rows=4, parts=5, debug=int(debug))

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        mixed: Int64,
        q: Int64,
        scale: Int64,
        payload: Int64,
        ba_weight: Int64,
        ba: Int64,
        partials: Int64,
        epochs: Int64,
        diag: Int64,
    ):
        sm = fx.SharedAllocator().allocate(Smem)
        hc_sm, quant_sm, ba_sm = sm.hc.peek(), sm.quant.peek(), sm.ba.peek()
        task, tid = fx.block_idx.x, fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        epoch = (
            fx.Int32(
                bo.buffer_load(
                    rsrc(epochs), task, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
                )
            )
            + 1
        )

        def store(col, value):
            _st_bf(mixed, col, value)
            store_payload(payload, col, value, epoch)

        emit_up(part, normed, weight, raw, stride, hc_sm, 4, True, store, down_parts=1)
        gpu.barrier()
        mb = Mailbox(epoch - 1, diag if debug else None, {"ba_partial": 1})
        region = sreg(partials, 0, "ba_partial")
        put, poll = mb.put, mb.poll
        if task < 4:
            words = load_payload(payload, task * 1280 + tid * 2, epoch, 2)
            tail = fx.Vector.filled(2, 0, fx.Int32)
            if tid < 128:
                tail = load_payload(payload, task * 1280 + 1024 + tid * 2, epoch, 2)
            values = [
                bf_lo(words[0]),
                bf_hi(words[0]),
                bf_lo(words[1]),
                bf_hi(words[1]),
            ]
            extra = [bf_lo(tail[0]), bf_hi(tail[0]), bf_lo(tail[1]), bf_hi(tail[1])]
            maximum = fx.Float32(0)
            for value in values + extra:
                maximum = fx.max(maximum, fx.max(value, -value))
            maximum = block_max(maximum, lane, wave, quant_sm.reduction.ptr)
            dequant = maximum / fx.Float32(FP8_MAX)
            inv = (dequant > 0).select(fx.Float32(1) / dequant, fx.Float32(0))
            bo.buffer_store(
                fp8_pack4(*[clamp_fp8(v * inv) for v in values]),
                rsrc(q),
                task * 640 + tid,
            )
            if tid < 128:
                bo.buffer_store(
                    fp8_pack4(*[clamp_fp8(v * inv) for v in extra]),
                    rsrc(q),
                    task * 640 + 512 + tid,
                )
            if tid == 0:
                bo.buffer_store(dequant, rsrc(scale), task)
            if tid < 96:
                parts = poll([(region, (s * 4 + task) * 96 + tid, 1) for s in range(5)])
                value = parts[0][0].bitcast(fx.Float32)
                for s in range_constexpr(1, 5):
                    value = value + parts[s][0].bitcast(fx.Float32)
                bo.buffer_store(value.to(fx.BFloat16), rsrc(ba), task * 96 + tid)
        elif task < 34:
            tile = (task - 4) % 6
            split = (task - 4) // 6
            row = tile * 16 + lane % 16
            token = fx.min(lane % 16, 3)
            first_at = split * 512 + wave * 32 + 8 * (lane // 16)
            first_w = fx.Vector(
                bo.buffer_load(
                    rsrc(ba_weight),
                    row * 1280 + first_at // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            ).bitcast(fx.BFloat16)
            if tid < 64:
                for t in range_constexpr(4):
                    words = load_payload(
                        payload, t * 1280 + split * 256 + tid * 4, epoch, 4
                    )
                    fx.ptr_store(words, ba_sm.activation.ptr + t * 256 + tid * 4)
            gpu.barrier()
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for k in range_constexpr(2):
                at = split * 512 + (wave + 8 * k) * 32 + 8 * (lane // 16)
                w = (
                    first_w
                    if k == 0
                    else fx.Vector(
                        bo.buffer_load(
                            rsrc(ba_weight),
                            row * 1280 + at // 2,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    ).bitcast(fx.BFloat16)
                )
                x = fx.Vector(
                    fx.ptr_load(
                        ba_sm.activation.ptr
                        + token * 256
                        + (wave + 8 * k) * 16
                        + 4 * (lane // 16),
                        result_type=T.vec(4, T.i32),
                    )
                ).bitcast(fx.BFloat16)
                acc = mfma_bf16(w, x, acc)
            fx.ptr_store(acc, ba_sm.reduction.ptr + (wave * 64 + lane) * 4)
            gpu.barrier()
            if tid < 64:
                value = row_sum(ba_sm.reduction.ptr, tid % 16, tid // 16)
                put(region, (split * 4 + tid // 16) * 96 + tile * 16 + tid % 16, value)
        if tid == 0:
            bo.buffer_store(epoch, rsrc(epochs), task, cache_modifier=CM_DEV)

    @launcher(name)
    def launch(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        mixed: Int64,
        q: Int64,
        scale: Int64,
        payload: Int64,
        ba_weight: Int64,
        ba: Int64,
        partials: Int64,
        epochs: Int64,
        diag: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCE, SOURCES
        kernel(
            part,
            normed,
            weight,
            raw,
            stride,
            mixed,
            q,
            scale,
            payload,
            ba_weight,
            ba,
            partials,
            epochs,
            diag,
        ).launch(grid=(160,), block=(512,), stream=stream)

    ABI.check(kernel, launch)
    return launch
