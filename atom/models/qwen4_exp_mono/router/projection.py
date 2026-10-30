"""Two K1280 router partitions, each with half-row staging and tagged FP32 output."""

import hashlib
from pathlib import Path

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.kernels.batch_hc_front import load_payload
from atom.mono.device.ops import mfma_bf16, row_sum, rsrc, traced

SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@fx.struct
class RouterSmem:
    activation: fx.Array[fx.Int32, 4 * 640, 16]
    reduction: fx.Array[fx.Float32, 8 * 64 * 4, 16]


@traced
def stage_half(payload, activation, tid, split, epoch):
    if tid < 160:
        for token in range_constexpr(4):
            words = load_payload(
                payload, token * 1280 + split * 640 + tid * 4, epoch, 4
            )
            fx.ptr_store(words, activation + token * 640 + tid * 4)
    gpu.barrier()


@traced
def router_projection(x, weight, out, task, tokens, sm, epoch, publish_value):
    tile = task % 33
    split = task // 33
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    token = fx.min(lane % 16, tokens - 1)
    row = fx.min(tile * 16 + lane % 16, 512)
    first_at = (wave + 8 * split * 5) * 32 + 8 * (lane // 16)
    first_w = fx.Vector(
        bo.buffer_load(
            rsrc(weight), row * 1280 + first_at // 2, vec_width=4, dtype=T.i32
        )
    ).bitcast(fx.BFloat16)
    stage_half(x, sm.activation.ptr, tid, split, epoch)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k in range_constexpr(5):
        at = (wave + 8 * (k + split * 5)) * 32 + 8 * (lane // 16)
        w = (
            first_w
            if k == 0
            else fx.Vector(
                bo.buffer_load(
                    rsrc(weight), row * 1280 + at // 2, vec_width=4, dtype=T.i32
                )
            ).bitcast(fx.BFloat16)
        )
        xval = fx.Vector(
            fx.ptr_load(
                sm.activation.ptr
                + token * 640
                + (wave + 8 * k) * 16
                + 4 * (lane // 16),
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(w, xval, acc)
    fx.ptr_store(acc, sm.reduction.ptr + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < 16 * tokens:
        t = tid // 16
        n = tile * 16 + tid % 16
        if n <= 512:
            value = row_sum(sm.reduction.ptr, tid % 16, t)
            publish_value((split * tokens + t) * 513 + n, value)
