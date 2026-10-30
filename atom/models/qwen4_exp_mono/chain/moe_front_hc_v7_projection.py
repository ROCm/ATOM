"""V7 router: same BF16 output plus an paired value/tag publication.

Activation staging, MFMA and reduction match V6. The exact BF16-rounded
value is also passed as FP32 to the existing tagged Mailbox publisher.
This module is isolated and does not patch the production helper.
"""

import hashlib
from pathlib import Path

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.stages.batch_projection import stage_activation
from atom.mono.device.ops import CM_DEV, mfma_bf16, row_sum, rsrc, traced

SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@traced
def router_projection(
    x,
    weight,
    out,
    task,
    tokens,
    sm,
    load=None,
    prefetch=False,
    split_k=1,
    publish_value=None,
):
    # A split publishes FP32 partials. The MoE preparation consumer combines
    # them before the router's BF16 rounding and top-k boundary.
    tile = task if split_k == 1 else task % 33
    split = 0 if split_k == 1 else task // 33
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    if not prefetch:
        stage_activation(x, sm.activation.ptr, tid, tokens, load)
    token = fx.min(lane % 16, tokens - 1)
    row = fx.min(tile * 16 + lane % 16, 512)
    first_at = (wave + 8 * split * (10 // split_k)) * 32 + 8 * (lane // 16)
    first_w = (
        fx.Vector(
            bo.buffer_load(
                rsrc(weight), row * 1280 + first_at // 2, vec_width=4, dtype=T.i32
            )
        ).bitcast(fx.BFloat16)
        if prefetch
        else None
    )
    if prefetch:
        stage_activation(x, sm.activation.ptr, tid, tokens, load)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k in range_constexpr(10 // split_k):
        at = (wave + 8 * (k + split * (10 // split_k))) * 32 + 8 * (lane // 16)
        w = (
            first_w
            if prefetch and k == 0
            else fx.Vector(
                bo.buffer_load(
                    rsrc(weight), row * 1280 + at // 2, vec_width=4, dtype=T.i32
                )
            ).bitcast(fx.BFloat16)
        )
        xval = fx.Vector(
            fx.ptr_load(
                sm.activation.ptr + token * 1280 + at // 2,
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
            if split_k == 1:
                value_bf = value.to(fx.BFloat16)
                bo.buffer_store(value_bf, rsrc(out), t * 513 + n, cache_modifier=CM_DEV)
                publish_value(t * 513 + n, fx.Float32(value_bf))
            else:
                bo.buffer_store(
                    value,
                    rsrc(out),
                    (split * tokens + t) * 513 + n,
                    cache_modifier=CM_DEV,
                )
