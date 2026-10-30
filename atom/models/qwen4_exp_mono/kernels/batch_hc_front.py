"""B2/B4 HC up and MoE front share a tagged BF16 activation hand-off.

Every N16 producer advances its own previous payload tag. The complete grid
runs once per seam on the same stream, so all producers use the same epoch.
Consumers load data and tags together; no separate completion flag or drain is
needed. The CTA barrier protects the HC/front LDS union before reuse.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_hc import Smem as HcSmem
from atom.models.qwen4_exp_mono.stages.batch_hc import emit_up
from atom.models.qwen4_exp_mono.stages.batch_projection import (
    G1Smem,
    _router_projection,
    _shared_projection,
)
from atom.models.qwen4_exp_mono.stages.quant import quantize_values
from atom.mono.device.ops import (
    CM_DEV,
    bf16_pair,
    bf_hi,
    bf_lo,
    kernel_symbol,
    launcher,
    rsrc,
    traced,
)


@traced
def load_payload(payload, pair, epoch, width):
    words = []
    for half in range_constexpr(width // 2):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(payload),
                (pair + half * 2) * 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        words += [v[0], v[1], v[2], v[3]]
    bad = words[1] != epoch
    for at in range_constexpr(1, width):
        bad = bad | (words[at * 2 + 1] != epoch)
    while bad:
        rocdl.s_sleep(1)
        words = []
        for half in range_constexpr(width // 2):
            v = fx.Vector(
                bo.buffer_load(
                    rsrc(payload),
                    (pair + half * 2) * 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            words += [v[0], v[1], v[2], v[3]]
        bad = words[1] != epoch
        for at in range_constexpr(1, width):
            bad = bad | (words[at * 2 + 1] != epoch)
    return fx.Vector.from_elements([words[at * 2] for at in range(width)], fx.Int32)


@fx.union
class Smem:
    hc: HcSmem
    front: G1Smem


@traced
def store_payload(payload, col, value, epoch):
    other = value.shuffle_xor(1, 64)
    if fx.thread_idx.x % 2 == 0:
        word = bf16_pair(value, other).bitcast(fx.Int32)
        bo.buffer_store(
            fx.Vector.from_elements([word, epoch], fx.Int32),
            rsrc(payload),
            col,
            cache_modifier=CM_DEV,
        )


@traced
def front_tasks(
    payload,
    router_weight,
    logits,
    shared_weight,
    shared_mid,
    q,
    scale,
    task,
    tokens,
    shared_intermediate,
    quant0,
    stride_bytes,
    front_sm,
    epoch,
):
    if task < 33:
        _router_projection(
            payload,
            router_weight,
            logits,
            task,
            tokens,
            front_sm,
            load=lambda pair, width: load_payload(payload, pair, epoch, width),
            prefetch=True,
        )
    elif task < quant0:
        _shared_projection(
            payload,
            shared_weight,
            shared_mid,
            task - 33,
            tokens,
            shared_intermediate,
            front_sm,
            load=lambda pair, width: load_payload(payload, pair, epoch, width),
            prefetch=True,
        )
    elif task < quant0 + tokens * 20:
        quant_task(payload, q, scale, task - quant0, stride_bytes, epoch)


@traced
def quant_task(payload, q, scale, task, stride_bytes, epoch):
    tid = fx.thread_idx.x
    token = fx.Int64(task // 20)
    group = task % 20
    words = fx.Vector.filled(2, 0, fx.Int32)
    if tid < 32:
        words = load_payload(payload, token * 1280 + group * 64 + tid * 2, epoch, 2)
    values = [bf_lo(words[0]), bf_hi(words[0]), bf_lo(words[1]), bf_hi(words[1])]
    packed, dequant = quantize_values(values)
    if tid < 32:
        bo.buffer_store(
            packed,
            rsrc(q + token * stride_bytes),
            group * 32 + tid,
            cache_modifier=CM_DEV,
        )
    if tid == 0:
        bo.buffer_store(
            dequant, rsrc(scale + token * stride_bytes), group, cache_modifier=CM_DEV
        )


_STREAM = fx.Stream(None)


def build(tokens, tp, stride_bytes):
    assert tokens in (2, 4)
    shared_intermediate = Dims(tp).shared_intermediate
    shared_tiles = shared_intermediate // 16
    quant0 = 33 + shared_tiles
    assert quant0 + tokens * 20 <= 160
    name = kernel_symbol("qwen_batch_hc_moe_front", b=tokens, tp=tp)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        router_weight: Int64,
        logits: Int64,
        shared_weight: Int64,
        shared_mid: Int64,
        q: Int64,
        scale: Int64,
        payload: Int64,
    ):
        sm = fx.SharedAllocator().allocate(Smem)
        task = fx.block_idx.x
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
        emit_up(
            part,
            normed,
            weight,
            raw,
            stride,
            sm.hc.peek(),
            tokens,
            True,
            lambda col, value: store_payload(payload, col, value, epoch),
        )
        # Join only to protect the union LDS; payload tags publish their own data.
        gpu.barrier()
        front_tasks(
            payload,
            router_weight,
            logits,
            shared_weight,
            shared_mid,
            q,
            scale,
            task,
            tokens,
            shared_intermediate,
            quant0,
            stride_bytes,
            sm.front.peek(),
            epoch,
        )

    @launcher(name)
    def launch(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        router_weight: Int64,
        logits: Int64,
        shared_weight: Int64,
        shared_mid: Int64,
        q: Int64,
        scale: Int64,
        payload: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(
            part,
            normed,
            weight,
            raw,
            stride,
            router_weight,
            logits,
            shared_weight,
            shared_mid,
            q,
            scale,
            payload,
        ).launch(grid=(160,), block=(512,), stream=stream)

    return launch
