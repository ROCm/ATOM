# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 GDN K1/K2 specialization for Qwen3.8-Flash-Next.

K1 computes one rank's six-head GDN shard and publishes its BF16 output through
the symmetric ``attention`` peer region. K2 pulls the eight rank partials in a
fixed order, then runs the accepted TP8 HC+MoE finish and publishes the next
layer's MoE partial. The TP1 GDN emitter stays untouched so this specialization
cannot perturb its ISA.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from .gdr_decode import create_vk_gdr_decode_kernel
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels import finish
from atom.models.qwen4_exp_mono.kernels.collective import (
    PARTS,
    publish_pair,
    pull_part_to_bf16,
)
from atom.models.qwen4_exp_mono.peer_layout import peer_offsets
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    hw_exp2,
    hw_rsq,
    kernel_symbol,
    launcher,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    wave_sum,
)
from atom.mono.device.sync import Mailbox, publish, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import (
    BLOCKS,
    PERSISTENT_VALUE_ATTRS,
    THREADS,
    WAVES,
)
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

K1_ABI = KernelAbi(
    (
        "x",
        "projection_weight",
        "conv_weight",
        "conv_state",
        "state_read",
        "state_write",
        "a_log",
        "dt_bias",
        "state",
        "norm_weight",
        "output_weight",
        "partial_out",
        "peer_sym",
        "scratch",
        "layer",
    )
)

K2_ABI = KernelAbi(
    (
        "peers",
        "streams",
        "attn_out",
        "attn_raw",
        "hc_norm",
        "hc_down_inject",
        "hc_up",
        "router",
        "shared_w13",
        "shared_w2",
        "routed_w13",
        "routed_w13_scale",
        "routed_w2",
        "routed_w2_scale",
        "expert_map",
        "peer_sym",
        "streams_out",
        "pending_out",
        "pending_raw",
        "moe_out",
        "scratch",
        "layer",
    )
)

_DIMS = Dims(8)
HIDDEN = _DIMS.hidden
KEY_HEADS = _DIMS.gdn_key_heads
VALUE_HEADS = _DIMS.gdn_value_heads
HEAD_DIM = 128
CONV_WIDTH = 4
KEY_DIM = KEY_HEADS * HEAD_DIM
VALUE_DIM = VALUE_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
PROJ_DIM = CONV_DIM + VALUE_DIM + 2 * VALUE_HEADS
B_ROW = CONV_DIM + VALUE_DIM
A_ROW = B_ROW + VALUE_HEADS

ROWS = 16
K_STEP = 32
K_STEPS_PER_WAVE = HIDDEN // (K_STEP * WAVES)
OUT_K_STEPS_PER_WAVE = VALUE_DIM // (K_STEP * WAVES)
PROJECTION_TASKS = (PROJ_DIM + ROWS - 1) // ROWS
CONV_TASKS = CONV_DIM // ROWS
Z_TASKS = VALUE_DIM // ROWS
V_SPLITS = 2
RECURRENCE_TASKS = VALUE_HEADS * V_SPLITS
OUTPUT_TASKS = HIDDEN // ROWS
WEIGHT_BYTES = PROJ_DIM * HIDDEN * 2
OUTPUT_WEIGHT_BYTES = HIDDEN * VALUE_DIM * 2

PAIR_BYTES = 8
def align(value, alignment=16):
    return (value + alignment - 1) // alignment * alignment


K1_BASE = align(finish.scratch_bytes())
PROJECTION_READY_OFFSET = K1_BASE
RECURRENCE_READY_OFFSET = PROJECTION_READY_OFFSET + PAIR_BYTES * PROJECTION_TASKS
HEAD_READY_OFFSET = RECURRENCE_READY_OFFSET + PAIR_BYTES * RECURRENCE_TASKS
NORM_READY_OFFSET = HEAD_READY_OFFSET + PAIR_BYTES * VALUE_HEADS
READY_END = NORM_READY_OFFSET + PAIR_BYTES
PROJECTED_OFFSET = align(READY_END)
CORE_OFFSET = align(PROJECTED_OFFSET + PROJ_DIM * 2)
NORMED_OFFSET = align(CORE_OFFSET + VALUE_DIM * 2)
K1_END = align(NORMED_OFFSET + VALUE_DIM * 2)
K2_PULL_READY_OFFSET = align(K1_END, PAIR_BYTES)
K2_ALL_READY_OFFSET = K2_PULL_READY_OFFSET + PARTS * PAIR_BYTES
SCRATCH_BYTES = align(K2_ALL_READY_OFFSET + PAIR_BYTES)
LOG2E = 1.4426950408889634
NORM_EPS = 1e-6

assert BLOCKS == 256 and THREADS == 512 and WAVES == 8
assert (KEY_DIM, VALUE_DIM, CONV_DIM, PROJ_DIM) == (256, 768, 1280, 2060)
assert (PROJECTION_TASKS, CONV_TASKS, Z_TASKS) == (129, 80, 48)
assert (RECURRENCE_TASKS, OUTPUT_TASKS) == (12, 160)
assert K_STEPS_PER_WAVE * K_STEP * WAVES == HIDDEN
assert OUT_K_STEPS_PER_WAVE * K_STEP * WAVES == VALUE_DIM


@fx.struct
class GdnSmem:
    activation: fx.Array[fx.Int32, HIDDEN // 2, 16]
    reduction: fx.Array[fx.Float32, WAVES * 64 * 4, 16]


def _raw_tensor(address, numeric, alignment=16):
    pointer = fx.PointerType.get(
        numeric.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=alignment,
    )
    view = fx.make_view(
        fx.inttoptr(pointer, fx.Int64(address)),
        fx.make_layout((0x7FFFFFFF,), (1,)),
    )
    return fx.Tensor(view)


def _projection_views(
    projected, dt_bias, a_log, read_indices, write_indices, state, core
):
    bf16_bytes = fx.Int64(2)
    query = _raw_tensor(projected, fx.BFloat16)
    key = _raw_tensor(projected + bf16_bytes * fx.Int64(KEY_DIM), fx.BFloat16)
    value = _raw_tensor(projected + bf16_bytes * fx.Int64(2 * KEY_DIM), fx.BFloat16)
    b = _raw_tensor(projected + bf16_bytes * fx.Int64(B_ROW), fx.BFloat16)
    a = _raw_tensor(projected + bf16_bytes * fx.Int64(A_ROW), fx.BFloat16)
    return (
        query,
        key,
        value,
        a,
        b,
        _raw_tensor(dt_bias, fx.BFloat16),
        _raw_tensor(a_log, fx.BFloat16),
        _raw_tensor(read_indices, fx.Int32, 4),
        _raw_tensor(write_indices, fx.Int32, 4),
        _raw_tensor(state, fx.Float32),
        _raw_tensor(core, fx.BFloat16),
    )


def _ld_bf(ptr, index):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), index, vec_width=1, dtype=T.bf16))
    )


def _st_bf(ptr, index, value, cache_modifier=0):
    bo.buffer_store(
        fx.Float32(value).to(fx.BFloat16),
        rsrc(ptr),
        index,
        cache_modifier=cache_modifier,
    )


def _silu(value):
    return fx.Float32(value) / (
        fx.Float32(1.0) + hw_exp2(fx.Float32(value) * fx.Float32(-LOG2E))
    )


def _sigmoid(value):
    return fx.Float32(1.0) / (
        fx.Float32(1.0) + hw_exp2(fx.Float32(value) * fx.Float32(-LOG2E))
    )


def _bf16_round_strict(value):
    bits = fx.Float32(value).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    return fx.Float32((rounded >> 16).to(fx.Uint16).bitcast(fx.BFloat16))


@traced
def _conv_value(
    projected, channel, conv_weight, conv_state, read_slot, write_slot
):
    state_stride = (CONV_WIDTH - 1) * CONV_DIM
    read_base = read_slot * state_stride + channel
    write_base = write_slot * state_stride + channel
    old0 = _ld_bf(conv_state, read_base)
    old1 = _ld_bf(conv_state, read_base + CONV_DIM)
    old2 = _ld_bf(conv_state, read_base + 2 * CONV_DIM)
    base = channel * CONV_WIDTH
    # Native causal_conv1d_update multiplies two BF16 operands before adding
    # to its FP32 accumulator. Preserve that product rounding at each tap.
    acc = _bf16_round_strict(old0 * _ld_bf(conv_weight, base))
    acc = acc + _bf16_round_strict(old1 * _ld_bf(conv_weight, base + 1))
    acc = acc + _bf16_round_strict(old2 * _ld_bf(conv_weight, base + 2))
    acc = acc + _bf16_round_strict(projected * _ld_bf(conv_weight, base + 3))
    _st_bf(conv_state, write_base, old1)
    _st_bf(conv_state, write_base + CONV_DIM, old2)
    _st_bf(conv_state, write_base + 2 * CONV_DIM, projected)
    return bf16_round(_silu(acc))


@traced
def _project_task(
    x,
    weight,
    conv_weight,
    conv_state,
    read_slot,
    write_slot,
    projected,
    task,
    tid,
    lane,
    wave,
    activation,
    reduction,
):
    row = task * ROWS + lane % ROWS
    row_live = row < PROJ_DIM
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(K_STEPS_PER_WAVE):
        k0 = (wave + WAVES * k_pass) * K_STEP + 8 * (lane // ROWS)
        w = fx.Vector(
            bo.buffer_load(
                rsrc(weight, nbytes=WEIGHT_BYTES),
                (row * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                mask=row_live,
            )
        ).bitcast(fx.BFloat16)
        xv = fx.Vector(
            fx.ptr_load(activation + k0 // 2, result_type=T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(w, xv, acc)
    fx.ptr_store(acc, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()

    if task < CONV_TASKS:
        if tid < ROWS:
            channel = task * ROWS + tid
            value = _conv_value(
                bf16_round(row_sum(reduction, tid, 0)),
                channel,
                conv_weight,
                conv_state,
                read_slot,
                write_slot,
            )
            _st_bf(projected, channel, value, CM_DEV)
    elif tid < ROWS // 2:
        row0 = task * ROWS + 2 * tid
        value = bf16_pair(
            bf16_round(row_sum(reduction, 2 * tid, 0)),
            bf16_round(row_sum(reduction, 2 * tid + 1, 0)),
        )
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(projected, nbytes=PROJ_DIM * 2),
            task * (ROWS // 2) + tid,
            mask=row0 + 1 < PROJ_DIM,
            cache_modifier=CM_DEV,
        )
    gpu.barrier()


def _projection_dependencies(task):
    head = task // V_SPLITS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    q0 = key_head * (HEAD_DIM // ROWS)
    k0 = KEY_DIM // ROWS + q0
    v0 = 2 * KEY_DIM // ROWS + head * (HEAD_DIM // ROWS)
    deps = []
    for offset in range_constexpr(HEAD_DIM // ROWS):
        deps.extend((q0 + offset, k0 + offset, v0 + offset))
    deps.append(PROJECTION_TASKS - 1)
    return deps


@traced
def _wait_projection_dependencies(poll, ready, task, tid):
    if tid == 0:
        poll([(ready, dep, 1) for dep in _projection_dependencies(task)])
    gpu.barrier()


@traced
def _wait_head_dependencies(
    poll, recurrence_ready, projection_ready, head, tid
):
    if tid == 0:
        specs = [
            (recurrence_ready, head * V_SPLITS + split, 1)
            for split in range_constexpr(V_SPLITS)
        ]
        z0 = CONV_TASKS + head * (HEAD_DIM // ROWS)
        specs.extend(
            (projection_ready, z0 + offset, 1)
            for offset in range_constexpr(HEAD_DIM // ROWS)
        )
        poll(specs)
    gpu.barrier()


@traced
def _wait_all_heads(poll, head_ready, tid):
    if tid == 0:
        poll(
            [
                (head_ready, head, 1)
                for head in range_constexpr(VALUE_HEADS)
            ]
        )
    gpu.barrier()


@traced
def _wait_norm(poll, norm_ready, tid):
    if tid == 0:
        poll([(norm_ready, 0, 1)])
    gpu.barrier()


@traced
def _norm_heads(core, projected, norm_weight, normed, wave, lane):
    if wave < VALUE_HEADS:
        index0 = wave * HEAD_DIM + lane
        index1 = index0 + 64
        x0 = _ld_bf(core, index0)
        x1 = _ld_bf(core, index1)
        total = wave_sum(x0 * x0 + x1 * x1)
        rrms = hw_rsq(total * fx.Float32(1.0 / HEAD_DIM) + NORM_EPS)
        normalized0 = _bf16_round_strict(x0 * rrms)
        normalized1 = _bf16_round_strict(x1 * rrms)
        affine0 = _bf16_round_strict(normalized0 * _ld_bf(norm_weight, lane))
        affine1 = _bf16_round_strict(
            normalized1 * _ld_bf(norm_weight, lane + 64)
        )
        _st_bf(
            normed,
            index0,
            affine0 * _sigmoid(_ld_bf(projected, CONV_DIM + index0)),
            CM_DEV,
        )
        _st_bf(
            normed,
            index1,
            affine1 * _sigmoid(_ld_bf(projected, CONV_DIM + index1)),
            CM_DEV,
        )


def _output_weights(weight, tile, lane, wave):
    row = tile * ROWS + lane % ROWS
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(weight, nbytes=OUTPUT_WEIGHT_BYTES),
                (
                    row * VALUE_DIM
                    + (wave + WAVES * k_pass) * K_STEP
                    + 8 * (lane // ROWS)
                )
                // 2,
                vec_width=4,
                dtype=T.i32,
            )
        ).bitcast(fx.BFloat16)
        for k_pass in range_constexpr(OUT_K_STEPS_PER_WAVE)
    ]


@traced
def _output_task(
    normed,
    out,
    tile,
    tid,
    lane,
    wave,
    reduction,
    weight_ops,
    mailbox,
    peer_sym,
    peer_region_off,
    layer,
    tokens=1,
    token=0,
):
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(OUT_K_STEPS_PER_WAVE):
        k0 = (wave + WAVES * k_pass) * K_STEP + 8 * (lane // ROWS)
        value = fx.Vector(
            bo.buffer_load(
                rsrc(normed, nbytes=VALUE_DIM * 2),
                k0 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(weight_ops[k_pass], value, acc)
    fx.ptr_store(acc, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        value0 = bf16_round(row_sum(reduction, 2 * tid, 0))
        value1 = bf16_round(row_sum(reduction, 2 * tid + 1, 0))
        value = bf16_pair(value0, value1)
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(out, nbytes=HIDDEN * 2),
            tile * (ROWS // 2) + tid,
            cache_modifier=CM_DEV,
        )
        publish_pair(
            mailbox,
            peer_sym,
            peer_region_off,
            "attention",
            layer,
            tokens,
            token,
            tile * ROWS + 2 * tid,
            value0.to(fx.BFloat16),
            value1.to(fx.BFloat16),
        )


@traced
def _run_full_graph(
    emit_gdr_decode_task,
    lds,
    mailbox,
    peer_mailbox,
    scratch,
    args,
    bid,
):
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    projected = scratch + fx.Int64(PROJECTED_OFFSET)
    core = scratch + fx.Int64(CORE_OFFSET)
    normed = scratch + fx.Int64(NORMED_OFFSET)
    projection_ready = sreg(
        scratch, PROJECTION_READY_OFFSET, "tp8_gdn_projection_ready"
    )
    recurrence_ready = sreg(
        scratch, RECURRENCE_READY_OFFSET, "tp8_gdn_recurrence_ready"
    )
    head_ready = sreg(scratch, HEAD_READY_OFFSET, "tp8_gdn_head_ready")
    norm_ready = sreg(scratch, NORM_READY_OFFSET, "tp8_gdn_norm_ready")
    read_slot = fx.Int32(
        bo.buffer_load(
            rsrc(args["state_read"]), 0, vec_width=1, dtype=T.i32
        )
    )
    write_slot = fx.Int32(
        bo.buffer_load(
            rsrc(args["state_write"]), 0, vec_width=1, dtype=T.i32
        )
    )

    if tid < HIDDEN // 8:
        at = tid * 8
        value = fx.Vector(
            bo.buffer_load(
                rsrc(args["x"], nbytes=HIDDEN * 2),
                at // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        fx.ptr_store(value, lds.activation.ptr + at // 2)
    gpu.barrier()

    if bid < PROJECTION_TASKS:
        _project_task(
            args["x"],
            args["projection_weight"],
            args["conv_weight"],
            args["conv_state"],
            read_slot,
            write_slot,
            projected,
            bid,
            tid,
            lane,
            wave,
            lds.activation.ptr,
            lds.reduction.ptr,
        )
        publish(
            mailbox.put,
            projection_ready,
            bid,
            fx.Int32(1),
            tid == 0,
        )

    def run_recurrence():
        _wait_projection_dependencies(
            mailbox.poll, projection_ready, bid, tid
        )
        emit_gdr_decode_task(
            *_projection_views(
                projected,
                args["dt_bias"],
                args["a_log"],
                args["state_read"],
                args["state_write"],
                args["state"],
                core,
            ),
            fx.Int32(1),
            tid,
            bid,
        )
        publish(
            mailbox.put,
            recurrence_ready,
            bid,
            fx.Int32(1),
            tid == 0,
        )

    if bid < RECURRENCE_TASKS:
        run_recurrence()

    if bid < VALUE_HEADS:
        _wait_head_dependencies(
            mailbox.poll, recurrence_ready, projection_ready, bid, tid
        )
        publish(mailbox.put, head_ready, bid, fx.Int32(1), tid == 0)

    if bid == 0:
        _wait_all_heads(mailbox.poll, head_ready, tid)
        _norm_heads(core, projected, args["norm_weight"], normed, wave, lane)
        publish(mailbox.put, norm_ready, 0, fx.Int32(1), tid == 0)

    if bid < OUTPUT_TASKS:
        weight_ops = _output_weights(args["output_weight"], bid, lane, wave)
        _wait_norm(mailbox.poll, norm_ready, tid)
        _output_task(
            normed,
            args["out"],
            bid,
            tid,
            lane,
            wave,
            lds.reduction.ptr,
            weight_ops,
            peer_mailbox,
            args["peer_sym"],
            args["peer_region_off"],
            args["layer"],
            args.get("tokens", 1),
            args.get("token", 0),
        )



def build_gdn_k1(key: MonoBuild):
    """Build rank-local TP8 GDN and publish its attention partial."""

    if key.tokens not in (1, 2, 4) or key.tp != 8:
        raise ValueError("Qwen TP8 GDN K1 supports S=1/S=2/S=4 TP8")
    keyed = key_tuple(key, SOURCES)
    peer_region_off = peer_offsets(key.tokens, 8)["attention"]
    from atom.models.qwen4_exp_mono.state import LAYER_SCRATCH_BYTES

    q_stride = (KEY_DIM, KEY_DIM, HEAD_DIM, 1)
    v_stride = (VALUE_DIM, VALUE_DIM, HEAD_DIM, 1)
    gate_stride = (VALUE_HEADS, VALUE_HEADS, 1)
    state_stride = (
        VALUE_HEADS * HEAD_DIM * HEAD_DIM,
        HEAD_DIM * HEAD_DIM,
        HEAD_DIM,
        1,
    )

    def compose(
        *, module_name, emit_gdr_decode_task, block_threads, tasks_per_batch
    ):
        assert block_threads == THREADS
        assert tasks_per_batch == RECURRENCE_TASKS
        name = kernel_symbol("qwen_tp8_gdn_k1", s=key.tokens)

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def kernel(
            x: Int64,
            projection_weight: Int64,
            conv_weight: Int64,
            conv_state: Int64,
            state_read: Int64,
            state_write: Int64,
            a_log: Int64,
            dt_bias: Int64,
            state: Int64,
            norm_weight: Int64,
            output_weight: Int64,
            partial_out: Int64,
            peer_sym: Int64,
            scratch: Int64,
            layer: Int32,
        ):
            bid = fx.block_idx.x
            lds = fx.SharedAllocator().allocate(GdnSmem).peek()
            mailbox = Mailbox(layer)
            args = {
                "x": x,
                "projection_weight": projection_weight,
                "conv_weight": conv_weight,
                "conv_state": conv_state,
                "state_read": state_read,
                "state_write": state_write,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "state": state,
                "norm_weight": norm_weight,
                "output_weight": output_weight,
                "out": partial_out,
                "peer_sym": peer_sym,
                "peer_region_off": peer_region_off,
                "layer": layer,
            }
            for token in range_constexpr(key.tokens):
                token_args = dict(args)
                token_args["x"] = x + fx.Int64(token * HIDDEN * 2)
                token_args["state_read"] = state_read + fx.Int64(token * 4)
                token_args["state_write"] = state_write + fx.Int64(token * 4)
                # The public GDN seam is strided within the layer scratch.
                token_args["out"] = partial_out + fx.Int64(token * LAYER_SCRATCH_BYTES)
                token_args["tokens"] = key.tokens
                token_args["token"] = token
                _run_full_graph(
                    emit_gdr_decode_task, lds, mailbox, mailbox,
                    scratch + fx.Int64(token * LAYER_SCRATCH_BYTES),
                    token_args, bid,
                )
            _ = module_name

        @launcher(name)
        def launch(
            x: Int64,
            projection_weight: Int64,
            conv_weight: Int64,
            conv_state: Int64,
            state_read: Int64,
            state_write: Int64,
            a_log: Int64,
            dt_bias: Int64,
            state: Int64,
            norm_weight: Int64,
            output_weight: Int64,
            partial_out: Int64,
            peer_sym: Int64,
            scratch: Int64,
            layer: Int32,
            stream: fx.Stream = _CURRENT_STREAM,
        ):
            kernel(
                x,
                projection_weight,
                conv_weight,
                conv_state,
                state_read,
                state_write,
                a_log,
                dt_bias,
                state,
                norm_weight,
                output_weight,
                partial_out,
                peer_sym,
                scratch,
                layer,
                value_attrs=PERSISTENT_VALUE_ATTRS,
            ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

        K1_ABI.check(kernel, launch)
        _ = keyed, module_name
        return launch

    return create_vk_gdr_decode_kernel(
        "bf16",
        "bf16",
        "f32",
        1,
        KEY_HEADS,
        VALUE_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        q_stride,
        q_stride,
        v_stride,
        state_stride,
        gate_stride,
        gate_stride,
        True,
        NUM_BLOCKS_PER_V_DIM=V_SPLITS,
        NUM_WARPS=WAVES,
        WARP_THREADS_K=16,
        _composition=compose,
        _composition_output_cache_modifier=CM_DEV,
        _composition_round_beta_bf16=True,
    )



@traced
def _run_attention_pull(c, bid):
    """Pull each N32 attention part, then expose one local all-ready tag."""

    pull_ready = sreg(
        c["scratch"], K2_PULL_READY_OFFSET, "tp8_attention_pull_ready"
    )
    all_ready = sreg(
        c["scratch"], K2_ALL_READY_OFFSET, "tp8_attention_all_ready"
    )
    if bid < PARTS:
        pull_part_to_bf16(
            c["mailbox"],
            c["peers"],
            c["args"]["attn_out"],
            c["attention_region_off"],
            "attention",
            c["layer"],
            1,
            0,
            bid,
            8,
        )
        publish(c["put"], pull_ready, bid, fx.Int32(1), c["tid"] == 0)
    if bid == PARTS - 1:
        if c["tid"] == 0:
            c["poll"](
                [
                    (pull_ready, part, 1)
                    for part in range_constexpr(PARTS)
                ]
            )
        gpu.barrier()
        publish(c["put"], all_ready, 0, fx.Int32(1), c["tid"] == 0)
    c["attn_ready"] = all_ready


def build_gdn_k2(key: MonoBuild):
    """Build TP8 attention pull followed by the common HC+MoE finish."""

    if key.tokens != 1 or key.tp != 8:
        raise ValueError("Qwen TP8 GDN K2 supports only S=1 TP8")
    keyed = key_tuple(key, SOURCES)
    finish_layout = finish.scratch_layout()
    attention_region_off = peer_offsets(1, 8)["attention"]
    moe_region_off = peer_offsets(1, 8)["moe"]
    name = kernel_symbol("qwen_tp8_gdn_k2", s=1)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        peers: Int64,
        streams: Int64,
        attn_out: Int64,
        attn_raw: Int64,
        hc_norm: Int64,
        hc_down_inject: Int64,
        hc_up: Int64,
        router: Int64,
        shared_w13: Int64,
        shared_w2: Int64,
        routed_w13: Int64,
        routed_w13_scale: Int64,
        routed_w2: Int64,
        routed_w2_scale: Int64,
        expert_map: Int64,
        peer_sym: Int64,
        streams_out: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        moe_out: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(finish.FinishSmem).peek()
        mailbox = Mailbox(layer)
        args = {
            "streams": streams,
            "attn_out": attn_out,
            "attn_raw": attn_raw,
            "hc_norm": hc_norm,
            "hc_down_inject": hc_down_inject,
            "hc_up": hc_up,
            "router": router,
            "shared_w13": shared_w13,
            "shared_w2": shared_w2,
            "routed_w13": routed_w13,
            "routed_w13_scale": routed_w13_scale,
            "routed_w2": routed_w2,
            "routed_w2_scale": routed_w2_scale,
            "expert_map": expert_map,
            "streams_out": streams_out,
            "pending_out": pending_out,
            "pending_raw": pending_raw,
            "moe_out": moe_out,
        }
        c = finish.finish_context(lds, mailbox, finish_layout, scratch, args)
        c.update(
            {
                "expert_owned": True,
                "shared_intermediate": key.dims.shared_intermediate,
                "publish_partial": True,
                "peer_sym": peer_sym,
                "peer_region_off": moe_region_off,
                "peers": peers,
                "attention_region_off": attention_region_off,
                "attn_cm": CM_DEV,
                "layer": layer,
            }
        )
        _run_attention_pull(c, bid)
        finish.run_finish_tp8(c, bid)
        _ = keyed

    @launcher(name)
    def launch(
        peers: Int64,
        streams: Int64,
        attn_out: Int64,
        attn_raw: Int64,
        hc_norm: Int64,
        hc_down_inject: Int64,
        hc_up: Int64,
        router: Int64,
        shared_w13: Int64,
        shared_w2: Int64,
        routed_w13: Int64,
        routed_w13_scale: Int64,
        routed_w2: Int64,
        routed_w2_scale: Int64,
        expert_map: Int64,
        peer_sym: Int64,
        streams_out: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        moe_out: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = keyed
        kernel(
            peers,
            streams,
            attn_out,
            attn_raw,
            hc_norm,
            hc_down_inject,
            hc_up,
            router,
            shared_w13,
            shared_w2,
            routed_w13,
            routed_w13_scale,
            routed_w2,
            routed_w2_scale,
            expert_map,
            peer_sym,
            streams_out,
            pending_out,
            pending_raw,
            moe_out,
            scratch,
            layer,
            value_attrs=PERSISTENT_VALUE_ATTRS,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    K2_ABI.check(kernel, launch)
    return launch
