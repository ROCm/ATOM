# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen common HC+MoE finish, standalone or composed after attention."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64
from flydsl.expr import range_constexpr

from atom.models.qwen4_exp_mono.config import (
    HC,
    HC_RANK,
    HIDDEN,
    INTERMEDIATE,
    ROUTER_OUTPUTS,
    TOPK,
)
from atom.models.qwen4_exp_mono.kernels.finish_shape import (
    G1_QUEUE_WORDS,
    G1_TASKS,
    HC_DOWN_TASKS,
    HC_NOUT,
    HC_OUT_TASKS,
    MIDQ_GROUPS,
    MIDQ_WORDS,
    ROUTER_TASKS,
    XQ_GROUPS,
    XQ_WORDS,
)
from atom.models.qwen4_exp_mono.kernels.g1 import (
    run_g1,
    run_g1_s2,
    run_g1_s2_partition,
    run_g1_tp8,
    run_g1_tp8_s2,
    _stage_shared_g1_s2,
    G1_START,
    G1_COLS,
)
from atom.models.qwen4_exp_mono.kernels.g2 import (
    run_g2,
    run_g2_s2,
    run_g2_s2_partition,
)
from atom.models.qwen4_exp_mono.kernels.hc import (
    run_hc_finish,
    run_hc_finish_s2,
    run_hc_finish_s2_partition,
    _stage_up_group,
    HC_UP_GROUPS,
    UP_GROUP0,
)
from atom.models.qwen4_exp_mono.kernels.quant import (
    run_input_quant,
    run_input_quant_s2,
)
from atom.models.qwen4_exp_mono.kernels.router import run_router, run_router_s2
from atom.models.qwen4_exp_mono.peer_layout import peer_offsets
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.mono.device.ops import kernel_symbol, launcher, traced
from atom.mono.device.sync import Mailbox, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, PERSISTENT_VALUE_ATTRS, THREADS, WAVES, first_task
from atom.mono.plan.layout import PAIR_BYTES
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

ABI = KernelAbi(
    (
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
        "streams_out",
        "pending_out",
        "pending_raw",
        "moe_out",
        "scratch",
        "layer",
    )
)

TP8_ABI = KernelAbi(
    (
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


@fx.struct
class FinishSmem:
    red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]
    up: fx.Array[fx.Float32, 4 * 16, 16]
    x: fx.Array[fx.Int32, HIDDEN // 2, 16]
    queue: fx.Array[fx.Int32, 1, 16]


@fx.struct
class FinishS2Smem:
    red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]
    up: fx.Array[fx.Float32, 2 * 4 * 16, 16]
    x: fx.Array[fx.Int32, 2 * HIDDEN // 2, 16]
    queue: fx.Array[fx.Int32, 1, 16]


def _align(offset: int, alignment: int = 16) -> int:
    return (offset + alignment - 1) // alignment * alignment


def hc_scratch_layout() -> dict[str, tuple[int, int]]:
    """HC mailbox and numerical hand-offs, in bytes."""

    out: dict[str, tuple[int, int]] = {}
    off = 0
    for name, pairs in (
        ("hc_norm_ready", HC),
        ("hc_down_ready", HC_DOWN_TASKS),
        ("hc_gate_ready", 1),
        ("hc_up_ready", HC_OUT_TASKS),
    ):
        off = _align(off, PAIR_BYTES)
        size = pairs * PAIR_BYTES
        out[name] = (off, size)
        off += size
    for name, size in (
        ("hc_normed", HC * HIDDEN * 2),
        ("hc_part", HC * HC_NOUT * 4),
        ("hc_gate", HC_RANK * 2),
    ):
        off = _align(off)
        out[name] = (off, size)
        off += size
    return out


def hc_scratch_bytes() -> int:
    layout = hc_scratch_layout()
    return _align(max(off + size for off, size in layout.values()))


def scratch_layout() -> dict[str, tuple[int, int]]:
    """All global hand-offs owned by the HC+MoE body."""

    out = hc_scratch_layout()
    off = hc_scratch_bytes()
    for name, pairs in (
        ("up_group_ready", 2),
        ("xq_ready", XQ_GROUPS),
        ("router_tile_ready", ROUTER_TASKS),
        ("route_ready", 1),
        ("g1_ready", G1_TASKS),
        ("midq_ready", TOPK * MIDQ_GROUPS),
        ("shared_mid_ready", 1),
    ):
        off = _align(off, PAIR_BYTES)
        size = pairs * PAIR_BYTES
        out[name] = (off, size)
        off += size
    for name, size in (
        ("xq", XQ_WORDS * 4),
        ("xq_scale", XQ_GROUPS * 4),
        ("router_logits", ROUTER_OUTPUTS * 2),
        ("route_ids", TOPK * 4),
        ("route_weights", TOPK * 4),
        ("shared_gate", 4),
        ("g1_queue", G1_QUEUE_WORDS * 4),
        ("routed_mid", TOPK * INTERMEDIATE * 2),
        ("shared_mid", INTERMEDIATE * 2),
        ("midq", MIDQ_WORDS * 4),
        ("midq_scale", TOPK * MIDQ_GROUPS * 4),
    ):
        off = _align(off)
        out[name] = (off, size)
        off += size
    return out


def scratch_bytes() -> int:
    layout = scratch_layout()
    return _align(max(off + size for off, size in layout.values()))


def finish_args(*values) -> dict:
    """Numerical kernel arguments by name, excluding the runtime tail."""

    names = ABI.names[: ABI.names.index("scratch")]
    assert len(values) == len(names)
    return dict(zip(names, values))


def finish_context(lds, mailbox, layout, scratch, args):
    """Stage context shared by the standalone and composed kernels."""

    tid = fx.thread_idx.x
    c = {
        "tid": tid,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "up": lds.up.ptr,
        "x": lds.x.ptr,
        "queue": lds.queue.ptr,
        "mailbox": mailbox,
        "put": mailbox.put,
        "poll": mailbox.poll,
        "tag": mailbox.tag,
        "scratch": scratch,
        "layout": layout,
        "args": args,
    }
    for name, (off, _) in layout.items():
        key = name.removeprefix("hc_")
        c[key] = (
            sreg(scratch, off, name)
            if name.endswith("_ready")
            else scratch + fx.Int64(off)
        )
    return c


def run_finish(c, bid):
    """Run the post-attention HC seam, quantization, routing, G1 and G2."""

    run_hc_finish(c, bid)
    run_input_quant(c, bid)
    run_router(c, bid)
    run_g1(c, bid)
    run_g2(c, bid)


def run_finish_tp8(c, bid):
    """Run S=1 TP8 finish and publish this rank's MoE partial."""

    run_hc_finish(c, bid)
    run_input_quant(c, bid)
    run_router(c, bid)
    run_g1_tp8(c, bid)
    run_g2(c, bid)


def run_finish_s2(c0, c1, bid):
    """Run S=2 by sharing dense weights and balancing routed work together."""

    run_hc_finish_s2(c0, c1, bid)
    run_input_quant_s2(c0, c1, bid)
    run_router_s2(c0, c1, bid)
    run_g1_s2(c0, c1, bid)
    run_g2_s2(c0, c1, bid)


@traced
def run_finish_s4(c0, c1, c2, c3, bid):
    """Run two independent S=2 pairs concurrently on half grids."""

    pair_blocks = BLOCKS // 2
    if bid < pair_blocks:
        local = bid
        run_hc_finish_s2_partition(c0, c1, local)
        run_input_quant_s2(c0, c1, local)
        run_router_s2(c0, c1, local)
        run_g1_s2_partition(c0, c1, local)
        run_g2_s2_partition(c0, c1, local)
    else:
        local = bid - pair_blocks
        run_hc_finish_s2_partition(c2, c3, local)
        run_input_quant_s2(c2, c3, local)
        run_router_s2(c2, c3, local)
        run_g1_s2_partition(c2, c3, local)
        run_g2_s2_partition(c2, c3, local)


@traced
def run_finish_tp8_hc_pair(c0, c1, bid, paired_g1=False, static_g1=False, shared_only=False):
    """Share HC/router weights; retain independent expert-owned G1/G2."""
    run_hc_finish_s2(c0, c1, bid)
    task = first_task(bid, UP_GROUP0)
    if task < HC_UP_GROUPS:
        _stage_up_group(c0, task)
        _stage_up_group(c1, task)
    run_input_quant_s2(c0, c1, bid)
    run_router_s2(c0, c1, bid)
    if shared_only:
        owner = (bid - G1_START + BLOCKS) % BLOCKS
        if owner < c0["shared_intermediate"] // G1_COLS:
            _stage_shared_g1_s2(c0, c1, owner)
        run_g1_tp8(c0, bid, skip_shared=True)
        run_g2(c0, bid)
        run_g1_tp8(c1, bid, skip_shared=True)
        run_g2(c1, bid)
    elif paired_g1:
        run_g1_tp8_s2(c0, c1, bid, static_g1)
        run_g2_s2(c0, c1, bid)
    else:
        run_g1_tp8(c0, bid)
        run_g2(c0, bid)
        run_g1_tp8(c1, bid)
        run_g2(c1, bid)


def _build_finish_tp8(key: MonoBuild, *, paired_hc=False, paired_g1=False, static_g1=False, shared_only=False, paired_shared_g2=False):
    if key.tokens not in (1, 2, 4):
        raise ValueError("Qwen TP8 finish supports only S=1/S=2/S=4")
    if paired_hc and key.tokens != 2:
        raise ValueError("paired TP8 HC probe requires S=2")
    if paired_g1 and not paired_hc:
        raise ValueError("paired TP8 G1 requires paired HC")
    if paired_shared_g2 and not (paired_hc and shared_only):
        raise ValueError("paired shared G2 requires paired HC and shared G1")
    keyed = key_tuple(key, SOURCES)
    kind = tuple(type(key.kind)).index(key.kind)
    layout = scratch_layout()
    region_off = peer_offsets(key.tokens, 8)["moe"]
    name = kernel_symbol(
        "qwen_finish_tp8_shared_g2_pair" if paired_shared_g2 else
        "qwen_finish_tp8_shared_pair" if shared_only else
        "qwen_finish_tp8_g1_static" if static_g1 else
        "qwen_finish_tp8_g1_pair" if paired_g1 else
        "qwen_finish_tp8_hc_pair" if paired_hc else "qwen_finish_tp8",
        s=key.tokens, k=kind,
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
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
        lds = fx.SharedAllocator().allocate(
            FinishS2Smem if paired_hc else FinishSmem
        ).peek()
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
        def token_context(token):
            token_args = dict(args)
            for field, columns in (
                ("streams", HC * HIDDEN), ("attn_out", HIDDEN),
                ("attn_raw", HC), ("streams_out", HC * HIDDEN),
                ("pending_out", HIDDEN), ("pending_raw", HC),
                ("moe_out", HIDDEN),
            ):
                token_args[field] = args[field] + fx.Int64(token * columns * 2)
            c = finish_context(
                lds, mailbox, layout,
                scratch + fx.Int64(token * scratch_bytes()), token_args,
            )
            c.update(
                {
                    "expert_owned": True,
                    "shared_intermediate": key.dims.shared_intermediate,
                    "publish_partial": True,
                    "peer_sym": peer_sym,
                    "peer_region_off": region_off,
                    "layer": layer,
                    "tokens": key.tokens,
                    "token": token,
                }
            )
            if paired_shared_g2:
                c.update({
                    "paired_shared_g2": token == 0,
                    "preloaded_shared_g2": token == 1,
                    "paired_shared_scratch": scratch + fx.Int64(scratch_bytes()),
                    "paired_shared_ready": sreg(
                        scratch + fx.Int64(scratch_bytes()),
                        layout["shared_mid_ready"][0], "shared_mid_ready"),
                    "paired_shared_output": moe_out + fx.Int64(HIDDEN * 2),
                })
            return c
        if paired_hc:
            run_finish_tp8_hc_pair(
                token_context(0), token_context(1), fx.block_idx.x,
                paired_g1, static_g1, shared_only
            )
        else:
            for token in range_constexpr(key.tokens):
                run_finish_tp8(token_context(token), fx.block_idx.x)
        _ = keyed

    @launcher(name)
    def launch(
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

    TP8_ABI.check(kernel, launch)
    return launch


def build_finish(key: MonoBuild):
    """Build the standalone finish used to validate width specializations."""

    if key.tp == 8:
        return _build_finish_tp8(key)

    if key.tokens not in (1, 2, 4):
        raise ValueError("Qwen finish currently has S=1/S=2/S=4 specializations")
    keyed = key_tuple(key, SOURCES)
    kind = tuple(type(key.kind)).index(key.kind)
    layout = scratch_layout()
    name = kernel_symbol("qwen_finish", s=key.tokens, k=kind)
    smem = FinishSmem if key.tokens == 1 else FinishS2Smem

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
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
        streams_out: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        moe_out: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        lds = fx.SharedAllocator().allocate(smem).peek()
        mailbox = Mailbox(layer)

        def token_context(token):
            token_scratch = scratch + fx.Int64(token * scratch_bytes())
            args = finish_args(
                streams + fx.Int64(token * HC * HIDDEN * 2),
                attn_out + fx.Int64(token * HIDDEN * 2),
                attn_raw + fx.Int64(token * HC * 2),
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
                streams_out + fx.Int64(token * HC * HIDDEN * 2),
                pending_out + fx.Int64(token * HIDDEN * 2),
                pending_raw + fx.Int64(token * HC * 2),
                moe_out + fx.Int64(token * HIDDEN * 2),
            )
            return finish_context(lds, mailbox, layout, token_scratch, args)

        contexts = tuple(token_context(token) for token in range(key.tokens))
        c0 = contexts[0]
        if key.tokens == 1:
            run_finish(c0, fx.block_idx.x)
        elif key.tokens == 2:
            run_finish_s2(c0, contexts[1], fx.block_idx.x)
        else:
            run_finish_s4(
                contexts[0],
                contexts[1],
                contexts[2],
                contexts[3],
                fx.block_idx.x,
            )
        _ = keyed

    @launcher(name)
    def launch(
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
            streams_out,
            pending_out,
            pending_raw,
            moe_out,
            scratch,
            layer,
            value_attrs=PERSISTENT_VALUE_ATTRS,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kernel, launch)
    return launch
