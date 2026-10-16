# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen GDN attention and HC+MoE finish composed into one layer kernel."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64

from atom.models.qwen4_exp_mono.config import (
    ALIGN,
    GDN_PROJ_DIM,
    GDN_VALUE_DIM,
    HC,
    HIDDEN,
)
from atom.models.qwen4_exp_mono.kernels.finish import (
    FinishS2Smem,
    FinishSmem,
    finish_args,
    finish_context,
    run_finish,
    run_finish_s2,
    run_finish_s4,
)
from atom.models.qwen4_exp_mono.kernels.finish import (
    scratch_bytes as finish_scratch_bytes,
)
from atom.models.qwen4_exp_mono.kernels.finish import (
    scratch_layout as finish_scratch_layout,
)
from atom.models.qwen4_exp_mono.kernels.gdn import (
    GDR_FULL_OUTPUT_GROUP_READY_OFFSET,
    GDR_FULL_READY_BYTES,
    GDR_OUT_EARLY_TASKS,
    GDR_OUT_PARTIAL_TASKS,
    ROWS,
    GdnS2Smem,
    GdnS4Smem,
    GdnSmem,
    compose_gdn,
    gdn_args,
)
from atom.models.qwen4_exp_mono.kernels.hc import (
    NORM_COLS_PER_PASS,
    NORM_EARLY_PASSES,
    combine_gdn_output_pair,
    combine_gdn_output_pair_raw,
)
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.mono.device.ops import kernel_symbol, launcher
from atom.mono.device.sync import Mailbox, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, PERSISTENT_VALUE_ATTRS, THREADS
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

assert GDR_OUT_EARLY_TASKS * ROWS == NORM_EARLY_PASSES * NORM_COLS_PER_PASS

ABI = KernelAbi(
    (
        "streams",
        "gdn_input",
        "attn_raw",
        "gdn_qkvzba",
        "gdn_conv",
        "gdn_a_log",
        "gdn_dt_bias",
        "gdn_norm",
        "gdn_out",
        "gdn_conv_state",
        "gdn_recurrent_state",
        "gdn_state_read",
        "gdn_state_write",
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


@fx.union
class LayerSmem:
    gdn: GdnSmem
    finish: FinishSmem


@fx.union
class LayerS2Smem:
    gdn: GdnS2Smem
    finish: FinishS2Smem


@fx.union
class LayerS4Smem:
    gdn: GdnS4Smem
    finish: FinishS2Smem


def _align(offset: int, alignment: int = ALIGN) -> int:
    return (offset + alignment - 1) // alignment * alignment


def scratch_layout() -> dict[str, tuple[int, int]]:
    """Disjoint global hand-offs shared by the two halves of the layer."""

    out: dict[str, tuple[int, int]] = {}
    offset = _align(finish_scratch_bytes())
    for name, size in (
        ("gdn_ready", GDR_FULL_READY_BYTES),
        ("gdn_projected", GDN_PROJ_DIM * 2),
        ("gdn_core", GDN_VALUE_DIM * 2),
        ("gdn_normed", GDN_VALUE_DIM * 2),
        ("gdn_out", HIDDEN * 2),
        ("gdn_partials", GDR_OUT_PARTIAL_TASKS * 16 * 4),
    ):
        offset = _align(offset)
        out[name] = (offset, size)
        offset += size
    return out


def scratch_bytes() -> int:
    layout = scratch_layout()
    return _align(max(off + size for off, size in layout.values()))


def _workspace(scratch, layout, name):
    return scratch + fx.Int64(layout[name][0])


def build_gdn_layer(key: MonoBuild):
    """Build TP1 GDN -> HC -> MoE for the accepted narrow widths."""

    if key.tokens not in (1, 2, 4):
        raise ValueError("Qwen GDN layer currently has S=1/S=2/S=4 specializations")
    keyed = key_tuple(key, SOURCES)
    kind = tuple(type(key.kind)).index(key.kind)
    finish_layout = finish_scratch_layout()
    layer_layout = scratch_layout()
    name = kernel_symbol("qwen_gdn_layer", s=key.tokens, k=kind)
    smem = (
        LayerSmem
        if key.tokens == 1
        else LayerS2Smem
        if key.tokens == 2
        else LayerS4Smem
    )

    def composition(
        *, module_name, emit_gdr_decode_task, run_gdn, run_gdn_s2, run_gdn_s4
    ):
        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def kernel(
            streams: Int64,
            gdn_input: Int64,
            attn_raw: Int64,
            gdn_qkvzba: Int64,
            gdn_conv: Int64,
            gdn_a_log: Int64,
            gdn_dt_bias: Int64,
            gdn_norm: Int64,
            gdn_out: Int64,
            gdn_conv_state: Int64,
            gdn_recurrent_state: Int64,
            gdn_state_read: Int64,
            gdn_state_write: Int64,
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
            bid = fx.block_idx.x
            halves = fx.SharedAllocator().allocate(smem)
            mailbox = Mailbox(layer)

            def gdn_token_context(token):
                token_scratch = scratch + fx.Int64(token * scratch_bytes())
                ready = _workspace(token_scratch, layer_layout, "gdn_ready")
                output = _workspace(token_scratch, layer_layout, "gdn_out")
                gargs = gdn_args(
                    gdn_input + fx.Int64(token * HIDDEN * 2),
                    gdn_qkvzba,
                    gdn_conv,
                    gdn_conv_state,
                    gdn_state_read + fx.Int64(token * 4),
                    gdn_state_write + fx.Int64(token * 4),
                    gdn_a_log,
                    gdn_dt_bias,
                    gdn_recurrent_state,
                    _workspace(token_scratch, layer_layout, "gdn_projected"),
                    _workspace(token_scratch, layer_layout, "gdn_core"),
                    gdn_norm,
                    gdn_out,
                    _workspace(token_scratch, layer_layout, "gdn_normed"),
                    output,
                    _workspace(token_scratch, layer_layout, "gdn_partials"),
                )
                return ready, gargs

            def finish_token_context(token, ready):
                token_scratch = scratch + fx.Int64(token * scratch_bytes())
                output = _workspace(token_scratch, layer_layout, "gdn_out")
                args = finish_args(
                    streams + fx.Int64(token * HC * HIDDEN * 2),
                    output,
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
                c = finish_context(
                    halves.finish.peek(),
                    mailbox,
                    finish_layout,
                    token_scratch,
                    args,
                )
                c["attn_group_ready"] = sreg(
                    ready,
                    GDR_FULL_OUTPUT_GROUP_READY_OFFSET,
                    "gdn_full_output_group_ready",
                )
                c["attn_precombined"] = True
                return c

            def token_context(token):
                ready, gargs = gdn_token_context(token)
                return finish_token_context(token, ready), ready, gargs

            if key.tokens == 1:
                c0, ready0, gargs0 = token_context(0)
                run_gdn(
                    emit_gdr_decode_task,
                    halves.gdn.peek(),
                    mailbox,
                    ready0,
                    gargs0,
                    bid,
                    True,
                    lambda tile, row0, value0, value1: combine_gdn_output_pair(
                        c0, tile, row0, value0, value1
                    ),
                )
                gpu.barrier()
                run_finish(c0, bid)
            elif key.tokens == 2:
                c0, ready0, gargs0 = token_context(0)
                c1, ready1, gargs1 = token_context(1)
                run_gdn_s2(
                    emit_gdr_decode_task,
                    halves.gdn.peek(),
                    mailbox,
                    ready0,
                    ready1,
                    gargs0,
                    gargs1,
                    bid,
                    True,
                    lambda tile, row0, value0, value1: combine_gdn_output_pair(
                        c0, tile, row0, value0, value1
                    ),
                    lambda tile, row0, value0, value1: combine_gdn_output_pair(
                        c1, tile, row0, value0, value1
                    ),
                )
                gpu.barrier()
                run_finish_s2(c0, c1, bid)
            else:
                ready0, gargs0 = gdn_token_context(0)
                ready1, gargs1 = gdn_token_context(1)
                ready2, gargs2 = gdn_token_context(2)
                ready3, gargs3 = gdn_token_context(3)
                run_gdn_s4(
                    emit_gdr_decode_task,
                    halves.gdn.peek(),
                    mailbox,
                    ready0,
                    ready1,
                    ready2,
                    ready3,
                    gargs0,
                    gargs1,
                    gargs2,
                    gargs3,
                    bid,
                    True,
                    lambda tile, row0, value0, value1: combine_gdn_output_pair_raw(
                        streams,
                        attn_raw,
                        streams_out,
                        tile,
                        row0,
                        value0,
                        value1,
                    ),
                    lambda tile, row0, value0, value1: combine_gdn_output_pair_raw(
                        streams + fx.Int64(HC * HIDDEN * 2),
                        attn_raw + fx.Int64(HC * 2),
                        streams_out + fx.Int64(HC * HIDDEN * 2),
                        tile,
                        row0,
                        value0,
                        value1,
                    ),
                    lambda tile, row0, value0, value1: combine_gdn_output_pair_raw(
                        streams + fx.Int64(2 * HC * HIDDEN * 2),
                        attn_raw + fx.Int64(2 * HC * 2),
                        streams_out + fx.Int64(2 * HC * HIDDEN * 2),
                        tile,
                        row0,
                        value0,
                        value1,
                    ),
                    lambda tile, row0, value0, value1: combine_gdn_output_pair_raw(
                        streams + fx.Int64(3 * HC * HIDDEN * 2),
                        attn_raw + fx.Int64(3 * HC * 2),
                        streams_out + fx.Int64(3 * HC * HIDDEN * 2),
                        tile,
                        row0,
                        value0,
                        value1,
                    ),
                )
                gpu.barrier()
                c0 = finish_token_context(0, ready0)
                c1 = finish_token_context(1, ready1)
                c2 = finish_token_context(2, ready2)
                c3 = finish_token_context(3, ready3)
                run_finish_s4(c0, c1, c2, c3, bid)
            _ = keyed, module_name

        @launcher(name)
        def launch(
            streams: Int64,
            gdn_input: Int64,
            attn_raw: Int64,
            gdn_qkvzba: Int64,
            gdn_conv: Int64,
            gdn_a_log: Int64,
            gdn_dt_bias: Int64,
            gdn_norm: Int64,
            gdn_out: Int64,
            gdn_conv_state: Int64,
            gdn_recurrent_state: Int64,
            gdn_state_read: Int64,
            gdn_state_write: Int64,
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
                gdn_input,
                attn_raw,
                gdn_qkvzba,
                gdn_conv,
                gdn_a_log,
                gdn_dt_bias,
                gdn_norm,
                gdn_out,
                gdn_conv_state,
                gdn_recurrent_state,
                gdn_state_read,
                gdn_state_write,
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

    return compose_gdn(composition)
