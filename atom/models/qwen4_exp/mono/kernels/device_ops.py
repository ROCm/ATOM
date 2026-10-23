# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Common AMD expression primitives for fused model-layer kernels."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels.dpp_utils import update_dpp_i32

_UMAX_DPP_ASM = "\n".join(
    [
        "s_nop 1\nv_max_u32_dpp $0, $0, $0 " + control
        for control in (
            "row_shr:1 bound_ctrl:0",
            "row_shr:2 bound_ctrl:0",
            "row_shr:4 bound_ctrl:0",
            "row_shr:8 bound_ctrl:0",
            "row_bcast:15 row_mask:0xa",
            "row_bcast:31 row_mask:0xc",
        )
    ]
)


def rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def uniform(value):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(value).ir_value()))


def opaque(value):
    """Wave-uniform ``value`` through an ``s_mov`` the compiler cannot see
    through: the result is not reused from, nor computed before, another."""
    v = as_ir_value(fx.Int32(value))
    asm = llvm.InlineAsmOp(
        T.i32, [v], "s_mov_b32 $0, $1", "=s,s", has_side_effects=True
    )
    return fx.Int32(asm.result)


def spin_pause() -> None:
    """Keep tagged-mailbox polling loads inside the retry loop."""

    llvm.InlineAsmOp(None, [], "s_nop 0", "", has_side_effects=True)


def read_lane_i32(value, lane):
    """Read one i32 from ``lane`` while keeping IR conversion local."""

    return fx.Int32(rocdl.readlane(T.i32, fx.Int32(value), fx.Int32(lane)))


def bpermute_i32(byte_offset, value):
    """Read an i32 VGPR value from the lane selected by a byte offset."""

    return fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(byte_offset), fx.Int32(value)))


def _hardware_f32(name, value):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(value).ir_value()], [], [])
    )


def rsq(value):
    return _hardware_f32("llvm.amdgcn.rsq.f32", value)


def rcp(value):
    return _hardware_f32("llvm.amdgcn.rcp.f32", value)


def exp(value):
    return _hardware_f32("llvm.amdgcn.exp2.f32", fx.Float32(value) * 1.4426950408889634)


def xshfl(value, offset):
    """Return the value from ``lane ^ offset`` using VALU/DPP operations."""

    if offset >= 16:
        return value.shuffle_xor(offset, 64)
    is_float = isinstance(value, fx.Float32)
    source = value.bitcast(fx.Int32) if is_float else fx.Int32(value)
    if offset == 8:
        result = fx.Int32(update_dpp_i32(source, source, 0x118, 0xF, 0xC, False))
        result = fx.Int32(update_dpp_i32(result, source, 0x108, 0xF, 0x3, False))
    elif offset == 4:
        result = fx.Int32(update_dpp_i32(source, source, 0x114, 0xF, 0xA, False))
        result = fx.Int32(update_dpp_i32(result, source, 0x104, 0xF, 0x5, False))
    elif offset == 2:
        result = fx.Int32(update_dpp_i32(source, source, 0x4E, 0xF, 0xF, False))
    else:
        result = fx.Int32(update_dpp_i32(source, source, 0xB1, 0xF, 0xF, False))
    return result.bitcast(fx.Float32) if is_float else result


def wave_umax_dpp(value):
    """Return a wave maximum through the tuned GLM/TileRT DPP schedule."""

    result = llvm.InlineAsmOp(
        T.i32, [as_ir_value(fx.Int32(value))], _UMAX_DPP_ASM, "=v,0"
    ).result
    return read_lane_i32(result, 63)


def xred(value, offset, op):
    """Combine a value with ``lane ^ offset`` using a symmetric operation."""

    if offset < 16:
        return op(value, xshfl(value, offset))
    is_float = isinstance(value, fx.Float32)
    source = as_ir_value(value.bitcast(fx.Int32) if is_float else fx.Int32(value))
    swap = rocdl.permlane32_swap if offset == 32 else rocdl.permlane16_swap
    pair = swap(
        llvm.StructType.get_literal([T.i32, T.i32]), source, source, False, False
    )
    lhs, rhs = (fx.Int32(llvm.extractvalue(T.i32, pair, [index])) for index in range(2))
    if is_float:
        return op(lhs.bitcast(fx.Float32), rhs.bitcast(fx.Float32))
    return op(type(value)(lhs), type(value)(rhs))
