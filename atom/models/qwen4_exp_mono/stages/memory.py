# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Raw-address views and integer cache accesses for Qwen stages."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from atom.mono.device.ops import CM_DEV, CM_NT, rsrc


def load_i32(ptr, index, cache_modifier=CM_DEV):
    return fx.Int32(
        bo.buffer_load(
            rsrc(ptr), index, vec_width=1, dtype=T.i32, cache_modifier=cache_modifier
        )
    )


def load_i64(ptr, index, cache_modifier=CM_DEV):
    return fx.Int64(
        bo.buffer_load(
            rsrc(ptr), index, vec_width=1, dtype=T.i64, cache_modifier=cache_modifier
        )
    )


def store_i64(ptr, index, value, cache_modifier=CM_DEV):
    bo.buffer_store(fx.Int64(value), rsrc(ptr), index, cache_modifier=cache_modifier)


def raw_tensor(address, numeric, alignment=16):
    """Flat global tensor view used at the raw-address production ABI."""

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


def workspace(scratch, layout, name):
    """Address one named byte region; never alter the layout or cache policy."""
    return scratch + fx.Int64(layout[name][0])


def i64_pair(words, pair):
    return fx.Vector.from_elements(
        [words[2 * pair], words[2 * pair + 1]], fx.Int32
    ).bitcast(fx.Int64)[0]


def shuffled_fp8_tile(
    weight, expert, row_group, kb, lane, rows, width, cache_modifier=CM_NT
):
    """Load a native shuffled N16/K128 tile in its four MFMA operand slots."""
    base = expert * (rows * width // 4)
    tile = row_group * (width // 64) + 2 * kb
    lo = fx.Vector(
        bo.buffer_load(
            rsrc(weight),
            base + tile * 256 + lane * 4,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    hi = fx.Vector(
        bo.buffer_load(
            rsrc(weight),
            base + (tile + 1) * 256 + lane * 4,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    return [i64_pair(lo, 0), i64_pair(lo, 1), i64_pair(hi, 0), i64_pair(hi, 1)]


def load_f32(ptr, index, cache_modifier=CM_DEV):
    return fx.Float32(
        bo.buffer_load(
            rsrc(ptr), index, vec_width=1, dtype=T.f32, cache_modifier=cache_modifier
        )
    )


def store_bf16(ptr, index, value, cache_modifier=CM_DEV):
    """Store a BF16 operand without introducing a new FP32 cast boundary."""
    bo.buffer_store(fx.BFloat16(value), rsrc(ptr), index, cache_modifier=cache_modifier)
