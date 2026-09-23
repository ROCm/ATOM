# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL building blocks and the qualified V4 FP8 sparse-prefill kernel.

The fused H=128 kernel is production-routed only for the conservative envelope
selected in ``paged_prefill._use_flydsl_native_fp8_prefill``. The remaining
builders and configuration switches are retained for benchmarking and tuning.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import T

from aiter import dtypes
from aiter.ops.flydsl.kernels.tensor_shim import (
    GTensor,
    _run_compiled,
    ptr_arg,
    ptr_buf_tensor,
)

V4_DIM_NOPE = 448
V4_DIM_ROPE = 64
V4_DIM_QK = 512
V4_PACKED_ROW_BYTES = 512
V4_NUM_MX_GROUPS = 7
V4_MX_GROUP_SIZE = 64
V4_SCALE_OFFSET_BYTES = 448
LOG2E = 1.4426950408889634

V_STAGE_ROWS = 32
V_STAGE_THREADS_PER_ROW = 16
V_STAGE_BLOCK_THREADS = V_STAGE_ROWS * V_STAGE_THREADS_PER_ROW

QK_BLOCK_ROWS = 128
QK_BLOCK_COLS = 32
QK_WAVES = 8
PREFILL_D_SPLITS = 1
PREFILL_D_PER_CTA = V4_DIM_QK // PREFILL_D_SPLITS


def _decode_fp8x4_to_bf16_words(raw_word, scale):
    """Decode one packed FP8 dword to two packed BF16 dwords."""
    lo = fx.Vector(
        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
            T.vec(2, T.bf16), raw_word.ir_value(), scale.ir_value(), False
        )
    ).bitcast(fx.Int32)[0]
    hi = fx.Vector(
        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
            T.vec(2, T.bf16), raw_word.ir_value(), scale.ir_value(), True
        )
    ).bitcast(fx.Int32)[0]
    return lo, hi


def _pack_opsel_scales(scale_words, lane_group):
    """Transpose four row-major scale dwords into one MFMA opsel dword.

    The packed V4 row stores duplicated scale bytes as
    ``[s0,s0,s1,s1, s2,s2,s3,s3, ...]``.  A scaled K128 MFMA lane group needs
    one byte from each of four consecutive K128 steps, selected by ``opsel``.
    """
    shift = lane_group * fx.Int32(8)
    packed = fx.Int32(0)
    for step in fx.range_constexpr(4):
        byte = (fx.Uint32(scale_words[step]) >> fx.Uint32(shift)) & fx.Uint32(0xFF)
        packed = packed | (fx.Int32(byte) << fx.Int32(step * 8))
    return packed


def _load_permuted_opsel_scales(scale_stage, scale_base, lane_group):
    """Load and pack four scale bytes with short-lived source pairs."""
    byte = lane_group
    pair_selector = byte | ((byte + fx.Int32(4)) << fx.Int32(8))
    scale0 = fx.Int32(scale_stage[scale_base])
    scale1 = fx.Int32(scale_stage[scale_base + fx.Int32(1)])
    pair01 = fx.Int32(
        fx.rocdl.perm_b32(
            scale1.ir_value(),
            scale0.ir_value(),
            pair_selector.ir_value(),
        )
    )
    scale2 = fx.Int32(scale_stage[scale_base + fx.Int32(2)])
    scale3 = fx.Int32(scale_stage[scale_base + fx.Int32(3)])
    pair23 = fx.Int32(
        fx.rocdl.perm_b32(
            scale3.ir_value(),
            scale2.ir_value(),
            pair_selector.ir_value(),
        )
    )
    return (pair01 & fx.Int32(0xFFFF)) | (pair23 << fx.Int32(16))


def _join_i32x4(lo, hi):
    return fx.Vector(lo).shuffle(fx.Vector(hi), list(range(8))).ir_value()


def _exp2_f32(value):
    return fx.Float32(fx.rocdl.exp2(T.f32, value.ir_value()))


def _ds_read_b64_tr_b16_start(address, d_offset_bytes: int, next_k_offset_bytes: int):
    """Start one two-read gfx950 BF16 transpose fragment."""
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    result_type = ir.Type.parse("!llvm.struct<(vector<2xi32>, vector<2xi32>)>")
    result = llvm.inline_asm(
        result_type,
        [fx.Int32(address).ir_value()],
        (
            f"ds_read_b64_tr_b16 $0, $2 offset:{d_offset_bytes};\n"
            "ds_read_b64_tr_b16 $1, $2 "
            f"offset:{d_offset_bytes + next_k_offset_bytes};\n"
        ),
        "=&v,=&v,v,~{memory}",
        has_side_effects=True,
    )
    return (
        llvm.extractvalue(raw_type, result, [0]),
        llvm.extractvalue(raw_type, result, [1]),
    )


def _ds_read_b64_tr_b16_advance(
    address,
    d_offset_bytes: int,
    next_k_offset_bytes: int,
    raw_lo,
    raw_hi,
):
    """Issue the next transpose fragment and publish the previous one."""
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    result_type = ir.Type.parse(
        "!llvm.struct<(vector<2xi32>, vector<2xi32>, " "vector<2xi32>, vector<2xi32>)>"
    )
    result = llvm.inline_asm(
        result_type,
        [fx.Int32(address).ir_value(), raw_lo, raw_hi],
        (
            f"ds_read_b64_tr_b16 $0, $4 offset:{d_offset_bytes};\n"
            "ds_read_b64_tr_b16 $1, $4 "
            f"offset:{d_offset_bytes + next_k_offset_bytes};\n"
            "s_waitcnt lgkmcnt(2);\n"
        ),
        "=&v,=&v,=v,=v,v,2,3,~{memory}",
        has_side_effects=True,
    )
    return (
        llvm.extractvalue(raw_type, result, [2]),
        llvm.extractvalue(raw_type, result, [3]),
        llvm.extractvalue(raw_type, result, [0]),
        llvm.extractvalue(raw_type, result, [1]),
    )


def _ds_read_b64_tr_b16_finish(raw_lo, raw_hi):
    """Wait for and publish the final outstanding transpose fragment."""
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    result_type = ir.Type.parse("!llvm.struct<(vector<2xi32>, vector<2xi32>)>")
    result = llvm.inline_asm(
        result_type,
        [raw_lo, raw_hi],
        "s_waitcnt lgkmcnt(0);\n",
        "=v,=v,0,1,~{memory}",
        has_side_effects=True,
    )
    return (
        llvm.extractvalue(raw_type, result, [0]),
        llvm.extractvalue(raw_type, result, [1]),
    )


def _ds_read_b64_tr_b16_pair_start(
    address, d_offset_bytes: int, next_k_offset_bytes: int
):
    """Start two adjacent D16 transpose fragments with four LDS reads."""
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    result_type = ir.Type.parse(
        "!llvm.struct<(vector<2xi32>, vector<2xi32>, " "vector<2xi32>, vector<2xi32>)>"
    )
    result = llvm.inline_asm(
        result_type,
        [fx.Int32(address).ir_value()],
        (
            f"ds_read_b64_tr_b16 $0, $4 offset:{d_offset_bytes};\n"
            "ds_read_b64_tr_b16 $1, $4 "
            f"offset:{d_offset_bytes + next_k_offset_bytes};\n"
            f"ds_read_b64_tr_b16 $2, $4 offset:{d_offset_bytes + 32};\n"
            "ds_read_b64_tr_b16 $3, $4 "
            f"offset:{d_offset_bytes + next_k_offset_bytes + 32};\n"
        ),
        "=&v,=&v,=&v,=&v,v,~{memory}",
        has_side_effects=True,
    )
    return tuple(llvm.extractvalue(raw_type, result, [i]) for i in range(4))


def _ds_read_b64_tr_b16_pair_advance(
    address,
    d_offset_bytes: int,
    next_k_offset_bytes: int,
    raw_values,
):
    """Issue the next D32 pair and publish the previous four LDS reads."""
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    result_type = ir.Type.parse(
        "!llvm.struct<(vector<2xi32>, vector<2xi32>, vector<2xi32>, "
        "vector<2xi32>, vector<2xi32>, vector<2xi32>, vector<2xi32>, "
        "vector<2xi32>)>"
    )
    result = llvm.inline_asm(
        result_type,
        [fx.Int32(address).ir_value(), *raw_values],
        (
            f"ds_read_b64_tr_b16 $0, $8 offset:{d_offset_bytes};\n"
            "ds_read_b64_tr_b16 $1, $8 "
            f"offset:{d_offset_bytes + next_k_offset_bytes};\n"
            f"ds_read_b64_tr_b16 $2, $8 offset:{d_offset_bytes + 32};\n"
            "ds_read_b64_tr_b16 $3, $8 "
            f"offset:{d_offset_bytes + next_k_offset_bytes + 32};\n"
            "s_waitcnt lgkmcnt(4);\n"
        ),
        "=&v,=&v,=&v,=&v,=v,=v,=v,=v,v,4,5,6,7,~{memory}",
        has_side_effects=True,
    )
    ready = tuple(llvm.extractvalue(raw_type, result, [i]) for i in range(4, 8))
    upcoming = tuple(llvm.extractvalue(raw_type, result, [i]) for i in range(4))
    return ready, upcoming


def _ds_read_b64_tr_b16_pair_wait(raw_values, wait_count: int):
    """Wait for and publish a D32 transpose-read pair."""
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    result_type = ir.Type.parse(
        "!llvm.struct<(vector<2xi32>, vector<2xi32>, " "vector<2xi32>, vector<2xi32>)>"
    )
    result = llvm.inline_asm(
        result_type,
        list(raw_values),
        f"s_waitcnt lgkmcnt({wait_count});\n",
        "=v,=v,=v,=v,0,1,2,3,~{memory}",
        has_side_effects=True,
    )
    return tuple(llvm.extractvalue(raw_type, result, [i]) for i in range(4))


def _pack_ds_read_b64_tr_b16(raw_lo, raw_hi):
    """Pack two transpose-read b64 results into an eight-BF16 operand."""
    return (
        fx.Vector(raw_lo)
        .shuffle(fx.Vector(raw_hi), list(range(4)))
        .bitcast(fx.BFloat16)
    )


@lru_cache(maxsize=1)
def _build_v_stage_launcher():
    @flyc.kernel(
        name="v4_fp8_flydsl_stage_v_tile",
        known_block_size=[V_STAGE_BLOCK_THREADS, 1, 1],
    )
    def kernel(
        packed: fx.Tensor,
        rope: fx.Tensor,
        indices: fx.Tensor,
        out: fx.Tensor,
        num_rows: fx.Int32,
    ):
        packed_i32 = ptr_buf_tensor(packed, fx.Int32)
        rope_i32 = ptr_buf_tensor(rope, fx.Int32)
        indices_i32 = ptr_buf_tensor(indices, fx.Int32)
        out_i32 = ptr_buf_tensor(out, fx.Int32)

        tid = fx.Int32(fx.thread_idx.x)
        tile = fx.Int32(fx.block_idx.x)
        row_in_tile = tid // V_STAGE_THREADS_PER_ROW
        dword_in_group = tid % V_STAGE_THREADS_PER_ROW
        out_row = tile * V_STAGE_ROWS + row_in_tile
        active = out_row < num_rows
        safe_out_row = active.select(out_row, fx.Int32(0))
        source_row = fx.Int32(indices_i32[safe_out_row])
        source_row = active.select(source_row, fx.Int32(0))

        packed_row_dwords = V4_PACKED_ROW_BYTES // 4
        out_row_dwords = V4_DIM_QK // 2
        packed_base = source_row * packed_row_dwords
        out_base = out_row * out_row_dwords

        # Hoist all independent global reads before conversion.  In particular,
        # each packed scale dword serves two adjacent 64-value groups; keeping it
        # in a VGPR avoids the duplicate buffer loads emitted by a per-group
        # spelling and lets the VMEM operations overlap the conversion chain.
        raw_words = [
            fx.Int32(
                packed_i32[
                    packed_base + group * (V4_MX_GROUP_SIZE // 4) + dword_in_group
                ]
            )
            for group in fx.range_constexpr(V4_NUM_MX_GROUPS)
        ]
        scale_words = [
            fx.Uint32(
                packed_i32[packed_base + V4_SCALE_OFFSET_BYTES // 4 + scale_dword]
            )
            for scale_dword in fx.range_constexpr((V4_NUM_MX_GROUPS + 1) // 2)
        ]
        rope_row_dwords = V4_DIM_ROPE // 2
        rope_word = source_row * rope_row_dwords + dword_in_group * 2
        rope_words = [
            fx.Int32(rope_i32[rope_word]),
            fx.Int32(rope_i32[rope_word + 1]),
        ]

        for group in fx.range_constexpr(V4_NUM_MX_GROUPS):
            raw_word = raw_words[group]
            scale_word = scale_words[group // 2]
            if fx.const_expr(group & 1):
                e8m0 = (scale_word >> fx.Uint32(16)) & fx.Uint32(0xFF)
            else:
                e8m0 = scale_word & fx.Uint32(0xFF)
            scale = (e8m0 << fx.Uint32(23)).bitcast(fx.Float32)
            lo, hi = _decode_fp8x4_to_bf16_words(raw_word, scale)
            out_dword = out_base + group * (V4_MX_GROUP_SIZE // 2) + dword_in_group * 2
            if active:
                out_i32[out_dword] = lo
                out_i32[out_dword + 1] = hi

        out_rope_word = out_base + V4_DIM_NOPE // 2 + dword_in_group * 2
        if active:
            out_i32[out_rope_word] = rope_words[0]
            out_i32[out_rope_word + 1] = rope_words[1]

    @flyc.jit
    def launch(
        packed: fx.Tensor,
        rope: fx.Tensor,
        indices: fx.Tensor,
        out: fx.Tensor,
        num_rows: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(packed, rope, indices, out, num_rows).launch(
            grid=(grid_x, 1, 1),
            block=(V_STAGE_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=1)
def _build_nope_qk_launcher():
    """Build one-wave 16x16 V4 NoPE QK validation kernel."""

    @flyc.kernel(
        name="v4_fp8_flydsl_nope_qk_tile",
        known_block_size=[64, 1, 1],
    )
    def kernel(q_packed: fx.Tensor, k_packed: fx.Tensor, out: fx.Tensor):
        lane = fx.Int32(fx.thread_idx.x)
        lane_row = lane % fx.Int32(16)
        lane_group = lane // fx.Int32(16)

        q_i32 = GTensor(q_packed, dtype=T.i32, shape=(-1,))
        k_i32 = GTensor(k_packed, dtype=T.i32, shape=(-1,))
        out_f32 = ptr_buf_tensor(out, fx.Float32)

        row_dwords = V4_PACKED_ROW_BYTES // 4
        q_base = lane_row * row_dwords
        k_base = lane_row * row_dwords
        q_scale_words = fx.Vector(q_i32.vec_load((q_base + 112,), vec_size=4))
        k_scale_words = fx.Vector(k_i32.vec_load((k_base + 112,), vec_size=4))
        scale_q = _pack_opsel_scales(q_scale_words, lane_group)
        scale_k = _pack_opsel_scales(k_scale_words, lane_group)

        q_operands = []
        k_operands = []
        for step in fx.range_constexpr(4):
            step_base = step * 32 + lane_group * fx.Int32(4)
            q_lo = q_i32.vec_load((q_base + step_base,), vec_size=4)
            k_lo = k_i32.vec_load((k_base + step_base,), vec_size=4)
            if fx.const_expr(step == 3):
                q_hi = fx.Vector.filled(4, 0, fx.Int32)
                k_hi = fx.Vector.filled(4, 0, fx.Int32)
            else:
                q_hi = q_i32.vec_load((q_base + step_base + 16,), vec_size=4)
                k_hi = k_i32.vec_load((k_base + step_base + 16,), vec_size=4)
            q_operands.append(_join_i32x4(q_lo, q_hi))
            k_operands.append(_join_i32x4(k_lo, k_hi))

        acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
        for step in fx.range_constexpr(4):
            acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [
                    q_operands[step],
                    k_operands[step],
                    acc,
                    0,
                    0,
                    step,
                    scale_q,
                    step,
                    scale_k,
                ],
            )

        values = fx.Vector(acc)
        col = lane_row
        row = lane_group * fx.Int32(4)
        for i in fx.range_constexpr(4):
            out_f32[(row + i) * 16 + col] = values[i]

    @flyc.jit
    def launch(
        q_packed: fx.Tensor,
        k_packed: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(q_packed, k_packed, out).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=1)
def _build_qk_launcher():
    """Build one-wave 16x16 V4 QK tile, including the BF16 RoPE tail."""

    @flyc.kernel(
        name="v4_fp8_flydsl_qk_tile",
        known_block_size=[64, 1, 1],
    )
    def kernel(
        q_packed: fx.Tensor,
        q_rope: fx.Tensor,
        k_packed: fx.Tensor,
        k_rope: fx.Tensor,
        out: fx.Tensor,
    ):
        lane = fx.Int32(fx.thread_idx.x)
        lane_row = lane % fx.Int32(16)
        lane_group = lane // fx.Int32(16)

        q_i32 = GTensor(q_packed, dtype=T.i32, shape=(-1,))
        k_i32 = GTensor(k_packed, dtype=T.i32, shape=(-1,))
        q_rope_i32 = GTensor(q_rope, dtype=T.i32, shape=(-1,))
        k_rope_i32 = GTensor(k_rope, dtype=T.i32, shape=(-1,))
        out_f32 = ptr_buf_tensor(out, fx.Float32)

        row_dwords = V4_PACKED_ROW_BYTES // 4
        q_base = lane_row * row_dwords
        k_base = lane_row * row_dwords
        q_scale_words = fx.Vector(q_i32.vec_load((q_base + 112,), vec_size=4))
        k_scale_words = fx.Vector(k_i32.vec_load((k_base + 112,), vec_size=4))
        scale_q = _pack_opsel_scales(q_scale_words, lane_group)
        scale_k = _pack_opsel_scales(k_scale_words, lane_group)

        q_operands = []
        k_operands = []
        for step in fx.range_constexpr(4):
            step_base = step * 32 + lane_group * fx.Int32(4)
            q_lo = q_i32.vec_load((q_base + step_base,), vec_size=4)
            k_lo = k_i32.vec_load((k_base + step_base,), vec_size=4)
            if fx.const_expr(step == 3):
                q_hi = fx.Vector.filled(4, 0, fx.Int32)
                k_hi = fx.Vector.filled(4, 0, fx.Int32)
            else:
                q_hi = q_i32.vec_load((q_base + step_base + 16,), vec_size=4)
                k_hi = k_i32.vec_load((k_base + step_base + 16,), vec_size=4)
            q_operands.append(_join_i32x4(q_lo, q_hi))
            k_operands.append(_join_i32x4(k_lo, k_hi))

        acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
        for step in fx.range_constexpr(4):
            acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [
                    q_operands[step],
                    k_operands[step],
                    acc,
                    0,
                    0,
                    step,
                    scale_q,
                    step,
                    scale_k,
                ],
            )

        # gfx950's BF16 K32 atom consumes eight consecutive BF16 values per
        # lane.  lane_group selects one of the four K8 groups within each K32
        # step, matching the row/lane mapping of the scaled-FP8 atom above.
        rope_row_dwords = V4_DIM_ROPE // 2
        q_rope_base = lane_row * rope_row_dwords
        k_rope_base = lane_row * rope_row_dwords
        rope_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        rope_a = fx.make_rmem_tensor(8, fx.BFloat16)
        rope_b = fx.make_rmem_tensor(8, fx.BFloat16)
        rope_c = fx.make_rmem_tensor(4, fx.Float32)
        rope_c.store(fx.Vector(acc))
        for step in fx.range_constexpr(2):
            step_base = step * 16 + lane_group * fx.Int32(4)
            q_values = fx.Vector(
                q_rope_i32.vec_load((q_rope_base + step_base,), vec_size=4)
            ).bitcast(fx.BFloat16)
            k_values = fx.Vector(
                k_rope_i32.vec_load((k_rope_base + step_base,), vec_size=4)
            ).bitcast(fx.BFloat16)
            rope_a.store(q_values)
            rope_b.store(k_values)
            fx.mma_atom_call(rope_mma, rope_c, rope_a, rope_b, rope_c)

        values = fx.Vector(rope_c.load())
        col = lane_row
        row = lane_group * fx.Int32(4)
        for i in fx.range_constexpr(4):
            out_f32[(row + i) * 16 + col] = values[i]

    @flyc.jit
    def launch(
        q_packed: fx.Tensor,
        q_rope: fx.Tensor,
        k_packed: fx.Tensor,
        k_rope: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(q_packed, q_rope, k_packed, k_rope, out).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=1)
def _build_qk_h128_launcher():
    """Build the OPUS-shaped 128x32 QK block with eight independent waves."""

    @flyc.kernel(
        name="v4_fp8_flydsl_qk_h128_tile",
        known_block_size=[QK_WAVES * 64, 1, 1],
    )
    def kernel(
        q_packed: fx.Tensor,
        q_rope: fx.Tensor,
        k_packed: fx.Tensor,
        k_rope: fx.Tensor,
        out: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        lane_row = lane % fx.Int32(16)
        lane_group = lane // fx.Int32(16)

        q_i32 = GTensor(q_packed, dtype=T.i32, shape=(-1,))
        k_i32 = GTensor(k_packed, dtype=T.i32, shape=(-1,))
        q_rope_i32 = GTensor(q_rope, dtype=T.i32, shape=(-1,))
        k_rope_i32 = GTensor(k_rope, dtype=T.i32, shape=(-1,))
        out_f32 = ptr_buf_tensor(out, fx.Float32)

        row_dwords = V4_PACKED_ROW_BYTES // 4
        q_row = wave * fx.Int32(16) + lane_row
        q_base = q_row * row_dwords
        q_scale_words = fx.Vector(q_i32.vec_load((q_base + 112,), vec_size=4))
        scale_q = _pack_opsel_scales(q_scale_words, lane_group)

        q_operands = []
        for step in fx.range_constexpr(4):
            step_base = step * 32 + lane_group * fx.Int32(4)
            q_lo = q_i32.vec_load((q_base + step_base,), vec_size=4)
            if fx.const_expr(step == 3):
                q_hi = fx.Vector.filled(4, 0, fx.Int32)
            else:
                q_hi = q_i32.vec_load((q_base + step_base + 16,), vec_size=4)
            q_operands.append(_join_i32x4(q_lo, q_hi))

        rope_row_dwords = V4_DIM_ROPE // 2
        q_rope_base = q_row * rope_row_dwords
        q_rope_operands = []
        for step in fx.range_constexpr(2):
            step_base = step * 16 + lane_group * fx.Int32(4)
            q_rope_operands.append(
                fx.Vector(
                    q_rope_i32.vec_load((q_rope_base + step_base,), vec_size=4)
                ).bitcast(fx.BFloat16)
            )

        rope_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        rope_a = fx.make_rmem_tensor(8, fx.BFloat16)
        rope_b = fx.make_rmem_tensor(8, fx.BFloat16)
        rope_c = fx.make_rmem_tensor(4, fx.Float32)

        for nt in fx.range_constexpr(2):
            k_row = fx.Int32(nt * 16) + lane_row
            k_base = k_row * row_dwords
            k_scale_words = fx.Vector(k_i32.vec_load((k_base + 112,), vec_size=4))
            scale_k = _pack_opsel_scales(k_scale_words, lane_group)

            acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
            for step in fx.range_constexpr(4):
                step_base = step * 32 + lane_group * fx.Int32(4)
                k_lo = k_i32.vec_load((k_base + step_base,), vec_size=4)
                if fx.const_expr(step == 3):
                    k_hi = fx.Vector.filled(4, 0, fx.Int32)
                else:
                    k_hi = k_i32.vec_load((k_base + step_base + 16,), vec_size=4)
                k_operand = _join_i32x4(k_lo, k_hi)
                acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.f32x4,
                    [
                        q_operands[step],
                        k_operand,
                        acc,
                        0,
                        0,
                        step,
                        scale_q,
                        step,
                        scale_k,
                    ],
                )

            rope_c.store(fx.Vector(acc))
            k_rope_base = k_row * rope_row_dwords
            for step in fx.range_constexpr(2):
                step_base = step * 16 + lane_group * fx.Int32(4)
                k_values = fx.Vector(
                    k_rope_i32.vec_load((k_rope_base + step_base,), vec_size=4)
                ).bitcast(fx.BFloat16)
                rope_a.store(q_rope_operands[step])
                rope_b.store(k_values)
                fx.mma_atom_call(rope_mma, rope_c, rope_a, rope_b, rope_c)

            values = fx.Vector(rope_c.load())
            col = fx.Int32(nt * 16) + lane_row
            row = wave * fx.Int32(16) + lane_group * fx.Int32(4)
            for i in fx.range_constexpr(4):
                out_f32[(row + i) * QK_BLOCK_COLS + col] = values[i]

    @flyc.jit
    def launch(
        q_packed: fx.Tensor,
        q_rope: fx.Tensor,
        k_packed: fx.Tensor,
        k_rope: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(q_packed, q_rope, k_packed, k_rope, out).launch(
            grid=(1, 1, 1),
            block=(QK_WAVES * 64, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=1)
def _build_qk_pv_h128_launcher():
    """Build a fused 128x32 QK followed by BF16 P@V mapping probe."""

    @fx.struct
    class SharedStorage:
        p: fx.Array[fx.BFloat16, 4096, 16]
        denom: fx.Array[fx.Float32, 128, 16]
        v: fx.Array[fx.BFloat16, 16384, 16]

    @flyc.kernel(
        name="v4_fp8_flydsl_qk_pv_h128_tile",
        known_block_size=[QK_WAVES * 64, 1, 1],
    )
    def kernel(
        q_packed: fx.Tensor,
        q_rope: fx.Tensor,
        k_packed: fx.Tensor,
        k_rope: fx.Tensor,
        attn_sink: fx.Tensor,
        out: fx.Tensor,
        softmax_scale: fx.Float32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        lane_row = lane % fx.Int32(16)
        lane_group = lane // fx.Int32(16)

        q_i32 = GTensor(q_packed, dtype=T.i32, shape=(-1,))
        k_i32 = GTensor(k_packed, dtype=T.i32, shape=(-1,))
        q_rope_i32 = GTensor(q_rope, dtype=T.i32, shape=(-1,))
        k_rope_i32 = GTensor(k_rope, dtype=T.i32, shape=(-1,))
        sink_ptr = ptr_buf_tensor(attn_sink, fx.Float32)
        out_f32 = ptr_buf_tensor(out, fx.Float32)
        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        p_smem = shared.p
        denom_smem = shared.denom
        v_smem = shared.v

        row_dwords = V4_PACKED_ROW_BYTES // 4
        q_row = wave * fx.Int32(16) + lane_row
        q_base = q_row * row_dwords
        q_scale_words = fx.Vector(q_i32.vec_load((q_base + 112,), vec_size=4))
        scale_q = _pack_opsel_scales(q_scale_words, lane_group)

        q_operands = []
        for step in fx.range_constexpr(4):
            step_base = step * 32 + lane_group * fx.Int32(4)
            q_lo = q_i32.vec_load((q_base + step_base,), vec_size=4)
            if fx.const_expr(step == 3):
                q_hi = fx.Vector.filled(4, 0, fx.Int32)
            else:
                q_hi = q_i32.vec_load((q_base + step_base + 16,), vec_size=4)
            q_operands.append(_join_i32x4(q_lo, q_hi))

        rope_row_dwords = V4_DIM_ROPE // 2
        q_rope_base = q_row * rope_row_dwords
        q_rope_operands = []
        for step in fx.range_constexpr(2):
            step_base = step * 16 + lane_group * fx.Int32(4)
            q_rope_operands.append(
                fx.Vector(
                    q_rope_i32.vec_load((q_rope_base + step_base,), vec_size=4)
                ).bitcast(fx.BFloat16)
            )

        rope_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        rope_a = fx.make_rmem_tensor(8, fx.BFloat16)
        rope_b = fx.make_rmem_tensor(8, fx.BFloat16)
        rope_c = fx.make_rmem_tensor(4, fx.Float32)
        score_fragments = []

        # Swap Q/K at the MFMA boundary.  The resulting C fragments are laid
        # out as four keys for one query row per lane and can be reinterpreted
        # directly as the P operand of the K32 PV MFMA.
        for nt in fx.range_constexpr(2):
            k_row = fx.Int32(nt * 16) + lane_row
            k_base = k_row * row_dwords
            k_scale_words = fx.Vector(k_i32.vec_load((k_base + 112,), vec_size=4))
            scale_k = _pack_opsel_scales(k_scale_words, lane_group)

            acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
            for step in fx.range_constexpr(4):
                step_base = step * 32 + lane_group * fx.Int32(4)
                k_lo = k_i32.vec_load((k_base + step_base,), vec_size=4)
                if fx.const_expr(step == 3):
                    k_hi = fx.Vector.filled(4, 0, fx.Int32)
                else:
                    k_hi = k_i32.vec_load((k_base + step_base + 16,), vec_size=4)
                k_operand = _join_i32x4(k_lo, k_hi)
                acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.f32x4,
                    [
                        k_operand,
                        q_operands[step],
                        acc,
                        0,
                        0,
                        step,
                        scale_k,
                        step,
                        scale_q,
                    ],
                )

            rope_c.store(fx.Vector(acc))
            k_rope_base = k_row * rope_row_dwords
            for step in fx.range_constexpr(2):
                step_base = step * 16 + lane_group * fx.Int32(4)
                k_values = fx.Vector(
                    k_rope_i32.vec_load((k_rope_base + step_base,), vec_size=4)
                ).bitcast(fx.BFloat16)
                rope_a.store(k_values)
                rope_b.store(q_rope_operands[step])
                fx.mma_atom_call(rope_mma, rope_c, rope_a, rope_b, rope_c)
            score_fragments.append(fx.Vector(rope_c.load()))

        temperature_scale = softmax_scale * fx.Float32(LOG2E)
        scaled_scores = []
        row_max = fx.Float32(-float("inf"))
        for nt in fx.range_constexpr(2):
            values = []
            for i in fx.range_constexpr(4):
                score = fx.Float32(score_fragments[nt][i]) * temperature_scale
                values.append(score)
                row_max = fx.max(row_max, score)
            scaled_scores.append(values)
        row_max = fx.max(row_max, row_max.shuffle_xor(16, 64))
        row_max = fx.max(row_max, row_max.shuffle_xor(32, 64))
        sink_log2 = fx.Float32(sink_ptr[q_row]) * fx.Float32(LOG2E)
        row_max = fx.max(row_max, sink_log2)

        p_fragments = []
        row_sum = fx.Float32(0.0)
        for nt in fx.range_constexpr(2):
            values = []
            for i in fx.range_constexpr(4):
                p = _exp2_f32(scaled_scores[nt][i] - row_max)
                values.append(p)
                row_sum = row_sum + p
            p_fragments.append(values)
        row_sum = row_sum + row_sum.shuffle_xor(16, 64)
        row_sum = row_sum + row_sum.shuffle_xor(32, 64)
        denom = row_sum + _exp2_f32(sink_log2 - row_max)
        denom_smem[q_row] = denom

        # Materialize P in row-major LDS for this mapping probe.  This makes the
        # required C-fragment -> A-operand permutation explicit and gives the
        # final fused kernel a correctness oracle before replacing it with
        # register permutes.
        for nt in fx.range_constexpr(2):
            p_col = fx.Int32(nt * 16) + lane_group * fx.Int32(4)
            for i in fx.range_constexpr(4):
                p_smem[q_row * QK_BLOCK_COLS + p_col + i] = p_fragments[nt][i].to(
                    fx.BFloat16
                )

        # The same K rows are the V rows for MLA.  All 512 threads cooperate:
        # sixteen threads own one KV row, each decoding four values from every
        # MXFP8 group and four BF16 RoPE values.  Store transposed [D, K] so a
        # PV K32 operand is one contiguous LDS vector per lane.
        v_row = tid // fx.Int32(V_STAGE_THREADS_PER_ROW)
        v_dword = tid % fx.Int32(V_STAGE_THREADS_PER_ROW)
        v_packed_base = v_row * row_dwords
        v_raw_words = [
            fx.Int32(k_i32[v_packed_base + group * (V4_MX_GROUP_SIZE // 4) + v_dword])
            for group in fx.range_constexpr(V4_NUM_MX_GROUPS)
        ]
        v_scale_words = [
            fx.Uint32(k_i32[v_packed_base + V4_SCALE_OFFSET_BYTES // 4 + scale_dword])
            for scale_dword in fx.range_constexpr((V4_NUM_MX_GROUPS + 1) // 2)
        ]
        for group in fx.range_constexpr(V4_NUM_MX_GROUPS):
            scale_word = v_scale_words[group // 2]
            if fx.const_expr(group & 1):
                e8m0 = (scale_word >> fx.Uint32(16)) & fx.Uint32(0xFF)
            else:
                e8m0 = scale_word & fx.Uint32(0xFF)
            scale = (e8m0 << fx.Uint32(23)).bitcast(fx.Float32)
            lo, hi = _decode_fp8x4_to_bf16_words(v_raw_words[group], scale)
            decoded = fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(fx.BFloat16)
            d_base = group * V4_MX_GROUP_SIZE + v_dword * fx.Int32(4)
            for i in fx.range_constexpr(4):
                v_smem[(d_base + i) * QK_BLOCK_COLS + v_row] = decoded[i]

        v_rope_base = v_row * rope_row_dwords + v_dword * fx.Int32(2)
        v_rope = fx.Vector(k_rope_i32.vec_load((v_rope_base,), vec_size=2)).bitcast(
            fx.BFloat16
        )
        v_d_base = V4_DIM_NOPE + v_dword * fx.Int32(4)
        for i in fx.range_constexpr(4):
            v_smem[(v_d_base + i) * QK_BLOCK_COLS + v_row] = v_rope[i]
        fx.rocdl.s_barrier()

        p_values = fx.Vector.from_elements(
            [
                fx.BFloat16(
                    p_smem[
                        q_row * QK_BLOCK_COLS + lane_group * fx.Int32(8) + fx.Int32(i)
                    ]
                )
                for i in fx.range_constexpr(8)
            ],
            fx.BFloat16,
        )
        pv_a = fx.make_rmem_tensor(8, fx.BFloat16)
        pv_b = fx.make_rmem_tensor(8, fx.BFloat16)
        pv_c = fx.make_rmem_tensor(4, fx.Float32)
        pv_a.store(p_values)

        for ds in fx.range_constexpr(V4_DIM_QK // 16):
            d_col = fx.Int32(ds * 16) + lane_row
            v_values = fx.Vector.from_elements(
                [
                    fx.BFloat16(
                        v_smem[
                            d_col * QK_BLOCK_COLS
                            + lane_group * fx.Int32(8)
                            + fx.Int32(j)
                        ]
                    )
                    for j in fx.range_constexpr(8)
                ],
                fx.BFloat16,
            )
            pv_b.store(v_values)
            pv_c.store(fx.Vector.filled(4, 0.0, fx.Float32))
            fx.mma_atom_call(rope_mma, pv_c, pv_a, pv_b, pv_c)
            values = fx.Vector(pv_c.load())
            out_row = wave * fx.Int32(16) + lane_group * fx.Int32(4)
            out_col = fx.Int32(ds * 16) + lane_row
            for i in fx.range_constexpr(4):
                out_f32[(out_row + i) * V4_DIM_QK + out_col] = values[i] / fx.Float32(
                    denom_smem[out_row + i]
                )

    @flyc.jit
    def launch(
        q_packed: fx.Tensor,
        q_rope: fx.Tensor,
        k_packed: fx.Tensor,
        k_rope: fx.Tensor,
        attn_sink: fx.Tensor,
        out: fx.Tensor,
        softmax_scale: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            q_packed,
            q_rope,
            k_packed,
            k_rope,
            attn_sink,
            out,
            softmax_scale,
        ).launch(
            grid=(1, 1, 1),
            block=(QK_WAVES * 64, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=16)
def _build_sparse_prefill_launcher(
    pipeline_two: bool = False,
    cluster_two: bool = False,
    wave_padded_k: bool = False,
    alpha_bpermute: bool = False,
    lds_padding: int = 0,
    rescale_once: bool = False,
    fixed_softmax_ref: bool = False,
    transpose_v: bool = False,
    register_p: bool = False,
    cache_q_step: int = -1,
    reuse_q_across_n: bool = False,
    broadcast_indices: bool = False,
    no_sentinel: bool = False,
    full_tile_fastpath: bool = False,
    p_lane_layout: bool = False,
    stage_q: bool = False,
    cache_all_q: bool = False,
    assume_full_tiles: bool = False,
    permute_k_scales: bool = False,
    pairwise_pv: bool = False,
    post_misched: bool = False,
    machine_sink: bool = False,
    setprio: bool = False,
    dynamic_full_prefix: bool = False,
    prefix_only: bool = False,
    split_partial: bool = False,
    waves_per_eu: int = 0,
):
    """Build the H128 paged sparse-prefill kernel.

    ``pipeline_two`` is a benchmark-only rolling K32 pipeline.  It double
    buffers raw K/RoPE plus the BF16 P/V operands so QK for the current tile
    and PV for the previous tile share one scheduling phase.

    ``cluster_two`` keeps the same double-buffered storage, but splits each
    fixed-reference softmax across adjacent tiles: the low K16 half is
    produced after the current PV phase, while the high K16 half is completed
    after the next tile's QK.  This mirrors the two-tile OPUS cluster ordering
    without changing the established ``pipeline_two`` benchmark path.

    ``assume_full_tiles`` is an exact-K32 benchmark specialization.  Callers
    must guarantee that every prefix and extend segment length is divisible by
    32; the specialization intentionally removes all per-tile bounds checks.

    ``permute_k_scales`` replaces the hot-loop shift/and/or scale transpose with
    two gfx950 ``v_perm_b32`` operations while keeping scale lifetimes local.
    ``pairwise_pv`` pipelines two adjacent D16 PV fragments as one D32 group.
    """

    rolling_pipeline = pipeline_two or cluster_two
    raw_wave_bytes = 64 * 16
    raw_wave_stride_bytes = raw_wave_bytes + (32 if wave_padded_k else 0)
    raw_wave_stride_dwords = raw_wave_stride_bytes // 4
    packed_wave_blocks = 16
    rope_wave_blocks = 4
    packed_slot_dwords = packed_wave_blocks * raw_wave_stride_bytes // 4
    rope_slot_dwords = rope_wave_blocks * raw_wave_stride_bytes // 4
    raw_stage_slots = 2 if rolling_pipeline else 1
    pv_stage_slots = 2 if rolling_pipeline else 1
    pv_stride = QK_BLOCK_COLS + lds_padding
    p_lane_stride = 10
    p_slot_elems = (
        QK_WAVES * 64 * p_lane_stride if p_lane_layout else QK_BLOCK_ROWS * pv_stride
    )
    v_stride = V4_DIM_QK + 16 if transpose_v else pv_stride
    v_slot_elems = QK_BLOCK_COLS * v_stride if transpose_v else V4_DIM_QK * v_stride
    p_stage_elems = (
        p_slot_elems * pv_stage_slots if rolling_pipeline and not register_p else 1
    )
    q_stage_stride = 25
    q_stage_elems = QK_WAVES * 64 * q_stage_stride if stage_q else 1

    @fx.struct
    class SharedStorage:
        k: fx.Array[fx.Int32, packed_slot_dwords * raw_stage_slots, 16]
        k_rope: fx.Array[fx.Int32, rope_slot_dwords * raw_stage_slots, 16]
        p: fx.Array[fx.BFloat16, p_stage_elems, 16]
        v: fx.Array[fx.BFloat16, v_slot_elems * pv_stage_slots, 16]
        valid: fx.Array[fx.Int32, 32 * raw_stage_slots, 16]
        q: fx.Array[fx.Int32, q_stage_elems, 16]
        alpha: fx.Array[
            fx.Float32, 128 if rolling_pipeline or not alpha_bpermute else 1, 16
        ]

    kernel_value_attrs = {
        "rocdl.flat_work_group_size": "512,512",
        "passthrough": [
            ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
            ["no-nans-fp-math", "true"],
            ["unsafe-fp-math", "true"],
        ],
    }
    if waves_per_eu > 0:
        kernel_value_attrs["rocdl.waves_per_eu"] = int(waves_per_eu)

    @flyc.kernel(
        name=(
            "v4_fp8_flydsl_sparse_prefill_h128"
            + ("_pipeline_two" if pipeline_two else "")
            + ("_cluster_two" if cluster_two else "")
            + ("_wavepadk" if wave_padded_k else "")
            + ("_alpha_bpermute" if alpha_bpermute else "")
            + (f"_pad{lds_padding}" if lds_padding else "")
            + ("_rescale_once" if rescale_once else "")
            + ("_fixed_ref" if fixed_softmax_ref else "")
            + ("_trv" if transpose_v else "")
            + ("_regp" if register_p else "")
            + (f"_cacheq{cache_q_step}" if cache_q_step >= 0 else "")
            + ("_reuseqn" if reuse_q_across_n else "")
            + ("_bcastidx" if broadcast_indices else "")
            + ("_nosentinel" if no_sentinel else "")
            + ("_fulltile" if full_tile_fastpath else "")
            + ("_planel" if p_lane_layout else "")
            + ("_stageq" if stage_q else "")
            + ("_cacheqall" if cache_all_q else "")
            + ("_assumefull" if assume_full_tiles else "")
            + ("_permkscale2" if permute_k_scales else "")
            + ("_pairpv4" if pairwise_pv else "")
            + ("_postmisched" if post_misched else "")
            + ("_msink" if machine_sink else "")
            + ("_setprio" if setprio else "")
            + ("_dynfullprefix" if dynamic_full_prefix else "")
            + ("_prefixonly" if prefix_only else "")
            + ("_splitpartial" if split_partial else "")
            + (f"_weu{waves_per_eu}" if waves_per_eu else "")
        ),
        known_block_size=[QK_WAVES * 64, 1, 1],
    )
    def kernel(
        q_packed: fx.Pointer,
        q_rope: fx.Pointer,
        prefix_packed: fx.Pointer,
        prefix_rope: fx.Pointer,
        prefix_indices: fx.Pointer,
        prefix_indptr: fx.Pointer,
        extend_packed: fx.Pointer,
        extend_rope: fx.Pointer,
        extend_indices: fx.Pointer,
        extend_indptr: fx.Pointer,
        attn_sink: fx.Pointer,
        out: fx.Pointer,
        split_task_query: fx.Pointer,
        split_task_start: fx.Pointer,
        split_task_len: fx.Pointer,
        split_task_row: fx.Pointer,
        partial_m: fx.Pointer,
        partial_l: fx.Pointer,
        softmax_scale: fx.Float32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        task = fx.Int32(fx.block_idx.x)
        split_task_query_i32 = ptr_buf_tensor(split_task_query, fx.Int32)
        split_task_start_i32 = ptr_buf_tensor(split_task_start, fx.Int32)
        split_task_len_i32 = ptr_buf_tensor(split_task_len, fx.Int32)
        split_task_row_i32 = ptr_buf_tensor(split_task_row, fx.Int32)
        if fx.const_expr(split_partial):
            query = fx.Int32(split_task_query_i32[task])
        else:
            query = task
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        lane_row = lane % fx.Int32(16)
        lane_group = lane // fx.Int32(16)
        head = wave * fx.Int32(16) + lane_row

        q_i32 = GTensor(q_packed, dtype=T.i32, shape=(-1,))
        q_rope_i32 = GTensor(q_rope, dtype=T.i32, shape=(-1,))
        prefix_indices_i32 = ptr_buf_tensor(prefix_indices, fx.Int32)
        prefix_indptr_i32 = ptr_buf_tensor(prefix_indptr, fx.Int32)
        extend_indices_i32 = ptr_buf_tensor(extend_indices, fx.Int32)
        extend_indptr_i32 = ptr_buf_tensor(extend_indptr, fx.Int32)
        sink_f32 = ptr_buf_tensor(attn_sink, fx.Float32)
        if fx.const_expr(split_partial):
            out_f16 = ptr_buf_tensor(out, fx.Float16)
            partial_m_f32 = ptr_buf_tensor(partial_m, fx.Float32)
            partial_l_f32 = ptr_buf_tensor(partial_l, fx.Float32)
        else:
            out_bf16 = ptr_buf_tensor(out, fx.BFloat16)

        def make_byte_div(pointer):
            byte_buffer = ptr_buf_tensor(pointer, fx.Int8)
            return fx.logical_divide(byte_buffer, fx.make_layout(1, 1))

        prefix_div = make_byte_div(prefix_packed)
        prefix_rope_div = make_byte_div(prefix_rope)
        extend_div = make_byte_div(extend_packed)
        extend_rope_div = make_byte_div(extend_rope)

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_smem = shared.k
        k_rope_smem = shared.k_rope
        p_smem = shared.p
        v_smem = shared.v
        valid_smem = shared.valid
        q_smem = shared.q
        alpha_smem = shared.alpha

        k_lds_base = fx.Index(fx.ptrtoint(k_smem.ptr))
        k_rope_lds_base = fx.Index(fx.ptrtoint(k_rope_smem.ptr))
        v_lds_base = fx.Index(fx.ptrtoint(v_smem.ptr))
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        lds_byte_ptr_type = fx.PointerType.get(fx.Int8.ir_type, 2, 128)

        def dma_128(source_div, source_byte, lds_byte):
            lds_ptr = fx.inttoptr(lds_byte_ptr_type, fx.Int32(lds_byte))
            destination = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            source = fx.slice(source_div, (None, fx.Int32(source_byte)))
            fx.copy(dma_atom, source, destination, soffset=fx.Int32(0))

        row_dwords = V4_PACKED_ROW_BYTES // 4
        rope_row_dwords = V4_DIM_ROPE // 2
        q_row = query * QK_BLOCK_ROWS + head
        q_base = q_row * row_dwords
        q_scale_words = fx.Vector(q_i32.vec_load((q_base + 112,), vec_size=4))
        scale_q = _pack_opsel_scales(q_scale_words, lane_group)

        q_rope_base = q_row * rope_row_dwords
        q_tail_words = fx.Vector(
            q_i32.vec_load(
                (q_base + 96 + lane_group * fx.Int32(4),),
                vec_size=4,
            )
        )
        q_rope_lo = fx.Vector(
            q_rope_i32.vec_load(
                (q_rope_base + lane_group * fx.Int32(4),),
                vec_size=4,
            )
        ).bitcast(fx.BFloat16)
        cached_q_rope_hi = fx.Vector.filled(4, 0, fx.Int32).ir_value()
        if fx.const_expr(cache_all_q):
            cached_q_rope_hi = q_rope_i32.vec_load(
                (q_rope_base + fx.Int32(16) + lane_group * fx.Int32(4),),
                vec_size=4,
            )
        cached_q_operand = fx.Vector.filled(8, 0, fx.Int32).ir_value()
        cached_q_operands = fx.make_rmem_tensor(24, fx.Int32)
        if fx.const_expr(cache_q_step >= 0):
            cached_q_step_base = cache_q_step * 32 + lane_group * fx.Int32(4)
            cached_q_lo = q_i32.vec_load((q_base + cached_q_step_base,), vec_size=4)
            cached_q_hi = q_i32.vec_load(
                (q_base + cached_q_step_base + 16,), vec_size=4
            )
            cached_q_operand = _join_i32x4(cached_q_lo, cached_q_hi)
        if fx.const_expr(cache_all_q):
            for step in fx.range_constexpr(3):
                cached_step_base = step * 32 + lane_group * fx.Int32(4)
                cached_step_lo = fx.Vector(
                    q_i32.vec_load((q_base + cached_step_base,), vec_size=4)
                )
                cached_step_hi = fx.Vector(
                    q_i32.vec_load((q_base + cached_step_base + 16,), vec_size=4)
                )
                for i in fx.range_constexpr(4):
                    cached_q_operands[step * 8 + i] = cached_step_lo[i]
                    cached_q_operands[step * 8 + 4 + i] = cached_step_hi[i]
        if fx.const_expr(stage_q):
            q_stage_base = tid * fx.Int32(q_stage_stride)
            for step in fx.range_constexpr(3):
                q_stage_step = step * 32 + lane_group * fx.Int32(4)
                q_stage_lo = fx.Vector(
                    q_i32.vec_load((q_base + q_stage_step,), vec_size=4)
                )
                q_stage_hi = fx.Vector(
                    q_i32.vec_load((q_base + q_stage_step + 16,), vec_size=4)
                )
                for i in fx.range_constexpr(4):
                    q_smem[q_stage_base + step * 8 + i] = q_stage_lo[i]
                    q_smem[q_stage_base + step * 8 + 4 + i] = q_stage_hi[i]
            fx.rocdl.s_barrier()

        bf16_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        mma_a = fx.make_rmem_tensor(8, fx.BFloat16)
        mma_b = fx.make_rmem_tensor(8, fx.BFloat16)
        mma_c = fx.make_rmem_tensor(4, fx.Float32)
        output_acc = [
            fx.make_rmem_tensor(4, fx.Float32)
            for _ in fx.range_constexpr(PREFILL_D_PER_CTA // 16)
        ]
        for ds in fx.range_constexpr(PREFILL_D_PER_CTA // 16):
            output_acc[ds].store(fx.Vector.filled(4, 0.0, fx.Float32))

        temperature_scale = softmax_scale * fx.Float32(LOG2E)
        if fx.const_expr(fixed_softmax_ref):
            m_row = fx.Float32(0.0)
        else:
            m_row = fx.Float32(-float("inf"))
        l_row = fx.Float32(0.0)

        def process_segment(
            source_tensor,
            source_rope_tensor,
            source_div,
            source_rope_div,
            indices_i32,
            start,
            length,
            m_state,
            l_state,
            k_stage,
            k_rope_stage,
            p_stage,
            v_stage,
            valid_stage,
            alpha_stage,
            frag_a,
            frag_b,
            frag_c,
            q_tensor,
            q_rope_tensor,
            assume_segment_full,
        ):
            tile_offset = fx.Int32(0)
            if fx.const_expr(rolling_pipeline):

                def stage_tile(stage_slot, stage_offset, stage_valid, stage_is_full):
                    packed_slot_bytes = stage_slot * fx.Int32(packed_slot_dwords * 4)
                    rope_slot_bytes = stage_slot * fx.Int32(rope_slot_dwords * 4)
                    valid_slot_base = stage_slot * fx.Int32(QK_BLOCK_COLS)

                    # Resolve all three gather indices first.  If an index load
                    # sits between two global_load_lds operations, LLVM emits
                    # vmcnt(0) before consuming it and accidentally drains the
                    # earlier DMA, destroying the intended overlap.
                    packed_stage_rows = []
                    packed_chunks_in_row = []
                    packed_source_rows = []
                    packed_row_valid = []
                    for dma_pass in fx.range_constexpr(2):
                        chunk = tid + fx.Int32(dma_pass * 512)
                        stage_row = chunk // fx.Int32(32)
                        chunk_in_row = chunk % fx.Int32(32)
                        logical_row = stage_offset + stage_row
                        if fx.const_expr(assume_segment_full):
                            safe_logical_row = logical_row
                        else:
                            in_range = logical_row < length
                            safe_logical_row = in_range.select(logical_row, fx.Int32(0))
                            if fx.const_expr(full_tile_fastpath):
                                if stage_is_full:
                                    in_range = logical_row >= fx.Int32(0)
                                    safe_logical_row = logical_row
                        if fx.const_expr(broadcast_indices):
                            source_row_seed = fx.Int32(0)
                            if chunk_in_row == fx.Int32(0):
                                source_row_seed = fx.Int32(
                                    indices_i32[start + safe_logical_row]
                                )
                            source_lane = (lane // fx.Int32(32)) * fx.Int32(32)
                            source_row = fx.Int32(
                                fx.rocdl.ds_bpermute(
                                    T.i32,
                                    (source_lane * fx.Int32(4)).ir_value(),
                                    source_row_seed.ir_value(),
                                )
                            )
                        else:
                            source_row = fx.Int32(indices_i32[start + safe_logical_row])
                        if fx.const_expr(assume_segment_full):
                            row_valid = source_row >= fx.Int32(0)
                            safe_source_row = source_row
                        elif fx.const_expr(no_sentinel):
                            row_valid = in_range
                            safe_source_row = row_valid.select(source_row, fx.Int32(0))
                        else:
                            row_valid = in_range & (source_row >= fx.Int32(0))
                            safe_source_row = row_valid.select(source_row, fx.Int32(0))
                        packed_stage_rows.append(stage_row)
                        packed_chunks_in_row.append(chunk_in_row)
                        packed_source_rows.append(safe_source_row)
                        packed_row_valid.append(row_valid)

                    rope_chunk = tid
                    rope_stage_row = rope_chunk // fx.Int32(8)
                    rope_chunk_in_row = rope_chunk % fx.Int32(8)
                    rope_logical_row = stage_offset + rope_stage_row
                    rope_active = wave < fx.Int32(4)
                    if fx.const_expr(assume_segment_full):
                        rope_in_range = rope_active
                        rope_safe_logical_row = rope_active.select(
                            rope_logical_row, fx.Int32(0)
                        )
                    else:
                        rope_in_range = rope_active & (rope_logical_row < length)
                        rope_safe_logical_row = rope_in_range.select(
                            rope_logical_row, fx.Int32(0)
                        )
                        if fx.const_expr(full_tile_fastpath):
                            if stage_is_full:
                                rope_in_range = rope_active
                                rope_safe_logical_row = rope_active.select(
                                    rope_logical_row, fx.Int32(0)
                                )
                    if fx.const_expr(broadcast_indices):
                        rope_source_row_seed = fx.Int32(0)
                        if (wave < fx.Int32(4)) & (rope_chunk_in_row == fx.Int32(0)):
                            rope_source_row_seed = fx.Int32(
                                indices_i32[start + rope_safe_logical_row]
                            )
                        rope_source_lane = (lane // fx.Int32(8)) * fx.Int32(8)
                        rope_source_row = fx.Int32(
                            fx.rocdl.ds_bpermute(
                                T.i32,
                                (rope_source_lane * fx.Int32(4)).ir_value(),
                                rope_source_row_seed.ir_value(),
                            )
                        )
                    else:
                        rope_source_row = fx.Int32(
                            indices_i32[start + rope_safe_logical_row]
                        )
                    if fx.const_expr(no_sentinel):
                        rope_row_valid = rope_in_range
                    else:
                        rope_row_valid = rope_in_range & (
                            rope_source_row >= fx.Int32(0)
                        )
                    rope_safe_source_row = rope_row_valid.select(
                        rope_source_row, fx.Int32(0)
                    )

                    # Two CTA-wide passes cover the 16 KiB packed tile.  The
                    # LDS address is wave-uniform; global_load_lds supplies the
                    # per-lane 16-byte destination stride implicitly.
                    for dma_pass in fx.range_constexpr(2):
                        stage_row = packed_stage_rows[dma_pass]
                        chunk_in_row = packed_chunks_in_row[dma_pass]
                        if fx.const_expr(wave_padded_k):
                            packed_wave_offset = fx.Int32(
                                (dma_pass * QK_WAVES) * raw_wave_stride_bytes
                            ) + wave * fx.Int32(raw_wave_stride_bytes)
                        else:
                            packed_wave_offset = fx.Int32(
                                dma_pass * 512 * 16
                            ) + wave * fx.Int32(64 * 16)
                        dma_128(
                            source_div,
                            packed_source_rows[dma_pass] * fx.Int32(V4_PACKED_ROW_BYTES)
                            + chunk_in_row * fx.Int32(16),
                            k_lds_base
                            + fx.Index(packed_slot_bytes)
                            + fx.Index(packed_wave_offset),
                        )
                        if fx.const_expr(not no_sentinel) and (
                            chunk_in_row == fx.Int32(0)
                        ):
                            stage_valid[valid_slot_base + stage_row] = packed_row_valid[
                                dma_pass
                            ].select(fx.Int32(1), fx.Int32(0))

                    # Four waves cover the 4 KiB RoPE tile.
                    if wave < fx.Int32(4):
                        if fx.const_expr(wave_padded_k):
                            rope_wave_offset = wave * fx.Int32(raw_wave_stride_bytes)
                        else:
                            rope_wave_offset = wave * fx.Int32(64 * 16)
                        dma_128(
                            source_rope_div,
                            rope_safe_source_row * fx.Int32(V4_DIM_ROPE * 2)
                            + rope_chunk_in_row * fx.Int32(16),
                            k_rope_lds_base
                            + fx.Index(rope_slot_bytes)
                            + fx.Index(rope_wave_offset),
                        )

                if length > fx.Int32(0):
                    stage_tile(
                        fx.Int32(0),
                        fx.Int32(0),
                        valid_stage,
                        length >= fx.Int32(QK_BLOCK_COLS),
                    )
                    fx.rocdl.s_waitcnt(0)
                    fx.rocdl.sched_barrier(0)
                    fx.rocdl.s_barrier()

            def run_staged_pv(
                pv_slot,
                rescale_output,
                register_p_values,
                pv_frag_a,
                pv_frag_b,
                pv_frag_c,
                pv_output_acc,
            ):
                p_slot_base = pv_slot * fx.Int32(p_slot_elems)
                v_slot_base = pv_slot * fx.Int32(v_slot_elems)
                if fx.const_expr(register_p):
                    p_values = register_p_values
                else:
                    if fx.const_expr(p_lane_layout):
                        p_read_base = (
                            p_slot_base
                            + wave * fx.Int32(64 * p_lane_stride)
                            + lane * fx.Int32(p_lane_stride)
                        )
                    else:
                        p_read_base = (
                            p_slot_base + head * pv_stride + lane_group * fx.Int32(8)
                        )
                    p_values = fx.Vector.from_elements(
                        [
                            fx.BFloat16(p_stage[p_read_base + fx.Int32(i)])
                            for i in fx.range_constexpr(8)
                        ],
                        fx.BFloat16,
                    )
                pv_frag_a.store(p_values)
                out_row = wave * fx.Int32(16) + lane_group * fx.Int32(4)
                alpha_vector = fx.Vector.from_elements(
                    [
                        fx.Float32(alpha_stage[out_row + i])
                        for i in fx.range_constexpr(4)
                    ],
                    fx.Float32,
                )

                def accumulate_pv(
                    ds,
                    v_values,
                    frag_b,
                    frag_c,
                    output_acc,
                    alpha,
                    apply_rescale,
                ):
                    old = fx.Vector(output_acc[ds].load())
                    frag_b.store(v_values)
                    if apply_rescale:
                        frag_c.store(old * alpha)
                    else:
                        frag_c.store(old)
                    fx.mma_atom_call(bf16_mma, frag_c, pv_frag_a, frag_b, frag_c)
                    output_acc[ds].store(fx.Vector(frag_c.load()))

                source_k = lane_group * fx.Int32(8) + lane_row // fx.Int32(4)
                source_d = (lane_row % fx.Int32(4)) * fx.Int32(4)
                v_address_base = v_lds_base + fx.Index(
                    v_slot_base + source_k * fx.Int32(v_stride) + source_d
                ) * fx.Index(2)
                if fx.const_expr(transpose_v and pairwise_pv):
                    raw_v_pair = _ds_read_b64_tr_b16_pair_start(
                        v_address_base, 0, 4 * v_stride * 2
                    )
                    for ds_pair in fx.range_constexpr(PREFILL_D_PER_CTA // 32):
                        if fx.const_expr(ds_pair + 1 < PREFILL_D_PER_CTA // 32):
                            ready_v_pair, raw_v_pair = _ds_read_b64_tr_b16_pair_advance(
                                v_address_base,
                                (ds_pair + 1) * 32 * 2,
                                4 * v_stride * 2,
                                raw_v_pair,
                            )
                        else:
                            ready_v_pair = _ds_read_b64_tr_b16_pair_wait(raw_v_pair, 0)
                        for inner in fx.range_constexpr(2):
                            ds = ds_pair * 2 + inner
                            v_values = _pack_ds_read_b64_tr_b16(
                                ready_v_pair[inner * 2],
                                ready_v_pair[inner * 2 + 1],
                            )
                            accumulate_pv(
                                ds,
                                v_values,
                                pv_frag_b,
                                pv_frag_c,
                                pv_output_acc,
                                alpha_vector,
                                rescale_output,
                            )
                else:
                    if fx.const_expr(transpose_v):
                        raw_v_lo, raw_v_hi = _ds_read_b64_tr_b16_start(
                            v_address_base, 0, 4 * v_stride * 2
                        )
                    for ds in fx.range_constexpr(PREFILL_D_PER_CTA // 16):
                        d_col = fx.Int32(ds * 16) + lane_row
                        if fx.const_expr(transpose_v):
                            if fx.const_expr(ds + 1 < PREFILL_D_PER_CTA // 16):
                                ready_v_lo, ready_v_hi, raw_v_lo, raw_v_hi = (
                                    _ds_read_b64_tr_b16_advance(
                                        v_address_base,
                                        (ds + 1) * 16 * 2,
                                        4 * v_stride * 2,
                                        raw_v_lo,
                                        raw_v_hi,
                                    )
                                )
                            else:
                                ready_v_lo, ready_v_hi = _ds_read_b64_tr_b16_finish(
                                    raw_v_lo, raw_v_hi
                                )
                            v_values = _pack_ds_read_b64_tr_b16(ready_v_lo, ready_v_hi)
                        else:
                            v_values = fx.Vector.from_elements(
                                [
                                    fx.BFloat16(
                                        v_stage[
                                            v_slot_base
                                            + d_col * pv_stride
                                            + lane_group * fx.Int32(8)
                                            + fx.Int32(j)
                                        ]
                                    )
                                    for j in fx.range_constexpr(8)
                                ],
                                fx.BFloat16,
                            )
                        accumulate_pv(
                            ds,
                            v_values,
                            pv_frag_b,
                            pv_frag_c,
                            pv_output_acc,
                            alpha_vector,
                            rescale_output,
                        )

            def stage_v_tile(k_slot_base, rope_slot_base, v_slot_base):
                tile_row = tid // fx.Int32(V_STAGE_THREADS_PER_ROW)
                tile_dword = tid % fx.Int32(V_STAGE_THREADS_PER_ROW)
                if fx.const_expr(wave_padded_k):
                    packed_row_base = (tile_row // fx.Int32(2)) * fx.Int32(
                        raw_wave_stride_dwords
                    ) + (tile_row & fx.Int32(1)) * fx.Int32(row_dwords)
                for group in fx.range_constexpr(V4_NUM_MX_GROUPS):
                    logical_dword = (
                        fx.Int32(group * (V4_MX_GROUP_SIZE // 4)) + tile_dword
                    )
                    if fx.const_expr(wave_padded_k):
                        raw_index = packed_row_base + logical_dword
                        scale_index = packed_row_base + fx.Int32(112 + group // 2)
                    else:
                        raw_index = tile_row * row_dwords + logical_dword
                        scale_index = tile_row * row_dwords + fx.Int32(112 + group // 2)
                    raw = fx.Int32(k_stage[k_slot_base + raw_index])
                    scale_word = fx.Uint32(k_stage[k_slot_base + scale_index])
                    if fx.const_expr(group & 1):
                        e8m0 = (scale_word >> fx.Uint32(16)) & fx.Uint32(0xFF)
                    else:
                        e8m0 = scale_word & fx.Uint32(0xFF)
                    scale = (e8m0 << fx.Uint32(23)).bitcast(fx.Float32)
                    lo, hi = _decode_fp8x4_to_bf16_words(raw, scale)
                    decoded = fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(
                        fx.BFloat16
                    )
                    d_base = group * V4_MX_GROUP_SIZE + tile_dword * fx.Int32(4)
                    for i in fx.range_constexpr(4):
                        if fx.const_expr(transpose_v):
                            v_stage[
                                v_slot_base + tile_row * fx.Int32(v_stride) + d_base + i
                            ] = decoded[i]
                        else:
                            v_stage[
                                v_slot_base + (d_base + i) * pv_stride + tile_row
                            ] = decoded[i]

                logical_rope_dword = tile_dword * fx.Int32(2)
                if fx.const_expr(wave_padded_k):
                    v_rope_base = (
                        (tile_row // fx.Int32(8)) * fx.Int32(raw_wave_stride_dwords)
                        + (tile_row & fx.Int32(7)) * fx.Int32(rope_row_dwords)
                        + logical_rope_dword
                    )
                else:
                    v_rope_base = tile_row * rope_row_dwords + logical_rope_dword
                v_rope_values = fx.Vector.from_elements(
                    [
                        fx.Int32(k_rope_stage[rope_slot_base + v_rope_base]),
                        fx.Int32(k_rope_stage[rope_slot_base + v_rope_base + 1]),
                    ],
                    fx.Int32,
                ).bitcast(fx.BFloat16)
                v_d_base = V4_DIM_NOPE + tile_dword * fx.Int32(4)
                for i in fx.range_constexpr(4):
                    if fx.const_expr(transpose_v):
                        v_stage[
                            v_slot_base + tile_row * fx.Int32(v_stride) + v_d_base + i
                        ] = v_rope_values[i]
                    else:
                        v_stage[v_slot_base + (v_d_base + i) * pv_stride + tile_row] = (
                            v_rope_values[i]
                        )

            previous_p_values = fx.Vector.filled(8, 0.0, fx.BFloat16)
            previous_high_scores = fx.Vector.filled(4, 0.0, fx.Float32)
            previous_low_sum = fx.Float32(0.0)
            while tile_offset < length:
                tile_row = tid // fx.Int32(V_STAGE_THREADS_PER_ROW)
                tile_dword = tid % fx.Int32(V_STAGE_THREADS_PER_ROW)
                if fx.const_expr(rolling_pipeline):
                    tile_index = tile_offset // fx.Int32(QK_BLOCK_COLS)
                    stage_slot = tile_index & fx.Int32(1)
                    k_slot_base = stage_slot * fx.Int32(packed_slot_dwords)
                    rope_slot_base = stage_slot * fx.Int32(rope_slot_dwords)
                    valid_slot_base = stage_slot * fx.Int32(QK_BLOCK_COLS)
                    next_tile_offset = tile_offset + fx.Int32(QK_BLOCK_COLS)
                    has_next_tile = next_tile_offset < length
                    if has_next_tile:
                        stage_tile(
                            stage_slot ^ fx.Int32(1),
                            next_tile_offset,
                            valid_stage,
                            next_tile_offset + fx.Int32(QK_BLOCK_COLS) <= length,
                        )
                else:
                    k_slot_base = fx.Int32(0)
                    rope_slot_base = fx.Int32(0)
                    valid_slot_base = fx.Int32(0)
                    logical_row = tile_offset + tile_row
                    in_range = logical_row < length
                    safe_logical_row = in_range.select(logical_row, fx.Int32(0))
                    source_row = fx.Int32(indices_i32[start + safe_logical_row])
                    row_valid = in_range & (source_row >= fx.Int32(0))
                    safe_source_row = row_valid.select(source_row, fx.Int32(0))
                    source_base = safe_source_row * row_dwords
                    copy_base = tile_dword * fx.Int32(8)
                    raw_lo = fx.Vector(
                        GTensor(source_tensor, dtype=T.i32, shape=(-1,)).vec_load(
                            (source_base + copy_base,), vec_size=4
                        )
                    )
                    raw_hi = fx.Vector(
                        GTensor(source_tensor, dtype=T.i32, shape=(-1,)).vec_load(
                            (source_base + copy_base + 4,), vec_size=4
                        )
                    )
                    for i in fx.range_constexpr(4):
                        k_stage[k_slot_base + tile_row * row_dwords + copy_base + i] = (
                            row_valid.select(fx.Int32(raw_lo[i]), fx.Int32(0))
                        )
                        k_stage[
                            k_slot_base + tile_row * row_dwords + copy_base + 4 + i
                        ] = row_valid.select(fx.Int32(raw_hi[i]), fx.Int32(0))
                    source_rope_base = safe_source_row * rope_row_dwords
                    rope_raw = fx.Vector(
                        GTensor(source_rope_tensor, dtype=T.i32, shape=(-1,)).vec_load(
                            (source_rope_base + tile_dword * 2,), vec_size=2
                        )
                    )
                    for part in fx.range_constexpr(2):
                        k_rope_stage[
                            rope_slot_base
                            + tile_row * rope_row_dwords
                            + tile_dword * 2
                            + part
                        ] = row_valid.select(fx.Int32(rope_raw[part]), fx.Int32(0))
                    if tile_dword == fx.Int32(0):
                        valid_stage[valid_slot_base + tile_row] = row_valid.select(
                            fx.Int32(1), fx.Int32(0)
                        )
                    fx.rocdl.s_barrier()

                def load_q_operand(step):
                    step_base = step * 32 + lane_group * fx.Int32(4)
                    if fx.const_expr(cache_all_q and step < 3):
                        return fx.Vector.from_elements(
                            [
                                fx.Int32(cached_q_operands[step * 8 + i])
                                for i in fx.range_constexpr(8)
                            ],
                            fx.Int32,
                        ).ir_value()
                    if fx.const_expr(stage_q and step < 3):
                        q_stage_base = tid * fx.Int32(q_stage_stride) + step * 8
                        return fx.Vector.from_elements(
                            [
                                fx.Int32(q_smem[q_stage_base + i])
                                for i in fx.range_constexpr(8)
                            ],
                            fx.Int32,
                        ).ir_value()
                    if fx.const_expr(step == cache_q_step):
                        return cached_q_operand
                    if fx.const_expr(step == 3):
                        q_lo = q_tail_words
                        q_hi = fx.Vector.filled(4, 0, fx.Int32)
                    else:
                        q_lo = GTensor(q_tensor, dtype=T.i32, shape=(-1,)).vec_load(
                            (q_base + step_base,), vec_size=4
                        )
                        q_hi = GTensor(q_tensor, dtype=T.i32, shape=(-1,)).vec_load(
                            (q_base + step_base + 16,), vec_size=4
                        )
                    return _join_i32x4(q_lo, q_hi)

                packed_k_step_bases = []
                packed_k_scale_bases = []
                rope_k_row_bases = []
                for nt in fx.range_constexpr(2):
                    k_row = fx.Int32(nt * 16) + lane_row
                    if fx.const_expr(wave_padded_k):
                        packed_row_base = (
                            k_slot_base
                            + (k_row // fx.Int32(2)) * fx.Int32(raw_wave_stride_dwords)
                            + (k_row & fx.Int32(1)) * fx.Int32(row_dwords)
                        )
                        packed_k_step_bases.append(
                            [
                                packed_row_base + fx.Int32(step * 32)
                                for step in fx.range_constexpr(4)
                            ]
                        )
                        packed_k_scale_bases.append(packed_row_base + fx.Int32(112))
                        rope_k_row_bases.append(
                            rope_slot_base
                            + (k_row // fx.Int32(8)) * fx.Int32(raw_wave_stride_dwords)
                            + (k_row & fx.Int32(7)) * fx.Int32(rope_row_dwords)
                        )
                    else:
                        packed_k_step_bases.append(
                            [
                                k_slot_base + k_row * row_dwords + fx.Int32(step * 32)
                                for step in fx.range_constexpr(4)
                            ]
                        )
                        packed_k_scale_bases.append(
                            k_slot_base + k_row * row_dwords + fx.Int32(112)
                        )
                        rope_k_row_bases.append(
                            rope_slot_base + k_row * rope_row_dwords
                        )

                def load_k_operand(nt, step):
                    step_base = lane_group * fx.Int32(4)
                    k_lo = fx.Vector.from_elements(
                        [
                            fx.Int32(
                                k_stage[
                                    packed_k_step_bases[nt][step]
                                    + step_base
                                    + fx.Int32(i)
                                ]
                            )
                            for i in fx.range_constexpr(4)
                        ],
                        fx.Int32,
                    )
                    if fx.const_expr(step == 3):
                        k_hi = fx.Vector.filled(4, 0, fx.Int32)
                    else:
                        k_hi = fx.Vector.from_elements(
                            [
                                fx.Int32(
                                    k_stage[
                                        packed_k_step_bases[nt][step]
                                        + step_base
                                        + fx.Int32(16 + i)
                                    ]
                                )
                                for i in fx.range_constexpr(4)
                            ],
                            fx.Int32,
                        )
                    return _join_i32x4(k_lo, k_hi)

                def load_q_rope(step):
                    step_base = step * 16 + lane_group * fx.Int32(4)
                    if fx.const_expr(step == 0):
                        return q_rope_lo
                    if fx.const_expr(cache_all_q and step == 1):
                        return fx.Vector(cached_q_rope_hi).bitcast(fx.BFloat16)
                    return fx.Vector(
                        GTensor(q_rope_tensor, dtype=T.i32, shape=(-1,)).vec_load(
                            (q_rope_base + step_base,), vec_size=4
                        )
                    ).bitcast(fx.BFloat16)

                def load_k_rope(nt, step):
                    step_base = step * 16 + lane_group * fx.Int32(4)
                    return fx.Vector.from_elements(
                        [
                            fx.Int32(
                                k_rope_stage[
                                    rope_k_row_bases[nt] + step_base + fx.Int32(i)
                                ]
                            )
                            for i in fx.range_constexpr(4)
                        ],
                        fx.Int32,
                    ).bitcast(fx.BFloat16)

                score_fragments = []
                if fx.const_expr(setprio):
                    fx.rocdl.s_setprio(1)
                if fx.const_expr(reuse_q_across_n):
                    qk_acc = [
                        fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
                        for _ in fx.range_constexpr(2)
                    ]
                    scale_k_values = []
                    for nt in fx.range_constexpr(2):
                        if fx.const_expr(permute_k_scales):
                            scale_k_values.append(
                                _load_permuted_opsel_scales(
                                    k_stage,
                                    packed_k_scale_bases[nt],
                                    lane_group,
                                )
                            )
                        else:
                            k_scale_words = fx.Vector.from_elements(
                                [
                                    fx.Int32(
                                        k_stage[packed_k_scale_bases[nt] + fx.Int32(i)]
                                    )
                                    for i in fx.range_constexpr(4)
                                ],
                                fx.Int32,
                            )
                            scale_k_values.append(
                                _pack_opsel_scales(k_scale_words, lane_group)
                            )
                    for step in fx.range_constexpr(4):
                        q_operand = load_q_operand(step)
                        for nt in fx.range_constexpr(2):
                            k_operand = load_k_operand(nt, step)
                            qk_acc[nt] = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.f32x4,
                                [
                                    k_operand,
                                    q_operand,
                                    qk_acc[nt],
                                    0,
                                    0,
                                    step,
                                    scale_k_values[nt],
                                    step,
                                    scale_q,
                                ],
                            )
                    for step in fx.range_constexpr(2):
                        q_values = load_q_rope(step)
                        for nt in fx.range_constexpr(2):
                            frag_c.store(fx.Vector(qk_acc[nt]))
                            frag_a.store(load_k_rope(nt, step))
                            frag_b.store(q_values)
                            fx.mma_atom_call(bf16_mma, frag_c, frag_a, frag_b, frag_c)
                            qk_acc[nt] = fx.Vector(frag_c.load()).ir_value()
                    for nt in fx.range_constexpr(2):
                        score_fragments.append(fx.Vector(qk_acc[nt]))
                else:
                    for nt in fx.range_constexpr(2):
                        if fx.const_expr(permute_k_scales):
                            scale_k = _load_permuted_opsel_scales(
                                k_stage,
                                packed_k_scale_bases[nt],
                                lane_group,
                            )
                        else:
                            k_scale_words = fx.Vector.from_elements(
                                [
                                    fx.Int32(
                                        k_stage[packed_k_scale_bases[nt] + fx.Int32(i)]
                                    )
                                    for i in fx.range_constexpr(4)
                                ],
                                fx.Int32,
                            )
                            scale_k = _pack_opsel_scales(k_scale_words, lane_group)
                        acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
                        for step in fx.range_constexpr(4):
                            q_operand = load_q_operand(step)
                            k_operand = load_k_operand(nt, step)
                            acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.f32x4,
                                [
                                    k_operand,
                                    q_operand,
                                    acc,
                                    0,
                                    0,
                                    step,
                                    scale_k,
                                    step,
                                    scale_q,
                                ],
                            )
                        frag_c.store(fx.Vector(acc))
                        for step in fx.range_constexpr(2):
                            q_values = load_q_rope(step)
                            frag_a.store(load_k_rope(nt, step))
                            frag_b.store(q_values)
                            fx.mma_atom_call(bf16_mma, frag_c, frag_a, frag_b, frag_c)
                        score_fragments.append(fx.Vector(frag_c.load()))

                # The cluster pipeline leaves the high K16 half of tile i-1
                # live across tile i's QK.  Completing that half here makes P
                # for the previous tile visible before its PV phase, matching
                # the dependency order of OPUS clusters 1/5.
                if fx.const_expr(cluster_two):
                    if tile_offset > fx.Int32(0):
                        previous_slot = stage_slot ^ fx.Int32(1)
                        previous_p_slot_base = previous_slot * fx.Int32(p_slot_elems)
                        previous_high_sum = fx.Float32(0.0)
                        previous_high_p = []
                        for i in fx.range_constexpr(4):
                            p = _exp2_f32(previous_high_scores[i])
                            previous_high_p.append(p)
                            previous_high_sum = previous_high_sum + p
                        previous_high_sum = (
                            previous_high_sum + previous_high_sum.shuffle_xor(16, 64)
                        )
                        previous_high_sum = (
                            previous_high_sum + previous_high_sum.shuffle_xor(32, 64)
                        )
                        l_state = l_state + previous_low_sum + previous_high_sum
                        previous_p_col = fx.Int32(16) + lane_group * fx.Int32(4)
                        for i in fx.range_constexpr(4):
                            p_stage[
                                previous_p_slot_base
                                + head * pv_stride
                                + previous_p_col
                                + i
                            ] = previous_high_p[i].to(fx.BFloat16)

                # OPUS clusters dequantize and publish V for the current tile
                # before consuming P/V from the previous tile.  Reuse the
                # normal end-of-iteration barrier here so the split softmax
                # does not add another CTA synchronization to the hot loop.
                if fx.const_expr(cluster_two):
                    current_v_slot_base = stage_slot * fx.Int32(v_slot_elems)
                    stage_v_tile(k_slot_base, rope_slot_base, current_v_slot_base)
                    if has_next_tile:
                        fx.rocdl.s_waitcnt(0)
                        fx.rocdl.sched_barrier(0)
                    fx.rocdl.s_barrier()

                # Keep the current QK result live while consuming the previous
                # tile's P/V operands.  This is the intended rolling compute
                # phase; current softmax must remain below it because alpha is
                # the recurrence factor belonging to the previous tile.
                if fx.const_expr(rolling_pipeline):
                    if tile_offset > fx.Int32(0):
                        if fx.const_expr(fixed_softmax_ref):
                            rescale_previous = False
                        elif fx.const_expr(rescale_once):
                            rescale_previous = tile_offset == fx.Int32(QK_BLOCK_COLS)
                        else:
                            rescale_previous = True
                        if fx.const_expr(not cluster_two):
                            run_staged_pv(
                                stage_slot ^ fx.Int32(1),
                                rescale_previous,
                                previous_p_values,
                                frag_a,
                                frag_b,
                                frag_c,
                                output_acc,
                            )
                        else:
                            run_staged_pv(
                                stage_slot ^ fx.Int32(1),
                                False,
                                previous_p_values,
                                frag_a,
                                frag_b,
                                frag_c,
                                output_acc,
                            )
                if fx.const_expr(setprio):
                    fx.rocdl.s_setprio(0)

                scaled_scores = []
                if fx.const_expr(not fixed_softmax_ref):
                    tile_max = fx.Float32(-float("inf"))
                for nt in fx.range_constexpr(2):
                    values = []
                    for i in fx.range_constexpr(4):
                        key = fx.Int32(nt * 16 + lane_group * 4 + i)
                        score = fx.Float32(score_fragments[nt][i]) * temperature_scale
                        if fx.const_expr(no_sentinel):
                            if fx.const_expr(assume_segment_full):
                                pass
                            elif fx.const_expr(full_tile_fastpath):
                                if tile_offset + fx.Int32(QK_BLOCK_COLS) <= length:
                                    pass
                                else:
                                    score = (tile_offset + key < length).select(
                                        score, fx.Float32(-float("inf"))
                                    )
                            else:
                                score = (tile_offset + key < length).select(
                                    score, fx.Float32(-float("inf"))
                                )
                        else:
                            score = (
                                valid_stage[valid_slot_base + key] != fx.Int32(0)
                            ).select(score, fx.Float32(-float("inf")))
                        values.append(score)
                        if fx.const_expr(not fixed_softmax_ref):
                            tile_max = fx.max(tile_max, score)
                    scaled_scores.append(values)
                if fx.const_expr(fixed_softmax_ref):
                    new_m = m_state
                    alpha = fx.Float32(1.0)
                elif fx.const_expr(rescale_once):
                    tile_max = fx.max(tile_max, tile_max.shuffle_xor(16, 64))
                    tile_max = fx.max(tile_max, tile_max.shuffle_xor(32, 64))
                    rescale_current = tile_offset == fx.Int32(0)
                    m_candidate = fx.max(m_state, tile_max)
                    new_m = rescale_current.select(m_candidate, m_state)
                    alpha_candidate = _exp2_f32(m_state - new_m)
                    alpha = rescale_current.select(alpha_candidate, fx.Float32(1.0))
                else:
                    tile_max = fx.max(tile_max, tile_max.shuffle_xor(16, 64))
                    tile_max = fx.max(tile_max, tile_max.shuffle_xor(32, 64))
                    new_m = fx.max(m_state, tile_max)
                    alpha = _exp2_f32(m_state - new_m)
                m_state = new_m

                p_fragments = []
                tile_sum = fx.Float32(0.0)
                softmax_fragments = 1 if cluster_two else 2
                for nt in fx.range_constexpr(softmax_fragments):
                    values = []
                    for i in fx.range_constexpr(4):
                        p = _exp2_f32(scaled_scores[nt][i] - new_m)
                        values.append(p)
                        tile_sum = tile_sum + p
                    p_fragments.append(values)
                tile_sum = tile_sum + tile_sum.shuffle_xor(16, 64)
                tile_sum = tile_sum + tile_sum.shuffle_xor(32, 64)
                if fx.const_expr(cluster_two):
                    previous_low_sum = tile_sum
                    previous_high_scores = fx.Vector.from_elements(
                        [scaled_scores[1][i] for i in fx.range_constexpr(4)],
                        fx.Float32,
                    )
                elif fx.const_expr(fixed_softmax_ref):
                    l_state = l_state + tile_sum
                elif fx.const_expr(rescale_once):
                    l_state = rescale_current.select(
                        l_state * alpha + tile_sum, l_state + tile_sum
                    )
                else:
                    l_state = l_state * alpha + tile_sum
                if fx.const_expr(rolling_pipeline):
                    if fx.const_expr(fixed_softmax_ref):
                        pass
                    elif fx.const_expr(rescale_once):
                        if rescale_current:
                            alpha_stage[head] = alpha
                    else:
                        alpha_stage[head] = alpha
                elif fx.const_expr(alpha_bpermute):
                    alpha_values = []
                    for i in fx.range_constexpr(4):
                        source_lane = lane_group * fx.Int32(4) + fx.Int32(i)
                        alpha_values.append(
                            fx.Int32(
                                fx.rocdl.ds_bpermute(
                                    T.i32,
                                    (source_lane * fx.Int32(4)).ir_value(),
                                    alpha.bitcast(fx.Int32).ir_value(),
                                )
                            ).bitcast(fx.Float32)
                        )
                    alpha_vector = fx.Vector.from_elements(alpha_values, fx.Float32)
                else:
                    alpha_stage[head] = alpha

                if fx.const_expr(rolling_pipeline and not register_p):
                    p_slot_base = stage_slot * fx.Int32(p_slot_elems)
                    p_store_fragments = 1 if cluster_two else 2
                    for nt in fx.range_constexpr(p_store_fragments):
                        p_col = fx.Int32(nt * 16) + lane_group * fx.Int32(4)
                        if fx.const_expr(p_lane_layout):
                            consumer_group = fx.Int32(nt * 2) + lane_group // fx.Int32(
                                2
                            )
                            consumer_lane = consumer_group * fx.Int32(16) + lane_row
                            p_write_base = (
                                p_slot_base
                                + wave * fx.Int32(64 * p_lane_stride)
                                + consumer_lane * fx.Int32(p_lane_stride)
                                + (lane_group & fx.Int32(1)) * fx.Int32(4)
                            )
                        for i in fx.range_constexpr(4):
                            if fx.const_expr(p_lane_layout):
                                p_stage[p_write_base + i] = p_fragments[nt][i].to(
                                    fx.BFloat16
                                )
                            else:
                                p_stage[p_slot_base + head * pv_stride + p_col + i] = (
                                    p_fragments[nt][i].to(fx.BFloat16)
                                )
                else:
                    # Convert the two QK C-fragments into the K32 PV A-operand
                    # layout without an LDS round trip.  Destination lane
                    # groups 0/1 consume keys 0..15, groups 2/3 keys 16..31.
                    source_lane0 = (lane_group & fx.Int32(1)) * fx.Int32(32) + lane_row
                    source_lane1 = source_lane0 + fx.Int32(16)
                    use_high_nt = lane_group >= fx.Int32(2)
                    p_values_f32 = []
                    for source_lane in (source_lane0, source_lane1):
                        for i in fx.range_constexpr(4):
                            source_byte = source_lane * fx.Int32(4)
                            p0 = fx.Float32(
                                fx.Int32(
                                    fx.rocdl.ds_bpermute(
                                        T.i32,
                                        source_byte.ir_value(),
                                        p_fragments[0][i].bitcast(fx.Int32).ir_value(),
                                    )
                                ).bitcast(fx.Float32)
                            )
                            p1 = fx.Float32(
                                fx.Int32(
                                    fx.rocdl.ds_bpermute(
                                        T.i32,
                                        source_byte.ir_value(),
                                        p_fragments[1][i].bitcast(fx.Int32).ir_value(),
                                    )
                                ).bitcast(fx.Float32)
                            )
                            p_values_f32.append(use_high_nt.select(p1, p0))
                    p_values = fx.Vector.from_elements(p_values_f32, fx.Float32).to(
                        fx.BFloat16
                    )
                    if fx.const_expr(register_p):
                        previous_p_values = p_values

                # Decode all 512 V values and transpose them to [D, K].
                if fx.const_expr(rolling_pipeline):
                    v_slot_base = stage_slot * fx.Int32(v_slot_elems)
                else:
                    v_slot_base = fx.Int32(0)
                if fx.const_expr(not cluster_two):
                    stage_v_tile(k_slot_base, rope_slot_base, v_slot_base)

                if fx.const_expr(rolling_pipeline):
                    if fx.const_expr(not cluster_two):
                        if has_next_tile:
                            fx.rocdl.s_waitcnt(0)
                            fx.rocdl.sched_barrier(0)
                        fx.rocdl.s_barrier()
                else:
                    fx.rocdl.s_barrier()
                    frag_a.store(p_values)
                    out_row = wave * fx.Int32(16) + lane_group * fx.Int32(4)
                    for ds in fx.range_constexpr(PREFILL_D_PER_CTA // 16):
                        d_col = fx.Int32(ds * 16) + lane_row
                        v_values = fx.Vector.from_elements(
                            [
                                fx.BFloat16(
                                    v_stage[
                                        d_col * QK_BLOCK_COLS
                                        + lane_group * fx.Int32(8)
                                        + fx.Int32(j)
                                    ]
                                )
                                for j in fx.range_constexpr(8)
                            ],
                            fx.BFloat16,
                        )
                        old = fx.Vector(output_acc[ds].load())
                        if fx.const_expr(alpha_bpermute):
                            scaled_old = old * alpha_vector
                        else:
                            scaled_old = fx.Vector.from_elements(
                                [
                                    old[i] * fx.Float32(alpha_stage[out_row + i])
                                    for i in fx.range_constexpr(4)
                                ],
                                fx.Float32,
                            )
                        frag_b.store(v_values)
                        frag_c.store(scaled_old)
                        fx.mma_atom_call(bf16_mma, frag_c, frag_a, frag_b, frag_c)
                        output_acc[ds].store(fx.Vector(frag_c.load()))
                    fx.rocdl.s_barrier()
                tile_offset = tile_offset + fx.Int32(QK_BLOCK_COLS)

            if fx.const_expr(rolling_pipeline):
                if length > fx.Int32(0):
                    last_slot = ((length - fx.Int32(1)) // fx.Int32(32)) & fx.Int32(1)
                    if fx.const_expr(cluster_two):
                        last_p_slot_base = last_slot * fx.Int32(p_slot_elems)
                        last_high_sum = fx.Float32(0.0)
                        last_high_p = []
                        for i in fx.range_constexpr(4):
                            p = _exp2_f32(previous_high_scores[i])
                            last_high_p.append(p)
                            last_high_sum = last_high_sum + p
                        last_high_sum = last_high_sum + last_high_sum.shuffle_xor(
                            16, 64
                        )
                        last_high_sum = last_high_sum + last_high_sum.shuffle_xor(
                            32, 64
                        )
                        l_state = l_state + previous_low_sum + last_high_sum
                        last_p_col = fx.Int32(16) + lane_group * fx.Int32(4)
                        for i in fx.range_constexpr(4):
                            p_stage[
                                last_p_slot_base + head * pv_stride + last_p_col + i
                            ] = last_high_p[i].to(fx.BFloat16)
                        fx.rocdl.s_waitcnt(lgkmcnt=0)
                        fx.rocdl.sched_barrier(0)
                        run_staged_pv(
                            last_slot,
                            False,
                            previous_p_values,
                            frag_a,
                            frag_b,
                            frag_c,
                            output_acc,
                        )
                    else:
                        if fx.const_expr(fixed_softmax_ref):
                            rescale_last = False
                        elif fx.const_expr(rescale_once):
                            rescale_last = length <= fx.Int32(QK_BLOCK_COLS)
                        else:
                            rescale_last = True
                        run_staged_pv(
                            last_slot,
                            rescale_last,
                            previous_p_values,
                            frag_a,
                            frag_b,
                            frag_c,
                            output_acc,
                        )
                    fx.rocdl.s_barrier()
            return m_state, l_state

        def run_segment(
            source_tensor,
            source_rope_tensor,
            source_div,
            source_rope_div,
            indices_i32,
            start,
            length,
            m_state,
            l_state,
            assume_segment_full,
        ):
            return process_segment(
                source_tensor,
                source_rope_tensor,
                source_div,
                source_rope_div,
                indices_i32,
                start,
                length,
                m_state,
                l_state,
                k_smem,
                k_rope_smem,
                p_smem,
                v_smem,
                valid_smem,
                alpha_smem,
                mma_a,
                mma_b,
                mma_c,
                q_packed,
                q_rope,
                assume_segment_full,
            )

        if fx.const_expr(split_partial):
            prefix_start = fx.Int32(split_task_start_i32[task])
            prefix_len = fx.Int32(split_task_len_i32[task])
        else:
            prefix_start = fx.Int32(prefix_indptr_i32[query])
            prefix_len = fx.Int32(prefix_indptr_i32[query + 1]) - prefix_start
        if fx.const_expr(dynamic_full_prefix):
            if (prefix_len & fx.Int32(QK_BLOCK_COLS - 1)) == fx.Int32(0):
                m_row, l_row = run_segment(
                    prefix_packed,
                    prefix_rope,
                    prefix_div,
                    prefix_rope_div,
                    prefix_indices_i32,
                    prefix_start,
                    prefix_len,
                    m_row,
                    l_row,
                    True,
                )
            else:
                m_row, l_row = run_segment(
                    prefix_packed,
                    prefix_rope,
                    prefix_div,
                    prefix_rope_div,
                    prefix_indices_i32,
                    prefix_start,
                    prefix_len,
                    m_row,
                    l_row,
                    False,
                )
        else:
            m_row, l_row = run_segment(
                prefix_packed,
                prefix_rope,
                prefix_div,
                prefix_rope_div,
                prefix_indices_i32,
                prefix_start,
                prefix_len,
                m_row,
                l_row,
                assume_full_tiles,
            )
        if fx.const_expr(not prefix_only):
            extend_start = fx.Int32(extend_indptr_i32[query])
            extend_len = fx.Int32(extend_indptr_i32[query + 1]) - extend_start
            m_row, l_row = run_segment(
                extend_packed,
                extend_rope,
                extend_div,
                extend_rope_div,
                extend_indices_i32,
                extend_start,
                extend_len,
                m_row,
                l_row,
                assume_full_tiles,
            )

        out_row = wave * fx.Int32(16) + lane_group * fx.Int32(4)
        if fx.const_expr(split_partial):
            partial_row = fx.Int32(split_task_row_i32[task])
            if lane_group == fx.Int32(0):
                partial_meta_offset = partial_row * QK_BLOCK_ROWS + head
                partial_m_f32[partial_meta_offset] = m_row
                partial_l_f32[partial_meta_offset] = l_row
            out_base = partial_row * QK_BLOCK_ROWS * V4_DIM_QK
            for ds in fx.range_constexpr(PREFILL_D_PER_CTA // 16):
                values = fx.Vector(output_acc[ds].load())
                out_col = fx.Int32(ds * 16) + lane_row
                for i in fx.range_constexpr(4):
                    out_f16[out_base + (out_row + i) * V4_DIM_QK + out_col] = values[
                        i
                    ].to(fx.Float16)
        else:
            sink_log2 = fx.Float32(sink_f32[head]) * fx.Float32(LOG2E)
            final_m = fx.max(m_row, sink_log2)
            final_alpha = _exp2_f32(m_row - final_m)
            denom = l_row * final_alpha + _exp2_f32(sink_log2 - final_m)
            final_scale = final_alpha / denom
            if fx.const_expr(alpha_bpermute):
                final_scale_values = []
                for i in fx.range_constexpr(4):
                    source_lane = lane_group * fx.Int32(4) + fx.Int32(i)
                    final_scale_values.append(
                        fx.Int32(
                            fx.rocdl.ds_bpermute(
                                T.i32,
                                (source_lane * fx.Int32(4)).ir_value(),
                                final_scale.bitcast(fx.Int32).ir_value(),
                            )
                        ).bitcast(fx.Float32)
                    )
                final_scale_vector = fx.Vector.from_elements(
                    final_scale_values, fx.Float32
                )
            else:
                alpha_smem[head] = final_scale
                fx.rocdl.s_barrier()

            out_base = query * QK_BLOCK_ROWS * V4_DIM_QK
            for ds in fx.range_constexpr(PREFILL_D_PER_CTA // 16):
                values = fx.Vector(output_acc[ds].load())
                out_col = fx.Int32(ds * 16) + lane_row
                for i in fx.range_constexpr(4):
                    if fx.const_expr(alpha_bpermute):
                        scale = fx.Float32(final_scale_vector[i])
                    else:
                        scale = fx.Float32(alpha_smem[out_row + i])
                    out_bf16[out_base + (out_row + i) * V4_DIM_QK + out_col] = (
                        values[i] * scale
                    ).to(fx.BFloat16)

    @flyc.jit
    def launch(
        q_packed: fx.Pointer,
        q_rope: fx.Pointer,
        prefix_packed: fx.Pointer,
        prefix_rope: fx.Pointer,
        prefix_indices: fx.Pointer,
        prefix_indptr: fx.Pointer,
        extend_packed: fx.Pointer,
        extend_rope: fx.Pointer,
        extend_indices: fx.Pointer,
        extend_indptr: fx.Pointer,
        attn_sink: fx.Pointer,
        out: fx.Pointer,
        split_task_query: fx.Pointer,
        split_task_start: fx.Pointer,
        split_task_len: fx.Pointer,
        split_task_row: fx.Pointer,
        partial_m: fx.Pointer,
        partial_l: fx.Pointer,
        softmax_scale: fx.Float32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q_packed,
            q_rope,
            prefix_packed,
            prefix_rope,
            prefix_indices,
            prefix_indptr,
            extend_packed,
            extend_rope,
            extend_indices,
            extend_indptr,
            attn_sink,
            out,
            split_task_query,
            split_task_start,
            split_task_len,
            split_task_row,
            partial_m,
            partial_l,
            softmax_scale,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(QK_WAVES * 64, 1, 1),
            stream=stream,
            value_attrs=kernel_value_attrs,
        )

    return launch


def stage_v4_paged_tile_flydsl(
    packed: torch.Tensor,
    rope: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Gather and decode V4 FP8 rows into contiguous BF16 rows."""
    if not (packed.is_cuda and rope.is_cuda and indices.is_cuda):
        raise RuntimeError("packed, rope, and indices must be CUDA/HIP tensors")
    if packed.dtype != dtypes.fp8 or packed.ndim != 2 or packed.shape[1] != 512:
        raise RuntimeError(
            f"packed must be {dtypes.fp8} shape [N,512], got "
            f"dtype={packed.dtype} shape={tuple(packed.shape)}"
        )
    if rope.dtype != torch.bfloat16 or rope.shape != (packed.shape[0], 64):
        raise RuntimeError(
            f"rope must be bf16 shape {(packed.shape[0], 64)}, got "
            f"dtype={rope.dtype} shape={tuple(rope.shape)}"
        )
    if indices.dtype != torch.int32 or indices.ndim != 1 or not indices.is_contiguous():
        raise RuntimeError("indices must be a contiguous int32 vector")
    if indices.numel() == 0:
        if out is None:
            return torch.empty((0, 512), dtype=torch.bfloat16, device=packed.device)
        return out
    if out is None:
        out = torch.empty(
            (indices.numel(), V4_DIM_QK),
            dtype=torch.bfloat16,
            device=packed.device,
        )
    elif (
        out.dtype != torch.bfloat16
        or out.shape != (indices.numel(), V4_DIM_QK)
        or not out.is_contiguous()
    ):
        raise RuntimeError(
            f"out must be contiguous bf16 shape {(indices.numel(), V4_DIM_QK)}, "
            f"got dtype={out.dtype} shape={tuple(out.shape)}"
        )

    if stream is None:
        stream = torch.cuda.current_stream(packed.device)
    launcher = _build_v_stage_launcher()
    grid_x = (indices.numel() + V_STAGE_ROWS - 1) // V_STAGE_ROWS
    with torch.cuda.device(packed.device.index):
        _run_compiled(
            launcher,
            packed,
            rope,
            indices,
            out,
            int(indices.numel()),
            int(grid_x),
            stream,
        )
    return out


def v4_nope_qk_tile_flydsl(
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute one 16x16 QK tile over the 448-dimensional V4 NoPE segment."""
    for name, tensor in (("q_packed", q_packed), ("k_packed", k_packed)):
        if (
            not tensor.is_cuda
            or tensor.dtype != dtypes.fp8
            or tensor.shape != (16, V4_PACKED_ROW_BYTES)
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"{name} must be contiguous {dtypes.fp8} shape (16,512), got "
                f"dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
    if out is None:
        out = torch.empty((16, 16), dtype=torch.float32, device=q_packed.device)
    elif out.dtype != torch.float32 or out.shape != (16, 16) or not out.is_contiguous():
        raise RuntimeError("out must be contiguous float32 shape (16,16)")
    if stream is None:
        stream = torch.cuda.current_stream(q_packed.device)
    with torch.cuda.device(q_packed.device.index):
        _run_compiled(_build_nope_qk_launcher(), q_packed, k_packed, out, stream)
    return out


def v4_qk_tile_flydsl(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute one 16x16 V4 QK tile over FP8 NoPE plus BF16 RoPE."""
    for name, tensor in (("q_packed", q_packed), ("k_packed", k_packed)):
        if (
            not tensor.is_cuda
            or tensor.dtype != dtypes.fp8
            or tensor.shape != (16, V4_PACKED_ROW_BYTES)
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"{name} must be contiguous {dtypes.fp8} shape (16,512), got "
                f"dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
    for name, tensor in (("q_rope", q_rope), ("k_rope", k_rope)):
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.bfloat16
            or tensor.shape != (16, V4_DIM_ROPE)
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"{name} must be contiguous bf16 shape (16,64), got "
                f"dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
    if not (q_packed.device == q_rope.device == k_packed.device == k_rope.device):
        raise RuntimeError("all Q/K tensors must be on the same device")
    if out is None:
        out = torch.empty((16, 16), dtype=torch.float32, device=q_packed.device)
    elif out.dtype != torch.float32 or out.shape != (16, 16) or not out.is_contiguous():
        raise RuntimeError("out must be contiguous float32 shape (16,16)")
    if stream is None:
        stream = torch.cuda.current_stream(q_packed.device)
    with torch.cuda.device(q_packed.device.index):
        _run_compiled(
            _build_qk_launcher(),
            q_packed,
            q_rope,
            k_packed,
            k_rope,
            out,
            stream,
        )
    return out


def v4_qk_h128_tile_flydsl(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute the eight-wave 128x32 QK tile used by sparse prefill."""
    expected = (
        ("q_packed", q_packed, (QK_BLOCK_ROWS, V4_PACKED_ROW_BYTES), dtypes.fp8),
        ("q_rope", q_rope, (QK_BLOCK_ROWS, V4_DIM_ROPE), torch.bfloat16),
        ("k_packed", k_packed, (QK_BLOCK_COLS, V4_PACKED_ROW_BYTES), dtypes.fp8),
        ("k_rope", k_rope, (QK_BLOCK_COLS, V4_DIM_ROPE), torch.bfloat16),
    )
    for name, tensor, shape, dtype in expected:
        if (
            not tensor.is_cuda
            or tensor.dtype != dtype
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"{name} must be contiguous {dtype} shape {shape}, got "
                f"dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
    if not (q_packed.device == q_rope.device == k_packed.device == k_rope.device):
        raise RuntimeError("all Q/K tensors must be on the same device")
    if out is None:
        out = torch.empty(
            (QK_BLOCK_ROWS, QK_BLOCK_COLS),
            dtype=torch.float32,
            device=q_packed.device,
        )
    elif (
        out.dtype != torch.float32
        or out.shape != (QK_BLOCK_ROWS, QK_BLOCK_COLS)
        or not out.is_contiguous()
    ):
        raise RuntimeError("out must be contiguous float32 shape (128,32)")
    if stream is None:
        stream = torch.cuda.current_stream(q_packed.device)
    with torch.cuda.device(q_packed.device.index):
        _run_compiled(
            _build_qk_h128_launcher(),
            q_packed,
            q_rope,
            k_packed,
            k_rope,
            out,
            stream,
        )
    return out


def v4_qk_pv_h128_tile_flydsl(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run the fused QK/PV register-layout probe for one 128x32 tile."""
    expected = (
        ("q_packed", q_packed, (QK_BLOCK_ROWS, V4_PACKED_ROW_BYTES), dtypes.fp8),
        ("q_rope", q_rope, (QK_BLOCK_ROWS, V4_DIM_ROPE), torch.bfloat16),
        ("k_packed", k_packed, (QK_BLOCK_COLS, V4_PACKED_ROW_BYTES), dtypes.fp8),
        ("k_rope", k_rope, (QK_BLOCK_COLS, V4_DIM_ROPE), torch.bfloat16),
        ("attn_sink", attn_sink, (QK_BLOCK_ROWS,), torch.float32),
    )
    for name, tensor, shape, dtype in expected:
        if (
            not tensor.is_cuda
            or tensor.dtype != dtype
            or tensor.shape != shape
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"{name} must be contiguous {dtype} shape {shape}, got "
                f"dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
    if len({tensor.device for _, tensor, _, _ in expected}) != 1:
        raise RuntimeError("all Q/K/V tensors must be on the same device")
    if out is None:
        out = torch.empty(
            (QK_BLOCK_ROWS, V4_DIM_QK),
            dtype=torch.float32,
            device=q_packed.device,
        )
    elif (
        out.dtype != torch.float32
        or out.shape != (QK_BLOCK_ROWS, V4_DIM_QK)
        or not out.is_contiguous()
    ):
        raise RuntimeError("out must be contiguous float32 shape (128,512)")
    if stream is None:
        stream = torch.cuda.current_stream(q_packed.device)
    with torch.cuda.device(q_packed.device.index):
        _run_compiled(
            _build_qk_pv_h128_launcher(),
            q_packed,
            q_rope,
            k_packed,
            k_rope,
            attn_sink,
            out,
            float(softmax_scale),
            stream,
        )
    return out


def sparse_attn_v4_paged_prefill_fp8_flydsl(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_packed: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
    waves_per_eu: int = 0,
    pipeline_two: bool = False,
    cluster_two: bool = False,
    wave_padded_k: bool = False,
    alpha_bpermute: bool = False,
    lds_padding: int = 0,
    rescale_once: bool = False,
    fixed_softmax_ref: bool = False,
    transpose_v: bool = False,
    register_p: bool = False,
    cache_q_step: int = -1,
    reuse_q_across_n: bool = False,
    broadcast_indices: bool = False,
    no_sentinel: bool = False,
    full_tile_fastpath: bool = False,
    p_lane_layout: bool = False,
    stage_q: bool = False,
    cache_all_q: bool = False,
    assume_full_tiles: bool = False,
    permute_k_scales: bool = False,
    pairwise_pv: bool = False,
    post_misched: bool = False,
    machine_sink: bool = False,
    setprio: bool = False,
    dynamic_full_prefix: bool = False,
    prefix_only: bool = False,
    split_partial: bool = False,
    split_task_query: torch.Tensor | None = None,
    split_task_start: torch.Tensor | None = None,
    split_task_len: torch.Tensor | None = None,
    split_task_row: torch.Tensor | None = None,
    partial_m: torch.Tensor | None = None,
    partial_l: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the benchmark-only H=128 FlyDSL sparse-prefill kernel.

    ``assume_full_tiles`` is a caller contract: every prefix and extend length
    must be divisible by 32 and the index arrays must not contain sentinels.
    ``permute_k_scales`` is a benchmark-only gfx950 scale-packing specialization.
    """
    if (
        not q_packed.is_cuda
        or q_packed.dtype != dtypes.fp8
        or q_packed.ndim != 3
        or q_packed.shape[1:] != (QK_BLOCK_ROWS, V4_PACKED_ROW_BYTES)
        or not q_packed.is_contiguous()
    ):
        raise RuntimeError(
            "q_packed must be contiguous FP8 shape [T,128,512], got "
            f"dtype={q_packed.dtype} shape={tuple(q_packed.shape)}"
        )
    tokens = q_packed.shape[0]
    if (
        q_rope.dtype != torch.bfloat16
        or q_rope.shape != (tokens, QK_BLOCK_ROWS, V4_DIM_ROPE)
        or not q_rope.is_contiguous()
    ):
        raise RuntimeError(
            f"q_rope must be contiguous bf16 shape {(tokens, 128, 64)}, got "
            f"dtype={q_rope.dtype} shape={tuple(q_rope.shape)}"
        )

    unified_kv_packed = unified_kv_packed.reshape(
        unified_kv_packed.shape[0], V4_PACKED_ROW_BYTES
    )
    unified_kv_rope = unified_kv_rope.reshape(unified_kv_rope.shape[0], V4_DIM_ROPE)
    k_packed = k_packed.reshape(k_packed.shape[0], V4_PACKED_ROW_BYTES)
    k_rope = k_rope.reshape(k_rope.shape[0], V4_DIM_ROPE)
    for name, packed, rope in (
        ("prefix", unified_kv_packed, unified_kv_rope),
        ("extend", k_packed, k_rope),
    ):
        if packed.dtype != dtypes.fp8 or not packed.is_contiguous():
            raise RuntimeError(f"{name} packed KV must be contiguous FP8")
        if rope.dtype != torch.bfloat16 or not rope.is_contiguous():
            raise RuntimeError(f"{name} RoPE KV must be contiguous bf16")

    # BufferCopyLDS128b consumes a byte-linear buffer descriptor.  Recasting a
    # rank-2 tensor keeps its multidimensional layout and makes a byte voffset
    # address the wrong logical element; flatten before the kernel builds the
    # byte view.  The serial path keeps its original rank-2 arguments.
    rolling_pipeline = pipeline_two or cluster_two
    if rolling_pipeline:
        unified_kv_packed = unified_kv_packed.reshape(-1)
        unified_kv_rope = unified_kv_rope.reshape(-1)
        k_packed = k_packed.reshape(-1)
        k_rope = k_rope.reshape(-1)

    indices = []
    for name, value in (
        ("kv_indices_prefix", kv_indices_prefix),
        ("kv_indptr_prefix", kv_indptr_prefix),
        ("kv_indices_extend", kv_indices_extend),
        ("kv_indptr_extend", kv_indptr_extend),
    ):
        value = value.to(dtype=torch.int32).contiguous()
        indices.append(value)
        if not value.is_cuda:
            raise RuntimeError(f"{name} must be on the GPU")
    (
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
    ) = indices
    if kv_indptr_prefix.numel() != tokens + 1 or kv_indptr_extend.numel() != tokens + 1:
        raise RuntimeError("prefix and extend indptr must each have T+1 entries")
    if attn_sink.dtype != torch.float32 or attn_sink.shape != (QK_BLOCK_ROWS,):
        raise RuntimeError("attn_sink must be float32 shape (128,)")
    if lds_padding < 0:
        raise RuntimeError("lds_padding must be non-negative")
    if rescale_once and fixed_softmax_ref:
        raise RuntimeError("rescale_once and fixed_softmax_ref are exclusive")
    if register_p and not rolling_pipeline:
        raise RuntimeError("register_p requires pipeline_two or cluster_two")
    if cache_q_step not in (-1, 0, 1, 2):
        raise RuntimeError("cache_q_step must be -1, 0, 1, or 2")
    if full_tile_fastpath and not no_sentinel:
        raise RuntimeError("full_tile_fastpath requires no_sentinel")
    if assume_full_tiles and not no_sentinel:
        raise RuntimeError("assume_full_tiles requires no_sentinel")
    if stage_q and (not rolling_pipeline or not register_p):
        raise RuntimeError("stage_q requires pipeline_two and register_p")
    if stage_q and cache_q_step >= 0:
        raise RuntimeError("stage_q and cache_q_step are exclusive")
    if cache_all_q and (cache_q_step >= 0 or stage_q):
        raise RuntimeError("cache_all_q is exclusive with cache_q_step and stage_q")
    if wave_padded_k and not rolling_pipeline:
        raise RuntimeError("wave_padded_k requires pipeline_two or cluster_two")
    if cluster_two:
        if pipeline_two:
            raise RuntimeError("cluster_two and pipeline_two are exclusive")
        if not fixed_softmax_ref:
            raise RuntimeError("cluster_two currently requires fixed_softmax_ref")
        if not transpose_v or not reuse_q_across_n:
            raise RuntimeError(
                "cluster_two currently requires transpose_v and reuse_q_across_n"
            )
        if not no_sentinel or not full_tile_fastpath:
            raise RuntimeError(
                "cluster_two is exact-full only and requires no_sentinel and "
                "full_tile_fastpath"
            )
        if register_p or p_lane_layout or rescale_once:
            raise RuntimeError(
                "cluster_two does not support register_p, p_lane_layout, or "
                "rescale_once"
            )
    if pairwise_pv and not transpose_v:
        raise RuntimeError("pairwise_pv requires transpose_v")
    if split_partial and (not prefix_only or not fixed_softmax_ref):
        raise RuntimeError(
            "split_partial currently requires prefix_only and fixed_softmax_ref"
        )

    split_tensors = (
        split_task_query,
        split_task_start,
        split_task_len,
        split_task_row,
    )
    if split_partial:
        if any(value is None for value in split_tensors):
            raise RuntimeError("split_partial requires all split task tensors")
        split_tensors = tuple(
            value.to(dtype=torch.int32).contiguous()  # type: ignore[union-attr]
            for value in split_tensors
        )
        if len({value.numel() for value in split_tensors}) != 1:
            raise RuntimeError("split task tensors must have matching lengths")
        if partial_m is None or partial_l is None:
            raise RuntimeError("split_partial requires partial_m and partial_l")
        if (
            partial_m.dtype != torch.float32
            or partial_l.dtype != torch.float32
            or partial_m.shape != partial_l.shape
            or partial_m.ndim != 3
            or partial_m.shape[0] != tokens
            or partial_m.shape[2] != QK_BLOCK_ROWS
            or not partial_m.is_contiguous()
            or not partial_l.is_contiguous()
        ):
            raise RuntimeError(
                "partial_m and partial_l must be contiguous float32 " "shape [T,S,128]"
            )
        partial_shape = (*partial_m.shape, V4_DIM_QK)
        if out is None:
            out = torch.empty(
                partial_shape, dtype=torch.float16, device=q_packed.device
            )
        elif (
            out.dtype != torch.float16
            or out.shape != partial_shape
            or not out.is_contiguous()
        ):
            raise RuntimeError(
                f"split partial out must be contiguous fp16 shape {partial_shape}"
            )
    else:
        # Compile-time-dead split arguments still need concrete ABI values.
        split_tensors = (
            kv_indptr_prefix,
            kv_indptr_prefix,
            kv_indptr_prefix,
            kv_indptr_prefix,
        )
        partial_m = attn_sink
        partial_l = attn_sink

    if out is None:
        out = torch.empty(
            (tokens, QK_BLOCK_ROWS, V4_DIM_QK),
            dtype=torch.bfloat16,
            device=q_packed.device,
        )
    elif not split_partial and (
        out.dtype != torch.bfloat16
        or out.shape != (tokens, QK_BLOCK_ROWS, V4_DIM_QK)
        or not out.is_contiguous()
    ):
        raise RuntimeError(f"out must be contiguous bf16 shape {(tokens, 128, 512)}")
    devices = {
        tensor.device
        for tensor in (
            q_packed,
            q_rope,
            unified_kv_packed,
            unified_kv_rope,
            kv_indices_prefix,
            kv_indptr_prefix,
            k_packed,
            k_rope,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            out,
            *split_tensors,
            partial_m,
            partial_l,
        )
    }
    if len(devices) != 1:
        raise RuntimeError("all sparse-prefill tensors must be on the same device")

    if stream is None:
        stream = torch.cuda.current_stream(q_packed.device)
    with torch.cuda.device(q_packed.device.index):
        compile_hints = {
            "fast_fp_math": True,
            "unsafe_fp_math": True,
            "llvm_options": {
                "enable-post-misched": bool(post_misched),
                "lsr-drop-solution": True,
                "disable-machine-sink": not bool(machine_sink),
            },
        }
        if waves_per_eu > 0:
            compile_hints["waves_per_eu"] = int(waves_per_eu)
        with CompilationContext.compile_hints(compile_hints):
            _run_compiled(
                _build_sparse_prefill_launcher(
                    bool(pipeline_two),
                    bool(cluster_two),
                    bool(wave_padded_k),
                    bool(alpha_bpermute),
                    int(lds_padding),
                    bool(rescale_once),
                    bool(fixed_softmax_ref),
                    bool(transpose_v),
                    bool(register_p),
                    int(cache_q_step),
                    bool(reuse_q_across_n),
                    bool(broadcast_indices),
                    bool(no_sentinel),
                    bool(full_tile_fastpath),
                    bool(p_lane_layout),
                    bool(stage_q),
                    bool(cache_all_q),
                    bool(assume_full_tiles),
                    bool(permute_k_scales),
                    bool(pairwise_pv),
                    bool(post_misched),
                    bool(machine_sink),
                    bool(setprio),
                    bool(dynamic_full_prefix),
                    bool(prefix_only),
                    bool(split_partial),
                    int(waves_per_eu),
                ),
                ptr_arg(q_packed),
                ptr_arg(q_rope),
                ptr_arg(unified_kv_packed),
                ptr_arg(unified_kv_rope),
                ptr_arg(kv_indices_prefix),
                ptr_arg(kv_indptr_prefix),
                ptr_arg(k_packed),
                ptr_arg(k_rope),
                ptr_arg(kv_indices_extend),
                ptr_arg(kv_indptr_extend),
                ptr_arg(attn_sink),
                ptr_arg(out),
                *(ptr_arg(value) for value in split_tensors),
                ptr_arg(partial_m),
                ptr_arg(partial_l),
                float(softmax_scale),
                int(split_tensors[0].numel() if split_partial else tokens),
                stream,
            )
    return out


__all__ = [
    "stage_v4_paged_tile_flydsl",
    "v4_nope_qk_tile_flydsl",
    "v4_qk_tile_flydsl",
    "v4_qk_h128_tile_flydsl",
    "v4_qk_pv_h128_tile_flydsl",
    "sparse_attn_v4_paged_prefill_fp8_flydsl",
]
