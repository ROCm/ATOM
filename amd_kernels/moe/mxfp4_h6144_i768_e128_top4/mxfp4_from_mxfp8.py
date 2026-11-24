# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 -> MXFP4 requant of the all-gathered MoE input under sequence parallelism: e4m3 codes [M, K] + e8m0 [M, K/32]
-> ``quant.mxfp4_quant``'s layouts, equal to ``mxfp4_quant`` of the dequantized input in bf16 (an e4m3 value times a
power of two is exact in bf16). One lane per 32-value block: two 16-B code loads and the scale byte, an exact decode to
f32 (``v_cvt_scalef32_pk_f32_fp8``), then ``quant``'s |max|, Quark even e8m0 and RNE e2m1 (gfx950 has no fp8 -> fp4
conversion).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common.launch import _run_compiled
from .common.loaders import _as_f32
from .mxfp8.gemm1_prefill_h6144_i3072 import (
    _fmax3_abs,
)

from .i3072.gemm1_prefill_h6144_i3072_persistent import _cvt_pk_fp4, _e8m0_even_e2m1
from amd_kernels.warm_up import launcher

_THREADS = 256


def _e4m3x2(word, scale_f, hi):
    """Bytes 0, 1 (``hi``: 2, 3) of ``word`` as f32 times ``scale_f``."""
    pk = _rocdl_ops.cvt_scalef32_pk_f32_fp8(
        T.vec(2, T.f32), fx.as_ir_value(word), fx.as_ir_value(scale_f), hi
    )
    v = Vec(pk)
    return fx.Float32(v[0]), fx.Float32(v[1])


@functools.cache
@launcher
def compile_mxfp4_from_mxfp8():
    @flyc.kernel(name="mxfp4_from_mxfp8_even")
    def kernel_mxfp4_from_mxfp8(
        C: fx.Tensor, CS: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, n_blocks: fx.Int32
    ):
        blk = fx.block_idx.x * _THREADS + fx.thread_idx.x
        c_rsrc = buffer_ops.create_buffer_resource(
            C, max_size=False, num_records_bytes=n_blocks * 32
        )
        cs_rsrc = buffer_ops.create_buffer_resource(
            CS, max_size=False, num_records_bytes=n_blocks
        )
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=n_blocks * 16
        )
        s_rsrc = buffer_ops.create_buffer_resource(
            S, max_size=False, num_records_bytes=n_blocks
        )
        # the block's scale byte, from the dword four neighbouring lanes share
        sw = fx.Int32(
            buffer_ops.buffer_load(cs_rsrc, blk >> 2, vec_width=1, dtype=fx.Int32)
        )
        s8 = (sw >> ((blk & fx.Int32(3)) * fx.Int32(8))) & fx.Int32(0xFF)
        scale_in = _as_f32(s8 << 23)
        vals = []
        for j in range_constexpr(2):
            v = Vec(
                buffer_ops.buffer_load(
                    c_rsrc, blk * 8 + j * 4, vec_width=4, dtype=fx.Int32
                )
            )
            for k in range_constexpr(4):
                w = fx.Int32(v[k])
                for hi in (False, True):
                    vals.extend(_e4m3x2(w, scale_in, hi))
        amax = _fmax3_abs(vals[0], vals[1], vals[2])
        for i in range_constexpr(3, 31, 2):
            amax = _fmax3_abs(vals[i], vals[i + 1], amax)
        amax = _fmax3_abs(vals[31], vals[31], amax)
        e8m0 = _e8m0_even_e2m1(amax)
        scale_f = _as_f32(e8m0 << 23)
        out = []
        for d in range_constexpr(4):
            p = fx.Int32(0)
            for k in range_constexpr(4):
                p = _cvt_pk_fp4(
                    p, vals[8 * d + 2 * k], vals[8 * d + 2 * k + 1], scale_f, k
                )
            out.append(p)
        buffer_ops.buffer_store(Vec.from_elements(out, fx.Int32), q_rsrc, blk * 4)
        buffer_ops.buffer_store(fx.Int32(e8m0).to(fx.Int8), s_rsrc, blk)

    @flyc.jit
    def launch_mxfp4_from_mxfp8(
        C: fx.Tensor,
        CS: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        n_blocks: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_mxfp4_from_mxfp8(C, CS, Q, S, n_blocks).launch(
            grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    return launch_mxfp4_from_mxfp8


def mxfp4_from_mxfp8(
    codes: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 ``(codes [M, K] e4m3, scales [M, K/32] e8m0)`` -> MXFP4 ``(codes [M, K/2], scales [M, K/32])``, all
    row-major u8 views."""
    m, k = codes.shape
    assert codes.is_contiguous() and scales.is_contiguous() and k % 128 == 0
    assert scales.shape == (m, k // 32), scales.shape
    q = torch.empty((m, k // 2), dtype=torch.uint8, device=codes.device)
    s = torch.empty((m, k // 32), dtype=torch.uint8, device=codes.device)
    n_blocks = m * (k // 32)
    _run_compiled(
        compile_mxfp4_from_mxfp8(),
        codes.view(torch.uint8).view(-1),
        scales.view(torch.uint8).view(-1),
        q.view(-1),
        s.view(-1),
        n_blocks,
        (n_blocks + _THREADS - 1) // _THREADS,
        torch.cuda.current_stream(),
    )
    return q, s
