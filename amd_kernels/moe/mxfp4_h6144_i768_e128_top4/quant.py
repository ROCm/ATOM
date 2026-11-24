# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP4 activation quant: bf16 [M, K] -> e2m1 codes [M, K/2] (low nibble first) + e8m0 scales [M, K/32] (token
order, what ``scale_sort_tile_map`` sorts), all row-major. One lane per 32-value block: four 16-B loads (a wave reads
4 KB contiguous), the block's |max|, Quark even e8m0 (floored at 1), RNE e2m1 by ``v_cvt_scalef32_pk_fp4_f32``, one
16-B store of codes and one scale byte.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from .common.launch import _run_compiled
from .common.loaders import _as_f32
from .mxfp8.gemm1_prefill_h6144_i3072 import (
    _fmax3_abs,
)

from .i3072.gemm1_prefill_h6144_i3072_persistent import _cvt_pk_fp4, _e8m0_even_e2m1
from amd_kernels.warm_up import launcher

_THREADS = 256


@functools.cache
@launcher
def compile_mxfp4_quant():
    @flyc.kernel(name="mxfp4_quant_even")
    def kernel_mxfp4_quant(
        X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, n_blocks: fx.Int32
    ):
        blk = fx.block_idx.x * _THREADS + fx.thread_idx.x
        x_rsrc = buffer_ops.create_buffer_resource(
            X, max_size=False, num_records_bytes=n_blocks * 64
        )
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=n_blocks * 16
        )
        s_rsrc = buffer_ops.create_buffer_resource(
            S, max_size=False, num_records_bytes=n_blocks
        )
        vals = []
        for j in range_constexpr(4):
            v = Vec(
                buffer_ops.buffer_load(
                    x_rsrc, blk * 16 + j * 4, vec_width=4, dtype=fx.Int32
                )
            )
            for k in range_constexpr(4):
                w = fx.Int32(v[k])
                vals.append(_as_f32(w << 16))
                vals.append(_as_f32(w & fx.Int32(-65536)))
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
    def launch_mxfp4_quant_i768(
        X: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        n_blocks: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_mxfp4_quant(X, Q, S, n_blocks).launch(
            grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    return launch_mxfp4_quant_i768


def mxfp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and k % 32 == 0
    q = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    s = torch.empty((m, k // 32), dtype=torch.uint8, device=x.device)
    n_blocks = m * (k // 32)
    _run_compiled(
        compile_mxfp4_quant(),
        x.view(-1),
        q.view(-1),
        s.view(-1),
        n_blocks,
        (n_blocks + _THREADS - 1) // _THREADS,
        torch.cuda.current_stream(),
    )
    return q, s
