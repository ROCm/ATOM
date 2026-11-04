# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``atom.mono.device.dcp_merge`` on the GPU: the LSE form, run in a small
FlyDSL kernel, against ``dcp_ops``'s rank-order Triton combine bit for bit
(the Kimi-K3 K2 merge's contract), and the max/sum form against a torch
reference. Rows include empty sources (LSE -inf, partial NaN), a NaN and a +inf
LSE, and a row whose every source is empty."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "drives FlyDSL and Triton kernels; needs a GPU", allow_module_level=True
    )
pytest.importorskip("flydsl")
pytest.importorskip("aiter", reason="the kernels use AITER's buffer ops")
triton = pytest.importorskip("triton")

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64, T

from atom.mono.device import dcp_merge as merge
from atom.mono.device.ops import rsrc, xred

N, D, ROWS = 8, 16, 64  # sources, row width, rows (a thread a row)


def _lse_kernel():
    @flyc.kernel(known_block_size=[ROWS, 1, 1])
    def k(lse_in: Int64, val_in: Int64, out: Int64, out_lse: Int64):
        row = fx.thread_idx.x
        lses = [
            merge.clean_lse(
                fx.Float32(
                    bo.buffer_load(rsrc(lse_in), row * N + n, vec_width=1, dtype=T.f32)
                )
            )
            for n in range(N)
        ]
        total = merge.lse_total(lses)
        acc = [fx.Float32(0.0)] * D
        for n in range_constexpr(N):
            vals = [
                fx.Float32(
                    bo.buffer_load(
                        rsrc(val_in), (row * N + n) * D + j, vec_width=1, dtype=T.f32
                    )
                )
                for j in range(D)
            ]
            acc = merge.lse_accumulate(acc, vals, merge.lse_factor(lses[n], total))
        for j in range_constexpr(D):
            bo.buffer_store(acc[j], rsrc(out), row * D + j)
        bo.buffer_store(total, rsrc(out_lse), row)

    @flyc.jit
    def launch(
        lse_in: Int64,
        val_in: Int64,
        out: Int64,
        out_lse: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        k(lse_in, val_in, out, out_lse).launch(grid=(1,), block=(ROWS,), stream=stream)

    return launch


def _maxsum_kernel():
    def wave_sum(v):
        for sh in range_constexpr(6):
            v = xred(v, 32 >> sh, lambda a, b: a + b)
        return v

    def wave_max(v):
        for sh in range_constexpr(6):
            v = xred(v, 32 >> sh, fx.max)
        return v

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(m_in: Int64, l_in: Int64, w_out: Int64):
        lane = fx.thread_idx.x
        row = fx.block_idx.x
        valid = lane < N
        src = fx.min(lane, N - 1)
        m = fx.Float32(
            bo.buffer_load(rsrc(m_in), row * N + src, vec_width=1, dtype=T.f32)
        )
        s = fx.Float32(
            bo.buffer_load(rsrc(l_in), row * N + src, vec_width=1, dtype=T.f32)
        )
        m = valid.select(m, fx.Float32(-1.0e30))
        s = valid.select(s, fx.Float32(0.0))
        w = merge.maxsum_weight(m, s, wave_max, wave_sum)
        if valid:
            bo.buffer_store(w, rsrc(w_out), row * N + lane)

    @flyc.jit
    def launch(
        m_in: Int64,
        l_in: Int64,
        w_out: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        k(m_in, l_in, w_out).launch(grid=(ROWS,), block=(64,), stream=stream)

    return launch


def _accumulate_kernel():
    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(w_in: Int64, v_in: Int64, o_out: Int64):
        row = fx.thread_idx.x
        acc = [fx.Float32(0.0), fx.Float32(0.0)]
        for src in range_constexpr(N):
            w = fx.Float32(
                bo.buffer_load(rsrc(w_in), row * N + src, vec_width=1, dtype=T.f32)
            )
            v = [
                fx.Float32(
                    bo.buffer_load(
                        rsrc(v_in), (row * N + src) * 2 + j, vec_width=1, dtype=T.f32
                    )
                )
                for j in range(2)
            ]
            acc = merge.maxsum_accumulate(acc, v, w)
        for j in range_constexpr(2):
            bo.buffer_store(acc[j], rsrc(o_out), row * 2 + j)

    @flyc.jit
    def launch(
        w_in: Int64,
        v_in: Int64,
        o_out: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        k(w_in, v_in, o_out).launch(grid=(1,), block=(64,), stream=stream)

    return launch


def _lse_inputs():
    g = torch.Generator().manual_seed(7)
    lse = torch.randn(ROWS, N, generator=g) * 3.0
    val = torch.randn(ROWS, N, D, generator=g)
    lse[1, 0] = val[1, 0] = float("nan")  # NaN LSE: contributes nothing
    lse[2, 3] = float("inf")
    lse[3, :5] = float("-inf")  # empty sources: LSE -inf, partial NaN
    val[3, :5] = float("nan")
    lse[4] = float("-inf")  # every source empty
    val[4] = float("nan")
    lse[5, 7] = -200.0  # a factor below fp32's normal range
    return lse.cuda().contiguous(), val.cuda().contiguous()


def test_lse_merge_is_the_rank_order_triton_combine_bit_for_bit():
    from atom.model_ops.dcp_ops import _dcp_a2a_unpack_combine_rank_order_kernel

    lse, val = _lse_inputs()
    out = torch.empty(ROWS, D, device="cuda")
    out_lse = torch.empty(ROWS, device="cuda")
    _lse_kernel()(lse.data_ptr(), val.data_ptr(), out.data_ptr(), out_lse.data_ptr())

    # the Triton combine's receive buffer: [source][row][head 1][D + LSE], fp32
    recv = torch.empty(N, ROWS, 1, D + 1, device="cuda")
    recv[:, :, 0, :D] = val.permute(1, 0, 2)
    recv[:, :, 0, D] = lse.t()
    ref = torch.empty(ROWS, 1, D, device="cuda")
    ref_lse = torch.empty(ROWS, 1, device="cuda")
    _dcp_a2a_unpack_combine_rank_order_kernel[(ROWS, 1)](
        recv,
        ref,
        ref_lse,
        recv.stride(0),
        recv.stride(1),
        recv.stride(2),
        ref.stride(0),
        ref.stride(1),
        ref_lse.stride(0),
        ref_lse.stride(1),
        N,
        HEAD_DIM=D,
        LSE_PACK=1,
        N_ROUNDED=N,
        WRITE_LSE=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int32), ref[:, 0].view(torch.int32))
    assert torch.equal(out_lse.view(torch.int32), ref_lse[:, 0].view(torch.int32))
    assert torch.isfinite(out).all()
    assert torch.equal(out[4], torch.zeros(D, device="cuda"))  # every source empty


def test_maxsum_weight_is_the_softmax_share():
    g = torch.Generator().manual_seed(11)
    m = (torch.randn(ROWS, N, generator=g) * 4.0).cuda().contiguous()
    s = (torch.rand(ROWS, N, generator=g) * 50.0 + 1.0).cuda().contiguous()
    s[0] = 0.0  # every source empty: weights 0, not NaN
    w = torch.full((ROWS, N), float("nan"), device="cuda")
    _maxsum_kernel()(m.data_ptr(), s.data_ptr(), w.data_ptr())
    torch.cuda.synchronize()
    z = s * torch.exp(m - m.amax(dim=1, keepdim=True))
    ref = z / z.sum(dim=1, keepdim=True)
    ref[0] = 0.0
    torch.testing.assert_close(w, ref, rtol=2e-6, atol=1e-7)


def test_maxsum_accumulate_drops_a_zero_weight_source_even_if_nan():
    """An empty rank's partial is whatever its padding slot holds; at weight 0
    it must add an exact 0, not NaN * 0."""
    g = torch.Generator().manual_seed(13)
    w = torch.rand(ROWS, N, generator=g)
    v = torch.randn(ROWS, N, 2, generator=g)
    w[:, 1::3] = 0.0
    v[:, 1::3] = float("nan")
    w[0] = 0.0  # every source empty
    v[0] = float("nan")
    w, v = w.cuda().contiguous(), v.cuda().contiguous()
    out = torch.full((ROWS, 2), float("nan"), device="cuda")
    _accumulate_kernel()(w.data_ptr(), v.data_ptr(), out.data_ptr())
    torch.cuda.synchronize()
    live = w != 0
    ref = (torch.where(live[..., None], v, 0.0) * w[..., None]).sum(dim=1)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], torch.zeros(2, device="cuda"))
    torch.testing.assert_close(out, ref, rtol=2e-6, atol=1e-6)
