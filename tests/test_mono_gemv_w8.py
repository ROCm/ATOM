# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The shared GEMV pieces (``atom.mono.device.gemv``) on a GPU: the bf16 and
FP8 (``w8``) paths against the dequantized product, and every finite E4M3 byte
read back exactly.
"""

import pytest
import torch

FP8 = torch.float8_e4m3fn


def _gemv_kernel(rows, k, s, w8):
    """One CTA a 16-row task of an [rows, k] GEMV against s token rows, from
    the ``gemv`` pieces K1 / K2 use: ``loads``, ``rows_to_lds``, ``mfmas``,
    then the reduce (``pair_scales`` / ``row_pair``) -> fp32 ``out[s, rows]``,
    each value bf16-rounded."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import Int64

    from atom.mono.device import gemv
    from atom.mono.device.ops import rsrc
    from atom.mono.plan.execution import THREADS, WAVES

    pairs, xrow = gemv.ROWS // 2, k + gemv.LDS_PAD
    default_stream = fx.Stream(None)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, s * xrow // 2, 16]

    name = f"gemv_test_k{k}_s{s}_{'w8' if w8 else 'bf16'}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def gemv_task(w: Int64, scale: Int64, x: Int64, out: Int64):
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        row0 = fx.block_idx.x * gemv.ROWS
        lds = fx.SharedAllocator().allocate(Smem).peek()
        red, xl = lds.red.ptr, lds.xl.ptr
        ops = gemv.loads(lane, wave, w, row0, k, w8=w8)
        scales = None
        if const_expr(w8):
            scales = gemv.pair_scales(scale, row0, tid)
        gemv.rows_to_lds(tid, x, k, s, xl, xrow)
        acc = gemv.mfmas(lane, wave, xl, xrow, k, s, ops, w8=w8)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < pairs * s:
            rp = tid % pairs
            t = tid // pairs
            vs = gemv.row_pair(red, rp, t, scales)
            for i in range_constexpr(2):
                bo.buffer_store(vs[i], rsrc(out), t * rows + row0 + 2 * rp + i)

    @flyc.jit
    def launch(
        w: Int64, scale: Int64, x: Int64, out: Int64, stream: fx.Stream = default_stream
    ):
        gemv_task(w, scale, x, out).launch(
            grid=(rows // gemv.ROWS,), block=(THREADS,), stream=stream
        )

    return launch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("k", [768, 1536])
@pytest.mark.parametrize("s", [1, 5, 16])
@pytest.mark.parametrize("w8", [False, True])
def test_gemv_matches_the_dequantized_product(k, s, w8):
    # small integers: every product and partial sum is exact in fp32, so the
    # kernel's K order cannot change the result and any misread element or
    # wrong row scale shows as a mismatch
    pytest.importorskip("flydsl")
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    rows, dev = 48, torch.device("cuda")
    gen = torch.Generator().manual_seed(k + s)
    plain = torch.randint(-16, 17, (rows, k), generator=gen).float()
    x = torch.randint(-8, 9, (s, k), generator=gen).to(torch.bfloat16)
    if w8:
        w = shuffle.shuffle_weight(plain.to(FP8).to(dev), layout=(16, 16))
        scale = torch.rand(rows, generator=gen) + 0.5
    else:
        w = plain.to(torch.bfloat16).to(dev)
        scale = torch.ones(rows)
    scale = scale.to(dev)
    x = x.to(dev)
    out = torch.zeros(s, rows, device=dev)
    _gemv_kernel(rows, k, s, w8)(
        w.data_ptr(), scale.data_ptr(), x.data_ptr(), out.data_ptr()
    )
    torch.cuda.synchronize()
    sums = x.float().cpu() @ plain.t()
    want = (sums * scale.cpu() if w8 else sums).to(torch.bfloat16).float()
    assert torch.equal(out.cpu(), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_w8_widens_every_finite_code_exactly():
    # all 254 finite E4M3 bytes (subnormals to 448), each in every row and at
    # every position of a lane's 8-byte run, read back through one-hot token
    # rows: each output is one widened weight, so it must be the byte's value
    pytest.importorskip("flydsl")
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    rows, k, s, dev = 48, 768, 16, torch.device("cuda")
    finite = torch.tensor(
        [b for b in range(256) if b & 0x7F != 0x7F], dtype=torch.uint8
    )
    at = (7 * torch.arange(rows)[:, None] + torch.arange(k)[None, :]) % finite.numel()
    plain = finite[at].view(FP8)
    w = shuffle.shuffle_weight(plain.to(dev), layout=(16, 16))
    scale = torch.ones(rows, device=dev)
    out = torch.zeros(s, rows, device=dev)
    launch = _gemv_kernel(rows, k, s, True)
    for c0 in range(0, k, s):
        x = torch.zeros(s, k, dtype=torch.bfloat16)
        x[torch.arange(s), c0 + torch.arange(s)] = 1
        x = x.to(dev)
        launch(w.data_ptr(), scale.data_ptr(), x.data_ptr(), out.data_ptr())
        torch.cuda.synchronize()
        want = plain.float()[:, c0 : c0 + s].t()
        assert torch.equal(out.cpu(), want), f"columns {c0} .. {c0 + s - 1}"
