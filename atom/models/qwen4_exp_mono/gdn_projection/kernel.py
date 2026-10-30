"""PTPC trial: reuse each FP8 activation operand across multiple channel tiles."""

from functools import cache
from hashlib import sha256
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.models.qwen4_exp_mono.chain.ptpc_kernel import (
    SOURCE as BASE_SOURCE,
)
from atom.models.qwen4_exp_mono.chain.ptpc_kernel import (
    load8,
    load_scale,
)
from atom.mono.device.ops import (
    CM_NT,
    bf16_round,
    kernel_symbol,
    launcher,
    mfma_fp8,
    rsrc,
)
from atom.mono.runtime.abi import KernelAbi

from .epilogue import complete

ABI = KernelAbi(
    (
        "x",
        "weight",
        "x_scale",
        "w_scale",
        "out",
        "conv_w",
        "history",
        "slots",
        "accepted",
        "cs0",
        "cs1",
        "cs2",
        "alpha",
        "beta_input",
        "abs0",
        "bbs0",
        "a_log",
        "dt_bias",
        "q",
        "k_out",
        "v",
        "g",
        "beta",
        "z",
    )
)
SOURCE = (
    sha256(
        Path(__file__).read_bytes()
        + (Path(__file__).parent / "epilogue.py").read_bytes()
    ).hexdigest()
    + BASE_SOURCE
)
STREAM = fx.Stream(None)


@cache
def smem_type(waves, columns):
    @fx.struct
    class Smem:
        partials: fx.Array[fx.Float32, columns * waves * 64 * 4, 16]
        projected: fx.Array[fx.Float32, 4 * 64, 16]

    return Smem


@cache
def build(debug=False):
    rows, n, k, nout = 4, 16384, 2560, 16384
    waves, ntiles, unroll, nt, columns, burst = 8, 1, 2, True, 4, True
    nout = n if nout is None else nout
    assert rows in (4, 8, 16) and nout == n and n % (16 * columns * ntiles) == 0
    kwaves = waves // ntiles
    assert waves in (4, 8) and waves % ntiles == 0 and k % (32 * kwaves * unroll) == 0
    smem = smem_type(waves, columns)
    rounds = k // (32 * kwaves * unroll)
    name = kernel_symbol(
        "qwen_mtp_projection_conv_" + SOURCE[:12],
        m=rows,
        n=n,
        k=k,
        w=waves,
        ntile=ntiles,
        u=unroll,
        nt=nt,
        cols=columns,
        burst=burst,
        debug=debug,
    )

    @flyc.kernel(name=name, known_block_size=[waves * 64, 1, 1])
    def kernel(
        x: Int64,
        weight: Int64,
        x_scale: Int64,
        w_scale: Int64,
        out: Int64,
        conv_w: Int64,
        history: Int64,
        slots: Int64,
        accepted: Int64,
        cs0: Int64,
        cs1: Int64,
        cs2: Int64,
        alpha: Int64,
        beta_input: Int64,
        abs0: Int64,
        bbs0: Int64,
        a_log: Int64,
        dt_bias: Int64,
        q: Int64,
        k_out: Int64,
        v: Int64,
        g: Int64,
        beta: Int64,
        z: Int64,
    ):
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        bundle = fx.block_idx.x * ntiles + wave // kwaves
        part = wave % kwaves
        token = fx.min(lane % 16, rows - 1)
        acc = fx.Vector.filled(columns * 4, 0.0, fx.Float32)
        for turn in range(rounds):
            xs, ws = [], []
            if burst:
                for sub in range_constexpr(unroll):
                    kb = (turn * unroll + sub) * kwaves + part
                    xs.append(load8(x, token * k + kb * 32 + (lane // 16) * 8))
                    for column in range_constexpr(columns):
                        tile = bundle * columns + column
                        at = (
                            tile * k * 16
                            + kb * 512
                            + (lane // 32) * 256
                            + (lane % 16) * 16
                            + ((lane // 16) % 2) * 8
                        )
                        ws.append(load8(weight, at, CM_NT if nt else 0))
            for sub in range_constexpr(unroll):
                kb = (turn * unroll + sub) * kwaves + part
                a = (
                    xs[sub]
                    if burst
                    else load8(x, token * k + kb * 32 + (lane // 16) * 8)
                )
                elements = []
                for column in range_constexpr(columns):
                    tile = bundle * columns + column
                    at = (
                        tile * k * 16
                        + kb * 512
                        + (lane // 32) * 256
                        + (lane % 16) * 16
                        + ((lane // 16) % 2) * 8
                    )
                    w = (
                        ws[sub * columns + column]
                        if burst
                        else load8(weight, at, CM_NT if nt else 0)
                    )
                    partial = fx.Vector.from_elements(
                        [acc[column * 4 + j] for j in range(4)], fx.Float32
                    )
                    updated = mfma_fp8(w, a, partial)
                    elements += [updated[j] for j in range(4)]
                acc = fx.Vector.from_elements(elements, fx.Float32)
        if kwaves == 1:
            if lane % 16 < rows:
                sx = load_scale(x_scale, lane % 16)
                for column in range_constexpr(columns):
                    for j in range_constexpr(4):
                        col = (bundle * columns + column) * 16 + 4 * (lane // 16) + j
                        y = (acc[column * 4 + j] * sx) * load_scale(w_scale, col)
                        bo.buffer_store(
                            y.to(fx.BFloat16), rsrc(out), (lane % 16) * nout + col
                        )
        else:
            sm = fx.SharedAllocator().allocate(smem).peek()
            for column in range_constexpr(columns):
                value = fx.Vector.from_elements(
                    [acc[column * 4 + j] for j in range(4)], fx.Float32
                )
                fx.ptr_store(
                    value, sm.partials.ptr + ((column * waves + wave) * 64 + lane) * 4
                )
            gpu.barrier()
            for store_pass in range_constexpr(
                (ntiles * columns * rows * 16 + waves * 64 - 1) // (waves * 64)
            ):
                index = tid + store_pass * waves * 64
                if index < ntiles * columns * rows * 16:
                    out_tile = index // (rows * 16)
                    t = (index // 16) % rows
                    channel = index % 16
                    col = (fx.block_idx.x * ntiles * columns + out_tile) * 16 + channel
                    value = fx.Float32(0.0)
                    for kp in range_constexpr(kwaves):
                        wave_read = (
                            (out_tile % columns) * waves
                            + (out_tile // columns) * kwaves
                            + kp
                        )
                        at = (
                            wave_read * 64 + 16 * (channel // 4) + t
                        ) * 4 + channel % 4
                        value = value + fx.ptr_load(sm.partials.ptr + at)
                    y = (value * load_scale(x_scale, t)) * load_scale(w_scale, col)
                    fx.ptr_store(
                        bf16_round(y),
                        sm.projected.ptr + t * 64 + out_tile * 16 + channel,
                    )
                    if debug:
                        bo.buffer_store(y.to(fx.BFloat16), rsrc(out), t * nout + col)
            complete(
                sm,
                conv_w,
                history,
                slots,
                accepted,
                cs0,
                cs1,
                cs2,
                alpha,
                beta_input,
                abs0,
                bbs0,
                a_log,
                dt_bias,
                q,
                k_out,
                v,
                g,
                beta,
                z,
            )

    @launcher(name)
    def launch(
        x: Int64,
        weight: Int64,
        x_scale: Int64,
        w_scale: Int64,
        out: Int64,
        conv_w: Int64,
        history: Int64,
        slots: Int64,
        accepted: Int64,
        cs0: Int64,
        cs1: Int64,
        cs2: Int64,
        alpha: Int64,
        beta_input: Int64,
        abs0: Int64,
        bbs0: Int64,
        a_log: Int64,
        dt_bias: Int64,
        q: Int64,
        k_out: Int64,
        v: Int64,
        g: Int64,
        beta: Int64,
        z: Int64,
        stream: fx.Stream = STREAM,
    ):
        _ = SOURCE
        kernel(
            x,
            weight,
            x_scale,
            w_scale,
            out,
            conv_w,
            history,
            slots,
            accepted,
            cs0,
            cs1,
            cs2,
            alpha,
            beta_input,
            abs0,
            bbs0,
            a_log,
            dt_bias,
            q,
            k_out,
            v,
            g,
            beta,
            z,
        ).launch(
            grid=(n // (ntiles * columns * 16),), block=(waves * 64,), stream=stream
        )

    ABI.check(kernel, launch)
    return launch
