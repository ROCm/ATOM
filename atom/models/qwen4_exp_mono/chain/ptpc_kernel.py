"""Small-row FP8 PTPC projection on stock AITER (16,16) shuffled weights."""

from functools import cache
from hashlib import sha256
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.mono.device.ops import CM_NT, kernel_symbol, launcher, mfma_fp8, rsrc, traced
from atom.mono.runtime.abi import KernelAbi

ABI = KernelAbi(("x", "weight", "x_scale", "w_scale", "out"))
SOURCE = sha256(Path(__file__).read_bytes()).hexdigest()
STREAM = fx.Stream(None)


@traced
def load8(addr, byte_offset, cm=0):
    return fx.Vector(
        bo.buffer_load(
            rsrc(addr), byte_offset // 4, vec_width=2, dtype=T.i32, cache_modifier=cm
        )
    ).bitcast(fx.Int64)[0]


@traced
def load_scale(addr, i):
    return fx.Float32(bo.buffer_load(rsrc(addr), i, vec_width=1, dtype=T.f32))


@cache
def smem_type(waves):
    @fx.struct
    class Smem:
        partials: fx.Array[fx.Float32, waves * 64 * 4, 16]

    return Smem


@cache
def build(rows, n, k, nout=None, *, waves=4, ntiles=1, unroll=4, nt=False):
    nout = n if nout is None else nout
    assert rows in (4, 8, 16) and n % 16 == 0 and 0 < nout <= n
    assert waves in (4, 8) and ntiles in (1, 2, 4, 8) and waves % ntiles == 0
    kwaves = waves // ntiles
    assert k % (32 * kwaves * unroll) == 0
    rounds = k // (32 * kwaves * unroll)
    smem = smem_type(waves)
    name = kernel_symbol(
        "qwen_trial_ptpc",
        m=rows,
        n=n,
        k=k,
        o=nout,
        w=waves,
        ntile=ntiles,
        u=unroll,
        nt=nt,
    )

    @flyc.kernel(name=name, known_block_size=[waves * 64, 1, 1])
    def kernel(x: Int64, weight: Int64, x_scale: Int64, w_scale: Int64, out: Int64):
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        tile = fx.block_idx.x * ntiles + wave // kwaves
        kpart = wave % kwaves
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        token = fx.min(lane % 16, rows - 1)
        # Shuffled storage: [N/16, K/32, 2, 16, 16] bytes.
        # MFMA A supplies channels, B supplies tokens; each lane loads 8 K values.
        if tile < n // 16:
            for turn in range(rounds):
                for sub in range_constexpr(unroll):
                    kb = (turn * unroll + sub) * kwaves + kpart
                    wa = (
                        tile * (k * 16)
                        + kb * 512
                        + (lane // 32) * 256
                        + (lane % 16) * 16
                        + ((lane // 16) % 2) * 8
                    )
                    xa = token * k + kb * 32 + (lane // 16) * 8
                    w = load8(weight, wa, CM_NT if nt else 0)
                    a = load8(x, xa)
                    acc = mfma_fp8(w, a, acc)
        if kwaves == 1:
            # Output lane mapping: token=lane%16, channel=4*(lane//16)+component.
            if lane % 16 < rows:
                sx = load_scale(x_scale, lane % 16)
                for component in range_constexpr(4):
                    col = tile * 16 + 4 * (lane // 16) + component
                    if col < nout:
                        sw = load_scale(w_scale, col)
                        y = (acc[component] * sx) * sw
                        bo.buffer_store(
                            y.to(fx.BFloat16), rsrc(out), (lane % 16) * nout + col
                        )
        else:
            sm = fx.SharedAllocator().allocate(smem).peek()
            fx.ptr_store(acc, sm.partials.ptr + (wave * 64 + lane) * 4)
            gpu.barrier()
            for store_pass in range_constexpr(
                (ntiles * rows * 16 + waves * 64 - 1) // (waves * 64)
            ):
                index = tid + store_pass * waves * 64
                if index < ntiles * rows * 16:
                    out_tile = index // (rows * 16)
                    t = (index // 16) % rows
                    channel = index % 16
                    col = (fx.block_idx.x * ntiles + out_tile) * 16 + channel
                    if col < nout:
                        value = fx.Float32(0.0)
                        for part in range_constexpr(kwaves):
                            read = (
                                (out_tile * kwaves + part) * 64
                                + 16 * (channel // 4)
                                + t
                            ) * 4 + channel % 4
                            value = value + fx.ptr_load(sm.partials.ptr + read)
                        y = (value * load_scale(x_scale, t)) * load_scale(w_scale, col)
                        bo.buffer_store(y.to(fx.BFloat16), rsrc(out), t * nout + col)

    @launcher(name)
    def launch(
        x: Int64,
        weight: Int64,
        x_scale: Int64,
        w_scale: Int64,
        out: Int64,
        stream: fx.Stream = STREAM,
    ):
        _ = SOURCE
        kernel(x, weight, x_scale, w_scale, out).launch(
            grid=((n + ntiles * 16 - 1) // (ntiles * 16),),
            block=(waves * 64,),
            stream=stream,
        )

    return launch
