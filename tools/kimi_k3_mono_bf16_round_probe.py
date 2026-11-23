# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""f32 -> bf16 two ways on gfx1250, every f32 bit pattern: the conversion the
compiler emits for ``.to(fx.BFloat16)`` (``v_cvt_pk_bf16_f32``) against an
integer round to nearest even (``bf16_round_bits``; a NaN keeps its high
half, quieted). A thread walks 2^16 patterns, the grid all 2^32; prints the
mismatches among the finite values and among the NaNs, with the first of a
thread. Measured on MI455X: none of either -- the instruction is the IEEE
rounding, NaNs quieted.

TIME=1 also times a chain of CHAIN dependent roundings a thread each way
(ns a step, the launch's overhead in it: CHAIN large). Measured: 8.2 ns a
step with the instruction, 16.0 ns with the integer form -- the instruction
is not a slow one, and K1 keeps it.

    python tools/kimi_k3_mono_bf16_round_probe.py
"""

import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr
from flydsl.expr.typing import Int64

from atom.mono.device.ops import rsrc

THREADS = 256
BLOCKS = 256
STEPS = 1 << 16
CHAIN = int(os.environ.get("CHAIN", "4096"))
WORDS = 8


def bf16_round_bits(bits):
    """An f32's bits rounded to bf16 precision (nearest, ties to even), the
    low half zero, by integer adds: the carry walks the mantissa into the
    exponent as the rounding does, the largest finite's half-way and up to
    infinity; a NaN keeps its high half, quieted."""
    bits = fx.Int32(bits)
    odd = fx.Int32((fx.Uint32(bits) >> fx.Uint32(16)) & fx.Uint32(1))
    rounded = (bits + fx.Int32(0x7FFF) + odd) & fx.Int32(-65536)
    nan = (bits & fx.Int32(0x7FFFFFFF)) > fx.Int32(0x7F800000)
    return nan.select((bits | fx.Int32(0x00400000)) & fx.Int32(-65536), rounded)


def build():
    @flyc.kernel(name="bf16_round_probe", known_block_size=[THREADS, 1, 1])
    def kernel(out: Int64):
        item = fx.Int32(fx.block_idx.x) * THREADS + fx.thread_idx.x
        high = item << 16
        all_bad = fx.Int32(0)
        nan_bad = fx.Int32(0)
        first_bits = fx.Int32(0)
        first_hard = fx.Int32(0)
        first_soft = fx.Int32(0)
        step = fx.Int32(0)
        while step < STEPS:
            bits = high | step
            value = bits.bitcast(fx.Float32)
            hard = fx.Float32(value.to(fx.BFloat16)).bitcast(fx.Int32)
            soft = bf16_round_bits(bits)
            bad = hard != soft
            nan = (bits & 0x7FFFFFFF) > fx.Int32(0x7F800000)
            none_yet = all_bad == 0
            first_bits = (bad & none_yet).select(bits, first_bits)
            first_hard = (bad & none_yet).select(hard, first_hard)
            first_soft = (bad & none_yet).select(soft, first_soft)
            all_bad = all_bad + bad.select(fx.Int32(1), fx.Int32(0))
            nan_bad = nan_bad + (bad & nan).select(fx.Int32(1), fx.Int32(0))
            step = step + fx.Int32(1)
        bo.buffer_store(
            fx.Vector.from_elements(
                [all_bad, nan_bad, first_bits, first_hard], fx.Int32
            ),
            rsrc(out),
            item * WORDS,
        )
        bo.buffer_store(
            fx.Vector.from_elements(
                [first_soft, fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32
            ),
            rsrc(out),
            item * WORDS + 4,
        )

    @flyc.jit
    def launch(out: Int64, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        kernel(out).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


def build_chain(soft: bool):
    """A thread's chain of ``CHAIN`` dependent roundings (each of the last
    one's result plus an increment), the last result stored."""

    @flyc.kernel(
        name=f"bf16_round_chain_{'soft' if soft else 'hard'}",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(out: Int64):
        item = fx.Int32(fx.block_idx.x) * THREADS + fx.thread_idx.x
        # 1.0 .. 1.008, a value a thread
        value = ((item & 0xFFFF) | fx.Int32(0x3F800000)).bitcast(fx.Float32)
        step = fx.Int32(0)
        while step < CHAIN:
            value = value * fx.Float32(1.0009765625) + fx.Float32(0.001953125)
            if const_expr(soft):
                value = bf16_round_bits(value.bitcast(fx.Int32)).bitcast(fx.Float32)
            else:
                value = fx.Float32(value.to(fx.BFloat16))
            step = step + fx.Int32(1)
        bo.buffer_store(value, rsrc(out), item)

    @flyc.jit
    def launch(out: Int64, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        kernel(out).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


def main():
    out = torch.zeros(BLOCKS * THREADS * WORDS, dtype=torch.int32, device="cuda")
    probe = build()
    probe(out.data_ptr(), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    got = out.view(BLOCKS * THREADS, WORDS).cpu().long() & 0xFFFFFFFF
    nan_bad = int(got[:, 1].sum())
    finite_bad = int(got[:, 0].sum()) - nan_bad
    firsts = [
        {"bits": hex(int(r[2])), "hard": hex(int(r[3])), "soft": hex(int(r[4]))}
        for r in got[got[:, 0] > 0][:16]
    ]
    report = {
        "patterns": BLOCKS * THREADS * STEPS,
        "finite_mismatched": finite_bad,
        "nan_mismatched": nan_bad,
        "first_of_a_thread": firsts,
    }
    if os.environ.get("TIME") == "1":
        times, results, held = {}, {}, [probe]
        for soft in (False, True):
            chain = build_chain(soft)
            buf = torch.zeros(BLOCKS * THREADS, dtype=torch.float32, device="cuda")
            held.append((chain, buf))
            chain(buf.data_ptr(), stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            samples = []
            for _ in range(9):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                chain(buf.data_ptr(), stream=torch.cuda.current_stream())
                b.record()
                b.synchronize()
                samples.append(a.elapsed_time(b) * 1e6 / CHAIN)
            key = "soft" if soft else "hard"
            times[key] = round(sorted(samples)[len(samples) // 2], 2)
            results[key] = buf.clone()
        report["chain_ns_a_step"] = times
        report["chain_results_equal"] = bool(
            torch.equal(
                results["hard"].view(torch.int32), results["soft"].view(torch.int32)
            )
        )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
