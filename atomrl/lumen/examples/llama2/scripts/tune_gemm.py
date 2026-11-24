#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tune hipBLASLt GEMM solutions for Llama2-70B TP=1 FP8 training.

Benchmarks all available hipBLASLt solutions for each core GEMM shape
and writes the best solution index per shape to a JSON file that can
be loaded at training time.

Run inside the Lumen Docker container on a single GPU:
    python tune_gemm.py [--output /path/to/tuned_gemms.json] [--warmup 50] [--iters 200]

The output JSON maps "(M,N,K)" string keys to solution indices (int).
"""

import argparse
import json

import torch

FP8_DTYPE = torch.float8_e4m3fnuz
OUT_DTYPE = torch.bfloat16

SHAPES = [
    (8192, 10240, 8192),  # attn QKV fwd
    (8192, 8192, 10240),  # attn QKV dgrad
    (8192, 8192, 8192),  # attn proj fwd/dgrad
    (8192, 57344, 8192),  # MLP FC1 fwd (gate+up)
    (8192, 8192, 57344),  # MLP FC1 dgrad
    (8192, 8192, 28672),  # MLP FC2 fwd
    (8192, 28672, 8192),  # MLP FC2 dgrad
]


def benchmark_solution(mat1, mat2, scale_a, scale_b, sol_idx, warmup, iters):
    """Return median time in ms for a single hipb_mm solution."""
    from aiter.ops.gradlib import hipb_mm

    for _ in range(warmup):
        hipb_mm(
            mat1, mat2, sol_idx, out_dtype=OUT_DTYPE, scaleA=scale_a, scaleB=scale_b
        )
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        hipb_mm(
            mat1, mat2, sol_idx, out_dtype=OUT_DTYPE, scaleA=scale_a, scaleB=scale_b
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    return median


def tune_shape(M, N, K, warmup, iters):
    """Find the best hipBLASLt solution for a single GEMM shape."""
    import aiter
    from aiter.ops.gradlib import hipb_findallsols

    aiter.hipb_create_extension()

    print(f"\n{'='*60}")
    print(f"Tuning M={M}, N={N}, K={K}  (FP8 -> BF16)")
    print(f"{'='*60}")

    mat1 = torch.randn(M, K, device="cuda").to(FP8_DTYPE)
    mat2_weight = torch.randn(N, K, device="cuda").to(FP8_DTYPE)
    mat2 = mat2_weight.t()

    scale_a = torch.tensor(0.5, dtype=torch.float32, device="cuda")
    scale_b = torch.tensor(0.5, dtype=torch.float32, device="cuda")

    sols = hipb_findallsols(
        mat1,
        mat2,
        out_dtype=OUT_DTYPE,
        scaleA=scale_a,
        scaleB=scale_b,
    )
    print(f"  Found {len(sols)} candidate solutions")

    if not sols:
        print("  WARNING: No solutions found!")
        return -1, float("inf")

    default_time = benchmark_solution(mat1, mat2, scale_a, scale_b, -1, warmup, iters)
    print(f"  Default (idx=-1): {default_time:.3f} ms")

    best_idx = -1
    best_time = default_time
    results = []

    for i, sol in enumerate(sols):
        try:
            t = benchmark_solution(
                mat1, mat2, scale_a, scale_b, sol, warmup // 2, iters // 2
            )
            results.append((sol, t))
            if t < best_time:
                best_time = t
                best_idx = sol
            if (i + 1) % 10 == 0 or i == len(sols) - 1:
                print(
                    f"  Tested {i+1}/{len(sols)} solutions, "
                    f"current best: idx={best_idx} ({best_time:.3f} ms)"
                )
        except RuntimeError:
            pass

    if best_idx != -1:
        best_time = benchmark_solution(
            mat1, mat2, scale_a, scale_b, best_idx, warmup, iters
        )

    speedup = (default_time - best_time) / default_time * 100
    print(
        f"\n  Best solution: idx={best_idx}, time={best_time:.3f} ms "
        f"(default={default_time:.3f} ms, speedup={speedup:+.1f}%)"
    )

    del mat1, mat2, mat2_weight, scale_a, scale_b
    torch.cuda.empty_cache()

    return best_idx, best_time


def main():
    parser = argparse.ArgumentParser(description="Tune hipBLASLt GEMMs")
    parser.add_argument(
        "--output", default="/data1/lumen/results/tp1_fp8/tuned_gemms.json"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    print("Llama2-70B TP=1 FP8 GEMM Tuner")
    print(f"  Shapes to tune: {len(SHAPES)}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iters}")
    print(f"  Output: {args.output}")

    results = {}
    total_default = 0.0
    total_tuned = 0.0

    for M, N, K in SHAPES:
        best_idx, best_time = tune_shape(M, N, K, args.warmup, args.iters)
        key = f"{M},{N},{K}"
        results[key] = {
            "solution_index": best_idx,
            "time_ms": best_time,
        }
        default_time = benchmark_solution(
            torch.randn(M, K, device="cuda").to(FP8_DTYPE),
            torch.randn(N, K, device="cuda").to(FP8_DTYPE).t(),
            torch.tensor(0.5, dtype=torch.float32, device="cuda"),
            torch.tensor(0.5, dtype=torch.float32, device="cuda"),
            -1,
            args.warmup,
            args.iters,
        )
        results[key]["default_time_ms"] = default_time
        total_default += default_time
        total_tuned += best_time
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Shape':<25} {'Default':>10} {'Tuned':>10} {'Speedup':>10}")
    print("-" * 55)
    for key, v in results.items():
        speedup = (v["default_time_ms"] - v["time_ms"]) / v["default_time_ms"] * 100
        print(
            f"{key:<25} {v['default_time_ms']:>9.3f}ms {v['time_ms']:>9.3f}ms {speedup:>+9.1f}%"
        )
    print("-" * 55)
    total_speedup = (total_default - total_tuned) / total_default * 100
    print(
        f"{'TOTAL':<25} {total_default:>9.3f}ms {total_tuned:>9.3f}ms {total_speedup:>+9.1f}%"
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
