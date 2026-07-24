#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone correctness + performance test for rotate_activation Triton kernel.

Usage:
    python tools/test_rotate_activation.py          # correctness only
    python tools/test_rotate_activation.py --bench  # correctness + benchmark
"""
import argparse
import os
import sys
import time

# Ensure atom/ is importable from the worktree root.
_srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _srcdir not in sys.path:
    sys.path.insert(0, _srcdir)

import torch

# Force Triton path (no aiter dependency needed for rotate_activation).
os.environ.setdefault("ATOM_WHT_TORCH_FALLBACK", "0")

from atom.model_ops.quant_v4 import rotate_activation, _rotate_activation_torch


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_orthogonality():
    """H @ H^T = I (within float32 precision)."""
    n = 128
    eye = torch.eye(n, dtype=torch.float32)
    h = rotate_activation(eye)
    hht = h @ h.T
    err = (hht - torch.eye(n)).abs().max().item()
    status = "OK" if err < 1e-5 else "FAIL"
    print(f"[orthogonality]   {status} max_abs_err={err:.2e}")
    return err < 1e-5


def test_involution():
    """H(H(x)) = x (Hadamard is involutive after normalization)."""
    n = 64
    x = torch.randn(2, 4, n, dtype=torch.float32)
    twice = rotate_activation(rotate_activation(x))
    err = (twice - x).abs().max().item()
    status = "OK" if err < 1e-5 else "FAIL"
    print(f"[involution]      {status} max_abs_err={err:.2e}")
    return err < 1e-5


def test_triton_matches_torch():
    """Triton output == torch fallback output (within numerical tolerance)."""
    torch.manual_seed(42)

    shapes_dtypes = [
        ((1, 128), torch.float32),
        ((4, 128), torch.float32),
        ((2, 4, 128), torch.bfloat16),
        ((8, 512), torch.float32),
        ((1, 16, 512), torch.bfloat16),
        ((128, 128), torch.float32),  # stress: many rows
    ]

    all_pass = True
    for shape, dtype in shapes_dtypes:
        x = torch.randn(shape, dtype=dtype)

        # Triton path
        os.environ["ATOM_WHT_TORCH_FALLBACK"] = "0"
        y_triton = rotate_activation(x.clone())

        # Torch path
        os.environ["ATOM_WHT_TORCH_FALLBACK"] = "1"
        y_torch = rotate_activation(x.clone())
        os.environ["ATOM_WHT_TORCH_FALLBACK"] = "0"

        diff = (y_triton.float() - y_torch.float()).abs().max().item()
        ok = diff < 1e-3
        status = "OK" if ok else "FAIL"
        print(
            f"[triton==torch]   {status} shape={list(shape)} dtype={dtype} "
            f"max_diff={diff:.2e}"
        )
        if not ok:
            all_pass = False

    return all_pass


def test_bfloat16_preservation():
    """Output dtype == input dtype (bf16 in → bf16 out)."""
    x = torch.randn(3, 8, 128, dtype=torch.bfloat16)
    y = rotate_activation(x)
    ok = y.dtype == torch.bfloat16
    status = "OK" if ok else "FAIL"
    print(f"[dtype-preserve]  {status} in={x.dtype} out={y.dtype}")
    return ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_forward(n_repeat: int = 10, n_warmup: int = 3):
    """Compare Triton vs torch fallback performance across workloads."""
    print("\n" + "=" * 72)
    print("Benchmark: rotate_activation — Triton vs Torch fallback")
    print("=" * 72)
    print(f"{'Shape':>20s}  {'N':>5s}  {'Triton(ms)':>10s}  {'Torch(ms)':>10s}  {'Speedup':>8s}")

    configs = [
        (128, 1, torch.float32),
        (128, 4, torch.float32),
        (128, 16, torch.float32),
        (128, 64, torch.float32),
        (128, 128, torch.float32),
        (128, 256, torch.float32),
        (128, 512, torch.float32),
        (128, 1024, torch.float32),
        (512, 1, torch.float32),
        (512, 4, torch.float32),
        (512, 16, torch.float32),
        (512, 64, torch.float32),
        (512, 128, torch.float32),
        (128, 64, torch.bfloat16),
        (512, 64, torch.bfloat16),
    ]

    for n, rows, dtype in configs:
        x = torch.randn(rows, n, dtype=dtype)

        with torch.no_grad():
            # Warmup Triton
            os.environ["ATOM_WHT_TORCH_FALLBACK"] = "0"
            for _ in range(n_warmup):
                _ = rotate_activation(x.clone())

            # Measure Triton
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_repeat):
                _ = rotate_activation(x.clone())
            torch.cuda.synchronize()
            t_triton = (time.perf_counter() - t0) / n_repeat * 1000

            # Warmup Torch
            os.environ["ATOM_WHT_TORCH_FALLBACK"] = "1"
            for _ in range(n_warmup):
                _ = rotate_activation(x.clone())

            # Measure Torch
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_repeat):
                _ = rotate_activation(x.clone())
            torch.cuda.synchronize()
            t_torch = (time.perf_counter() - t0) / n_repeat * 1000

            os.environ["ATOM_WHT_TORCH_FALLBACK"] = "0"

        speedup = t_torch / t_triton
        label = f"({rows}, {n})"
        print(f"{label:>20s}  {n:>5d}  {t_triton:>10.4f}  {t_torch:>10.4f}  {speedup:>7.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test rotate_activation Triton kernel")
    parser.add_argument("--bench", action="store_true", help="run benchmark")
    args = parser.parse_args()

    print("=== rotate_activation correctness ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print()

    results = {
        "orthogonality": test_orthogonality(),
        "involution": test_involution(),
        "triton_matches_torch": test_triton_matches_torch(),
        "dtype_preserve": test_bfloat16_preservation(),
    }

    print()
    all_ok = all(results.values())
    print(f"=== {'ALL PASSED' if all_ok else 'SOME FAILED'} ===")

    if args.bench and all_ok:
        bench_forward()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
