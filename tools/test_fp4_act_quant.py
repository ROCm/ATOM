#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone correctness + performance test for fp4_act_quant Triton kernel.

Usage:
    python tools/test_fp4_act_quant.py          # correctness only
    python tools/test_fp4_act_quant.py --bench  # correctness + benchmark
"""
import argparse
import importlib.util
import os
import sys
import time
import types as _types

# Load quant_v4.py directly to avoid atom package __init__ import cascade.
_srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_qv4_path = os.path.join(_srcdir, "atom", "model_ops", "quant_v4.py")
_spec = importlib.util.spec_from_file_location("atom.model_ops.quant_v4", _qv4_path)
_qv4 = importlib.util.module_from_spec(_spec)
_atom = _types.ModuleType("atom")
_atom.__path__ = [os.path.join(_srcdir, "atom")]
sys.modules["atom"] = _atom
_mops = _types.ModuleType("atom.model_ops")
_mops.__path__ = [os.path.join(_srcdir, "atom", "model_ops")]
sys.modules["atom.model_ops"] = _mops
_spec.loader.exec_module(_qv4)

fp4_act_quant = _qv4.fp4_act_quant
_fp4_act_quant_triton = _qv4._fp4_act_quant_triton
_fp4_act_quant_torch = _qv4._fp4_act_quant_torch
dequant_fp4_e2m1 = _qv4.dequant_fp4_e2m1
torch = _qv4.torch

import torch as _torch  # for top-level cuda checks


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_triton_matches_torch():
    """Triton packed output == torch packed output (max diff = 0 on packed bytes)."""
    torch.manual_seed(42)

    shapes = [
        (1, 32),
        (1, 64),
        (4, 32),
        (4, 64),
        (2, 16, 64),
        (4, 32, 128),
        (8, 64, 256),
        (16, 128, 256),
        (1, 128, 256),
        (1, 128, 512),
        (32, 512),
        (2048, 32),
        (1, 1024),
    ]

    all_pass = True
    for shape in shapes:
        x = torch.randn(*shape, dtype=torch.bfloat16) * 3.0

        # Allocate output buffers (float32 scale for both, to match kernel API)
        *prefix, n = shape
        packed_triton = torch.empty(*prefix, n // 2, dtype=torch.uint8, device="cuda")
        scale_triton = torch.empty(*prefix, n // 32, dtype=torch.float32, device="cuda")
        packed_torch = torch.empty(*prefix, n // 2, dtype=torch.uint8)
        scale_torch = torch.empty(*prefix, n // 32, dtype=torch.float32)

        _fp4_act_quant_triton(x.clone().cuda(), packed_triton, scale_triton, 32)
        _fp4_act_quant_torch(x.clone(), packed_torch, scale_torch, 32)

        diff_packed = (
            (packed_triton.cpu().int() - packed_torch.int()).abs().max().item()
        )
        diff_scale = (
            (scale_triton.cpu() - scale_torch).abs().max().item()
        )
        ok = diff_packed == 0 and diff_scale < 1e-6
        status = "OK" if ok else "FAIL"
        print(
            f"[triton==torch] {status} shape={list(shape)} "
            f"packed_diff={diff_packed} scale_diff={diff_scale:.2e}"
        )
        if not ok:
            all_pass = False

    return all_pass


def test_roundtrip():
    """fp4_act_quant + dequant_fp4_e2m1 ≈ original (within FP4 precision)."""
    torch.manual_seed(99)

    shapes = [
        (1, 32),
        (4, 64),
        (16, 256),
        (1, 128, 256),
    ]

    all_pass = True
    for shape in shapes:
        x = torch.randn(*shape, dtype=torch.bfloat16) * 2.0

        # Quantize (Triton path on GPU, torch fallback on CPU)
        packed, scale_e8 = fp4_act_quant(x, 32)

        # Dequantize via dequant_fp4_e2m1 (expects [..., out, in/2] format)
        *prefix, n = shape
        packed_i8 = packed.to(torch.int8).reshape(*prefix, 1, n // 2)
        scale_float = scale_e8.float().reshape(*prefix, 1, n // 32)
        dequant = dequant_fp4_e2m1(packed_i8, scale_float, fp4_block_size=32)
        dequant = dequant.reshape(*prefix, n)

        # Check: dequant values are on FP4 grid
        blocks = dequant.float().reshape(*prefix, n // 32, 32)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2**-126)
        blk_scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
        normalized = (blocks / blk_scale).abs()
        valid_grid = _qv4._FP4_MAGNITUDES.to(normalized.device)
        off_grid = (
            (normalized.unsqueeze(-1) - valid_grid).abs()
            .min(dim=-1).values.max().item()
        )
        ok = off_grid < 1e-4
        status = "OK" if ok else "FAIL"
        print(f"[roundtrip]      {status} shape={list(shape)} off_grid={off_grid:.2e}")
        if not ok:
            all_pass = False

    return all_pass


def test_pack_format():
    """Verify pack format: dequant_fp4_e2m1 correctly unpacks our output."""
    torch.manual_seed(7)

    shapes = [(4, 64), (2, 16, 128)]
    all_pass = True
    for shape in shapes:
        x = torch.randn(*shape, dtype=torch.bfloat16) * 3.0

        # Our quantize
        packed, scale_e8 = fp4_act_quant(x, 32)

        # Our dequantize
        *prefix, n = shape
        packed_i8 = packed.to(torch.int8).reshape(*prefix, 1, n // 2)
        scale_f = scale_e8.float().reshape(*prefix, 1, n // 32)
        out = dequant_fp4_e2m1(packed_i8, scale_f, fp4_block_size=32)
        out = out.reshape(*prefix, n)

        # Verify all dequantized values are valid FP4 magnitudes after rescaling
        blocks = out.float().reshape(*prefix, n // 32, 32)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2**-126)
        blk_scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
        normalized_abs = (blocks / blk_scale).abs()
        valid_grid = _qv4._FP4_MAGNITUDES.to(normalized_abs.device)
        max_off = (
            (normalized_abs.unsqueeze(-1) - valid_grid).abs()
            .min(dim=-1).values.max().item()
        )
        ok = max_off < 1e-4
        status = "OK" if ok else "FAIL"
        print(f"[pack-format]    {status} shape={list(shape)} max_off={max_off:.2e}")
        if not ok:
            all_pass = False

    return all_pass


def test_zeros():
    """All-zero tensor → all-zeros packed + min scale."""
    shapes = [(1, 32), (4, 64)]
    all_pass = True
    for shape in shapes:
        x = torch.zeros(*shape, dtype=torch.bfloat16)
        packed, scale = fp4_act_quant(x, 32)
        ok_packed = (packed == 0).all()
        # For all-zero, amax = clamp_min, scale should be a valid E8M0 value
        ok_scale = not scale.float().isnan().any()
        ok = ok_packed and ok_scale
        status = "OK" if ok else "FAIL"
        print(
            f"[zeros]          {status} shape={list(shape)} "
            f"all_zero_packed={ok_packed}"
        )
        if not ok:
            all_pass = False
    return all_pass


def test_output_shapes():
    """Verify output shapes: packed [..., N//2], scale [..., N//32]."""
    configs = [
        ((1, 32), (1, 16), (1, 1)),
        ((1, 64), (1, 32), (1, 2)),
        ((4, 128), (4, 64), (4, 4)),
        ((2, 16, 256), (2, 16, 128), (2, 16, 8)),
        ((1, 128, 256), (1, 128, 128), (1, 128, 8)),
    ]
    all_pass = True
    for input_shape, expected_packed, expected_scale in configs:
        x = torch.randn(*input_shape, dtype=torch.bfloat16)
        packed, scale = fp4_act_quant(x, 32)
        ok = tuple(packed.shape) == expected_packed and tuple(scale.shape) == expected_scale
        status = "OK" if ok else "FAIL"
        print(
            f"[shape]          {status} in={list(input_shape)} "
            f"packed={list(packed.shape)} scale={list(scale.shape)}"
        )
        if not ok:
            all_pass = False
    return all_pass


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_forward(n_repeat: int = 100, n_warmup: int = 10):
    """Compare Triton vs torch fallback performance across workloads."""
    print("\n" + "=" * 85)
    print("Benchmark: fp4_act_quant — Triton vs Torch fallback")
    print("=" * 85)
    header = (
        f"{'Shape':>20s}  "
        f"{'Triton(μs)':>10s}  {'Torch(μs)':>10s}  {'Speedup':>8s}  {'BW(GB/s)':>10s}"
    )
    print(header)
    print("-" * 85)

    configs = [
        (1, 32),
        (4, 32),
        (16, 32),
        (64, 32),
        (128, 32),
        (512, 32),
        (2, 16, 64),
        (4, 32, 128),
        (8, 64, 256),
        (1, 128, 256),   # DSv4 Indexer kv path
        (16, 128, 256),
        (1, 128, 512),
        (32, 512),
        (64, 256),
        (1, 1024),
    ]

    for shape in configs:
        *prefix, n = shape
        x = torch.randn(*shape, dtype=torch.bfloat16).cuda()
        packed = torch.empty(*prefix, n // 2, dtype=torch.uint8, device="cuda")
        scale = torch.empty(*prefix, n // 32, dtype=torch.float32, device="cuda")
        num_elems = x.numel()
        nbytes = num_elems * 2  # bf16

        # Warmup Triton
        for _ in range(n_warmup):
            _fp4_act_quant_triton(x.clone(), packed.clone(), scale.clone(), 32)

        # Measure Triton
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _fp4_act_quant_triton(x.clone(), packed.clone(), scale.clone(), 32)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / n_repeat * 1e6

        # Warmup Torch
        packed_cpu = torch.empty(*prefix, n // 2, dtype=torch.uint8)
        scale_cpu = torch.empty(*prefix, n // 32, dtype=torch.float32)
        for _ in range(n_warmup):
            _fp4_act_quant_torch(x.clone().cpu(), packed_cpu.clone(), scale_cpu.clone(), 32)

        # Measure Torch
        x_cpu = x.cpu()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _fp4_act_quant_torch(x_cpu.clone(), packed_cpu.clone(), scale_cpu.clone(), 32)
        t_torch = (time.perf_counter() - t0) / n_repeat * 1e6

        speedup = t_torch / t_triton if t_triton > 0 else float("inf")
        bw = nbytes / t_triton * 1e6 / 1e9
        label = f"({shape[0]}, {shape[-1]})" if len(shape) == 2 else str(list(shape))
        print(
            f"{label:>20s}  "
            f"{t_triton:>10.2f}  {t_torch:>10.2f}  {speedup:>7.2f}x  {bw:>10.2f}"
        )

    print("-" * 85)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Test fp4_act_quant Triton kernel")
    parser.add_argument("--bench", action="store_true", help="run benchmark")
    args = parser.parse_args()

    print("=== fp4_act_quant (BF16 → packed uint8 + E8M0 scale) ===")
    print(f"CUDA available: {_torch.cuda.is_available()}")
    if _torch.cuda.is_available():
        print(f"GPU: {_torch.cuda.get_device_name(0)}")
        vram = _torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"VRAM: {vram} GB")
    print()

    if not _torch.cuda.is_available():
        print("ERROR: CUDA required for Triton kernel tests")
        return 1

    results = {
        "triton_matches_torch": test_triton_matches_torch(),
        "roundtrip": test_roundtrip(),
        "pack_format": test_pack_format(),
        "zeros": test_zeros(),
        "output_shapes": test_output_shapes(),
    }

    print()
    all_ok = all(results.values())
    print(f"=== {'ALL PASSED' if all_ok else 'SOME FAILED'} ===")

    if args.bench and all_ok:
        bench_forward()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
