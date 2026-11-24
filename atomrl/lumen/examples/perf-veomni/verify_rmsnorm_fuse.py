#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Single-GPU check of Lumen's Diffusers RMSNorm fusion.

  1. does it fuse -- kernel count per call, forward and backward
  2. is it faster at the two shapes the Qwen-Image DiT uses
  3. does it change the answer -- SNR against an FP32 reference, compared with
     what diffusers' own BF16 chain scores
  4. what happens with a DTensor weight. FSDP2 unshards parameters to plain
     tensors before a module's forward, so training never hands one over; if
     something did, the compiled function would reject it exactly as eager
     torch does (a plain activation times a DTensor weight is an error either
     way). The patch checks for one and takes diffusers' forward instead, which
     this confirms on a real diffusers RMSNorm.

Expected on MI350X: image-branch forward 8 kernels -> 1, ~91 -> ~22 us; SNR
55.6 dB fused against 52.6 dB stock.

    HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python verify_rmsnorm_fuse.py
"""

import os
import sys

import torch
import torch.profiler as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomrl.lumen.ops.normalization.rmsnorm import (
    _diffusers_rmsnorm_weighted,
    fused_diffusers_rmsnorm,
)

SHAPES = [("image", (1, 4096, 24, 128)), ("text", (1, 13, 24, 128))]
ITERS, WARMUP = 30, 10


def stock(hidden_states, weight, eps):
    """diffusers 0.37.0 RMSNorm.forward, weight path, eager."""
    return _diffusers_rmsnorm_weighted(hidden_states, weight, eps)


def snr_db(ref, got):
    ref = ref.float()
    err = got.float() - ref
    return 10 * torch.log10(ref.pow(2).mean() / err.pow(2).mean()).item()


def count_kernels(fn, args, backward):
    out = fn(*args)
    if backward:
        out.sum().backward(retain_graph=True)
    torch.cuda.synchronize()
    with P.profile(activities=[P.ProfilerActivity.CUDA]) as pr:
        out = fn(*args)
        if backward:
            out.sum().backward(retain_graph=True)
        torch.cuda.synchronize()
    return sum(int(e.self_device_time_total > 0) for e in pr.key_averages())


def timed(fn, args, backward):
    for _ in range(WARMUP):
        out = fn(*args)
        if backward:
            out.sum().backward(retain_graph=True)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS):
        out = fn(*args)
        if backward:
            out.sum().backward(retain_graph=True)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS * 1e3  # us


def main():
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")
    torch.manual_seed(0)
    eps = 1e-6

    print("=== 1+2. fusion and speed ===")
    print(
        f"  {'shape':7} {'pass':10} {'stock us':>9} {'fused us':>9} {'speedup':>8} {'k.stock':>8} {'k.fused':>8}"
    )
    for tag, shape in SHAPES:
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
        for label, bwd in (("forward", False), ("fwd+bwd", True)):
            if bwd:
                a = (
                    x.clone().requires_grad_(True),
                    w.clone().requires_grad_(True),
                    eps,
                )
            else:
                a = (x, w, eps)
            s_us = timed(stock, a, bwd)
            f_us = timed(fused_diffusers_rmsnorm, a, bwd)
            s_k = count_kernels(stock, a, bwd)
            f_k = count_kernels(fused_diffusers_rmsnorm, a, bwd)
            print(
                f"  {tag:7} {label:10} {s_us:9.1f} {f_us:9.1f} {s_us / f_us:7.2f}x {s_k:8d} {f_k:8d}"
            )
        torch.cuda.empty_cache()

    print("\n=== 3. accuracy: both BF16 paths against an FP32 reference ===")
    for tag, shape in SHAPES:
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
        ref = stock(x.float(), w.float(), eps)
        s = stock(x, w, eps)
        f = fused_diffusers_rmsnorm(x, w, eps)
        print(
            f"  {tag:7} stock {snr_db(ref, s):6.1f} dB   fused {snr_db(ref, f):6.1f} dB   "
            f"fused-vs-stock {snr_db(s, f):6.1f} dB"
        )
        torch.cuda.empty_cache()

    print(
        "\n=== 4. DTensor weight: the patch must hand it to diffusers, not compile it ==="
    )
    import torch.distributed as dist
    from atomrl.lumen.ops.normalization import rmsnorm as mod
    from diffusers.models.normalization import RMSNorm
    from torch.distributed.tensor import Replicate, distribute_tensor, init_device_mesh

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("nccl", rank=0, world_size=1)
    mesh = init_device_mesh("cuda", (1,))
    norm = RMSNorm(128, eps=eps).cuda().to(torch.bfloat16)
    norm.weight = torch.nn.Parameter(
        distribute_tensor(norm.weight.detach(), mesh, [Replicate()])
    )
    x = distribute_tensor(
        torch.randn(1, 4096, 24, 128, device="cuda", dtype=torch.bfloat16),
        mesh,
        [Replicate()],
    )
    mod.patch_diffusers_rmsnorm()
    before = mod.diffusers_rmsnorm_stats()
    out = norm(x)
    after = mod.diffusers_rmsnorm_stats()
    mod.unpatch_diffusers_rmsnorm()
    ref = norm(x)
    took = (
        "diffusers' forward"
        if after["fallback"] == before["fallback"] + 1
        else "the COMPILED path"
    )
    print(
        f"  weight {type(norm.weight).__name__}: patch took {took}; "
        f"output identical to unpatched: {torch.equal(out.to_local(), ref.to_local())}"
    )


if __name__ == "__main__":
    main()
