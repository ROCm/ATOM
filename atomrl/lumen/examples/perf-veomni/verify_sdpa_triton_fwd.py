#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Single-GPU check of Lumen's DiT SDPA route at both attention shapes.

For each of Qwen-Image self-attention, Wan2.1 self-attention and Wan2.1 cross-
attention:

  1. the LSE seam: aiter's Triton forward LSE against the efficient forward's
     and against an FP32 natural-log reference -- the efficient backward reads
     it, so a base-2 or differently laid out LSE would corrupt every gradient
     without raising;
  2. accuracy of out, dQ, dK, dV against FP32 SDPA, for the efficient backend
     alone (sdpa_efficient) and for the Triton-forward splice (attn_triton_fwd);
  3. fwd+bwd time of each.

Expected on MI350X: LSE within ~2e-6 of FP32 for both; SNR 52.5-52.9 dB for
every tensor on both paths.

    HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python verify_sdpa_triton_fwd.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomrl.lumen.ops.attention.hf_patch import (
    TritonFwdEfficientBwd,
    _triton_forward,
)

SHAPES = [
    ("Qwen self", 24, 4109, 4109),
    ("Wan self", 12, 16422, 16422),
    ("Wan cross", 12, 16422, 512),
]


def snr(ref, got):
    ref = ref.float()
    return (
        10 * torch.log10(ref.pow(2).mean() / (got.float() - ref).pow(2).mean()).item()
    )


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def main():
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    for label, h, sq, sk in SHAPES:
        torch.manual_seed(0)
        q = torch.randn(1, h, sq, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, h, sk, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, h, sk, 128, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(1, h, sq, 128, device="cuda", dtype=torch.bfloat16)
        scale = 1.0 / math.sqrt(128)

        _, lse_t = _triton_forward(q, k, v, scale)
        _, lse_e, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, None, True, scale=scale
        )
        ref_lse = torch.logsumexp(
            (q.float() @ k.float().transpose(-1, -2)) * scale, dim=-1
        )
        print(f"\n=== {label}: B=1 H={h} Sq={sq} Sk={sk} ===")
        print(f"  LSE shape triton {tuple(lse_t.shape)} efficient {tuple(lse_e.shape)}")
        print(
            f"  LSE max |triton - FP32 ln ref| {(lse_t - ref_lse).abs().max().item():.3e}   "
            f"max |efficient - FP32 ln ref| {(lse_e - ref_lse).abs().max().item():.3e}"
        )
        del ref_lse

        qf, kf, vf = (t.float().requires_grad_(True) for t in (q, k, v))
        ref_out = F.scaled_dot_product_attention(qf, kf, vf, scale=scale)
        ref_g = torch.autograd.grad(ref_out, [qf, kf, vf], g.float())
        ref_out = ref_out.detach()
        del qf, kf, vf

        def run_splice(g=g, k=k, q=q, scale=scale, v=v):
            qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
            o = TritonFwdEfficientBwd.apply(qq, kk, vv, scale)
            return (o,) + torch.autograd.grad(o, [qq, kk, vv], g)

        def run_efficient(g=g, k=k, q=q, scale=scale, v=v):
            qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
            ):
                o = F.scaled_dot_product_attention(qq, kk, vv, scale=scale)
            return (o,) + torch.autograd.grad(o, [qq, kk, vv], g)

        for name, fn in (
            ("sdpa_efficient", run_efficient),
            ("attn_triton_fwd", run_splice),
        ):
            o, dq, dk, dv = fn()
            t = timed(fn)
            print(
                f"  {name:16} fwd+bwd {t:8.0f} us   SNR vs FP32: out {snr(ref_out, o):5.1f}  "
                f"dQ {snr(ref_g[0], dq):5.1f}  dK {snr(ref_g[1], dk):5.1f}  dV {snr(ref_g[2], dv):5.1f} dB"
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
